import { execFile } from "node:child_process";
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";

import { pipelineIrSchema, verifResultSchema, verificationSidecarSchema } from "./schemas.js";
import type { PipelineIR, VerificationSidecar, VerifResult, VerilogModule } from "./types.js";

const execFileAsync = promisify(execFile);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(
  __dirname,
  path.basename(__dirname) === "dist" ? ".." : ".",
  "..",
);
export const TB_SOURCE_PATH = path.resolve(repoRoot, "tb", "static_verilator_tb.cpp");
export const TB_JSON_HPP_PATH = path.resolve(repoRoot, "tb", "third_party", "json.hpp");

function resolveTmpDirRoot(): string {
  if (process.platform === "win32") {
    return os.tmpdir();
  }
  return process.env.TMPDIR && path.isAbsolute(process.env.TMPDIR)
    ? process.env.TMPDIR
    : "/tmp";
}

type CommandOptions = {
  cwd?: string;
  env?: NodeJS.ProcessEnv;
};

type CommandResult = {
  stderr: string;
  stdout: string;
};

export type CommandRunner = (
  file: string,
  args: string[],
  options?: CommandOptions,
) => Promise<CommandResult>;

export type ToolsRuntime = {
  commandRunner: CommandRunner;
  cwd: string;
  env: NodeJS.ProcessEnv;
  tmpDirRoot: string;
};

const DEFAULT_TOOLS_RUNTIME: ToolsRuntime = {
  async commandRunner(file, args, options) {
    const { stdout, stderr } = await execFileAsync(file, args, options);
    return {
      stdout: typeof stdout === "string" ? stdout : stdout.toString("utf8"),
      stderr: typeof stderr === "string" ? stderr : stderr.toString("utf8"),
    };
  },
  cwd: repoRoot,
  env: process.env,
  tmpDirRoot: resolveTmpDirRoot(),
};

export function createToolsRuntime(
  overrides: Partial<ToolsRuntime> = {},
): ToolsRuntime {
  return {
    ...DEFAULT_TOOLS_RUNTIME,
    ...overrides,
  };
}

export async function withTempDir<T>(
  prefix: string,
  fn: (tempDir: string) => Promise<T>,
  runtime: ToolsRuntime = createToolsRuntime(),
): Promise<T> {
  const tempDir = await mkdtemp(path.join(runtime.tmpDirRoot, prefix));
  try {
    return await fn(tempDir);
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
}

export function stderrFromUnknown(error: unknown): string {
  if (typeof error === "object" && error !== null && "stderr" in error) {
    const stderr = (error as { stderr?: string | Buffer }).stderr;
    if (typeof stderr === "string") {
      return stderr;
    }
    if (stderr instanceof Buffer) {
      return stderr.toString("utf8");
    }
  }

  if (error instanceof Error) {
    return error.message;
  }

  return String(error);
}

export function parseYosysReport(
  report: string,
): { fmax_mhz: number; lut_count: number } {
  const lutMatch = report.match(/LUT4\s+(\d+)/);
  const fmaxMatch = report.match(/([0-9]+(?:\.[0-9]+)?)\s*MHz/i);
  return {
    lut_count: lutMatch ? Number(lutMatch[1]) : 0,
    fmax_mhz: fmaxMatch ? Number(fmaxMatch[1]) : 0,
  };
}

export function resolveOutputRoot(outputDir: string): string {
  return path.resolve(process.cwd(), outputDir);
}

export function resolveRepoRootFromEnv(env: NodeJS.ProcessEnv = process.env): string {
  const override = env.NN2RTL_REPO_ROOT;
  return override ? path.resolve(override) : repoRoot;
}

function requireAbsoluteSidecarPaths(sidecar: VerificationSidecar): void {
  const pathFields = [
    "golden_inputs_path",
    "golden_outputs_path",
    "results_path",
  ] as const;

  for (const field of pathFields) {
    if (!path.isAbsolute(sidecar[field])) {
      throw new Error(
        `run_verilator: sidecar field '${field}' must be an absolute path; got '${sidecar[field]}'.`,
      );
    }
  }
}

async function readVerilatorResults(resultsPath: string): Promise<VerifResult> {
  const raw = await readFile(resultsPath, "utf8");
  const parsed: unknown = JSON.parse(raw);
  const validated = verifResultSchema.safeParse(parsed);
  if (!validated.success) {
    throw new Error(
      `run_verilator: results JSON at '${resultsPath}' failed schema validation:\n${JSON.stringify(validated.error.issues, null, 2)}`,
    );
  }
  return validated.data;
}

async function readVerilatorResultsIfPresent(resultsPath: string): Promise<VerifResult | null> {
  try {
    return await readVerilatorResults(resultsPath);
  } catch (error: unknown) {
    if (
      typeof error === "object" &&
      error !== null &&
      "code" in error &&
      (error as { code?: string }).code === "ENOENT"
    ) {
      return null;
    }
    throw error;
  }
}

export async function run_iverilog(
  verilog_source: string,
  module_name: string,
  runtimeOverrides: Partial<ToolsRuntime> = {},
): Promise<{ success: boolean; stderr: string }> {
  const runtime = createToolsRuntime(runtimeOverrides);
  return withTempDir("nn2rtl-iverilog-", async (tempDir) => {
    const verilogPath = path.join(tempDir, `${module_name}.v`);
    await writeFile(verilogPath, verilog_source, "utf8");

    try {
      await runtime.commandRunner("iverilog", ["-o", os.devNull, "-g2012", verilogPath], {
        cwd: tempDir,
      });
      return { success: true, stderr: "" };
    } catch (error: unknown) {
      return { success: false, stderr: stderrFromUnknown(error) };
    }
  }, runtime);
}

export async function run_verilator(
  verilog_source: string,
  module_name: string,
  sidecar_path: string,
  runtimeOverrides: Partial<ToolsRuntime> = {},
): Promise<VerifResult> {
  const runtime = createToolsRuntime(runtimeOverrides);
  return withTempDir("nn2rtl-verilator-", async (tempDir) => {
    const sidecar = await readSidecarIfPresent(sidecar_path);
    if (!sidecar) {
      throw new Error(`run_verilator: sidecar '${sidecar_path}' was not found.`);
    }
    requireAbsoluteSidecarPaths(sidecar);

    const verilogPath = path.join(tempDir, `${module_name}.v`);
    const tempTbPath = path.join(tempDir, "static_verilator_tb.cpp");
    const tempJsonDir = path.join(tempDir, "third_party");
    const tempJsonPath = path.join(tempJsonDir, "json.hpp");

    await writeFile(verilogPath, verilog_source, "utf8");
    await mkdir(tempJsonDir, { recursive: true });
    await copyFile(TB_SOURCE_PATH, tempTbPath);
    await copyFile(TB_JSON_HPP_PATH, tempJsonPath);

    try {
      await runtime.commandRunner(
        "verilator",
        [
          "--cc",
          "--exe",
          "--build",
          "--Mdir",
          "obj_dir",
          "-Wall",
          "-Wno-fatal",
          "--top-module",
          module_name,
          "-CFLAGS",
          `-std=c++17 -DVMODEL_HEADER="\\\"V${module_name}.h\\\"" -DVMODEL_CLASS=V${module_name}`,
          "static_verilator_tb.cpp",
          `${module_name}.v`,
        ],
        { cwd: tempDir },
      );
    } catch (error: unknown) {
      return {
        module_id: sidecar.module_id,
        status: "syntax_error",
        timing_pass: false,
        timing_actual_cycles: 0,
        timing_expected_cycles: 0,
        verilator_stderr: stderrFromUnknown(error),
        fix_hint: `Verilator build failed while compiling '${module_name}' with the static testbench.`,
      };
    }

    const binaryName = `V${module_name}${process.platform === "win32" ? ".exe" : ""}`;
    const binaryPath = path.join(tempDir, "obj_dir", binaryName);
    let simulationError: unknown = null;

    try {
      await runtime.commandRunner(binaryPath, [sidecar_path], { cwd: tempDir });
    } catch (error: unknown) {
      simulationError = error;
    }

    const parsedResults = await readVerilatorResultsIfPresent(sidecar.results_path);
    if (parsedResults) {
      return parsedResults;
    }

    return {
      module_id: sidecar.module_id,
      status: "fail",
      timing_pass: false,
      timing_actual_cycles: 0,
      timing_expected_cycles: sidecar.pipeline_latency_cycles,
      expected: [],
      got: [],
      failure_class: null,
      verilator_stderr: simulationError ? stderrFromUnknown(simulationError) : "",
      fix_hint: `Static testbench did not produce results JSON at '${sidecar.results_path}'.`,
    };
  }, runtime);
}

export async function run_yosys(
  verilog_source: string,
  module_name: string,
  runtimeOverrides: Partial<ToolsRuntime> = {},
): Promise<{ success: boolean; lut_count: number; fmax_mhz: number; report: string }> {
  const runtime = createToolsRuntime(runtimeOverrides);
  return withTempDir("nn2rtl-yosys-", async (tempDir) => {
    const verilogPath = path.join(tempDir, `${module_name}.v`);
    await writeFile(verilogPath, verilog_source, "utf8");

    try {
      const { stdout, stderr } = await runtime.commandRunner(
        "yosys",
        ["-p", "synth_ice40 -abc9; stat", verilogPath],
        { cwd: tempDir },
      );

      const report = [stdout, stderr].filter(Boolean).join("\n");
      return {
        success: true,
        report,
        ...parseYosysReport(report),
      };
    } catch (error: unknown) {
      return {
        success: false,
        lut_count: 0,
        fmax_mhz: 0,
        report: stderrFromUnknown(error),
      };
    }
  }, runtime);
}

export async function read_weights(
  checkpoint_path: string,
  quantization_config: object,
  runtimeOverrides: Partial<ToolsRuntime> = {},
): Promise<PipelineIR> {
  const runtime = createToolsRuntime(runtimeOverrides);
  const scriptPath = path.join(repoRoot, "scripts", "generate_golden.py");
  const outputPath = path.join(resolveRepoRootFromEnv(runtime.env), "output", "golden_vectors.json");

  await runtime.commandRunner("python3", [scriptPath, checkpoint_path], {
    cwd: runtime.cwd,
    env: {
      ...runtime.env,
      NN2RTL_QUANTIZATION_CONFIG: JSON.stringify(quantization_config),
    },
  });

  const raw = await readFile(outputPath, "utf8");
  const parsed: unknown = JSON.parse(raw);
  const validated = pipelineIrSchema.safeParse(parsed);
  if (!validated.success) {
    throw new Error(
      `read_weights: '${outputPath}' is not a valid PipelineIR:\n${JSON.stringify(validated.error.issues, null, 2)}`,
    );
  }
  return validated.data as PipelineIR;
}

export async function write_verilog(
  module: VerilogModule,
  output_dir: string,
): Promise<string> {
  const outputRoot = resolveOutputRoot(output_dir);
  const rtlDir = path.join(outputRoot, "rtl");
  const verilogPath = path.join(rtlDir, `${module.module_id}.v`);
  const metadataPath = path.join(rtlDir, `${module.module_id}.meta.json`);

  await mkdir(rtlDir, { recursive: true });
  await writeFile(verilogPath, module.verilog_source, "utf8");
  await writeFile(metadataPath, `${JSON.stringify(module, null, 2)}\n`, "utf8");

  return verilogPath;
}

export async function readSidecarIfPresent(
  filePath: string,
): Promise<VerificationSidecar | null> {
  let raw: string;
  try {
    raw = await readFile(filePath, "utf8");
  } catch (error: unknown) {
    if (
      typeof error === "object" &&
      error !== null &&
      "code" in error &&
      (error as { code?: string }).code === "ENOENT"
    ) {
      return null;
    }
    throw error;
  }

  const parsed: unknown = JSON.parse(raw);
  const validated = verificationSidecarSchema.safeParse(parsed);
  if (!validated.success) {
    throw new Error(
      `run_verilator: '${filePath}' is not a valid VerificationSidecar:\n${JSON.stringify(validated.error.issues, null, 2)}`,
    );
  }
  return validated.data;
}
