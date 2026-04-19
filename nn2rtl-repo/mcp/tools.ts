import { execFile, spawn } from "node:child_process";
import { existsSync } from "node:fs";
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
export const SKY130_LIB_PATH = path.resolve(
  repoRoot,
  "vendor",
  "sky130",
  "sky130_fd_sc_hd__tt_025C_1v80.lib",
);

// Hard wall-clock cap for Yosys synthesis. Set to 25 minutes — the manual
// run that proved synthesizability completed in ~16 min, so 25 min gives
// a ~55% margin above the proven baseline. Designs that can't hit that
// budget get capped; raise per-run if you know you're exercising a
// pathological layer.
export const YOSYS_TIMEOUT_MS = 1_500_000;
// Deprecated — kept exported for callers that may reference it. Size control
// is now done via rolling head/tail buffers inside the yosys spawn path, not
// via execFile maxBuffer.
export const YOSYS_MAX_BUFFER_BYTES = 64 * 1024 * 1024;
// Per-stream rolling buffer caps for yosys stdout/stderr. Yosys can emit
// tens of MB of repetitive dead-port / no-latch warnings on large designs.
// We keep the first N bytes ("head") verbatim and the last M bytes ("tail")
// as a ring, dropping the middle. That way the orchestrator-side capYosysReport
// head+ERRORS+tail extraction still has real content to work with, and the
// MCP side can't blow memory on a runaway print.
const YOSYS_STREAM_HEAD_CAP_BYTES = 512 * 1024;
const YOSYS_STREAM_TAIL_CAP_BYTES = 1024 * 1024;
const SKY130_ABC_DRIVING_CELL = "sky130_fd_sc_hd__buf_1";
const SKY130_ABC_OUTPUT_LOAD_FF = 10.0;

// On Windows, the `verilator` entry point is a Perl script that OSS CAD Suite
// does not ship Perl for; `verilator_bin.exe` is the native Windows binary and
// works directly. Linux/macOS installs ship the Perl wrapper with a working
// system Perl, so `verilator` is the right command there. An env override lets
// callers force a specific binary (e.g. a distro-managed `verilator`).
export const VERILATOR_COMMAND =
  process.env.NN2RTL_VERILATOR_BIN ??
  (process.platform === "win32" ? "verilator_bin" : "verilator");

// On Windows, the `python3` command often resolves to a broken Microsoft Store
// alias; `python` typically points at a real interpreter. On Linux/macOS
// `python3` is the conventional entry point. An env override lets callers
// force a specific interpreter (e.g. a virtualenv binary).
export const PYTHON_COMMAND =
  process.env.NN2RTL_PYTHON_BIN ??
  (process.platform === "win32" ? "python" : "python3");

function resolveTmpDirRoot(): string {
  // os.tmpdir() handles TMPDIR / TMP / TEMP / USERPROFILE / platform defaults
  // correctly across Windows, macOS, and Linux. Preserve an explicit absolute
  // TMPDIR override for callers that deliberately redirect tmp (e.g. the
  // Windows cross-env TMPDIR=/tmp workaround in the vitest scripts), but fall
  // through to os.tmpdir() for anything else rather than hardcoding "/tmp".
  const override = process.env.TMPDIR;
  if (override && path.isAbsolute(override)) {
    return override;
  }
  return os.tmpdir();
}

// On Windows, OSS CAD Suite binaries (yosys, verilator, iverilog, abc, ...)
// need YOSYSHQ_ROOT set + both `bin/` and `lib/` prepended to PATH or they
// spawn silently and exit non-zero with no stderr (DLLs not found). The
// shipped `environment.bat` does this setup; Node's execFile inherits the
// parent process env, which usually does *not* have these set unless the
// user launched their shell from that batch file.
//
// Detect the suite root by walking up from the first `yosys`/`yosys.exe` on
// PATH. If found, return an env object with YOSYSHQ_ROOT populated and
// bin/lib prepended to PATH. The env override NN2RTL_YOSYSHQ_ROOT lets
// callers force a specific location.
function resolveOssCadSuiteRoot(env: NodeJS.ProcessEnv): string | null {
  const override = env.NN2RTL_YOSYSHQ_ROOT;
  if (override) {
    return path.resolve(override);
  }
  if (env.YOSYSHQ_ROOT) {
    return path.resolve(env.YOSYSHQ_ROOT);
  }
  const pathVar = env.PATH ?? env.Path ?? "";
  const sep = process.platform === "win32" ? ";" : ":";
  const candidates = pathVar.split(sep).filter(Boolean);
  const binaries = process.platform === "win32" ? ["yosys.exe", "yosys"] : ["yosys"];
  for (const dir of candidates) {
    for (const bin of binaries) {
      if (existsSync(path.join(dir, bin))) {
        // bin dir -> suite root is one level up, IFF sibling `lib/` exists.
        const root = path.resolve(dir, "..");
        if (existsSync(path.join(root, "lib"))) {
          return root;
        }
      }
    }
  }
  return null;
}

// Verilator on Windows needs three things on PATH, in this order:
//  1. A modern g++ (w64devkit / GCC 13+). Without it the C++ compile step
//     fails on `-faligned-new`, `-fcf-protection=none`, etc.
//  2. A real python3 (oss-cad-suite ships one at `lib/python3.exe`). Without
//     it, Verilator's makefile hits the Microsoft Store `python3.exe` shim
//     which prints "Python was not found" and exits with error 0x2331.
//  3. oss-cad-suite's own DLLs (lib/) for anything it shells out to.
// We layer both augmentations: Verilator first (so its g++ wins), then
// oss-cad-suite lib (without touching g++). Override the C++ toolchain
// with NN2RTL_WIN_CXX_TOOLCHAIN_BIN.
export function augmentEnvForVerilatorCxx(env: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  if (process.platform !== "win32") {
    return env;
  }
  const candidates = [
    env.NN2RTL_WIN_CXX_TOOLCHAIN_BIN,
    env.USERPROFILE ? path.join(env.USERPROFILE, "w64devkit", "bin") : undefined,
    "C:\\w64devkit\\bin",
  ].filter((c): c is string => typeof c === "string" && c.length > 0);
  for (const dir of candidates) {
    if (existsSync(path.join(dir, "g++.exe"))) {
      const augmented: NodeJS.ProcessEnv = { ...env };
      const currentPath = env.PATH ?? env.Path ?? "";
      const sep = ";";
      const norm = (d: string) => path.resolve(d).toLowerCase();
      const target = norm(dir);
      const remaining = currentPath
        .split(sep)
        .filter((entry) => entry && norm(entry) !== target);
      const newPath = [dir, ...remaining].join(sep);
      augmented.PATH = newPath;
      augmented.Path = newPath;
      return augmented;
    }
  }
  return env;
}

// Like augmentEnvForOssCadSuite but only prepends `lib/` (DLLs + python3.exe).
// Used for Verilator, which needs oss-cad-suite's python3 but must NOT see
// oss-cad-suite's older g++ (that lives elsewhere on PATH; we want the
// modern w64devkit g++ from augmentEnvForVerilatorCxx to win).
export function augmentEnvForOssCadSuiteLibOnly(env: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const root = resolveOssCadSuiteRoot(env);
  if (!root) {
    return env;
  }
  const augmented: NodeJS.ProcessEnv = { ...env, YOSYSHQ_ROOT: root };
  const sep = process.platform === "win32" ? ";" : ":";
  const libDir = path.join(root, "lib");
  const currentPath = env.PATH ?? env.Path ?? "";
  const norm = (dir: string) => path.resolve(dir).toLowerCase();
  const target = norm(libDir);
  const remaining = currentPath
    .split(sep)
    .filter((entry) => entry && norm(entry) !== target);
  const newPath = [libDir, ...remaining].join(sep);
  augmented.PATH = newPath;
  if (process.platform === "win32") {
    augmented.Path = newPath;
  }
  return augmented;
}

export function augmentEnvForOssCadSuite(env: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const root = resolveOssCadSuiteRoot(env);
  if (!root) {
    return env;
  }
  const augmented: NodeJS.ProcessEnv = { ...env, YOSYSHQ_ROOT: root };
  const sep = process.platform === "win32" ? ";" : ":";
  const binDir = path.join(root, "bin");
  const libDir = path.join(root, "lib");
  // On Windows, Node exposes process.env with BOTH `PATH` and `Path`
  // populated (duplicated). If we update only one, the child process may
  // read the other and see the un-augmented value. So read from whichever
  // is set and write to BOTH keys.
  const currentPath = env.PATH ?? env.Path ?? "";
  // Strip any existing copies of binDir/libDir and prepend them — Windows
  // DLL resolution walks PATH in order, and if another toolchain (e.g.
  // git's mingw64) appears earlier it can load an incompatible
  // libstdc++/libgcc before yosys's own DLLs are seen. Force oss-cad-suite
  // to win by putting it at position 0.
  const norm = (dir: string) => path.resolve(dir).toLowerCase();
  const targets = new Set([norm(binDir), norm(libDir)]);
  const remaining = currentPath
    .split(sep)
    .filter((entry) => entry && !targets.has(norm(entry)));
  const newPath = [binDir, libDir, ...remaining].join(sep);
  augmented.PATH = newPath;
  if (process.platform === "win32") {
    augmented.Path = newPath;
  }
  const certFile = path.join(root, "etc", "cacert.pem");
  if (!augmented.SSL_CERT_FILE && existsSync(certFile)) {
    augmented.SSL_CERT_FILE = certFile;
  }
  return augmented;
}

type CommandOptions = {
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  timeout?: number;
  maxBuffer?: number;
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
    const result = await execFileAsync(file, args, options);
    const stdoutRaw: unknown = result.stdout;
    const stderrRaw: unknown = result.stderr;
    return {
      stdout: typeof stdoutRaw === "string" ? stdoutRaw : Buffer.isBuffer(stdoutRaw) ? stdoutRaw.toString("utf8") : String(stdoutRaw ?? ""),
      stderr: typeof stderrRaw === "string" ? stderrRaw : Buffer.isBuffer(stderrRaw) ? stderrRaw.toString("utf8") : String(stderrRaw ?? ""),
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

// System-level spawn errors (ENOENT, EACCES, timeout, OOM) must not be
// laundered into Verilog syntax/synthesis failures: Surgeon would then try to
// "fix" an out-of-memory error by rewriting correct code. A genuine tool exit
// from iverilog/yosys has a numeric exit code on the Error object; a Node
// spawn failure surfaces `code` as a string like "ENOENT" and typically has no
// `signal` or `stdout`.
export function isSystemSpawnError(error: unknown): boolean {
  if (typeof error !== "object" || error === null) return false;
  const err = error as { code?: unknown; killed?: boolean; signal?: unknown };
  if (typeof err.code === "string") {
    const c = err.code;
    if (
      c === "ENOENT" ||
      c === "EACCES" ||
      c === "EPERM" ||
      c === "ENOMEM" ||
      c === "ETIMEDOUT" ||
      c === "EMFILE" ||
      c === "ENFILE" ||
      // Node's execFile kills the child when stdout/stderr exceeds
      // `maxBuffer`. That is infra (our buffer config is too small), not an
      // RTL bug — must not be routed to Surgeon as a synthesis failure.
      c === "ERR_CHILD_PROCESS_STDIO_MAXBUFFER"
    ) {
      return true;
    }
  }
  if (err.killed === true && err.signal) return true;
  return false;
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
): { fmax_mhz: number; lut_count: number; area_um2: number } {
  // Yosys cell-naming varies by backend (and by version): synth_ice40 emits
  // `SB_LUT4`, other targets emit bare `LUT`, `LUT4`, `LUT5`, `LUT6`, and some
  // paths emit `ICESTORM_LC`. Sum any cell row whose name contains a LUT-ish
  // token so a non-iCE40 target does not silently report `lut_count: 0`.
  let lut_count = 0;
  const lutLineRe = /^\s*(?:\$?[A-Z0-9_]*(?:LUT|ICESTORM_LC)[A-Z0-9_]*)\s+(\d+)\s*$/gim;
  for (const m of report.matchAll(lutLineRe)) {
    lut_count += Number(m[1]);
  }

  // Sky130 / standard-cell flow: `stat -liberty ...` prints a line like
  //   Chip area for module '\layer1_0_conv1': 12345.678900
  // in um^2. When a .lib is used, fall back to total cell count as a proxy
  // "complexity" metric if no LUT-style lines were found.
  let area_um2 = 0;
  const areaMatch = report.match(/Chip\s+area\s+for\s+module[^:]*:\s*([0-9]+(?:\.[0-9]+)?)/i);
  if (areaMatch) {
    area_um2 = Number(areaMatch[1]);
  }
  if (lut_count === 0) {
    // `stat -liberty` on standard-cell flows prints summary lines like either:
    //   147911 1.11E+006 cells
    // or the older:
    //   Number of cells: 147911
    // Reports often contain multiple stats blocks (pre/post mapping), so take
    // the last total-cells line to capture the mapped netlist.
    let totalCells: number | null = null;
    const cellsLineRe = /^\s*(\d+)(?:\s+[0-9]+(?:\.[0-9]+)?(?:E[+-]?\d+)?)?\s+cells\s*$/gim;
    for (const m of report.matchAll(cellsLineRe)) {
      totalCells = Number(m[1]);
    }
    if (totalCells === null) {
      const cellsMatch = report.match(/Number\s+of\s+cells:\s*(\d+)/i);
      if (cellsMatch) {
        totalCells = Number(cellsMatch[1]);
      }
    }
    if (totalCells !== null) {
      lut_count = totalCells;
    }
  }

  // Fmax extraction order — most specific first:
  // 1. Explicit "X MHz" (nextpnr-style report or test mocks)
  // 2. ABC/abc9 "Delay = X.XX ns" (Yosys sta/abc9 output)
  // 3. ABC/abc9 "Delay = X.XX ps"
  // 4. ABC "Current delay (X.XX ps/ns)" fallback
  // With Sky130 timing enabled via `abc -constr ... -D ...`, the `stime -p`
  // summary line is the primary production signal; the "Current delay (...)"
  // line is a useful fallback if that summary is absent.
  const mhzMatch = report.match(/([0-9]+(?:\.[0-9]+)?)\s*MHz/i);
  if (mhzMatch) {
    return { lut_count, fmax_mhz: Number(mhzMatch[1]), area_um2 };
  }

  const nsMatch = report.match(/Delay\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ns/i);
  if (nsMatch) {
    const ns = Number(nsMatch[1]);
    return { lut_count, fmax_mhz: ns > 0 ? 1_000 / ns : 0, area_um2 };
  }

  const psMatch = report.match(/Delay\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ps/i);
  if (psMatch) {
    const ps = Number(psMatch[1]);
    return { lut_count, fmax_mhz: ps > 0 ? 1_000_000 / ps : 0, area_um2 };
  }

  const currentNsMatch = report.match(/Current\s+delay\s*\(\s*([0-9]+(?:\.[0-9]+)?)\s*ns\s*\)/i);
  if (currentNsMatch) {
    const ns = Number(currentNsMatch[1]);
    return { lut_count, fmax_mhz: ns > 0 ? 1_000 / ns : 0, area_um2 };
  }

  const currentPsMatch = report.match(/Current\s+delay\s*\(\s*([0-9]+(?:\.[0-9]+)?)\s*ps\s*\)/i);
  if (currentPsMatch) {
    const ps = Number(currentPsMatch[1]);
    return { lut_count, fmax_mhz: ps > 0 ? 1_000_000 / ps : 0, area_um2 };
  }

  return { lut_count, fmax_mhz: 0, area_um2 };
}

function yosysTimingTargetPs(clock_period_ns: number): number | null {
  if (!Number.isFinite(clock_period_ns) || clock_period_ns <= 0) {
    return null;
  }
  return Math.max(1, Math.round(clock_period_ns * 1_000));
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
        env: augmentEnvForOssCadSuite(runtime.env),
      });
      return { success: true, stderr: "" };
    } catch (error: unknown) {
      if (isSystemSpawnError(error)) {
        throw error;
      }
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

    // The sidecar carries `module_name` and `module_id` fields that the bench
    // never rechecks against the DUT it was given. If a caller passes a
    // mismatched pair we want to fail loudly here rather than silently build
    // the wrong module.
    if (sidecar.module_name !== module_name) {
      throw new Error(
        `run_verilator: sidecar.module_name='${sidecar.module_name}' does not match the module_name argument '${module_name}'.`,
      );
    }

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
        VERILATOR_COMMAND,
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
        { cwd: tempDir, env: augmentEnvForVerilatorCxx(augmentEnvForOssCadSuiteLibOnly(runtime.env)) },
      );
    } catch (error: unknown) {
      if (isSystemSpawnError(error)) {
        throw error;
      }
      const stderr = stderrFromUnknown(error);
      const tbSetupFailure =
        /static_verilator_tb\.cpp|VlWide<|verilated_types\.h|VMODEL_HEADER|VMODEL_CLASS/.test(stderr);
      return {
        module_id: sidecar.module_id,
        status: tbSetupFailure ? "fail" : "syntax_error",
        status_class: tbSetupFailure ? "tb_setup_error" : undefined,
        timing_pass: false,
        timing_actual_cycles: tbSetupFailure ? -1 : 0,
        timing_expected_cycles: sidecar.pipeline_latency_cycles,
        verilator_stderr: stderr,
        fix_hint: tbSetupFailure
          ? `Static Verilator testbench build failed while compiling '${module_name}'. The RTL may be fine; inspect the external C++ / bus-width diagnostics before attempting module-local repair.`
          : `Verilator build failed while compiling '${module_name}' with the static testbench.`,
      };
    }

    const binaryName = `V${module_name}${process.platform === "win32" ? ".exe" : ""}`;
    const binaryPath = path.join(tempDir, "obj_dir", binaryName);
    let simulationError: unknown = null;

    try {
      await runtime.commandRunner(binaryPath, [sidecar_path], {
        cwd: tempDir,
        env: augmentEnvForVerilatorCxx(augmentEnvForOssCadSuiteLibOnly(runtime.env)),
      });
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
      status_class: "tb_setup_error",
      timing_pass: false,
      timing_actual_cycles: -1,
      timing_expected_cycles: sidecar.pipeline_latency_cycles,
      expected: [],
      got: [],
      failure_class: null,
      verilator_stderr: simulationError ? stderrFromUnknown(simulationError) : "",
      fix_hint: `Static testbench did not produce results JSON at '${sidecar.results_path}'.`,
    };
  }, runtime);
}

// Rolling-buffer stdout/stderr capture for yosys. Head is kept verbatim up
// to HEAD_CAP; after that, further output accumulates into a ring-limited
// tail capped at TAIL_CAP. Middle bytes are elided but their count is
// reported so capYosysReport downstream can reconstruct the structure.
type RollingBuffer = {
  head: string[];
  headBytes: number;
  tail: string[];
  tailBytes: number;
  elided: number;
};

function createRollingBuffer(): RollingBuffer {
  return { head: [], headBytes: 0, tail: [], tailBytes: 0, elided: 0 };
}

function appendRolling(buf: RollingBuffer, chunk: string): void {
  if (!chunk) return;
  let remaining = chunk;
  if (buf.headBytes < YOSYS_STREAM_HEAD_CAP_BYTES) {
    const room = YOSYS_STREAM_HEAD_CAP_BYTES - buf.headBytes;
    const take = remaining.length <= room ? remaining : remaining.slice(0, room);
    buf.head.push(take);
    buf.headBytes += take.length;
    remaining = remaining.length <= room ? "" : remaining.slice(room);
    if (!remaining) return;
  }
  buf.tail.push(remaining);
  buf.tailBytes += remaining.length;
  while (buf.tailBytes > YOSYS_STREAM_TAIL_CAP_BYTES && buf.tail.length > 1) {
    const drop = buf.tail.shift()!;
    buf.tailBytes -= drop.length;
    buf.elided += drop.length;
  }
}

function materializeRolling(buf: RollingBuffer): string {
  const head = buf.head.join("");
  const tail = buf.tail.join("");
  if (!buf.elided) return head + tail;
  return `${head}\n...[${buf.elided} bytes of middle output elided]...\n${tail}`;
}

type YosysSpawnResult = {
  stdout: string;
  stderr: string;
  exitCode: number | null;
  signal: NodeJS.Signals | null;
  timedOut: boolean;
  spawnError?: NodeJS.ErrnoException;
};

async function runYosysViaSpawn(
  args: string[],
  options: { cwd: string; env: NodeJS.ProcessEnv; timeoutMs: number },
): Promise<YosysSpawnResult> {
  return new Promise((resolve) => {
    const stdoutBuf = createRollingBuffer();
    const stderrBuf = createRollingBuffer();
    let timedOut = false;
    let spawnError: NodeJS.ErrnoException | undefined;
    let settled = false;

    const child = spawn("yosys", args, {
      cwd: options.cwd,
      env: options.env,
      windowsHide: true,
      // stdio default: pipes for stdout/stderr, no stdin
    });

    child.stdout?.setEncoding("utf8");
    child.stderr?.setEncoding("utf8");
    child.stdout?.on("data", (chunk: string) => appendRolling(stdoutBuf, chunk));
    child.stderr?.on("data", (chunk: string) => appendRolling(stderrBuf, chunk));

    const timer = setTimeout(() => {
      timedOut = true;
      // SIGTERM first; on Windows spawn translates this to TerminateProcess.
      try {
        child.kill("SIGTERM");
      } catch {
        // already exited
      }
    }, options.timeoutMs);

    const settle = (result: YosysSpawnResult) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(result);
    };

    child.on("error", (err: NodeJS.ErrnoException) => {
      spawnError = err;
      settle({
        stdout: materializeRolling(stdoutBuf),
        stderr: materializeRolling(stderrBuf),
        exitCode: null,
        signal: null,
        timedOut: false,
        spawnError,
      });
    });

    child.on("close", (code, signal) => {
      settle({
        stdout: materializeRolling(stdoutBuf),
        stderr: materializeRolling(stderrBuf),
        exitCode: code,
        signal,
        timedOut,
        spawnError,
      });
    });
  });
}

export async function run_yosys(
  verilog_source: string,
  module_name: string,
  clockPeriodNsOrRuntimeOverrides: number | Partial<ToolsRuntime> = 0,
  runtimeOverrides: Partial<ToolsRuntime> = {},
): Promise<{
  success: boolean;
  lut_count: number;
  fmax_mhz: number;
  area_um2: number;
  report: string;
}> {
  const clock_period_ns =
    typeof clockPeriodNsOrRuntimeOverrides === "number" ? clockPeriodNsOrRuntimeOverrides : 0;
  const runtime = createToolsRuntime(
    typeof clockPeriodNsOrRuntimeOverrides === "number"
      ? runtimeOverrides
      : clockPeriodNsOrRuntimeOverrides,
  );
  return withTempDir("nn2rtl-yosys-", async (tempDir) => {
    const verilogPath = path.join(tempDir, `${module_name}.v`);
    await writeFile(verilogPath, verilog_source, "utf8");

    // Sky130 standard-cell flow. Previously targeted iCE40 with synth_ice40
    // -abc9; that path hung indefinitely on deep combinational blobs because
    // abc9 delay-aware mapping does unbounded search on large LUT cones.
    // Sky130 (free Google/SkyWater standard-cell library) gives real area
    // (um^2), mapped standard-cell counts, and constrained timing when ABC is
    // given a load/driving-cell constraint file plus a target delay derived
    // from the LayerIR clock period. See vendor/sky130/.
    const sky130Lib = SKY130_LIB_PATH.replace(/\\/g, "/");
    const targetDelayPs = yosysTimingTargetPs(clock_period_ns);
    let abcCommand = `abc -liberty ${sky130Lib}`;
    if (targetDelayPs !== null) {
      const constrPath = path.join(tempDir, "sky130.abc.constr");
      await writeFile(
        constrPath,
        [
          `set_driving_cell ${SKY130_ABC_DRIVING_CELL}`,
          `set_load ${SKY130_ABC_OUTPUT_LOAD_FF.toFixed(1)}`,
          "",
        ].join("\n"),
        "utf8",
      );
      // `abc -constr` switches Yosys to ABC's timing-aware default script,
      // which ends in `stime -p` and prints the critical-path delay.
      abcCommand =
        `abc -liberty ${sky130Lib} ` +
        `-constr ${constrPath.replace(/\\/g, "/")} -D ${targetDelayPs}`;
    }
    // memory_share merges structurally equivalent read ports across memory
    // arrays (e.g. the same weight index read by multiple MAC lanes in a
    // generate loop). This reduces cell count without requiring BRAM macros
    // and is safe to run before synth. The Verilog file is loaded via the
    // positional command-line argument so no read_verilog is needed here.
    const yosysScript = [
      `memory_share`,
      `synth -top ${module_name}`,
      `dfflibmap -liberty ${sky130Lib}`,
      abcCommand,
      `stat -liberty ${sky130Lib}`,
    ].join("; ");
    // Use spawn directly (not runtime.commandRunner / execFileAsync) because
    // execFile only returns stdout/stderr on the rejection object AFTER the
    // child has exited; on kill-by-timeout, Node on Windows can drop buffered
    // output. Streaming with spawn and rolling buffers preserves whatever the
    // child managed to print right up to the kill signal — critical for
    // Surgeon repairs, which used to get empty "Command failed: ..." messages
    // and repair blind.
    const spawnResult = await runYosysViaSpawn(
      ["-p", yosysScript, verilogPath],
      {
        cwd: tempDir,
        env: augmentEnvForOssCadSuite(runtime.env),
        timeoutMs: YOSYS_TIMEOUT_MS,
      },
    );

    if (spawnResult.spawnError) {
      // ENOENT / EACCES / etc. — propagate as a system error so the caller
      // doesn't route it to Surgeon as a synthesis failure.
      if (isSystemSpawnError(spawnResult.spawnError)) {
        throw spawnResult.spawnError;
      }
      // Unusual spawn failure without a classifiable errno. Surface the
      // captured output plus the error message.
      const captured = [spawnResult.stdout, spawnResult.stderr].filter(Boolean).join("\n");
      return {
        success: false,
        lut_count: 0,
        fmax_mhz: 0,
        area_um2: 0,
        report: [
          `Yosys spawn error: ${spawnResult.spawnError.message}`,
          captured,
        ].filter(Boolean).join("\n"),
      };
    }

    const captured = [spawnResult.stdout, spawnResult.stderr].filter(Boolean).join("\n");
    const cleanExit = spawnResult.exitCode === 0 && !spawnResult.timedOut && !spawnResult.signal;

    if (cleanExit) {
      return {
        success: true,
        report: captured,
        ...parseYosysReport(captured),
      };
    }

    const prefix = spawnResult.timedOut
      ? `Yosys synthesis timed out after ${YOSYS_TIMEOUT_MS / 1000}s. ` +
        `Likely cause: the design is a single very deep/wide combinational blob ` +
        `(e.g. all MACs of a conv unrolled into one always block) that abc cannot ` +
        `map quickly. Rewrite the module to use the intended registered ` +
        `output-stationary MAC-array structure so the combinational cone ` +
        `between any two registers stays small.`
      : spawnResult.signal
        ? `Yosys was killed by signal ${spawnResult.signal}.`
        : `Yosys exited with code ${spawnResult.exitCode}.`;
    const report = [prefix, "---", captured].filter(Boolean).join("\n");
    return {
      success: false,
      lut_count: 0,
      fmax_mhz: 0,
      area_um2: 0,
      report,
    };
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

  await runtime.commandRunner(PYTHON_COMMAND, [scriptPath, checkpoint_path], {
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
