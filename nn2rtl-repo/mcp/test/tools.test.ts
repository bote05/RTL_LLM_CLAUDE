import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  VERILATOR_COMMAND,
  parseYosysReport,
  readSidecarIfPresent,
  read_weights,
  resolveOutputRoot,
  resolveRepoRootFromEnv,
  run_iverilog,
  run_verilator,
  run_yosys,
  stderrFromUnknown,
  withTempDir,
  write_verilog,
} from "../tools.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "../..");
const fixtureRoot = path.join(repoRoot, "test", "fixtures");
const tempDirs: string[] = [];

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
});

async function makeTempDir(prefix: string): Promise<string> {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), prefix));
  tempDirs.push(tempDir);
  return tempDir;
}

async function writeSidecar(
  tempDir: string,
  overrides: Record<string, unknown> = {},
): Promise<string> {
  const sidecarPath = path.join(tempDir, "sidecar.json");
  const sidecar = {
    module_name: "unit_module",
    module_id: "unit_module",
    clock_signal: "clk",
    reset_signal: "rst_n",
    valid_in_signal: "valid_in",
    valid_out_signal: "valid_out",
    ready_in_signal: "ready_in",
    data_in_signal: "data_in",
    data_out_signal: "data_out",
    bus_bytes_per_sample: 1,
    input_width_bits: 8,
    output_width_bits: 8,
    pipeline_latency_cycles: 1,
    clock_period_ns: 20,
    golden_inputs_path: path.join(tempDir, "inputs.json"),
    golden_outputs_path: path.join(tempDir, "outputs.json"),
    results_path: path.join(tempDir, "results.json"),
    testbench_template_path: path.join(repoRoot, "tb", "static_verilator_tb.cpp"),
    ...overrides,
  };

  // Only materialize the vector files when the sidecar points at absolute
  // paths — the "rejects relative path" test deliberately supplies a bare
  // filename like `relative-inputs.json`, and writing to it would leak into
  // the test's cwd (which is the mcp package root on Windows runs).
  if (path.isAbsolute(sidecar.golden_inputs_path)) {
    await writeFile(sidecar.golden_inputs_path, JSON.stringify([[0, 1, 2]]), "utf8");
  }
  if (path.isAbsolute(sidecar.golden_outputs_path)) {
    await writeFile(sidecar.golden_outputs_path, JSON.stringify([[0, 1, 2]]), "utf8");
  }
  await writeFile(sidecarPath, JSON.stringify(sidecar), "utf8");
  return sidecarPath;
}

describe("mcp tools", () => {
  it("extracts stderr text from unknown errors", () => {
    expect(stderrFromUnknown({ stderr: Buffer.from("boom") })).toBe("boom");
    expect(stderrFromUnknown(new Error("plain-error"))).toBe("plain-error");
  });

  it("creates and cleans temporary directories", async () => {
    let createdDir = "";
    await withTempDir("nn2rtl-tools-test-", async (tempDir) => {
      createdDir = tempDir;
      await writeFile(path.join(tempDir, "fixture.txt"), "ok", "utf8");
      expect(await readFile(path.join(tempDir, "fixture.txt"), "utf8")).toBe("ok");
    });

    await expect(readFile(path.join(createdDir, "fixture.txt"), "utf8")).rejects.toThrow();
  });

  it("resolves repo root overrides from the environment", () => {
    expect(resolveRepoRootFromEnv({ NN2RTL_REPO_ROOT: "/tmp/override" })).toBe(
      path.resolve("/tmp/override"),
    );
    expect(resolveRepoRootFromEnv({})).toBe(repoRoot);
  });

  it("runs iverilog successfully through the command abstraction", async () => {
    const commandRunner = vi.fn(async (_file, args: string[]) => {
      expect(args[3]).toContain("unit_module.v");
      expect(await readFile(args[3], "utf8")).toContain("module unit_module");
      return { stdout: "", stderr: "" };
    });

    await expect(
      run_iverilog("module unit_module; endmodule", "unit_module", { commandRunner }),
    ).resolves.toEqual({
      success: true,
      stderr: "",
    });
    expect(commandRunner).toHaveBeenCalledOnce();
  });

  it("returns syntax errors from iverilog failures", async () => {
    const result = await run_iverilog("module broken", "broken_module", {
      commandRunner: async () => {
        throw { stderr: "syntax error" };
      },
    });
    expect(result).toEqual({
      success: false,
      stderr: "syntax error",
    });
  });

  it("parses Yosys reports for LUTs and MHz", () => {
    expect(parseYosysReport("LUT4 12\nEstimated fmax: 50.5 MHz")).toEqual({
      lut_count: 12,
      fmax_mhz: 50.5,
      area_um2: 0,
    });
  });

  it("parses abc9 delay reported in ns into Fmax", () => {
    const report = "LUT4 8\nABC: Best Delay = 10.000 ns";
    const parsed = parseYosysReport(report);
    expect(parsed.lut_count).toBe(8);
    expect(parsed.fmax_mhz).toBeCloseTo(100, 5);
  });

  it("parses abc9 delay reported in ps into Fmax", () => {
    const report = "LUT4 2\nABC: Delay = 2500.0 ps";
    const parsed = parseYosysReport(report);
    expect(parsed.lut_count).toBe(2);
    expect(parsed.fmax_mhz).toBeCloseTo(400, 5);
  });

  it("returns fmax_mhz=0 when no delay/MHz information is present", () => {
    expect(parseYosysReport("LUT4 3\nnothing measurable here")).toEqual({
      lut_count: 3,
      fmax_mhz: 0,
      area_um2: 0,
    });
  });

  it("runs yosys successfully when the command layer returns a valid report", async () => {
    const result = await run_yosys("module passthrough; endmodule", "passthrough", {
      commandRunner: async () => ({
        stdout: "LUT4 4\nEstimated fmax: 42.0 MHz",
        stderr: "",
      }),
    });
    expect(result).toEqual({
      success: true,
      lut_count: 4,
      fmax_mhz: 42,
      area_um2: 0,
      report: "LUT4 4\nEstimated fmax: 42.0 MHz",
    });
  });

  it("returns a failure report when yosys execution fails", async () => {
    const result = await run_yosys("module bad; endmodule", "bad", {
      commandRunner: async () => {
        throw { stderr: "yosys failed" };
      },
    });
    expect(result).toEqual({
      success: false,
      lut_count: 0,
      fmax_mhz: 0,
      area_um2: 0,
      report: "yosys failed",
    });
  });

  it("writes Verilog source and metadata sidecars", async () => {
    const tempDir = await makeTempDir("nn2rtl-write-verilog-");
    const module = JSON.parse(
      await readFile(path.join(fixtureRoot, "verilog_module.json"), "utf8"),
    );

    const writtenPath = await write_verilog(module, tempDir);

    expect(resolveOutputRoot(tempDir)).toBe(path.resolve(process.cwd(), tempDir));
    expect(await readFile(writtenPath, "utf8")).toContain("module unit_module");
    expect(
      JSON.parse(await readFile(path.join(tempDir, "rtl", "unit_module.meta.json"), "utf8")),
    ).toEqual(module);
  });

  it("loads sidecars when present and returns null when missing", async () => {
    const tempDir = await makeTempDir("nn2rtl-sidecar-");
    const sidecarPath = await writeSidecar(tempDir);

    await expect(readSidecarIfPresent(sidecarPath)).resolves.toMatchObject({
      module_id: "unit_module",
      ready_in_signal: "ready_in",
    });
    await expect(readSidecarIfPresent(path.join(tempDir, "missing.json"))).resolves.toBeNull();
  });

  it("rejects invalid sidecars", async () => {
    const tempDir = await makeTempDir("nn2rtl-sidecar-invalid-");
    const sidecarPath = path.join(tempDir, "sidecar.json");
    await writeFile(sidecarPath, JSON.stringify({ module_name: "bad" }), "utf8");
    await expect(readSidecarIfPresent(sidecarPath)).rejects.toThrow("VerificationSidecar");
  });

  it("returns syntax_error when verilator compilation fails", async () => {
    const tempDir = await makeTempDir("nn2rtl-verilator-build-");
    const sidecarPath = await writeSidecar(tempDir);

    const result = await run_verilator("module unit_module; endmodule", "unit_module", sidecarPath, {
      commandRunner: async (file) => {
        if (file === VERILATOR_COMMAND) {
          throw { stderr: "compile boom" };
        }
        return { stdout: "", stderr: "" };
      },
    });

    expect(result.status).toBe("syntax_error");
    expect(result.verilator_stderr).toBe("compile boom");
  });

  it("rejects non-absolute sidecar vector paths before invoking verilator", async () => {
    const tempDir = await makeTempDir("nn2rtl-verilator-sidecar-");
    const sidecarPath = await writeSidecar(tempDir, {
      golden_inputs_path: "relative-inputs.json",
    });

    await expect(
      run_verilator("module unit_module; endmodule", "unit_module", sidecarPath, {
        commandRunner: async () => ({ stdout: "", stderr: "" }),
      }),
    ).rejects.toThrow("must be an absolute path");
  });

  it("returns a fallback verification result when simulation finishes without results", async () => {
    const tempDir = await makeTempDir("nn2rtl-verilator-results-");
    const sidecarPath = await writeSidecar(tempDir);
    const commandRunner = vi.fn(async () => ({ stdout: "", stderr: "" }));

    const result = await run_verilator("module unit_module; endmodule", "unit_module", sidecarPath, {
      commandRunner,
    });

    expect(result).toMatchObject({
      module_id: "unit_module",
      status: "fail",
      timing_expected_cycles: 1,
    });
    expect(result.fix_hint).toContain("did not produce results JSON");
    expect(commandRunner).toHaveBeenCalledTimes(2);
  });

  it("validates emitted PipelineIR in read_weights when the script layer succeeds", async () => {
    const tempRepoRoot = await makeTempDir("nn2rtl-read-weights-");
    const pipelineIr = JSON.parse(
      await readFile(path.join(fixtureRoot, "pipeline_ir.json"), "utf8"),
    );
    const outputPath = path.join(tempRepoRoot, "output", "golden_vectors.json");
    await mkdir(path.dirname(outputPath), { recursive: true });
    await writeFile(outputPath, JSON.stringify(pipelineIr, null, 2), "utf8");

    const commandRunner = vi.fn(async () => ({ stdout: "", stderr: "" }));

    await expect(
      read_weights(
        "checkpoint.pth",
        { quantization: "int8" },
        {
          commandRunner,
          env: {
            ...process.env,
            NN2RTL_REPO_ROOT: tempRepoRoot,
          },
        },
      ),
    ).resolves.toEqual(pipelineIr);

    expect(commandRunner).toHaveBeenCalledOnce();
  });

  it("rejects invalid PipelineIR emitted by the Python layer", async () => {
    const tempRepoRoot = await makeTempDir("nn2rtl-read-weights-invalid-");
    const outputPath = path.join(tempRepoRoot, "output", "golden_vectors.json");
    await mkdir(path.dirname(outputPath), { recursive: true });
    await writeFile(outputPath, JSON.stringify({ nope: true }), "utf8");

    await expect(
      read_weights(
        "checkpoint.pth",
        { quantization: "int8" },
        {
          commandRunner: async () => ({ stdout: "", stderr: "" }),
          env: {
            ...process.env,
            NN2RTL_REPO_ROOT: tempRepoRoot,
          },
        },
      ),
    ).rejects.toThrow("is not a valid PipelineIR");
  });
});
