import { execFile } from "node:child_process";
import { mkdtemp, readFile, rm, unlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";

import { afterEach, describe, expect, it } from "vitest";

import { read_weights, run_iverilog, run_verilator, run_yosys } from "../tools.js";

const execFileAsync = promisify(execFile);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "../..");
const fixtureRoot = path.join(repoRoot, "test", "fixtures");
const verilatorFixtureRoot = path.join(fixtureRoot, "verilator");
const tempDirs: string[] = [];

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
});

async function makeTempDir(prefix: string): Promise<string> {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), prefix));
  tempDirs.push(tempDir);
  return tempDir;
}

async function loadFixture(directory: string, fileName: string): Promise<string> {
  return readFile(path.join(directory, fileName), "utf8");
}

async function writeJson(filePath: string, value: unknown): Promise<void> {
  await writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

async function writeSidecar(
  tempDir: string,
  moduleName: string,
  inputs: number[],
  outputs: number[],
  pipelineLatencyCycles: number,
  overrides: Record<string, unknown> = {},
): Promise<string> {
  const sidecarPath = path.join(tempDir, `${moduleName}.sidecar.json`);
  const sidecar = {
    module_name: moduleName,
    module_id: moduleName,
    clock_signal: "clk",
    reset_signal: "rst_n",
    valid_in_signal: "valid_in",
    valid_out_signal: "valid_out",
    ready_in_signal: "ready_in",
    data_in_signal: "data_in",
    data_out_signal: "data_out",
    input_width_bits: 8,
    output_width_bits: 8,
    pipeline_latency_cycles: pipelineLatencyCycles,
    clock_period_ns: 20,
    golden_inputs_path: path.join(tempDir, `${moduleName}.inputs.json`),
    golden_outputs_path: path.join(tempDir, `${moduleName}.outputs.json`),
    results_path: path.join(tempDir, `${moduleName}.results.json`),
    testbench_template_path: path.join(repoRoot, "tb", "static_verilator_tb.cpp"),
    ...overrides,
  };

  await writeJson(sidecar.golden_inputs_path, [inputs]);
  await writeJson(sidecar.golden_outputs_path, [outputs]);
  await writeJson(sidecarPath, sidecar);
  return sidecarPath;
}

describe("mcp tools full integration", () => {
  it("runs iverilog successfully on a real fixture module", async () => {
    const verilog = await loadFixture(verilatorFixtureRoot, "stream_passthrough.v");
    await expect(run_iverilog(verilog, "stream_passthrough")).resolves.toEqual({
      success: true,
      stderr: "",
    });
  });

  it("returns syntax errors from a real iverilog invocation", async () => {
    const verilog = await loadFixture(fixtureRoot, "broken_module.v");
    const result = await run_iverilog(verilog, "broken_module");
    expect(result.success).toBe(false);
    expect(result.stderr.length).toBeGreaterThan(0);
  });

  it("runs yosys against a real fixture module", async () => {
    const verilog = await loadFixture(verilatorFixtureRoot, "stream_passthrough.v");
    const result = await run_yosys(verilog, "stream_passthrough");
    expect(result.success).toBe(true);
    expect(result.report.length).toBeGreaterThan(0);
    expect(result.lut_count).toBeGreaterThanOrEqual(0);
    expect(result.fmax_mhz).toBeGreaterThanOrEqual(0);
  });

  it("passes a valid module through the static C++ testbench", async () => {
    const tempDir = await makeTempDir("nn2rtl-verilator-pass-");
    const verilog = await loadFixture(verilatorFixtureRoot, "stream_passthrough.v");
    const sidecarPath = await writeSidecar(tempDir, "stream_passthrough", [1, 2, 7], [1, 2, 7], 1);

    const result = await run_verilator(verilog, "stream_passthrough", sidecarPath);

    expect(result).toMatchObject({
      module_id: "stream_passthrough",
      status: "pass",
      timing_pass: true,
      timing_actual_cycles: 1,
      timing_expected_cycles: 1,
      expected: [1, 2, 7],
      got: [1, 2, 7],
      max_error: 0,
    });
  });

  it("surfaces numerical mismatches above the allowed tolerance", async () => {
    const tempDir = await makeTempDir("nn2rtl-verilator-mismatch-");
    const verilog = await loadFixture(verilatorFixtureRoot, "stream_offset.v");
    const sidecarPath = await writeSidecar(tempDir, "stream_offset", [1, 2, 7], [1, 2, 7], 1);

    const result = await run_verilator(verilog, "stream_offset", sidecarPath);

    expect(result.status).toBe("fail");
    expect(result.timing_pass).toBe(true);
    expect(result.max_error).toBe(10);
    expect(result.expected).toEqual([1, 2, 7]);
    expect(result.got).toEqual([11, 12, 17]);
  });

  it("fails when the observed latency does not match the sidecar contract", async () => {
    const tempDir = await makeTempDir("nn2rtl-verilator-latency-");
    const verilog = await loadFixture(verilatorFixtureRoot, "stream_latency2.v");
    const sidecarPath = await writeSidecar(tempDir, "stream_latency2", [4, 5], [4, 5], 1);

    const result = await run_verilator(verilog, "stream_latency2", sidecarPath);

    expect(result.status).toBe("fail");
    expect(result.timing_pass).toBe(false);
    expect(result.timing_actual_cycles).toBe(2);
    expect(result.timing_expected_cycles).toBe(1);
  });

  it("handles ready_in backpressure correctly", async () => {
    const tempDir = await makeTempDir("nn2rtl-verilator-stall-");
    const verilog = await loadFixture(verilatorFixtureRoot, "stream_stall.v");
    const sidecarPath = await writeSidecar(tempDir, "stream_stall", [3, 6, 9], [3, 6, 9], 1);

    const result = await run_verilator(verilog, "stream_stall", sidecarPath);

    expect(result.status).toBe("pass");
    expect(result.timing_pass).toBe(true);
    expect(result.expected).toEqual([3, 6, 9]);
    expect(result.got).toEqual([3, 6, 9]);
  });

  it("tolerates valid_out bubbles between samples", async () => {
    const tempDir = await makeTempDir("nn2rtl-verilator-bubble-");
    const verilog = await loadFixture(verilatorFixtureRoot, "stream_bubble.v");
    const sidecarPath = await writeSidecar(tempDir, "stream_bubble", [8, 9], [8, 9], 2);

    const result = await run_verilator(verilog, "stream_bubble", sidecarPath);

    expect(result.status).toBe("pass");
    expect(result.timing_pass).toBe(true);
    expect(result.timing_actual_cycles).toBe(2);
    expect(result.expected).toEqual([8, 9]);
    expect(result.got).toEqual([8, 9]);
  });

  it("rejects sidecars that do not use canonical signal names", async () => {
    const tempDir = await makeTempDir("nn2rtl-verilator-canonical-");
    const verilog = await loadFixture(verilatorFixtureRoot, "stream_passthrough.v");
    const sidecarPath = await writeSidecar(tempDir, "stream_passthrough", [1], [1], 1, {
      data_out_signal: "stream_data_out",
    });

    await expect(
      run_verilator(verilog, "stream_passthrough", sidecarPath),
    ).rejects.toThrow("VerificationSidecar");
  });

  it("returns the fallback error JSON when a vector file is missing", async () => {
    const tempDir = await makeTempDir("nn2rtl-verilator-missing-vector-");
    const verilog = await loadFixture(verilatorFixtureRoot, "stream_passthrough.v");
    const sidecarPath = await writeSidecar(tempDir, "stream_passthrough", [1, 2], [1, 2], 1);
    await unlink(path.join(tempDir, "stream_passthrough.outputs.json"));

    const result = await run_verilator(verilog, "stream_passthrough", sidecarPath);

    expect(result.status).toBe("fail");
    expect(result.max_error).toBe(-1);
    expect(result.fix_hint).toContain("Could not open vector file");
    expect(result.verilator_stderr).toContain("Could not open vector file");
  });

  it("reads weights through the real toy Python flow and emits weight artifacts", async () => {
    const tempRepoRoot = await makeTempDir("nn2rtl-read-weights-full-");

    await execFileAsync(
      "python3",
      [path.join(repoRoot, "scripts", "quantize_model.py")],
      {
        cwd: repoRoot,
        env: {
          ...process.env,
          NN2RTL_REPO_ROOT: tempRepoRoot,
        },
      },
    );

    const pipelineIr = await read_weights(
      "checkpoints/resnet50_int8.pth",
      { calibration: "toy" },
      {
        cwd: repoRoot,
        env: {
          ...process.env,
          NN2RTL_REPO_ROOT: tempRepoRoot,
        },
      },
    );

    expect(pipelineIr.layers[0]).toMatchObject({
      module_id: "toy_conv1x1",
      ready_in_signal: "ready_in",
      data_in_signal: "data_in",
      data_out_signal: "data_out",
      golden_inputs: [[0, 1, 2, 7]],
      golden_outputs: [[1, 3, 5, 15]],
    });
    expect(await readFile(pipelineIr.layers[0].weights_path, "utf8")).toBe("02\n");
    expect(await readFile(pipelineIr.layers[0].bias_path!, "utf8")).toBe("01\n");
  });
});
