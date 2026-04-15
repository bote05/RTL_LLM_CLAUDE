import { execFile } from "node:child_process";
import { mkdtemp, readFile, rm, unlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";

import { afterEach, describe, expect, it } from "vitest";

import {
  PYTHON_COMMAND,
  read_weights,
  run_iverilog,
  run_verilator,
  run_yosys,
} from "../tools.js";

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

// Binary vector file format matches scripts/golden_impl.py's
// write_golden_vector_file + tb/static_verilator_tb.cpp's loadVectorFile:
//   4 bytes ASCII "NN2V", uint32 LE version=1, uint32 num_vectors,
//   uint32 samples_per_vector, then num*samples int32 LE samples.
async function writeGoldenVectorFile(
  filePath: string,
  vectors: number[][],
): Promise<void> {
  const numVectors = vectors.length;
  const samplesPerVector = numVectors > 0 ? vectors[0].length : 0;
  for (const row of vectors) {
    if (row.length !== samplesPerVector) {
      throw new Error(
        `writeGoldenVectorFile: row length mismatch, expected ${samplesPerVector} got ${row.length}`,
      );
    }
  }
  const buf = Buffer.alloc(16 + numVectors * samplesPerVector * 4);
  buf.write("NN2V", 0, 4, "ascii");
  buf.writeUInt32LE(1, 4);
  buf.writeUInt32LE(numVectors, 8);
  buf.writeUInt32LE(samplesPerVector, 12);
  let offset = 16;
  for (const row of vectors) {
    for (const sample of row) {
      buf.writeInt32LE(sample | 0, offset);
      offset += 4;
    }
  }
  await writeFile(filePath, buf);
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
    golden_inputs_path: path.join(tempDir, `${moduleName}.goldin`),
    golden_outputs_path: path.join(tempDir, `${moduleName}.goldout`),
    results_path: path.join(tempDir, `${moduleName}.results.json`),
    testbench_template_path: path.join(repoRoot, "tb", "static_verilator_tb.cpp"),
    ...overrides,
  };

  await writeGoldenVectorFile(sidecar.golden_inputs_path, [inputs]);
  await writeGoldenVectorFile(sidecar.golden_outputs_path, [outputs]);
  await writeJson(sidecarPath, sidecar);
  return sidecarPath;
}

// Real Verilator builds take 10–15s on Windows with w64devkit; the vitest
// default 5s timeout is far too tight for this suite.
const VERILATOR_TEST_TIMEOUT_MS = 180_000;

describe("mcp tools full integration", { timeout: VERILATOR_TEST_TIMEOUT_MS }, () => {
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
    await unlink(path.join(tempDir, "stream_passthrough.goldout"));

    const result = await run_verilator(verilog, "stream_passthrough", sidecarPath);

    expect(result.status).toBe("fail");
    expect(result.max_error).toBe(-1);
    expect(result.fix_hint).toContain("Could not open vector file");
    expect(result.verilator_stderr).toContain("Could not open vector file");
  });

  it("reads weights through the real Python frontend and emits ResNet-50 layer1 artifacts", async () => {
    const tempRepoRoot = await makeTempDir("nn2rtl-read-weights-full-");

    await execFileAsync(
      PYTHON_COMMAND,
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
      { calibration: "synthetic" },
      {
        cwd: repoRoot,
        env: {
          ...process.env,
          NN2RTL_REPO_ROOT: tempRepoRoot,
        },
      },
    );

    // The real ResNet-50 PTQ flow emits 17 modules: stem + 3 layer1 blocks +
    // downsample + post_add_relus. The first is the fused stem.
    expect(pipelineIr.layers).toHaveLength(17);
    expect(pipelineIr.model_name).toBe("resnet50");
    expect(pipelineIr.layers[0]).toMatchObject({
      module_id: "layer0_0_conv1",
      op_type: "conv2d",
      clock_signal: "clk",
      ready_in_signal: "ready_in",
      data_in_signal: "data_in",
      data_out_signal: "data_out",
    });
    // Output is post-MaxPool (fusion is invisible to downstream): 1 x 64 x 56 x 56.
    expect(pipelineIr.layers[0].output_shape).toEqual([1, 64, 56, 56]);

    // Weight + bias hex files are materialized on disk (one uppercase hex
    // value per line; widths vary by op, so just assert the format).
    expect(await readFile(pipelineIr.layers[0].weights_path, "utf8")).toMatch(/^[0-9A-F]+\n/);
    expect(await readFile(pipelineIr.layers[0].bias_path!, "utf8")).toMatch(/^[0-9A-F]+\n/);

    // Binary vector files: 16-byte NN2V header + int32 LE samples. The stem
    // input is [1, 3, 224, 224] = 150528 samples per vector, 8 vectors.
    expect(pipelineIr.layers[0].golden_inputs_path).toMatch(/layer0_0_conv1\.goldin$/);
    expect(pipelineIr.layers[0].golden_outputs_path).toMatch(/layer0_0_conv1\.goldout$/);
    const goldinBuf = await readFile(pipelineIr.layers[0].golden_inputs_path);
    expect(goldinBuf.subarray(0, 4).toString("ascii")).toBe("NN2V");
    expect(goldinBuf.readUInt32LE(4)).toBe(1); // version
    expect(goldinBuf.readUInt32LE(8)).toBe(8); // num_vectors
    expect(goldinBuf.readUInt32LE(12)).toBe(1 * 3 * 224 * 224); // samples_per_vector
  });
});
