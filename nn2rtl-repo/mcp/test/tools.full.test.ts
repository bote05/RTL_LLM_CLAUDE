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
  run_vivado,
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

function wordsPerSample(bytesPerSample: number): number {
  return Math.ceil(bytesPerSample / 4);
}

function packBusSamples(samples: number[][]): number[] {
  return samples.flatMap((sample) => {
    const packed = new Array(wordsPerSample(sample.length)).fill(0);
    sample.forEach((value, byteIndex) => {
      const wordIndex = Math.floor(byteIndex / 4);
      const shift = (byteIndex % 4) * 8;
      packed[wordIndex] |= (value & 0xFF) << shift;
    });
    return packed.map((word) => word | 0);
  });
}

// Binary vector file format matches scripts/golden_impl.py's
// write_golden_vector_file + tb/static_verilator_tb.cpp's loadVectorFile:
//   4 bytes ASCII "NN2V", uint32 LE version=2, uint32 num_vectors,
//   uint32 samples_per_vector, uint32 bytes_per_sample, then
//   num_vectors * samples_per_vector * ceil(bytes_per_sample / 4)
//   int32 LE packed-word payload.
async function writeGoldenVectorFile(
  filePath: string,
  vectors: number[][],
  bytesPerSample: number,
): Promise<void> {
  const numVectors = vectors.length;
  const sampleWords = wordsPerSample(bytesPerSample);
  const samplesPerVector = numVectors > 0 ? vectors[0].length / sampleWords : 0;
  for (const row of vectors) {
    if (row.length % sampleWords !== 0) {
      throw new Error(
        `writeGoldenVectorFile: row length ${row.length} is not a multiple of ${sampleWords} words/sample`,
      );
    }
    if (row.length !== samplesPerVector * sampleWords) {
      throw new Error(
        `writeGoldenVectorFile: row length mismatch, expected ${samplesPerVector * sampleWords} got ${row.length}`,
      );
    }
  }
  const buf = Buffer.alloc(20 + numVectors * samplesPerVector * sampleWords * 4);
  buf.write("NN2V", 0, 4, "ascii");
  buf.writeUInt32LE(2, 4);
  buf.writeUInt32LE(numVectors, 8);
  buf.writeUInt32LE(samplesPerVector, 12);
  buf.writeUInt32LE(bytesPerSample, 16);
  let offset = 20;
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
    bus_bytes_per_sample: 1,
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

  await writeGoldenVectorFile(
    sidecar.golden_inputs_path,
    [inputs],
    Number(sidecar.bus_bytes_per_sample),
  );
  await writeGoldenVectorFile(
    sidecar.golden_outputs_path,
    [outputs],
    Math.ceil(Number(sidecar.output_width_bits) / 8),
  );
  await writeJson(sidecarPath, sidecar);
  return sidecarPath;
}

const CONTRACT_IDS = [
  "flat-bus",
  "tiled-streaming",
  "dram-backed-weights",
  "activation-double-buffering",
  "weight-tiling",
] as const;

async function contractPassthroughVerilog(moduleName: string, contractId: string): Promise<string> {
  const metadataPath = path.join(repoRoot, "contracts", contractId, "metadata.json");
  const metadata = JSON.parse(await readFile(metadataPath, "utf8")) as {
    interface_signals: Array<{ name: string; direction: "input" | "output"; width_bits?: number; width_expr?: string }>;
  };
  const canonical = new Set(["clk", "rst_n", "valid_in", "ready_in", "data_in", "valid_out", "data_out"]);
  const extraSignals = metadata.interface_signals.filter((signal) => !canonical.has(signal.name));
  const extraPorts = extraSignals
    .map((signal) => {
      const width = signal.width_bits ?? 1;
      const range = width > 1 ? ` [${width - 1}:0]` : "      ";
      return `  ${signal.direction} wire${range} ${signal.name}`;
    })
    .join(",\n");
  const extraAssigns = extraSignals
    .filter((signal) => signal.direction === "output")
    .map((signal) => {
      const width = signal.width_bits ?? 1;
      const zero = width > 1 ? `${width}'d0` : "1'b0";
      return `  assign ${signal.name} = ${zero};`;
    })
    .join("\n");
  const commaExtra = extraPorts ? `,\n${extraPorts}` : "";

  return `
module ${moduleName} (
  input  wire       clk,
  input  wire       rst_n,
  input  wire       valid_in,
  input  wire [7:0] data_in,
  output wire       ready_in,
  output reg        valid_out,
  output reg  [7:0] data_out${commaExtra}
);
  assign ready_in = 1'b1;
${extraAssigns}

  always @(posedge clk) begin
    if (!rst_n) begin
      valid_out <= 1'b0;
      data_out <= 8'd0;
    end else begin
      valid_out <= valid_in;
      if (valid_in) begin
        data_out <= data_in;
      end
    end
  end
endmodule
`;
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

  it.skipIf(!process.env.NN2RTL_VIVADO_BIN)("runs vivado against a real fixture module", async () => {
    const verilog = await loadFixture(verilatorFixtureRoot, "stream_passthrough.v");
    const result = await run_vivado(verilog, "stream_passthrough", 20);
    expect(result.success).toBe(true);
    expect(result.report.length).toBeGreaterThan(0);
    expect(result.lut_count).toBeGreaterThanOrEqual(0);
    expect(result.fmax_mhz).toBeGreaterThan(0);
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

  it("passes a valid module through the flat-bus contract template", async () => {
    const tempDir = await makeTempDir("nn2rtl-verilator-contract-template-");
    const verilog = await loadFixture(verilatorFixtureRoot, "stream_passthrough.v");
    const sidecarPath = await writeSidecar(tempDir, "stream_passthrough", [4, 5, 6], [4, 5, 6], 1, {
      testbench_template_path: path.join(repoRoot, "contracts", "flat-bus", "testbench.cpp"),
      contract_id: "flat-bus",
      contract_name: "Flat Bus",
      contract_metadata_path: path.join(repoRoot, "contracts", "flat-bus", "metadata.json"),
      beat_width_bits: 8,
      beats_per_input_sample: 1,
      beats_per_output_sample: 1,
      contract_params: {},
    });

    const result = await run_verilator(verilog, "stream_passthrough", sidecarPath);

    expect(result).toMatchObject({
      module_id: "stream_passthrough",
      status: "pass",
      timing_pass: true,
      timing_actual_cycles: 1,
      expected: [4, 5, 6],
      got: [4, 5, 6],
      max_error: 0,
    });
  });

  it("passes template smoke modules for every contract infrastructure set", async () => {
    for (const contractId of CONTRACT_IDS) {
      const moduleName = `contract_${contractId.replace(/-/g, "_")}_passthrough`;
      const tempDir = await makeTempDir(`nn2rtl-verilator-${contractId}-`);
      const verilog = await contractPassthroughVerilog(moduleName, contractId);
      const sidecarPath = await writeSidecar(tempDir, moduleName, [7, 8], [7, 8], 1, {
        testbench_template_path: path.join(repoRoot, "contracts", contractId, "testbench.cpp"),
        contract_id: contractId,
        contract_name: contractId,
        contract_metadata_path: path.join(repoRoot, "contracts", contractId, "metadata.json"),
        beat_width_bits: 8,
        beats_per_input_sample: 1,
        beats_per_output_sample: 1,
        contract_params: {},
      });

      const result = await run_verilator(verilog, moduleName, sidecarPath);
      expect(result.status, contractId).toBe("pass");
      expect(result.expected, contractId).toEqual([7, 8]);
      expect(result.got, contractId).toEqual([7, 8]);
    }
  });

  it("packs and unpacks 32-bit multi-channel bus samples correctly", async () => {
    const tempDir = await makeTempDir("nn2rtl-verilator-wide32-");
    const verilog = await loadFixture(verilatorFixtureRoot, "stream_wide32_passthrough.v");
    const channelSamples = [
      [1, -2, 3, -4],
      [5, 6, -7, 8],
    ];
    const packedSamples = packBusSamples(channelSamples);
    const sidecarPath = await writeSidecar(
      tempDir,
      "stream_wide32_passthrough",
      packedSamples,
      packedSamples,
      1,
      {
        bus_bytes_per_sample: 4,
        input_width_bits: 32,
        output_width_bits: 32,
      },
    );

    const result = await run_verilator(verilog, "stream_wide32_passthrough", sidecarPath);

    expect(result.status).toBe("pass");
    expect(result.timing_pass).toBe(true);
    expect(result.expected).toEqual(channelSamples.flat());
    expect(result.got).toEqual(channelSamples.flat());
    expect(result.max_error).toBe(0);
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
    // Output is post-ReLU, pre-MaxPool: on the legacy .pth path MaxPool is NOT
    // folded into layer0_0_conv1, so output is stride-2 from 224x224 → 112x112.
    expect(pipelineIr.layers[0].output_shape).toEqual([1, 64, 112, 112]);

    // Weight + bias hex files are materialized on disk (one uppercase hex
    // value per line; widths vary by op, so just assert the format).
    expect(await readFile(pipelineIr.layers[0].weights_path, "utf8")).toMatch(/^[0-9A-F]+\n/);
    expect(await readFile(pipelineIr.layers[0].bias_path!, "utf8")).toMatch(/^[0-9A-F]+\n/);

    // Binary vector files: 20-byte NN2V v2 header + packed int32 words. The
    // stem input is [1, 3, 224, 224], so each vector has 224*224 pixel samples
    // and each sample carries 3 packed input bytes.
    expect(pipelineIr.layers[0].golden_inputs_path).toMatch(/layer0_0_conv1\.goldin$/);
    expect(pipelineIr.layers[0].golden_outputs_path).toMatch(/layer0_0_conv1\.goldout$/);
    const goldinBuf = await readFile(pipelineIr.layers[0].golden_inputs_path);
    expect(goldinBuf.subarray(0, 4).toString("ascii")).toBe("NN2V");
    expect(goldinBuf.readUInt32LE(4)).toBe(2); // version
    expect(goldinBuf.readUInt32LE(8)).toBe(8); // num_vectors
    expect(goldinBuf.readUInt32LE(12)).toBe(224 * 224); // samples_per_vector
    expect(goldinBuf.readUInt32LE(16)).toBe(3); // bytes_per_sample
  });
});
