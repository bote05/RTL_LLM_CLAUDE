import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  evaluateImprovementTargets,
  parseImproveCliArgs,
  runImprove,
  type FoundryImproveInput,
  type ImprovementMetrics,
  type SynthesisReport,
} from "../improve.js";
import type {
  LayerIR,
  PipelineIR,
  RetrospectorAdvice,
  VerifResult,
  VerilogModule,
} from "../types.js";

const fixedNow = () => new Date("2026-05-02T12:00:00Z");

let tempRoot: string | null = null;

async function writeJson(filePath: string, value: unknown): Promise<void> {
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

async function exists(filePath: string): Promise<boolean> {
  try {
    await readFile(filePath, "utf8");
    return true;
  } catch (error: unknown) {
    if (
      typeof error === "object" &&
      error !== null &&
      "code" in error &&
      (error as { code?: string }).code === "ENOENT"
    ) {
      return false;
    }
    throw error;
  }
}

function layerFixture(): LayerIR {
  return {
    module_id: "unit_module",
    op_type: "conv2d",
    input_shape: [1, 1, 1, 1],
    output_shape: [1, 1, 1, 1],
    weights_path: "/tmp/unit_module_weights.hex",
    bias_path: "/tmp/unit_module_bias.hex",
    weight_shape: [1, 1, 1, 1],
    num_weights: 1,
    scale_factor: 0.125,
    zero_point: 0,
    pipeline_latency_cycles: 10,
    clock_period_ns: 20,
    input_width_bits: 8,
    output_width_bits: 8,
    clock_signal: "clk",
    reset_signal: "rst_n",
    valid_in_signal: "valid_in",
    valid_out_signal: "valid_out",
    ready_in_signal: "ready_in",
    data_in_signal: "data_in",
    data_out_signal: "data_out",
    golden_inputs_path: "/tmp/unit_module.goldin",
    golden_outputs_path: "/tmp/unit_module.goldout",
    stride: [1, 1],
    padding: [0, 0],
    dilation: [1, 1],
    groups: 1,
    mac_parallelism: 1,
  };
}

function originalModule(): VerilogModule {
  return {
    module_id: "unit_module",
    spec_hash: "fixture-hash",
    verilog_source: "module unit_module; // original\nendmodule\n",
    generated_by: "Foundry",
    attempt: 1,
  };
}

function improvedModule(attempt: number): VerilogModule {
  return {
    module_id: "unit_module",
    spec_hash: "fixture-hash",
    verilog_source: `module unit_module; // improved attempt ${attempt}\nendmodule\n`,
    generated_by: "Foundry",
    attempt,
  };
}

function scalarizedLineBufModule(attempt: number): VerilogModule {
  return {
    module_id: "unit_module",
    spec_hash: "fixture-hash",
    verilog_source: [
      "module unit_module;",
      "  localparam IH = 14;",
      "  localparam IW = 14;",
      "  localparam IC = 512;",
      "  reg signed [7:0] line_buf [0:IH*IW-1][0:IC-1];",
      "endmodule",
    ].join("\n"),
    generated_by: "Foundry",
    attempt,
  };
}

function vivadoReport(overrides: Partial<SynthesisReport> = {}): SynthesisReport {
  return {
    success: true,
    tool: "vivado",
    part: "xczu9eg-ffvb1156-2-e",
    stage: "synth",
    lut_count: 100,
    ff_count: 10,
    dsp_count: 0,
    bram18_count: 0,
    bram36_count: 0,
    bram18_equiv: 0,
    wns_ns: 1,
    setup_wns_ns: 1,
    hold_wns_ns: null,
    timing_met: true,
    fmax_mhz: 100,
    report: "mock vivado",
    ...overrides,
  };
}

function verifResult(overrides: Partial<VerifResult> = {}): VerifResult {
  return {
    module_id: "unit_module",
    status: "pass",
    timing_pass: true,
    timing_actual_cycles: 10,
    timing_expected_cycles: 10,
    initiation_interval_cycles: 4,
    expected: [0],
    got: [0],
    max_error: 0,
    mean_error: 0,
    ...overrides,
  };
}

async function seedProject(): Promise<string> {
  tempRoot = await mkdtemp(path.join(os.tmpdir(), "nn2rtl-improve-"));
  const outputRoot = path.join(tempRoot, "output");
  const reportsDir = path.join(outputRoot, "reports");
  const rtlDir = path.join(outputRoot, "rtl");
  await mkdir(reportsDir, { recursive: true });
  await mkdir(rtlDir, { recursive: true });
  const layer = layerFixture();
  const pipeline: PipelineIR = {
    model_name: "fixture-net",
    quantization: "int8_symmetric_per_tensor",
    generated_at: "2026-05-02T12:00:00Z",
    layers: [layer],
  };
  const module = originalModule();
  await writeJson(path.join(outputRoot, "layer_ir.json"), pipeline);
  await writeFile(path.join(rtlDir, "unit_module.v"), module.verilog_source, "utf8");
  await writeJson(path.join(rtlDir, "unit_module.meta.json"), module);
  await writeJson(path.join(reportsDir, "unit_module.vivado.json"), vivadoReport());
  await writeJson(path.join(reportsDir, "unit_module.results.json"), verifResult());
  return tempRoot;
}

function foundryMock(calls: Array<{
  attempt_index: number;
  resume_session_id?: string;
  previous_attempt_count: number;
  has_retrospector_advice: boolean;
}> = []) {
  return vi.fn(async (input: FoundryImproveInput) => {
    calls.push({
      attempt_index: input.attempt_index,
      resume_session_id: input.resume_session_id,
      previous_attempt_count: input.previous_attempts.length,
      has_retrospector_advice: input.retrospector_advice !== undefined,
    });
    const resultMessage = {
      type: "result",
      subtype: "success",
      result: "{}",
      total_cost_usd: 0,
      modelUsage: {},
      session_id: "shared-foundry-session",
    } as const;
    return {
      module: improvedModule(input.attempt_index),
      result: resultMessage,
      messages: [resultMessage],
      session_id: "shared-foundry-session",
    };
  });
}

afterEach(async () => {
  if (tempRoot) {
    await rm(tempRoot, { recursive: true, force: true });
    tempRoot = null;
  }
});

describe("evaluateImprovementTargets", () => {
  it("requires every requested target to pass its deterministic rule", () => {
    const baseline: ImprovementMetrics = {
      lut: 100,
      dsp: 0,
      bram: 0,
      latency_cycles: 10,
      ii: 4,
    };
    const next: ImprovementMetrics = {
      lut: 90,
      dsp: 8,
      bram: 1,
      latency_cycles: 9,
      ii: 3,
    };

    const verdict = evaluateImprovementTargets(
      baseline,
      next,
      ["use-dsp", "use-bram", "reduce-lut", "reduce-latency", "increase-throughput"],
    );

    expect(verdict.overall).toBe(true);
    expect(verdict.targets.map((target) => target.satisfied)).toEqual([
      true,
      true,
      true,
      true,
      true,
    ]);
  });

  it("fails the whole improvement when any target misses its threshold", () => {
    const verdict = evaluateImprovementTargets(
      { lut: 100, dsp: 0, bram: 0, latency_cycles: 10, ii: 4 },
      { lut: 96, dsp: 1, bram: 0, latency_cycles: 10, ii: 5 },
      ["reduce-lut", "use-bram"],
    );

    expect(verdict.overall).toBe(false);
    expect(verdict.targets).toMatchObject([
      { target: "use-bram", satisfied: false },
      { target: "reduce-lut", satisfied: false },
    ]);
  });
});

describe("parseImproveCliArgs", () => {
  it("parses the module, predefined targets, and keep-reference flag", () => {
    expect(parseImproveCliArgs([
      "unit_module",
      "--targets=reduce-lut,use-dsp,reduce-lut",
      "--keep-reference",
    ])).toEqual({
      moduleId: "unit_module",
      targets: ["use-dsp", "reduce-lut"],
      keepReference: true,
    });
  });
});

describe("runImprove", () => {
  it("refuses to improve a module whose saved baseline is not already passing", async () => {
    const root = await seedProject();
    await writeJson(
      path.join(root, "output", "reports", "unit_module.vivado.json"),
      vivadoReport({ success: false, timing_met: false, fmax_mhz: 0 }),
    );
    const foundryFn = foundryMock();

    await expect(runImprove("unit_module", {
      targets: ["reduce-lut"],
      paths: { repoRoot: root },
      runtime: {
        now: fixedNow,
        foundryFn,
        assayerFn: vi.fn(async () => verifResult()),
        synthesisFn: vi.fn(async () => vivadoReport({ lut_count: 80 })),
      },
    })).rejects.toThrow("baseline Vivado report is not passing");
    expect(foundryFn).not.toHaveBeenCalled();
  });

  it("replaces the canonical RTL only after Verilator, Vivado, and target checks pass", async () => {
    const root = await seedProject();
    const foundryFn = foundryMock();
    const assayerFn = vi.fn(async () => verifResult());
    const synthesisFn = vi.fn(async () => vivadoReport({ lut_count: 80 }));

    const result = await runImprove("unit_module", {
      targets: ["reduce-lut"],
      paths: { repoRoot: root },
      runtime: { now: fixedNow, foundryFn, assayerFn, synthesisFn },
    });

    expect(result.success).toBe(true);
    expect(result.final_action).toBe("replaced");
    expect(foundryFn).toHaveBeenCalledTimes(1);
    expect(assayerFn).toHaveBeenCalledTimes(1);
    expect(synthesisFn).toHaveBeenCalledTimes(1);
    await expect(readFile(path.join(root, "output", "rtl", "unit_module.v"), "utf8"))
      .resolves.toContain("improved attempt 1");
    await expect(readFile(path.join(root, "output", "rtl", "archive", "unit_module__20260502T120000Z.v"), "utf8"))
      .resolves.toContain("original");
    await expect(readFile(path.join(
      root,
      "output",
      "improve",
      "unit_module",
      "20260502T120000Z",
      "attempt_1.messages.json",
    ), "utf8")).resolves.toContain("shared-foundry-session");
  });

  it("saves a verified improved variant without touching the canonical RTL when requested", async () => {
    const root = await seedProject();
    const foundryFn = foundryMock();
    const assayerFn = vi.fn(async () => verifResult());
    const synthesisFn = vi.fn(async () => vivadoReport({ bram18_equiv: 1 }));

    const result = await runImprove("unit_module", {
      targets: ["use-bram"],
      keepReference: true,
      paths: { repoRoot: root },
      runtime: { now: fixedNow, foundryFn, assayerFn, synthesisFn },
    });

    expect(result.success).toBe(true);
    expect(result.final_action).toBe("kept-as-variant");
    await expect(readFile(path.join(root, "output", "rtl", "unit_module.v"), "utf8"))
      .resolves.toContain("original");
    await expect(readFile(path.join(root, "knowledge", "references", "improved", "unit_module__use-bram.v"), "utf8"))
      .resolves.toContain("improved attempt 1");
    const lifecycle = JSON.parse(await readFile(path.join(root, "knowledge", "doc_lifecycle.json"), "utf8"));
    expect(lifecycle.docs["improved_unit_module__use-bram"]).toMatchObject({
      status: "active",
      reference_path: "knowledge/references/improved/unit_module__use-bram.v",
      improvement_targets: ["use-bram"],
    });
  });

  it("does not run Vivado for a Verilator-failed attempt and retries the same conversation", async () => {
    const root = await seedProject();
    const calls: Array<{
      attempt_index: number;
      resume_session_id?: string;
      previous_attempt_count: number;
      has_retrospector_advice: boolean;
    }> = [];
    const foundryFn = foundryMock(calls);
    const assayerFn = vi
      .fn()
      .mockResolvedValueOnce(verifResult({ status: "fail", timing_pass: false }))
      .mockResolvedValueOnce(verifResult());
    const synthesisFn = vi.fn(async () => vivadoReport({ lut_count: 80 }));

    const result = await runImprove("unit_module", {
      targets: ["reduce-lut"],
      paths: { repoRoot: root },
      runtime: { now: fixedNow, foundryFn, assayerFn, synthesisFn },
    });

    expect(result.success).toBe(true);
    expect(result.attempts).toMatchObject([
      { attempt_index: 1, failed_gate: "verilator" },
      { attempt_index: 2, failed_gate: null },
    ]);
    expect(synthesisFn).toHaveBeenCalledTimes(1);
    expect(calls).toMatchObject([
      {
        attempt_index: 1,
        resume_session_id: undefined,
        previous_attempt_count: 0,
        has_retrospector_advice: false,
      },
      {
        attempt_index: 2,
        resume_session_id: "shared-foundry-session",
        previous_attempt_count: 1,
        has_retrospector_advice: false,
      },
    ]);
  });

  it("skips Vivado for a synthesis-preflight failure and retries the same conversation", async () => {
    const root = await seedProject();
    const spatialLayer: LayerIR = {
      ...layerFixture(),
      input_shape: [1, 512, 14, 14],
      output_shape: [1, 512, 7, 7],
      weight_shape: [512, 512, 3, 3],
      num_weights: 512 * 512 * 3 * 3,
      input_width_bits: 256,
      output_width_bits: 256,
      channel_tile: 32,
      pipeline_latency_cycles: 100,
    };
    await writeJson(path.join(root, "output", "layer_ir.json"), {
      model_name: "fixture-net",
      quantization: "int8_symmetric_per_tensor",
      generated_at: "2026-05-02T12:00:00Z",
      layers: [spatialLayer],
    });
    const resultMessage = {
      type: "result",
      subtype: "success",
      result: "{}",
      total_cost_usd: 0,
      modelUsage: {},
      session_id: "shared-foundry-session",
    } as const;
    const foundryFn = vi
      .fn()
      .mockResolvedValueOnce({
        module: scalarizedLineBufModule(1),
        result: resultMessage,
        messages: [resultMessage],
        session_id: "shared-foundry-session",
      })
      .mockResolvedValueOnce({
        module: improvedModule(2),
        result: resultMessage,
        messages: [resultMessage],
        session_id: "shared-foundry-session",
      });
    const assayerFn = vi.fn(async () => verifResult({ timing_expected_cycles: 100, timing_actual_cycles: 100 }));
    const synthesisFn = vi.fn(async () => vivadoReport({ lut_count: 80 }));

    const result = await runImprove("unit_module", {
      targets: ["reduce-lut"],
      paths: { repoRoot: root },
      runtime: { now: fixedNow, foundryFn, assayerFn, synthesisFn },
    });

    expect(result.success).toBe(true);
    expect(result.attempts).toMatchObject([
      {
        attempt_index: 1,
        failed_gate: "vivado",
        vivado_report: {
          success: false,
        },
      },
      { attempt_index: 2, failed_gate: null },
    ]);
    expect(result.attempts[0].vivado_report?.report).toContain("large_scalarized_activation_memory");
    expect(synthesisFn).toHaveBeenCalledTimes(1);
    expect(foundryFn).toHaveBeenCalledTimes(2);
    expect(foundryFn.mock.calls[1][0]).toMatchObject({
      attempt_index: 2,
      resume_session_id: "shared-foundry-session",
    });
  });

  it("keeps the original after three failed deterministic improvements and logs Retrospector advice", async () => {
    const root = await seedProject();
    const calls: Array<{
      attempt_index: number;
      resume_session_id?: string;
      previous_attempt_count: number;
      has_retrospector_advice: boolean;
    }> = [];
    const foundryFn = foundryMock(calls);
    const assayerFn = vi.fn(async () => verifResult());
    const synthesisFn = vi
      .fn()
      .mockResolvedValueOnce(vivadoReport({ lut_count: 99 }))
      .mockResolvedValueOnce(vivadoReport({ lut_count: 98 }))
      .mockResolvedValueOnce(vivadoReport({ lut_count: 97 }));
    const retrospectorAdvice: RetrospectorAdvice = {
      analysis: "All attempts preserved correctness but missed the 5 percent LUT delta.",
      suggestion: "Try a structural sharing change instead of local rewrites.",
    };
    const retrospectorFn = vi.fn(async () => retrospectorAdvice);

    const result = await runImprove("unit_module", {
      targets: ["reduce-lut"],
      paths: { repoRoot: root },
      runtime: {
        now: fixedNow,
        foundryFn,
        assayerFn,
        synthesisFn,
        retrospectorFn,
      },
    });

    expect(result.success).toBe(false);
    expect(result.final_action).toBe("no-change");
    expect(result.attempts).toHaveLength(3);
    expect(result.attempts.map((attempt) => attempt.failed_gate)).toEqual([
      "improvement_checker",
      "improvement_checker",
      "improvement_checker",
    ]);
    expect(retrospectorFn).toHaveBeenCalledTimes(1);
    expect(calls).toMatchObject([
      {
        attempt_index: 1,
        resume_session_id: undefined,
        previous_attempt_count: 0,
        has_retrospector_advice: false,
      },
      {
        attempt_index: 2,
        resume_session_id: "shared-foundry-session",
        previous_attempt_count: 1,
        has_retrospector_advice: false,
      },
      {
        attempt_index: 3,
        resume_session_id: "shared-foundry-session",
        previous_attempt_count: 2,
        has_retrospector_advice: true,
      },
    ]);
    await expect(readFile(path.join(root, "output", "rtl", "unit_module.v"), "utf8"))
      .resolves.toContain("original");
    expect(await exists(path.join(root, "output", "rtl", "archive", "unit_module__20260502T120000Z.v")))
      .toBe(false);
    const report = JSON.parse(await readFile(result.report_path, "utf8"));
    expect(report).toMatchObject({
      final_action: "no-change",
      retrospector_advice: retrospectorAdvice,
    });
  });
});
