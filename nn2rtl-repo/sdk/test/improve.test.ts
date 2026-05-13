import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  buildFoundryImprovePrompt,
  buildImproveSweepPlan,
  evaluateImprovementTargets,
  parseImproveCliArgs,
  parseImproveSweepCliArgs,
  runImprove,
  runImproveSequence,
  type FoundryImproveInput,
  type ImprovementRetrospectorInput,
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

vi.setConfig({ testTimeout: 30_000 });

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
      ff: 1000,
      dsp: 0,
      bram: 0,
      fmax_mhz: 200,
      latency_cycles: 10,
      ii: 4,
    };
    // For a 200 MHz baseline (below the 300 MHz floor), improve-fmax now
    // requires max(200*1.05=210, min(300, 200+50)=250) = 250. The next
    // metric clears that with margin so the rule passes.
    const next: ImprovementMetrics = {
      lut: 90,
      ff: 800,
      dsp: 8,
      bram: 8,
      fmax_mhz: 260,
      latency_cycles: 9,
      ii: 3,
    };

    const verdict = evaluateImprovementTargets(
      baseline,
      next,
      ["use-dsp", "use-bram", "reduce-lut", "reduce-ff", "improve-fmax", "reduce-latency", "increase-throughput"],
    );

    expect(verdict.overall).toBe(true);
    expect(verdict.targets.map((target) => target.satisfied)).toEqual([
      true,
      true,
      true,
      true,
      true,
      true,
      true,
    ]);
  });

  it("improve-fmax rejects a tiny relative bump when the baseline is well below the floor", () => {
    const baseline: ImprovementMetrics = {
      lut: 100, ff: 1000, dsp: 0, bram: 0, fmax_mhz: 167, latency_cycles: 10, ii: 4,
    };
    // Old rule required >= min(300, 167*1.05=175.35) = 175.35 — a trivial
    // sliver of progress for a module that needs to reach ~300 to be
    // competitive. New rule requires max(175.35, min(300, 167+50)=217) = 217.
    // 200 MHz "improvement" no longer clears the new bar.
    const weakImprovement: ImprovementMetrics = {
      lut: 100, ff: 1000, dsp: 0, bram: 0, fmax_mhz: 200, latency_cycles: 10, ii: 4,
    };
    const weakVerdict = evaluateImprovementTargets(baseline, weakImprovement, ["improve-fmax"]);
    expect(weakVerdict.overall).toBe(false);
    expect(weakVerdict.targets[0]).toMatchObject({
      target: "improve-fmax",
      satisfied: false,
      new_value: 200,
    });

    // A meaningful 220 MHz still doesn't clear the 217 bar by enough... actually 220 > 217 does pass.
    // Confirm a real jump (250 MHz) passes the rule.
    const strongImprovement: ImprovementMetrics = {
      ...baseline, fmax_mhz: 250,
    };
    const strongVerdict = evaluateImprovementTargets(baseline, strongImprovement, ["improve-fmax"]);
    expect(strongVerdict.overall).toBe(true);
  });

  it("improve-fmax above the floor falls back to pure relative bump (no additive penalty)", () => {
    // Baseline 350 MHz, already above the 300 MHz floor. The additive
    // component min(300, 350+50)=300 is then DOMINATED by the relative
    // 350*1.05=367.5, so required = 367.5 (the pure relative bump). The
    // test asserts a 5% lift exactly at 368 MHz satisfies and 360 does not.
    const baseline: ImprovementMetrics = {
      lut: 100, ff: 1000, dsp: 0, bram: 0, fmax_mhz: 350, latency_cycles: 10, ii: 4,
    };
    const justBelowRelative = evaluateImprovementTargets(
      baseline,
      { ...baseline, fmax_mhz: 360 },
      ["improve-fmax"],
    );
    expect(justBelowRelative.targets[0].satisfied).toBe(false);

    const justAboveRelative = evaluateImprovementTargets(
      baseline,
      { ...baseline, fmax_mhz: 368 },
      ["improve-fmax"],
    );
    expect(justAboveRelative.targets[0].satisfied).toBe(true);
  });

  it("fails the whole improvement when any target misses its threshold", () => {
    const verdict = evaluateImprovementTargets(
      { lut: 100, ff: 1000, dsp: 0, bram: 0, fmax_mhz: 200, latency_cycles: 10, ii: 4 },
      { lut: 96, ff: 950, dsp: 1, bram: 0, fmax_mhz: 205, latency_cycles: 10, ii: 5 },
      ["reduce-lut", "use-bram"],
    );

    expect(verdict.overall).toBe(false);
    expect(verdict.targets).toMatchObject([
      { target: "reduce-lut", satisfied: false },
      { target: "use-bram", satisfied: false },
    ]);
  });

  it("rejects token BRAM usage without meaningful LUT or FF reduction", () => {
    const baseline: ImprovementMetrics = {
      lut: 1000,
      ff: 1000,
      dsp: 1,
      bram: 0,
      fmax_mhz: 250,
      latency_cycles: 10,
      ii: 4,
    };
    const tokenBram = evaluateImprovementTargets(
      baseline,
      { ...baseline, bram: 7, lut: 1001, ff: 1001 },
      ["use-bram"],
    );
    expect(tokenBram.overall).toBe(false);

    const realBram = evaluateImprovementTargets(
      baseline,
      { ...baseline, bram: 15, lut: 900, ff: 990 },
      ["use-bram"],
    );
    expect(realBram.overall).toBe(true);
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
      targets: ["reduce-lut", "use-dsp"],
      keepReference: true,
    });
  });
});

describe("parseImproveSweepCliArgs", () => {
  it("parses sweep preset, run mode, keep-reference, and module cap", () => {
    expect(parseImproveSweepCliArgs([
      "--preset=reduce-ff",
      "--run",
      "--keep-reference",
      "--max-modules",
      "12",
    ])).toEqual({
      preset: "reduce-ff",
      run: true,
      keepReference: true,
      maxModules: 12,
    });
  });
});

describe("buildImproveSweepPlan", () => {
  it("selects passing modules and assigns recommended target bundles", async () => {
    tempRoot = await mkdtemp(path.join(os.tmpdir(), "nn2rtl-improve-sweep-"));
    const outputRoot = path.join(tempRoot, "output");
    const reportsDir = path.join(outputRoot, "reports");
    await mkdir(reportsDir, { recursive: true });
    const layers: LayerIR[] = [
      {
        ...layerFixture(),
        module_id: "conv_big",
        op_type: "conv2d",
        weight_shape: [512, 512, 3, 3],
        num_weights: 512 * 512 * 3 * 3,
      },
      {
        ...layerFixture(),
        module_id: "add_ff",
        op_type: "add",
        weight_shape: [],
        num_weights: 0,
      },
      {
        ...layerFixture(),
        module_id: "relu_skip",
        op_type: "relu",
        weight_shape: [],
        num_weights: 0,
      },
    ];
    await writeJson(path.join(outputRoot, "layer_ir.json"), {
      model_name: "fixture-net",
      quantization: "int8_symmetric_per_tensor",
      generated_at: "2026-05-02T12:00:00Z",
      layers,
    });
    await writeJson(path.join(reportsDir, "conv_big.vivado.json"), vivadoReport({
      lut_count: 150_000,
      ff_count: 260_000,
      dsp_count: 1,
      fmax_mhz: 205,
    }));
    await writeJson(path.join(reportsDir, "conv_big.results.json"), verifResult());
    await writeJson(path.join(reportsDir, "add_ff.vivado.json"), vivadoReport({
      lut_count: 40_000,
      ff_count: 49_000,
      dsp_count: 0,
      fmax_mhz: 300,
    }));
    await writeJson(path.join(reportsDir, "add_ff.results.json"), verifResult());
    await writeJson(path.join(reportsDir, "relu_skip.vivado.json"), vivadoReport({
      lut_count: 200,
      ff_count: 50,
      dsp_count: 0,
      fmax_mhz: 500,
    }));
    await writeJson(path.join(reportsDir, "relu_skip.results.json"), verifResult());

    const plan = await buildImproveSweepPlan({
      paths: { repoRoot: tempRoot },
      runtime: { now: fixedNow },
    });

    expect(plan.generated_at).toBe("2026-05-02T12:00:00.000Z");
    expect(plan.recommendations.map((item) => item.module_id)).toEqual(["conv_big", "add_ff"]);
    expect(plan.recommendations[0]).toMatchObject({
      module_id: "conv_big",
      targets: ["use-dsp", "reduce-lut", "reduce-ff", "improve-fmax"],
    });
    expect(plan.recommendations[1]).toMatchObject({
      module_id: "add_ff",
      targets: ["reduce-ff"],
    });
  });
});

describe("buildFoundryImprovePrompt", () => {
  it("includes preloaded pattern markdown but omits reference Verilog in improve mode", () => {
    const prompt = buildFoundryImprovePrompt({
      attempt_index: 1,
      module_id: "unit_module",
      targets: ["use-bram"],
      original_module: originalModule(),
      baseline_metrics: { lut: 100, ff: 100, dsp: 0, bram: 0, fmax_mhz: 200, latency_cycles: 10, ii: 4 },
      baseline_vivado_report: vivadoReport(),
      layer_ir: layerFixture(),
      preloaded_rtl_patterns: {
        pattern_markdown: "PATTERN_DOC_SENT_TO_IMPROVE_FOUNDRY",
        reference_verilog: "module reference_should_not_be_sent; endmodule",
        license_notice: null,
      },
      previous_attempts: [],
    });

    expect(prompt).toContain("PATTERN_DOC_SENT_TO_IMPROVE_FOUNDRY");
    expect(prompt).toContain("Reference Verilog is intentionally omitted in improve mode");
    expect(prompt).not.toContain("reference_verilog:");
    expect(prompt).not.toContain("reference_should_not_be_sent");
  });
});

describe("runImprove", () => {
  it("rejects multiple targets so a single Foundry turn cannot combine improvements", async () => {
    const root = await seedProject();
    await expect(runImprove("unit_module", {
      targets: ["use-dsp", "reduce-lut"],
      paths: { repoRoot: root },
      runtime: {
        now: fixedNow,
        foundryFn: foundryMock(),
        assayerFn: vi.fn(async () => verifResult()),
        synthesisFn: vi.fn(async () => vivadoReport()),
      },
    })).rejects.toThrow("single-target primitive");
  });

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
    const synthesisFn = vi.fn(async () => vivadoReport({ lut_count: 90, bram18_equiv: 8 }));

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

describe("runImproveSequence", () => {
  it("runs requested targets one by one and feeds each step the previous improved RTL", async () => {
    const root = await seedProject();
    const calls: Array<{ target: string; original: string }> = [];
    const foundryFn = vi.fn(async (input: FoundryImproveInput) => {
      const target = input.targets[0]!;
      calls.push({
        target,
        original: input.original_module.verilog_source,
      });
      const resultMessage = {
        type: "result",
        subtype: "success",
        result: "{}",
        total_cost_usd: 0,
        modelUsage: {},
        session_id: `session-${target}`,
      } as const;
      return {
        module: {
          module_id: "unit_module",
          spec_hash: "fixture-hash",
          verilog_source: `module unit_module; // ${target} after ${input.original_module.verilog_source.includes("use-dsp") ? "use-dsp" : "original"}\nendmodule\n`,
          generated_by: "Foundry",
          attempt: input.attempt_index,
        },
        result: resultMessage,
        messages: [resultMessage],
        session_id: `session-${target}`,
      };
    });
    const assayerFn = vi.fn(async () => verifResult());
    const synthesisFn = vi.fn(async (module: VerilogModule) => {
      if (module.verilog_source.includes("reduce-lut")) {
        return vivadoReport({ lut_count: 80, dsp_count: 8 });
      }
      return vivadoReport({ lut_count: 100, dsp_count: 8 });
    });

    const result = await runImproveSequence("unit_module", {
      targets: ["use-dsp", "reduce-lut"],
      keepReference: true,
      paths: { repoRoot: root },
      runtime: { now: fixedNow, foundryFn, assayerFn, synthesisFn },
    });

    expect(result.success).toBe(true);
    expect(result.final_action).toBe("kept-as-variant");
    expect(result.sequence_steps).toMatchObject([
      { target: "use-dsp", success: true },
      { target: "reduce-lut", success: true },
    ]);
    expect(calls).toEqual([
      { target: "use-dsp", original: "module unit_module; // original\nendmodule\n" },
      { target: "reduce-lut", original: "module unit_module; // use-dsp after original\nendmodule\n" },
    ]);
    expect(result.final_verdict?.overall).toBe(true);
    expect(result.final_verdict?.targets.map((target) => target.target)).toEqual(["use-dsp", "reduce-lut"]);
    await expect(readFile(path.join(root, "output", "rtl", "unit_module.v"), "utf8"))
      .resolves.toContain("original");
    await expect(readFile(path.join(root, "knowledge", "references", "improved", "unit_module__use-dsp-reduce-lut.v"), "utf8"))
      .resolves.toContain("reduce-lut after use-dsp");
  }, 20_000);

  it("skips a failed target, tries later targets on the last accepted RTL, and gives Retrospector prior-step context", async () => {
    const root = await seedProject();
    const calls: Array<{ target: string; original: string }> = [];
    const foundryFn = vi.fn(async (input: FoundryImproveInput) => {
      const target = input.targets[0]!;
      calls.push({
        target,
        original: input.original_module.verilog_source,
      });
      const resultMessage = {
        type: "result",
        subtype: "success",
        result: "{}",
        total_cost_usd: 0,
        modelUsage: {},
        session_id: `session-${target}`,
      } as const;
      const prior = input.original_module.verilog_source.includes("use-dsp") ? "after-use-dsp" : "original";
      return {
        module: {
          module_id: "unit_module",
          spec_hash: "fixture-hash",
          verilog_source: `module unit_module; // ${target} ${prior}\nendmodule\n`,
          generated_by: "Foundry",
          attempt: input.attempt_index,
        },
        result: resultMessage,
        messages: [resultMessage],
        session_id: `session-${target}`,
      };
    });
    const assayerFn = vi.fn(async () => verifResult());
    const synthesisFn = vi.fn(async (module: VerilogModule) => {
      if (module.verilog_source.includes("reduce-ff")) {
        return vivadoReport({ lut_count: 100, ff_count: 5, dsp_count: 8 });
      }
      if (module.verilog_source.includes("use-dsp")) {
        return vivadoReport({ lut_count: 100, dsp_count: 8 });
      }
      return vivadoReport({ lut_count: 99, dsp_count: 8 });
    });
    const retrospectorInputs: ImprovementRetrospectorInput[] = [];
    const retrospectorFn = vi.fn(async (input: ImprovementRetrospectorInput) => {
      retrospectorInputs.push(input);
      return {
        analysis: "The reduce-lut step is not clearing the threshold.",
        suggestion: "Try a structural LUT reduction while preserving prior DSP mapping.",
      };
    });

    const result = await runImproveSequence("unit_module", {
      targets: ["use-dsp", "reduce-lut", "reduce-ff"],
      keepReference: true,
      paths: { repoRoot: root },
      runtime: { now: fixedNow, foundryFn, assayerFn, synthesisFn, retrospectorFn },
    });

    expect(result.success).toBe(true);
    expect(result.partial_success).toBe(true);
    expect(result.targets).toEqual(["use-dsp", "reduce-ff"]);
    expect(result.completed_targets).toEqual(["use-dsp", "reduce-ff"]);
    expect(result.failed_targets).toEqual(["reduce-lut"]);
    expect(result.unattempted_targets).toEqual([]);
    expect(result.remaining_targets).toEqual(["reduce-lut"]);
    expect(result.overall_success).toBe(false);
    expect(result.sequence_steps).toMatchObject([
      { target: "use-dsp", success: true },
      { target: "reduce-lut", success: false },
      { target: "reduce-ff", success: true },
    ]);
    expect(calls.at(-1)).toEqual({
      target: "reduce-ff",
      original: "module unit_module; // use-dsp original\nendmodule\n",
    });
    expect(retrospectorFn).toHaveBeenCalledTimes(1);
    expect(retrospectorInputs[0].sequence_context).toMatchObject([
      {
        target: "use-dsp",
        final_action: "replaced",
      },
    ]);
    await expect(readFile(path.join(root, "output", "rtl", "unit_module.v"), "utf8"))
      .resolves.toContain("original");
    await expect(readFile(path.join(root, "knowledge", "references", "improved", "unit_module__use-dsp-reduce-ff.v"), "utf8"))
      .resolves.toContain("reduce-ff after-use-dsp");
    const report = JSON.parse(await readFile(result.report_path, "utf8"));
    expect(report).toMatchObject({
      targets: ["use-dsp", "reduce-ff"],
      requested_targets: ["use-dsp", "reduce-lut", "reduce-ff"],
      completed_targets: ["use-dsp", "reduce-ff"],
      failed_targets: ["reduce-lut"],
      unattempted_targets: [],
      remaining_targets: ["reduce-lut"],
      partial_success: true,
      overall_success: false,
    });
  }, 20_000);

  it("rejects a locally passing step when it regresses a prior accepted target", async () => {
    const root = await seedProject();
    const calls: Array<{ target: string; original: string }> = [];
    const foundryFn = vi.fn(async (input: FoundryImproveInput) => {
      const target = input.targets[0]!;
      calls.push({
        target,
        original: input.original_module.verilog_source,
      });
      const resultMessage = {
        type: "result",
        subtype: "success",
        result: "{}",
        total_cost_usd: 0,
        modelUsage: {},
        session_id: `session-${target}`,
      } as const;
      const prior = input.original_module.verilog_source.includes("use-dsp") ? "after-use-dsp" : "original";
      return {
        module: {
          module_id: "unit_module",
          spec_hash: "fixture-hash",
          verilog_source: `module unit_module; // ${target} ${prior}\nendmodule\n`,
          generated_by: "Foundry",
          attempt: input.attempt_index,
        },
        result: resultMessage,
        messages: [resultMessage],
        session_id: `session-${target}`,
      };
    });
    const assayerFn = vi.fn(async () => verifResult());
    const synthesisFn = vi.fn(async (module: VerilogModule) => {
      if (module.verilog_source.includes("reduce-lut")) {
        // This clears reduce-lut locally, but drops DSP usage back to 0.
        // The sequence-level gate must reject it because the prior use-dsp
        // target no longer passes against the original baseline.
        return vivadoReport({ lut_count: 80, ff_count: 10, dsp_count: 0 });
      }
      if (module.verilog_source.includes("reduce-ff")) {
        return vivadoReport({ lut_count: 100, ff_count: 5, dsp_count: 8 });
      }
      return vivadoReport({ lut_count: 100, ff_count: 10, dsp_count: 8 });
    });

    const result = await runImproveSequence("unit_module", {
      targets: ["use-dsp", "reduce-lut", "reduce-ff"],
      keepReference: true,
      paths: { repoRoot: root },
      runtime: { now: fixedNow, foundryFn, assayerFn, synthesisFn },
    });

    expect(result.success).toBe(true);
    expect(result.partial_success).toBe(true);
    expect(result.overall_success).toBe(false);
    expect(result.targets).toEqual(["use-dsp", "reduce-ff"]);
    expect(result.completed_targets).toEqual(["use-dsp", "reduce-ff"]);
    expect(result.failed_targets).toEqual(["reduce-lut"]);
    expect(result.unattempted_targets).toEqual([]);
    expect(result.sequence_steps).toMatchObject([
      { target: "use-dsp", success: true },
      { target: "reduce-lut", success: false, final_action: "replaced" },
      { target: "reduce-ff", success: true },
    ]);
    expect(calls).toEqual([
      { target: "use-dsp", original: "module unit_module; // original\nendmodule\n" },
      { target: "reduce-lut", original: "module unit_module; // use-dsp original\nendmodule\n" },
      { target: "reduce-ff", original: "module unit_module; // use-dsp original\nendmodule\n" },
    ]);
    expect(result.final_verdict?.targets.map((target) => target.target)).toEqual(["use-dsp", "reduce-ff"]);
    await expect(readFile(path.join(root, "knowledge", "references", "improved", "unit_module__use-dsp-reduce-ff.v"), "utf8"))
      .resolves.toContain("reduce-ff after-use-dsp");
    const report = JSON.parse(await readFile(result.report_path, "utf8"));
    expect(report).toMatchObject({
      targets: ["use-dsp", "reduce-ff"],
      requested_targets: ["use-dsp", "reduce-lut", "reduce-ff"],
      completed_targets: ["use-dsp", "reduce-ff"],
      failed_targets: ["reduce-lut"],
      unattempted_targets: [],
      remaining_targets: ["reduce-lut"],
      partial_success: true,
      overall_success: false,
    });
  }, 20_000);
});
