import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { z } from "zod";

import {
  query,
  type OutputFormat,
  type SDKMessage,
  type SDKResultMessage,
} from "./claude-agent-sdk-compat.js";
import {
  AGENT_CONFIG,
  RETROSPECTOR_CONFIG,
  parsePositiveIntEnv,
} from "./config.js";
import {
  CONTRACT_PLANS,
  applyContractPlan,
  appendRunLog,
  createOrchestratorRuntime,
  findLayer,
  loadPluginAgentDefinition,
  pathExists,
  readJsonFile,
  requireStructuredOutput,
  writeJsonFile,
  type ContractPlan,
  type OrchestratorRuntime,
} from "./orchestrate.js";
import {
  layerIrSchema,
  pipelineIrSchema,
  retrospectorAdviceSchema,
  synthesisReportSchema,
  verifResultSchema,
  verilogModuleSchema,
} from "./schemas.js";
import type {
  LayerIR,
  PipelineIR,
  RetrospectorAdvice,
  VerifResult,
  VerilogModule,
} from "./types.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const sdkRoot = path.resolve(
  __dirname,
  path.basename(__dirname) === "dist" ? ".." : ".",
);
const defaultRepoRoot = path.resolve(sdkRoot, "..");
const pluginPath = path.resolve(defaultRepoRoot, "nn2rtl-plugin");

export const IMPROVEMENT_TARGETS = [
  "use-dsp",
  "use-bram",
  "reduce-lut",
  "reduce-latency",
  "increase-throughput",
] as const;

export type ImprovementTarget = typeof IMPROVEMENT_TARGETS[number];
export type SynthesisReport = z.infer<typeof synthesisReportSchema>;

export type ImprovementMetrics = {
  lut: number;
  dsp: number;
  bram: number;
  latency_cycles?: number;
  ii?: number;
};

export type ImprovementCheckerConfig = {
  useDspThresholdMin: number;
  reduceLutMinDelta: number;
};

export const DEFAULT_IMPROVEMENT_CHECKER_CONFIG: ImprovementCheckerConfig = {
  useDspThresholdMin: parsePositiveIntEnv(process.env, "NN2RTL_IMPROVE_USE_DSP_MIN", 8),
  reduceLutMinDelta: Number(process.env.NN2RTL_IMPROVE_REDUCE_LUT_MIN_DELTA ?? "") || 0.05,
};

export type ImprovementTargetResult = {
  target: ImprovementTarget;
  satisfied: boolean;
  baseline_value?: number;
  new_value?: number;
  required: string;
  reason: string;
};

export type ImprovementVerdict = {
  overall: boolean;
  targets: ImprovementTargetResult[];
};

export type ImprovePaths = {
  repoRoot: string;
  outputRoot: string;
  reportsDir: string;
  rtlDir: string;
  knowledgeRoot: string;
};

export type FoundryImproveInput = {
  attempt_index: number;
  module_id: string;
  targets: ImprovementTarget[];
  original_module: VerilogModule;
  baseline_metrics: ImprovementMetrics;
  baseline_vivado_report: SynthesisReport;
  layer_ir: LayerIR;
  previous_attempts: ImprovementAttemptRecord[];
  resume_session_id?: string;
  retrospector_advice?: RetrospectorAdvice;
};

export type FoundryImproveResult = {
  module: VerilogModule;
  result?: SDKResultMessage;
  messages?: SDKMessage[];
  session_id?: string | null;
};

export type ImprovementRetrospectorInput = {
  module_id: string;
  targets: ImprovementTarget[];
  original_module: VerilogModule;
  baseline_metrics: ImprovementMetrics;
  attempts: ImprovementAttemptRecord[];
};

export type ImproveRuntime = Pick<OrchestratorRuntime, "now" | "queryFn" | "assayerFn" | "synthesisFn"> & {
  foundryFn: (input: FoundryImproveInput, runtime: ImproveRuntime) => Promise<FoundryImproveResult>;
  retrospectorFn: (input: ImprovementRetrospectorInput, runtime: ImproveRuntime) => Promise<RetrospectorAdvice>;
};

export type ImproveRuntimeOverrides = Partial<Pick<ImproveRuntime, "now" | "queryFn" | "assayerFn" | "synthesisFn" | "foundryFn" | "retrospectorFn">>;

export type ImprovementAttemptRecord = {
  attempt_index: number;
  module: VerilogModule;
  session_id?: string | null;
  verilog_path: string;
  assayer_result?: VerifResult;
  vivado_report?: SynthesisReport;
  metrics?: ImprovementMetrics;
  verdict?: ImprovementVerdict;
  failed_gate: "verilator" | "vivado" | "improvement_checker" | null;
  messages?: SDKMessage[];
};

export type ImproveResult = {
  module_id: string;
  targets: ImprovementTarget[];
  final_action: "replaced" | "kept-as-variant" | "no-change";
  success: boolean;
  baseline_metrics: ImprovementMetrics;
  attempts: ImprovementAttemptRecord[];
  final_verdict?: ImprovementVerdict;
  report_path: string;
  committed_module_path?: string;
  archived_original_path?: string;
  improved_reference_path?: string;
  retrospector_advice?: RetrospectorAdvice;
};

export type RunImproveOptions = {
  targets: ImprovementTarget[];
  keepReference?: boolean;
  runtime?: ImproveRuntimeOverrides;
  paths?: Partial<ImprovePaths>;
  checkerConfig?: Partial<ImprovementCheckerConfig>;
};

export type ImproveCliArgs = {
  moduleId: string;
  targets: ImprovementTarget[];
  keepReference: boolean;
};

function toOutputFormat(schema: z.ZodType): OutputFormat {
  return {
    type: "json_schema",
    schema: z.toJSONSchema(schema) as Record<string, unknown>,
  };
}

// Improve Foundry SHOULD emit metadata only and persist `verilog_source`
// via `mcp__nn2rtl-tools__write_verilog`. In practice Opus has a strong
// prior from `foundry.md` to inline the source in the final JSON, and
// even with the addendum it sometimes does that anyway. So the schema
// accepts BOTH shapes:
//   - metadata-only: `{module_id, spec_hash, generated_by, attempt}`
//   - full:          `{module_id, spec_hash, verilog_source, generated_by, attempt}`
// The orchestrator's hydrate step then either reads the .v from disk
// (write_verilog path) OR writes the inline source to disk first (inline
// path). Either way the canonical RTL ends up at the expected location
// before Verilator + Vivado run, and the improve run survives Foundry's
// choice of output style.
const verilogModuleAgentOutputSchema = verilogModuleSchema
  .omit({ verilog_source: true })
  .extend({
    verilog_source: z.string().optional(),
  });
type VerilogModuleAgentOutput = z.infer<typeof verilogModuleAgentOutputSchema>;
const verilogModuleAgentOutputFormat = toOutputFormat(verilogModuleAgentOutputSchema);
const retrospectorAdviceOutputFormat = toOutputFormat(retrospectorAdviceSchema);

function isSdkResultMessage(message: SDKMessage): message is SDKResultMessage {
  return message.type === "result" && "total_cost_usd" in message && "modelUsage" in message;
}

export function defaultImprovePaths(repoRoot = defaultRepoRoot): ImprovePaths {
  const outputRoot = path.join(repoRoot, "output");
  return {
    repoRoot,
    outputRoot,
    reportsDir: path.join(outputRoot, "reports"),
    rtlDir: path.join(outputRoot, "rtl"),
    knowledgeRoot: path.join(repoRoot, "knowledge"),
  };
}

function resolveImprovePaths(overrides: Partial<ImprovePaths> = {}): ImprovePaths {
  const base = defaultImprovePaths(overrides.repoRoot ?? defaultRepoRoot);
  return { ...base, ...overrides };
}

function sanitizePathPart(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 100) || "improve";
}

function improvementStamp(date: Date): string {
  return date.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
}

function uniqueTargets(targets: ImprovementTarget[]): ImprovementTarget[] {
  const requested = new Set(targets);
  return IMPROVEMENT_TARGETS.filter((target) => requested.has(target));
}

export function parseImprovementTargets(raw: string): ImprovementTarget[] {
  const allowed = new Set<string>(IMPROVEMENT_TARGETS);
  const targets = raw
    .split(",")
    .map((target) => target.trim())
    .filter(Boolean);
  if (targets.length === 0) {
    throw new Error("--targets requires at least one target.");
  }
  for (const target of targets) {
    if (!allowed.has(target)) {
      throw new Error(
        `Unknown improvement target '${target}'. Allowed targets: ${IMPROVEMENT_TARGETS.join(", ")}.`,
      );
    }
  }
  return uniqueTargets(targets as ImprovementTarget[]);
}

export function parseImproveCliArgs(argv: string[]): ImproveCliArgs {
  const positional: string[] = [];
  let targets: ImprovementTarget[] | null = null;
  let keepReference = false;

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--keep-reference") {
      keepReference = true;
    } else if (arg === "--targets") {
      const next = argv[++i];
      if (next === undefined || next.startsWith("--")) {
        throw new Error("--targets requires a comma-separated target list.");
      }
      targets = parseImprovementTargets(next);
    } else if (arg.startsWith("--targets=")) {
      targets = parseImprovementTargets(arg.slice("--targets=".length));
    } else if (arg.startsWith("--")) {
      throw new Error(`Unknown improve flag '${arg}'.`);
    } else {
      positional.push(arg);
    }
  }

  if (positional.length !== 1 || targets === null) {
    throw new Error(
      "Usage: nn2rtl improve <module_id> --targets=<target1>,<target2>,... [--keep-reference]",
    );
  }

  return { moduleId: positional[0], targets, keepReference };
}

function metricValue(metrics: ImprovementMetrics, target: ImprovementTarget): number | undefined {
  switch (target) {
    case "use-dsp":
      return metrics.dsp;
    case "use-bram":
      return metrics.bram;
    case "reduce-lut":
      return metrics.lut;
    case "reduce-latency":
      return metrics.latency_cycles;
    case "increase-throughput":
      return metrics.ii;
  }
}

function missingMetric(target: ImprovementTarget): ImprovementTargetResult {
  return {
    target,
    satisfied: false,
    required: "baseline and new metrics must both contain this metric",
    reason: `Target '${target}' cannot be checked because a required metric is missing.`,
  };
}

export function evaluateImprovementTargets(
  baseline: ImprovementMetrics,
  next: ImprovementMetrics,
  targets: ImprovementTarget[],
  config: ImprovementCheckerConfig = DEFAULT_IMPROVEMENT_CHECKER_CONFIG,
): ImprovementVerdict {
  const results = uniqueTargets(targets).map((target): ImprovementTargetResult => {
    const baseValue = metricValue(baseline, target);
    const newValue = metricValue(next, target);
    if (baseValue === undefined || newValue === undefined) {
      return missingMetric(target);
    }

    switch (target) {
      case "use-dsp": {
        const required = Math.max(baseline.dsp + 1, config.useDspThresholdMin);
        const satisfied = next.dsp >= required;
        return {
          target,
          satisfied,
          baseline_value: baseline.dsp,
          new_value: next.dsp,
          required: `new.dsp >= max(baseline.dsp + 1, ${config.useDspThresholdMin}) = ${required}`,
          reason: satisfied ? "DSP usage target satisfied." : `new.dsp=${next.dsp} is below required ${required}.`,
        };
      }
      case "use-bram": {
        const satisfied = next.bram > 0;
        return {
          target,
          satisfied,
          baseline_value: baseline.bram,
          new_value: next.bram,
          required: "new.bram > 0",
          reason: satisfied ? "BRAM usage target satisfied." : "new.bram is 0.",
        };
      }
      case "reduce-lut": {
        const required = baseline.lut * (1 - config.reduceLutMinDelta);
        const satisfied = next.lut < required;
        return {
          target,
          satisfied,
          baseline_value: baseline.lut,
          new_value: next.lut,
          required: `new.lut < baseline.lut * (1 - ${config.reduceLutMinDelta}) = ${required}`,
          reason: satisfied ? "LUT reduction target satisfied." : `new.lut=${next.lut} is not below ${required}.`,
        };
      }
      case "reduce-latency": {
        const satisfied = next.latency_cycles! < baseline.latency_cycles!;
        return {
          target,
          satisfied,
          baseline_value: baseline.latency_cycles,
          new_value: next.latency_cycles,
          required: "new.latency_cycles < baseline.latency_cycles",
          reason: satisfied ? "Latency target satisfied." : `new.latency_cycles=${next.latency_cycles} is not lower than baseline ${baseline.latency_cycles}.`,
        };
      }
      case "increase-throughput": {
        const satisfied = next.ii! < baseline.ii!;
        return {
          target,
          satisfied,
          baseline_value: baseline.ii,
          new_value: next.ii,
          required: "new.ii < baseline.ii",
          reason: satisfied ? "Throughput target satisfied." : `new.ii=${next.ii} is not lower than baseline ${baseline.ii}.`,
        };
      }
    }
  });

  return {
    overall: results.every((result) => result.satisfied),
    targets: results,
  };
}

function metricsFromReports(synthesis: SynthesisReport, verif?: VerifResult): ImprovementMetrics {
  return {
    lut: synthesis.lut_count,
    dsp: synthesis.dsp_count,
    bram: synthesis.bram18_equiv || synthesis.bram18_count + synthesis.bram36_count * 2,
    latency_cycles: verif?.timing_actual_cycles !== undefined && verif.timing_actual_cycles >= 0
      ? verif.timing_actual_cycles
      : undefined,
    ii: verif && typeof (verif as { initiation_interval_cycles?: unknown }).initiation_interval_cycles === "number"
      ? (verif as { initiation_interval_cycles: number }).initiation_interval_cycles
      : undefined,
  };
}

export const improvementMetricsSchema = z
  .object({
    lut: z.number().nonnegative(),
    dsp: z.number().nonnegative(),
    bram: z.number().nonnegative(),
    latency_cycles: z.number().nonnegative().optional(),
    ii: z.number().nonnegative().optional(),
  })
  .strict();

async function readOptionalJson<T>(filePath: string, schema: z.ZodType<T>): Promise<T | null> {
  if (!(await pathExists(filePath))) return null;
  return readJsonFile<T>(filePath, schema);
}

async function loadBaselineMetrics(paths: ImprovePaths, moduleId: string): Promise<{
  vivadoReport: SynthesisReport;
  verifResult?: VerifResult;
  metrics: ImprovementMetrics;
}> {
  const metricsPath = path.join(paths.reportsDir, `${moduleId}.metrics.json`);
  const metrics = await readOptionalJson<ImprovementMetrics>(metricsPath, improvementMetricsSchema);
  const vivadoReport = await readJsonFile<SynthesisReport>(
    path.join(paths.reportsDir, `${moduleId}.vivado.json`),
    synthesisReportSchema,
  );
  const verifResult = await readOptionalJson<VerifResult>(
    path.join(paths.reportsDir, `${moduleId}.results.json`),
    verifResultSchema,
  ) ?? undefined;
  return {
    vivadoReport,
    verifResult,
    metrics: metrics ?? metricsFromReports(vivadoReport, verifResult),
  };
}

async function loadOriginalModule(paths: ImprovePaths, moduleId: string): Promise<VerilogModule> {
  const metaPath = path.join(paths.rtlDir, `${moduleId}.meta.json`);
  if (await pathExists(metaPath)) {
    return readJsonFile<VerilogModule>(metaPath, verilogModuleSchema);
  }
  const verilogPath = path.join(paths.rtlDir, `${moduleId}.v`);
  const verilogSource = await readFile(verilogPath, "utf8");
  return {
    module_id: moduleId,
    spec_hash: "unknown",
    verilog_source: verilogSource,
    generated_by: "Foundry",
    attempt: 1,
  };
}

/**
 * Re-apply the contract that was selected for this module during the
 * original pipeline run. The on-disk `output/layer_ir.json` is the BASE
 * LayerIR (pre-contract-plan). The runtime layer that actually shipped to
 * Foundry / the assayer / Vivado was the result of `applyContractPlan` —
 * with `contract_id`, `io_mode`, `channel_tile`, and the contract-specific
 * `input_width_bits` / `output_width_bits`. We recover that runtime shape
 * by reading the contract id out of the persisted `.meta.json`'s
 * `spec_hash` (which encodes `iotiled-streaming_tile32` etc.) and applying
 * the same plan again. Without this, the bus-width validator inside
 * `runAssayerDeterministic` rejects the layer because the base shape's
 * `input_width_bits` / `output_width_bits` exceed flat-bus's 4096-bit cap
 * for any layer that was originally escalated to a heavier contract.
 */
function inferContractIdFromSpecHash(specHash: string): ContractPlan["id"] {
  // Spec-hash suffix uses `_io<contract>` so we look for an explicit infix.
  for (const plan of CONTRACT_PLANS) {
    if (plan.id === "flat-bus") continue;
    if (specHash.includes(`_io${plan.id}_`) || specHash.includes(`_io${plan.id}`)) {
      return plan.id;
    }
  }
  return "flat-bus";
}

async function loadLayer(paths: ImprovePaths, moduleId: string): Promise<LayerIR> {
  const pipelineIr = await readJsonFile<PipelineIR>(
    path.join(paths.outputRoot, "layer_ir.json"),
    pipelineIrSchema,
  );
  const baseLayer = layerIrSchema.parse(findLayer(pipelineIr, moduleId));
  const metaPath = path.join(paths.rtlDir, `${moduleId}.meta.json`);
  if (!(await pathExists(metaPath))) {
    return baseLayer;
  }
  const meta = await readJsonFile<VerilogModule>(metaPath, verilogModuleSchema);
  const contractId = inferContractIdFromSpecHash(meta.spec_hash);
  if (contractId === "flat-bus") {
    return baseLayer;
  }
  const plan = CONTRACT_PLANS.find((candidate) => candidate.id === contractId);
  if (!plan) return baseLayer;
  return applyContractPlan(baseLayer, plan);
}

function extractSessionId(messages: SDKMessage[] | undefined, result?: SDKResultMessage): string | null {
  const candidates: unknown[] = [...(messages ?? [])].reverse();
  if (result) candidates.unshift(result);
  for (const candidate of candidates) {
    if (
      typeof candidate === "object" &&
      candidate !== null &&
      "session_id" in candidate &&
      typeof (candidate as { session_id?: unknown }).session_id === "string"
    ) {
      return (candidate as { session_id: string }).session_id;
    }
  }
  return null;
}

/**
 * Per-target guidance Foundry sees before the prompt's evidence section.
 * Each entry has three parts: GOAL (what the deterministic checker is going
 * to measure), HOW (concrete RTL idioms that move the metric), PITFALLS
 * (failure modes observed in past runs or known to break Vivado inference).
 *
 * The orchestrator runs `evaluateImprovementTargets` after Vivado, so the
 * agent doesn't have to estimate — it just has to produce RTL that the
 * checker accepts. Keep this guidance specific to the metric and the
 * Vivado/UltraScale+ inference rules; do not list general "good Verilog"
 * advice here.
 */
const TARGET_GUIDANCE: Record<ImprovementTarget, string> = {
  "use-dsp": [
    "GOAL: map multipliers / MAC operations onto DSP48E2 slices instead of LUT-based ripple multipliers.",
    "HOW (correctness-preserving levers, in order of preference):",
    "  - Annotate the multiply with `(* use_dsp = \"yes\" *)` on the line before the assignment, OR factor the multiply into a registered intermediate `reg signed [W-1:0] mul_q; always @(posedge clk) mul_q <= a * b;` so Vivado can pattern-match a DSP cell. This alone moves the existing scalar multiply into a DSP without changing throughput.",
    "  - Make both operands `signed [N-1:0]` of the same width before the multiply. Mixed signed/unsigned or width-mismatched operands routinely keep multipliers in LUT.",
    "  - Register the multiplier output BEFORE feeding into shifts, saturation, or accumulation. A direct `(a*b) >>> N` combinational chain is one of the most common DSP-inference rejections.",
    "  - Controlled banking / parallel MACs are ALLOWED if you keep correctness and timing. If a single annotated multiply does not push DSP count to the required threshold, you may bank the MAC across N parallel lanes (each lane its own `(* use_dsp = \"yes\" *)` registered multiply) and a balanced adder tree — provided the public interface, latency contract, and bit-exact output are unchanged. This is often necessary when the threshold is much higher than the baseline.",
    "PITFALLS:",
    "  - Do NOT blindly unroll a serial loop into a structurally different datapath that changes when `valid_out` fires or which weights pair with which inputs — that breaks Verilator. Banking is correctness-preserving only when each lane consumes exactly the same operands the original sequential MAC would have, just in parallel.",
    "  - Multiplications inside `for ... if (...)` conditional generate-for blocks may map differently across Vivado versions; prefer unconditional registered multiplies inside an `always_ff`.",
    "  - Adding lanes increases LUT fan-out around the adder tree. If you also have `reduce-lut` in the targets, balance carefully — banking too aggressively trades DSPs for LUTs.",
  ].join("\n"),
  "use-bram": [
    "GOAL: store weight (and any large constant) memories in BRAM18/BRAM36 instead of distributed LUT-RAM.",
    "HOW:",
    "  - Annotate the array on its declaration line: `(* rom_style = \"block\", ram_style = \"block\" *) reg signed [7:0] weights [0:OC*K_TOTAL-1];`. The attribute MUST sit immediately before the `reg` declaration.",
    "  - Reads MUST be synchronous: `reg signed [7:0] w_q; always @(posedge clk) w_q <= weights[addr];`. Async reads (`assign w = weights[addr];`) force LUTRAM no matter what attribute is set.",
    "  - Initialize with `$readmemh` inside an `initial begin ... end` block. Element-by-element assignment in initial blocks (`weights[0] = ..; weights[1] = ..;`) defeats Vivado's BRAM/ROM init pattern matching.",
    "PITFALLS:",
    "  - Adding a registered read stage shifts the MAC schedule by one cycle; verify the FSM still drives `valid_out` after exactly `pipeline_latency_cycles` cycles.",
    "  - BRAM has fixed port counts. A weight memory read by N parallel MAC lanes either needs N replicated BRAMs (depth/width tradeoff) or a banked layout — see `weight_bank_paths` in the LayerIR.",
  ].join("\n"),
  "reduce-lut": [
    "GOAL: reduce CLB LUT count by at least the configured delta (`new.lut < baseline.lut * (1 - reduceLutMinDelta)`).",
    "HOW:",
    "  - Move large constant tables into BRAM-backed ROMs (overlaps with `use-bram`).",
    "  - Replace wide `case` / nested `if` chains over a discrete encoder with arithmetic / table lookups.",
    "  - Combine repeated comparators against a counter (e.g. `cnt == 0 || cnt == 1 || ...`) into a single bound check.",
    "PITFALLS:",
    "  - Do not eliminate logic that's required by the contract. The Verilator gate runs first; functional regressions are caught before the LUT count is even read.",
    "  - LUT count includes `LUT as Logic` AND `LUT as Memory` rows — moving distributed RAM to BRAM moves the count from `LUT as Memory` to BRAM18 only if the read is registered (see `use-bram`).",
  ].join("\n"),
  "reduce-latency": [
    "GOAL: reduce cycles-to-first-output (`new.latency_cycles < baseline.latency_cycles`).",
    "HOW:",
    "  - Identify pipeline stages that exist only because the original was conservative. The setup WNS in the baseline Vivado report tells you how much slack each stage has — merging two stages is safe iff the merged combinational path still meets the period.",
    "  - For multi-cycle MAC loops, look at whether successive lanes can share a stage rather than each having its own.",
    "PITFALLS:",
    "  - This is the only target that legitimately changes `pipeline_latency_cycles`. The deterministic verifier compares `timing_actual_cycles` against `timing_expected_cycles` from the LayerIR — both must move together. The improve flow CANNOT regenerate goldens, so latency reduction is only safe if the testbench treats latency as the cycle of the FIRST `valid_out` pulse and not as a fixed checkpoint. Read the goldens' header before changing this.",
    "  - Removing a register that breaks DSP inference (see `use-dsp`) loses LUT savings instantly.",
  ].join("\n"),
  "increase-throughput": [
    "GOAL: lower initiation interval (II) — the design accepts a new input every fewer cycles.",
    "HOW:",
    "  - Identify the resource that's serialized: shared single-port BRAM, single MAC lane, single accumulator. Replicate it (BRAM banks, MAC lanes) so independent inputs don't fight for the same port.",
    "  - Switch single-port BRAM to true dual-port (`RAMB36E2` with two independent read addresses) when read patterns conflict.",
    "PITFALLS:",
    "  - II=1 designs need every BRAM read to be deterministic from the inputs. Address-pipelining can make II=1 hard to verify functionally.",
    "  - Doubling MACs doubles DSP usage. Confirm the board has the budget (ZCU102 / XCZU9EG has 2,520 DSP48E2 — plenty for a single layer, less so for a whole pipeline).",
  ].join("\n"),
};

const TARGETS_COMMON_GUIDANCE = [
  "COMMON RULES (apply to every target):",
  "  - The public interface (clk, rst_n, valid_in, ready_in, data_in, valid_out, data_out) is byte-identical to the original. Same widths, same names, same directions. The verifier rejects any port-level change.",
  "  - The `spec_hash` you return MUST equal the original's `spec_hash`. The orchestrator overrides it anyway, but emitting a different value indicates you misunderstood the contract.",
  "  - Functional behavior is verified bit-exact by Verilator using the same goldens that pass against the original. Any output-value change is a hard fail and burns the attempt.",
  "  - Do NOT use `Bash`, `Read`, or `Write` tools. There are none registered for this turn. The original RTL is fully embedded below.",
  "  - Do NOT introduce non-synthesizable constructs: `#delay` outside testbench, `force`/`release`, `initial weights[i] = expr` outside `$readmemh`, dynamic `for` loops outside generate, `wait`, `fork/join`.",
  "  - Do NOT widen, narrow, or reorder ports. Internal width changes are fine.",
  "  - Output: a single JSON object matching `verilogModuleSchema` — `module_id`, `spec_hash`, `verilog_source` (the FULL improved RTL as a JSON-escaped string), `generated_by: \"Foundry\"`, `attempt: <attempt_index>`. No markdown fences, no commentary, nothing else.",
].join("\n");

function summarizeBaselineMetrics(metrics: ImprovementMetrics): string {
  const parts = [
    `LUT: ${metrics.lut}`,
    `DSP: ${metrics.dsp}`,
    `BRAM18-equivalent: ${metrics.bram}`,
  ];
  if (metrics.latency_cycles !== undefined) {
    parts.push(`latency_cycles: ${metrics.latency_cycles}`);
  }
  if (metrics.ii !== undefined) {
    parts.push(`II: ${metrics.ii}`);
  }
  return parts.join(", ");
}

function checkerRulesForTargets(
  targets: ImprovementTarget[],
  baseline: ImprovementMetrics,
  config: ImprovementCheckerConfig = DEFAULT_IMPROVEMENT_CHECKER_CONFIG,
): string {
  const lines: string[] = [];
  for (const target of targets) {
    switch (target) {
      case "use-dsp": {
        const required = Math.max(baseline.dsp + 1, config.useDspThresholdMin);
        lines.push(
          `  - use-dsp: pass iff new.dsp >= ${required} (= max(${baseline.dsp} + 1, ${config.useDspThresholdMin})).`,
        );
        break;
      }
      case "use-bram":
        lines.push(`  - use-bram: pass iff new.bram18_equiv > 0 (baseline = ${baseline.bram}).`);
        break;
      case "reduce-lut": {
        const required = baseline.lut * (1 - config.reduceLutMinDelta);
        lines.push(
          `  - reduce-lut: pass iff new.lut < ${required.toFixed(0)} (= baseline.lut ${baseline.lut} * (1 - ${config.reduceLutMinDelta})).`,
        );
        break;
      }
      case "reduce-latency":
        lines.push(
          `  - reduce-latency: pass iff new.latency_cycles < ${baseline.latency_cycles ?? "(missing — baseline metrics need timing_actual_cycles)"}.`,
        );
        break;
      case "increase-throughput":
        lines.push(
          `  - increase-throughput: pass iff new.ii < ${baseline.ii ?? "(missing — baseline metrics need initiation_interval_cycles)"}.`,
        );
        break;
    }
  }
  return ["DETERMINISTIC CHECKER RULES (ALL must pass):", ...lines].join("\n");
}

function summarizeAttemptForPrompt(attempt: ImprovementAttemptRecord): Record<string, unknown> {
  return {
    attempt_index: attempt.attempt_index,
    failed_gate: attempt.failed_gate,
    verilog_source: attempt.module.verilog_source,
    assayer_result: attempt.assayer_result
      ? {
          status: attempt.assayer_result.status,
          status_class: attempt.assayer_result.status_class,
          failure_class: attempt.assayer_result.failure_class,
          first_mismatch_index: attempt.assayer_result.first_mismatch_index,
          max_error: attempt.assayer_result.max_error,
          fix_hint: attempt.assayer_result.fix_hint,
        }
      : undefined,
    vivado_report: attempt.vivado_report
      ? {
          success: attempt.vivado_report.success,
          timing_met: attempt.vivado_report.timing_met,
          lut_count: attempt.vivado_report.lut_count,
          ff_count: attempt.vivado_report.ff_count,
          dsp_count: attempt.vivado_report.dsp_count,
          bram18_equiv: attempt.vivado_report.bram18_equiv,
          fmax_mhz: attempt.vivado_report.fmax_mhz,
          setup_wns_ns: attempt.vivado_report.setup_wns_ns ?? attempt.vivado_report.wns_ns,
          hold_wns_ns: attempt.vivado_report.hold_wns_ns,
        }
      : undefined,
    metrics: attempt.metrics,
    verdict: attempt.verdict,
  };
}

function buildFoundryImprovePrompt(
  input: FoundryImproveInput,
  config: ImprovementCheckerConfig = DEFAULT_IMPROVEMENT_CHECKER_CONFIG,
): string {
  const targets = uniqueTargets(input.targets);
  const targetGuidance = targets
    .map((target) => `[${target}]\n${TARGET_GUIDANCE[target]}`)
    .join("\n\n");
  const sections: string[] = [
    "You are Foundry in quality-improvement mode.",
    "",
    "== TASK ==",
    `Improve the passing RTL for module '${input.module_id}' so that the deterministic improvement checker accepts it for ALL of: ${targets.join(", ")}.`,
    "Correctness is verified first: the same Verilator goldens that pass against the original will run against your output, and any value mismatch fails the attempt instantly.",
    input.attempt_index === 1
      ? "This is attempt 1; you are seeing the original passing RTL and the baseline Vivado report."
      : `This is attempt ${input.attempt_index}; the orchestrator has resumed your prior session — your earlier attempt(s) and their gate failures are listed under ATTEMPT HISTORY. Do NOT repeat a failed approach; pick a different lever from the per-target guidance.`,
    input.retrospector_advice
      ? "A Retrospector advisory is included in this turn (attempt 3 only). Treat its analysis and suggestion as evidence-based; do not interpret it as permission to alter functionality."
      : "",
    "",
    "== TARGET GUIDANCE ==",
    targetGuidance,
    "",
    checkerRulesForTargets(targets, input.baseline_metrics, config),
    "",
    TARGETS_COMMON_GUIDANCE,
    "",
    "== BASELINE EVIDENCE ==",
    `module_id: ${input.module_id}`,
    `op_type: ${input.layer_ir.op_type}`,
    `weight_shape: [${input.layer_ir.weight_shape.join(", ")}]`,
    `contract_id: ${input.layer_ir.contract_id ?? "flat-bus"}`,
    `io_mode: ${input.layer_ir.io_mode ?? "packed_full"}`,
    `channel_tile: ${input.layer_ir.channel_tile ?? "n/a"}`,
    `pipeline_latency_cycles (LayerIR): ${input.layer_ir.pipeline_latency_cycles}`,
    `clock_period_ns: ${input.layer_ir.clock_period_ns}`,
    `Baseline metrics: ${summarizeBaselineMetrics(input.baseline_metrics)}`,
    `Baseline Vivado: setup_wns_ns=${input.baseline_vivado_report.setup_wns_ns ?? input.baseline_vivado_report.wns_ns ?? "?"}, hold_wns_ns=${input.baseline_vivado_report.hold_wns_ns ?? "?"}, fmax_mhz=${input.baseline_vivado_report.fmax_mhz?.toFixed?.(2) ?? "?"}`,
    "",
    "== ORIGINAL RTL (the source of truth — improve this) ==",
    "```verilog",
    input.original_module.verilog_source,
    "```",
  ];

  if (input.previous_attempts.length > 0) {
    sections.push(
      "",
      "== ATTEMPT HISTORY ==",
      JSON.stringify(input.previous_attempts.map(summarizeAttemptForPrompt), null, 2),
    );
  }

  if (input.retrospector_advice) {
    sections.push(
      "",
      "== RETROSPECTOR ADVICE (attempt 3 only) ==",
      JSON.stringify(input.retrospector_advice, null, 2),
    );
  }

  sections.push(
    "",
    "== OUTPUT CONTRACT ==",
    "Return EXACTLY this JSON object and nothing else:",
    `  { "module_id": "${input.module_id}", "spec_hash": "${input.original_module.spec_hash}", "verilog_source": "<full improved RTL as a JSON-escaped string>", "generated_by": "Foundry", "attempt": ${input.attempt_index} }`,
    "Re-emit the FULL improved Verilog source as a single JSON-escaped string. Escape every backslash as `\\\\`, every double quote as `\\\"`, every newline as `\\n`. No markdown fences, no commentary.",
  );

  return sections.filter((line) => line !== undefined).join("\n");
}

async function hydrateImprovedModuleFromDisk(
  metadata: VerilogModuleAgentOutput,
  attemptIndex: number,
  paths: ImprovePaths,
  agentTurnStartTime: Date,
): Promise<VerilogModule> {
  // Two paths to a canonical .v on disk:
  //
  //   1. `write_verilog` was called — the agent persisted the source via
  //      the MCP tool, so `<rtlDir>/<module_id>.v` is fresh (mtime is
  //      after the agent turn started). This is the cost-efficient path.
  //
  //   2. The agent inlined `verilog_source` in the final JSON instead of
  //      calling `write_verilog`. We extract that string from the
  //      structured output and write it to the same canonical .v path
  //      ourselves. The improve run survives the agent's choice of
  //      output style.
  //
  // Either way, after this function returns, the canonical .v reflects
  // THIS attempt's candidate RTL, ready for Verilator + Vivado.
  const verilogPath = path.join(paths.rtlDir, `${metadata.module_id}.v`);
  await mkdir(path.dirname(verilogPath), { recursive: true });

  // Path 2 first: if the agent inlined verilog_source, prefer that — it's
  // the producer's authoritative output for this turn. Only fall back to
  // a fresh-mtime disk file if the agent omitted the inline source.
  if (typeof metadata.verilog_source === "string" && metadata.verilog_source.trim()) {
    await writeFile(verilogPath, metadata.verilog_source, "utf8");
    return {
      module_id: metadata.module_id,
      spec_hash: metadata.spec_hash,
      verilog_source: metadata.verilog_source,
      generated_by: metadata.generated_by,
      attempt: metadata.attempt,
    };
  }

  // Path 1: read whatever write_verilog persisted. The mtime check guards
  // against the case where the agent did NOT call write_verilog AND did
  // NOT inline the source — without that guard we'd silently re-run
  // Verilator on the previous attempt's RTL.
  let source: string;
  let stat: { mtime: Date };
  try {
    [source, stat] = await Promise.all([
      readFile(verilogPath, "utf8"),
      (async () => {
        const { stat: statFn } = await import("node:fs/promises");
        return await statFn(verilogPath);
      })(),
    ]);
  } catch {
    throw new Error(
      `improve Foundry returned metadata for module '${metadata.module_id}' (attempt ${attemptIndex}) ` +
        `but did NOT inline verilog_source AND did NOT persist via mcp__nn2rtl-tools__write_verilog. ` +
        `Expected ${verilogPath} but the file is missing.`,
    );
  }
  if (!source.trim()) {
    throw new Error(
      `improve Foundry left an empty Verilog file for module '${metadata.module_id}' (attempt ${attemptIndex}).`,
    );
  }
  if (stat.mtime.getTime() < agentTurnStartTime.getTime()) {
    throw new Error(
      `improve Foundry returned for module '${metadata.module_id}' (attempt ${attemptIndex}) ` +
        `but the canonical .v at ${verilogPath} is older than the agent turn (mtime=${stat.mtime.toISOString()}, ` +
        `turn started=${agentTurnStartTime.toISOString()}). The agent neither inlined verilog_source nor ` +
        `called mcp__nn2rtl-tools__write_verilog; the improve attempt would silently re-run on the previous RTL.`,
    );
  }
  return {
    module_id: metadata.module_id,
    spec_hash: metadata.spec_hash,
    verilog_source: source,
    generated_by: metadata.generated_by,
    attempt: metadata.attempt,
  };
}

// Improve-mode addendum spliced onto the END of foundry.md's body. The
// canonical Foundry contract (canonical port list, packed-channel bus
// convention, INT8 quantization rules, scale-shift rounding mandate,
// memory-inference hints, common bugs catalog, split-architecture rule
// for spatial convs, invariant-marker policy, sign-extension warnings,
// procedural-declaration scoping warnings, weights_packed_forbidden,
// etc.) all still apply during improvement runs — Foundry must keep them
// or the deterministic Verilator gate will fail. The addendum below adds
// only the improve-specific output contract.
const IMPROVE_MODE_PROMPT_ADDENDUM = [
  "",
  "---",
  "",
  "## Improvement-mode addendum",
  "",
  "You are running in QUALITY-IMPROVEMENT mode. The original RTL has already passed Verilator + Vivado on the configured target part; your job is to produce a functionally-equivalent variant that satisfies the per-target deterministic checker rules listed in the user prompt.",
  "",
  "Every rule in this system prompt about the canonical interface, INT8 quantization, scale-shift rounding, memory inference, the split-architecture for spatial convs, invariant markers, sign extension, and procedural declarations STILL APPLIES. Improvement does not waive correctness — Verilator runs first against the same goldens that pass the original, and any value mismatch fails the attempt instantly.",
  "",
  "Output / persistence contract for improvement runs:",
  "- Persist the improved RTL via the `mcp__nn2rtl-tools__write_verilog` MCP tool BEFORE returning. Use the same `module_id` as the original; the orchestrator reads `<output_dir>/<module_id>.v` from disk after you return.",
  "- Your final JSON message contains METADATA ONLY: `{module_id, spec_hash, generated_by: \"Foundry\", attempt: <attempt_index>}`. Do NOT include `verilog_source` in the final JSON — re-serializing the source as a JSON-escaped string burns output tokens and routinely produces unparseable final messages.",
  "- The orchestrator does not have `Bash`, `Read`, or `Write` available to you for this turn. Everything you need (original RTL, baseline metrics, prior attempts, Retrospector advice on attempt 3) is embedded in the user prompt.",
].join("\n");

function makeDefaultFoundryImproveFn(paths: ImprovePaths): (input: FoundryImproveInput, runtime: ImproveRuntime) => Promise<FoundryImproveResult> {
  return async function defaultFoundryImproveFn(
    input: FoundryImproveInput,
    runtime: ImproveRuntime,
  ): Promise<FoundryImproveResult> {
    const messages: SDKMessage[] = [];
    let finalResult: SDKResultMessage | null = null;
    const agentTurnStartTime = runtime.now();

    // Load the canonical Foundry agent definition (`nn2rtl-plugin/agents/foundry.md`)
    // so improve-mode Foundry sees the same ~200-line system prompt the
    // main pipeline gives it: canonical interface, packed-channel bus
    // convention, INT8 quantization rules, scale-shift rounding, memory
    // inference hints, common bugs catalog, etc. Without this, improve
    // mode runs blind to the codebase's accumulated Verilog contract.
    const foundryAgent = await loadPluginAgentDefinition("foundry");

    for await (const message of runtime.queryFn({
      prompt: buildFoundryImprovePrompt(input),
      options: {
        cwd: defaultRepoRoot,
        model: AGENT_CONFIG.Foundry.model,
        systemPrompt: {
          type: "preset",
          preset: "claude_code",
          append: `${foundryAgent.prompt}${IMPROVE_MODE_PROMPT_ADDENDUM}`,
        },
        // The agent needs `write_verilog` to persist the improved source;
        // everything else (Bash, Read, file Write) stays disabled to keep
        // the call as close to pure-reasoning-with-one-side-effect as
        // possible.
        tools: ["mcp__nn2rtl-tools__write_verilog"],
        allowedTools: ["mcp__nn2rtl-tools__write_verilog"],
        disallowedTools: ["Agent", "Task", "Bash", "Read", "Write"],
        plugins: [{ type: "local", path: pluginPath }],
        outputFormat: verilogModuleAgentOutputFormat,
        maxTurns: AGENT_CONFIG.Foundry.maxTurns,
        ...(input.resume_session_id ? { resume: input.resume_session_id } : {}),
      },
    })) {
      messages.push(message);
      if (isSdkResultMessage(message)) {
        finalResult = message;
      }
    }

    if (!finalResult) {
      throw new Error("No final result message was received for improve Foundry.");
    }
    let metadata: VerilogModuleAgentOutput;
    try {
      metadata = requireStructuredOutput<VerilogModuleAgentOutput>(
        finalResult,
        "improve_foundry",
        verilogModuleAgentOutputSchema,
      );
    } catch (err) {
      // Foundry's structured output is fragile (escapes outside string
      // literals, truncated final messages, BOMs). Recover by synthesizing
      // metadata from `input` (we know module_id, spec_hash, attempt) and
      // hydrating from the on-disk RTL Foundry already persisted via
      // `write_verilog`. Without this fallback, a single malformed JSON byte
      // wastes the entire turn's spend.
      const verilogPath = path.join(paths.rtlDir, `${input.module_id}.v`);
      const verilogStat = await pathExists(verilogPath);
      if (!verilogStat) {
        throw err;
      }
      metadata = {
        module_id: input.module_id,
        spec_hash: input.original_module.spec_hash,
        generated_by: "Foundry",
        attempt: input.attempt_index,
      };
      await appendRunLog({
        event: "improve_foundry_parse_recovery",
        agent: "Foundry",
        module_id: input.module_id,
        attempt: input.attempt_index,
        reason: err instanceof Error ? err.message : String(err),
      });
    }
    const module = await hydrateImprovedModuleFromDisk(metadata, input.attempt_index, paths, agentTurnStartTime);
    return {
      module,
      result: finalResult,
      messages,
      session_id: extractSessionId(messages, finalResult),
    };
  };
}

function buildImproveRetrospectorPrompt(input: ImprovementRetrospectorInput): string {
  const targets = uniqueTargets(input.targets);
  return [
    "You are Retrospector for an RTL quality-improvement run.",
    "Two improvement attempts have failed against deterministic checker rules. Analyze the evidence and emit ONE advisory JSON object Foundry will see on its third attempt.",
    "",
    "== HARD RULES ==",
    "  - Do NOT suggest changing functional behavior. Verilator runs first; correctness is non-negotiable.",
    "  - Do NOT recommend abandoning a target. Foundry must satisfy ALL of the requested targets.",
    "  - Suggest a DIFFERENT optimization lever per target — Foundry already tried what it tried twice; pick something it has not tried.",
    "  - Be specific to the failed gate: a Verilator-fail attempt needs a different fix than a Vivado-fail attempt or an improvement-checker-fail attempt.",
    "",
    "== TARGETS REMAINING UNSATISFIED ==",
    targets.map((target) => `  - ${target}`).join("\n"),
    "",
    "== EVIDENCE ==",
    `module_id: ${input.module_id}`,
    `Baseline metrics: ${summarizeBaselineMetrics(input.baseline_metrics)}`,
    "",
    "Original passing RTL (FYI; do not re-derive):",
    "```verilog",
    input.original_module.verilog_source,
    "```",
    "",
    "Failed attempts:",
    JSON.stringify(input.attempts.map(summarizeAttemptForPrompt), null, 2),
    "",
    "== OUTPUT CONTRACT ==",
    "Return EXACTLY this JSON object:",
    "  { \"analysis\": \"<concise reading of the two failures>\", \"suggestion\": \"<concrete, actionable next strategy>\" }",
    "No markdown, no commentary, no doc_fault flag (improve runs do not own knowledge docs).",
  ].join("\n");
}

async function defaultImproveRetrospectorFn(
  input: ImprovementRetrospectorInput,
  runtime: ImproveRuntime,
): Promise<RetrospectorAdvice> {
  const messages: SDKMessage[] = [];
  let finalResult: SDKResultMessage | null = null;
  for await (const message of runtime.queryFn({
    prompt: buildImproveRetrospectorPrompt(input),
    options: {
      cwd: defaultRepoRoot,
      model: RETROSPECTOR_CONFIG.model,
      systemPrompt: {
        type: "preset",
        preset: "claude_code",
        append: "Return only the requested JSON advisory; do not use tools.",
      },
      outputFormat: retrospectorAdviceOutputFormat,
      maxTurns: RETROSPECTOR_CONFIG.maxTurns,
      tools: [],
      allowedTools: [],
      disallowedTools: ["Agent", "Task", "Bash", "Read", "Write"],
    },
  })) {
    messages.push(message);
    if (isSdkResultMessage(message)) {
      finalResult = message;
    }
  }
  if (!finalResult) {
    throw new Error("No final result message was received for improve Retrospector.");
  }
  return requireStructuredOutput<RetrospectorAdvice>(
    finalResult,
    "improve_retrospector",
    retrospectorAdviceSchema,
  );
}

export function createImproveRuntime(
  overrides: ImproveRuntimeOverrides = {},
  paths: ImprovePaths = defaultImprovePaths(),
): ImproveRuntime {
  const base = createOrchestratorRuntime(overrides);
  const runtime = {
    now: overrides.now ?? base.now,
    queryFn: overrides.queryFn ?? query,
    assayerFn: overrides.assayerFn ?? base.assayerFn,
    synthesisFn: overrides.synthesisFn ?? base.synthesisFn,
  } as ImproveRuntime;
  runtime.foundryFn = overrides.foundryFn ?? makeDefaultFoundryImproveFn(paths);
  runtime.retrospectorFn = overrides.retrospectorFn ?? defaultImproveRetrospectorFn;
  return runtime;
}

function targetSlug(targets: ImprovementTarget[]): string {
  return uniqueTargets(targets).map(sanitizePathPart).join("-");
}

async function persistAttempt(paths: ImprovePaths, moduleId: string, runId: string, attempt: ImprovementAttemptRecord): Promise<void> {
  const dir = path.join(paths.outputRoot, "improve", moduleId, runId);
  await mkdir(dir, { recursive: true });
  const prefix = path.join(dir, `attempt_${attempt.attempt_index}`);
  await writeFile(`${prefix}.v`, attempt.module.verilog_source, "utf8");
  await writeJsonFile(`${prefix}.module.json`, attempt.module);
  if (attempt.assayer_result) await writeJsonFile(`${prefix}.verif.json`, attempt.assayer_result);
  if (attempt.vivado_report) await writeJsonFile(`${prefix}.vivado.json`, attempt.vivado_report);
  if (attempt.metrics) await writeJsonFile(`${prefix}.metrics.json`, attempt.metrics);
  if (attempt.verdict) await writeJsonFile(`${prefix}.verdict.json`, attempt.verdict);
  if (attempt.messages) await writeJsonFile(`${prefix}.messages.json`, attempt.messages);
}

export async function commitReplacement(input: {
  paths: ImprovePaths;
  moduleId: string;
  module: VerilogModule;
  metrics: ImprovementMetrics;
  vivadoReport: SynthesisReport;
  verifResult: VerifResult;
  runtime: ImproveRuntime;
}): Promise<{ committedPath: string; archivedOriginalPath: string }> {
  const stamp = improvementStamp(input.runtime.now());
  const archiveDir = path.join(input.paths.rtlDir, "archive");
  const reportsArchiveDir = path.join(input.paths.reportsDir, "archive");
  await mkdir(archiveDir, { recursive: true });
  await mkdir(reportsArchiveDir, { recursive: true });

  const currentVerilog = path.join(input.paths.rtlDir, `${input.moduleId}.v`);
  const currentMeta = path.join(input.paths.rtlDir, `${input.moduleId}.meta.json`);
  const currentVivado = path.join(input.paths.reportsDir, `${input.moduleId}.vivado.json`);
  const currentResults = path.join(input.paths.reportsDir, `${input.moduleId}.results.json`);
  const currentMetrics = path.join(input.paths.reportsDir, `${input.moduleId}.metrics.json`);

  // Archive the originals (RTL + meta + canonical reports). The canonical
  // reports are the bit-exact `.vivado.json` / `.results.json` written by
  // the original pipeline run; without archiving them the archived RTL
  // becomes orphaned from its measurements.
  const archivedOriginalPath = path.join(archiveDir, `${input.moduleId}__${stamp}.v`);
  await copyFile(currentVerilog, archivedOriginalPath);
  if (await pathExists(currentMeta)) {
    await copyFile(currentMeta, path.join(archiveDir, `${input.moduleId}__${stamp}.meta.json`));
  }
  if (await pathExists(currentVivado)) {
    await copyFile(currentVivado, path.join(reportsArchiveDir, `${input.moduleId}__${stamp}.vivado.json`));
  }
  if (await pathExists(currentResults)) {
    await copyFile(currentResults, path.join(reportsArchiveDir, `${input.moduleId}__${stamp}.results.json`));
  }
  if (await pathExists(currentMetrics)) {
    await copyFile(currentMetrics, path.join(reportsArchiveDir, `${input.moduleId}__${stamp}.metrics.json`));
  }

  // Replace canonical artifacts with the improved attempt's RTL, meta,
  // Vivado report, Verilator result, and rolled-up metrics. After this
  // step, downstream tools that read the canonical paths see the improved
  // run, not the original.
  await writeFile(currentVerilog, input.module.verilog_source, "utf8");
  await writeJsonFile(currentMeta, input.module);
  await writeJsonFile(currentVivado, input.vivadoReport);
  await writeJsonFile(currentResults, input.verifResult);
  await writeJsonFile(currentMetrics, input.metrics);
  return { committedPath: currentVerilog, archivedOriginalPath };
}

async function ensureDocLifecycle(paths: ImprovePaths): Promise<{ version: 1; docs: Record<string, Record<string, unknown>> }> {
  const lifecyclePath = path.join(paths.knowledgeRoot, "doc_lifecycle.json");
  if (!(await pathExists(lifecyclePath))) {
    return { version: 1, docs: {} };
  }
  const raw = JSON.parse(await readFile(lifecyclePath, "utf8")) as unknown;
  if (typeof raw === "object" && raw !== null && "docs" in raw) {
    return raw as { version: 1; docs: Record<string, Record<string, unknown>> };
  }
  return { version: 1, docs: {} };
}

async function commitImprovedReference(input: {
  paths: ImprovePaths;
  moduleId: string;
  module: VerilogModule;
  layer: LayerIR;
  targets: ImprovementTarget[];
  metrics: ImprovementMetrics;
  verdict: ImprovementVerdict;
  runtime: ImproveRuntime;
}): Promise<string> {
  const slug = `${sanitizePathPart(input.moduleId)}__${targetSlug(input.targets)}`;
  const referenceRel = `knowledge/references/improved/${slug}.v`;
  const patternRel = `knowledge/patterns/improved/${slug}.md`;
  const referenceAbs = path.join(input.paths.repoRoot, referenceRel);
  const patternAbs = path.join(input.paths.repoRoot, patternRel);
  await mkdir(path.dirname(referenceAbs), { recursive: true });
  await mkdir(path.dirname(patternAbs), { recursive: true });
  await writeFile(referenceAbs, input.module.verilog_source, "utf8");
  await writeFile(
    patternAbs,
    [
      "---",
      `tier: improved`,
      `op_type: ${input.layer.op_type}`,
      `contract_id: ${input.layer.contract_id ?? "flat-bus"}`,
      `module_id: ${input.moduleId}`,
      `targets: [${input.targets.join(", ")}]`,
      `created_at: ${input.runtime.now().toISOString()}`,
      "---",
      "",
      `# Improved ${input.moduleId}`,
      "",
      "Deterministically verified improvement variant.",
      "",
      "Targets:",
      ...input.verdict.targets.map((target) => `- ${target.target}: ${target.reason}`),
      "",
      "Metrics:",
      "```json",
      JSON.stringify(input.metrics, null, 2),
      "```",
      "",
    ].join("\n"),
    "utf8",
  );

  const lifecycle = await ensureDocLifecycle(input.paths);
  const id = `improved_${slug}`;
  lifecycle.docs[id] = {
    id,
    op_type: input.layer.op_type,
    contract_id: input.layer.contract_id ?? "flat-bus",
    contract_key: input.module.spec_hash,
    spec_hash: input.module.spec_hash,
    status: "active",
    pattern_path: patternRel,
    reference_path: referenceRel,
    created_by_module: input.moduleId,
    created_by_agent: "Foundry",
    created_at: input.runtime.now().toISOString(),
    creation_reason: "quality_improvement",
    improvement_targets: input.targets,
    improvement_metrics: input.metrics,
    used_by_modules: [],
    successful_modules: [input.moduleId],
    failed_modules: [],
  };
  await writeJsonFile(path.join(input.paths.knowledgeRoot, "doc_lifecycle.json"), lifecycle);
  return referenceAbs;
}

async function writeImproveReport(paths: ImprovePaths, moduleId: string, targets: ImprovementTarget[], result: Omit<ImproveResult, "report_path">): Promise<string> {
  const reportPath = path.join(paths.reportsDir, `improve_${sanitizePathPart(moduleId)}__${targetSlug(targets)}.json`);
  await writeJsonFile(reportPath, { ...result, report_path: reportPath });
  return reportPath;
}

function normalizeImprovedModule(module: VerilogModule, original: VerilogModule, attemptIndex: number): VerilogModule {
  return {
    ...module,
    module_id: original.module_id,
    spec_hash: original.spec_hash,
    generated_by: "Foundry",
    attempt: attemptIndex,
  };
}

export async function runImprove(
  moduleId: string,
  options: RunImproveOptions,
): Promise<ImproveResult> {
  const targets = uniqueTargets(options.targets);
  if (targets.length === 0) {
    throw new Error("runImprove requires at least one target.");
  }
  // `reduce-latency` legitimately changes `pipeline_latency_cycles`. The
  // deterministic Verilator gate compares `timing_actual_cycles` against
  // `timing_expected_cycles` from the LayerIR (and from the goldens file
  // header), so any latency change must regenerate goldens + LayerIR
  // together. The improve flow does not own that regeneration; allowing
  // this target without a regen pass would silently fail Verilator on
  // every attempt and burn the entire 3-attempt budget. Until a
  // golden-regen step is wired in, refuse the target with a clear message.
  if (
    targets.includes("reduce-latency") &&
    process.env.NN2RTL_IMPROVE_ALLOW_LATENCY_CHANGE !== "1"
  ) {
    throw new Error(
      "reduce-latency cannot be improved without regenerating goldens and the LayerIR's pipeline_latency_cycles. " +
        "The improve flow does not currently own that regeneration step, so the deterministic Verilator gate would " +
        "fail every attempt. Re-run without --targets=reduce-latency, or set NN2RTL_IMPROVE_ALLOW_LATENCY_CHANGE=1 " +
        "if you have a separate process that updates goldens and the LayerIR latency contract before invoking improve.",
    );
  }
  const paths = resolveImprovePaths(options.paths);
  const runtime = createImproveRuntime(options.runtime, paths);
  const checkerConfig = {
    ...DEFAULT_IMPROVEMENT_CHECKER_CONFIG,
    ...options.checkerConfig,
  };
  const runId = improvementStamp(runtime.now());

  const originalModule = await loadOriginalModule(paths, moduleId);
  const layer = await loadLayer(paths, moduleId);
  const baseline = await loadBaselineMetrics(paths, moduleId);
  if (!baseline.vivadoReport.success || !baseline.vivadoReport.timing_met) {
    throw new Error(
      `Cannot improve '${moduleId}': baseline Vivado report is not passing. ` +
        "Run improve only on already-passing RTL.",
    );
  }
  if (baseline.verifResult && baseline.verifResult.status !== "pass") {
    throw new Error(
      `Cannot improve '${moduleId}': baseline Verilator result is '${baseline.verifResult.status}'. ` +
        "Run improve only on already-passing RTL.",
    );
  }
  // Foundry persists each attempt via `write_verilog`, which overwrites the
  // canonical `<rtl>/<moduleId>.v` and `<rtl>/<moduleId>.meta.json`. The
  // deterministic verilator run then writes `<reports>/<moduleId>.results.json`
  // and Vivado overwrites `<reports>/<moduleId>.vivado.json`. If the run
  // ends in `no-change` or `kept-as-variant`, all of these canonical
  // artifacts must still describe the ORIGINAL (passing) RTL — otherwise
  // downstream consumers see RTL or measurements from a failing attempt that
  // don't match the layer's verified contract. Snapshot now and restore on
  // the non-replace paths.
  const canonicalSnapshotFiles = [
    path.join(paths.rtlDir, `${moduleId}.v`),
    path.join(paths.rtlDir, `${moduleId}.meta.json`),
    path.join(paths.reportsDir, `${moduleId}.vivado.json`),
    path.join(paths.reportsDir, `${moduleId}.results.json`),
    path.join(paths.reportsDir, `${moduleId}.metrics.json`),
  ];
  const canonicalSnapshots = new Map<string, string>();
  for (const filePath of canonicalSnapshotFiles) {
    if (await pathExists(filePath)) {
      canonicalSnapshots.set(filePath, await readFile(filePath, "utf8"));
    }
  }

  const attempts: ImprovementAttemptRecord[] = [];
  let resumeSessionId: string | undefined;
  let retrospectorAdvice: RetrospectorAdvice | undefined;
  let successfulAttempt: ImprovementAttemptRecord | null = null;
  // Flipped to true only on the actual `replaced` commit branch — every other
  // exit (no-change, kept-as-variant, or thrown error mid-loop) must restore
  // the canonical snapshot. Using try/finally instead of a post-loop branch
  // ensures restore runs even when Foundry's structured-output parse throws
  // partway through the loop.
  let canonicalCommitted = false;
  let finalAction: ImproveResult["final_action"] = "no-change";
  let committedModulePath: string | undefined;
  let archivedOriginalPath: string | undefined;
  let improvedReferencePath: string | undefined;

  try {
  for (let attemptIndex = 1; attemptIndex <= 3; attemptIndex += 1) {
    if (attemptIndex === 3) {
      retrospectorAdvice = await runtime.retrospectorFn(
        {
          module_id: moduleId,
          targets,
          original_module: originalModule,
          baseline_metrics: baseline.metrics,
          attempts,
        },
        runtime,
      );
    }

    const foundry = await runtime.foundryFn(
      {
        attempt_index: attemptIndex,
        module_id: moduleId,
        targets,
        original_module: originalModule,
        baseline_metrics: baseline.metrics,
        baseline_vivado_report: baseline.vivadoReport,
        layer_ir: layer,
        previous_attempts: attempts,
        resume_session_id: resumeSessionId,
        retrospector_advice: attemptIndex === 3 ? retrospectorAdvice : undefined,
      },
      runtime,
    );
    const module = normalizeImprovedModule(foundry.module, originalModule, attemptIndex);
    const attempt: ImprovementAttemptRecord = {
      attempt_index: attemptIndex,
      module,
      session_id: foundry.session_id ?? extractSessionId(foundry.messages, foundry.result),
      verilog_path: path.join(paths.outputRoot, "improve", moduleId, runId, `attempt_${attemptIndex}.v`),
      failed_gate: null,
      messages: foundry.messages,
    };
    resumeSessionId = attempt.session_id ?? resumeSessionId;

    const verif = await runtime.assayerFn(module, layer);
    attempt.assayer_result = verifResultSchema.parse(verif);
    if (attempt.assayer_result.status !== "pass") {
      attempt.failed_gate = "verilator";
      attempts.push(attempt);
      await persistAttempt(paths, moduleId, runId, attempt);
      if (attemptIndex === 2) continue;
      if (attemptIndex === 3) break;
      continue;
    }

    const vivado = await runtime.synthesisFn(module, layer);
    attempt.vivado_report = synthesisReportSchema.parse(vivado);
    if (!attempt.vivado_report.success || !attempt.vivado_report.timing_met) {
      attempt.failed_gate = "vivado";
      attempts.push(attempt);
      await persistAttempt(paths, moduleId, runId, attempt);
      if (attemptIndex === 2) continue;
      if (attemptIndex === 3) break;
      continue;
    }

    attempt.metrics = metricsFromReports(attempt.vivado_report, attempt.assayer_result);
    attempt.verdict = evaluateImprovementTargets(baseline.metrics, attempt.metrics, targets, checkerConfig);
    if (!attempt.verdict.overall) {
      attempt.failed_gate = "improvement_checker";
      attempts.push(attempt);
      await persistAttempt(paths, moduleId, runId, attempt);
      if (attemptIndex === 2) continue;
      if (attemptIndex === 3) break;
      continue;
    }

    attempts.push(attempt);
    await persistAttempt(paths, moduleId, runId, attempt);
    successfulAttempt = attempt;
    break;
  }

  if (successfulAttempt?.metrics && successfulAttempt.verdict) {
    if (options.keepReference) {
      finalAction = "kept-as-variant";
      improvedReferencePath = await commitImprovedReference({
        paths,
        moduleId,
        module: successfulAttempt.module,
        layer,
        targets,
        metrics: successfulAttempt.metrics,
        verdict: successfulAttempt.verdict,
        runtime,
      });
    } else {
      finalAction = "replaced";
      if (!successfulAttempt.vivado_report || !successfulAttempt.assayer_result) {
        throw new Error(
          "internal: successful improve attempt is missing vivado_report or assayer_result; cannot commit canonical reports.",
        );
      }
      const commit = await commitReplacement({
        paths,
        moduleId,
        module: successfulAttempt.module,
        metrics: successfulAttempt.metrics,
        vivadoReport: successfulAttempt.vivado_report,
        verifResult: successfulAttempt.assayer_result,
        runtime,
      });
      committedModulePath = commit.committedPath;
      archivedOriginalPath = commit.archivedOriginalPath;
      canonicalCommitted = true;
    }
  }
  } finally {
    // Every exit branch except `replaced` (and every thrown error) leaves the
    // canonical RTL on disk pointing at the original — so the canonical
    // reports must match. Restore the pre-run snapshots that were clobbered
    // by per-attempt write_verilog / verilator / vivado side effects.
    if (!canonicalCommitted) {
      for (const filePath of canonicalSnapshotFiles) {
        const snapshot = canonicalSnapshots.get(filePath);
        if (snapshot !== undefined) {
          await writeFile(filePath, snapshot, "utf8");
        }
      }
    }
  }

  const reportWithoutPath: Omit<ImproveResult, "report_path"> = {
    module_id: moduleId,
    targets,
    final_action: finalAction,
    success: successfulAttempt !== null,
    baseline_metrics: baseline.metrics,
    attempts,
    final_verdict: successfulAttempt?.verdict ?? attempts.at(-1)?.verdict,
    committed_module_path: committedModulePath,
    archived_original_path: archivedOriginalPath,
    improved_reference_path: improvedReferencePath,
    retrospector_advice: retrospectorAdvice,
  };
  const reportPath = await writeImproveReport(paths, moduleId, targets, reportWithoutPath);
  return {
    ...reportWithoutPath,
    report_path: reportPath,
  };
}

export async function runImproveCli(argv: string[] = process.argv.slice(2)): Promise<void> {
  const cli = parseImproveCliArgs(argv);
  const result = await runImprove(cli.moduleId, {
    targets: cli.targets,
    keepReference: cli.keepReference,
  });
  console.log(
    `Improve ${result.module_id} [${result.targets.join(", ")}]: ${result.final_action}; report: ${result.report_path}`,
  );
}
