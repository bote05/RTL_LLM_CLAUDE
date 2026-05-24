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
  appendForeignMcpToolWarnings,
  appendRunLog,
  appendToolUseAudits,
  createOrchestratorRuntime,
  extractToolUseAudits,
  findLayer,
  getActiveNetworkId,
  loadRetrospectorKnowledgeDoc,
  pathExists,
  readJsonFile,
  requireStructuredOutput,
  setActiveNetwork,
  synthesisPreflightReport,
  synthesisPreflightViolations,
  writeJsonFile,
  type ContractPlan,
  type OrchestratorRuntime,
  type RtlKnowledgeDoc,
} from "./orchestrate.js";
import { resolveLayerContractId } from "./contracts.js";
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
import { applicabilityForSignature, signatureBundle } from "./signatures.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const sdkRoot = path.resolve(
  __dirname,
  path.basename(__dirname) === "dist" ? ".." : ".",
);
const defaultRepoRoot = path.resolve(sdkRoot, "..");
const pluginPath = path.resolve(defaultRepoRoot, "nn2rtl-plugin");
const improveFoundryPromptPath = path.join(pluginPath, "agents", "improve_foundry.md");

export const IMPROVEMENT_TARGETS = [
  "use-dsp",
  "use-bram",
  "reduce-lut",
  "reduce-ff",
  "improve-fmax",
  "reduce-latency",
  "increase-throughput",
] as const;

export type ImprovementTarget = typeof IMPROVEMENT_TARGETS[number];
export type SynthesisReport = z.infer<typeof synthesisReportSchema>;

export type ImprovementMetrics = {
  lut: number;
  ff: number;
  dsp: number;
  bram: number;
  fmax_mhz?: number;
  latency_cycles?: number;
  ii?: number;
};

export type ImprovementCheckerConfig = {
  useDspThresholdMin: number;
  useBramMinGain: number;
  useBramMinLutDelta: number;
  useBramMinFfDelta: number;
  reduceLutMinDelta: number;
  reduceFfMinDelta: number;
  improveFmaxMinDelta: number;
  improveFmaxFloorMhz: number;
  improveFmaxMinAdditiveMhz: number;
  increaseThroughputMinFpsDelta: number;
  increaseThroughputMinDspMultiplier: number;
};

export const DEFAULT_IMPROVEMENT_CHECKER_CONFIG: ImprovementCheckerConfig = {
  useDspThresholdMin: parsePositiveIntEnv(process.env, "NN2RTL_IMPROVE_USE_DSP_MIN", 8),
  useBramMinGain: parsePositiveIntEnv(process.env, "NN2RTL_IMPROVE_USE_BRAM_MIN_GAIN", 8),
  useBramMinLutDelta: Number(process.env.NN2RTL_IMPROVE_USE_BRAM_MIN_LUT_DELTA ?? "") || 0.05,
  useBramMinFfDelta: Number(process.env.NN2RTL_IMPROVE_USE_BRAM_MIN_FF_DELTA ?? "") || 0.05,
  reduceLutMinDelta: Number(process.env.NN2RTL_IMPROVE_REDUCE_LUT_MIN_DELTA ?? "") || 0.05,
  reduceFfMinDelta: Number(process.env.NN2RTL_IMPROVE_REDUCE_FF_MIN_DELTA ?? "") || 0.10,
  improveFmaxMinDelta: Number(process.env.NN2RTL_IMPROVE_FMAX_MIN_DELTA ?? "") || 0.05,
  improveFmaxFloorMhz: Number(process.env.NN2RTL_IMPROVE_FMAX_FLOOR_MHZ ?? "") || 300,
  // Below-floor baselines must close MEANINGFUL ground per attempt, not just
  // clear the 5% relative bump. A 167 MHz module that only has to reach 175
  // doesn't move the comparison-vs-deterministic-tool story. Require at
  // least baseline + this many MHz, capped at the absolute floor.
  improveFmaxMinAdditiveMhz:
    Number(process.env.NN2RTL_IMPROVE_FMAX_MIN_ADDITIVE_MHZ ?? "") || 50,
  // increase-throughput acceptance is FPS-based, not II-only. A 0.5% II
  // reduction that costs 10% Fmax is a net wall-clock regression. Require
  // real fps headroom to count.
  increaseThroughputMinFpsDelta:
    Number(process.env.NN2RTL_IMPROVE_THROUGHPUT_MIN_FPS_DELTA ?? "") || 0.10,
  // Parallelization on a 1-DSP baseline must add MACs. Without the multiplier
  // gate, Foundry returns single-DSP "tweaks" that satisfy II-on-paper but
  // do not represent the architectural change the target is asking for.
  increaseThroughputMinDspMultiplier:
    Number(process.env.NN2RTL_IMPROVE_THROUGHPUT_MIN_DSP_MULTIPLIER ?? "") || 2,
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
  preloaded_rtl_patterns?: RtlKnowledgeDoc;
  sequence_context?: ImproveSequenceContext[];
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
  sequence_context?: ImproveSequenceContext[];
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

export type ImproveSequenceStepSummary = {
  target: ImprovementTarget;
  success: boolean;
  final_action: ImproveResult["final_action"];
  report_path: string;
  error?: string;
};

export type ImproveSequenceContext = {
  target: ImprovementTarget;
  report_path: string;
  final_action: ImproveResult["final_action"];
  metrics?: ImprovementMetrics;
  verdict?: ImprovementVerdict;
};

export type ImproveSequenceResult = ImproveResult & {
  sequence_steps: ImproveSequenceStepSummary[];
  requested_targets: ImprovementTarget[];
  completed_targets: ImprovementTarget[];
  failed_targets: ImprovementTarget[];
  unattempted_targets: ImprovementTarget[];
  remaining_targets: ImprovementTarget[];
  partial_success: boolean;
  overall_success: boolean;
};

export type RunImproveOptions = {
  targets: ImprovementTarget[];
  keepReference?: boolean;
  runtime?: ImproveRuntimeOverrides;
  paths?: Partial<ImprovePaths>;
  checkerConfig?: Partial<ImprovementCheckerConfig>;
  sequenceContext?: ImproveSequenceContext[];
};

export type ImproveCliArgs = {
  moduleId: string;
  networkId: string | undefined;
  targets: ImprovementTarget[];
  keepReference: boolean;
};

export const IMPROVE_SWEEP_PRESETS = [
  "ppa",
  "ppa-no-dsp",
  "use-dsp",
  "reduce-lut",
  "reduce-ff",
  "improve-fmax",
] as const;

export type ImproveSweepPreset = typeof IMPROVE_SWEEP_PRESETS[number];

export type ImproveSweepRecommendation = {
  module_id: string;
  op_type: string;
  targets: ImprovementTarget[];
  priority: number;
  reasons: string[];
  metrics: ImprovementMetrics;
  num_weights: number;
};

export type ImproveSweepPlan = {
  generated_at: string;
  preset: ImproveSweepPreset;
  recommendations: ImproveSweepRecommendation[];
  skipped: Array<{ module_id: string; reason: string }>;
};

export type ImproveSweepResult = {
  plan: ImproveSweepPlan;
  ran: boolean;
  keep_reference: boolean;
  results: Array<{
    module_id: string;
    targets: ImprovementTarget[];
    success: boolean;
    final_action: ImproveResult["final_action"];
    report_path: string;
    error?: string;
  }>;
  report_path: string;
};

export type ImproveSweepCliArgs = {
  networkId?: string;
  preset: ImproveSweepPreset;
  run: boolean;
  keepReference: boolean;
  maxModules?: number;
};

function toOutputFormat(schema: z.ZodType): OutputFormat {
  return {
    type: "json_schema",
    schema: z.toJSONSchema(schema) as Record<string, unknown>,
  };
}

// Improve Foundry SHOULD persist `verilog_source` via
// `mcp__nn2rtl-tools__write_verilog`, but the improve pipeline also accepts an
// The improve schema requires `verilog_source` to be inlined. When Foundry
// was allowed to omit it (relying on `write_verilog`), the agent took the
// permission as a signal to skip inlining AND skipped the tool call,
// producing metadata-only turns that the orchestrator could not hydrate.
// Mandatory inline makes the orchestrator's hydrate path always succeed:
// the source is written to disk from the structured output. `write_verilog`
// becomes a redundant tool call (still allowed, but no longer load-bearing).
const verilogModuleAgentOutputSchema = verilogModuleSchema;
type VerilogModuleAgentOutput = z.infer<typeof verilogModuleAgentOutputSchema>;
const verilogModuleAgentOutputFormat = toOutputFormat(verilogModuleAgentOutputSchema);
const retrospectorAdviceOutputFormat = toOutputFormat(retrospectorAdviceSchema);

function isSdkResultMessage(message: SDKMessage): message is SDKResultMessage {
  return message.type === "result" && "total_cost_usd" in message && "modelUsage" in message;
}

export function defaultImprovePaths(repoRoot = defaultRepoRoot): ImprovePaths {
  const outputRoot = repoRoot === defaultRepoRoot
    ? (process.env.NN2RTL_OUTPUT_DIR ? path.resolve(repoRoot, process.env.NN2RTL_OUTPUT_DIR) : path.join(repoRoot, "output"))
    : path.join(repoRoot, "output");
  // NN2RTL_KNOWLEDGE_DIR lets parallel improve runners give each worker an
  // isolated knowledge sandbox (own doc_lifecycle.json + own improved/ tier).
  // Without this, multiple workers race on knowledge/doc_lifecycle.json's
  // read-modify-write and lose lifecycle entries.
  const knowledgeRoot = repoRoot === defaultRepoRoot
    ? (process.env.NN2RTL_KNOWLEDGE_DIR
        ? path.resolve(repoRoot, process.env.NN2RTL_KNOWLEDGE_DIR)
        : path.join(repoRoot, "knowledge"))
    : path.join(repoRoot, "knowledge");
  return {
    repoRoot,
    outputRoot,
    reportsDir: path.join(outputRoot, "reports"),
    rtlDir: path.join(outputRoot, "rtl"),
    knowledgeRoot,
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

function stripMarkdownFrontmatter(markdown: string): string {
  if (!markdown.startsWith("---\n")) return markdown;
  const end = markdown.indexOf("\n---", 4);
  if (end === -1) return markdown;
  return markdown.slice(end + "\n---".length).replace(/^\r?\n/, "");
}

async function loadImproveFoundrySystemPrompt(): Promise<string> {
  return stripMarkdownFrontmatter(await readFile(improveFoundryPromptPath, "utf8"));
}

function uniqueTargets(targets: ImprovementTarget[]): ImprovementTarget[] {
  const seen = new Set<ImprovementTarget>();
  const out: ImprovementTarget[] = [];
  for (const target of targets) {
    if (seen.has(target)) continue;
    seen.add(target);
    out.push(target);
  }
  return out;
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
  let networkId: string | undefined;
  let targets: ImprovementTarget[] | null = null;
  let keepReference = false;

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--keep-reference") {
      keepReference = true;
    } else if (arg === "--network") {
      const next = argv[++i];
      if (next === undefined || next.startsWith("--")) {
        throw new Error("--network requires a network id.");
      }
      networkId = next;
    } else if (arg.startsWith("--network=")) {
      networkId = arg.slice("--network=".length);
      if (!networkId) {
        throw new Error("--network= requires a non-empty network id.");
      }
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

  return { moduleId: positional[0], networkId, targets, keepReference };
}

export function parseImproveSweepCliArgs(argv: string[]): ImproveSweepCliArgs {
  let networkId: string | undefined;
  let preset: ImproveSweepPreset = "ppa";
  let run = false;
  let keepReference = true;
  let maxModules: number | undefined;

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--run") {
      run = true;
    } else if (arg === "--network") {
      const next = argv[++i];
      if (next === undefined || next.startsWith("--")) {
        throw new Error("--network requires a network id.");
      }
      networkId = next;
    } else if (arg.startsWith("--network=")) {
      networkId = arg.slice("--network=".length);
      if (!networkId) {
        throw new Error("--network= requires a non-empty network id.");
      }
    } else if (arg === "--dry-run" || arg === "--plan") {
      run = false;
    } else if (arg === "--keep-reference") {
      keepReference = true;
    } else if (arg === "--replace") {
      keepReference = false;
    } else if (arg === "--preset") {
      const next = argv[++i];
      if (next === undefined || next.startsWith("--")) {
        throw new Error("--preset requires a sweep preset.");
      }
      if (!IMPROVE_SWEEP_PRESETS.includes(next as ImproveSweepPreset)) {
        throw new Error(`Unknown sweep preset '${next}'. Allowed presets: ${IMPROVE_SWEEP_PRESETS.join(", ")}.`);
      }
      preset = next as ImproveSweepPreset;
    } else if (arg.startsWith("--preset=")) {
      const raw = arg.slice("--preset=".length);
      if (!IMPROVE_SWEEP_PRESETS.includes(raw as ImproveSweepPreset)) {
        throw new Error(`Unknown sweep preset '${raw}'. Allowed presets: ${IMPROVE_SWEEP_PRESETS.join(", ")}.`);
      }
      preset = raw as ImproveSweepPreset;
    } else if (arg === "--max-modules") {
      const next = argv[++i];
      if (next === undefined || next.startsWith("--")) {
        throw new Error("--max-modules requires a positive integer.");
      }
      const parsed = Number(next);
      if (!Number.isInteger(parsed) || parsed <= 0) {
        throw new Error(`--max-modules must be a positive integer, got '${next}'.`);
      }
      maxModules = parsed;
    } else if (arg.startsWith("--max-modules=")) {
      const raw = arg.slice("--max-modules=".length);
      const parsed = Number(raw);
      if (!Number.isInteger(parsed) || parsed <= 0) {
        throw new Error(`--max-modules must be a positive integer, got '${raw}'.`);
      }
      maxModules = parsed;
    } else {
      throw new Error(`Unknown improve sweep flag '${arg}'.`);
    }
  }

  return { networkId, preset, run, keepReference, maxModules };
}

function metricValue(metrics: ImprovementMetrics, target: ImprovementTarget): number | undefined {
  switch (target) {
    case "use-dsp":
      return metrics.dsp;
    case "use-bram":
      return metrics.bram;
    case "reduce-lut":
      return metrics.lut;
    case "reduce-ff":
      return metrics.ff;
    case "improve-fmax":
      return metrics.fmax_mhz;
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
        const requiredBram = baseline.bram + config.useBramMinGain;
        const requiredLut = baseline.lut * (1 - config.useBramMinLutDelta);
        const requiredFf = baseline.ff * (1 - config.useBramMinFfDelta);
        const bramGainSatisfied = next.bram >= requiredBram;
        const lutDropSatisfied = next.lut < requiredLut;
        const ffDropSatisfied = next.ff < requiredFf;
        const satisfied = bramGainSatisfied && (lutDropSatisfied || ffDropSatisfied);
        return {
          target,
          satisfied,
          baseline_value: baseline.bram,
          new_value: next.bram,
          required:
            `new.bram >= baseline.bram + ${config.useBramMinGain} = ${requiredBram} ` +
            `AND (new.lut < ${requiredLut} OR new.ff < ${requiredFf})`,
          reason: satisfied
            ? "BRAM usage target satisfied with meaningful LUT/FF reduction."
            : `BRAM/PPA target missed: new.bram=${next.bram} (need >=${requiredBram}), ` +
              `new.lut=${next.lut} (need <${requiredLut}), new.ff=${next.ff} (need <${requiredFf}).`,
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
      case "reduce-ff": {
        const required = baseline.ff * (1 - config.reduceFfMinDelta);
        const satisfied = next.ff < required;
        return {
          target,
          satisfied,
          baseline_value: baseline.ff,
          new_value: next.ff,
          required: `new.ff < baseline.ff * (1 - ${config.reduceFfMinDelta}) = ${required}`,
          reason: satisfied ? "FF reduction target satisfied." : `new.ff=${next.ff} is not below ${required}.`,
        };
      }
      case "improve-fmax": {
        // Two floors, take the stricter:
        //   relative = baseline * (1 + delta)         (always-on percentage bump)
        //   additive = min(floor, baseline + addMhz)  (meaningful absolute jump,
        //                                              capped at the absolute
        //                                              quality floor)
        // required = max(relative, additive)
        // For an already-above-floor baseline (e.g. 350 MHz), `additive`
        // collapses to `floor` (= 300) which is <= `relative`, so the rule
        // reduces to the pure relative bump. For a below-floor baseline
        // (e.g. 167 MHz, addMhz=50, floor=300), `additive` = 217 and
        // `relative` = 175.35; max = 217, so the module has to close real
        // ground per attempt instead of taking a 5% sliver.
        const relative = baseline.fmax_mhz! * (1 + config.improveFmaxMinDelta);
        const additive = Math.min(
          config.improveFmaxFloorMhz,
          baseline.fmax_mhz! + config.improveFmaxMinAdditiveMhz,
        );
        const required = Math.max(relative, additive);
        const satisfied = next.fmax_mhz! > required;
        return {
          target,
          satisfied,
          baseline_value: baseline.fmax_mhz,
          new_value: next.fmax_mhz,
          required:
            `new.fmax_mhz > max(baseline.fmax_mhz * (1 + ${config.improveFmaxMinDelta}), ` +
            `min(${config.improveFmaxFloorMhz}, baseline.fmax_mhz + ${config.improveFmaxMinAdditiveMhz})) = ${required}`,
          reason: satisfied ? "Fmax improvement target satisfied." : `new.fmax_mhz=${next.fmax_mhz} is not above ${required}.`,
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
        // FPS = Fmax * 1e6 / II. II-only gates are gameable: Foundry can
        // trade Fmax for cycles and "win" on paper while regressing the
        // wall-clock throughput. Score the actual fps.
        const baselineFps = (baseline.fmax_mhz! * 1e6) / baseline.ii!;
        const newFps = (next.fmax_mhz! * 1e6) / next.ii!;
        const requiredFps = baselineFps * (1 + config.increaseThroughputMinFpsDelta);
        const fpsSatisfied = newFps > requiredFps;
        // Parallelization mandate. If the baseline already uses a non-trivial
        // number of DSPs we don't require doubling (replication isn't the
        // only path to throughput); but on a baseline-DSP=1 module, returning
        // a 1-DSP variant proves the agent didn't replicate MAC lanes.
        const dspGateRequired = baseline.dsp >= 1
          ? Math.max(1, Math.ceil(baseline.dsp * config.increaseThroughputMinDspMultiplier))
          : 1;
        const dspSatisfied = next.dsp >= dspGateRequired;
        const satisfied = fpsSatisfied && dspSatisfied;
        const reasons: string[] = [];
        if (!fpsSatisfied) {
          reasons.push(
            `new.fps=${newFps.toFixed(4)} is not above baseline.fps=${baselineFps.toFixed(4)} * (1 + ${config.increaseThroughputMinFpsDelta}) = ${requiredFps.toFixed(4)}.`,
          );
        }
        if (!dspSatisfied) {
          reasons.push(
            `new.dsp=${next.dsp} is below the required ${dspGateRequired} (= max(1, ceil(baseline.dsp=${baseline.dsp} * ${config.increaseThroughputMinDspMultiplier}))). MAC parallelization must add DSPs on a serialized-MAC baseline.`,
          );
        }
        return {
          target,
          satisfied,
          baseline_value: baselineFps,
          new_value: newFps,
          required:
            `new.fps > baseline.fps * (1 + ${config.increaseThroughputMinFpsDelta}) AND ` +
            `new.dsp >= ${dspGateRequired}`,
          reason: satisfied ? "Throughput target satisfied." : reasons.join(" "),
        };
      }
    }
  });

  return {
    overall: results.every((result) => result.satisfied),
    targets: results,
  };
}

// Decide whether an attempt's Verilator result is acceptable as the input to
// the synthesis + acceptance-gate stages, given the requested target list.
//
// For most targets the rule is simply `status === "pass"` — outputs match the
// goldens exactly AND first_valid_out lands on the LayerIR's expected cycle.
//
// `increase-throughput` is the deliberate exception. The whole point of that
// target is to lower per-frame cycles in steady state — but the static
// testbench enforces per-vector `actual_cycles == timing_expected_cycles`,
// which a successful parallelization breaks by design. We accept the attempt
// when it is bit-exact (mismatch_count == 0, max_error == 0) AND first-frame
// cycles still match the pipeline-fill contract (so time-to-first-output is
// preserved). Subsequent vectors are allowed to finish faster — that is the
// target's literal mechanism.
function isAssayerResultAcceptableForTargets(
  verif: VerifResult,
  targets: ImprovementTarget[],
): boolean {
  if (verif.status === "pass") return true;
  if (!targets.includes("increase-throughput")) return false;
  const bitExact =
    verif.mismatch_count === 0 &&
    verif.max_error === 0 &&
    verif.exact_match_count !== undefined &&
    verif.sample_count !== undefined &&
    verif.exact_match_count === verif.sample_count;
  if (!bitExact) return false;
  const perVector = verif.per_vector ?? [];
  if (perVector.length === 0) return false;
  const firstFrameCycles = perVector[0]?.actual_cycles;
  const expectedFill = verif.timing_expected_cycles;
  if (
    typeof firstFrameCycles !== "number" ||
    typeof expectedFill !== "number" ||
    firstFrameCycles !== expectedFill
  ) {
    return false;
  }
  return true;
}

function metricsFromReports(synthesis: SynthesisReport, verif?: VerifResult): ImprovementMetrics {
  return {
    lut: synthesis.lut_count,
    ff: synthesis.ff_count,
    dsp: synthesis.dsp_count,
    bram: synthesis.bram18_equiv || synthesis.bram18_count + synthesis.bram36_count * 2,
    fmax_mhz: synthesis.fmax_mhz,
    latency_cycles: verif?.timing_actual_cycles !== undefined && verif.timing_actual_cycles >= 0
      ? verif.timing_actual_cycles
      : undefined,
    ii: deriveInitiationIntervalCycles(verif),
  };
}

// Per-frame initiation interval. Prefer the explicit field if the testbench
// emits it; otherwise derive from steady-state pulse positions:
//   ii = (last_valid_out_cycle - first_valid_in_cycle) / num_frames
// where num_frames = per_vector.length. This is the same quantity used in
// output/reports/throughput_per_module.csv and is what increase-throughput
// must lower for the acceptance gate to fire on real numbers.
function deriveInitiationIntervalCycles(verif?: VerifResult): number | undefined {
  if (!verif) return undefined;
  const explicit = (verif as { initiation_interval_cycles?: unknown }).initiation_interval_cycles;
  if (typeof explicit === "number" && Number.isFinite(explicit) && explicit >= 0) return explicit;
  const first = verif.first_valid_in_cycle;
  const last = verif.last_valid_out_cycle;
  const numFrames = Array.isArray(verif.per_vector) ? verif.per_vector.length : 0;
  if (typeof first !== "number" || typeof last !== "number" || numFrames <= 0) return undefined;
  const span = last - first;
  if (!Number.isFinite(span) || span <= 0) return undefined;
  return span / numFrames;
}

export const improvementMetricsSchema = z
  .object({
    lut: z.number().nonnegative(),
    ff: z.number().nonnegative(),
    dsp: z.number().nonnegative(),
    bram: z.number().nonnegative(),
    fmax_mhz: z.number().nonnegative().optional(),
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
  const vivadoReport = await readJsonFile<SynthesisReport>(
    path.join(paths.reportsDir, `${moduleId}.vivado.json`),
    synthesisReportSchema,
  );
  const verifResult = await readOptionalJson<VerifResult>(
    path.join(paths.reportsDir, `${moduleId}.results.json`),
    verifResultSchema,
  ) ?? undefined;
  const derived = metricsFromReports(vivadoReport, verifResult);
  const metricsRaw = await readOptionalJson<Record<string, unknown>>(
    metricsPath,
    z.record(z.string(), z.unknown()),
  );
  const metrics: ImprovementMetrics = metricsRaw
    ? {
        lut: typeof metricsRaw.lut === "number" ? metricsRaw.lut : derived.lut,
        ff: typeof metricsRaw.ff === "number" ? metricsRaw.ff : derived.ff,
        dsp: typeof metricsRaw.dsp === "number" ? metricsRaw.dsp : derived.dsp,
        bram: typeof metricsRaw.bram === "number" ? metricsRaw.bram : derived.bram,
        fmax_mhz: typeof metricsRaw.fmax_mhz === "number" ? metricsRaw.fmax_mhz : derived.fmax_mhz,
        latency_cycles: typeof metricsRaw.latency_cycles === "number" ? metricsRaw.latency_cycles : derived.latency_cycles,
        ii: typeof metricsRaw.ii === "number" ? metricsRaw.ii : derived.ii,
      }
    : derived;
  return {
    vivadoReport,
    verifResult,
    metrics,
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

const DEFAULT_SWEEP_THRESHOLDS = {
  bigConvMinWeights: parsePositiveIntEnv(process.env, "NN2RTL_SWEEP_BIG_CONV_MIN_WEIGHTS", 4096),
  bigConvMinLut: parsePositiveIntEnv(process.env, "NN2RTL_SWEEP_BIG_CONV_MIN_LUT", 10_000),
  reduceLutMin: parsePositiveIntEnv(process.env, "NN2RTL_SWEEP_REDUCE_LUT_MIN", 100_000),
  reduceFfMin: parsePositiveIntEnv(process.env, "NN2RTL_SWEEP_REDUCE_FF_MIN", 100_000),
  reduceFfAddMin: parsePositiveIntEnv(process.env, "NN2RTL_SWEEP_REDUCE_FF_ADD_MIN", 20_000),
  fmaxBelowMhz: Number(process.env.NN2RTL_SWEEP_FMAX_BELOW_MHZ ?? "") || 220,
} as const;

function presetIncludesTarget(preset: ImproveSweepPreset, target: ImprovementTarget): boolean {
  if (preset === "ppa") return true;
  // ppa-no-dsp: everything ppa would include, minus the (structurally hard
  // and frequently unattainable on Foundry's default MAC ladder) use-dsp
  // target. Useful when you want to harvest LUT/FF/Fmax wins without
  // burning attempts on the absolute DSP>=8 threshold.
  if (preset === "ppa-no-dsp") return target !== "use-dsp";
  return preset === target;
}

function targetPriority(target: ImprovementTarget): number {
  switch (target) {
    case "use-dsp":
      return 1;
    case "reduce-lut":
      return 2;
    case "reduce-ff":
      return 3;
    case "improve-fmax":
      return 4;
    case "use-bram":
      return 5;
    case "reduce-latency":
      return 6;
    case "increase-throughput":
      return 7;
  }
}

function recommendationPriority(targets: ImprovementTarget[]): number {
  return Math.min(...targets.map(targetPriority));
}

function buildRecommendationForLayer(
  layer: LayerIR,
  metrics: ImprovementMetrics,
  preset: ImproveSweepPreset,
): ImproveSweepRecommendation | null {
  const targets: ImprovementTarget[] = [];
  const reasons: string[] = [];

  const isBigConv =
    layer.op_type === "conv2d" &&
    metrics.dsp < DEFAULT_IMPROVEMENT_CHECKER_CONFIG.useDspThresholdMin &&
    ((layer.num_weights ?? 0) >= DEFAULT_SWEEP_THRESHOLDS.bigConvMinWeights ||
      metrics.lut >= DEFAULT_SWEEP_THRESHOLDS.bigConvMinLut);
  if (presetIncludesTarget(preset, "use-dsp") && isBigConv) {
    targets.push("use-dsp");
    reasons.push(
      `conv2d uses ${metrics.dsp} DSPs with ${layer.num_weights ?? 0} weights / ${metrics.lut} LUTs; candidate for DSP-parallel MAC rewrite.`,
    );
  }

  if (presetIncludesTarget(preset, "reduce-lut") && metrics.lut >= DEFAULT_SWEEP_THRESHOLDS.reduceLutMin) {
    targets.push("reduce-lut");
    reasons.push(`LUT count ${metrics.lut} >= sweep threshold ${DEFAULT_SWEEP_THRESHOLDS.reduceLutMin}.`);
  }

  const ffThreshold = layer.op_type === "add"
    ? DEFAULT_SWEEP_THRESHOLDS.reduceFfAddMin
    : DEFAULT_SWEEP_THRESHOLDS.reduceFfMin;
  if (presetIncludesTarget(preset, "reduce-ff") && metrics.ff >= ffThreshold) {
    targets.push("reduce-ff");
    reasons.push(`FF count ${metrics.ff} >= sweep threshold ${ffThreshold}.`);
  }

  if (
    presetIncludesTarget(preset, "improve-fmax") &&
    metrics.fmax_mhz !== undefined &&
    metrics.fmax_mhz > 0 &&
    metrics.fmax_mhz < DEFAULT_SWEEP_THRESHOLDS.fmaxBelowMhz
  ) {
    targets.push("improve-fmax");
    reasons.push(`Fmax ${metrics.fmax_mhz.toFixed(2)} MHz < sweep threshold ${DEFAULT_SWEEP_THRESHOLDS.fmaxBelowMhz} MHz.`);
  }

  const unique = uniqueTargets(targets);
  if (unique.length === 0) return null;
  return {
    module_id: layer.module_id,
    op_type: layer.op_type,
    targets: unique,
    priority: recommendationPriority(unique),
    reasons,
    metrics,
    num_weights: layer.num_weights ?? 0,
  };
}

export async function buildImproveSweepPlan(input: {
  preset?: ImproveSweepPreset;
  maxModules?: number;
  paths?: Partial<ImprovePaths>;
  runtime?: Pick<ImproveRuntime, "now">;
} = {}): Promise<ImproveSweepPlan> {
  const preset = input.preset ?? "ppa";
  const paths = resolveImprovePaths(input.paths);
  const now = input.runtime?.now ?? (() => new Date());
  const pipelineIr = await readJsonFile<PipelineIR>(
    path.join(paths.outputRoot, "layer_ir.json"),
    pipelineIrSchema,
  );
  const recommendations: ImproveSweepRecommendation[] = [];
  const skipped: ImproveSweepPlan["skipped"] = [];
  for (const layer of pipelineIr.layers) {
    let baseline: Awaited<ReturnType<typeof loadBaselineMetrics>>;
    try {
      baseline = await loadBaselineMetrics(paths, layer.module_id);
    } catch (error: unknown) {
      skipped.push({
        module_id: layer.module_id,
        reason: `missing or invalid baseline reports: ${error instanceof Error ? error.message : String(error)}`,
      });
      continue;
    }
    if (!baseline.vivadoReport.success || !baseline.vivadoReport.timing_met) {
      skipped.push({
        module_id: layer.module_id,
        reason: "baseline Vivado report is not passing",
      });
      continue;
    }
    const recommendation = buildRecommendationForLayer(layer, baseline.metrics, preset);
    if (recommendation) {
      recommendations.push(recommendation);
    }
  }
  recommendations.sort((a, b) => {
    const priorityDelta = a.priority - b.priority;
    if (priorityDelta !== 0) return priorityDelta;
    const targetDelta = b.targets.length - a.targets.length;
    if (targetDelta !== 0) return targetDelta;
    const lutDelta = b.metrics.lut - a.metrics.lut;
    if (lutDelta !== 0) return lutDelta;
    return a.module_id.localeCompare(b.module_id);
  });
  return {
    generated_at: now().toISOString(),
    preset,
    recommendations: input.maxModules ? recommendations.slice(0, input.maxModules) : recommendations,
    skipped,
  };
}

export async function runImproveSweep(input: {
  preset?: ImproveSweepPreset;
  run?: boolean;
  keepReference?: boolean;
  maxModules?: number;
  paths?: Partial<ImprovePaths>;
  checkerConfig?: Partial<ImprovementCheckerConfig>;
  runtime?: ImproveRuntimeOverrides;
} = {}): Promise<ImproveSweepResult> {
  const paths = resolveImprovePaths(input.paths);
  const runtime = createImproveRuntime(input.runtime, paths);
  const plan = await buildImproveSweepPlan({
    preset: input.preset,
    maxModules: input.maxModules,
    paths,
    runtime,
  });
  const keepReference = input.keepReference ?? true;
  const results: ImproveSweepResult["results"] = [];
  if (input.run === true) {
    for (const item of plan.recommendations) {
      try {
        const result = await runImproveSequence(item.module_id, {
          targets: item.targets,
          keepReference,
          paths,
          checkerConfig: input.checkerConfig,
          runtime: input.runtime,
        });
        results.push({
          module_id: item.module_id,
          targets: result.targets,
          success: result.success,
          final_action: result.final_action,
          report_path: result.report_path,
        });
      } catch (error: unknown) {
        results.push({
          module_id: item.module_id,
          targets: item.targets,
          success: false,
          final_action: "no-change",
          report_path: "",
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }
  }

  const report: Omit<ImproveSweepResult, "report_path"> = {
    plan,
    ran: input.run === true,
    keep_reference: keepReference,
    results,
  };
  const reportPath = path.join(paths.reportsDir, `sweep_improve_${improvementStamp(runtime.now())}.json`);
  await writeJsonFile(reportPath, { ...report, report_path: reportPath });
  return { ...report, report_path: reportPath };
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
    "GOAL: store the actual weight / large-constant memories in BRAM18/BRAM36 instead of distributed LUT-RAM, and produce a measurable PPA win. Token BRAM allocations that leave LUT/FF essentially unchanged are rejected.",
    "HOW:",
    "  - First identify which large arrays dominate LUT-as-memory / scalar ROM cost in the original RTL. Move THOSE arrays to BRAM-backed synchronous memories; do not create a small unused or duplicate BRAM just to make `bram18_equiv` nonzero.",
    "  - Annotate the array on its declaration line: `(* rom_style = \"block\", ram_style = \"block\" *) reg signed [7:0] weights [0:OC*K_TOTAL-1];`. The attribute MUST sit immediately before the `reg` declaration.",
    "  - Reads MUST be synchronous: `reg signed [7:0] w_q; always @(posedge clk) w_q <= weights[addr];`. Async reads (`assign w = weights[addr];`) force LUTRAM no matter what attribute is set.",
    "  - Initialize with `$readmemh` inside an `initial begin ... end` block. Element-by-element assignment in initial blocks (`weights[0] = ..; weights[1] = ..;`) defeats Vivado's BRAM/ROM init pattern matching.",
    "  - If a synchronous BRAM read adds a cycle, retime the address/control pipeline so `valid_out` still appears at exactly the LayerIR `pipeline_latency_cycles`. The checker treats a pure latency slip as a failure even when values match.",
    "  - A good use-bram attempt should reduce LUT or FF materially while increasing BRAM. If your rewrite only changes BRAM by a few blocks with ~0% LUT/FF movement, it is the wrong approach.",
    "PITFALLS:",
    "  - Adding a registered read stage shifts the MAC schedule by one cycle; verify the FSM still drives `valid_out` after exactly `pipeline_latency_cycles` cycles.",
    "  - BRAM has fixed port counts. A weight memory read by N parallel MAC lanes either needs N replicated BRAMs (depth/width tradeoff) or a banked layout — see `weight_bank_paths` in the LayerIR.",
    "  - Do not keep the old LUT ROM live beside the new BRAM ROM. If both copies remain addressable, Vivado may preserve both and the improvement becomes a token BRAM pass instead of a real area reduction.",
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
  "reduce-ff": [
    "GOAL: reduce flip-flop count by at least the configured delta (`new.ff < baseline.ff * (1 - reduceFfMinDelta)`).",
    "HOW:",
    "  - Move large activation or staging buffers out of scalar registers. Vivado infers memory most reliably from one-dimensional unpacked arrays with a packed-wide word: `reg [WORD_BITS-1:0] mem [0:DEPTH-1]`.",
    "  - For line buffers, flatten bank/beat dimensions into one address (`addr = pixel_bank_addr * IN_BEATS + beat`) rather than declaring 2D unpacked memories like `[pixel][beat]`, which often map to FFs.",
    "  - Remove write-only diagnostic windows or dummy structural arrays. If a buffer is not read by the datapath, it should not exist in the improved RTL.",
    "  - For residual/add pipelines, stream or bank partial state instead of holding a full channel vector in registers when the interface is already tiled.",
    "PITFALLS:",
    "  - Do not trade the FF explosion for a single monolithic Verilog variable over Vivado's size limit. Bank large memories and keep each variable comfortably below the tool cap.",
    "  - A BRAM/LUTRAM read usually adds a registered cycle. Preserve the verified valid_out timing contract or adjust only if the goldens support the changed latency.",
  ].join("\n"),
  "improve-fmax": [
    "GOAL: improve post-synth Fmax by at least the configured delta, with a practical floor target for slow modules.",
    "HOW:",
    "  - Pipeline long multiply/shift/saturate chains. Register multiplier outputs, scaling results, and saturation decisions separately when WNS is dominated by arithmetic.",
    "  - Break high-fanout control signals by registering local enables in the state that consumes them.",
    "  - Replace very wide combinational muxes with registered memory reads or a balanced two-stage mux tree.",
    "  - Keep the public latency contract in mind: extra internal registers are allowed only if the output timing remains bit/cycle exact against Verilator.",
    "PITFALLS:",
    "  - Do not reduce critical path by duplicating huge datapaths unless LUT/FF usage remains sane. The checker still requires Vivado timing to pass and area regressions will show up in the report.",
    "  - Fmax is a timing result, not a cycle-count result. Do not pursue `reduce-latency` changes unless that target is explicitly requested.",
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
    "GOAL: increase real wall-clock throughput (FPS = Fmax / II). The acceptance gate evaluates FPS, not II alone — trading Fmax for II is rejected.",
    "II is measured per frame by the deterministic verifier as `(last_valid_out_cycle - first_valid_in_cycle) / num_frames`. FPS = `fmax_mhz * 1e6 / ii`. Acceptance gate: `new.fps > baseline.fps * (1 + delta)`.",
    "DSP PARALLELIZATION MANDATE: if baseline DSP count is N, you must return at least `ceil(N * minDspMultiplier)` DSPs (typically 2N). On a baseline-DSP=1 module, returning 1 DSP proves you did not replicate MAC lanes and the gate will reject the attempt. Adding parallel MACs costs DSPs — that is the explicit mechanism for this target.",
    "PRECEDENCE OVERRIDE: the protected pattern docs and reference modules describe the conservative serialized-MAC datapath used for first-pass correctness. When this target is active, those serialization guidelines DO NOT APPLY. Specifically the conv1x1/3x3/7x7 references' \"flat weights_path, one read per cycle, do not switch to parallel bank reads\" rule is SUSPENDED for this turn. Parallel MAC lanes and banked weight memories are the explicit mechanism for this target and are mandatory if you want to lower II.",
    "TIMING CONTRACT: the deterministic assayer checks `timing_actual_cycles == timing_expected_cycles`, where both refer to the PIPELINE-FILL latency only (cycles to first valid_out after first valid_in). Per-frame II is a separate measurement. You can lower per-frame II while preserving pipeline-fill latency — that is exactly what this target is asking for. Do NOT change time-to-first-output.",
    "HOW:",
    "  - Identify the resource that's serialized: shared single-port BRAM, single MAC lane, single accumulator. Replicate it (BRAM banks, MAC lanes) so independent inputs don't fight for the same port. Doubling MAC lanes typically halves II.",
    "  - Bank the weight memory by output-channel groups. If the baseline reads one weight per cycle into one MAC, fanning out to N banks lets N MACs work in parallel and cuts per-output cycles by ~N.",
    "  - Switch single-port BRAM to true dual-port (`RAMB36E2` with two independent read addresses) when read patterns conflict.",
    "  - Output ordering and bit-exactness MUST be preserved. The same goldens still apply. Parallel lanes are allowed only if their reduction matches the serial datapath's per-output value exactly.",
    "DSP-PACKING DISCIPLINE (mandatory — parallel lanes that synthesize as LUT multipliers fail the gate):",
    "  - Each parallel multiply MUST be annotated `(* use_dsp = \"yes\" *)` on the line before the assignment AND factored into a registered intermediate: `(* use_dsp = \"yes\" *) reg signed [W-1:0] mul_q [0:N-1]; always @(posedge clk) for (g = 0; g < N; g = g + 1) mul_q[g] <= a[g] * b[g];`. Vivado pattern-matches a DSP48E2 cell off this exact shape.",
    "  - Both operands of every parallel multiply MUST be `signed [N-1:0]` of identical width. Mixed signed/unsigned or width-mismatched operands routinely keep multipliers in LUT — that is the most common DSP-inference rejection.",
    "  - DO NOT put `(* dont_touch *)` on the multiplier output register. The DSP48E2 cell absorbs the output flop; `dont_touch` blocks that absorption and forces the multiply back into LUTs even when other annotations are correct.",
    "  - Register the multiplier output BEFORE feeding into shifts, saturation, or accumulation. A direct combinational `(a*b) >>> N` chain is rejected by the DSP inferrer.",
    "  - Put each lane's multiply in its own `always @(posedge clk)` registered statement inside an unconditional `for ... generate` loop or unrolled block. Multiplies hidden inside `for ... if (...)` conditional generate blocks may map inconsistently across Vivado versions; prefer unconditional registered multiplies.",
    "  - Sanity check: if your baseline used `DSP=2` and you doubled MAC lanes to 8, the synthesized DSP count should be at least 4 (typically 8). A reported `dsp_count == 1` with much higher LUT/FF and `fmax_mhz == 0` is the classic signature of failed DSP inference — the design parallelizes in simulation but goes back to LUT-ripple multipliers in synth and timing collapses.",
    "PITFALLS:",
    "  - II=1 designs need every BRAM read to be deterministic from the inputs. Address-pipelining can make II=1 hard to verify functionally.",
    "  - Doubling MACs doubles DSP usage. Confirm the board has the budget (ZCU102 / XCZU9EG has 2,520 DSP48E2 — plenty for a single layer, less so for a whole pipeline).",
    "  - Do NOT also touch pipeline-fill latency. Extra fill stages fail `timing_actual_cycles == timing_expected_cycles` and burn the attempt. For conv stems, pipeline-fill is dominated by line-buffer fill (the wait for the first complete receptive field of input rows), NOT by per-output MAC time — so parallelizing the per-output MAC compute does not change time-to-first-output in practice.",
    "  - Returning RTL with the same II as baseline is treated as a no-op and fails the acceptance gate.",
    "  - Returning RTL whose II is lower but whose synthesized `fmax_mhz` is 0 (timing failed to close) ALSO fails the gate. Bit-exact simulation is necessary but not sufficient — the design must close timing on xczu9eg with the original clock period. If you cannot pack parallel multiplies into DSPs, the design will not close timing.",
    "MANDATORY: you MUST emit a complete `verilogModuleSchema` JSON with a modified `verilog_source`. Refusing the task or returning only an explanation message is treated as agent_max_turns_exhausted and burns the attempt. Even an aggressive attempt with high DSP cost is preferred over no attempt.",
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
  "  - Output: a single JSON object matching `verilogModuleSchema` — `module_id`, `spec_hash`, `verilog_source` (the FULL improved RTL as a JSON-escaped string — MANDATORY, the schema rejects metadata-only output), `generated_by: \"Foundry\"`, `attempt: <attempt_index>`. No markdown fences, no commentary before or after the JSON object.",
].join("\n");

function summarizeBaselineMetrics(metrics: ImprovementMetrics): string {
  const parts = [
    `LUT: ${metrics.lut}`,
    `FF: ${metrics.ff}`,
    `DSP: ${metrics.dsp}`,
    `BRAM18-equivalent: ${metrics.bram}`,
  ];
  if (metrics.fmax_mhz !== undefined) {
    parts.push(`fmax_mhz: ${metrics.fmax_mhz}`);
  }
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
        lines.push(
          `  - use-bram: pass iff new.bram18_equiv >= ${baseline.bram + config.useBramMinGain} ` +
          `(baseline ${baseline.bram} + ${config.useBramMinGain}) AND either ` +
          `new.lut < ${(baseline.lut * (1 - config.useBramMinLutDelta)).toFixed(0)} ` +
          `or new.ff < ${(baseline.ff * (1 - config.useBramMinFfDelta)).toFixed(0)}. ` +
          "Token BRAM allocations with no meaningful LUT/FF reduction fail.",
        );
        break;
      case "reduce-lut": {
        const required = baseline.lut * (1 - config.reduceLutMinDelta);
        lines.push(
          `  - reduce-lut: pass iff new.lut < ${required.toFixed(0)} (= baseline.lut ${baseline.lut} * (1 - ${config.reduceLutMinDelta})).`,
        );
        break;
      }
      case "reduce-ff": {
        const required = baseline.ff * (1 - config.reduceFfMinDelta);
        lines.push(
          `  - reduce-ff: pass iff new.ff < ${required.toFixed(0)} (= baseline.ff ${baseline.ff} * (1 - ${config.reduceFfMinDelta})).`,
        );
        break;
      }
      case "improve-fmax": {
        const required = baseline.fmax_mhz !== undefined
          ? baseline.fmax_mhz < config.improveFmaxFloorMhz
            ? Math.min(config.improveFmaxFloorMhz, baseline.fmax_mhz * (1 + config.improveFmaxMinDelta))
            : baseline.fmax_mhz * (1 + config.improveFmaxMinDelta)
          : undefined;
        lines.push(
          `  - improve-fmax: pass iff new.fmax_mhz > ${required?.toFixed(2) ?? "(missing — baseline Vivado report needs fmax_mhz)"} (= relative +${config.improveFmaxMinDelta * 100}% with floor ${config.improveFmaxFloorMhz} MHz for slow modules).`,
        );
        break;
      }
      case "reduce-latency":
        lines.push(
          `  - reduce-latency: pass iff new.latency_cycles < ${baseline.latency_cycles ?? "(missing — baseline metrics need timing_actual_cycles)"}.`,
        );
        break;
      case "increase-throughput": {
        const baselineFps = baseline.ii !== undefined && baseline.fmax_mhz !== undefined
          ? (baseline.fmax_mhz * 1e6) / baseline.ii
          : undefined;
        const requiredFps = baselineFps !== undefined
          ? baselineFps * (1 + config.increaseThroughputMinFpsDelta)
          : undefined;
        const dspGateRequired = baseline.dsp >= 1
          ? Math.max(1, Math.ceil(baseline.dsp * config.increaseThroughputMinDspMultiplier))
          : 1;
        lines.push(
          `  - increase-throughput: pass iff new.fps > ${requiredFps?.toFixed(4) ?? "(missing — baseline needs fmax + ii)"} ` +
          `(= baseline.fps=${baselineFps?.toFixed(4) ?? "?"} * (1 + ${config.increaseThroughputMinFpsDelta})) ` +
          `AND new.dsp >= ${dspGateRequired} (= ceil(baseline.dsp=${baseline.dsp} * ${config.increaseThroughputMinDspMultiplier})). ` +
          "II-only wins are rejected; trading Fmax for cycles loses. Single-DSP returns on a serial-MAC baseline are rejected.",
        );
        break;
      }
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

export function buildFoundryImprovePrompt(
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
  ];

  if (input.preloaded_rtl_patterns) {
    sections.push(
      "",
      "== PRELOADED RTL KNOWLEDGE ==",
      "The orchestrator fetched this deterministically for the current LayerIR/op/contract. Use it as local architectural guidance; do not spend a tool turn re-fetching it. Reference Verilog is intentionally omitted in improve mode; the ORIGINAL RTL below is the implementation source of truth.",
      "",
      "pattern_markdown:",
      input.preloaded_rtl_patterns.pattern_markdown,
    );
    if (input.preloaded_rtl_patterns.license_notice) {
      sections.push("", "license_notice:", input.preloaded_rtl_patterns.license_notice);
    }
  }

  if (input.sequence_context && input.sequence_context.length > 0) {
    sections.push(
      "",
      "== PRIOR SEQUENCE SUCCESSES (LOCKED) ==",
      "This improve call is one step in a multi-target sequence. The ORIGINAL RTL below already includes these previously verified improvements. Preserve them; the sequence-level checker will re-check them after this step.",
      JSON.stringify(input.sequence_context.map((step) => ({
        target: step.target,
        final_action: step.final_action,
        metrics: step.metrics,
        verdict: step.verdict,
        report_path: step.report_path,
      })), null, 2),
    );
  }

  sections.push(
    "",
    "== ORIGINAL RTL (the source of truth — improve this) ==",
    "```verilog",
    input.original_module.verilog_source,
    "```",
  );

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
    "== OUTPUT CONTRACT (PERSISTENCE IS MANDATORY) ==",
    "Your improved RTL MUST reach disk through ONE of the two paths below. Skipping both means the entire turn is discarded — the orchestrator detects empty turns and aborts the run.",
    "",
    "Path A — preferred — call the MCP tool BEFORE your final message:",
    "  ```",
    "  mcp__nn2rtl-tools__write_verilog({",
    `    module: {`,
    `      module_id: "${input.module_id}",`,
    `      spec_hash: "${input.original_module.spec_hash}",`,
    "      verilog_source: \"<your full improved Verilog source>\",",
    "      generated_by: \"Foundry\",",
    `      attempt: ${input.attempt_index}`,
    "    },",
    "    output_dir: \"output\"",
    "  })",
    "  ```",
    "  The tool writes `output/rtl/<module_id>.v` and `output/rtl/<module_id>.meta.json` for you. The orchestrator reads the .v back from disk after your turn ends.",
    "",
    "Path B — acceptable fallback — inline the full source in your final structured-output JSON:",
    `  { "module_id": "${input.module_id}", "spec_hash": "${input.original_module.spec_hash}", "generated_by": "Foundry", "attempt": ${input.attempt_index}, "verilog_source": "<full improved Verilog>" }`,
    "  The orchestrator extracts the string and writes it to disk itself.",
    "",
    "If you used Path A, your final structured-output JSON may omit `verilog_source`:",
    `  { "module_id": "${input.module_id}", "spec_hash": "${input.original_module.spec_hash}", "generated_by": "Foundry", "attempt": ${input.attempt_index} }`,
    "",
    "Do NOT return ONLY metadata without first calling write_verilog. The orchestrator cannot improve the original RTL using metadata alone — it needs the new source.",
    "No markdown fences in the final JSON, no commentary outside it.",
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

function makeDefaultFoundryImproveFn(paths: ImprovePaths): (input: FoundryImproveInput, runtime: ImproveRuntime) => Promise<FoundryImproveResult> {
  return async function defaultFoundryImproveFn(
    input: FoundryImproveInput,
    runtime: ImproveRuntime,
  ): Promise<FoundryImproveResult> {
    const messages: SDKMessage[] = [];
    let finalResult: SDKResultMessage | null = null;
    const agentTurnStartTime = runtime.now();

    // Improve mode uses a dedicated prompt instead of `foundry.md + addendum`.
    // The normal generation prompt contains tool-reading and metadata-only
    // instructions that are correct for fresh generation but confusing here.
    const improveFoundryPrompt = await loadImproveFoundrySystemPrompt();

    for await (const message of runtime.queryFn({
      prompt: buildFoundryImprovePrompt(input),
      options: {
        cwd: defaultRepoRoot,
        model: AGENT_CONFIG.Foundry.model,
        systemPrompt: {
          type: "preset",
          preset: "claude_code",
          append: improveFoundryPrompt,
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
        // Match the foundry/improve_foundry agent-definition frontmatter:
        // both pin `effort: high`. The normal pipeline picks this up via
        // loadPluginAgentDefinition; the improve flow builds options
        // manually and was previously omitting it, which downgraded the
        // Opus thinking budget and produced shorter, less coherent turns.
        effort: "high",
        ...(input.resume_session_id ? { resume: input.resume_session_id } : {}),
      },
    })) {
      messages.push(message);
      if (isSdkResultMessage(message)) {
        finalResult = message;
      }
    }

    // Flush tool-use audits BEFORE anything else can throw. Without this,
    // hydrate failures (empty turns, malformed structured output) blow away
    // the only evidence of whether Foundry actually called write_verilog —
    // making it impossible to tell "agent skipped the tool" from "agent
    // called the tool and the tool failed".
    try {
      const improveAudits = extractToolUseAudits(messages, {
        agent: "Foundry",
        module_id: input.module_id,
        nowIso: agentTurnStartTime.toISOString(),
      });
      await appendToolUseAudits(improveAudits);
      await appendForeignMcpToolWarnings(improveAudits);
      await appendRunLog({
        event: "improve_foundry_turn_audit",
        agent: "Foundry",
        module_id: input.module_id,
        attempt: input.attempt_index,
        tool_call_count: improveAudits.filter((a) => a.kind === "tool_use").length,
        tools_called: improveAudits
          .filter((a) => a.kind === "tool_use")
          .map((a) => a.tool_name),
        message_count: messages.length,
        had_final_result: finalResult !== null,
      });
    } catch (auditErr) {
      // Audit logging is best-effort; never let it mask the real failure.
      await appendRunLog({
        event: "improve_foundry_audit_failed",
        module_id: input.module_id,
        attempt: input.attempt_index,
        reason: auditErr instanceof Error ? auditErr.message : String(auditErr),
      });
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
      // pulling `verilog_source` off the on-disk RTL — but only if it was
      // freshly written by `write_verilog` during this turn. Otherwise we'd
      // silently score the previous attempt's RTL as this turn's output.
      const verilogPath = path.join(paths.rtlDir, `${input.module_id}.v`);
      const verilogStat = await pathExists(verilogPath);
      if (!verilogStat) {
        throw err;
      }
      const { stat: statFn } = await import("node:fs/promises");
      const stat = await statFn(verilogPath);
      if (stat.mtime.getTime() < agentTurnStartTime.getTime()) {
        throw err;
      }
      const diskSource = await readFile(verilogPath, "utf8");
      metadata = {
        module_id: input.module_id,
        spec_hash: input.original_module.spec_hash,
        verilog_source: diskSource,
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
  const sections = [
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
  ];
  if (input.sequence_context && input.sequence_context.length > 0) {
    sections.push(
      "",
      "== PRIOR SEQUENCE SUCCESSES (MUST PRESERVE) ==",
      "The current target is running after earlier sequence steps already passed. The original RTL shown below includes those changes. Your advice must preserve them while fixing the current failed target.",
      JSON.stringify(input.sequence_context.map((step) => ({
        target: step.target,
        metrics: step.metrics,
        verdict: step.verdict,
        report_path: step.report_path,
      })), null, 2),
    );
  }
  sections.push(
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
  );
  return sections.join("\n");
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
  const contractId = resolveLayerContractId(input.layer);
  const signatures = signatureBundle({
    baseLayer: input.layer,
    runtimeLayer: input.layer,
    baseContractId: contractId,
    runtimeContractId: contractId,
    modelQuantization: input.layer.quantization_family,
  });
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
      `contract_id: ${contractId}`,
      `module_id: ${input.moduleId}`,
      `signature_hash: ${signatures.signature_hash}`,
      `exact_reference_key: ${signatures.exact_reference_key ?? "none"}`,
      `derived_from_networks: [${getActiveNetworkId()}]`,
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
    contract_id: contractId,
    contract_key: input.module.spec_hash,
    spec_hash: input.module.spec_hash,
    signature_hashes: [signatures.signature_hash],
    exact_reference_keys: signatures.exact_reference_key ? [signatures.exact_reference_key] : [],
    derived_from_networks: [getActiveNetworkId()],
    derived_from_modules: [input.moduleId],
    applicability: applicabilityForSignature({
      networkId: getActiveNetworkId(),
      signatures,
    }),
    contraindications: [],
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

async function writeImproveReport(
  paths: ImprovePaths,
  moduleId: string,
  targets: ImprovementTarget[],
  result: Omit<ImproveResult, "report_path"> & Partial<Pick<
    ImproveSequenceResult,
    | "sequence_steps"
    | "requested_targets"
    | "completed_targets"
    | "failed_targets"
    | "unattempted_targets"
    | "remaining_targets"
    | "partial_success"
    | "overall_success"
  >>,
): Promise<string> {
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
  if (targets.length > 1) {
    throw new Error(
      "runImprove is a single-target primitive. Use runImproveSequence for multiple targets so each improvement runs on the previous improved RTL.",
    );
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
  let preloadedRtlPatterns: RtlKnowledgeDoc | undefined;
  if (options.runtime?.foundryFn === undefined) {
    try {
      const loadedPatterns = await loadRetrospectorKnowledgeDoc(layer);
      preloadedRtlPatterns = {
        ...loadedPatterns,
        reference_verilog: null,
      };
      await appendRunLog({
        event: "improve_rtl_patterns_preloaded",
        module_id: moduleId,
        op_type: layer.op_type,
        contract_id: layer.contract_id ?? "flat-bus",
        pattern_markdown_chars: loadedPatterns.pattern_markdown.length,
        reference_verilog_chars: 0,
        reference_verilog_omitted: true,
      });
    } catch (error: unknown) {
      await appendRunLog({
        event: "improve_rtl_patterns_preload_failed",
        module_id: moduleId,
        reason: error instanceof Error ? error.message : String(error),
      });
    }
  }
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
          sequence_context: options.sequenceContext,
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
        preloaded_rtl_patterns: preloadedRtlPatterns,
        sequence_context: options.sequenceContext,
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
    const assayerResult = verifResultSchema.parse(verif);
    attempt.assayer_result = assayerResult;
    if (!isAssayerResultAcceptableForTargets(assayerResult, targets)) {
      attempt.failed_gate = "verilator";
      attempts.push(attempt);
      await persistAttempt(paths, moduleId, runId, attempt);
      if (attemptIndex === 2) continue;
      if (attemptIndex === 3) break;
      continue;
    }

    const synthPreflightIssues = synthesisPreflightViolations(module, layer);
    if (synthPreflightIssues.length > 0) {
      attempt.failed_gate = "vivado";
      attempt.vivado_report = synthesisPreflightReport(module, layer, synthPreflightIssues);
      attempts.push(attempt);
      await persistAttempt(paths, moduleId, runId, attempt);
      await appendRunLog({
        event: "improve_synthesis_preflight_failed",
        module_id: moduleId,
        attempt: attemptIndex,
        rules: synthPreflightIssues.map((issue) => issue.rule),
        violations: synthPreflightIssues,
      });
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

async function snapshotCanonicalFiles(paths: ImprovePaths, moduleId: string): Promise<Map<string, string>> {
  const files = [
    path.join(paths.rtlDir, `${moduleId}.v`),
    path.join(paths.rtlDir, `${moduleId}.meta.json`),
    path.join(paths.reportsDir, `${moduleId}.vivado.json`),
    path.join(paths.reportsDir, `${moduleId}.results.json`),
    path.join(paths.reportsDir, `${moduleId}.metrics.json`),
  ];
  const snapshots = new Map<string, string>();
  for (const filePath of files) {
    if (await pathExists(filePath)) {
      snapshots.set(filePath, await readFile(filePath, "utf8"));
    }
  }
  return snapshots;
}

async function restoreCanonicalFiles(snapshots: Map<string, string>): Promise<void> {
  for (const [filePath, snapshot] of snapshots) {
    await writeFile(filePath, snapshot, "utf8");
  }
}

function cloneAttemptsWithoutMessages(attempts: ImprovementAttemptRecord[]): ImprovementAttemptRecord[] {
  return attempts.map((attempt, index) => ({
    ...attempt,
    attempt_index: index + 1,
    messages: undefined,
  }));
}

function successfulAttemptFromResult(result: ImproveResult): ImprovementAttemptRecord | undefined {
  return result.attempts.find((attempt) => attempt.verdict?.overall === true && attempt.metrics !== undefined);
}

export async function runImproveSequence(
  moduleId: string,
  options: RunImproveOptions,
): Promise<ImproveSequenceResult> {
  const targets = uniqueTargets(options.targets);
  if (targets.length === 0) {
    throw new Error("runImproveSequence requires at least one target.");
  }
  if (targets.length === 1) {
    const single = await runImprove(moduleId, {
      ...options,
      targets,
    });
    return {
      ...single,
      sequence_steps: [{
        target: targets[0],
        success: single.success,
        final_action: single.final_action,
        report_path: single.report_path,
      }],
      requested_targets: targets,
      completed_targets: single.success ? targets : [],
      failed_targets: single.success ? [] : targets,
      unattempted_targets: [],
      remaining_targets: single.success ? [] : targets,
      partial_success: false,
      overall_success: single.success,
    };
  }

  const paths = resolveImprovePaths(options.paths);
  const runtime = createImproveRuntime(options.runtime, paths);
  const checkerConfig = {
    ...DEFAULT_IMPROVEMENT_CHECKER_CONFIG,
    ...options.checkerConfig,
  };
  const originalSnapshots = await snapshotCanonicalFiles(paths, moduleId);
  const originalBaseline = await loadBaselineMetrics(paths, moduleId);
  const sequenceStartedAt = runtime.now();
  const steps: ImproveResult[] = [];
  const sequenceSteps: ImproveSequenceStepSummary[] = [];
  const sequenceContext: ImproveSequenceContext[] = [];
  let bestTargets: ImprovementTarget[] = [];
  let bestSnapshots: Map<string, string> | null = null;
  let bestMetrics: ImprovementMetrics | undefined;
  let bestVerdict: ImprovementVerdict | undefined;
  let finalVerdict: ImprovementVerdict | undefined;
  let improvedReferencePath: string | undefined;
  let committedModulePath: string | undefined;
  let archivedOriginalPath: string | undefined;
  let finalAction: ImproveResult["final_action"] = "no-change";
  let success = false;
  let shouldRestoreOriginal = true;

  try {
    for (let index = 0; index < targets.length; index += 1) {
      const target = targets[index];
      const stepNow = new Date(sequenceStartedAt.getTime() + index * 1000);
      let step: ImproveResult;
      try {
        step = await runImprove(moduleId, {
          targets: [target],
          keepReference: false,
          paths,
          checkerConfig,
          runtime: {
            ...options.runtime,
            now: () => stepNow,
          },
          sequenceContext: [...sequenceContext],
        });
      } catch (error: unknown) {
        await appendRunLog({
          event: "improve_sequence_step_error",
          module_id: moduleId,
          target,
          requested_targets: targets,
          accepted_targets: bestTargets,
          error: error instanceof Error ? error.message : String(error),
        });
        await restoreCanonicalFiles(bestSnapshots ?? originalSnapshots);
        sequenceSteps.push({
          target,
          success: false,
          final_action: "no-change",
          report_path: "",
          error: error instanceof Error ? error.message : String(error),
        });
        continue;
      }
      steps.push(step);
      const stepSummary: ImproveSequenceStepSummary = {
        target,
        success: false,
        final_action: step.final_action,
        report_path: step.report_path,
      };
      if (!step.success) {
        await restoreCanonicalFiles(bestSnapshots ?? originalSnapshots);
        sequenceSteps.push(stepSummary);
        continue;
      }
      const successfulAttempt = successfulAttemptFromResult(step);
      const candidateTargets = [...bestTargets, target];
      const candidateBaseline = await loadBaselineMetrics(paths, moduleId);
      const candidateVerdict = evaluateImprovementTargets(
        originalBaseline.metrics,
        candidateBaseline.metrics,
        candidateTargets,
        checkerConfig,
      );
      if (!candidateVerdict.overall) {
        await appendRunLog({
          event: "improve_sequence_prefix_regressed",
          module_id: moduleId,
          requested_targets: targets,
          candidate_targets: candidateTargets,
          final_verdict: candidateVerdict,
          step_report_path: step.report_path,
        });
        await restoreCanonicalFiles(bestSnapshots ?? originalSnapshots);
        sequenceSteps.push(stepSummary);
        continue;
      }
      sequenceContext.push({
        target,
        report_path: step.report_path,
        final_action: step.final_action,
        metrics: successfulAttempt?.metrics,
        verdict: step.final_verdict,
      });
      bestTargets = candidateTargets;
      bestSnapshots = await snapshotCanonicalFiles(paths, moduleId);
      bestMetrics = candidateBaseline.metrics;
      bestVerdict = candidateVerdict;
      archivedOriginalPath ??= step.archived_original_path;
      committedModulePath = step.committed_module_path ?? committedModulePath;
      stepSummary.success = true;
      sequenceSteps.push(stepSummary);
    }

    if (bestTargets.length > 0 && bestSnapshots && bestMetrics && bestVerdict) {
      await restoreCanonicalFiles(bestSnapshots);
      finalVerdict = bestVerdict;
      success = true;
      if (options.keepReference) {
        const finalModule = await loadOriginalModule(paths, moduleId);
        const layer = await loadLayer(paths, moduleId);
        improvedReferencePath = await commitImprovedReference({
          paths,
          moduleId,
          module: finalModule,
          layer,
          targets: bestTargets,
          metrics: bestMetrics,
          verdict: bestVerdict,
          runtime,
        });
        finalAction = "kept-as-variant";
        shouldRestoreOriginal = true;
      } else {
        finalAction = "replaced";
        shouldRestoreOriginal = false;
      }
    }
  } finally {
    if (shouldRestoreOriginal) {
      await restoreCanonicalFiles(originalSnapshots);
    }
  }

  const attempts = cloneAttemptsWithoutMessages(steps.flatMap((step) => step.attempts));
  const retrospectorAdvice = [...steps]
    .reverse()
    .find((step) => step.retrospector_advice !== undefined)
    ?.retrospector_advice;
  const attemptedTargets = sequenceSteps.map((step) => step.target);
  const reportTargets = success ? bestTargets : targets;
  const completedTargets = success ? bestTargets : [];
  const failedTargets = targets.filter(
    (target) => attemptedTargets.includes(target) && !completedTargets.includes(target),
  );
  const unattemptedTargets = targets.filter((target) => !attemptedTargets.includes(target));
  const remainingTargets = [...failedTargets, ...unattemptedTargets];
  const overallSuccess = completedTargets.length === targets.length;
  const reportWithoutPath: Omit<ImproveResult, "report_path"> & Pick<
    ImproveSequenceResult,
    | "sequence_steps"
    | "requested_targets"
    | "completed_targets"
    | "failed_targets"
    | "unattempted_targets"
    | "remaining_targets"
    | "partial_success"
    | "overall_success"
  > = {
    module_id: moduleId,
    targets: reportTargets,
    final_action: finalAction,
    success,
    baseline_metrics: originalBaseline.metrics,
    attempts,
    final_verdict: finalVerdict ?? steps.at(-1)?.final_verdict,
    committed_module_path: success && finalAction === "replaced" ? committedModulePath : undefined,
    archived_original_path: success && finalAction === "replaced" ? archivedOriginalPath : undefined,
    improved_reference_path: improvedReferencePath,
    retrospector_advice: retrospectorAdvice,
    sequence_steps: sequenceSteps,
    requested_targets: targets,
    completed_targets: completedTargets,
    failed_targets: failedTargets,
    unattempted_targets: unattemptedTargets,
    remaining_targets: remainingTargets,
    partial_success: success && !overallSuccess,
    overall_success: overallSuccess,
  };
  const reportPath = await writeImproveReport(paths, moduleId, reportTargets, reportWithoutPath);
  return {
    ...reportWithoutPath,
    report_path: reportPath,
  };
}

export async function runImproveCli(argv: string[] = process.argv.slice(2)): Promise<void> {
  if (argv[0] === "sweep") {
    const cli = parseImproveSweepCliArgs(argv.slice(1));
    if (cli.networkId) {
      setActiveNetwork(cli.networkId);
    }
    const result = await runImproveSweep({
      preset: cli.preset,
      run: cli.run,
      keepReference: cli.keepReference,
      maxModules: cli.maxModules,
    });
    console.log(
      `Improve sweep ${result.plan.preset}: ${result.plan.recommendations.length} recommendation(s); ` +
        `${result.ran ? `ran ${result.results.length}` : "dry run"}; report: ${result.report_path}`,
    );
    for (const item of result.plan.recommendations) {
      console.log(`  ${item.module_id}: ${item.targets.join(", ")} (${item.reasons.join("; ")})`);
    }
    return;
  }
  const cli = parseImproveCliArgs(argv);
  if (cli.networkId) {
    setActiveNetwork(cli.networkId);
  }
  const result = await runImproveSequence(cli.moduleId, {
    targets: cli.targets,
    keepReference: cli.keepReference,
  });
  console.log(
    `Improve ${result.module_id} [${result.targets.join(", ")}]: ${result.final_action}; report: ${result.report_path}`,
  );
}
