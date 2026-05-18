import { existsSync } from "node:fs";
import { appendFile, access, mkdir, readFile, rename, unlink, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { parse as parseYaml } from "yaml";
import { z } from "zod";

import {
  query,
  type AgentDefinition,
  type EffortLevel,
  type OutputFormat,
  type SDKMessage,
  type SDKResultMessage,
} from "./claude-agent-sdk-compat.js";
import {
  AGENT_CONFIG,
  FAILURE_CLASSIFIER_CONFIG,
  PIPELINE_CONFIG,
  RETROSPECTOR_CONFIG,
  parseBooleanEnv,
  type AgentName,
} from "./config.js";
import {
  defaultNetworkId,
  getNetwork,
  isKnownNetworkId,
  listNetworks,
  outputDirForNetwork,
} from "./networks.js";
import { PipelineStateManager } from "./pipeline.js";
import {
  layerIrSchema as layerIrZod,
  failureClassificationSchema as failureClassificationZod,
  pipelineIrSchema as pipelineIrZod,
  retrospectorAdviceSchema as retrospectorAdviceZod,
  synthesisReportSchema as synthesisReportZod,
  verifResultSchema as verifResultZod,
  verilogModuleSchema as verilogModuleZod,
} from "./schemas.js";
import {
  applicabilityForSignature,
  signatureBundle,
  signatureCandidateMatchLevel,
  signatureMatchRank,
  signaturePaddingMatches,
  type LayerSignature,
  type SignatureMatchLevel,
  type SignatureTarget,
} from "./signatures.js";
import {
  contractFitFailure,
  contractSelectionForLayer,
  contractSidecarFields,
  contractTestbenchTemplatePath,
  loadContractMetadata,
  resolveLayerContractId,
} from "./contracts.js";
import type {
  ContractId,
  LayerIR,
  FailureClassification,
  ModelUsageEntry,
  PipelineIR,
  RetrospectorAdvice,
  RetrospectorBaseArtifact,
  RetrospectorNextActor,
  VerifResult,
  VerilogModule,
} from "./types.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const sdkRoot = path.resolve(
  __dirname,
  path.basename(__dirname) === "dist" ? ".." : ".",
);
const repoRoot = path.resolve(sdkRoot, "..");
const pluginPath = path.resolve(repoRoot, "nn2rtl-plugin");
let activeNetworkId = process.env.NN2RTL_NETWORK_ID || defaultNetworkId();
let activeOutputRoot: string | null = process.env.NN2RTL_OUTPUT_DIR
  ? path.resolve(repoRoot, process.env.NN2RTL_OUTPUT_DIR)
  : null;

export const AGENT_SLUGS = {
  Cartographer: "cartographer",
  Foundry: "foundry",
  Surgeon: "surgeon",
} as const satisfies Record<AgentName, string>;

const AGENT_MCP_TOOLS = {
  cartographer: ["mcp__nn2rtl-tools__read_weights"],
  foundry: [
    "mcp__nn2rtl-tools__write_verilog",
    "mcp__nn2rtl-tools__get_rtl_patterns",
    "mcp__nn2rtl-tools__get_failure_corpus",
    "mcp__nn2rtl-tools__compute_layer_reference",
  ],
  surgeon: [
    "mcp__nn2rtl-tools__write_verilog",
    "mcp__nn2rtl-tools__get_rtl_patterns",
    "mcp__nn2rtl-tools__get_failure_corpus",
    "mcp__nn2rtl-tools__compute_layer_reference",
  ],
} as const;

const APPEND_SKILL_TO_PROMPT = {
  cartographer: true,
  foundry: false,
  surgeon: false,
} as const;

type SynthesisReport = z.infer<typeof synthesisReportZod>;
type AgentSlug = (typeof AGENT_SLUGS)[AgentName];
type AgentRunResult<T> = {
  payload: T;
  result: SDKResultMessage;
  messages: SDKMessage[];
};
type RtlAgentRunResult = AgentRunResult<VerilogModule> & {
  draft_doc?: DocDraft | null;
  doc_request?: CreateNewDocRequest | null;
};
type DelegatedAgentRunOptions = {
  prompt?: string;
  resumeSessionId?: string;
};
type FrontmatterRecord = Record<string, unknown>;

type PortDirection = "input" | "output" | "inout";
type ParsedTopPort = {
  declaration: string;
  direction: PortDirection;
  width_bits: number | null;
};

export type SynthesisFn = (module: VerilogModule, layer: LayerIR) => Promise<SynthesisReport>;
export type AssayerFn = (
  module: VerilogModule,
  layer: LayerIR,
) => Promise<VerifResult>;

export type ReadWeightsFn = (
  checkpoint_path: string,
  quantization_config: object,
) => Promise<PipelineIR>;

export type OrchestratorRuntime = {
  now: () => Date;
  queryFn: typeof query;
  synthesisFn: SynthesisFn;
  assayerFn: AssayerFn;
  // Deterministic LayerIR extraction. Defaults to a dynamic import of
  // `mcp/tools.ts::read_weights` (which shells out to Python). Tests inject
  // a stub so they don't need a real .pth file or a Python install.
  readWeightsFn: ReadWeightsFn;
};

export type RunPipelineOptions = {
  networkId?: string;
  resume?: boolean;
  runtime?: Partial<OrchestratorRuntime>;
  maxRetries?: number;
  // Optional programmatic override for tests/experiments. The default comes
  // from PIPELINE_CONFIG.self_improve / NN2RTL_SELF_IMPROVE.
  selfImprove?: boolean;
  // When set, restrict the pipeline to a single module (scoped run for
  // testing). The pipeline state is built with only that module in the
  // module set, so isDone() resolves after it passes or fail_aborts.
  only?: string;
  // Inverse of `only` — list of module_ids to exclude from the run. The
  // pipeline state is built with every other module; excluded ones never
  // appear in moduleOrder and don't block isDone().
  except?: string[];
};

const DEFAULT_ORCHESTRATOR_RUNTIME: OrchestratorRuntime = {
  now: () => new Date(),
  queryFn: query,
  synthesisFn: (module, layer) => invokeVivado(module, layer),
  assayerFn: (module, layer) => runAssayerDeterministic(module, layer),
  readWeightsFn: async (checkpoint_path, quantization_config) => {
    const mcpTools = (await import(MCP_TOOLS_MODULE_PATH)) as {
      read_weights: ReadWeightsFn;
    };
    return mcpTools.read_weights(checkpoint_path, quantization_config);
  },
};

function toOutputFormat(schema: z.ZodType): OutputFormat {
  return {
    type: "json_schema",
    schema: z.toJSONSchema(schema) as Record<string, unknown>,
  };
}

const failureClassificationOutputFormat = toOutputFormat(failureClassificationZod);
const retrospectorAdviceOutputFormat = toOutputFormat(retrospectorAdviceZod);

// Foundry / Surgeon emit METADATA only in their final structured output.
// `verilog_source` is intentionally NOT in this schema — the agents persist
// the actual RTL via the `mcp__nn2rtl-tools__write_verilog` tool, and the
// orchestrator hydrates the full `VerilogModule` from disk after the agent
// returns. Re-serializing 10+ KB of Verilog as a JSON-escaped string in the
// final message was responsible for an ~30-50% structured-output parse-fail
// rate on long generations (any unescaped backslash, embedded `"`, or stray
// newline broke the whole final JSON), recovered through a heavy disk
// fallback path that lost cost / session / draft_doc telemetry. Dropping
// `verilog_source` from the agent contract makes the structured output
// ~200 bytes instead of ~12 KB and effectively eliminates the parse-fail
// mode while keeping the schema gate over the metadata + draft_doc.
// Lenient: agent output is metadata-only OR may inline verilog_source. The
// system prompt asks for metadata-only (write_verilog persists the source),
// but foundry.md has a strong prior to inline that's hard to suppress; the
// optional field absorbs that without a parse failure. `hydrateVerilogModuleFromDisk`
// prefers the inline string when present and falls back to
// the on-disk write_verilog path otherwise.
const verilogModuleAgentOutputZod = verilogModuleZod
  .omit({ verilog_source: true })
  .extend({ verilog_source: z.string().optional() });
type VerilogModuleAgentOutput = z.infer<typeof verilogModuleAgentOutputZod>;
const verilogModuleAgentOutputFormat = toOutputFormat(verilogModuleAgentOutputZod);

const docDraftZod = z
  .object({
    title: z.string().min(1),
    pattern_markdown: z.string().min(1),
    reference_verilog: z.string().min(1),
    notes: z.string().optional(),
  })
  .strict();
type DocDraft = z.infer<typeof docDraftZod>;

type ClosestDocSnippet = {
  id: string;
  tier: DocTier;
  kind: "pattern" | "reference";
  op_type: LayerIR["op_type"] | "shared";
  contract_id?: ContractId;
  relative_path: string;
  text: string;
};
type NewDocFailureContext = {
  reason: string;
  previous_contract_id?: ContractId;
  previous_contract_key?: string;
  failure_result?: VerifResult;
  flagged_contracts?: ContractFlag[];
};
type CreateNewDocRequest = {
  enabled: true;
  destination_tier: "probationary";
  contract_id: ContractId;
  contract_key: string;
  reason: string;
  no_external_retrieval: true;
  closest_existing_docs: ClosestDocSnippet[];
  failure_context: NewDocFailureContext;
};

const rtlAgentWithDocZod = z
  .object({
    module: verilogModuleAgentOutputZod,
    draft_doc: docDraftZod,
  })
  .strict();
type RtlAgentWithDoc = z.infer<typeof rtlAgentWithDocZod>;
const rtlAgentWithDocOutputFormat = toOutputFormat(rtlAgentWithDocZod);

/**
 * Hydrate a full `VerilogModule` (with `verilog_source`) from the agent's
 * metadata-only payload by reading the .v that the agent persisted via
 * `write_verilog`. Throws when the .v is missing — that means the agent
 * skipped its only required side effect.
 */
async function hydrateVerilogModuleFromDisk(
  metadata: VerilogModuleAgentOutput,
  layerIr: LayerIR,
): Promise<VerilogModule> {
  const expectedSpecHash = computeExpectedSpecHash(layerIr);
  if (metadata.module_id !== layerIr.module_id) {
    throw new SpecHashMismatchError(
      `${metadata.generated_by} returned module_id='${metadata.module_id}' for LayerIR.module_id='${layerIr.module_id}'. ` +
        `Retry with the exact module_id and selected contract '${currentContractId(layerIr)}'.`,
    );
  }
  if (metadata.spec_hash !== expectedSpecHash) {
    throw new SpecHashMismatchError(
      `${metadata.generated_by} returned spec_hash='${metadata.spec_hash}', expected '${expectedSpecHash}' ` +
        `for selected contract '${currentContractId(layerIr)}'. This usually means the agent reused a prior-contract ` +
        `response after a contract switch. Retry using the current LayerIR.contract_id/io_mode and return the exact ` +
        `expected_spec_hash.`,
    );
  }

  const rtlDir = resolvePipelineConfigPath(PIPELINE_CONFIG.rtl_dir);
  const verilogPath = path.join(rtlDir, `${metadata.module_id}.v`);
  let source: string | undefined;
  // Path 1: agent inlined verilog_source in the structured output. Persist
  // it to the canonical .v path so the rest of the pipeline (verilator,
  // vivado, write_verilog audit) sees the same file shape it would in the
  // metadata-only path.
  if (typeof metadata.verilog_source === "string" && metadata.verilog_source.trim()) {
    source = metadata.verilog_source;
    await mkdir(path.dirname(verilogPath), { recursive: true });
    await writeFile(verilogPath, source, "utf8");
  } else {
    // Path 2: agent persisted via write_verilog and returned metadata only.
    try {
      source = await readFile(verilogPath, "utf8");
    } catch {
      throw new Error(
        `${metadata.generated_by} returned metadata for module '${metadata.module_id}' ` +
          `but no Verilog source was persisted. Expected ${verilogPath} from a prior ` +
          `mcp__nn2rtl-tools__write_verilog call (or an inline verilog_source in the ` +
          `final structured output). The agent must persist RTL one of those two ways.`,
      );
    }
  }
  if (!source.trim()) {
    throw new Error(
      `${metadata.generated_by} persisted an empty Verilog file at ${verilogPath} ` +
        `for module '${metadata.module_id}'.`,
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

export function createOrchestratorRuntime(
  overrides: Partial<OrchestratorRuntime> = {},
): OrchestratorRuntime {
  return {
    ...DEFAULT_ORCHESTRATOR_RUNTIME,
    ...overrides,
  };
}

export function resolveFromSdk(relativePath: string): string {
  return path.resolve(sdkRoot, relativePath);
}

function resolveRepoPathMaybeAbsolute(inputPath: string): string {
  const normalized = normalizePathForCurrentHost(inputPath);
  return path.isAbsolute(normalized) || isWindowsAbsolutePath(normalized)
    ? normalized
    : path.resolve(repoRoot, normalized);
}

export function setActiveNetwork(networkId: string): void {
  const network = getNetwork(networkId);
  activeNetworkId = network.id;
  activeOutputRoot = outputDirForNetwork(network.id, repoRoot);
  process.env.NN2RTL_NETWORK_ID = network.id;
  process.env.NN2RTL_OUTPUT_DIR = activeOutputRoot;
}

export function getActiveNetworkId(): string {
  const envId = process.env.NN2RTL_NETWORK_ID;
  return isKnownNetworkId(envId) ? envId : activeNetworkId;
}

export function getPipelineOutputRoot(): string {
  const envRoot = process.env.NN2RTL_OUTPUT_DIR;
  if (envRoot && envRoot.trim()) {
    return resolveRepoPathMaybeAbsolute(envRoot);
  }
  return activeOutputRoot ?? outputDirForNetwork(getActiveNetworkId(), repoRoot);
}

export function resolvePipelineConfigPath(configPath: string): string {
  const legacyOutputRoot = resolveFromSdk(PIPELINE_CONFIG.output_dir);
  const resolved = resolveFromSdk(configPath);
  const outputRoot = getPipelineOutputRoot();
  if (resolved === legacyOutputRoot) return outputRoot;
  if (resolved.startsWith(legacyOutputRoot + path.sep)) {
    return path.join(outputRoot, path.relative(legacyOutputRoot, resolved));
  }
  return resolved;
}

function isWindowsAbsolutePath(inputPath: string): boolean {
  return /^[a-zA-Z]:[\\/]/.test(inputPath);
}

function normalizePathForCurrentHost(inputPath: string): string {
  const normalized = inputPath.replace(/\\/g, "/");
  if (process.platform !== "win32") {
    const drivePath = normalized.match(/^([a-zA-Z]):\/(.*)$/);
    if (drivePath) {
      return `/mnt/${drivePath[1].toLowerCase()}/${drivePath[2]}`;
    }
  }
  if (process.platform === "win32") {
    const wslPath = normalized.match(/^\/mnt\/([a-zA-Z])(?:\/(.*))?$/);
    if (wslPath) {
      const rest = wslPath[2] ?? "";
      return rest ? `${wslPath[1].toUpperCase()}:/${rest}` : `${wslPath[1].toUpperCase()}:/`;
    }
  }
  return normalized;
}

function resolveInputPathForCurrentHost(inputPath: string): string {
  const normalized = normalizePathForCurrentHost(inputPath);
  return path.isAbsolute(normalized) || isWindowsAbsolutePath(normalized)
    ? normalized
    : path.resolve(normalized);
}

function pathFingerprintKey(inputPath: string): string {
  const normalized = inputPath.replace(/\\/g, "/");
  const drivePath = normalized.match(/^([a-zA-Z]):\/(.*)$/);
  if (drivePath) {
    return `/${drivePath[1].toLowerCase()}/${drivePath[2]}`.toLowerCase();
  }
  const wslPath = normalized.match(/^\/mnt\/([a-zA-Z])(?:\/(.*))?$/);
  if (wslPath) {
    return `/${wslPath[1].toLowerCase()}/${wslPath[2] ?? ""}`.toLowerCase();
  }
  return path.resolve(normalized).replace(/\\/g, "/").toLowerCase();
}

function normalizeLayerFilePathsForCurrentHost(layer: LayerIR): LayerIR {
  return {
    ...layer,
    weights_path: normalizePathForCurrentHost(layer.weights_path),
    bias_path: layer.bias_path ? normalizePathForCurrentHost(layer.bias_path) : null,
    golden_inputs_path: normalizePathForCurrentHost(layer.golden_inputs_path),
    golden_outputs_path: normalizePathForCurrentHost(layer.golden_outputs_path),
    weight_bank_paths: layer.weight_bank_paths?.map(normalizePathForCurrentHost),
  };
}

function normalizePipelineIrForCurrentHost(pipelineIr: PipelineIR): PipelineIR {
  return {
    ...pipelineIr,
    layers: pipelineIr.layers.map(normalizeLayerFilePathsForCurrentHost),
  };
}

// PPA gates — a module that passes functional verification but fails these
// is treated as a real hardware failure and routed back to Surgeon via a
// synthesized VerifResult. Thresholds come from the README's FPGA targets.
const FMAX_TARGET_MHZ = 50;

// Maps a Vivado outcome to either a pass (null) or a synthesized VerifResult
// with the correct failure_class. The classification matters because Surgeon
// uses it to pick the repair strategy — "add a pipeline register" is very
// different from "remove a non-synthesizable construct."
// Vivado reports can be large; summarize as head + ERROR/CRITICAL/WARNING
// lines + tail so Surgeon sees root-cause diagnostics without megabytes of
// table noise.
const SYNTH_REPORT_HEAD_BYTES = 2_500;
const SYNTH_REPORT_TAIL_BYTES = 3_500;
const SYNTH_REPORT_ERRORS_BYTES = 4_000;
function capSynthesisReport(report: string): string {
  if (report.length <= SYNTH_REPORT_HEAD_BYTES + SYNTH_REPORT_TAIL_BYTES) {
    return report;
  }
  const head = report.slice(0, SYNTH_REPORT_HEAD_BYTES);
  const tail = report.slice(-SYNTH_REPORT_TAIL_BYTES);
  const errorLines = report
    .split(/\r?\n/)
    .filter((line) => /CRITICAL WARNING|ERROR|error:|Error:|VIOLATED/.test(line))
    .join("\n");
  const errorBlock =
    errorLines.length > SYNTH_REPORT_ERRORS_BYTES
      ? errorLines.slice(0, SYNTH_REPORT_ERRORS_BYTES) +
        `\n...[${errorLines.length - SYNTH_REPORT_ERRORS_BYTES} more diagnostic-line bytes elided]...`
      : errorLines;
  const elided = report.length - head.length - tail.length;
  return [
    "--- HEAD ---",
    head,
    errorBlock ? "--- ERRORS ---" : "",
    errorBlock,
    `--- (middle ${elided} bytes elided) ---`,
    "--- TAIL ---",
    tail,
  ]
    .filter(Boolean)
    .join("\n");
}

function evaluateSynthesis(
  moduleId: string,
  verifiedResult: VerifResult,
  report: SynthesisReport,
): VerifResult | null {
  if (!report.success) {
    // Vivado emitted a syntax/elaboration/synthesis error after simulation
    // had already passed. Fix strategy: rewrite only the synth-hostile RTL.
    return {
      ...verifiedResult,
      module_id: moduleId,
      status: "fail",
      failure_class: "synthesis_failed",
      fix_hint: [
        "Vivado synthesis failed after functional verification passed.",
        "Repair the RTL so ZCU102 (xczu9eg-ffvb1156-2-e) synth_design succeeds.",
        "Look at the HEAD and DIAGNOSTICS sections below for the root cause; the TAIL is usually table/log noise.",
        "Vivado output summary (head + diagnostics + tail):",
        capSynthesisReport(report.report),
      ].join("\n\n"),
    };
  }

  // Vivado prints `WNS = NA` for trivially-meeting designs that have no
  // inter-FF setup paths -- e.g. a stream-through ReLU where every output
  // register is driven from primary inputs only. In that case the parser
  // reads `timing_met = true` from the "All user specified timing
  // constraints are met" line in the report but cannot extract a numeric
  // WNS / Fmax. Treat the explicit timing-met assertion as authoritative
  // and only fail here when timing actually has no result AND was not
  // confirmed met. The downstream Fmax-vs-target gate further down still
  // applies when there IS a measurable result.
  if (!report.timing_met && (report.fmax_mhz <= 0 || report.wns_ns === null)) {
    return {
      ...verifiedResult,
      module_id: moduleId,
      status: "fail",
      failure_class: "synthesis_failed",
      fix_hint: [
        "Vivado synthesis succeeded but did not emit a measurable timing result.",
        "Repair the RTL or synthesis flow so report_timing_summary reports WNS.",
        "Vivado output summary (head + diagnostics + tail):",
        capSynthesisReport(report.report),
      ].join("\n\n"),
    };
  }

  // Fmax-vs-target gate. Only meaningful when there's a numeric WNS to
  // compute Fmax from -- the trivial WNS=NA case (handled above) was
  // accepted with a `timing_met = true` from the report's explicit
  // "constraints met" line, and there's no critical path to push back on
  // for those designs.
  if (
    report.wns_ns !== null &&
    (!report.timing_met || report.fmax_mhz < FMAX_TARGET_MHZ)
  ) {
    // Synthesis succeeded but critical path is too long. Fix strategy:
    // insert a pipeline register in the longest combinational path.
    // Note the latency-contract implication — adding a register changes
    // pipeline_latency_cycles, so Surgeon also has to update the handshake.
    return {
      ...verifiedResult,
      module_id: moduleId,
      status: "fail",
      failure_class: "missing_pipeline_register",
      fix_hint: [
        `Synthesis passed but Fmax ${report.fmax_mhz.toFixed(2)} MHz is below the ${FMAX_TARGET_MHZ} MHz target.`,
        `Vivado WNS: ${report.wns_ns.toFixed(3)} ns.`,
        "Insert a pipeline register to break the critical path, and update pipeline_latency_cycles to match.",
        "Vivado output summary (head + diagnostics + tail):",
        capSynthesisReport(report.report),
      ].join("\n\n"),
    };
  }

  return null;
}

function reportPath(fileName: string): string {
  return path.join(resolvePipelineConfigPath(PIPELINE_CONFIG.reports_dir), fileName);
}

/**
 * Bus-width capability gate. Returns a `fix_hint` string when a layer's
 * per-stream input or output bus exceeds
 * `PIPELINE_CONFIG.MAX_SUPPORTED_BUS_BITS`, or null when the layer is within
 * capability. Add layers pack lhs and rhs into one top-level data_in port, so
 * they are checked by each operand width rather than by the concatenated
 * `input_width_bits`.
 */
export function checkBusWidthCapability(layer: LayerIR): string | null {
  return contractFitFailure(layer);
}

export function normalizeAgentName(slug: AgentSlug): AgentName {
  const match = Object.entries(AGENT_SLUGS).find(([, value]) => value === slug);
  if (!match) {
    throw new Error(`No AgentName mapping found for slug '${slug}'.`);
  }

  return match[0] as AgentName;
}

export function parseFrontmatter(
  markdown: string,
): { frontmatter: FrontmatterRecord; body: string } {
  const match = markdown.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/);
  if (!match) {
    throw new Error("Expected agent markdown to start with YAML frontmatter.");
  }

  const [, rawFrontmatter, body] = match;
  const parsed: unknown = parseYaml(rawFrontmatter);
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Agent frontmatter must be a YAML mapping.");
  }

  const frontmatter: FrontmatterRecord = {};
  for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
    frontmatter[key] = value;
  }

  return { frontmatter, body: body.trim() };
}

// Normalize a frontmatter value into the string list the dispatcher expects.
// The parser accepts either an explicit YAML list (`tools: [Bash, Read]`) or
// the legacy inline CSV form (`tools: Bash, Read`) used in the existing agent
// markdown files.
export function toStringList(value: unknown): string[] | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }

  if (Array.isArray(value)) {
    const parts = value
      .map((entry) => (typeof entry === "string" ? entry.trim() : String(entry).trim()))
      .filter(Boolean);
    return parts.length > 0 ? parts : undefined;
  }

  if (typeof value === "string") {
    const parts = value
      .split(",")
      .map((entry) => entry.trim())
      .filter(Boolean);
    return parts.length > 0 ? parts : undefined;
  }

  throw new Error(`Unsupported frontmatter list value: ${JSON.stringify(value)}`);
}

function isResultMessage(message: SDKMessage): message is SDKResultMessage {
  return message.type === "result" && "modelUsage" in message;
}

export async function readText(filePath: string): Promise<string> {
  return readFile(filePath, "utf8");
}

export async function pathExists(filePath: string): Promise<boolean> {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

type GoldenVectorFile = {
  numVectors: number;
  samplesPerVector: number;
  bytesPerSample: number;
  wordsPerSample: number;
  vectors: number[][];
};

const GOLDEN_MAGIC = "NN2V";
const GOLDEN_VERSION = 2;
const GOLDEN_HEADER_BYTES = 20;

function wordsPerSampleForBytes(bytesPerSample: number): number {
  return Math.ceil(bytesPerSample / 4);
}

async function readGoldenVectorFile(filePath: string): Promise<GoldenVectorFile> {
  const hostPath = normalizePathForCurrentHost(filePath);
  const buffer = await readFile(hostPath);
  if (buffer.length < GOLDEN_HEADER_BYTES) {
    throw new Error(`Golden vector file '${hostPath}' is truncated.`);
  }
  if (buffer.subarray(0, 4).toString("ascii") !== GOLDEN_MAGIC) {
    throw new Error(`Golden vector file '${hostPath}' has wrong magic; expected '${GOLDEN_MAGIC}'.`);
  }
  const version = buffer.readUInt32LE(4);
  if (version !== GOLDEN_VERSION) {
    throw new Error(`Golden vector file '${hostPath}' has unsupported version ${version}.`);
  }
  const numVectors = buffer.readUInt32LE(8);
  const samplesPerVector = buffer.readUInt32LE(12);
  const bytesPerSample = buffer.readUInt32LE(16);
  const wordsPerSample = wordsPerSampleForBytes(bytesPerSample);
  const expectedWords = numVectors * samplesPerVector * wordsPerSample;
  const expectedBytes = GOLDEN_HEADER_BYTES + expectedWords * 4;
  if (buffer.length < expectedBytes) {
    throw new Error(
      `Golden vector file '${hostPath}' data is truncated; expected ${expectedBytes} bytes, found ${buffer.length}.`,
    );
  }

  const vectors: number[][] = [];
  let offset = GOLDEN_HEADER_BYTES;
  for (let vectorIndex = 0; vectorIndex < numVectors; vectorIndex += 1) {
    const row: number[] = [];
    for (let word = 0; word < samplesPerVector * wordsPerSample; word += 1) {
      row.push(buffer.readInt32LE(offset));
      offset += 4;
    }
    vectors.push(row);
  }

  return { numVectors, samplesPerVector, bytesPerSample, wordsPerSample, vectors };
}

async function writeGoldenVectorFile(filePath: string, file: GoldenVectorFile): Promise<void> {
  const totalWords = file.vectors.reduce((sum, row) => sum + row.length, 0);
  const buffer = Buffer.alloc(GOLDEN_HEADER_BYTES + totalWords * 4);
  buffer.write(GOLDEN_MAGIC, 0, "ascii");
  buffer.writeUInt32LE(GOLDEN_VERSION, 4);
  buffer.writeUInt32LE(file.numVectors, 8);
  buffer.writeUInt32LE(file.samplesPerVector, 12);
  buffer.writeUInt32LE(file.bytesPerSample, 16);
  let offset = GOLDEN_HEADER_BYTES;
  for (const row of file.vectors) {
    for (const word of row) {
      buffer.writeInt32LE(word | 0, offset);
      offset += 4;
    }
  }
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(filePath, buffer);
}

function unpackSampleBytes(words: readonly number[], bytesPerSample: number): number[] {
  const bytes: number[] = [];
  for (let byteIndex = 0; byteIndex < bytesPerSample; byteIndex += 1) {
    const word = words[Math.floor(byteIndex / 4)] >>> 0;
    bytes.push((word >>> (8 * (byteIndex % 4))) & 0xff);
  }
  return bytes;
}

function packSampleBytes(bytes: readonly number[], bytesPerSample: number): number[] {
  const padded = [...bytes];
  while (padded.length < bytesPerSample) {
    padded.push(0);
  }
  const words: number[] = [];
  for (let byteStart = 0; byteStart < bytesPerSample; byteStart += 4) {
    let word = 0;
    for (let byteOffset = 0; byteOffset < 4 && byteStart + byteOffset < bytesPerSample; byteOffset += 1) {
      word |= (padded[byteStart + byteOffset] & 0xff) << (8 * byteOffset);
    }
    words.push(word | 0);
  }
  return words;
}

function retileGoldenVectors(source: GoldenVectorFile, targetBytesPerSample: number): GoldenVectorFile {
  if (targetBytesPerSample <= 0) {
    throw new Error(`targetBytesPerSample must be positive, got ${targetBytesPerSample}.`);
  }
  if (source.bytesPerSample === targetBytesPerSample) {
    return source;
  }

  const targetWordsPerSample = wordsPerSampleForBytes(targetBytesPerSample);
  const tilesPerSourceSample = Math.max(1, Math.ceil(source.bytesPerSample / targetBytesPerSample));
  const vectors = source.vectors.map((row) => {
    const out: number[] = [];
    for (let sample = 0; sample < source.samplesPerVector; sample += 1) {
      const wordStart = sample * source.wordsPerSample;
      const sampleWords = row.slice(wordStart, wordStart + source.wordsPerSample);
      const sampleBytes = unpackSampleBytes(sampleWords, source.bytesPerSample);
      for (let byteStart = 0; byteStart < source.bytesPerSample; byteStart += targetBytesPerSample) {
        const tileBytes = sampleBytes.slice(byteStart, byteStart + targetBytesPerSample);
        out.push(...packSampleBytes(tileBytes, targetBytesPerSample));
      }
    }
    return out;
  });

  return {
    numVectors: source.numVectors,
    samplesPerVector: source.samplesPerVector * tilesPerSourceSample,
    bytesPerSample: targetBytesPerSample,
    wordsPerSample: targetWordsPerSample,
    vectors,
  };
}

async function materializeContractGoldenFile(inputPath: string, outputPath: string, targetBusBits: number): Promise<string> {
  const targetBytes = targetBusBits / 8;
  const source = await readGoldenVectorFile(inputPath);
  if (source.bytesPerSample === targetBytes) {
    return normalizePathForCurrentHost(inputPath);
  }
  const transformed = retileGoldenVectors(source, targetBytes);
  await writeGoldenVectorFile(outputPath, transformed);
  return outputPath;
}

// For an add layer, both operands have shape == output_shape (elementwise),
// so per-pixel channel count is the inner channel dim. Use NCHW index 1 when
// the shape has >=2 dims; fall back to product/spatial otherwise. Verified
// against the source goldin (bytesPerSample == OC*2) inside the retile.
function addOutputChannelsFromLayer(layer: LayerIR): number {
  const shape = layer.output_shape;
  if (!Array.isArray(shape) || shape.length === 0) {
    throw new Error(`add layer '${layer.module_id}' has no output_shape; cannot derive channel count.`);
  }
  if (shape.length >= 2) {
    return shape[1]; // NCHW: [N, C, H, W] or [N, C]
  }
  return shape[0]; // 1-D fallback
}

// Add ops carry two operands packed `lhs[0..OC) | rhs[0..OC)` in each source
// sample. Naive retile chops that flat byte stream into target-sized chunks,
// scrambling the layout for tiled-streaming (which requires every beat to
// carry `lhs_tile (CT bytes) | rhs_tile (CT bytes)` interleaved, where
// CT = channel_tile). Tile-pair retile fixes that.
function retileAddTiledGoldenInputs(
  source: GoldenVectorFile,
  channelTile: number,
  outChannels: number,
): GoldenVectorFile {
  if (channelTile <= 0) {
    throw new Error(`channelTile must be positive, got ${channelTile}.`);
  }
  if (outChannels <= 0 || outChannels % channelTile !== 0) {
    throw new Error(
      `outChannels (${outChannels}) must be a positive multiple of channelTile (${channelTile}) for add tile-pair retile.`,
    );
  }
  const expectedSourceBytes = outChannels * 2; // lhs + rhs
  if (source.bytesPerSample !== expectedSourceBytes) {
    throw new Error(
      `add goldin tile-pair retile expects bytesPerSample=${expectedSourceBytes} (lhs+rhs concat), got ${source.bytesPerSample}.`,
    );
  }
  const targetBytesPerSample = channelTile * 2; // lhs_tile + rhs_tile per beat
  const targetWordsPerSample = wordsPerSampleForBytes(targetBytesPerSample);
  const beatsPerSourceSample = outChannels / channelTile;

  const vectors = source.vectors.map((row) => {
    const out: number[] = [];
    for (let sample = 0; sample < source.samplesPerVector; sample += 1) {
      const wordStart = sample * source.wordsPerSample;
      const sampleWords = row.slice(wordStart, wordStart + source.wordsPerSample);
      const sampleBytes = unpackSampleBytes(sampleWords, source.bytesPerSample);
      // sampleBytes layout: [lhs[0..outChannels-1] | rhs[0..outChannels-1]].
      for (let beat = 0; beat < beatsPerSourceSample; beat += 1) {
        const base = beat * channelTile;
        const beatBytes: number[] = [];
        for (let i = 0; i < channelTile; i += 1) {
          beatBytes.push(sampleBytes[base + i]); // lhs_tile byte i
        }
        for (let i = 0; i < channelTile; i += 1) {
          beatBytes.push(sampleBytes[outChannels + base + i]); // rhs_tile byte i
        }
        out.push(...packSampleBytes(beatBytes, targetBytesPerSample));
      }
    }
    return out;
  });

  return {
    numVectors: source.numVectors,
    samplesPerVector: source.samplesPerVector * beatsPerSourceSample,
    bytesPerSample: targetBytesPerSample,
    wordsPerSample: targetWordsPerSample,
    vectors,
  };
}

async function materializeAddTiledGoldenInputs(
  inputPath: string,
  outputPath: string,
  targetBusBits: number,
  channelTile: number,
  outChannels: number,
): Promise<string> {
  const targetBytes = targetBusBits / 8;
  const source = await readGoldenVectorFile(inputPath);
  if (source.bytesPerSample === targetBytes) {
    return normalizePathForCurrentHost(inputPath);
  }
  const transformed = retileAddTiledGoldenInputs(source, channelTile, outChannels);
  if (transformed.bytesPerSample !== targetBytes) {
    throw new Error(
      `add tile-pair retile produced bytesPerSample=${transformed.bytesPerSample}, expected ${targetBytes} (channel_tile=${channelTile}).`,
    );
  }
  await writeGoldenVectorFile(outputPath, transformed);
  return outputPath;
}

async function materializeContractGoldens(layer: LayerIR): Promise<{
  goldenInputsPath: string;
  goldenOutputsPath: string;
}> {
  const key = sanitizePathPart(`${layer.module_id}_${contractStateKeyForLayer(layer)}`);
  const dir = path.join(resolvePipelineConfigPath(PIPELINE_CONFIG.output_dir), "goldens", "contracts", key);
  const channelTile = layer.channel_tile;
  const isTiledAdd =
    layer.op_type === "add" &&
    currentContractId(layer) !== "flat-bus" &&
    typeof channelTile === "number" &&
    channelTile > 0;
  const goldenInputsPath = isTiledAdd
    ? await materializeAddTiledGoldenInputs(
        layer.golden_inputs_path,
        path.join(dir, `${layer.module_id}.goldin`),
        layer.input_width_bits,
        channelTile as number,
        addOutputChannelsFromLayer(layer),
      )
    : await materializeContractGoldenFile(
        layer.golden_inputs_path,
        path.join(dir, `${layer.module_id}.goldin`),
        layer.input_width_bits,
      );
  const goldenOutputsPath = await materializeContractGoldenFile(
    layer.golden_outputs_path,
    path.join(dir, `${layer.module_id}.goldout`),
    layer.output_width_bits,
  );
  return { goldenInputsPath, goldenOutputsPath };
}

export async function readJsonFile<T>(
  filePath: string,
  schema?: z.ZodType<T>,
): Promise<T> {
  const raw = await readFile(filePath, "utf8");
  const parsed: unknown = JSON.parse(raw);

  if (schema) {
    const result = schema.safeParse(parsed);
    if (!result.success) {
      throw new Error(
        `Invalid JSON at '${filePath}':\n${JSON.stringify(result.error.issues, null, 2)}`,
      );
    }
    return result.data;
  }

  return parsed as T;
}

export async function writeJsonFile(filePath: string, value: unknown): Promise<void> {
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function recordUsageFromResult(
  manager: PipelineStateManager,
  result: SDKResultMessage,
): void {
  manager.recordAgentUsage(
    result.total_cost_usd,
    result.modelUsage as Record<string, ModelUsageEntry>,
  );
}

/**
 * Per-tool-call audit record. One entry per tool_use block observed in the
 * agent's message stream, plus one entry per matching tool_result. Appended
 * to `output/reports/agent_tool_use.jsonl` for post-hoc inspection: what
 * files Foundry/Surgeon actually read, what Bash commands they ran, what
 * came back. Independent of agent obedience — we read the SDK's own
 * message stream, which the agent cannot suppress.
 */
export type ToolUseAuditEntry = {
  timestamp: string;
  agent: string;
  module_id: string | null;
  turn_index: number;
  kind: "tool_use" | "tool_result";
  tool_use_id: string | null;
  tool_name: string | null;
  // tool_use fields
  input: unknown;
  // tool_result fields (truncated to keep logs manageable)
  is_error: boolean | null;
  output_preview: string | null;
  output_length: number | null;
};

const TOOL_RESULT_PREVIEW_BYTES = 2000;

function isRecordLike(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null;
}

/**
 * Walk the SDK message stream and extract tool-use + tool-result blocks.
 * Tolerant of shape variance: the SDK's message types are loose records, so
 * we defensively guard each field lookup.
 */
export function extractToolUseAudits(
  messages: SDKMessage[],
  meta: { agent: string; module_id?: string | null; nowIso: string },
): ToolUseAuditEntry[] {
  const audits: ToolUseAuditEntry[] = [];
  let turnIndex = 0;

  for (const msg of messages) {
    if (!isRecordLike(msg)) continue;
    const msgType = typeof msg.type === "string" ? msg.type : "";
    // SDKMessage is a loose union; cast through Record<string, unknown> to
    // reach optional fields like `message` that only some variants carry.
    const loose = msg as Record<string, unknown>;

    // Assistant messages may carry tool_use blocks in their content array.
    if (msgType === "assistant") {
      const inner = isRecordLike(loose.message) ? loose.message : {};
      const content = Array.isArray(inner.content) ? inner.content : [];
      for (const block of content) {
        if (!isRecordLike(block)) continue;
        if (block.type === "tool_use") {
          audits.push({
            timestamp: meta.nowIso,
            agent: meta.agent,
            module_id: meta.module_id ?? null,
            turn_index: turnIndex,
            kind: "tool_use",
            tool_use_id: typeof block.id === "string" ? block.id : null,
            tool_name: typeof block.name === "string" ? block.name : null,
            input: block.input ?? null,
            is_error: null,
            output_preview: null,
            output_length: null,
          });
        }
      }
      turnIndex += 1;
      continue;
    }

    // User messages may carry tool_result blocks routed back to the agent.
    if (msgType === "user") {
      const inner = isRecordLike(loose.message) ? loose.message : {};
      const content = Array.isArray(inner.content) ? inner.content : [];
      for (const block of content) {
        if (!isRecordLike(block)) continue;
        if (block.type === "tool_result") {
          const rawContent = block.content;
          const rawStr = typeof rawContent === "string"
            ? rawContent
            : JSON.stringify(rawContent ?? "");
          const isError =
            typeof block.is_error === "boolean" ? block.is_error : null;
          audits.push({
            timestamp: meta.nowIso,
            agent: meta.agent,
            module_id: meta.module_id ?? null,
            turn_index: turnIndex,
            kind: "tool_result",
            tool_use_id:
              typeof block.tool_use_id === "string" ? block.tool_use_id : null,
            tool_name: null,
            input: null,
            is_error: isError,
            output_preview: rawStr.slice(0, TOOL_RESULT_PREVIEW_BYTES),
            output_length: rawStr.length,
          });
        }
      }
    }
  }

  return audits;
}

/** Append per-tool-call audit entries to `output/reports/agent_tool_use.jsonl`. */
export async function appendToolUseAudits(
  audits: ToolUseAuditEntry[],
): Promise<void> {
  if (audits.length === 0) return;
  const logPath = reportPath("agent_tool_use.jsonl");
  await mkdir(path.dirname(logPath), { recursive: true });
  const body = audits.map((a) => JSON.stringify(a)).join("\n") + "\n";
  await appendFile(logPath, body, "utf8");
}

/**
 * Emit a compact summary of what tools the agent actually used, to the
 * main run_log.jsonl. Makes post-hoc inspection fast: one line per agent
 * dispatch telling you `{tool_call_count, tools_called: [...], bytes_read}`.
 */
export function summarizeToolUse(
  audits: ToolUseAuditEntry[],
): Record<string, unknown> {
  const toolUseEntries = audits.filter((a) => a.kind === "tool_use");
  const toolsCalled = toolUseEntries
    .map((a) => a.tool_name)
    .filter((n): n is string => typeof n === "string" && n.length > 0);
  const toolsCounts: Record<string, number> = {};
  for (const name of toolsCalled) {
    toolsCounts[name] = (toolsCounts[name] ?? 0) + 1;
  }
  const totalResultBytes = audits
    .filter((a) => a.kind === "tool_result")
    .reduce((sum, a) => sum + (a.output_length ?? 0), 0);
  const errorCount = audits
    .filter((a) => a.kind === "tool_result" && a.is_error === true).length;
  return {
    tool_call_count: toolUseEntries.length,
    tools_called: toolsCalled,
    tools_counts: toolsCounts,
    total_result_bytes: totalResultBytes,
    tool_error_count: errorCount,
  };
}

const NN2RTL_MCP_TOOL_PREFIX = "mcp__nn2rtl-tools__";
const MCP_TOOL_PREFIX = "mcp__";

export function isForeignMcpToolName(name: string): boolean {
  return name.startsWith(MCP_TOOL_PREFIX) && !name.startsWith(NN2RTL_MCP_TOOL_PREFIX);
}

export function foreignMcpToolNames(audits: ToolUseAuditEntry[]): string[] {
  return [
    ...new Set(
      audits
        .filter((a) => a.kind === "tool_use" && typeof a.tool_name === "string")
        .map((a) => a.tool_name as string)
        .filter(isForeignMcpToolName),
    ),
  ];
}

export async function appendForeignMcpToolWarnings(
  audits: ToolUseAuditEntry[],
  runtime: OrchestratorRuntime = createOrchestratorRuntime(),
): Promise<void> {
  const tools = foreignMcpToolNames(audits);
  if (tools.length === 0) return;
  const first = audits.find((a) => a.kind === "tool_use" && a.tool_name && tools.includes(a.tool_name));
  await appendRunLog(
    {
      event: "foreign_mcp_tool_used",
      agent: first?.agent ?? null,
      module_id: first?.module_id ?? null,
      tools_called: tools,
      tool_call_count: audits.filter((a) => a.kind === "tool_use" && a.tool_name && tools.includes(a.tool_name)).length,
      policy: "warning_only",
      note: "A non-nn2rtl MCP connector was visible to the agent. Persistence and verification gates still decide whether the attempt is usable.",
    },
    runtime,
  );
}

export async function appendRunLog(
  entry: Record<string, unknown>,
  runtime: OrchestratorRuntime = createOrchestratorRuntime(),
): Promise<void> {
  const logPath = reportPath("run_log.jsonl");
  await mkdir(path.dirname(logPath), { recursive: true });
  await appendFile(
    logPath,
    `${JSON.stringify({ timestamp: runtime.now().toISOString(), ...entry })}\n`,
    "utf8",
  );
}

export async function ensureOutputLayout(): Promise<void> {
  await Promise.all([
    mkdir(resolvePipelineConfigPath(PIPELINE_CONFIG.output_dir), { recursive: true }),
    mkdir(resolvePipelineConfigPath(PIPELINE_CONFIG.rtl_dir), { recursive: true }),
    mkdir(resolvePipelineConfigPath(PIPELINE_CONFIG.tb_dir), { recursive: true }),
    mkdir(resolvePipelineConfigPath(PIPELINE_CONFIG.weights_dir), { recursive: true }),
    mkdir(resolvePipelineConfigPath(PIPELINE_CONFIG.reports_dir), { recursive: true }),
  ]);
}

function buildSidecarPath(moduleId: string): string {
  return path.join(resolvePipelineConfigPath(PIPELINE_CONFIG.tb_dir), `${moduleId}.sidecar.json`);
}

export async function loadPluginAgentDefinition(slug: AgentSlug): Promise<AgentDefinition> {
  const agentName = normalizeAgentName(slug);
  const markdownPath = path.join(pluginPath, "agents", `${slug}.md`);
  const markdown = await readText(markdownPath);
  const { frontmatter, body } = parseFrontmatter(markdown);
  const skillMarkdownPath = path.join(pluginPath, "skills", slug, "SKILL.md");
  const skillMarkdown = (await pathExists(skillMarkdownPath))
    ? await readText(skillMarkdownPath)
    : "";
  const parsedSkill = skillMarkdown ? parseFrontmatter(skillMarkdown) : null;

  const builtInTools = toStringList(frontmatter.tools) ?? [];
  const disallowedTools = toStringList(frontmatter.disallowedTools);
  const mcpTools = [...AGENT_MCP_TOOLS[slug]];
  const combinedTools = [...new Set([...builtInTools, ...mcpTools])];
  const skills = toStringList(frontmatter.skills);
  const prompt = parsedSkill && APPEND_SKILL_TO_PROMPT[slug]
    ? `${body}\n\nSupplemental skill reference:\n\n${parsedSkill.body}`
    : body;

  const effortRaw = typeof frontmatter.effort === "string" ? frontmatter.effort : undefined;
  const effort: EffortLevel | undefined =
    effortRaw === "low" || effortRaw === "medium" || effortRaw === "high" ||
    effortRaw === "xhigh" || effortRaw === "max"
      ? effortRaw
      : undefined;

  return {
    description: AGENT_CONFIG[agentName].description,
    prompt,
    tools: combinedTools.length > 0 ? combinedTools : undefined,
    disallowedTools,
    model: AGENT_CONFIG[agentName].model,
    maxTurns: AGENT_CONFIG[agentName].maxTurns,
    skills,
    effort,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function formatIntVector(values: readonly number[] | undefined): string {
  return values && values.length > 0 ? `[${values.join(", ")}]` : "unknown";
}

function buildFoundryGenerationBrief(payload: unknown): string | null {
  if (!isRecord(payload) || !isRecord(payload.layer_ir)) {
    return null;
  }
  const layer = payload.layer_ir as unknown as LayerIR;
  const expectedSpecHash =
    typeof payload.expected_spec_hash === "string"
      ? payload.expected_spec_hash
      : computeExpectedSpecHash(layer);
  const effectiveLatency = expectedLatencyCyclesForContract(layer, contractSidecarFields(layer));
  const lines = [
    "Compact generation brief:",
    `- op_type=${layer.op_type}; module_id=${layer.module_id}; return spec_hash=${expectedSpecHash} exactly.`,
    `- bus contract: data_in=${layer.input_width_bits} bits, data_out=${layer.output_width_bits} bits.`,
    `- base pipeline_latency_cycles=${layer.pipeline_latency_cycles}; Assayer expects valid_out after ${effectiveLatency} cycle(s) for the selected contract.`,
  ];
  const contractSelection = contractSelectionForLayer(layer);
  lines.push(
    `- selected contract: ${contractSelection.selected.name} (rank ${contractSelection.selected.complexity_rank}).`,
    `- contract template: ${contractTestbenchTemplatePath(contractSelection.selected.name)}.`,
    `- contract constraints: max_bus_width_bits=${contractSelection.selected.fit_constraints.max_bus_width_bits}; dependencies=${contractSelection.selected.dependencies.join(", ") || "none"}.`,
    "- use `preloaded_rtl_patterns` from the payload as the authoritative local knowledge context; it is already filtered by selected contract.",
  );
  if (currentContractId(layer) !== "flat-bus") {
    lines.push(
      `- contract variant: ${currentContractId(layer)}; io_mode=${layer.io_mode}; channel_tile=${layer.channel_tile ?? "n/a"}.`,
      "- contract switching rule: implement the requested interface variant exactly; do not fall back to the full flat-bus contract.",
    );
  }

  if (layer.op_type === "conv2d" && layer.weight_shape.length >= 4) {
    const kh = layer.weight_shape[2];
    const kw = layer.weight_shape[3];
    const pointwise = kh === 1 && kw === 1;
    const padding = layer.padding;
    lines.push(
      `- conv geometry: kernel=${kh}x${kw}; pointwise=${pointwise ? "yes" : "no"}; stride=${formatIntVector(layer.stride)}; padding=${formatIntVector(padding)}; dilation=${formatIntVector(layer.dilation)}; groups=${layer.groups ?? 1}.`,
    );
    if ((layer.groups ?? 1) !== 1 || (layer.dilation && layer.dilation.some((value) => value !== 1))) {
      lines.push(
        "- conv variant warning: grouped/depthwise/dilated conv semantics are explicit in LayerIR. Do not silently treat them as ordinary dense dilation=1 convolution.",
      );
    }
    if (!pointwise) {
      const needsDrain =
        padding !== undefined &&
        ((padding[0] ?? 0) > 0 || (padding[1] ?? 0) > 0);
      lines.push(
        `- spatial conv rule: use a real line buffer + sliding window; ${needsDrain ? "include a padding drain path." : "no padding drain is required unless the LayerIR says otherwise."}`,
      );
    }
    if (layer.module_id === "layer0_0_conv1") {
      lines.push(
        "- stem rule: do not fuse extra stages from stale docs. Follow the current LayerIR/golden contract; on the legacy .pth path this is not a fused MaxPool stage.",
      );
    }
  } else if (layer.op_type === "add") {
    lines.push(
      `- add rule: data_in is packed lhs|rhs where W=${layer.input_width_bits / 2}; use lhs_scale_factor, rhs_scale_factor, and scale_factor exactly.`,
    );
  } else if (layer.op_type === "maxpool") {
    lines.push(
      `- maxpool geometry: kernel=${formatIntVector(layer.kernel_size)}; stride=${formatIntVector(layer.pool_stride)}; padding=${formatIntVector(layer.pool_padding)}.`,
    );
  }

  lines.push(
    "- invariant markers: only ROUNDING, READY_IN_GATING, and VALID_OUT_LATENCY may use [INVARIANT:*] comments. Do not mark drain, reset, counter, or memory lines invariant.",
  );
  if (Array.isArray(payload.failure_memory) && payload.failure_memory.length > 0) {
    lines.push(
      `- failure memory: ${payload.failure_memory.length} scored failed RTL attempt(s) are attached in payload.failure_memory. Each entry has rtl_path/failure_path; read the RTL only if it helps avoid repeating a known-bad structure.`,
    );
  }
  return lines.join("\n");
}

function buildSurgeonRepairBrief(payload: unknown): string | null {
  if (!isRecord(payload) || !isRecord(payload.layer_ir) || !isRecord(payload.verif_result)) {
    return null;
  }
  const layer = payload.layer_ir as unknown as LayerIR;
  const verif = payload.verif_result as unknown as VerifResult;
  const retrySeed = typeof payload.retry_seed === "string" ? payload.retry_seed : null;
  const effectiveLatency = expectedLatencyCyclesForContract(layer, contractSidecarFields(layer));

  // Distinguish the sim-passed / synth-only failure from a full functional
  // failure. When sim passed, the datapath is already correct by evidence —
  // Surgeon must NOT rewrite the numerical logic, only the constructs that
  // upset Vivado (wide unrolled blocks, non-synthesizable $signed patterns,
  // deep combinational cones, etc.). Framing this narrowly prevents Surgeon
  // from regressing sim in the process of "fixing" synth.
  const isSynthOnlyFailure =
    verif.status_class === "sim_passed" && verif.failure_class === "synthesis_failed";

  const lines = [
    "Compact repair brief:",
    `- op_type=${layer.op_type}; bus contract=data_in ${layer.input_width_bits} bits, data_out ${layer.output_width_bits} bits; preserve the public interface exactly.`,
    `- return spec_hash=${
      typeof payload.expected_spec_hash === "string"
        ? payload.expected_spec_hash
        : computeExpectedSpecHash(layer)
    } exactly; do not copy a stale broken_module.spec_hash after a contract or geometry change.`,
    `- authoritative latency contract: base pipeline_latency_cycles=${layer.pipeline_latency_cycles}; Assayer expected latency=${effectiveLatency}.`,
    `- selected contract: ${resolveLayerContractId(layer)}. Preserve every metadata-declared interface signal for that contract.`,
    "- use `preloaded_rtl_patterns` from the payload as the authoritative local knowledge context; it is already filtered by selected contract.",
    `- current failure: status=${verif.status}; status_class=${verif.status_class ?? "n/a"}; failure_class=${verif.failure_class ?? "n/a"}.`,
    "- compiler-first rule: if status=syntax_error or compiler stderr is populated, read iverilog/verilator stderr before touching datapath logic.",
    "- setup-failure rule: if evidence points only to static_verilator_tb.cpp, sidecar JSON, or toolchain glue, do not rewrite the RTL datapath in response to it.",
  ];

  // Post-Retrospector final attempt. The advice block names the failure
  // mode and the repair scope; the broken_module is already the
  // best-known artifact picked by the orchestrator (or the latest if
  // Retrospector overrode that choice). Surgeon should treat the scope
  // verdict as the upper bound of edits — a `targeted_fsm_or_datapath_fix`
  // verdict is permission for narrow surgery, not a green light to rewrite
  // the architecture.
  if (isRecord(payload.retrospector_advice)) {
    const advice = payload.retrospector_advice as unknown as RetrospectorAdvice;
    lines.push(
      `- post-retrospector final attempt: scope=${advice.repair_scope ?? "unspecified"}; the orchestrator chose Surgeon (not Foundry) because the prior artifact is salvageable. Do NOT rewrite the architecture; perform the smallest possible repair consistent with retrospector_advice.suggestion.`,
    );
    if (advice.base_artifact === "best_known") {
      lines.push(
        "- broken_module is the highest-scoring artifact across all prior attempts (Foundry + Surgeon), not necessarily the most recent one. A later attempt may have regressed; this picker rolled it back automatically.",
      );
    }
  }
  if (retrySeed) {
    lines.push(`- retry seed: ${retrySeed}. Use this as a fresh-attempt discriminator; do not repeat a prior unsuccessful patch shape.`);
  }
  if (Array.isArray(payload.prior_foundry_attempts) && payload.prior_foundry_attempts.length > 0) {
    lines.push(
      `- foundry tried this contract ${payload.prior_foundry_attempts.length} time(s) before you. Their RTL and the verifier's verdict on each are in payload.prior_foundry_attempts. Read them — every entry tells you a fix shape that ALREADY didn't work. Pick a different lever.`,
    );
  }
  if (Array.isArray(payload.failure_memory) && payload.failure_memory.length > 0) {
    lines.push(
      `- failure memory: ${payload.failure_memory.length} scored failed RTL attempt(s) are attached in payload.failure_memory. Each entry has rtl_path/failure_path; inspect only the relevant source paths before repeating a prior failed structure.`,
    );
  }
  if (isRecord(payload.reference_evidence)) {
    lines.push(
      "- reference evidence: payload.reference_evidence was computed deterministically for the first mismatch via compute_layer_reference. Compare its expected INT8 output and integer-domain intermediates with the observed got value before changing rounding, channel indexing, or pixel traversal.",
    );
  }
  if (currentContractId(layer) !== "flat-bus") {
    lines.push(
      `- contract variant: ${currentContractId(layer)}; io_mode=${layer.io_mode}; channel_tile=${layer.channel_tile ?? "n/a"}. Preserve this interface variant exactly.`,
    );
  }
  if (layer.op_type === "conv2d") {
    lines.push(
      `- conv semantics: stride=${formatIntVector(layer.stride)}; padding=${formatIntVector(layer.padding)}; dilation=${formatIntVector(layer.dilation)}; groups=${layer.groups ?? 1}. Preserve these exactly.`,
    );
  }

  if (verif.failure_class === "verilator_timeout") {
    lines.push(
      "- VERILATOR TIMEOUT: the DUT compiled but the simulation never terminated." +
      " Do NOT assume the RTL is partially correct — a timeout means the FSM is structurally" +
      " wrong enough that it can never reach the end of the output stream." +
      " The TB's hang_budget only catches TOTAL silence on valid_out, so intermittent" +
      " firings keep the sim alive indefinitely. Check output-counter bounds, drain-exit" +
      " conditions, and any state that re-enters a wait on a signal that cannot arrive." +
      " Fix the FSM control flow — do not rewrite the datapath.",
    );
  } else if (verif.failure_class === "structural_preflight_failed") {
    lines.push(
      "- STRUCTURAL PREFLIGHT FAILURE: the RTL parsed but violated a structural rule" +
      " before simulation. The fix_hint names the exact rule (e.g. line_buffer_missing," +
      " window_not_registered, weights_packed_forbidden, readmemh_missing," +
      " procedural_declaration_forbidden, output_counter_missing, coord_scheduler_missing)." +
      " Repair the indicted construct and do not touch unrelated logic.",
    );
  } else if (isSynthOnlyFailure) {
    lines.push(
      "- SYNTHESIS-ONLY FAILURE: simulation passed with correct outputs and exact timing." +
      " The datapath is proven correct — DO NOT rewrite numerical logic, MAC ordering," +
      " requantisation, ready/valid handshaking, or state transitions." +
      " Your ONLY job is to make the existing logic synthesizable." +
      " Typical synth-hostile patterns to target: deep combinational cones that Vivado can't meet timing on," +
      " unsynthesizable constructs (non-constant array indices into large regs, dynamic $signed," +
      " latch inference from incomplete case statements), or register/wire width issues." +
      " Read the Vivado error output below carefully and make the minimum change that addresses it.",
    );
  } else {
    // Any non-synth-only failure means the module has not yet been proven to
    // simulate correctly. No line is invariant regardless of [INVARIANT:*]
    // markers — markers are placed by Foundry before verification and may sit
    // on the exact buggy lines.
    lines.push(
      "- invariant scope: this module has NOT yet passed functional verification." +
      " [INVARIANT:*] markers were placed speculatively by Foundry and may cover the bug." +
      " Treat every line as mutable — no marker confers protection until sim+synth both pass.",
    );
  }

  // Flag when verif.expected/got have been windowed (synth-only drops them
  // entirely; sim-failure trims them to a window around first_mismatch_index)
  // so Surgeon doesn't try to scan the arrays for a "late" mismatch it won't
  // find. outputs_expected is authoritative for the total vector length.
  const expLen = Array.isArray(verif.expected) ? verif.expected.length : 0;
  const totalLen = verif.outputs_expected ?? 0;
  if (isSynthOnlyFailure && expLen === 0 && totalLen > 0) {
    lines.push(
      `- verif arrays: expected/got were dropped (sim passed, full ${totalLen}-sample vectors are not diagnostic for a synth-only failure).`,
    );
  } else if (expLen > 0 && totalLen > expLen) {
    lines.push(
      `- verif arrays: expected/got are a ±${SURGEON_MISMATCH_WINDOW}-sample window around first_mismatch_index=${verif.first_mismatch_index} (${expLen} of ${totalLen} total samples shown).`,
    );
  }

  // Per-vector breakdown: a single line that names which goldin vectors
  // matched 100% vs partial. A "v0=100%, v1=98%, v2-7=98%" pattern is the
  // signature of a multi-vector pipeline desync (active-pixel-counter or
  // FSM reset bug). v0 alone failing usually means the cold-start path is
  // wrong; vN alone failing means a per-frame reset issue.
  if (Array.isArray(verif.per_vector) && verif.per_vector.length > 0) {
    const summary = verif.per_vector
      .map((pv) => {
        const total = pv.exact_match_count + pv.mismatch_count;
        if (total === 0) return `v${pv.vector_idx}=∅`;
        const pct = ((pv.exact_match_count / total) * 100).toFixed(1);
        return `v${pv.vector_idx}=${pct}%(max_err=${pv.max_error})`;
      })
      .join(", ");
    lines.push(`- per-vector breakdown: ${summary}.`);
  }

  // Verilator stdout: surface only the existence and size, not the body.
  // Surgeon can read payload.verif_result.verilator_stdout directly when
  // they have planted $display probes; the prompt brief stays compact.
  if (typeof verif.verilator_stdout === "string" && verif.verilator_stdout.length > 0) {
    lines.push(
      `- verilator_stdout: ${verif.verilator_stdout.length} bytes captured in payload.verif_result.verilator_stdout (read it only if you embedded $display/$write probes; otherwise it's just simulator banners).`,
    );
  }

  if (layer.op_type === "conv2d" && layer.weight_shape.length >= 4) {
    const kh = layer.weight_shape[2];
    const kw = layer.weight_shape[3];
    const pointwise = kh === 1 && kw === 1;
    const padding = layer.padding;
    lines.push(
      `- conv geometry: kernel=${kh}x${kw}; pointwise=${pointwise ? "yes" : "no"}; stride=${formatIntVector(layer.stride)}; padding=${formatIntVector(padding)}.`,
    );
    if (!pointwise) {
      const needsDrain =
        padding !== undefined &&
        ((padding[0] ?? 0) > 0 || (padding[1] ?? 0) > 0);
      lines.push(
        `- spatial conv repair rule: ${needsDrain ? "expect a drain path near the tail; fix the existing one before inventing a new one." : "no padding drain should be needed unless the RTL explicitly contains one."}`,
      );
    }
    lines.push("- conv architecture rule: single-MAC rewrites are forbidden; preserve the output-stationary OC-lane design.");
  } else if (layer.op_type === "add") {
    lines.push(
      `- add rule: lhs/rhs are packed into one input bus; preserve the packed interface and the scale-aware requantisation path.`,
    );
  } else if (layer.op_type === "maxpool") {
    lines.push(
      `- maxpool geometry: kernel=${formatIntVector(layer.kernel_size)}; stride=${formatIntVector(layer.pool_stride)}; padding=${formatIntVector(layer.pool_padding)}.`,
    );
  }

  return lines.join("\n");
}

export type ModuleContractSummary = {
  interface: {
    clock: "clk";
    reset: "rst_n";
    valid_in: "valid_in";
    ready_in: "ready_in";
    valid_out: "valid_out";
    data_in_bits: number;
    data_out_bits: number;
  };
  timing: {
    pipeline_latency_cycles: number;
    clock_period_ns: number;
    fmax_target_mhz: number;
  };
  capability_limits: {
    max_supported_bus_bits: number;
    target_part: "xczu9eg-ffvb1156-2-e";
    // XCZU9EG / ZCU102 fabric numbers from AMD Zynq UltraScale+ MPSoC
    // Product Selection Guide (XMP104 v2.8), ZU9 column. The ZU9 device
    // does NOT include UltraRAM — it is unique to ZU3T/4/5/7 in the CG/EG
    // families and to selected EV variants — so no `uram` field here.
    zcu102_capacity: {
      lut: 274080;
      ff: 548160;
      dsp: 2520;
      bram18: 1824;
    };
  };
  operation: Record<string, unknown>;
};

export type FailureClassifierInput = {
  module_spec: LayerIR;
  contract: ModuleContractSummary;
  failure_result: VerifResult;
  logs: Record<string, unknown>;
  module?: Pick<VerilogModule, "module_id" | "spec_hash" | "generated_by" | "attempt">;
};

export type RtlKnowledgeDoc = {
  pattern_markdown: string;
  reference_verilog: string | null;
  license_notice: string | null;
};

export type FoundryVersionRecord = {
  version_index: number;
  module: VerilogModule;
  session_id: string | null;
  tool_use_summary: Record<string, unknown>;
  documents_used: Array<{
    tool_name: string | null;
    input: unknown;
  }>;
};

export type FailureAttemptRecord = {
  attempt_index: number;
  stage: string;
  module: Pick<VerilogModule, "module_id" | "spec_hash" | "generated_by" | "attempt"> | null;
  result: VerifResult;
  logs: Record<string, unknown>;
};

export type RetrospectorInput = {
  original_spec: LayerIR;
  contract: ModuleContractSummary;
  current_contract: ContractPlan;
  available_contracts: ContractPlan[];
  doc_used: RtlKnowledgeDoc;
  knowledge_docs_used: KnowledgeDocRecord[];
  foundry_versions: FoundryVersionRecord[];
  failure_attempts: FailureAttemptRecord[];
  failure_corpus: FailureCorpusIndexEntry[];
};

type DocTier = "protected" | "active" | "probationary" | "archive";
type GeneratedDocStatus = "probationary" | "active" | "archived";
type KnowledgeDocRecord = {
  id: string;
  tier: DocTier;
  kind: "pattern" | "reference";
  op_type: LayerIR["op_type"];
  contract_id?: ContractId;
  path: string;
  relative_path: string;
  match_level?: SignatureMatchLevel;
};
type DocLifecycleEntry = {
  id: string;
  op_type: LayerIR["op_type"];
  contract_id?: ContractId;
  contract_key?: string;
  spec_hash: string;
  signature_hashes?: string[];
  exact_reference_keys?: string[];
  derived_from_networks?: string[];
  derived_from_modules?: string[];
  applicability?: Record<string, unknown>;
  contraindications?: Array<string | Record<string, unknown>>;
  status: GeneratedDocStatus;
  pattern_path: string;
  reference_path: string;
  created_by_module: string;
  created_by_agent: VerilogModule["generated_by"];
  created_at: string;
  creation_reason?: string;
  source_doc_ids?: string[];
  replacement_for?: string[];
  used_by_modules: string[];
  successful_modules: string[];
  failed_modules: string[];
  promoted_at?: string;
  archived_at?: string;
  archive_reason?: string;
  archived_pattern_path?: string;
  archived_reference_path?: string;
};
type DocLifecycleState = {
  version: 1;
  docs: Record<string, DocLifecycleEntry>;
};

export type ContractPlan = {
  id: ContractId;
  complexity: number;
  description: string;
};
type ContractFlag = {
  key: string;
  contract_id: ContractId;
  spec_hash: string;
  op_type: LayerIR["op_type"];
  status: "manual_correction_needed";
  flagged_at: string;
  updated_at: string;
  module_ids: string[];
  reason: string;
  report_path?: string;
  last_failure_result?: VerifResult;
};
type ContractResponseState = {
  version: 1;
  contracts: Record<string, ContractFlag>;
};

export const CONTRACT_PLANS: ContractPlan[] = [
  {
    id: "flat-bus",
    complexity: 0,
    description: "Full packed activation bus, current default contract.",
  },
  {
    id: "tiled-streaming",
    complexity: 1,
    description: "Channel-tiled stream contract that keeps each activation stream within the configured bus cap.",
  },
  {
    id: "dram-backed-weights",
    complexity: 2,
    description: "External-memory-backed weights contract for layers that cannot fit the simpler streaming contracts.",
  },
  {
    id: "activation-double-buffering",
    complexity: 3,
    description: "Ping-pong activation buffers for overlapping activation load and compute.",
  },
  {
    id: "weight-tiling",
    complexity: 4,
    description: "Weight-tiled execution with external weight reads and partial-sum accumulation.",
  },
  {
    id: "depthwise-conv",
    complexity: 1,
    description: "Depthwise convolution contract — groups == in_channels == out_channels, per-channel filters, no cross-channel reduction. Powers MobileNet-style separable conv blocks.",
  },
];

function contractStatePath(): string {
  return resolvePipelineConfigPath(PIPELINE_CONFIG.contract_state_path);
}

function emptyContractResponseState(): ContractResponseState {
  return { version: 1, contracts: {} };
}

async function loadContractResponseState(): Promise<ContractResponseState> {
  try {
    const raw = await readFile(contractStatePath(), "utf8");
    const parsed = JSON.parse(raw) as Partial<ContractResponseState>;
    return {
      version: 1,
      contracts: parsed.contracts && typeof parsed.contracts === "object"
        ? (parsed.contracts as Record<string, ContractFlag>)
        : {},
    };
  } catch {
    return emptyContractResponseState();
  }
}

async function saveContractResponseState(state: ContractResponseState): Promise<void> {
  await writeJsonFile(contractStatePath(), state);
}

function currentContractId(layer: LayerIR): ContractId {
  if (layer.contract_id) return layer.contract_id;
  if (layer.io_mode === "channel_tiled") return "tiled-streaming";
  if (layer.io_mode === "dram_backed_weights") return "dram-backed-weights";
  if (layer.io_mode === "activation_double_buffered") return "activation-double-buffering";
  if (layer.io_mode === "weight_tiled") return "weight-tiling";
  // Depthwise fallback: a conv2d whose groups equal both input and output
  // channel counts is depthwise, even if the LayerIR was emitted without an
  // explicit contract_id (e.g. legacy IR files predating multi-network). Mirror
  // resolveLayerContractId in sdk/contracts.ts so downstream dispatch is
  // consistent whether callers go through that helper or read currentContractId
  // directly.
  if (
    layer.op_type === "conv2d" &&
    typeof layer.groups === "number" &&
    layer.groups > 1
  ) {
    const inCh = layer.input_shape?.[1];
    const outCh = layer.output_shape?.[1];
    if (
      typeof inCh === "number" &&
      typeof outCh === "number" &&
      layer.groups === inCh &&
      layer.groups === outCh
    ) {
      return "depthwise-conv";
    }
  }
  return "flat-bus";
}

function contractStateKeyForLayer(layer: LayerIR): string {
  return `${currentContractId(layer)}:${computeExpectedSpecHash(layer)}`;
}

function signatureMetadataForLayer(
  baseLayer: LayerIR,
  runtimeLayer: LayerIR,
  modelQuantization?: string,
): LayerSignatureMetadata {
  return signatureBundle({
    baseLayer,
    runtimeLayer,
    baseContractId: currentContractId(baseLayer),
    runtimeContractId: currentContractId(runtimeLayer),
    modelQuantization,
  });
}

function withSignatureMetadata(
  baseLayer: LayerIR,
  runtimeLayer: LayerIR,
  modelQuantization?: string,
): LayerIR {
  const metadata = signatureMetadataForLayer(baseLayer, runtimeLayer, modelQuantization);
  return {
    ...runtimeLayer,
    quantization_family: metadata.runtime_layer_signature.quantization_family,
    ...metadata,
  };
}

function fullInputBusWidthBits(layer: LayerIR): number {
  const inputChannels = getShapeChannels(layer.input_shape, "input_shape", layer.module_id);
  return layer.op_type === "add" ? inputChannels * 16 : inputChannels * 8;
}

function fullOutputBusWidthBits(layer: LayerIR): number {
  const outputChannels = getShapeChannels(layer.output_shape, "output_shape", layer.module_id);
  return outputChannels * 8;
}

/**
 * Pick a `channel_tile` for the chosen tiled-style contract such that the
 * resulting per-beat bus width fits BOTH the global pipeline cap
 * (`PIPELINE_CONFIG.MAX_SUPPORTED_BUS_BITS`) AND the target contract's own
 * `fit_constraints.max_bus_width_bits` from its metadata. Without this the
 * orchestrator could pick a tile that overshoots a contract's own spec
 * (e.g. tiled-streaming caps at 256 bits/beat, so a 512-channel tile at
 * 4096 bits would be rejected by tiled-streaming's own bus-width gate
 * before Foundry ever ran).
 *
 * For `op_type=add` data_in is the packed `lhs|rhs` pair (16 bits per
 * channel-pair on the input side), so the cap is divided by 16 instead of
 * by 8 for that op.
 */
function chooseContractChannelTile(layer: LayerIR, plan: ContractPlan): number {
  const inputChannels = getShapeChannels(layer.input_shape, "input_shape", layer.module_id);
  const outputChannels = getShapeChannels(layer.output_shape, "output_shape", layer.module_id);
  const maxChannels = Math.max(inputChannels, outputChannels);
  const metadata = loadContractMetadata(plan.id);
  const contractCapBits = metadata.fit_constraints.max_bus_width_bits;
  const globalCapBits = PIPELINE_CONFIG.MAX_SUPPORTED_BUS_BITS;
  const defaultBeatBits = metadata.fit_constraints.default_beat_width_bits;
  const capBits = Math.max(
    1,
    Math.min(contractCapBits, globalCapBits, defaultBeatBits ?? Number.POSITIVE_INFINITY),
  );
  // tiledInputBusWidthBits packs 2 bytes per add-operand-pair channel and
  // 1 byte per channel for every other op. Output bus is always 1 byte per
  // channel (single int8 stream), so input is the binding side here.
  const bytesPerInputChannel = layer.op_type === "add" ? 2 : 1;
  const capChannels = Math.max(1, Math.floor(capBits / (bytesPerInputChannel * 8)));
  return Math.max(1, Math.min(maxChannels, capChannels));
}

function tiledInputBusWidthBits(layer: LayerIR, channelTile: number): number {
  return layer.op_type === "add" ? channelTile * 16 : channelTile * 8;
}

function tiledOutputBusWidthBits(channelTile: number): number {
  return channelTile * 8;
}

function ioModeForContract(contractId: ContractId): LayerIR["io_mode"] {
  switch (contractId) {
    case "flat-bus":
      return "packed_full";
    case "tiled-streaming":
      return "channel_tiled";
    case "dram-backed-weights":
      return "dram_backed_weights";
    case "activation-double-buffering":
      return "activation_double_buffered";
    case "weight-tiling":
      return "weight_tiled";
  }
}

export function applyContractPlan(baseLayer: LayerIR, plan: ContractPlan): LayerIR {
  const layer = jsonClone(baseLayer);
  layer.contract_id = plan.id;
  layer.io_mode = ioModeForContract(plan.id);
  if (plan.id === "flat-bus") {
    delete layer.channel_tile;
    delete layer.contract_params;
    layer.input_width_bits = fullInputBusWidthBits(layer);
    layer.output_width_bits = fullOutputBusWidthBits(layer);
    return layer;
  }

  const channelTile = chooseContractChannelTile(baseLayer, plan);
  const metadata = loadContractMetadata(plan.id);
  layer.channel_tile = channelTile;
  layer.contract_params = {
    ...(currentContractId(baseLayer) === plan.id ? baseLayer.contract_params ?? {} : {}),
    ...(metadata.fit_constraints.default_beat_width_bits
      ? { beat_width_bits: metadata.fit_constraints.default_beat_width_bits }
      : {}),
  };
  layer.input_width_bits = tiledInputBusWidthBits(baseLayer, channelTile);
  layer.output_width_bits = tiledOutputBusWidthBits(channelTile);
  return layer;
}

function contractPlanForLayer(layer: LayerIR): ContractPlan {
  const id = currentContractId(layer);
  const plan = CONTRACT_PLANS.find((candidate) => candidate.id === id);
  if (!plan) {
    throw new Error(`No contract plan registered for '${id}'.`);
  }
  return plan;
}

function isContractFlagged(state: ContractResponseState, layer: LayerIR): boolean {
  return state.contracts[contractStateKeyForLayer(layer)]?.status === "manual_correction_needed";
}

function selectAvailableContract(
  baseLayer: LayerIR,
  state: ContractResponseState,
  afterContractId?: ContractId,
): { plan: ContractPlan; layer: LayerIR } | null {
  const startIndex = afterContractId
    ? CONTRACT_PLANS.findIndex((plan) => plan.id === afterContractId) + 1
    : CONTRACT_PLANS.findIndex((plan) => plan.id === currentContractId(baseLayer));
  for (const plan of CONTRACT_PLANS.slice(Math.max(0, startIndex))) {
    const layer = applyContractPlan(baseLayer, plan);
    if (contractFitFailure(layer)) {
      continue;
    }
    if (!isContractFlagged(state, layer)) {
      return { plan, layer };
    }
  }
  return null;
}

function setActiveLayerForModule(
  pipelineIr: PipelineIR,
  activeLayers: Map<string, LayerIR>,
  moduleId: string,
  layer: LayerIR,
): void {
  activeLayers.set(moduleId, layer);
  const index = pipelineIr.layers.findIndex((candidate) => candidate.module_id === moduleId);
  if (index >= 0) {
    pipelineIr.layers[index] = layer;
  }
}

async function writeManualCorrectionReport(input: {
  moduleId: string;
  layer: LayerIR;
  result: VerifResult;
  reason: string;
  runtime: OrchestratorRuntime;
}): Promise<string> {
  const key = contractStateKeyForLayer(input.layer);
  const reportAbs = reportPath(`manual_correction_${sanitizePathPart(key)}.json`);
  await writeJsonFile(reportAbs, {
    event: "manual_correction_needed",
    generated_at: input.runtime.now().toISOString(),
    module_id: input.moduleId,
    contract_key: key,
    contract_id: currentContractId(input.layer),
    contract_plan: contractPlanForLayer(input.layer),
    contract: buildModuleContractSummary(input.layer),
    layer_ir: input.layer,
    reason: input.reason,
    final_result: input.result,
    foundry_versions: foundryVersionsFor(input.layer),
    failure_attempts: failureAttemptsFor(input.layer),
    available_contracts: CONTRACT_PLANS,
  });
  return relFromRepo(reportAbs);
}

async function flagContractForManualCorrection(
  state: ContractResponseState,
  layer: LayerIR,
  moduleId: string,
  reason: string,
  result: VerifResult,
  runtime: OrchestratorRuntime,
): Promise<ContractFlag> {
  const key = contractStateKeyForLayer(layer);
  const existing = state.contracts[key];
  const reportPathRel = await writeManualCorrectionReport({
    moduleId,
    layer,
    result,
    reason,
    runtime,
  });
  const moduleIds = new Set(existing?.module_ids ?? []);
  moduleIds.add(moduleId);
  const now = runtime.now().toISOString();
  const flag: ContractFlag = {
    key,
    contract_id: currentContractId(layer),
    spec_hash: computeExpectedSpecHash(layer),
    op_type: layer.op_type,
    status: "manual_correction_needed",
    flagged_at: existing?.flagged_at ?? now,
    updated_at: now,
    module_ids: [...moduleIds].sort(),
    reason,
    report_path: reportPathRel,
    last_failure_result: result,
  };
  state.contracts[key] = flag;
  await appendRunLog(
    {
      event: "contract_manual_correction_needed",
      module_id: moduleId,
      contract_key: key,
      contract_id: flag.contract_id,
      reason,
      report_path: reportPathRel,
      result,
    },
    runtime,
  );
  return flag;
}

function buildModuleContractSummary(layer: LayerIR): ModuleContractSummary {
  const operation: Record<string, unknown> = {
    op_type: layer.op_type,
    contract_id: currentContractId(layer),
    contract_complexity: contractPlanForLayer(layer).complexity,
    input_shape: layer.input_shape,
    output_shape: layer.output_shape,
    weight_shape: layer.weight_shape,
    scale_factor: layer.scale_factor,
    zero_point: layer.zero_point,
  };

  if (layer.op_type === "conv2d") {
    operation.stride = layer.stride;
    operation.padding = layer.padding;
    operation.dilation = layer.dilation ?? [1, 1];
    operation.groups = layer.groups ?? 1;
    operation.mac_parallelism = layer.mac_parallelism;
    operation.io_mode = layer.io_mode ?? "packed_full";
    operation.channel_tile = layer.channel_tile ?? null;
    operation.weight_bank_paths = layer.weight_bank_paths ?? [];
  } else if (layer.op_type === "add") {
    operation.lhs_scale_factor = layer.lhs_scale_factor ?? null;
    operation.rhs_scale_factor = layer.rhs_scale_factor ?? null;
    operation.packed_operand_bits = layer.input_width_bits / 2;
  } else if (layer.op_type === "maxpool") {
    operation.kernel_size = layer.kernel_size;
    operation.pool_stride = layer.pool_stride;
    operation.pool_padding = layer.pool_padding;
  }

  return {
    interface: {
      clock: "clk",
      reset: "rst_n",
      valid_in: "valid_in",
      ready_in: "ready_in",
      valid_out: "valid_out",
      data_in_bits: layer.input_width_bits,
      data_out_bits: layer.output_width_bits,
    },
    timing: {
      pipeline_latency_cycles: layer.pipeline_latency_cycles,
      clock_period_ns: layer.clock_period_ns,
      fmax_target_mhz: FMAX_TARGET_MHZ,
    },
    capability_limits: {
      max_supported_bus_bits: PIPELINE_CONFIG.MAX_SUPPORTED_BUS_BITS,
      target_part: "xczu9eg-ffvb1156-2-e",
      zcu102_capacity: {
        lut: 274080,
        ff: 548160,
        dsp: 2520,
        bram18: 1824,
      },
    },
    operation,
  };
}

function buildFailureLogs(
  result: VerifResult,
  extraLogs: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    iverilog_stderr: result.iverilog_stderr ?? "",
    verilator_stderr: result.verilator_stderr ?? "",
    fix_hint: result.fix_hint ?? "",
    verif_result_json: result,
    ...extraLogs,
  };
}

function failureText(result: VerifResult): string {
  return [
    result.iverilog_stderr,
    result.verilator_stderr,
    result.fix_hint,
    result.classifier_reason,
    result.violated_constraint,
  ]
    .filter((value): value is string => typeof value === "string" && value.length > 0)
    .join("\n");
}

function isToolchainInfrastructureFailure(result: VerifResult): boolean {
  if (result.status_class === "tb_setup_error") return true;
  const text = failureText(result);
  return (
    /exited non-zero without diagnostic output/i.test(text) ||
    /exit_code=3221225794/i.test(text) ||
    /0xC0000002/i.test(text) ||
    /STATUS_NOT_IMPLEMENTED/i.test(text) ||
    /toolchain|runtime setup failure|Assayer runner crashed|Vivado failed before producing/i.test(text)
  );
}

function isDramBackedVerificationEnvironmentFailure(result: VerifResult, layer?: LayerIR): boolean {
  if (!layer || currentContractId(layer) !== "dram-backed-weights") return false;
  if (result.status_class !== "sim_stalled" || (result.outputs_received ?? -1) !== 0) return false;

  // Old verifier results had no AXI trace at all; for dram-backed-weights
  // that means the external-memory contract did not receive a meaningful
  // protocol verdict and retries would only burn Foundry calls.
  if (result.axi_weight_memory_model_enabled === undefined) return true;
  if (result.axi_weight_memory_model_enabled === false) return true;

  const arvalid = result.axi_weight_arvalid_cycles ?? 0;
  const arready = result.axi_weight_arready_cycles ?? 0;
  const handshakes = result.axi_weight_ar_handshakes ?? 0;
  return arvalid > 0 && arready === 0 && handshakes === 0;
}

function deterministicFailureClassification(
  result: VerifResult,
  layer?: LayerIR,
): FailureClassification | null {
  if (isToolchainInfrastructureFailure(result)) {
    return {
      category: "toolchain_infra",
      violated_resource: null,
      violated_constraint: result.violated_constraint ?? result.failure_class ?? "toolchain_or_testbench_setup",
      rationale:
        "Deterministic evidence says the RTL did not receive a trustworthy compiler/simulation verdict because the local toolchain or testbench setup failed.",
    };
  }

  if (isDramBackedVerificationEnvironmentFailure(result, layer)) {
    return {
      category: "verification_env",
      violated_resource: null,
      violated_constraint: result.violated_constraint ?? "axi_weight_memory_model_missing",
      rationale:
        "Deterministic dram-backed-weights evidence says the module stalled before any outputs while the AXI weight protocol lacked a usable verifier response/trace. Do not route this to Surgeon as a module-local RTL repair.",
    };
  }

  if (result.failure_class === "architectural_unsupported") {
    return {
      category: "architectural_fit",
      violated_resource: result.violated_resource ?? null,
      violated_constraint: result.violated_constraint ?? "architectural_unsupported",
      rationale: "The deterministic capability gate reported an unsupported architecture/contract fit.",
    };
  }

  if (result.status === "syntax_error") {
    return {
      category: "code_bug",
      violated_resource: null,
      violated_constraint: result.failure_class ?? "verilog_syntax",
      rationale: "The compiler produced a real syntax/elaboration failure for the generated RTL.",
    };
  }

  return null;
}

function isTransientAgentOrQuotaFailure(reason: string, result: VerifResult): boolean {
  const text = `${reason}\n${failureText(result)}`;
  return /dispatch_failed|classifier_unavailable|rate limit|usage limit|quota|You've hit your limit|API|overloaded|timeout/i.test(text);
}

function shouldPersistContractManualCorrection(reason: string, result: VerifResult): boolean {
  if (isToolchainInfrastructureFailure(result)) return false;
  if (result.failure_category === "toolchain_infra") return false;
  if (result.failure_category === "verification_env") return false;
  if (isTransientAgentOrQuotaFailure(reason, result)) return false;
  return true;
}

const FOUNDRY_HISTORY = new Map<string, FoundryVersionRecord[]>();
const FAILURE_ATTEMPT_HISTORY = new Map<string, FailureAttemptRecord[]>();

type FailureCorpusScore = {
  syntax_ok: boolean;
  sim_completed: boolean;
  timing_delta_cycles: number | null;
  timing_abs_delta_cycles: number | null;
  outputs_received: number | null;
  outputs_expected: number | null;
  output_completion_ratio: number | null;
  first_mismatch_index: number | null;
  max_error: number | null;
  mean_error: number | null;
  exact_match_count: number | null;
  mismatch_count: number | null;
  signed_error_sum: number | null;
  positive_error_count: number | null;
  negative_error_count: number | null;
  axi_out_of_range_reads: number | null;
};

type FailureCorpusIndexEntry = {
  id: string;
  created_at: string;
  network_id?: string;
  model_name?: string;
  module_id: string;
  stage: string;
  attempt_index: number;
  parent_id?: string;
  op_type: LayerIR["op_type"];
  contract_id: ContractId;
  spec_hash: string;
  generated_by: VerilogModule["generated_by"] | null;
  module_attempt: number | null;
  rtl_path: string | null;
  failure_path: string;
  score: FailureCorpusScore;
  summary: Record<string, unknown>;
  shape: Record<string, unknown>;
  runtime_layer_signature?: Record<string, unknown>;
  signature_hash?: string;
  exact_reference_key?: string | null;
  applicability?: Record<string, unknown>;
  contraindications?: Array<string | Record<string, unknown>>;
};

type LayerSignatureMetadata = {
  base_layer_signature: LayerSignature;
  runtime_layer_signature: LayerSignature;
  signature_hash: string;
  exact_reference_key: string | null;
};

const FAILURE_CORPUS_VISIBLE_TIER = "visible";
const FAILURE_CORPUS_ARCHIVE_TIER = "archive";

function failureCorpusRoot(): string {
  return path.join(resolvePipelineConfigPath(PIPELINE_CONFIG.output_dir), "failure_corpus");
}

function failureCorpusTierRoot(tier: typeof FAILURE_CORPUS_VISIBLE_TIER | typeof FAILURE_CORPUS_ARCHIVE_TIER): string {
  return path.join(failureCorpusRoot(), tier);
}

function failureCorpusPathPart(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 96) || "entry";
}

function failureCorpusTimestamp(date: Date): string {
  return date.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
}

function verifScore(result: VerifResult): FailureCorpusScore {
  const actual = result.timing_actual_cycles;
  const expected = result.timing_expected_cycles;
  const timingDelta =
    typeof actual === "number" && typeof expected === "number" && actual >= 0 && expected >= 0
      ? actual - expected
      : null;
  const outputsReceived = result.outputs_received ?? null;
  const outputsExpected = result.outputs_expected ?? null;
  const completionRatio =
    typeof outputsReceived === "number" && typeof outputsExpected === "number" && outputsExpected > 0
      ? outputsReceived / outputsExpected
      : null;
  return {
    syntax_ok: result.status !== "syntax_error",
    sim_completed:
      result.status_class === "sim_passed" ||
      result.status_class === "sim_completed_mismatch" ||
      result.status === "pass",
    timing_delta_cycles: timingDelta,
    timing_abs_delta_cycles: timingDelta === null ? null : Math.abs(timingDelta),
    outputs_received: outputsReceived,
    outputs_expected: outputsExpected,
    output_completion_ratio: completionRatio,
    first_mismatch_index: result.first_mismatch_index ?? null,
    max_error: result.max_error ?? null,
    mean_error: result.mean_error ?? null,
    exact_match_count: result.exact_match_count ?? null,
    mismatch_count: result.mismatch_count ?? null,
    signed_error_sum: result.signed_error_sum ?? null,
    positive_error_count: result.positive_error_count ?? null,
    negative_error_count: result.negative_error_count ?? null,
    axi_out_of_range_reads: result.axi_weight_out_of_range_reads ?? null,
  };
}

function verifDiagnosticSummary(result: VerifResult): Record<string, unknown> {
  const score = verifScore(result);
  return {
    status: result.status,
    status_class: result.status_class ?? null,
    failure_class: result.failure_class ?? null,
    failure_category: result.failure_category ?? null,
    timing_pass: result.timing_pass ?? null,
    timing_actual_cycles: result.timing_actual_cycles ?? null,
    timing_expected_cycles: result.timing_expected_cycles ?? null,
    timing_delta_cycles: score.timing_delta_cycles,
    outputs_received: result.outputs_received ?? null,
    outputs_expected: result.outputs_expected ?? null,
    output_completion_ratio: score.output_completion_ratio,
    first_mismatch: {
      flat_index: result.first_mismatch_index ?? null,
      vector_index: result.first_mismatch_vector_index ?? null,
      output_index: result.first_mismatch_output_index ?? null,
      channel_index: result.first_mismatch_channel_index ?? null,
      expected: result.first_mismatch_expected ?? null,
      got: result.first_mismatch_got ?? null,
    },
    error_stats: {
      max_error: result.max_error ?? null,
      mean_error: result.mean_error ?? null,
      exact_match_count: result.exact_match_count ?? null,
      mismatch_count: result.mismatch_count ?? null,
      signed_error_sum: result.signed_error_sum ?? null,
      positive_error_count: result.positive_error_count ?? null,
      negative_error_count: result.negative_error_count ?? null,
    },
    gap: {
      missing_index_start: result.missing_index_start ?? null,
      missing_index_end: result.missing_index_end ?? null,
      output_gap_histogram: result.output_gap_histogram ?? null,
      last_valid_out_cycle: result.last_valid_out_cycle ?? null,
      simulation_end_cycle: result.simulation_end_cycle ?? null,
    },
    axi_weight_trace: {
      model_enabled: result.axi_weight_memory_model_enabled ?? null,
      model_status: result.axi_weight_memory_model_status ?? null,
      ar_handshakes: result.axi_weight_ar_handshakes ?? null,
      r_beats: result.axi_weight_r_beats ?? null,
      completed_bursts: result.axi_weight_completed_bursts ?? null,
      out_of_range_reads: result.axi_weight_out_of_range_reads ?? null,
    },
    per_vector: result.per_vector ?? null,
  };
}

function layerShapeSummary(layer: LayerIR): Record<string, unknown> {
  return {
    input_shape: layer.input_shape,
    output_shape: layer.output_shape,
    weight_shape: layer.weight_shape,
    input_width_bits: layer.input_width_bits,
    output_width_bits: layer.output_width_bits,
    stride: layer.stride ?? null,
    padding: layer.padding ?? null,
    dilation: layer.dilation ?? null,
    groups: layer.groups ?? null,
    mac_parallelism: layer.mac_parallelism ?? null,
    io_mode: layer.io_mode ?? null,
    channel_tile: layer.channel_tile ?? null,
  };
}

async function appendFailureCorpusIndex(entry: FailureCorpusIndexEntry): Promise<void> {
  const indexPath = path.join(failureCorpusTierRoot(FAILURE_CORPUS_VISIBLE_TIER), "index.jsonl");
  await mkdir(path.dirname(indexPath), { recursive: true });
  await appendFile(indexPath, `${JSON.stringify(entry)}\n`, "utf8");
}

async function readFailureCorpusIndexFile(indexPath: string): Promise<FailureCorpusIndexEntry[]> {
  let raw: string;
  try {
    raw = await readFile(indexPath, "utf8");
  } catch {
    return [];
  }
  const entries: FailureCorpusIndexEntry[] = [];
  for (const line of raw.split(/\r?\n/)) {
    if (!line.trim()) continue;
    try {
      const parsed = JSON.parse(line) as FailureCorpusIndexEntry;
      if (parsed && parsed.module_id && parsed.failure_path) {
        entries.push(parsed);
      }
    } catch {
      // Ignore malformed historical lines; the corpus is evidence, not state.
    }
  }
  return entries;
}

async function readVisibleFailureCorpusIndex(): Promise<FailureCorpusIndexEntry[]> {
  const indexPath = path.join(failureCorpusTierRoot(FAILURE_CORPUS_VISIBLE_TIER), "index.jsonl");
  return readFailureCorpusIndexFile(indexPath);
}

async function readCrossNetworkFailureCorpusIndex(): Promise<FailureCorpusIndexEntry[]> {
  const activeRoot = getPipelineOutputRoot();
  const seen = new Set<string>();
  const all: FailureCorpusIndexEntry[] = [];
  for (const network of listNetworks()) {
    const outputRoot = outputDirForNetwork(network.id, repoRoot);
    const indexPath = path.join(outputRoot, "failure_corpus", FAILURE_CORPUS_VISIBLE_TIER, "index.jsonl");
    const key = path.resolve(indexPath);
    if (seen.has(key)) continue;
    seen.add(key);
    const entries = await readFailureCorpusIndexFile(indexPath);
    for (const entry of entries) {
      all.push({
        ...entry,
        network_id: entry.network_id ?? network.id,
        model_name: entry.model_name ?? network.modelName,
      });
    }
  }
  const activeIndex = path.join(activeRoot, "failure_corpus", FAILURE_CORPUS_VISIBLE_TIER, "index.jsonl");
  if (!seen.has(path.resolve(activeIndex))) {
    all.push(...await readFailureCorpusIndexFile(activeIndex));
  }
  return all;
}

function failureCorpusSimilarityRank(
  layer: LayerIR,
  entry: FailureCorpusIndexEntry,
  matchLevel: SignatureMatchLevel,
): number {
  let rank = signatureMatchRank(matchLevel) * 100_000;
  const target = signatureTargetForLayer(layer);
  if (entry.signature_hash === target.signature_hash) rank -= 80;
  if (entry.module_id === layer.module_id) rank -= 100;
  if (entry.spec_hash === computeExpectedSpecHash(layer)) rank -= 50;
  if (entry.op_type === layer.op_type) rank -= 20;
  if (entry.contract_id === currentContractId(layer)) rank -= 10;
  if (signaturePaddingMatches({
    op_type: entry.op_type,
    contract_id: entry.contract_id,
    signature_hash: entry.signature_hash,
    exact_reference_key: entry.exact_reference_key,
    runtime_layer_signature: entry.runtime_layer_signature,
    applicability: entry.applicability,
    shape: entry.shape,
  }, target)) {
    rank -= 3;
  }
  const entryKh = Array.isArray(entry.shape.weight_shape) ? entry.shape.weight_shape[2] : undefined;
  const entryKw = Array.isArray(entry.shape.weight_shape) ? entry.shape.weight_shape[3] : undefined;
  if (layer.op_type === "conv2d" && entryKh === layer.weight_shape[2] && entryKw === layer.weight_shape[3]) {
    rank -= 5;
  }
  rank += entry.score.timing_abs_delta_cycles ?? 1_000_000;
  rank += entry.score.max_error ?? 10_000;
  return rank;
}

async function failureMemoryForLayer(layer: LayerIR, limit = 5): Promise<FailureCorpusIndexEntry[]> {
  const entries = await readCrossNetworkFailureCorpusIndex();
  return entries
    .map((entry) => ({ entry, matchLevel: failureEntryMatchLevelForLayer(entry, layer) }))
    .filter(
      (candidate): candidate is { entry: FailureCorpusIndexEntry; matchLevel: SignatureMatchLevel } =>
        candidate.matchLevel !== null,
    )
    .sort((a, b) => {
      const rankDelta =
        failureCorpusSimilarityRank(layer, a.entry, a.matchLevel) -
        failureCorpusSimilarityRank(layer, b.entry, b.matchLevel);
      return rankDelta !== 0 ? rankDelta : b.entry.created_at.localeCompare(a.entry.created_at);
    })
    .map((candidate) => candidate.entry)
    .slice(0, limit);
}

async function archiveVisibleFailureCorpusForModule(
  moduleId: string,
  runtime: OrchestratorRuntime,
  reason: string,
): Promise<void> {
  const source = path.join(
    failureCorpusTierRoot(FAILURE_CORPUS_VISIBLE_TIER),
    failureCorpusPathPart(moduleId),
  );
  if (!(await pathExists(source))) return;
  const targetBase = path.join(
    failureCorpusTierRoot(FAILURE_CORPUS_ARCHIVE_TIER),
    failureCorpusPathPart(moduleId),
    `${failureCorpusTimestamp(runtime.now())}_${failureCorpusPathPart(reason)}`,
  );
  let target = targetBase;
  for (let i = 1; await pathExists(target); i += 1) {
    target = `${targetBase}_${i}`;
  }
  await mkdir(path.dirname(target), { recursive: true });
  await rename(source, target);
  const indexPath = path.join(failureCorpusTierRoot(FAILURE_CORPUS_VISIBLE_TIER), "index.jsonl");
  const retained = (await readVisibleFailureCorpusIndex())
    .filter((entry) => entry.module_id !== moduleId);
  await writeFile(
    indexPath,
    retained.length > 0 ? `${retained.map((entry) => JSON.stringify(entry)).join("\n")}\n` : "",
    "utf8",
  );
  await appendRunLog(
    {
      event: "failure_corpus_archived",
      module_id: moduleId,
      reason,
      archived_path: relFromRepo(target),
      visibility: "hidden_from_agent_retrieval",
    },
    runtime,
  );
}

function jsonClone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function moduleContractKey(layer: LayerIR): string {
  return `${layer.module_id}:${computeExpectedSpecHash(layer)}`;
}

function extractSessionId(messages: SDKMessage[], result: SDKResultMessage): string | null {
  const candidates: unknown[] = [result, ...messages.slice().reverse()];
  for (const candidate of candidates) {
    if (!isRecord(candidate)) continue;
    const sessionId = candidate.session_id;
    if (typeof sessionId === "string" && sessionId.length > 0) {
      return sessionId;
    }
  }
  return null;
}

function recordFoundryVersion(
  layer: LayerIR,
  run: AgentRunResult<VerilogModule>,
  audits: ToolUseAuditEntry[],
): void {
  const key = moduleContractKey(layer);
  const history = FOUNDRY_HISTORY.get(key) ?? [];
  const tool_use_summary = summarizeToolUse(audits);
  const documents_used = audits
    .filter((entry) => entry.kind === "tool_use" && entry.tool_name === "mcp__nn2rtl-tools__get_rtl_patterns")
    .map((entry) => ({
      tool_name: entry.tool_name,
      input: entry.input,
    }));

  history.push({
    version_index: history.length + 1,
    module: jsonClone(run.payload),
    session_id: extractSessionId(run.messages, run.result),
    tool_use_summary,
    documents_used,
  });
  FOUNDRY_HISTORY.set(key, history);
}

async function persistFailureCorpusAttempt(input: {
  layer: LayerIR;
  stage: string;
  result: VerifResult;
  module: VerilogModule | null;
  attemptIndex: number;
  parentId?: string;
  runtime: OrchestratorRuntime;
  extraLogs?: Record<string, unknown>;
}): Promise<FailureCorpusIndexEntry | null> {
  if (input.result.status === "pass") return null;
  const createdAt = input.runtime.now().toISOString();
  const timestamp = failureCorpusTimestamp(input.runtime.now());
  const id = [
    failureCorpusPathPart(input.layer.module_id),
    String(input.attemptIndex).padStart(3, "0"),
    failureCorpusPathPart(input.stage),
    timestamp,
  ].join("__");
  const dir = path.join(
    failureCorpusTierRoot(FAILURE_CORPUS_VISIBLE_TIER),
    failureCorpusPathPart(input.layer.module_id),
    id,
  );
  await mkdir(dir, { recursive: true });

  const rtlAbs = input.module ? path.join(dir, `${input.layer.module_id}.v`) : null;
  if (rtlAbs && input.module) {
    await writeFile(rtlAbs, input.module.verilog_source, "utf8");
  }
  const failureAbs = path.join(dir, "failure.json");
  const signatures = signatureBundle({
    baseLayer: input.layer,
    runtimeLayer: input.layer,
    baseContractId: currentContractId(input.layer),
    runtimeContractId: currentContractId(input.layer),
  });
  const entry: FailureCorpusIndexEntry = {
    id,
    created_at: createdAt,
    network_id: getActiveNetworkId(),
    model_name: getNetwork(getActiveNetworkId()).modelName,
    module_id: input.layer.module_id,
    stage: input.stage,
    attempt_index: input.attemptIndex,
    ...(input.parentId ? { parent_id: input.parentId } : {}),
    op_type: input.layer.op_type,
    contract_id: currentContractId(input.layer),
    spec_hash: computeExpectedSpecHash(input.layer),
    generated_by: input.module?.generated_by ?? null,
    module_attempt: input.module?.attempt ?? null,
    rtl_path: rtlAbs ? relFromRepo(rtlAbs) : null,
    failure_path: relFromRepo(failureAbs),
    score: verifScore(input.result),
    summary: verifDiagnosticSummary(input.result),
    shape: layerShapeSummary(input.layer),
    runtime_layer_signature: signatures.runtime_layer_signature,
    signature_hash: signatures.signature_hash,
    exact_reference_key: signatures.exact_reference_key,
    applicability: applicabilityForSignature({
      networkId: getActiveNetworkId(),
      signatures,
    }),
    contraindications: [],
  };
  await writeJsonFile(failureAbs, {
    entry,
    layer_ir: input.layer,
    module: input.module
      ? {
          module_id: input.module.module_id,
          spec_hash: input.module.spec_hash,
          generated_by: input.module.generated_by,
          attempt: input.module.attempt,
          rtl_path: entry.rtl_path,
        }
      : null,
    verif_result: input.result,
    logs: buildFailureLogs(input.result, input.extraLogs ?? {}),
  });
  await appendFailureCorpusIndex(entry);
  await appendRunLog(
    {
      event: "failure_corpus_recorded",
      module_id: input.layer.module_id,
      id,
      stage: input.stage,
      rtl_path: entry.rtl_path,
      failure_path: entry.failure_path,
      score: entry.score,
    },
    input.runtime,
  );
  return entry;
}

// Sibling of FAILURE_ATTEMPT_HISTORY that retains the full VerilogModule
// (with verilog_source) for every recorded failure. Necessary so the
// post-Retrospector "best_known" artifact picker can hand Surgeon a complete
// module to repair without rehydrating from disk. Memory-only; cleared in
// the same lifecycle hooks as the other in-memory histories.
type AttemptArtifact = {
  attempt_index: number;
  stage: string;
  module: VerilogModule;
  result: VerifResult;
};
const ATTEMPT_ARTIFACT_HISTORY = new Map<string, AttemptArtifact[]>();

async function recordFailureAttempt(
  layer: LayerIR,
  stage: string,
  result: VerifResult,
  module: VerilogModule | null,
  runtime: OrchestratorRuntime,
  extraLogs: Record<string, unknown> = {},
): Promise<void> {
  if (result.status === "pass") return;
  const key = moduleContractKey(layer);
  const history = FAILURE_ATTEMPT_HISTORY.get(key) ?? [];
  const parentId = history.length > 0
    ? `${layer.module_id}:attempt_${history[history.length - 1].attempt_index}`
    : undefined;
  history.push({
    attempt_index: history.length + 1,
    stage,
    module: module
      ? {
          module_id: module.module_id,
          spec_hash: module.spec_hash,
          generated_by: module.generated_by,
          attempt: module.attempt,
        }
      : null,
    result: jsonClone(result),
    logs: buildFailureLogs(result, extraLogs),
  });
  FAILURE_ATTEMPT_HISTORY.set(key, history);
  if (module) {
    const artifacts = ATTEMPT_ARTIFACT_HISTORY.get(key) ?? [];
    artifacts.push({
      attempt_index: history.length,
      stage,
      module: jsonClone(module),
      result: jsonClone(result),
    });
    ATTEMPT_ARTIFACT_HISTORY.set(key, artifacts);
  }
  await persistFailureCorpusAttempt({
    layer,
    stage,
    result,
    module,
    attemptIndex: history.length,
    parentId,
    runtime,
    extraLogs,
  });
}

function failureAttemptsFor(layer: LayerIR): FailureAttemptRecord[] {
  return FAILURE_ATTEMPT_HISTORY.get(moduleContractKey(layer)) ?? [];
}

function attemptArtifactsFor(layer: LayerIR): AttemptArtifact[] {
  return ATTEMPT_ARTIFACT_HISTORY.get(moduleContractKey(layer)) ?? [];
}

/**
 * Rank a verifier outcome from "best" (most worth resuming from) to "worst".
 * Tuple is read greater-is-better: status_pass first, then sim-completed,
 * then numerical agreement, then timing closeness, then output completeness,
 * with attempt_index as the final tiebreaker (later wins on equal scores so
 * we don't regress to an older attempt that happened to score identically).
 */
function attemptRankTuple(a: AttemptArtifact): number[] {
  const r = a.result;
  const score = verifScore(r);
  const passBit = r.status === "pass" ? 1 : 0;
  const simCompletedBit = score.sim_completed ? 1 : 0;
  const sampleCount =
    typeof r.exact_match_count === "number" && typeof r.mismatch_count === "number"
      ? r.exact_match_count + r.mismatch_count
      : 0;
  // Numerical agreement: ratio of exact matches over comparable samples.
  // Falls back to 0 when no samples were ever produced (status=syntax_error
  // or stalled before first valid_out) — those rank below any attempt that
  // at least emitted a few correct bytes.
  const exactRatio =
    sampleCount > 0 && typeof r.exact_match_count === "number"
      ? r.exact_match_count / sampleCount
      : 0;
  // max_error and timing_abs_delta are minimised: invert them with a finite
  // bound so the tuple stays purely greater-is-better.
  const maxErrorRank = -(score.max_error ?? Number.POSITIVE_INFINITY);
  const timingDeltaRank = -(score.timing_abs_delta_cycles ?? Number.POSITIVE_INFINITY);
  const completionRatio = score.output_completion_ratio ?? 0;
  return [
    passBit,
    simCompletedBit,
    exactRatio,
    maxErrorRank,
    timingDeltaRank,
    completionRatio,
    a.attempt_index,
  ];
}

function compareAttemptArtifacts(a: AttemptArtifact, b: AttemptArtifact): number {
  const ta = attemptRankTuple(a);
  const tb = attemptRankTuple(b);
  for (let i = 0; i < ta.length; i += 1) {
    if (ta[i] === tb[i]) continue;
    if (Number.isNaN(ta[i]) || Number.isNaN(tb[i])) continue;
    return ta[i] > tb[i] ? -1 : 1;
  }
  return 0;
}

/**
 * Best-scoring attempt across the (module, contract)'s in-memory history.
 * Returns null when no attempt has produced a verilog module yet (only
 * possible when every prior dispatch crashed before write_verilog).
 *
 * Used by `maybeRunRetrospectorFinalAttempt` when Retrospector picks
 * `next_actor: "surgeon"` with `base_artifact: "best_known"` (the default):
 * Surgeon should repair the artifact closest to passing, not necessarily
 * the most recent attempt — Surgeon may have regressed to a worse state.
 */
function pickBestKnownAttempt(layer: LayerIR): AttemptArtifact | null {
  const artifacts = attemptArtifactsFor(layer);
  if (artifacts.length === 0) return null;
  const ranked = [...artifacts].sort(compareAttemptArtifacts);
  return ranked[0] ?? null;
}

function foundryVersionsFor(layer: LayerIR): FoundryVersionRecord[] {
  return FOUNDRY_HISTORY.get(moduleContractKey(layer)) ?? [];
}

function latestFoundrySessionId(layer: LayerIR): string | null {
  const versions = foundryVersionsFor(layer);
  for (let i = versions.length - 1; i >= 0; i -= 1) {
    const sessionId = versions[i].session_id;
    if (sessionId) return sessionId;
  }
  return null;
}

const KNOWLEDGE_ROOT = path.join(repoRoot, "knowledge");
const DOC_LIFECYCLE_STATE_PATH = path.join(KNOWLEDGE_ROOT, "doc_lifecycle.json");

function emptyDocLifecycleState(): DocLifecycleState {
  return { version: 1, docs: {} };
}

async function loadDocLifecycleState(): Promise<DocLifecycleState> {
  try {
    const raw = await readFile(DOC_LIFECYCLE_STATE_PATH, "utf8");
    const parsed = JSON.parse(raw) as Partial<DocLifecycleState>;
    return {
      version: 1,
      docs: parsed.docs && typeof parsed.docs === "object" ? parsed.docs : {},
    };
  } catch {
    return emptyDocLifecycleState();
  }
}

async function saveDocLifecycleState(state: DocLifecycleState): Promise<void> {
  await writeJsonFile(DOC_LIFECYCLE_STATE_PATH, state);
}

function relFromRepo(absPath: string): string {
  return path.relative(repoRoot, absPath).split(path.sep).join("/");
}

function absFromRepo(relPath: string): string {
  return path.resolve(repoRoot, relPath);
}

function sanitizePathPart(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 80) || "doc";
}

function docTimestamp(date: Date): string {
  return date.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
}

function docContractId(doc: Pick<DocLifecycleEntry, "contract_id">): ContractId {
  return doc.contract_id ?? "flat-bus";
}

function signatureTargetForLayer(layer: LayerIR): SignatureTarget {
  return {
    ...signatureMetadataForLayer(layer, layer),
    network_id: getActiveNetworkId(),
  };
}

function docSignatureCandidate(doc: DocLifecycleEntry): Parameters<typeof signatureCandidateMatchLevel>[0] {
  return {
    op_type: doc.op_type,
    contract_id: docContractId(doc),
    signature_hashes: doc.signature_hashes,
    exact_reference_keys: doc.exact_reference_keys,
    applicability: doc.applicability,
  };
}

const COVERING_DOC_MATCH_LEVELS = new Set<SignatureMatchLevel>([
  "exact_signature",
  "exact_reference_key",
  "op_contract_kernel_stride_groups",
  "op_contract_kernel",
  // `op_contract` covers legacy / generic contract-level docs that didn't
  // record kernel info. Without this, a doc seeded with only
  // `(op_type, contract_id)` would fail to cover its own contract's traffic
  // and force a redundant create_new_doc_request on every layer.
  "op_contract",
]);

function docMatchLevelForLayer(
  doc: DocLifecycleEntry,
  layer: LayerIR,
  levels?: ReadonlySet<SignatureMatchLevel>,
): SignatureMatchLevel | null {
  if (contraindicationVetoesLayer(doc.contraindications, layer)) return null;
  const level = signatureCandidateMatchLevel(
    docSignatureCandidate(doc),
    signatureTargetForLayer(layer),
  );
  if (level === null) return null;
  return levels && !levels.has(level) ? null : level;
}

function docPaddingRank(doc: DocLifecycleEntry, layer: LayerIR): number {
  return signaturePaddingMatches(docSignatureCandidate(doc), signatureTargetForLayer(layer)) ? 0 : 1;
}

function failureEntryMatchLevelForLayer(
  entry: FailureCorpusIndexEntry,
  layer: LayerIR,
): SignatureMatchLevel | null {
  if (contraindicationVetoesLayer(entry.contraindications, layer)) return null;
  return signatureCandidateMatchLevel(
    {
      op_type: entry.op_type,
      contract_id: entry.contract_id,
      signature_hash: entry.signature_hash,
      exact_reference_key: entry.exact_reference_key,
      runtime_layer_signature: entry.runtime_layer_signature,
      applicability: entry.applicability,
      shape: entry.shape,
    },
    signatureTargetForLayer(layer),
  );
}

function stringList(value: unknown): string[] {
  if (typeof value === "string") return [value];
  if (Array.isArray(value)) return value.filter((item): item is string => typeof item === "string");
  return [];
}

function valueMatchesRule(value: unknown, actual: string | null | undefined): boolean {
  if (actual === null || actual === undefined) return false;
  return stringList(value).some((candidate) => candidate === actual);
}

function contraindicationVetoesLayer(
  contraindications: Array<string | Record<string, unknown>> | undefined,
  layer: LayerIR,
): boolean {
  if (!contraindications || contraindications.length === 0) return false;
  const signatures = signatureMetadataForLayer(layer, layer);
  const exactReferenceKey = signatures.exact_reference_key;
  const networkId = getActiveNetworkId();
  const contractId = currentContractId(layer);
  for (const rule of contraindications) {
    if (typeof rule === "string") {
      if (
        rule === signatures.signature_hash ||
        rule === exactReferenceKey ||
        rule === layer.op_type ||
        rule === contractId ||
        rule === networkId ||
        rule === `op_type:${layer.op_type}` ||
        rule === `contract_id:${contractId}` ||
        rule === `network_id:${networkId}` ||
        rule === `signature_hash:${signatures.signature_hash}`
      ) {
        return true;
      }
      continue;
    }
    if (
      valueMatchesRule(rule.op_type, layer.op_type) ||
      valueMatchesRule(rule.contract_id, contractId) ||
      valueMatchesRule(rule.network_id, networkId) ||
      valueMatchesRule(rule.signature_hash, signatures.signature_hash) ||
      valueMatchesRule(rule.signature_hashes, signatures.signature_hash) ||
      valueMatchesRule(rule.exact_reference_key, exactReferenceKey) ||
      valueMatchesRule(rule.exact_reference_keys, exactReferenceKey)
    ) {
      return true;
    }
  }
  return false;
}

function protectedPatternCoveragePath(layer: LayerIR): string | null {
  // Depthwise convs (groups == in == out) are covered by 12_depthwise_conv.md
  // regardless of kernel size — that doc owns the per-channel datapath. Check
  // before the regular kernel-based conv2d dispatch so a 3x3 depthwise doesn't
  // collide with the standard 3x3 reference.
  if (layer.op_type === "conv2d" && currentContractId(layer) === "depthwise-conv") {
    return "knowledge/patterns/protected/12_depthwise_conv.md";
  }
  if (layer.op_type === "conv2d") {
    const kh = layer.weight_shape[2];
    const kw = layer.weight_shape[3];
    const file =
      kh === 1 && kw === 1
        ? "02_conv1x1.md"
        : kh === 3 && kw === 3
          ? "03_conv3x3_pad1.md"
          : kh === 7 && kw === 7
            ? "04_conv7x7_pad3.md"
            : null;
    return file ? `knowledge/patterns/protected/${file}` : null;
  }
  const byOp: Partial<Record<LayerIR["op_type"], string>> = {
    add: "05_add_quantized.md",
    relu: "06_relu.md",
    maxpool: "07_maxpool.md",
    global_avg_pool: "10_global_avg_pool.md",
    gemm: "11_gemm.md",
  };
  const file = byOp[layer.op_type];
  return file ? `knowledge/patterns/protected/${file}` : null;
}

function protectedPatternCandidatesForLayer(layer: LayerIR): Array<{ id: string; op_type: LayerIR["op_type"] | "shared"; relPath: string }> {
  const common = [
    { id: "protected_context", op_type: "shared" as const, relPath: "knowledge/patterns/protected/01_context.md" },
    { id: "protected_common_bugs", op_type: "shared" as const, relPath: "knowledge/patterns/protected/08_common_bugs.md" },
  ];
  if (layer.op_type === "conv2d" && currentContractId(layer) === "depthwise-conv") {
    return [
      ...common,
      {
        id: "protected_12_depthwise_conv_md",
        op_type: "conv2d" as const,
        relPath: "knowledge/patterns/protected/12_depthwise_conv.md",
      },
    ];
  }
  if (layer.op_type === "conv2d") {
    const exact =
      layer.weight_shape[2] === 1 && layer.weight_shape[3] === 1
        ? "02_conv1x1.md"
        : layer.weight_shape[2] === 3 && layer.weight_shape[3] === 3
          ? "03_conv3x3_pad1.md"
          : layer.weight_shape[2] === 7 && layer.weight_shape[3] === 7
            ? "04_conv7x7_pad3.md"
            : null;
    const convDocs = [
      "02_conv1x1.md",
      "03_conv3x3_pad1.md",
      "04_conv7x7_pad3.md",
    ];
    const ordered = exact
      ? [exact, ...convDocs.filter((file) => file !== exact)]
      : convDocs;
    return [
      ...common,
      ...ordered.map((file) => ({
        id: `protected_${file.replace(/[^a-z0-9]+/gi, "_")}`,
        op_type: "conv2d" as const,
        relPath: `knowledge/patterns/protected/${file}`,
      })),
    ];
  }
  const byOp: Partial<Record<LayerIR["op_type"], string>> = {
    add: "05_add_quantized.md",
    relu: "06_relu.md",
    maxpool: "07_maxpool.md",
    global_avg_pool: "10_global_avg_pool.md",
    gemm: "11_gemm.md",
  };
  const file = byOp[layer.op_type];
  return file
    ? [
        ...common,
        {
          id: `protected_${file.replace(/[^a-z0-9]+/gi, "_")}`,
          op_type: layer.op_type,
          relPath: `knowledge/patterns/protected/${file}`,
        },
      ]
    : common;
}

function protectedReferenceCandidatesForLayer(layer: LayerIR): Array<{ id: string; op_type: LayerIR["op_type"]; relPath: string }> {
  if (layer.op_type !== "conv2d") return [];
  // dram-backed-weights conv2d has a fundamentally different MAC pipeline
  // (AXI weight prefetch + ping-pong cache) than the on-chip-weights conv.
  // Surface a contract-specific protected reference so agents see the
  // correct prefetch-guard / cache-loaded gating shape, not the wrong one
  // from conv1x1/3x3/7x7_passing_reference.v which assume on-chip weights.
  const isDramBacked =
    layer.contract_id === "dram-backed-weights" || layer.io_mode === "dram_backed_weights";
  if (isDramBacked && layer.weight_shape[2] === 3 && layer.weight_shape[3] === 3) {
    const file = "conv3x3_drambacked_passing_reference.v";
    return [
      {
        id: `protected_${file.replace(/[^a-z0-9]+/gi, "_")}`,
        op_type: "conv2d",
        relPath: `knowledge/references/protected/${file}`,
      },
    ];
  }
  const file =
    layer.weight_shape[2] === 1 && layer.weight_shape[3] === 1
      ? "conv1x1_passing_reference.v"
      : layer.weight_shape[2] === 3 && layer.weight_shape[3] === 3
        ? "conv3x3_passing_reference.v"
        : layer.weight_shape[2] === 7 && layer.weight_shape[3] === 7
          ? "conv7x7_passing_reference.v"
          : null;
  return file
    ? [
        {
          id: `protected_${file.replace(/[^a-z0-9]+/gi, "_")}`,
          op_type: "conv2d",
          relPath: `knowledge/references/protected/${file}`,
        },
      ]
    : [];
}

// Auto-promote a passing module's RTL to a probationary reference when the
// active contract has protected pattern coverage but NO protected reference
// Verilog. This closes the "first-time-passing-on-new-contract" gap that the
// existing self-improve doc-creation flow misses (it only acts when no
// pattern doc exists; for new contracts like depthwise-conv that ship with
// a pattern doc but no proven .v, the passing module would otherwise be
// stranded under output/<network>/rtl/ and never seed future runs).
//
// Gated on `NN2RTL_AUTO_PROMOTE_REFERENCE` (default ON). Skip cases logged
// at debug level so the user can see why a passing module wasn't promoted.
async function autoPromotePassingReference(
  layer: LayerIR,
  module: VerilogModule,
  verif: VerifResult,
  synth: SynthesisReport,
  runtime: OrchestratorRuntime,
): Promise<void> {
  const enabled = parseBooleanEnv(process.env, "NN2RTL_AUTO_PROMOTE_REFERENCE", true);
  if (!enabled) return;

  // Eligibility gate. Reject anything that doesn't represent a clean,
  // template-worthy outcome. Off-by-one max_error is tolerated (matches the
  // verifier's `sim_passed` policy) but bigger drift means the body is
  // numerically off and shouldn't be a template.
  const reasonToSkip = async (msg: string): Promise<void> => {
    await appendRunLog(
      { event: "auto_promote_reference_skipped", module_id: layer.module_id, reason: msg },
      runtime,
    );
  };
  if (verif.status !== "pass") return reasonToSkip("verif_not_pass");
  if (verif.timing_pass !== true) return reasonToSkip("timing_pass_false");
  if (typeof verif.max_error === "number" && verif.max_error > 1) return reasonToSkip(`max_error_${verif.max_error}`);
  if (typeof verif.mismatch_count === "number" && verif.sample_count && verif.mismatch_count / verif.sample_count > 0.01) {
    return reasonToSkip(`mismatch_ratio_above_1pct`);
  }
  if (synth.timing_met !== true) return reasonToSkip("synth_timing_not_met");
  if (typeof synth.fmax_mhz !== "number" || synth.fmax_mhz <= 0) return reasonToSkip("synth_fmax_zero");
  if (typeof module.attempt === "number" && module.attempt > 3) return reasonToSkip(`attempt_${module.attempt}_too_late`);

  // Skip if a protected reference is already injected for this layer's
  // CONTRACT. protectedReferenceCandidatesForLayer is kernel-aware but
  // not contract-aware — for any 3×3 conv2d it returns
  // conv3x3_passing_reference.v whether the layer is standard or depthwise.
  // For depthwise-conv (and any future contract whose conv geometry overlaps
  // standard convs), we MUST ignore those candidates because they describe
  // a different datapath. Restrict the skip-trigger to flat-bus + the
  // dram-backed variant the candidate set actually targets.
  const contractIdForRef = currentContractId(layer);
  if (contractIdForRef === "flat-bus" || contractIdForRef === "dram-backed-weights") {
    const protectedRefs = protectedReferenceCandidatesForLayer(layer)
      .filter((c) => existsSync(absFromRepo(c.relPath)));
    if (protectedRefs.length > 0) return reasonToSkip(`protected_reference_exists:${protectedRefs[0].relPath}`);
  }

  // Only promote when this layer's CONTRACT has protected pattern doc
  // coverage. Mirror findCoveringDoc()'s contract filter exactly — that's
  // the same set of contracts whose protected pattern docs actually apply
  // (flat-bus + depthwise-conv today). For tiled-streaming and other
  // contracts without protected coverage, the self-improve `create_new_doc`
  // flow owns the layer and writes both pattern + reference itself.
  // Stepping on that path here causes duplicate-write collisions and
  // clobbers the Foundry-authored doc body.
  if (contractIdForRef !== "flat-bus" && contractIdForRef !== "depthwise-conv") {
    return reasonToSkip(`contract_${contractIdForRef}_owned_by_create_new_doc_flow`);
  }
  const protectedPatternPath = protectedPatternCoveragePath(layer);
  if (!protectedPatternPath || !existsSync(absFromRepo(protectedPatternPath))) {
    return reasonToSkip("no_protected_pattern_doc_create_new_doc_owns_layer");
  }

  // Skip non-conv ops for now. The reference-injection plumbing is conv-only
  // on the lifecycle-doc retrieval side; promoting a relu/add/gemm body
  // here would write a file no consumer reads. That's a separate plumbing
  // upgrade (extend lifecycle retrieval to non-conv ops).
  if (layer.op_type !== "conv2d") return reasonToSkip(`op_type_${layer.op_type}_unsupported`);

  // Don't double-promote when the lifecycle already covers this signature.
  const contractId = currentContractId(layer);
  const signatures = signatureBundle({
    baseLayer: layer,
    runtimeLayer: layer,
    baseContractId: contractId,
    runtimeContractId: contractId,
    modelQuantization: layer.quantization_family,
  });
  const lifecycleState = await loadDocLifecycleState();
  const contractKey = contractStateKeyForLayer(layer);
  for (const doc of Object.values(lifecycleState.docs)) {
    if (doc.contract_key === contractKey) return reasonToSkip(`lifecycle_doc_exists:${doc.id}`);
    if (doc.signature_hashes?.includes(signatures.signature_hash)) return reasonToSkip(`lifecycle_signature_match:${doc.id}`);
    if (signatures.exact_reference_key && doc.exact_reference_keys?.includes(signatures.exact_reference_key)) {
      return reasonToSkip(`lifecycle_exact_key_match:${doc.id}`);
    }
  }

  // Build the slug and persist. Naming mirrors the format the in-line
  // self-improve flow already uses: `auto_<contract>_<module>_<spec>_<stamp>`.
  const now = runtime.now ? runtime.now() : new Date();
  const stamp = docTimestamp(now);
  const slugBody = sanitizePathPart(`${contractId}_${layer.module_id}_${module.spec_hash}`);
  const id = `auto_${slugBody}_${stamp}`;
  const refRel = `knowledge/references/probationary/${id}.v`;
  const patternRel = `knowledge/patterns/probationary/${id}.md`;
  await mkdir(path.dirname(absFromRepo(refRel)), { recursive: true });
  await mkdir(path.dirname(absFromRepo(patternRel)), { recursive: true });

  // Derive a short pattern doc from the RTL's leading comment block. Foundry
  // already writes a descriptive header (architecture choices, latency math,
  // scale derivation) so we don't need a second LLM call to summarize.
  const rtlBody = module.verilog_source;
  const headerLines: string[] = [];
  for (const line of rtlBody.split(/\r?\n/)) {
    if (line.trim().startsWith("//") || line.trim() === "") {
      headerLines.push(line);
      continue;
    }
    break;
  }
  const headerBlock = headerLines.join("\n").trim() || "// (RTL body had no leading comment block.)";
  const networkId = (process.env.NN2RTL_NETWORK_ID ?? "").trim() || "unknown";
  const applicability = applicabilityForSignature({ networkId, signatures });

  const frontmatter = [
    "---",
    `id: ${id}`,
    "tier: probationary",
    `op_type: ${layer.op_type}`,
    `contract_id: ${contractId}`,
    `contract_key: ${contractKey}`,
    `module_id: ${layer.module_id}`,
    `spec_hash: ${module.spec_hash}`,
    `signature_hash: ${signatures.signature_hash}`,
    `exact_reference_key: ${signatures.exact_reference_key ?? "null"}`,
    `generated_by: ${module.generated_by}`,
    `created_at: ${now.toISOString()}`,
    "creation_reason: auto_promote_first_passing_reference",
    "---",
  ].join("\n");
  const patternMd = [
    frontmatter,
    "",
    `# Auto-promoted reference for \`${contractId}\` (${layer.op_type})`,
    "",
    `**Module:** \`${layer.module_id}\` from network \`${networkId}\``,
    `**Verif:** \`status=pass\`, \`mismatch=${verif.mismatch_count ?? 0}/${verif.sample_count ?? "?"}\`, \`max_error=${verif.max_error ?? 0}\``,
    `**Synth (xczu9eg):** LUT=${synth.lut_count}, FF=${synth.ff_count}, DSP=${synth.dsp_count}, BRAM18=${synth.bram18_equiv}, fmax=${synth.fmax_mhz?.toFixed?.(2) ?? synth.fmax_mhz}MHz, timing_met=${synth.timing_met}`,
    "",
    "## Why this reference exists",
    "",
    "The active contract's protected pattern doc describes the math, but no",
    "protected reference module was checked in. This module is the first",
    "passing RTL on this contract; future modules with matching signatures",
    "(see frontmatter applicability + signature_hash) can crib from it.",
    "",
    "## RTL header (authored by Foundry/Surgeon)",
    "",
    "```verilog",
    headerBlock,
    "```",
    "",
    "## Companion artifacts",
    "",
    "- Canonical RTL on disk: `output/<network-id>/rtl/" + layer.module_id + ".v`",
    "- Failure corpus (if any retries happened): `output/<network-id>/failure_corpus/visible/" + layer.module_id + "/`",
    "",
    "Promoted via `autoPromotePassingReference()`. Disable with `NN2RTL_AUTO_PROMOTE_REFERENCE=0`.",
  ].join("\n");

  await writeFile(absFromRepo(refRel), rtlBody, "utf8");
  await writeFile(absFromRepo(patternRel), `${patternMd}\n`, "utf8");

  const entry: DocLifecycleEntry = {
    id,
    op_type: layer.op_type,
    contract_id: contractId,
    contract_key: contractKey,
    spec_hash: module.spec_hash,
    status: "probationary",
    pattern_path: patternRel,
    reference_path: refRel,
    created_by_module: layer.module_id,
    created_by_agent: module.generated_by,
    created_at: now.toISOString(),
    creation_reason: "auto_promote_first_passing_reference",
    used_by_modules: [],
    successful_modules: [layer.module_id],
    failed_modules: [],
    signature_hashes: [signatures.signature_hash],
    exact_reference_keys: signatures.exact_reference_key ? [signatures.exact_reference_key] : [],
    derived_from_networks: networkId !== "unknown" ? [networkId] : [],
    derived_from_modules: [layer.module_id],
    applicability,
    contraindications: [],
  };
  lifecycleState.docs[id] = entry;
  await saveDocLifecycleState(lifecycleState);

  await appendRunLog(
    {
      event: "auto_promote_reference",
      module_id: layer.module_id,
      doc_id: id,
      contract_id: contractId,
      reference_path: refRel,
      pattern_path: patternRel,
      signature_hash: signatures.signature_hash,
      exact_reference_key: signatures.exact_reference_key,
    },
    runtime,
  );
}

async function readSnippet(relPath: string, maxChars = 12000): Promise<string | null> {
  const text = await readText(absFromRepo(relPath)).catch(() => null);
  if (text === null) return null;
  return text.length <= maxChars ? text : `${text.slice(0, maxChars)}\n[... truncated ...]`;
}

async function closestDocsForNewDoc(
  state: DocLifecycleState,
  layer: LayerIR,
): Promise<ClosestDocSnippet[]> {
  const snippets: ClosestDocSnippet[] = [];
  for (const candidate of protectedPatternCandidatesForLayer(layer)) {
    const text = await readSnippet(candidate.relPath);
    if (text !== null) {
      snippets.push({
        id: candidate.id,
        tier: "protected",
        kind: "pattern",
        op_type: candidate.op_type,
        relative_path: candidate.relPath,
        text,
      });
    }
  }
  for (const candidate of protectedReferenceCandidatesForLayer(layer)) {
    const text = await readSnippet(candidate.relPath);
    if (text !== null) {
      snippets.push({
        id: candidate.id,
        tier: "protected",
        kind: "reference",
        op_type: candidate.op_type,
        relative_path: candidate.relPath,
        text,
      });
    }
  }

  const generated = Object.values(state.docs)
    .map((doc) => ({ doc, matchLevel: docMatchLevelForLayer(doc, layer) }))
    .filter(
      (entry): entry is { doc: DocLifecycleEntry; matchLevel: SignatureMatchLevel } =>
        entry.matchLevel !== null &&
        (entry.doc.status === "active" || entry.doc.status === "probationary"),
    )
    .sort((a, b) => {
      const matchDelta = signatureMatchRank(a.matchLevel) - signatureMatchRank(b.matchLevel);
      if (matchDelta !== 0) return matchDelta;
      const paddingDelta = docPaddingRank(a.doc, layer) - docPaddingRank(b.doc, layer);
      if (paddingDelta !== 0) return paddingDelta;
      const aTier = a.doc.status === "active" ? 0 : 1;
      const bTier = b.doc.status === "active" ? 0 : 1;
      return aTier !== bTier ? aTier - bTier : a.doc.id.localeCompare(b.doc.id);
    })
    .slice(0, 4);
  for (const { doc } of generated) {
    const patternText = await readSnippet(doc.pattern_path);
    if (patternText !== null) {
      snippets.push({
        id: doc.id,
        tier: doc.status === "active" ? "active" : "probationary",
        kind: "pattern",
        op_type: doc.op_type,
        contract_id: doc.contract_id,
        relative_path: doc.pattern_path,
        text: patternText,
      });
    }
    const referenceText = await readSnippet(doc.reference_path, 16000);
    if (referenceText !== null) {
      snippets.push({
        id: doc.id,
        tier: doc.status === "active" ? "active" : "probationary",
        kind: "reference",
        op_type: doc.op_type,
        contract_id: doc.contract_id,
        relative_path: doc.reference_path,
        text: referenceText,
      });
    }
  }
  return snippets.slice(0, 10);
}

type CoveringDoc = {
  tier: "protected" | "active" | "probationary";
  path: string;
  doc_id?: string;
};

/**
 * Locate any existing pattern doc that covers the layer's
 * (contract_id, op_type, kernel signature). Walks all three live tiers
 * (`protected`, `active`, `probationary`) and returns the first match in
 * priority order, or `null` when no doc covers this layer.
 *
 * Used by:
 *   - `maybeBuildCreateNewDocRequest` to decide whether Foundry needs
 *     `create_new_doc_request` context with closest-family docs.
 *   - The `self_improve_doc_request` guard: when coverage exists we do not
 *     ask Foundry to emit a `draft_doc`, so the probationary tier does not
 *     accumulate redundant timestamped duplicates of contracts that already
 *     have a stable doc.
 */
function findCoveringDoc(state: DocLifecycleState, layer: LayerIR): CoveringDoc | null {
  // Tier 1: protected. Flat-bus owns kernel-specific conv docs +
  // 05_add_quantized.md / 06_relu.md / 07_maxpool.md /
  // 10_global_avg_pool.md / 11_gemm.md; depthwise-conv owns
  // 12_depthwise_conv.md. Other contracts inherit no protected coverage
  // (dram-backed-weights and friends are tier-1-on-09 anyway, handled via
  // the contract-walker, not findCoveringDoc).
  const contractId = currentContractId(layer);
  if (contractId === "flat-bus" || contractId === "depthwise-conv") {
    const coveragePath = protectedPatternCoveragePath(layer);
    if (coveragePath !== null && existsSync(absFromRepo(coveragePath))) {
      return { tier: "protected", path: coveragePath };
    }
  }

  // Tier 2 + 3: generated docs must match through the same post-contract
  // signature/applicability ladder used for retrieval. A weakest `op` match
  // can be useful context, but it is too broad to suppress new-doc creation.
  const tierRank: Record<"active" | "probationary", number> = { active: 0, probationary: 1 };
  const candidates = Object.values(state.docs)
    .map((doc) => ({
      doc,
      matchLevel: docMatchLevelForLayer(doc, layer, COVERING_DOC_MATCH_LEVELS),
    }))
    .filter(
      (entry): entry is { doc: DocLifecycleEntry; matchLevel: SignatureMatchLevel } =>
        entry.matchLevel !== null &&
        (entry.doc.status === "active" || entry.doc.status === "probationary"),
    )
    .sort((a, b) => {
      const matchDelta = signatureMatchRank(a.matchLevel) - signatureMatchRank(b.matchLevel);
      if (matchDelta !== 0) return matchDelta;
      const paddingDelta = docPaddingRank(a.doc, layer) - docPaddingRank(b.doc, layer);
      if (paddingDelta !== 0) return paddingDelta;
      const ta = tierRank[a.doc.status as "active" | "probationary"];
      const tb = tierRank[b.doc.status as "active" | "probationary"];
      return ta !== tb ? ta - tb : a.doc.id.localeCompare(b.doc.id);
    });

  if (candidates.length === 0) return null;
  const match = candidates[0].doc;
  return {
    tier: match.status === "active" ? "active" : "probationary",
    path: match.pattern_path,
    doc_id: match.id,
  };
}

function contractHasDocCoverage(state: DocLifecycleState, layer: LayerIR): boolean {
  return findCoveringDoc(state, layer) !== null;
}

async function maybeBuildCreateNewDocRequest(
  layer: LayerIR,
  failureContext: NewDocFailureContext | undefined,
  runtime: OrchestratorRuntime,
): Promise<CreateNewDocRequest | null> {
  const state = await loadDocLifecycleState();
  if (contractHasDocCoverage(state, layer)) {
    return null;
  }
  const request: CreateNewDocRequest = {
    enabled: true,
    destination_tier: "probationary",
    contract_id: currentContractId(layer),
    contract_key: contractStateKeyForLayer(layer),
    reason: failureContext?.reason ?? "no_existing_doc_matches_selected_contract",
    no_external_retrieval: true,
    closest_existing_docs: await closestDocsForNewDoc(state, layer),
    failure_context: failureContext ?? {
      reason: "no_existing_doc_matches_selected_contract",
    },
  };
  await appendRunLog(
    {
      event: "create_new_doc_requested",
      module_id: layer.module_id,
      contract_id: request.contract_id,
      contract_key: request.contract_key,
      reason: request.reason,
      closest_doc_ids: request.closest_existing_docs.map((doc) => doc.id),
      no_external_retrieval: true,
    },
    runtime,
  );
  return request;
}

function lifecycleDocsForLayer(
  state: DocLifecycleState,
  layer: LayerIR,
  tiers: Array<"active" | "probationary"> = ["active", "probationary"],
): KnowledgeDocRecord[] {
  const tierRank: Record<"active" | "probationary", number> = { active: 0, probationary: 1 };
  return Object.values(state.docs)
    .map((doc) => ({ doc, matchLevel: docMatchLevelForLayer(doc, layer) }))
    .filter(
      (entry): entry is { doc: DocLifecycleEntry; matchLevel: SignatureMatchLevel } =>
        entry.matchLevel !== null &&
        tiers.includes(entry.doc.status as "active" | "probationary")
    )
    .sort((a, b) => {
      const matchDelta = signatureMatchRank(a.matchLevel) - signatureMatchRank(b.matchLevel);
      if (matchDelta !== 0) return matchDelta;
      const paddingDelta = docPaddingRank(a.doc, layer) - docPaddingRank(b.doc, layer);
      if (paddingDelta !== 0) return paddingDelta;
      const tierDelta =
        tierRank[a.doc.status as "active" | "probationary"] -
        tierRank[b.doc.status as "active" | "probationary"];
      return tierDelta !== 0 ? tierDelta : a.doc.id.localeCompare(b.doc.id);
    })
    .flatMap(({ doc, matchLevel }): KnowledgeDocRecord[] => [
      {
        id: doc.id,
        tier: doc.status === "active" ? "active" : "probationary",
        kind: "pattern",
        op_type: doc.op_type,
        contract_id: doc.contract_id,
        path: absFromRepo(doc.pattern_path),
        relative_path: doc.pattern_path,
        match_level: matchLevel,
      },
      {
        id: doc.id,
        tier: doc.status === "active" ? "active" : "probationary",
        kind: "reference",
        op_type: doc.op_type,
        contract_id: doc.contract_id,
        path: absFromRepo(doc.reference_path),
        relative_path: doc.reference_path,
        match_level: matchLevel,
      },
    ]);
}

async function recordDocUsageForAgent(
  layer: LayerIR,
  moduleId: string,
  audits: ToolUseAuditEntry[],
  runtime: OrchestratorRuntime,
): Promise<void> {
  const calledPatternTool = audits.some(
    (entry) =>
      entry.kind === "tool_use" &&
      entry.tool_name === "mcp__nn2rtl-tools__get_rtl_patterns",
  );
  if (!calledPatternTool) return;

  const state = await loadDocLifecycleState();
  let changed = false;
  for (const doc of Object.values(state.docs)) {
    if (docMatchLevelForLayer(doc, layer) === null) continue;
    if (doc.status !== "probationary" && doc.status !== "active") continue;
    if (!doc.used_by_modules.includes(moduleId)) {
      doc.used_by_modules.push(moduleId);
      changed = true;
    }
  }
  if (!changed) return;

  await saveDocLifecycleState(state);
  await appendRunLog(
    {
      event: "doc_lifecycle_usage_recorded",
      module_id: moduleId,
      docs: Object.values(state.docs)
        .map((doc) => ({ doc, matchLevel: docMatchLevelForLayer(doc, layer) }))
        .filter((entry) => entry.matchLevel !== null && entry.doc.used_by_modules.includes(moduleId))
        .map((entry) => ({
          id: entry.doc.id,
          status: entry.doc.status,
          contract_id: docContractId(entry.doc),
          match_level: entry.matchLevel,
        })),
    },
    runtime,
  );
}

async function moveLifecycleFile(
  relPath: string,
  targetTier: "active" | "archive",
  suffix = "",
): Promise<string> {
  const sourceAbs = absFromRepo(relPath);
  const collection = relPath.includes("/references/") ? "references" : "patterns";
  const ext = path.extname(relPath);
  const stem = path.basename(relPath, ext);
  const targetName = `${stem}${suffix}${ext}`;
  const targetAbs = path.join(KNOWLEDGE_ROOT, collection, targetTier, targetName);
  await mkdir(path.dirname(targetAbs), { recursive: true });
  await rename(sourceAbs, targetAbs);
  return relFromRepo(targetAbs);
}

async function archiveDocEntry(
  doc: DocLifecycleEntry,
  reason: string,
  runtime: OrchestratorRuntime,
): Promise<void> {
  if (doc.status === "archived") return;
  if (doc.status !== "active" && doc.status !== "probationary") return;

  const suffix = `.archived_${docTimestamp(runtime.now())}`;
  const archivedPattern = await moveLifecycleFile(doc.pattern_path, "archive", suffix);
  const archivedReference = await moveLifecycleFile(doc.reference_path, "archive", suffix);
  doc.status = "archived";
  doc.archived_at = runtime.now().toISOString();
  doc.archive_reason = reason;
  doc.archived_pattern_path = archivedPattern;
  doc.archived_reference_path = archivedReference;

  await appendRunLog(
    {
      event: "doc_lifecycle_archived",
      doc_id: doc.id,
      reason,
      archived_pattern_path: archivedPattern,
      archived_reference_path: archivedReference,
    },
    runtime,
  );
}

async function archiveProbationaryDocsForFailure(
  layer: LayerIR,
  moduleId: string,
  reason: string,
  runtime: OrchestratorRuntime,
): Promise<void> {
  const state = await loadDocLifecycleState();
  const docs = Object.values(state.docs).filter(
    (doc) =>
      docMatchLevelForLayer(doc, layer) !== null &&
      doc.status === "probationary" &&
      doc.used_by_modules.includes(moduleId),
  );
  if (docs.length === 0) return;
  for (const doc of docs) {
    if (!doc.failed_modules.includes(moduleId)) {
      doc.failed_modules.push(moduleId);
    }
    await archiveDocEntry(doc, reason, runtime);
  }
  await saveDocLifecycleState(state);
}

async function archiveActiveDocsConfirmedByRetrospector(
  layer: LayerIR,
  moduleId: string,
  advice: RetrospectorAdvice,
  runtime: OrchestratorRuntime,
): Promise<string[]> {
  if (advice.doc_fault !== true) return [];

  const state = await loadDocLifecycleState();
  const requested = new Set(advice.faulty_doc_paths ?? []);
  const docs = Object.values(state.docs).filter((doc) => {
    if (docMatchLevelForLayer(doc, layer) === null || doc.status !== "active") return false;
    if (!doc.used_by_modules.includes(moduleId)) return false;
    if (requested.size === 0) return true;
    return (
      requested.has(doc.pattern_path) ||
      requested.has(doc.reference_path) ||
      requested.has(doc.id)
    );
  });
  if (docs.length === 0) return [];

  const archivedIds: string[] = [];
  for (const doc of docs) {
    if (!doc.failed_modules.includes(moduleId)) {
      doc.failed_modules.push(moduleId);
    }
    await archiveDocEntry(
      doc,
      `retrospector_doc_fault:${moduleId}`,
      runtime,
    );
    archivedIds.push(doc.id);
  }
  await saveDocLifecycleState(state);
  return archivedIds;
}

async function promoteEligibleProbationaryDocs(
  state: DocLifecycleState,
  runtime: OrchestratorRuntime,
): Promise<void> {
  const threshold = PIPELINE_CONFIG.doc_promotion_success_threshold;
  for (const doc of Object.values(state.docs)) {
    if (doc.status !== "probationary") continue;
    if (new Set(doc.successful_modules).size < threshold) continue;

    const activePattern = await moveLifecycleFile(doc.pattern_path, "active");
    const activeReference = await moveLifecycleFile(doc.reference_path, "active");
    doc.status = "active";
    doc.pattern_path = activePattern;
    doc.reference_path = activeReference;
    doc.promoted_at = runtime.now().toISOString();

    await appendRunLog(
      {
        event: "doc_lifecycle_promoted",
        doc_id: doc.id,
        threshold,
        successful_modules: doc.successful_modules,
        active_pattern_path: activePattern,
        active_reference_path: activeReference,
      },
      runtime,
    );
  }
}

async function recordSuccessfulUseOfProbationaryDocs(
  layer: LayerIR,
  moduleId: string,
  runtime: OrchestratorRuntime,
): Promise<void> {
  const state = await loadDocLifecycleState();
  let changed = false;
  for (const doc of Object.values(state.docs)) {
    if (docMatchLevelForLayer(doc, layer) === null) continue;
    if (doc.status !== "probationary") continue;
    if (!doc.used_by_modules.includes(moduleId)) continue;
    if (!doc.successful_modules.includes(moduleId)) {
      doc.successful_modules.push(moduleId);
      changed = true;
    }
  }
  if (!changed) return;
  await promoteEligibleProbationaryDocs(state, runtime);
  await saveDocLifecycleState(state);
}

async function writeProbationaryDocDraft(
  module: VerilogModule,
  layer: LayerIR,
  draft: DocDraft | null | undefined,
  runtime: OrchestratorRuntime,
  replacementFor: string[] = [],
  createNewDocRequest: CreateNewDocRequest | null = null,
): Promise<void> {
  if (!draft) {
    // Suppress the "draft missing" warning when the doc-coverage guard
    // intentionally suppressed the wrapper schema for this layer — there
    // is no agent contract for `draft_doc` to honour in that case.
    const coverageState = await loadDocLifecycleState();
    if (findCoveringDoc(coverageState, layer) !== null) {
      return;
    }
    await appendRunLog(
      {
        event: "doc_lifecycle_draft_missing",
        module_id: module.module_id,
        reason: "RTL agent did not return a draft_doc with its successful module.",
      },
      runtime,
    );
    return;
  }

  const state = await loadDocLifecycleState();
  const signatures = signatureBundle({
    baseLayer: layer,
    runtimeLayer: layer,
    baseContractId: currentContractId(layer),
    runtimeContractId: currentContractId(layer),
  });
  const sourceDocIds = createNewDocRequest?.closest_existing_docs.map((doc) => doc.id) ?? [];
  const base = [
    "auto",
    sanitizePathPart(layer.op_type),
    sanitizePathPart(currentContractId(layer)),
    sanitizePathPart(module.module_id),
    sanitizePathPart(module.spec_hash).slice(0, 32),
    docTimestamp(runtime.now()),
  ].join("_");
  const id = base;
  const patternAbs = path.join(KNOWLEDGE_ROOT, "patterns", "probationary", `${base}.md`);
  const referenceAbs = path.join(KNOWLEDGE_ROOT, "references", "probationary", `${base}.v`);

  await mkdir(path.dirname(patternAbs), { recursive: true });
  await mkdir(path.dirname(referenceAbs), { recursive: true });
  const header = [
    "---",
    `id: ${id}`,
    `tier: probationary`,
    `op_type: ${layer.op_type}`,
    `contract_id: ${currentContractId(layer)}`,
    `contract_key: ${contractStateKeyForLayer(layer)}`,
    `signature_hash: ${signatures.signature_hash}`,
    `exact_reference_key: ${signatures.exact_reference_key ?? "none"}`,
    `derived_from_networks: [${getActiveNetworkId()}]`,
    `derived_from_modules: [${module.module_id}]`,
    `module_id: ${module.module_id}`,
    `spec_hash: ${module.spec_hash}`,
    `generated_by: ${module.generated_by}`,
    `created_at: ${runtime.now().toISOString()}`,
    createNewDocRequest ? `creation_reason: create_new_doc` : `creation_reason: successful_module`,
    sourceDocIds.length > 0 ? `source_doc_ids: [${sourceDocIds.join(", ")}]` : "",
    replacementFor.length > 0 ? `replacement_for: [${replacementFor.join(", ")}]` : "",
    "---",
    "",
    `# ${draft.title}`,
    "",
  ].filter(Boolean).join("\n");

  await writeFile(patternAbs, `${header}${draft.pattern_markdown.trim()}\n`, "utf8");
  await writeFile(referenceAbs, `${draft.reference_verilog.trim()}\n`, "utf8");

  state.docs[id] = {
    id,
    op_type: layer.op_type,
    contract_id: currentContractId(layer),
    contract_key: contractStateKeyForLayer(layer),
    spec_hash: module.spec_hash,
    signature_hashes: [signatures.signature_hash],
    exact_reference_keys: signatures.exact_reference_key ? [signatures.exact_reference_key] : [],
    derived_from_networks: [getActiveNetworkId()],
    derived_from_modules: [module.module_id],
    applicability: applicabilityForSignature({
      networkId: getActiveNetworkId(),
      signatures,
    }),
    contraindications: [],
    status: "probationary",
    pattern_path: relFromRepo(patternAbs),
    reference_path: relFromRepo(referenceAbs),
    created_by_module: module.module_id,
    created_by_agent: module.generated_by,
    created_at: runtime.now().toISOString(),
    creation_reason: createNewDocRequest ? "create_new_doc" : "successful_module",
    source_doc_ids: sourceDocIds.length > 0 ? sourceDocIds : undefined,
    replacement_for: replacementFor.length > 0 ? replacementFor : undefined,
    used_by_modules: [],
    successful_modules: [],
    failed_modules: [],
  };
  await saveDocLifecycleState(state);
  await appendRunLog(
    {
      event: "doc_lifecycle_probationary_created",
      doc_id: id,
      module_id: module.module_id,
      pattern_path: state.docs[id].pattern_path,
      reference_path: state.docs[id].reference_path,
      contract_id: state.docs[id].contract_id,
      contract_key: state.docs[id].contract_key,
      creation_reason: state.docs[id].creation_reason,
      replacement_for: replacementFor,
    },
    runtime,
  );
  await archiveVisibleFailureCorpusForModule(
    module.module_id,
    runtime,
    `successful_reference_created:${id}`,
  );
}

async function finalizeSuccessfulRtlDocs(
  module: VerilogModule,
  layer: LayerIR,
  draft: DocDraft | null | undefined,
  runtime: OrchestratorRuntime,
  selfImproveEnabled: boolean,
  replacementFor: string[] = [],
  createNewDocRequest: CreateNewDocRequest | null = null,
): Promise<void> {
  if (!selfImproveEnabled) return;
  await recordSuccessfulUseOfProbationaryDocs(layer, module.module_id, runtime);
  await writeProbationaryDocDraft(module, layer, draft, runtime, replacementFor, createNewDocRequest);
}

export function buildFailureClassifierPrompt(input: FailureClassifierInput): string {
  return [
    "You are the `failure_classifier` for nn2rtl.",
    "Classify the failed module from the evidence below. Return JSON only.",
    "",
    "Categories:",
    "- code_bug: the current contract is still viable; a retry/repair could fix typos, syntax errors, signedness, simple logic mistakes, latency mismatches, indexing bugs, or missing pipeline registers.",
    "- architectural_fit: the current contract is not viable; a different contract or tiling/resource strategy is needed. Examples: bus width exceeds the configured cap, resource utilization exceeds ZCU102 / XCZU9EG capacity, DSP48E2/BRAM/LUT/FF overflow, memory cannot map to BRAM (the XCZU9EG has no UltraRAM), BRAM port/capacity pressure, or an explicit synthesis/place constraint cannot be satisfied by local RTL repair.",
    "- toolchain_infra: the RTL did not receive a trustworthy verdict because the compiler, simulator, Vivado runner, sidecar paths, Windows/WSL path translation, or API/tool dispatch failed before useful RTL diagnostics existed.",
    "- verification_env: the selected contract's deterministic testbench environment did not provide the required external protocol response, so the RTL did not receive a meaningful protocol verdict. Example: dram-backed-weights produced zero outputs and the AXI trace shows the weight memory model was missing/disabled or never asserted ready despite DUT requests.",
    "- unknown: evidence is insufficient or contradictory; escalate instead of spending blind retries.",
    "",
    "Contract-fit indicators to look for:",
    "- input/output stream width greater than max_supported_bus_bits.",
    "- Vivado/resource lines saying over-utilized, exceeds available, cannot place, no legal placement, too many LUT/FF/DSP/BRAM resources, RAMB18/RAMB36 exhausted, DSP48 exhausted, or memory too large for the target.",
    "- hard timing/resource constraints that imply the contract needs tiling, serialization, banking, or a different architecture rather than a local typo fix.",
    "- explicit unsupported-mode/capability gates from the orchestrator.",
    "",
    "For architectural_fit, set `violated_resource` or `violated_constraint` to the specific resource/constraint named by the logs, for example `DSP`, `BRAM18`, `LUT`, `MAX_SUPPORTED_BUS_BITS`, `FMAX_TARGET_MHZ`, or `weight_memory_bram_capacity`.",
    "For toolchain_infra, set `violated_constraint` to the failing setup surface, for example `iverilog_no_diagnostics`, `windows_path_translation`, `vivado_tool_error`, or `agent_dispatch_failed`.",
    "For verification_env, set `violated_constraint` to the missing verifier/protocol surface, for example `axi_weight_memory_model_missing`, `axi_weight_arready_never_asserted`, or `contract_testbench_protocol_nonresponse`.",
    "For code_bug or unknown, set both fields to null unless a specific hard constraint is still useful evidence.",
    "",
    "Output schema:",
    JSON.stringify(z.toJSONSchema(failureClassificationZod), null, 2),
    "",
    "Evidence JSON:",
    JSON.stringify(input, null, 2),
  ].join("\n");
}

async function invokeFailureClassifier(
  input: FailureClassifierInput,
  runtime: OrchestratorRuntime,
): Promise<AgentRunResult<FailureClassification>> {
  const messages: SDKMessage[] = [];
  let finalResult: SDKResultMessage | null = null;

  for await (const message of runtime.queryFn({
    prompt: buildFailureClassifierPrompt(input),
    options: {
      cwd: repoRoot,
      model: FAILURE_CLASSIFIER_CONFIG.model,
      systemPrompt: {
        type: "preset",
        preset: "claude_code",
        append:
          "You classify hardware-generation failures. Be terse, evidence-based, and return only the requested JSON.",
      },
      outputFormat: failureClassificationOutputFormat,
      maxTurns: FAILURE_CLASSIFIER_CONFIG.maxTurns,
      tools: [],
      allowedTools: [],
      disallowedTools: ["Agent", "Task", "Bash", "Read", "Write"],
    },
  })) {
    messages.push(message);
    if (isResultMessage(message)) {
      finalResult = message;
    }
  }

  if (!finalResult) {
    throw new Error("No final result message was received for failure_classifier.");
  }

  return {
    payload: requireStructuredOutput<FailureClassification>(
      finalResult,
      "failure_classifier",
      failureClassificationZod,
    ),
    result: finalResult,
    messages,
  };
}

function enforceKnownContractBreach(
  result: VerifResult,
  classification: FailureClassification,
): FailureClassification {
  if (
    result.failure_class === "architectural_unsupported" &&
    classification.category !== "architectural_fit"
  ) {
    return {
      category: "architectural_fit",
      violated_resource: classification.violated_resource ?? null,
      violated_constraint:
        classification.violated_constraint ?? "MAX_SUPPORTED_BUS_BITS",
      rationale:
        "Orchestrator capability gate already identified this as an unsupported architectural contract.",
    };
  }
  if (
    classification.category === "architectural_fit" &&
    !classification.violated_resource &&
    !classification.violated_constraint
  ) {
    return {
      ...classification,
      violated_constraint: result.failure_class ?? "architectural_fit_unspecified",
    };
  }
  return classification;
}

async function classifyFailedModule(
  manager: PipelineStateManager,
  result: VerifResult,
  layer: LayerIR,
  module: VerilogModule | null,
  runtime: OrchestratorRuntime,
  extraLogs: Record<string, unknown> = {},
): Promise<VerifResult> {
  if (result.status === "pass") {
    return result;
  }
  if (result.failure_category) {
    return result;
  }

  const deterministic = deterministicFailureClassification(result, layer);
  if (deterministic) {
    const enriched: VerifResult = {
      ...result,
      failure_category: deterministic.category,
      violated_resource: deterministic.violated_resource ?? undefined,
      violated_constraint: deterministic.violated_constraint ?? undefined,
      classifier_reason: deterministic.rationale,
    };
    await appendRunLog(
      {
        event: "failure_classifier_result",
        module_id: layer.module_id,
        classification: deterministic,
        failure_class: result.failure_class ?? null,
        status: result.status,
        status_class: result.status_class ?? null,
        deterministic: true,
      },
      runtime,
    );
    return enriched;
  }

  const input: FailureClassifierInput = {
    module_spec: layer,
    contract: buildModuleContractSummary(layer),
    failure_result: result,
    logs: buildFailureLogs(result, extraLogs),
    ...(module
      ? {
          module: {
            module_id: module.module_id,
            spec_hash: module.spec_hash,
            generated_by: module.generated_by,
            attempt: module.attempt,
          },
        }
      : {}),
  };

  let classification: FailureClassification;
  let classifierResult: SDKResultMessage | null = null;
  try {
    const run = await invokeFailureClassifier(input, runtime);
    classification = enforceKnownContractBreach(result, run.payload);
    classifierResult = run.result;
    recordUsageFromResult(manager, run.result);
  } catch (error: unknown) {
    classification =
      deterministicFailureClassification(result, layer) ??
      enforceKnownContractBreach(result, {
        category: "code_bug",
        violated_resource: null,
        violated_constraint: "failure_classifier_unavailable",
        rationale:
          `Failure classifier did not return a usable verdict; using code_bug as the retryable fallback for a structured RTL failure: ${error instanceof Error ? error.message : String(error)}`,
      });
  }

  const enriched: VerifResult = {
    ...result,
    failure_category: classification.category,
    violated_resource: classification.violated_resource ?? undefined,
    violated_constraint: classification.violated_constraint ?? undefined,
    classifier_reason: classification.rationale,
  };

  await appendRunLog(
    {
      event: "failure_classifier_result",
      module_id: layer.module_id,
      classification,
      failure_class: result.failure_class ?? null,
      status: result.status,
      status_class: result.status_class ?? null,
      ...(classifierResult
        ? {
            total_cost_usd: classifierResult.total_cost_usd,
            modelUsage: classifierResult.modelUsage,
          }
        : {}),
    },
    runtime,
  );

  return enriched;
}

export function buildRetrospectorPrompt(input: RetrospectorInput): string {
  return [
    "You are the `retrospector` for nn2rtl.",
    "One Foundry attempt and one Surgeon repair have failed for this (module, contract). You see the entire evidence trail; emit ONE concrete advisory the next Foundry can act on.",
    "",
    "== HARD RULES ==",
    "- Do not write Verilog. Emit advisory JSON only.",
    "- Do not ask for more retries beyond the one final Foundry call.",
    "- Do not propose abandoning the layer; the orchestrator owns contract switching.",
    "- Pick ONE root cause. Be specific. 'Try harder' is not an answer.",
    "",
    "== WHAT TO ANALYZE ==",
    "Read in order: the LayerIR (input/output shapes, weight_shape, scale_factor, mac_parallelism, contract_id, io_mode, pipeline_latency_cycles), the contract metadata, the RTL knowledge docs used, EVERY Foundry RTL version, and the failure logs from every attempt.",
    "",
    "Failure modes split into two families. Diagnose which family the evidence supports, then go specific.",
    "",
    "(A) Structural / FSM failures — present when you see:",
    "  - status_class `sim_stalled` or failure_class `verilator_timeout` (FSM deadlock or stuck wait state)",
    "  - failure_class `structural_preflight_failed` (named structural rule violated)",
    "  - failure_class `synthesis_failed` (Vivado rejects the construct)",
    "  - sample_count = 0 or partial output count with no further valid_out (drain/exit condition wrong)",
    "  Diagnose: which FSM state never transitions, which counter never reaches its bound, which AXI handshake never completes. Reference exact register / state names from the most recent .v.",
    "",
    "(B) Arithmetic / numerical failures — present when you see:",
    "  - status_class `sim_completed_mismatch` with full output count and small max_error (typically <= 64)",
    "  - timing_pass=true but values disagree",
    "  - error histogram clusters near zero with a long tail (Gaussian-ish), or a constant signed offset",
    "  Candidate root causes (rank by evidence — pick ONE):",
    "    1. ROUNDING DIRECTION. SCALE_ROUND_BIAS added before `>>>` rounds toward +inf for negative results. Asymmetric signed-diff distribution (e.g. +1 mismatches dominate -1 by >=1.5×) is the signature.",
    "    2. ACCUMULATOR / BIASED / SCALED WIDTH. ACC_W or BIASED_W truncates the LSBs before requantize. Errors are constant in magnitude and span all output channels uniformly.",
    "    3. WEIGHT/WINDOW LAYOUT MISMATCH. Window stored in [KH][KW][IC] order but weights laid out [IC][KH][KW] (or vice-versa). Mismatched-tuple multiplications, errors mostly small but non-zero, distributed across all outputs. Smoking gun: errors increase in magnitude with weight-magnitude.",
    "    4. SATURATION CLIPPING. Missing or wrong-direction saturation lets over-range values wrap, producing rare but very large errors. Smoking gun: max_error >> mean_error and a small number of outliers near ±128.",
    "    5. SCALE_SHIFT WRONG. The shift constant doesn't match the layer's quantization scale_factor. Errors are ALL outputs shifted by a constant power-of-2 factor. Smoking gun: error pattern is uniform multiplicative.",
    "    6. ZERO-POINT / BIAS-ADD WIDTH. Bias added at wrong width or with wrong sign; one OC group consistently off.",
    "  Read `first_mismatch_index`, `expected/got` samples, and `output_gap_histogram` from the verifier evidence. Reference exact constants/registers from the most recent .v in your suggestion.",
    "",
    "== DOC-FAULT CARVE-OUT ==",
    "- Set `doc_fault: true` only when an active/probationary generated doc used by this module is the likely root cause, rather than an implementation mistake in the RTL.",
    "- If `doc_fault` is true, set `faulty_doc_paths` to the exact lifecycle doc ids or paths from `knowledge_docs_used` that should be archived. Empty list if the specific doc can't be isolated.",
    "",
    "== EXHAUSTION SIGNAL ==",
    "- If the current contract truly cannot host this layer (e.g. weight memory architecturally wrong for the contract's storage assumption), say so clearly in `analysis`; the orchestrator will walk to the next contract.",
    "",
    "== ROUTING (next_actor / base_artifact / repair_scope) ==",
    "Decide who runs the final attempt:",
    "  - `next_actor: \"surgeon\"` (DEFAULT) — pick this when the prior RTL is salvageable: the module compiles, emits at least some correct outputs, has timing close or exact, and the failure is localized (FSM stall, boundary desync, off-by-N, missing reset, wrong transition). Surgeon preserves the working structure and applies the smallest possible fix. THIS IS THE COMMON CASE.",
    "  - `next_actor: \"foundry\"` — pick this ONLY when the design is structurally wrong, the contract is incompatible, the module never compiled, or the architecture itself must change. Foundry will discard the existing artifact and regenerate.",
    "Pair with `repair_scope`:",
    "  - `targeted_fsm_or_datapath_fix` — single-state / single-counter / single-register edits.",
    "  - `numerical_pipeline_fix` — accumulator / scale / saturation edits.",
    "  - `interface_or_contract_fix` — port / handshake / sidecar edits.",
    "  - `architecture_replacement` — only valid with `next_actor: \"foundry\"`.",
    "Pair with `base_artifact`:",
    "  - `best_known` (DEFAULT for `next_actor: \"surgeon\"`) — orchestrator picks the highest-scoring artifact across all prior attempts (a later Surgeon attempt may have regressed; this rolls back automatically).",
    "  - `latest` — force Surgeon to repair the most recent failed artifact even if it scored lower than an earlier one.",
    "  - `fresh` — discard all prior artifacts; only meaningful with `next_actor: \"foundry\"`.",
    "When in doubt, omit these fields entirely; the orchestrator's defaults are `next_actor: \"surgeon\"` + `base_artifact: \"best_known\"`, which is correct for ~80% of code-bug failures.",
    "",
    "Output schema:",
    JSON.stringify(z.toJSONSchema(retrospectorAdviceZod), null, 2),
    "",
    "Evidence JSON:",
    JSON.stringify(input, null, 2),
  ].join("\n");
}

async function invokeRetrospector(
  input: RetrospectorInput,
  runtime: OrchestratorRuntime,
): Promise<AgentRunResult<RetrospectorAdvice>> {
  const messages: SDKMessage[] = [];
  let finalResult: SDKResultMessage | null = null;

  for await (const message of runtime.queryFn({
    prompt: buildRetrospectorPrompt(input),
    options: {
      cwd: repoRoot,
      model: RETROSPECTOR_CONFIG.model,
      systemPrompt: {
        type: "preset",
        preset: "claude_code",
        append:
          "You diagnose repeated RTL-generation failures. Return only the requested JSON advisory; do not use tools.",
      },
      outputFormat: retrospectorAdviceOutputFormat,
      maxTurns: RETROSPECTOR_CONFIG.maxTurns,
      tools: [],
      allowedTools: [],
      disallowedTools: ["Agent", "Task", "Bash", "Read", "Write"],
    },
  })) {
    messages.push(message);
    if (isResultMessage(message)) {
      finalResult = message;
    }
  }

  if (!finalResult) {
    throw new Error("No final result message was received for retrospector.");
  }

  return {
    payload: requireStructuredOutput<RetrospectorAdvice>(
      finalResult,
      "retrospector",
      retrospectorAdviceZod,
    ),
    result: finalResult,
    messages,
  };
}

/**
 * Continuation-prompt for a Foundry call that resumes an existing session
 * (the post-retrospector final retry). The prompt
 * deliberately does NOT re-emit the system prompt or LayerIR — those already
 * live in the resumed session's history. It surfaces only the new evidence
 * Foundry didn't have on its previous turn: the prior verifier result,
 * Surgeon's failed repair attempt (if any), and the Retrospector advisory
 * (if any). All of this lands in Foundry's session as a SINGLE USER MESSAGE,
 * which is critical — Foundry must see Surgeon's RTL and the Retrospector
 * advice as orchestrator-provided context, not as a fake assistant turn that
 * came from itself.
 */
/**
 * Foundry should not probe-debug from raw simulator stdout — `$display` /
 * `$write` traces are a Surgeon/Assayer evidence channel, not part of
 * Foundry's first-attempt generation flow. Strip the captured stdout
 * before serialising a VerifResult into a Foundry continuation prompt to
 * keep the prompt under control (the field is capped at 64 KiB, but every
 * KiB of irrelevant text dilutes the actually-load-bearing facts).
 */
function stripVerilatorStdoutForFoundry(verif: VerifResult): VerifResult {
  if (typeof verif.verilator_stdout !== "string" || verif.verilator_stdout.length === 0) {
    return verif;
  }
  const { verilator_stdout: _stripped, ...rest } = verif;
  return rest as VerifResult;
}

export function buildFoundryContinuationPrompt(input: {
  expected_spec_hash: string;
  write_verilog_output_dir: string;
  attempt_index: number;
  prior_verif_result?: VerifResult;
  surgeon_attempt?: { module: VerilogModule; verif_result: VerifResult };
  retrospector_advice?: RetrospectorAdvice;
  is_final_attempt?: boolean;
  failure_memory?: FailureCorpusIndexEntry[];
  self_improve_doc_request?: Record<string, unknown>;
  create_new_doc_request?: CreateNewDocRequest;
}): string {
  const wantsDocOutput = input.self_improve_doc_request !== undefined;
  const wantsNewDoc = input.create_new_doc_request !== undefined;
  const lines: string[] = [
    "You are continuing the same `foundry` agent conversation. This message is from the orchestrator (a user turn) and contains evidence from your prior attempt(s).",
    "Do not re-derive the LayerIR or contract — they are unchanged from earlier turns. Build on the working memory you already have.",
    "",
  ];

  if (input.prior_verif_result) {
    lines.push(
      "Your previous attempt failed verification. The verifier reported:",
      "",
      JSON.stringify(stripVerilatorStdoutForFoundry(input.prior_verif_result), null, 2),
      "",
      "The canonical RTL on disk is your previous output; you may read it back via the Read tool if needed (it is at the same module_id .v path you wrote earlier).",
      "",
    );
  }

  if (input.surgeon_attempt) {
    lines.push(
      "After your attempt, a separate repair agent (`surgeon`) tried to fix your latest output and ALSO failed. Surgeon ran in its own conversation; what follows is its produced RTL and the verifier's verdict on it. Treat this as one more data point, NOT as a turn from yourself.",
      "",
      JSON.stringify(
        {
          surgeon_module: {
            module_id: input.surgeon_attempt.module.module_id,
            spec_hash: input.surgeon_attempt.module.spec_hash,
            generated_by: input.surgeon_attempt.module.generated_by,
            attempt: input.surgeon_attempt.module.attempt,
            verilog_source: input.surgeon_attempt.module.verilog_source,
          },
          surgeon_verif_result: stripVerilatorStdoutForFoundry(input.surgeon_attempt.verif_result),
        },
        null,
        2,
      ),
      "",
    );
  }

  if (input.retrospector_advice) {
    lines.push(
      "A separate analyst agent (`retrospector`) reviewed the full evidence trail and produced this advisory. It is from a different agent — read it as orchestrator-provided guidance, not as something you said.",
      "",
      JSON.stringify(input.retrospector_advice, null, 2),
      "",
    );
  }

  if (input.failure_memory && input.failure_memory.length > 0) {
    lines.push(
      "Visible failure memory for this module/contract family follows. These are scored failed RTL attempts saved by the orchestrator. Use the summaries to avoid repeating already-failed structures; if the source matters, read the listed rtl_path from disk.",
      "",
      JSON.stringify(input.failure_memory, null, 2),
      "",
    );
  }

  if (input.is_final_attempt) {
    lines.push(
      "This is your FINAL attempt on this contract. If it fails, the orchestrator will advance to a different contract entirely. Apply the analyst's advice concretely.",
      "",
    );
  }

  lines.push(
    "Persist the corrected RTL with `mcp__nn2rtl-tools__write_verilog` exactly once, then return the standard structured-output JSON.",
    wantsDocOutput
      ? 'Final JSON shape: { "module": VerilogModule, "draft_doc": { "title", "pattern_markdown", "reference_verilog", "notes"? } }.'
      : `Final JSON shape: { "module_id", "spec_hash", "verilog_source", "generated_by": "Foundry", "attempt": ${input.attempt_index} }.`,
    `Set generated_by to "Foundry" and attempt to ${input.attempt_index}.`,
    "Keep the public contract unchanged: base stream ports plus any selected contract-metadata extra ports, bus widths, pipeline_latency_cycles, quantization, and spec_hash all stay authoritative.",
  );

  if (wantsNewDoc) {
    lines.push(
      "",
      "This contract has no covering lifecycle doc yet. The draft_doc you emit must explain the technique using only your model knowledge plus the resumed-session context — no external retrieval.",
    );
  }

  lines.push(
    "",
    "Payload JSON:",
    JSON.stringify(
      {
        expected_spec_hash: input.expected_spec_hash,
        write_verilog_output_dir: input.write_verilog_output_dir,
        ...(input.self_improve_doc_request
          ? { self_improve_doc_request: input.self_improve_doc_request }
          : {}),
        ...(input.create_new_doc_request
          ? { create_new_doc_request: input.create_new_doc_request }
          : {}),
        ...(input.failure_memory && input.failure_memory.length > 0
          ? { failure_memory: input.failure_memory }
          : {}),
      },
      null,
      2,
    ),
  );

  return lines.join("\n");
}

export function buildFoundryRetrospectorInjectionPrompt(input: {
  layer_ir: LayerIR;
  expected_spec_hash: string;
  write_verilog_output_dir: string;
  retrospector_advice: RetrospectorAdvice;
  contract_options?: Record<string, unknown>;
  final_attempt?: number;
  self_improve_doc_request?: Record<string, unknown>;
  create_new_doc_request?: CreateNewDocRequest;
}): string {
  const finalAttempt = input.final_attempt ?? 1;
  const wantsDocOutput = input.self_improve_doc_request !== undefined;
  const wantsNewDoc = input.create_new_doc_request !== undefined;
  return [
    "You are continuing the existing `foundry` agent conversation for this same module.",
    "Do not start over mentally: preserve the working memory from the resumed session, and add the Retrospector's advice below as a new message in that conversation.",
    "",
    "Make exactly one final RTL attempt for the same LayerIR and spec_hash.",
    "Persist the RTL with mcp__nn2rtl-tools__write_verilog exactly once.",
    wantsDocOutput
      ? 'The final JSON must contain exactly { "module": VerilogModule, "draft_doc": { "title": string, "pattern_markdown": string, "reference_verilog": string, "notes"?: string } }.'
      : 'The final JSON must contain exactly { "module_id": string, "spec_hash": string, "verilog_source": string, "generated_by": "Foundry", "attempt": integer >= 1 }.',
    `Set generated_by to "Foundry" and attempt to ${finalAttempt}.`,
    "`verilog_source` must be the full Verilog source code as a single string.",
    wantsDocOutput
      ? "`draft_doc.reference_verilog` must be the reusable reference Verilog from this final attempt, and `draft_doc.pattern_markdown` must explain the transferable RTL pattern learned."
      : "",
    wantsNewDoc
      ? "This contract has no matching lifecycle doc. Use only the provided closest existing docs plus your model knowledge; do not retrieve external material. The draft_doc must explain the new technique and tag the selected contract."
      : "",
    "Keep the public contract unchanged: base stream ports plus any selected contract-metadata extra ports, bus widths, latency, quantization, and spec_hash remain authoritative.",
    "",
    "Retrospector advisory JSON:",
    JSON.stringify(input.retrospector_advice, null, 2),
    "",
    "Final-attempt payload JSON:",
    JSON.stringify(
      {
        layer_ir: input.layer_ir,
        expected_spec_hash: input.expected_spec_hash,
        ...(input.contract_options ? { contract_options: input.contract_options } : {}),
        write_verilog_output_dir: input.write_verilog_output_dir,
        ...(input.self_improve_doc_request
          ? { self_improve_doc_request: input.self_improve_doc_request }
          : {}),
        ...(input.create_new_doc_request
          ? { create_new_doc_request: input.create_new_doc_request }
          : {}),
      },
      null,
      2,
    ),
  ].join("\n");
}

export function buildDelegationPrompt(slug: AgentSlug, payload: unknown): string {
  // The outer query() IS the agent (see runDelegatedAgent: no Task/Agent
  // dispatch, agent body is attached via systemPrompt.append). This prompt is
  // therefore a direct task instruction, not a "dispatch to subagent" message.
  const lines = [
    `You are the \`${slug}\` agent. Execute the task described below.`,
    "The payload is embedded as JSON at the end of this message.",
  ];

  if (slug === "foundry" || slug === "surgeon") {
    const wantsDocOutput = isRecord(payload) && isRecord(payload.self_improve_doc_request);
    const wantsNewDoc = isRecord(payload) && isRecord(payload.create_new_doc_request);
    lines.push(
      "",
      "HARD CONTRACT — do not accept any other output:",
      "1. You MUST call the mcp__nn2rtl-tools__write_verilog tool exactly once to persist the RTL before returning.",
    );
    if (wantsDocOutput) {
      lines.push(
        "2. Your final message MUST be a single JSON object with exactly `module` and `draft_doc`:",
        '   { "module": { "module_id": string, "spec_hash": string, "verilog_source": string, "generated_by": "Foundry"|"Surgeon", "attempt": integer >= 1 }, "draft_doc": { "title": string, "pattern_markdown": string, "reference_verilog": string, "notes"?: string } }',
        "3. `module.verilog_source` MUST be the full Verilog source code as a single string (the same string passed to write_verilog).",
        "4. `draft_doc.pattern_markdown` MUST describe the reusable RTL pattern, constraints, invariants, and failure lessons. `draft_doc.reference_verilog` MUST be a reusable reference based on this exact successful RTL.",
        "5. Do NOT invent other top-level keys. Do NOT wrap the JSON in markdown fences.",
      );
      if (wantsNewDoc) {
        lines.push(
          "6. `create_new_doc_request` means no existing lifecycle doc covers this selected contract/technique.",
          "7. Use only the provided closest existing docs and your model knowledge. Do NOT use web search, external retrieval, curl, package downloads, or third-party source lookup.",
          "8. The `draft_doc.pattern_markdown` MUST name the selected `contract_id`, describe why this technique is needed, and state the interface/resource invariants that future modules can reuse.",
        );
      }
    } else {
      lines.push(
        "2. Your final message MUST be a single JSON object with exactly these five fields and NOTHING else:",
        '   { "module_id": string, "spec_hash": string, "verilog_source": string, "generated_by": "Foundry"|"Surgeon", "attempt": integer >= 1 }',
        "3. `verilog_source` MUST be the full Verilog source code as a single string (the same string passed to write_verilog).",
        "4. Do NOT invent other keys (no `source_path`, no `port_list`, no `module_name`). Do NOT wrap the JSON in markdown fences.",
        "5. If you cannot comply, still return the five-field JSON with a best-effort `verilog_source`.",
      );
    }
  }

  const brief =
    slug === "foundry"
      ? buildFoundryGenerationBrief(payload)
      : slug === "surgeon"
        ? buildSurgeonRepairBrief(payload)
        : null;
  if (brief) {
    lines.push("", brief);
  }

  lines.push("", "Payload JSON:", JSON.stringify(payload, null, 2));
  return lines.join("\n");
}

function stripJsonFences(text: string): string {
  // Agents sometimes wrap JSON in ```json ... ``` fences even when the
  // prompt asks for a bare object. Strip fences and any leading prose before
  // the first '{' so the final JSON.parse stays permissive about wrappers.
  const trimmed = text.trim();
  const fenceMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/);
  const candidate = fenceMatch ? fenceMatch[1].trim() : trimmed;
  const firstBrace = candidate.indexOf("{");
  const sliced = firstBrace > 0 ? candidate.slice(firstBrace) : candidate;
  return sliceFirstBalancedJsonObject(sliced);
}

// Trim trailing commentary after a valid top-level JSON object so strict
// JSON.parse can still consume the structured output. Walk once, respecting
// string-literal escapes; cut at the closing brace of the outermost object.
function sliceFirstBalancedJsonObject(text: string): string {
  if (!text.startsWith("{")) return text;
  let depth = 0;
  let inString = false;
  let escape = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (escape) escape = false;
      else if (ch === "\\") escape = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') {
      inString = true;
      continue;
    }
    if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) return text.slice(0, i + 1);
    }
  }
  return text;
}

/**
 * Wraps a parse failure with the underlying SDKResultMessage so the caller
 * can still pull cost/session/usage out of the failed agent turn instead of
 * losing it to the recovery-from-disk path. Without this, recovered runs
 * report `total_cost_usd: 0` even when Foundry burned $3+ producing
 * malformed final JSON (the RTL itself was already on disk via
 * `write_verilog`).
 */
export class StructuredOutputParseError extends Error {
  public readonly result: SDKResultMessage;
  public readonly messages: SDKMessage[];
  constructor(message: string, result: SDKResultMessage, messages: SDKMessage[]) {
    super(message);
    this.name = "StructuredOutputParseError";
    this.result = result;
    this.messages = messages;
  }
}

export class SpecHashMismatchError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SpecHashMismatchError";
  }
}

/**
 * Convert a SpecHashMismatchError into a retryable VerifResult so a stale or
 * wrong-contract spec_hash from Foundry/Surgeon doesn't crash the whole
 * pipeline run. The error message itself names the bad hash and the expected
 * hash; we hand that to the next attempt as `fix_hint`. failure_category =
 * "code_bug" is what tells the state machine to fail_retry within the
 * existing per-(module, contract) attempt budget — the next Foundry call
 * (which resumes the same session) reads this VerifResult in its
 * continuation prompt and can re-emit the right hash.
 */
function specHashMismatchAsVerifResult(
  err: SpecHashMismatchError,
  layer: LayerIR,
): VerifResult {
  return {
    module_id: layer.module_id,
    status: "fail",
    timing_pass: false,
    timing_actual_cycles: 0,
    timing_expected_cycles: layer.pipeline_latency_cycles,
    failure_class: "spec_hash_mismatch",
    failure_category: "code_bug",
    classifier_reason:
      "Agent returned a spec_hash that does not match the LayerIR's selected contract. Treat as a retryable code-bug so the next attempt can re-emit with the correct hash.",
    fix_hint: err.message,
  };
}

/**
 * The Anthropic SDK throws when an agent hits its `maxTurns` cap. The error
 * message is shaped like
 *   "Claude Code returned an error result: Reached maximum number of turns (40)"
 * Match by string fragment because the SDK does not expose a typed error class
 * for this case. Both the english phrase and the SDK's `error_max_turns`
 * subtype name are checked so detection survives a future SDK polish.
 */
function isAgentMaxTurnsError(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  return (
    /Reached maximum number of turns/i.test(err.message) ||
    /error_max_turns/i.test(err.message)
  );
}

/**
 * Wrap a max-turns dispatch error as a retryable VerifResult so the
 * per-(module, contract) attempt budget can absorb it the same way as a
 * Foundry/Surgeon spec-hash mismatch. Without this, a single max-turns
 * blow-up crashes the whole pipeline run via handlePipelineError.
 */
function agentMaxTurnsAsVerifResult(
  err: unknown,
  layer: LayerIR,
  agentName: "Foundry" | "Surgeon",
): VerifResult {
  const message = err instanceof Error ? err.message : String(err);
  return {
    module_id: layer.module_id,
    status: "fail",
    timing_pass: false,
    timing_actual_cycles: 0,
    timing_expected_cycles: layer.pipeline_latency_cycles,
    failure_class: "agent_max_turns_exhausted",
    failure_category: "code_bug",
    classifier_reason: `${agentName} hit its maxTurns cap before producing a complete RTL deliverable. Treat as a retryable code-bug so the next attempt resumes the conversation and converges with the remaining budget.`,
    fix_hint: message,
  };
}

/**
 * Best-effort scrape of a `draft_doc` object out of an RTL agent's raw final
 * message text when the wrapper-schema JSON parse fails entirely. Looks for
 * a `"draft_doc"` key followed by an opening `{` and returns whatever
 * passes a relaxed `JSON.parse` of that substring; null if nothing
 * recoverable. This lets the doc-lifecycle keep growing the probationary
 * tier even when Foundry's structured output is shape-broken — the win
 * (real RTL on disk) shouldn't be invisible to Phase 4 just because the
 * final-message JSON had an unescaped backslash.
 */
function scrapeDraftDocFromText(text: string): unknown | null {
  if (!text) return null;
  const idx = text.search(/"draft_doc"\s*:\s*\{/);
  if (idx < 0) return null;
  // Walk braces from the first `{` after the key to find a balanced object.
  const start = text.indexOf("{", idx + '"draft_doc"'.length);
  if (start < 0) return null;
  let depth = 0;
  let inString = false;
  let escape = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (escape) {
      escape = false;
      continue;
    }
    if (ch === "\\") {
      escape = true;
      continue;
    }
    if (ch === '"') {
      inString = !inString;
      continue;
    }
    if (inString) continue;
    if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) {
        const candidate = text.slice(start, i + 1);
        try {
          return JSON.parse(candidate) as unknown;
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

export function requireStructuredOutput<T>(
  result: SDKResultMessage,
  label: string,
  schema: z.ZodType<T>,
): T {
  if (result.subtype !== "success") {
    throw new Error(`${label} query did not succeed: ${result.subtype}`);
  }

  const raw: unknown =
    result.structured_output !== undefined
      ? result.structured_output
      : JSON.parse(stripJsonFences(result.result));

  const parsed = schema.safeParse(raw);
  if (!parsed.success) {
    throw new Error(
      `${label} returned invalid output:\n${JSON.stringify(parsed.error.issues, null, 2)}`,
    );
  }

  return parsed.data;
}

async function runDelegatedAgent<T>(
  slug: AgentSlug,
  payload: unknown,
  outputFormat: OutputFormat,
  resultSchema: z.ZodType<T>,
  runtime: OrchestratorRuntime,
  runOptions: DelegatedAgentRunOptions = {},
): Promise<AgentRunResult<T>> {
  const agentName = normalizeAgentName(slug);
  // Single-layer dispatch: the agent body runs AS the query()'s main agent,
  // not as a Task/Agent-dispatched subagent. This eliminates the outer-driver
  // middleman that was burning ~60k output tokens wrapping the subagent's
  // tool-use loop. We load one agent definition instead of all three, and we
  // do not register `agents` or expose the `Agent`/`Task` tools.
  const agent = await loadPluginAgentDefinition(slug);
  const agentTools = agent.tools ?? [...AGENT_MCP_TOOLS[slug]];
  const disallowed = [
    ...new Set([...(agent.disallowedTools ?? []), "Agent", "Task"]),
  ];
  const messages: SDKMessage[] = [];
  let finalResult: SDKResultMessage | null = null;

  for await (const message of runtime.queryFn({
    prompt: runOptions.prompt ?? buildDelegationPrompt(slug, payload),
    options: {
      cwd: repoRoot,
      model: AGENT_CONFIG[agentName].model,
      systemPrompt: {
        type: "preset",
        preset: "claude_code",
        append: agent.prompt,
      },
      tools: agentTools,
      allowedTools: agentTools,
      disallowedTools: disallowed,
      plugins: [{ type: "local", path: pluginPath }],
      outputFormat,
      maxTurns: AGENT_CONFIG[agentName].maxTurns,
      ...(runOptions.resumeSessionId ? { resume: runOptions.resumeSessionId } : {}),
      ...(agent.effort ? { effort: agent.effort } : {}),
    },
  })) {
    messages.push(message);

    if (isResultMessage(message)) {
      finalResult = message;
    }
  }

  if (!finalResult) {
    throw new Error(`No final result message was received for agent '${slug}'.`);
  }

  let parsedPayload: T;
  try {
    parsedPayload = requireStructuredOutput<T>(finalResult, slug, resultSchema);
  } catch (err) {
    // Wrap the parse failure with the SDK message so the caller's recovery
    // path (e.g. `tryRecoverVerilogModuleFromDisk`) can still extract cost,
    // session, modelUsage and the raw text for `draft_doc` scraping.
    throw new StructuredOutputParseError(
      err instanceof Error ? err.message : String(err),
      finalResult,
      messages,
    );
  }

  return {
    payload: parsedPayload,
    result: finalResult,
    messages,
  };
}

export function findLayer(pipelineIr: PipelineIR, moduleId: string): LayerIR {
  const layer = pipelineIr.layers.find((candidate) => candidate.module_id === moduleId);
  if (!layer) {
    throw new Error(`LayerIR for module '${moduleId}' was not found in output/layer_ir.json.`);
  }

  return layer;
}

export async function loadPersistedVerilogModule(moduleId: string): Promise<VerilogModule> {
  const metaPath = path.join(resolvePipelineConfigPath(PIPELINE_CONFIG.rtl_dir), `${moduleId}.meta.json`);
  return readJsonFile<VerilogModule>(metaPath, verilogModuleZod);
}

async function logStateTransition(
  manager: PipelineStateManager,
  moduleId: string,
  from: string,
  to: string,
  reason: string,
  runtime: OrchestratorRuntime,
): Promise<void> {
  await appendRunLog(
    {
      event: "state_transition",
      module_id: moduleId,
      from,
      to,
      reason,
      pipeline_state: manager.getState(),
    },
    runtime,
  );
}

// The add-module wire contract: int8 operands are packed as
//   data_in[W-1:0]    = lhs
//   data_in[2W-1:W]   = rhs
// so `input_width_bits` for an add layer must be twice the operand (= output)
// width. This contract lives only in a Foundry prompt today; catch desync at
// LayerIR load time before Foundry silently emits garbage.
function validateAddModulePacking(pipelineIr: PipelineIR): void {
  for (const layer of pipelineIr.layers) {
    if (layer.op_type !== "add") continue;
    const expected = 2 * layer.output_width_bits;
    if (layer.input_width_bits !== expected) {
      throw new Error(
        `LayerIR '${layer.module_id}' (op_type=add): input_width_bits=${layer.input_width_bits} ` +
          `but expected ${expected} (= 2 * output_width_bits=${layer.output_width_bits}). ` +
          `Add modules must pack lhs/rhs operands into a single data_in bus.`,
      );
    }
  }
}

function getShapeChannels(shape: number[], fieldName: string, moduleId: string): number {
  if (shape.length < 2) {
    throw new Error(
      `LayerIR '${moduleId}' field '${fieldName}' must include a channel dimension; got [${shape.join(", ")}].`,
    );
  }
  return shape[1];
}

function expectedInputBusWidthBits(layer: LayerIR): number {
  if (currentContractId(layer) !== "flat-bus" && layer.channel_tile) {
    return tiledInputBusWidthBits(layer, layer.channel_tile);
  }
  return fullInputBusWidthBits(layer);
}

function expectedOutputBusWidthBits(layer: LayerIR): number {
  if (currentContractId(layer) !== "flat-bus" && layer.channel_tile) {
    return tiledOutputBusWidthBits(layer.channel_tile);
  }
  return fullOutputBusWidthBits(layer);
}

const assayerLayerBusContractZod = layerIrZod.superRefine((layer, ctx) => {
  if (currentContractId(layer) !== "flat-bus" && !layer.channel_tile) {
    ctx.addIssue({
      code: "custom",
      path: ["channel_tile"],
      message: `io_mode='${layer.io_mode}' requires channel_tile so the Assayer can derive tiled bus widths.`,
    });
  }
  if (layer.input_width_bits % 8 !== 0) {
    ctx.addIssue({
      code: "custom",
      path: ["input_width_bits"],
      message: `input_width_bits must be a multiple of 8, got ${layer.input_width_bits}.`,
    });
  }
  if (layer.output_width_bits % 8 !== 0) {
    ctx.addIssue({
      code: "custom",
      path: ["output_width_bits"],
      message: `output_width_bits must be a multiple of 8, got ${layer.output_width_bits}.`,
    });
  }

  const contractId = resolveLayerContractId(layer);
  const metadata = loadContractMetadata(contractId);
  if (contractId === "flat-bus" || layer.channel_tile) {
    const expectedInput = expectedInputBusWidthBits(layer);
    if (layer.input_width_bits !== expectedInput) {
      ctx.addIssue({
        code: "custom",
        path: ["input_width_bits"],
        message:
          `input_width_bits=${layer.input_width_bits} does not match the ${contractId} channel contract ` +
          `for op_type='${layer.op_type}' (expected ${expectedInput}).`,
      });
    }

    const expectedOutput = expectedOutputBusWidthBits(layer);
    if (layer.output_width_bits !== expectedOutput) {
      ctx.addIssue({
        code: "custom",
        path: ["output_width_bits"],
        message:
          `output_width_bits=${layer.output_width_bits} does not match the ${contractId} channel contract ` +
          `for op_type='${layer.op_type}' (expected ${expectedOutput}).`,
      });
    }
  }
  if (
    (layer.op_type === "add" ? layer.input_width_bits / 2 : layer.input_width_bits) >
      metadata.fit_constraints.max_bus_width_bits ||
    layer.output_width_bits > metadata.fit_constraints.max_bus_width_bits
  ) {
    ctx.addIssue({
      code: "custom",
      path: ["contract_id"],
      message:
        `LayerIR '${layer.module_id}' exceeds ${contractId} max_bus_width_bits=` +
        `${metadata.fit_constraints.max_bus_width_bits}.`,
    });
  }
});

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function findMatchingParen(source: string, openIndex: number): number {
  let depth = 0;
  for (let i = openIndex; i < source.length; i++) {
    const ch = source[i];
    if (ch === "(") {
      depth += 1;
    } else if (ch === ")") {
      depth -= 1;
      if (depth === 0) {
        return i;
      }
    }
  }
  return -1;
}

function extractModulePortBlock(source: string, moduleName: string): string | null {
  const moduleMatch = new RegExp(`\\bmodule\\s+${escapeRegExp(moduleName)}\\b`).exec(source);
  if (!moduleMatch) {
    return null;
  }

  let cursor = moduleMatch.index + moduleMatch[0].length;
  while (cursor < source.length && /\s/.test(source[cursor])) {
    cursor += 1;
  }

  if (source[cursor] === "#") {
    const paramOpen = source.indexOf("(", cursor);
    if (paramOpen === -1) {
      return null;
    }
    const paramClose = findMatchingParen(source, paramOpen);
    if (paramClose === -1) {
      return null;
    }
    cursor = paramClose + 1;
  }

  const portOpen = source.indexOf("(", cursor);
  if (portOpen === -1) {
    return null;
  }
  const portClose = findMatchingParen(source, portOpen);
  if (portClose === -1) {
    return null;
  }
  return source.slice(portOpen + 1, portClose);
}

function splitTopLevelCommaList(text: string): string[] {
  const parts: string[] = [];
  let start = 0;
  let parenDepth = 0;
  let bracketDepth = 0;
  let braceDepth = 0;

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === "(") {
      parenDepth += 1;
    } else if (ch === ")") {
      parenDepth = Math.max(0, parenDepth - 1);
    } else if (ch === "[") {
      bracketDepth += 1;
    } else if (ch === "]") {
      bracketDepth = Math.max(0, bracketDepth - 1);
    } else if (ch === "{") {
      braceDepth += 1;
    } else if (ch === "}") {
      braceDepth = Math.max(0, braceDepth - 1);
    } else if (ch === "," && parenDepth === 0 && bracketDepth === 0 && braceDepth === 0) {
      parts.push(text.slice(start, i));
      start = i + 1;
    }
  }

  parts.push(text.slice(start));
  return parts;
}

function stripVerilogComments(text: string): string {
  return text
    .replace(/\/\*[\s\S]*?\*\//g, " ")
    .replace(/\/\/.*$/gm, " ");
}

function lastIdentifier(text: string): string | null {
  const match = text.match(/([A-Za-z_][A-Za-z0-9_$]*)\s*$/);
  return match ? match[1] : null;
}

function parseDeclaredPortWidthBits(declaration: string): number | null {
  const numericRange = declaration.match(/\[\s*(\d+)\s*:\s*(\d+)\s*\]/);
  if (numericRange) {
    return Math.abs(Number(numericRange[1]) - Number(numericRange[2])) + 1;
  }
  return declaration.includes("[") && declaration.includes("]") ? null : 1;
}

function parseAnsiTopPorts(portBlock: string): Map<string, ParsedTopPort> {
  const ports = new Map<string, ParsedTopPort>();
  let currentDirection: PortDirection | null = null;
  let currentWidthBits: number | null = null;

  // Strip comments BEFORE splitting on commas — inline `//` comments may
  // contain commas (e.g. "// bits [7:0]=ch0, [15:8]=ch1") that would otherwise
  // fragment the port list and cause downstream ports to inherit the wrong
  // direction / width from their predecessor.
  const cleanBlock = stripVerilogComments(portBlock);

  for (const rawEntry of splitTopLevelCommaList(cleanBlock)) {
    const declaration = rawEntry.replace(/\s+/g, " ").trim();
    if (!declaration) {
      continue;
    }

    const directionMatch = declaration.match(/^(input|output|inout)\b/i);
    const name = lastIdentifier(declaration);
    if (!name) {
      continue;
    }

    if (directionMatch) {
      currentDirection = directionMatch[1].toLowerCase() as PortDirection;
      currentWidthBits = parseDeclaredPortWidthBits(declaration);
      ports.set(name, {
        declaration,
        direction: currentDirection,
        width_bits: currentWidthBits,
      });
      continue;
    }

    if (!currentDirection) {
      continue;
    }

    ports.set(name, {
      declaration: `${currentDirection} ${declaration}`,
      direction: currentDirection,
      width_bits: currentWidthBits,
    });
  }

  return ports;
}

function evaluateContractPortWidthBits(widthExpr: string | undefined, widthBits: number | undefined, layer: LayerIR): number {
  if (typeof widthBits === "number") return widthBits;
  switch (widthExpr) {
    case "input_width_bits":
      return layer.input_width_bits;
    case "output_width_bits":
      return layer.output_width_bits;
    default:
      return 1;
  }
}

/**
 * Structural preflight checks — run AFTER the ANSI port preflight. These
 * rules derive purely from LayerIR fields (op_type, weight_shape) plus
 * generic RTL safety. They catch classes of bugs that parse cleanly but
 * are known to wedge downstream tools (Vivado synth, Verilator hang from
 * missing output bounds, etc.).
 *
 * Each violation is returned as {rule, detail}. `rule` is one of a small
 * named set (`line_buffer_missing`, `window_not_registered`,
 * `weights_packed_forbidden`, `readmemh_missing`,
 * `procedural_declaration_forbidden`, `output_counter_missing`);
 * `detail` explains the specific failure and — when applicable — cites the
 * offending line range. Callers synthesize a VerifResult with
 * failure_class="structural_preflight_failed" and the rule name surfaced
 * in fix_hint.
 */
export type StructuralPreflightViolation = { rule: string; detail: string };

/**
 * Extract the body of every `always @(posedge clk ...)` block. Returned
 * text is the concatenation of those blocks with non-clocked regions
 * elided — sufficient for pattern-matching "is `window[...] <= ...` inside
 * a clocked always block" without a full Verilog parser.
 */
function extractClockedAlwaysBlocks(source: string): string {
  const blocks: string[] = [];
  const re = /always\s*@\s*\(\s*posedge\s+clk[^)]*\)/gi;
  let m: RegExpExecArray | null;
  while ((m = re.exec(source)) !== null) {
    let cursor = m.index + m[0].length;
    // Skip whitespace
    while (cursor < source.length && /\s/.test(source[cursor])) cursor += 1;
    if (source.slice(cursor, cursor + 5) !== "begin") {
      // Single-statement always — take until the next ";".
      const semi = source.indexOf(";", cursor);
      if (semi !== -1) blocks.push(source.slice(cursor, semi + 1));
      continue;
    }
    // Walk begin/end depth.
    let depth = 0;
    let i = cursor;
    while (i < source.length) {
      const tok = source.slice(i).match(/^(\bbegin\b|\bend\b)/);
      if (tok) {
        if (tok[1] === "begin") depth += 1;
        else {
          depth -= 1;
          if (depth === 0) {
            i += tok[1].length;
            blocks.push(source.slice(cursor, i));
            break;
          }
        }
        i += tok[1].length;
      } else {
        i += 1;
      }
    }
  }
  return blocks.join("\n");
}

function forbiddenNegativeRoundingBiasSnippets(source: string): string[] {
  const snippets: string[] = [];
  const seen = new Set<string>();
  const addMatches = (re: RegExp): void => {
    let match: RegExpExecArray | null;
    while ((match = re.exec(source)) !== null) {
      const snippet = match[0].replace(/\s+/g, " ").trim();
      if (!seen.has(snippet)) {
        seen.add(snippet);
        snippets.push(snippet);
      }
    }
  };

  addMatches(
    /\b(?:localparam|parameter)\b[^;]*\b[A-Za-z_][A-Za-z0-9_$]*(?:ROUND|RND)[A-Za-z0-9_$]*(?:NEG|NEGATIVE)[A-Za-z0-9_$]*\b\s*=\s*-[^;]+;/gi,
  );
  addMatches(
    /\b(?:localparam|parameter)\b[^;]*\b[A-Za-z_][A-Za-z0-9_$]*(?:NEG|NEGATIVE)[A-Za-z0-9_$]*(?:ROUND|RND)[A-Za-z0-9_$]*\b\s*=\s*-[^;]+;/gi,
  );
  addMatches(
    /\b(?:localparam|parameter)\b[^;]*\b[A-Za-z_][A-Za-z0-9_$]*(?:BIAS)[A-Za-z0-9_$]*(?:NEG|NEGATIVE)[A-Za-z0-9_$]*\b\s*=\s*-[^;]+;/gi,
  );
  addMatches(
    /\?\s*\(?\s*-\s*(?:[A-Za-z_][A-Za-z0-9_$]*HALF[A-Za-z0-9_$]*|HALF)\s*\)?\s*:/gi,
  );

  return snippets;
}

/** Find $readmemh calls whose first arg is a non-absolute string literal.
 *  Accepted absolute forms: leading `/` (POSIX) or `[A-Za-z]:` drive (Windows).
 *  $readmemh(IDENTIFIER, ...) where the first arg is a parameter is also OK
 *  (the orchestrator can substitute it). */
function forbiddenRelativeReadmemhSnippets(source: string): string[] {
  const snippets: string[] = [];
  const seen = new Set<string>();
  const re = /\$readmemh\s*\(\s*"([^"]*)"\s*,/g;
  let match: RegExpExecArray | null;
  while ((match = re.exec(source)) !== null) {
    const literal = match[1];
    const isAbsolutePosix = literal.startsWith("/");
    const isAbsoluteWindows = /^[A-Za-z]:[\\/]/.test(literal);
    if (!isAbsolutePosix && !isAbsoluteWindows) {
      const snippet = match[0].replace(/\s+/g, " ").trim();
      if (!seen.has(snippet)) {
        seen.add(snippet);
        snippets.push(snippet);
      }
    }
  }
  return snippets;
}

/** Find ST_DONE (or any state-localparam suffixed _DONE / _IDLE_DONE) blocks
 *  whose body sets `ready_in <= 1'b0;` AND does not transition state away
 *  from itself. Uses a relaxed regex against the per-state `case` arm body. */
function forbiddenTerminalDoneLockSnippets(source: string): string[] {
  const snippets: string[] = [];
  const seen = new Set<string>();
  // Match "ST_DONE: begin ... end" (with non-greedy body capture). Tolerates
  // a state localparam name suffix `DONE` or `_DONE` (e.g. ST_DONE, ST_FINAL_DONE).
  const blockRe = /\b([A-Z][A-Z0-9_]*_?DONE)\s*:\s*begin([\s\S]*?)\bend\b/g;
  let match: RegExpExecArray | null;
  while ((match = blockRe.exec(source)) !== null) {
    const stateName = match[1];
    const body = match[2];
    // Lock signature: ready_in <= 0 AND no state <= NEW_STATE assignment.
    const setsReadyLow = /ready_in\s*<=\s*1'b0/.test(body);
    const transitionsAway = new RegExp(
      `\\bstate\\s*<=\\s*(?!${stateName}\\b)[A-Za-z_]`,
    ).test(body);
    if (setsReadyLow && !transitionsAway) {
      const snippet = `${stateName}: begin ${body.replace(/\s+/g, " ").trim().slice(0, 80)}…`;
      if (!seen.has(snippet)) {
        seen.add(snippet);
        snippets.push(snippet);
      }
    }
  }
  return snippets;
}

export function structuralPreflightViolations(
  module: VerilogModule,
  layer: LayerIR,
): StructuralPreflightViolation[] {
  const violations: StructuralPreflightViolation[] = [];
  const rawSource = module.verilog_source;
  const source = stripVerilogComments(rawSource);
  const isSpatialConv =
    layer.op_type === "conv2d" &&
    layer.weight_shape.length >= 4 &&
    layer.weight_shape[2] * layer.weight_shape[3] > 1;

  // Split-architecture detection. When the top-level instantiates the
  // handwritten library modules, the invariants those modules own (line
  // buffer, registered window, $readmemh weight/bias loading, output
  // counter) are guaranteed by the library source and don't need to be
  // textually present in the top-level file. Skip the corresponding
  // checks when the library modules are instantiated.
  const usesLineBufWindow  = /\bline_buf_window\b/.test(source);
  const usesConvDatapath   = /\bconv_datapath\b/.test(source);
  const clockedAlwaysBlocks = extractClockedAlwaysBlocks(source);

  // Rule 1: spatial conv requires a `line_buf*` reg memory declaration.
  // Accepts the literal name `line_buf` AND banked variants
  // (`line_buf_b0`, `line_buf_bank3`, `line_buf_lo`, etc.) so that a
  // module which legitimately banks the activation buffer per
  // [INVARIANT:ACTIVATION_BUFFER_BANKING] in pattern doc 09 is not forced
  // to also keep a dummy `line_buf` stub just to pass this check (the
  // stub then collides with the no-async-reset rule). Skipped when the
  // top-level instantiates `line_buf_window` — the library module owns
  // the line buffer in that case.
  if (isSpatialConv && !usesLineBufWindow) {
    const lineBufRe = /\breg\s+(?:signed\s+)?(?:\[[^\]]+\]\s+)?line_buf(?:_[A-Za-z0-9$]+)?\s*\[/;
    if (!lineBufRe.test(source)) {
      violations.push({
        rule: "line_buffer_missing",
        detail:
          `Spatial conv2d (kernel=${layer.weight_shape[2]}x${layer.weight_shape[3]}) must ` +
          `either instantiate line_buf_window or declare a line buffer 'line_buf' (or a ` +
          `banked variant like 'line_buf_b0', 'line_buf_bank3', 'line_buf_lo') ` +
          `as a multi-dimensional reg memory. Neither was found.`,
      });
    }
  }

  // Rule 2: spatial conv requires a registered window. Skipped when
  // line_buf_window is instantiated (the library owns the window).
  if (isSpatialConv && !usesLineBufWindow) {
    const windowDeclRe = /\breg\s+(?:signed\s+)?(?:\[[^\]]+\]\s+)?window\b/;
    if (!windowDeclRe.test(source)) {
      violations.push({
        rule: "window_not_registered",
        detail:
          "Spatial conv must either instantiate line_buf_window or declare a " +
          "shift-register window as 'reg [...] window [...]'. No such reg was " +
          "found — a wire/assign 'window' is rebuilt combinationally every " +
          "cycle, which blows up synth cones and loses the sliding-window invariant.",
      });
    } else {
      const windowAssignRe = /\bwindow\s*\[[^\]]+\][^;]*<=/;
      if (!windowAssignRe.test(clockedAlwaysBlocks)) {
        violations.push({
          rule: "window_not_registered",
          detail:
            "Spatial conv must update 'window' via a non-blocking assignment " +
            "(<=) inside an always @(posedge clk ...) block. No such " +
            "clocked window assignment was found.",
        });
      }
    }
  }

  // Rule 3: forbid weights_packed / packed weight initializers.
  if (/\bweights_packed\b/.test(source)) {
    violations.push({
      rule: "weights_packed_forbidden",
      detail:
        "Identifier 'weights_packed' is forbidden: packed weight arrays " +
        "block BRAM inference. Use a flat 'reg signed [7:0] " +
        "weights [0:OC*K_TOTAL-1]' array initialized via $readmemh, and " +
        "serialize reads via a lane_counter if the combinational mux is too wide.",
    });
  }
  // initial weights[...] = <expression>; (anything other than $readmemh)
  const initWeightsRe = /\binitial\b[\s\S]*?\bweights\s*\[[^\]]*\]\s*=/;
  if (initWeightsRe.test(source)) {
    violations.push({
      rule: "weights_packed_forbidden",
      detail:
        "Explicit 'initial weights[...] = ...' assignment is forbidden. " +
        "Initialize 'weights' only via $readmemh to keep the initializer " +
        "constant and compatible with Vivado memory inference.",
    });
  }
  // assign weights[...] = ... (continuous assign on the memory)
  if (/\bassign\s+weights\s*\[/.test(source)) {
    violations.push({
      rule: "weights_packed_forbidden",
      detail:
        "Continuous assignment 'assign weights[...] = ...' is forbidden — " +
        "it produces a non-constant memory initializer that Vivado cannot infer as ROM.",
    });
  }

  // Rule 3b: no declarations inside procedural blocks. Icarus versions differ
  // on how much SystemVerilog block scoping they accept, and Vivado rejects
  // many of these constructs in otherwise Verilog-style modules. Make the
  // rule deterministic before the external tools get a vote.
  const proceduralDeclRe =
    /\b(?:reg|wire|logic|integer)\b\s+(?:signed\s+)?(?:\[[^\]]+\]\s*)?[A-Za-z_][A-Za-z0-9_$]*\b/;
  const proceduralDecl = clockedAlwaysBlocks.match(proceduralDeclRe);
  if (proceduralDecl) {
    violations.push({
      rule: "procedural_declaration_forbidden",
      detail:
        "Declarations inside always blocks are forbidden for Vivado / " +
        `Verilog-2001 compatibility. Move '${proceduralDecl[0].trim()}' ` +
        "to module scope and assign it procedurally instead.",
    });
  }

  // Rule 3c: sign-aware fixed-point rounding must not subtract HALF for
  // negatives. Verilog arithmetic shift already floors toward -inf, so the
  // negative bias is +(HALF - 1), not -HALF.
  const negativeRoundingBiasSnippets = forbiddenNegativeRoundingBiasSnippets(source);
  if (negativeRoundingBiasSnippets.length > 0) {
    violations.push({
      rule: "rounding_negative_half_forbidden",
      detail:
        "Forbidden fixed-point rounding pattern: negative values may not use `-HALF`, " +
        "`-SCALE_ROUND_HALF`, or a negative `ROUND_BIAS_NEG` constant before `>>> SCALE_SHIFT`. " +
        "Use `(scaled + (scaled[MSB] ? (HALF - 1) : HALF)) >>> SHIFT` instead. " +
        `Found: ${negativeRoundingBiasSnippets.slice(0, 3).join("; ")}.`,
    });
  }

  // Rule 3d: $readmemh string-literal arguments must be absolute paths.
  // Verilator/iverilog (under the Assayer) compile in a temp build dir with
  // no `output/` subtree, so a relative path silently leaves the array
  // zero-initialised. The ROM holds plausible-but-wrong data and the RTL
  // looks "close to correct" while producing systematically biased outputs.
  // Accepted forms:
  //   $readmemh(SOME_PARAMETER, rom)         — module parameter (substitutable)
  //   $readmemh("/abs/posix/path", rom)
  //   $readmemh("C:/abs/win/path", rom)      — Windows drive letter
  const readmemhRelativeSnippets = forbiddenRelativeReadmemhSnippets(source);
  if (readmemhRelativeSnippets.length > 0) {
    violations.push({
      rule: "readmemh_relative_path_forbidden",
      detail:
        "$readmemh string-literal first arg must be an absolute path " +
        "(POSIX `/...` or Windows `[A-Z]:/...`) or a top-level Verilog `parameter`. " +
        "Verilator runs from a temp build dir; relative paths silently fail and the " +
        "ROM stays zero, producing wrong-but-plausible numerics. " +
        `Found: ${readmemhRelativeSnippets.slice(0, 3).join("; ")}.`,
    });
  }

  // Rule 3e: dram-backed-weights and other multi-vector contracts must NOT
  // make ST_DONE (or any equivalent terminal state) lock the FSM. The
  // testbench drives multiple input vectors; a terminal state holds
  // ready_in=0 forever and stalls vector N+1 input. Accept the rule as
  // contract-gated since flat-bus modules are single-vector by convention.
  const multiVectorContract =
    currentContractId(layer) === "dram-backed-weights" ||
    currentContractId(layer) === "tiled-streaming" ||
    currentContractId(layer) === "activation-double-buffering" ||
    currentContractId(layer) === "weight-tiling";
  if (multiVectorContract) {
    const terminalDoneLockSnippets = forbiddenTerminalDoneLockSnippets(source);
    if (terminalDoneLockSnippets.length > 0) {
      violations.push({
        rule: "dram_backed_weights_terminal_done_lock",
        detail:
          "Multi-vector contracts (dram-backed-weights, tiled-streaming, " +
          "activation-double-buffering, weight-tiling) test multiple input " +
          "vectors per simulation. A terminal ST_DONE that holds " +
          "`ready_in <= 1'b0` and stays in itself locks out vector N+1's input. " +
          "ST_DONE must reset per-vector state and transition back to " +
          "ST_INIT_BOOT (or equivalent) so vector N+1 starts cleanly. " +
          `Found: ${terminalDoneLockSnippets.slice(0, 3).join("; ")}.`,
      });
    }
  }

  // Rule 4: weights and biases must use $readmemh. Skipped when the
  // top-level instantiates `conv_datapath`: that library module owns the
  // weight/bias arrays and their $readmemh loaders, driven by WEIGHTS_PATH /
  // BIAS_PATH module parameters. Accept the current flat single-array form
  // `$readmemh(..., weights)` and the future banked form
  // `$readmemh(..., weights_bank<N>)`.
  const externalWeightContract =
    currentContractId(layer) === "dram-backed-weights" ||
    currentContractId(layer) === "weight-tiling";
  if (layer.op_type === "conv2d" && !usesConvDatapath && !externalWeightContract) {
    const readmemhWeightsRe = /\$readmemh\s*\(\s*"[^"]*"\s*,\s*weights(?:_bank\d+)?\s*\)/;
    if (!readmemhWeightsRe.test(source)) {
      violations.push({
        rule: "readmemh_missing",
        detail:
          "Weights must be loaded via $readmemh(\"<weights_path>\", weights) " +
          "or $readmemh(\"<bank_path>\", weights_bank<N>) inside an initial " +
          "block, OR the top-level must instantiate conv_datapath with " +
          "WEIGHTS_PATH/BIAS_PATH parameters. None was found.",
      });
    }
    if (layer.bias_path) {
      const readmemhBiasesRe = /\$readmemh\s*\(\s*"[^"]*"\s*,\s*biases\s*\)/;
      if (!readmemhBiasesRe.test(source)) {
        violations.push({
          rule: "readmemh_missing",
          detail:
            "Biases must be loaded via $readmemh(\"<bias_path>\", biases) " +
            "or via conv_datapath's BIAS_PATH parameter. Neither was found.",
        });
      }
    }
  }

  // Rule 5: output counter / completion guard must exist — but ONLY for
  // ops where the input-to-output mapping is NOT 1:1-per-input-pixel.
  // Spatial conv (KH*KW > 1) and maxpool both traverse padding regions
  // without consuming real input, so without an `outputs_emitted` bound
  // the FSM can emit an unbounded valid_out stream. Pointwise 1x1 conv,
  // add, and relu are all 1:1 with input and terminate naturally via
  // the per-pixel FSM (oc_group / K_TOTAL exhaustion) — forcing a
  // frame-level counter on them BREAKS back-to-back frames by latching
  // the FSM into a terminal state after the first frame.
  const needsFrameCounter =
    isSpatialConv || layer.op_type === "maxpool";
  if (needsFrameCounter) {
    const counterRe = /\breg\s+(?:signed\s+)?(?:\[[^\]]+\]\s+)?(?:out_row|out_col|outputs_emitted)\b/;
    // coord_scheduler exports its own `outputs_emitted` as an output port,
    // so a module that wires its own `outputs_emitted` reg to the
    // scheduler's `outputs_emitted` output satisfies the bounded-counter
    // invariant even if the reg shape varies. The reg-declaration regex
    // above captures the canonical hand-written case; the check below
    // additionally accepts an instantiation of coord_scheduler.
    const coordSchedulerInstantiated = /\bcoord_scheduler\b/.test(source);
    if (!counterRe.test(source) && !coordSchedulerInstantiated) {
      violations.push({
        rule: "output_counter_missing",
        detail:
          "Spatial conv / maxpool must bound its output count with either " +
          "an `outputs_emitted` reg (or `out_row` / `out_col`) or a " +
          "coord_scheduler instantiation. Without a bounded counter the " +
          "FSM has no frame-level stop condition and Verilator can hang " +
          "on partial valid_out firings.",
      });
    }
  }

  // Rule 6: spatial conv / maxpool must instantiate `coord_scheduler`. The
  // coordinate/wrap/stride/termination math is the single most bug-prone
  // piece of the pipeline; the handwritten module in rtl_library/ is its
  // authoritative implementation.
  const needsCoordScheduler =
    isSpatialConv || layer.op_type === "maxpool";
  if (needsCoordScheduler) {
    if (!/\bcoord_scheduler\b/.test(source)) {
      violations.push({
        rule: "coord_scheduler_missing",
        detail:
          "Spatial conv / maxpool modules must instantiate coord_scheduler " +
          "from rtl_library/coord_scheduler.v. No coord_scheduler reference " +
          "was found in the RTL. Rolling your own coordinate/wrap/stride/" +
          "termination logic is the historically most bug-prone piece of " +
          "the pipeline — do not reinvent it.",
      });
    }
  }

  return violations;
}

const SYNTH_PREFLIGHT_SCALAR_MEMORY_CELL_THRESHOLD = 16_384;
// Vivado errors with [Synth 8-4556] on any single unpacked reg variable
// whose total bits exceed ~1,048,576 (= 2^20). We cap at 900 Kb to leave
// margin for parameterised dimension drift and to keep one bank fitting
// in a clean LUT-RAM / BRAM mapping. Larger storage MUST be split into
// multiple smaller arrays (banking).
const SYNTH_PREFLIGHT_PER_VARIABLE_BIT_LIMIT = 900_000;
// Minimum element width that counts as a "wide-word" entry (i.e. one
// addressable cell holds a packed beat, not a single byte). Byte-cell
// arrays are caught by `largeScalarizedActivationMemoryViolations` and
// must not also fire the per-variable-bit-limit rule.
const SYNTH_PREFLIGHT_WIDEWORD_ELEMENT_BITS_MIN = 32;
// Total-bits threshold above which a 2D-unpacked activation memory
// stops being acceptable: Vivado refuses to infer LUT-RAM for
// 2D-unpacked × wide-packed arrays, and the resulting FF mapping
// cripples post-synth `report_timing_summary -check_timing_verbose`.
// Below this, FF mapping is tolerable (post-synth analysis still
// completes in seconds-to-minutes — proven by conv_292's 200k-FF
// pass that finished in ~7 min total wall).
const SYNTH_PREFLIGHT_MULTIDIM_FF_TOLERATED_BITS = 300_000;

function constantMapForLayer(layer: LayerIR): Map<string, number> {
  const ic = layer.input_shape[1] ?? 1;
  const oc = layer.output_shape[1] ?? 1;
  const ih = layer.input_shape[2] ?? 1;
  const iw = layer.input_shape[3] ?? 1;
  const oh = layer.output_shape[2] ?? 1;
  const ow = layer.output_shape[3] ?? 1;
  const kh = layer.weight_shape[2] ?? 1;
  const kw = layer.weight_shape[3] ?? 1;
  const channelTile = layer.channel_tile ?? Math.max(1, Math.floor(layer.input_width_bits / 8));
  const constants = new Map<string, number>();
  const add = (name: string, value: number): void => {
    if (Number.isFinite(value)) {
      constants.set(name, value);
      constants.set(name.toUpperCase(), value);
    }
  };

  add("IC", ic);
  add("OC", oc);
  add("IH", ih);
  add("IW", iw);
  add("OH", oh);
  add("OW", ow);
  add("KH", kh);
  add("KW", kw);
  add("MP", layer.mac_parallelism ?? 1);
  add("K_TOTAL", ic * kh * kw);
  add("KH_KW", kh * kw);
  add("TOTAL_IN_PIXELS", ih * iw);
  add("TOTAL_OUT_PIXELS", oh * ow);
  add("CHANNEL_TILE", channelTile);
  add("BEAT_BITS", layer.input_width_bits);
  add("INPUT_WIDTH_BITS", layer.input_width_bits);
  add("OUTPUT_WIDTH_BITS", layer.output_width_bits);
  add("IN_BEATS", Math.max(1, Math.ceil(ic / channelTile)));
  add("OUT_BEATS", Math.max(1, Math.ceil(oc / channelTile)));
  return constants;
}

function tokenizeConstantExpr(expr: string): string[] | null {
  // Strip Verilog's numeric-literal underscores (1_000 -> 1000) but
  // preserve underscores inside identifiers (BEAT_BITS, K_TOTAL, etc.).
  const compact = expr.replace(/(\d)_(?=\d)/g, "$1").trim();
  const tokens: string[] = [];
  const re = /\s*([A-Za-z_][A-Za-z0-9_$]*|\d+|[()+\-*/])\s*/gy;
  let cursor = 0;
  while (cursor < compact.length) {
    re.lastIndex = cursor;
    const match = re.exec(compact);
    if (!match) return null;
    tokens.push(match[1]);
    cursor = re.lastIndex;
  }
  return tokens;
}

function evaluateConstantExpr(expr: string, constants: Map<string, number>): number | null {
  const tokens = tokenizeConstantExpr(expr);
  if (!tokens) return null;
  const tokenList = tokens;
  let cursor = 0;

  const parseFactor = (): number | null => {
    const token = tokenList[cursor];
    if (token === undefined) return null;
    if (token === "+") {
      cursor += 1;
      return parseFactor();
    }
    if (token === "-") {
      cursor += 1;
      const value = parseFactor();
      return value === null ? null : -value;
    }
    if (token === "(") {
      cursor += 1;
      const value = parseExpr();
      if (value === null || tokenList[cursor] !== ")") return null;
      cursor += 1;
      return value;
    }
    if (/^\d+$/.test(token)) {
      cursor += 1;
      return Number(token);
    }
    const value = constants.get(token) ?? constants.get(token.toUpperCase());
    if (value === undefined) return null;
    cursor += 1;
    return value;
  };

  const parseTerm = (): number | null => {
    let value = parseFactor();
    if (value === null) return null;
    while (tokenList[cursor] === "*" || tokenList[cursor] === "/") {
      const op = tokenList[cursor];
      cursor += 1;
      const rhs = parseFactor();
      if (rhs === null) return null;
      if (op === "*") {
        value *= rhs;
      } else {
        if (rhs === 0) return null;
        value = Math.trunc(value / rhs);
      }
    }
    return value;
  };

  function parseExpr(): number | null {
    let value = parseTerm();
    if (value === null) return null;
    while (tokenList[cursor] === "+" || tokenList[cursor] === "-") {
      const op = tokenList[cursor];
      cursor += 1;
      const rhs = parseTerm();
      if (rhs === null) return null;
      value = op === "+" ? value + rhs : value - rhs;
    }
    return value;
  }

  const value = parseExpr();
  if (value === null || cursor !== tokenList.length || !Number.isFinite(value)) return null;
  return value;
}

function constantsForModule(source: string, layer: LayerIR): Map<string, number> {
  const constants = constantMapForLayer(layer);
  const localparamRe =
    /\b(?:localparam|parameter)\b(?:\s+(?:integer|signed))*\s+(?:\[[^\]]+\]\s*)?([A-Za-z_][A-Za-z0-9_$]*)\s*=\s*([^;]+);/g;

  for (let pass = 0; pass < 4; pass += 1) {
    let changed = false;
    let match: RegExpExecArray | null;
    localparamRe.lastIndex = 0;
    while ((match = localparamRe.exec(source)) !== null) {
      const name = match[1];
      if (constants.has(name)) continue;
      const value = evaluateConstantExpr(match[2], constants);
      if (value !== null) {
        constants.set(name, value);
        constants.set(name.toUpperCase(), value);
        changed = true;
      }
    }
    if (!changed) break;
  }

  return constants;
}

function extractRanges(text: string): string[] {
  return [...text.matchAll(/\[([^\]]+)\]/g)].map((match) => match[1]);
}

function rangeSize(range: string, constants: Map<string, number>): number | null {
  const colon = range.indexOf(":");
  if (colon === -1) return evaluateConstantExpr(range, constants);
  const left = evaluateConstantExpr(range.slice(0, colon), constants);
  const right = evaluateConstantExpr(range.slice(colon + 1), constants);
  if (left === null || right === null) return null;
  return Math.abs(left - right) + 1;
}

function rangeProduct(ranges: string[], constants: Map<string, number>, defaultValue = 1): number | null {
  if (ranges.length === 0) return defaultValue;
  let product = 1;
  for (const range of ranges) {
    const size = rangeSize(range, constants);
    if (size === null || size <= 0) return null;
    product *= size;
  }
  return product;
}

function largeScalarizedActivationMemoryViolations(
  source: string,
  layer: LayerIR,
): StructuralPreflightViolation[] {
  const violations: StructuralPreflightViolation[] = [];
  const constants = constantsForModule(source, layer);
  const memoryDeclRe =
    /\b(?:reg|logic)\b\s+(?:signed\s+)?((?:\[[^\]]+\]\s*)*)([A-Za-z_][A-Za-z0-9_$]*)\s*((?:\[[^\]]+\]\s*)+)\s*;/g;
  const activationMemoryNameRe =
    /(?:^|_)(?:line_?buf|activation|feature|pixel|frame|input|in_?buf|act_?buf)(?:_|$)/i;
  let match: RegExpExecArray | null;

  while ((match = memoryDeclRe.exec(source)) !== null) {
    const packedRanges = extractRanges(match[1]);
    const name = match[2];
    const unpackedRanges = extractRanges(match[3]);
    if (!activationMemoryNameRe.test(name) || unpackedRanges.length < 2) continue;

    const elementWidthBits = rangeProduct(packedRanges, constants, 1);
    const entryCount = rangeProduct(unpackedRanges, constants, 1);
    if (elementWidthBits === null || entryCount === null) continue;
    if (elementWidthBits > 16 || entryCount < SYNTH_PREFLIGHT_SCALAR_MEMORY_CELL_THRESHOLD) continue;

    violations.push({
      rule: "large_scalarized_activation_memory",
      detail:
        `Memory '${name}' is declared as a large scalarized activation buffer ` +
        `(element_width_bits=${elementWidthBits}, entries=${entryCount}, unpacked_dims=${unpackedRanges.length}). ` +
        "This shape creates thousands of independently addressable byte/word cells and wide mux fabric that can make " +
        "Vivado synthesis explode in runtime/RSS. Use a RAM-inferable packed beat/pixel memory with synchronous reads " +
        "instead of a per-channel scalar cell array.",
    });
  }

  return violations;
}

function activationMemoryBitLimitViolations(
  source: string,
  layer: LayerIR,
): StructuralPreflightViolation[] {
  const violations: StructuralPreflightViolation[] = [];
  const constants = constantsForModule(source, layer);
  const memoryDeclRe =
    /\b(?:reg|logic)\b\s+(?:signed\s+)?((?:\[[^\]]+\]\s*)*)([A-Za-z_][A-Za-z0-9_$]*)\s*((?:\[[^\]]+\]\s*)+)\s*;/g;
  // Match line_buf / activation / window-style names AND banked variants
  // (line_buf_b0, in_buf_bank3, etc.) so a partially-banked memory doesn't
  // skate by because one bank is still oversized.
  const activationMemoryNameRe =
    /(?:^|_)(?:line_?buf|activation|feature|pixel|frame|input|in_?buf|act_?buf|window)(?:_|$|\d|b)/i;
  let match: RegExpExecArray | null;

  while ((match = memoryDeclRe.exec(source)) !== null) {
    const packedRanges = extractRanges(match[1]);
    const name = match[2];
    const unpackedRanges = extractRanges(match[3]);
    if (!activationMemoryNameRe.test(name) || unpackedRanges.length === 0) continue;

    const elementWidthBits = rangeProduct(packedRanges, constants, 1);
    if (elementWidthBits === null || elementWidthBits < SYNTH_PREFLIGHT_WIDEWORD_ELEMENT_BITS_MIN) continue;

    const unpackedSizes: number[] = [];
    let totalEntries = 1;
    let unresolved = false;
    for (const range of unpackedRanges) {
      const size = rangeSize(range, constants);
      if (size === null || size <= 0) { unresolved = true; break; }
      unpackedSizes.push(size);
      totalEntries *= size;
    }
    if (unresolved) continue;

    const totalBits = totalEntries * elementWidthBits;
    if (totalBits <= SYNTH_PREFLIGHT_PER_VARIABLE_BIT_LIMIT) continue;

    // Compute the bank shape we'd recommend. Prefer 64-deep banks (clean
    // LUT-RAM granule); if a 64-deep bank would still bust the cap because
    // the inner dimensions are huge, drop to 32. The rule below is
    // architecture-agnostic — same shape for any conv/op layer.
    const innerDims = unpackedSizes.slice(1);
    const innerEntries = innerDims.reduce((acc, dim) => acc * dim, 1);
    const bitsPerOuterEntry = innerEntries * elementWidthBits;
    let preferredBankDepth = 64;
    if (preferredBankDepth * bitsPerOuterEntry > SYNTH_PREFLIGHT_PER_VARIABLE_BIT_LIMIT) {
      preferredBankDepth = 32;
    }
    const logicalDepth = unpackedSizes[0];
    const bankCount = Math.ceil(logicalDepth / preferredBankDepth);
    const bitsPerBank = preferredBankDepth * bitsPerOuterEntry;

    violations.push({
      rule: "activation_memory_exceeds_vivado_variable_bit_limit",
      detail:
        `Memory '${name}' is a single unpacked reg variable totalling ${totalBits} bits ` +
        `(depth=${logicalDepth}, inner=${innerDims.join("x") || "1"}, element_width_bits=${elementWidthBits}). ` +
        `Vivado rejects any single variable above ${SYNTH_PREFLIGHT_PER_VARIABLE_BIT_LIMIT} bits (hard limit ~1,048,576 with [Synth 8-4556]). ` +
        "Bank the memory into multiple smaller variables, each well under the cap. " +
        `Recommended: ${bankCount} bank(s) of depth ${preferredBankDepth} ` +
        `(bits_per_bank=${bitsPerBank}). ` +
        "Sketch:\n" +
        Array.from({ length: bankCount }, (_, i) =>
          `  reg [${elementWidthBits - 1}:0] ${name}_b${i} [0:${preferredBankDepth - 1}]${innerDims.map((d) => `[0:${d - 1}]`).join("")};`,
        ).join("\n") +
        `\n  bank_idx = pixel_index / ${preferredBankDepth};` +
        `\n  bank_addr = pixel_index % ${preferredBankDepth};` +
        "\nDo NOT collapse the buffer further than the layer's logical pixel count; each bank just holds a contiguous slice.",
    });
  }

  return violations;
}

function multiDimWideWordActivationMemoryViolations(
  source: string,
  layer: LayerIR,
): StructuralPreflightViolation[] {
  const constants = constantsForModule(source, layer);
  const memoryDeclRe =
    /\b(?:reg|logic)\b\s+(?:signed\s+)?((?:\[[^\]]+\]\s*)*)([A-Za-z_][A-Za-z0-9_$]*)\s*((?:\[[^\]]+\]\s*)+)\s*;/g;
  // Match the line_buf-family names plus banked variants. Window and
  // weight-style memories are NOT in scope: window is the dummy
  // sliding-window register (small, dead in dram-backed), weight ROMs
  // ARE allowed to be 2D unpacked because they're rarely read in this
  // contract — the AXI cache holds the read-hot copy.
  const activationMemoryNameRe =
    /(?:^|_)(?:line_?buf|activation|feature|pixel|frame|input|in_?buf|act_?buf)(?:_|$|\d|b)/i;
  let match: RegExpExecArray | null;

  type Offender = {
    name: string;
    elementWidthBits: number;
    unpackedSizes: number[];
    totalBits: number;
  };
  const offenders: Offender[] = [];

  while ((match = memoryDeclRe.exec(source)) !== null) {
    const packedRanges = extractRanges(match[1]);
    const name = match[2];
    const unpackedRanges = extractRanges(match[3]);
    if (!activationMemoryNameRe.test(name) || unpackedRanges.length < 2) continue;

    const elementWidthBits = rangeProduct(packedRanges, constants, 1);
    if (elementWidthBits === null || elementWidthBits < SYNTH_PREFLIGHT_WIDEWORD_ELEMENT_BITS_MIN) continue;

    const unpackedSizes: number[] = [];
    let unresolved = false;
    let totalEntries = 1;
    for (const range of unpackedRanges) {
      const size = rangeSize(range, constants);
      if (size === null || size <= 0) { unresolved = true; break; }
      unpackedSizes.push(size);
      totalEntries *= size;
    }
    if (unresolved) continue;

    offenders.push({
      name,
      elementWidthBits,
      unpackedSizes,
      totalBits: totalEntries * elementWidthBits,
    });
  }

  if (offenders.length === 0) return [];

  // The cost is the aggregate FF mapping across every multi-dim
  // activation memory in this module. One small array is tolerable
  // (conv_292 with ~262k bits synthesised in ~7 min total). Many small
  // arrays summing >300k bits is the conv_284 pathology that stalls
  // post-synth analysis. Compare the SUM, not the max.
  const aggregateBits = offenders.reduce((acc, o) => acc + o.totalBits, 0);
  if (aggregateBits <= SYNTH_PREFLIGHT_MULTIDIM_FF_TOLERATED_BITS) return [];

  // Build one violation per offender so each banked variant gets its
  // own concrete fix suggestion.
  return offenders.map<StructuralPreflightViolation>((o) => {
    const outerDepth = o.unpackedSizes[0];
    const innerDims = o.unpackedSizes.slice(1);
    const innerEntries = innerDims.reduce((acc, dim) => acc * dim, 1);
    const collapsedPackedBits = innerEntries * o.elementWidthBits;
    return {
      rule: "multidim_wideword_activation_memory",
      detail:
        `Memory '${o.name}' is a wide-word activation buffer with 2D+ unpacked dims ` +
        `(element_width_bits=${o.elementWidthBits}, unpacked=[${o.unpackedSizes.join(",")}], total_bits=${o.totalBits}). ` +
        `Aggregate across all such buffers in this module is ${aggregateBits} bits, ` +
        `above the FF-mapping-tolerable threshold of ${SYNTH_PREFLIGHT_MULTIDIM_FF_TOLERATED_BITS}. ` +
        "Vivado refuses to infer distributed LUT-RAM for multi-dim unpacked wide-word arrays and instead " +
        "FF-maps the entire memory, which cripples post-synth report_timing_summary " +
        "-check_timing_verbose wall time without changing PPA. " +
        "Collapse the inner unpacked dimension(s) into the packed width so the array becomes 1D unpacked × " +
        "wide packed — exactly the shape that the AXI prefetch cache_a / cache_b use and that Vivado maps " +
        "cleanly to RAM64M8 primitives. " +
        "Recommended:\n" +
        `  reg [${collapsedPackedBits - 1}:0] ${o.name} [0:${outerDepth - 1}];\n` +
        `  // write one inner slice: ${o.name}[addr][beat*${o.elementWidthBits} +: ${o.elementWidthBits}] <= narrow_data;\n` +
        `  // read whole word, then bit-select downstream: word_q1 = ${o.name}[addr]; narrow_q2 = word_q1[beat*${o.elementWidthBits} +: ${o.elementWidthBits}];`,
    };
  });
}

function activationMemoryWithAsyncResetViolations(
  source: string,
  _layer: LayerIR,
): StructuralPreflightViolation[] {
  // Vivado's RAM-inference engine refuses to map a reg array to BRAM /
  // distributed RAM if the array is written inside an always block that
  // ALSO has an async reset edge (`always @(posedge clk or negedge rst_n)`).
  // The failure shape is [Synth 8-4767] then [Synth 8-3391] when the
  // dissolved-bit count exceeds the elaboration cap, hard-failing
  // synth_design in ~8 seconds.
  //
  // This rule scans every async-reset always block and reports any
  // activation-memory name written inside it. The check is conservative:
  // a memory's NAME is matched against the activation-memory regex (same
  // family the other gates use), and the violation only fires if there's
  // a non-blocking assignment to `name[...]` inside the async-reset block.
  const violations: StructuralPreflightViolation[] = [];
  const activationMemoryNameRe =
    /(?:^|_)(?:line_?buf|activation|feature|pixel|frame|input|in_?buf|act_?buf)(?:_|$|\d|b)/i;

  // Use a tolerant matcher for always blocks that consumes nested begin/end.
  // We don't try to perfectly parse SystemVerilog — just slice from the
  // `always @(...)` header to the next top-level `endmodule` or next
  // `always @` (whichever comes first), then look for writes inside.
  const alwaysHeaderRe =
    /always\s*@\s*\(([^)]*)\)\s*begin/g;
  let match: RegExpExecArray | null;
  while ((match = alwaysHeaderRe.exec(source)) !== null) {
    const sensitivity = match[1];
    // Async-reset sensitivity has `negedge` (or `posedge`) on a non-clock
    // signal, paired with the main clock edge. We detect by presence of
    // BOTH `posedge` AND `negedge` (the canonical pattern Vivado checks
    // against). `always @(posedge clk)` alone is fine.
    if (!/\bposedge\b/.test(sensitivity) || !/\bnegedge\b/.test(sensitivity)) continue;

    // Slice the body up to the next `always @` header or end of file.
    const start = match.index + match[0].length;
    const nextAlways = source.indexOf("always", start);
    const bodyEnd = nextAlways === -1 ? source.length : nextAlways;
    const body = source.slice(start, bodyEnd);

    // Find every non-blocking assignment to `name[...]` and check the
    // name against the activation-memory regex. Skip names already
    // recognised as non-memory (the test is on the name, not the index).
    const writeRe =
      /\b([A-Za-z_][A-Za-z0-9_$]*)\s*\[[^;]*<=/g;
    const offenders = new Set<string>();
    let w: RegExpExecArray | null;
    while ((w = writeRe.exec(body)) !== null) {
      const name = w[1];
      if (activationMemoryNameRe.test(name)) {
        offenders.add(name);
      }
    }
    if (offenders.size === 0) continue;

    for (const name of offenders) {
      violations.push({
        rule: "activation_memory_in_async_reset_block",
        detail:
          `Memory '${name}' is written inside an always block with async reset ` +
          `(sensitivity = '${sensitivity.trim()}'). Vivado refuses RAM inference for ` +
          "reset-sensitive memories and emits [Synth 8-4767] then hard-fails with " +
          "[Synth 8-3391] because the dissolved-bit fallback exceeds the elaboration cap. " +
          "Move the memory writes to a dedicated `always @(posedge clk) begin ... end` block " +
          "with no reset clause — the cache_a / cache_b AXI prefetch arrays in this same " +
          "contract use exactly that pattern and Vivado maps them to RAM64M8 primitives. " +
          "The FSM state, counters, and handshake regs stay in the original async-reset block; " +
          "only the memory writes move out. Pre-fill of the memory by the input-streaming path " +
          "before the MAC reads from it is sufficient — no power-on reset of memory is needed.",
      });
    }
  }

  return violations;
}

export function synthesisPreflightViolations(
  module: VerilogModule,
  layer: LayerIR,
): StructuralPreflightViolation[] {
  const source = stripVerilogComments(module.verilog_source);
  return [
    ...largeScalarizedActivationMemoryViolations(source, layer),
    ...activationMemoryBitLimitViolations(source, layer),
    ...multiDimWideWordActivationMemoryViolations(source, layer),
    ...activationMemoryWithAsyncResetViolations(source, layer),
  ];
}

function synthesisPreflightMessage(violations: StructuralPreflightViolation[]): string {
  const rules = violations.map((v) => v.rule).join(", ");
  return [
    "Deterministic synthesis preflight rejected the RTL before Vivado.",
    `Violated rule(s): ${rules}.`,
    "Functional simulation may pass, but this structural shape is known to cause pathological Vivado runtime/RSS.",
    "Repair the indicted storage organization; do not change the public interface or functional behavior.",
    "Violations:",
    ...violations.map((v) => `- [${v.rule}] ${v.detail}`),
  ].join("\n");
}

export function synthesisPreflightReport(
  module: VerilogModule,
  layer: LayerIR,
  violations = synthesisPreflightViolations(module, layer),
): SynthesisReport {
  const report = synthesisPreflightMessage(violations);
  return {
    success: false,
    tool: "vivado",
    part: "xczu9eg-ffvb1156-2-e",
    stage: "synth",
    lut_count: 0,
    ff_count: 0,
    dsp_count: 0,
    bram18_count: 0,
    bram36_count: 0,
    bram18_equiv: 0,
    wns_ns: null,
    setup_wns_ns: null,
    hold_wns_ns: null,
    timing_met: false,
    fmax_mhz: 0,
    report,
  };
}

function synthesisPreflightAsVerifResult(
  moduleId: string,
  verifiedResult: VerifResult,
  violations: StructuralPreflightViolation[],
): VerifResult {
  const rules = violations.map((v) => v.rule).join(",");
  const message = synthesisPreflightMessage(violations);
  return {
    ...verifiedResult,
    module_id: moduleId,
    status: "fail",
    failure_class: "structural_preflight_failed",
    failure_category: "code_bug",
    violated_constraint: `synthesis_preflight:${rules}`,
    fix_hint: message,
    classifier_reason:
      "Deterministic synthesis preflight caught a Vivado-pathological storage shape before running synthesis.",
  };
}

function normalizedExpr(text: string): string {
  return text.replace(/\s+/g, "").toLowerCase();
}

function fullWeightElementCount(layer: LayerIR): number {
  if (Number.isInteger(layer.num_weights) && layer.num_weights > 0) {
    return layer.num_weights;
  }
  return layer.weight_shape.reduce((acc, value) => acc * value, 1);
}

function isFullWeightRangeExpr(expr: string, layer: LayerIR): boolean {
  const n = fullWeightElementCount(layer);
  const compact = normalizedExpr(expr);
  return (
    compact === `${n - 1}` ||
    compact === `${n}-1` ||
    compact === "num_weights-1" ||
    compact === "oc*k_total-1" ||
    compact === "k_total*oc-1" ||
    compact === "(oc*k_total)-1" ||
    compact === "(k_total*oc)-1"
  );
}

function fullWeightMemoryDeclarations(source: string, layer: LayerIR): Array<{ name: string; rangeExpr: string }> {
  const declarations: Array<{ name: string; rangeExpr: string }> = [];
  const memoryDeclRe =
    /\b(?:reg|logic)\b[^;]*?\b([A-Za-z_][A-Za-z0-9_$]*)\s*\[\s*0\s*:\s*([^\]]+)\]\s*;/g;
  let match: RegExpExecArray | null;
  while ((match = memoryDeclRe.exec(source)) !== null) {
    const name = match[1];
    const rangeExpr = match[2];
    if (/weight/i.test(name) && isFullWeightRangeExpr(rangeExpr, layer)) {
      declarations.push({ name, rangeExpr });
    }
  }
  return declarations;
}

function hasBareFullWeightReadmemh(source: string): boolean {
  return /\$readmemh\s*\(\s*[^,]+,\s*weights\s*\)\s*;/i.test(source);
}

function hasTiedOffWeightsArvalid(source: string): boolean {
  return /\bassign\s+weights_arvalid\s*=\s*(?:1\s*'\s*[bhd]\s*0|[0]+)\s*;/i.test(source);
}

function hasFakeCoordSchedulerValidIn(source: string): boolean {
  return /\bcoord_scheduler\b[\s\S]*?\.valid_in\s*\(\s*(?:1\s*'\s*[bhd]\s*0|[0]+)\s*\)/i.test(source);
}

function hasBeatCounterDeclaration(source: string): boolean {
  return /\b(?:reg|logic|integer)\b[^;]*\b[A-Za-z_][A-Za-z0-9_$]*beat[A-Za-z0-9_$]*\b[^;]*;/i.test(source);
}

export function contractConformanceViolations(
  module: VerilogModule,
  layer: LayerIR,
): StructuralPreflightViolation[] {
  const violations: StructuralPreflightViolation[] = [];
  const source = stripVerilogComments(module.verilog_source);
  const contractId = currentContractId(layer);

  if (contractId === "flat-bus") {
    return violations;
  }

  const fullWeightDecls = fullWeightMemoryDeclarations(source, layer);

  if (contractId === "dram-backed-weights") {
    if (hasTiedOffWeightsArvalid(source)) {
      violations.push({
        rule: "contract_dram_weights_arvalid_tied_off",
        detail:
          "dram-backed-weights requires a real AXI read-address FSM. " +
          "Found `assign weights_arvalid = 0`, which ties off external weight reads and reverts to an on-chip-weight design.",
      });
    }
    if (fullWeightDecls.length > 0) {
      violations.push({
        rule: "contract_dram_full_weight_array",
        detail:
          "dram-backed-weights may not allocate the full OC*K_TOTAL weight tensor on chip. " +
          `Found full-weight memory declaration(s): ${fullWeightDecls
            .map((decl) => `${decl.name}[0:${decl.rangeExpr}]`)
            .join(", ")}. Use an AXI-backed prefetch/cache window instead.`,
      });
    }
    if (hasBareFullWeightReadmemh(source)) {
      violations.push({
        rule: "contract_dram_full_weight_readmemh",
        detail:
          "dram-backed-weights may not load the entire layer via `$readmemh(..., weights)`. " +
          "The full weight tensor must be external-memory-backed; simulation may use the AXI memory model, not a full on-chip ROM.",
      });
    }
  }

  if (contractId === "weight-tiling") {
    if (hasFakeCoordSchedulerValidIn(source)) {
      violations.push({
        rule: "contract_weight_tiling_fake_scheduler",
        detail:
          "weight-tiling may not satisfy the spatial-conv scheduler invariant with `.valid_in(1'b0)`. " +
          "Wire coord_scheduler to the real frame/compute advance path, or remove the fake instantiation and use the shared scheduler correctly.",
      });
    }
    if (fullWeightDecls.length > 0) {
      violations.push({
        rule: "contract_weight_tiling_full_active_tile",
        detail:
          "weight-tiling active storage may not be sized as the full OC*K_TOTAL weight tensor. " +
          `Found full-weight tile/memory declaration(s): ${fullWeightDecls
            .map((decl) => `${decl.name}[0:${decl.rangeExpr}]`)
            .join(", ")}. Store only the active weight tile plus partial accumulators.`,
      });
    }
  }

  if (contractId === "tiled-streaming") {
    const channelTile = layer.channel_tile;
    if (!channelTile || channelTile <= 0) {
      violations.push({
        rule: "contract_tiled_streaming_missing_tile",
        detail:
          "tiled-streaming requires LayerIR.channel_tile so the bus contract is deterministic. " +
          "Regenerate/select the contract with a positive channel_tile.",
      });
    } else {
      const expectedInputWidth = tiledInputBusWidthBits(layer, channelTile);
      const expectedOutputWidth = tiledOutputBusWidthBits(channelTile);
      if (layer.input_width_bits !== expectedInputWidth || layer.output_width_bits !== expectedOutputWidth) {
        violations.push({
          rule: "contract_tiled_streaming_bus_width",
          detail:
            `tiled-streaming bus width must match channel_tile=${channelTile}: ` +
            `expected input_width_bits=${expectedInputWidth}, output_width_bits=${expectedOutputWidth}; ` +
            `LayerIR has input_width_bits=${layer.input_width_bits}, output_width_bits=${layer.output_width_bits}.`,
        });
      }
    }
    if (!hasBeatCounterDeclaration(source)) {
      violations.push({
        rule: "contract_tiled_streaming_beat_counter_missing",
        detail:
          "tiled-streaming RTL must declare beat counters for per-pixel input/output channel tiles " +
          "(for example in_beat_count/out_beat_count/cur_beat_stream). No beat counter declaration was found.",
      });
    }
  }

  return violations;
}

export function preflightVerilogModule(module: VerilogModule, layer: LayerIR): string[] {
  const issues: string[] = [];
  const expectedSpecHash = computeExpectedSpecHash(layer);

  if (module.module_id !== layer.module_id) {
    issues.push(
      `VerilogModule.module_id='${module.module_id}' does not match LayerIR.module_id='${layer.module_id}'.`,
    );
  }
  if (module.spec_hash !== expectedSpecHash) {
    issues.push(
      `VerilogModule.spec_hash='${module.spec_hash}' does not match expected spec_hash='${expectedSpecHash}' for selected contract '${currentContractId(layer)}'.`,
    );
  }

  const portBlock = extractModulePortBlock(module.verilog_source, module.module_id);
  if (!portBlock) {
    issues.push(
      `Module '${module.module_id}' is missing a parseable ANSI-style top-level port list.`,
    );
    return issues;
  }

  const ports = parseAnsiTopPorts(portBlock);
  const contract = loadContractMetadata(resolveLayerContractId(layer));
  const expectedPorts = contract.interface_signals;
  for (const expected of expectedPorts) {
    const portName = expected.name;
    const parsed = ports.get(portName);
    if (!parsed) {
      issues.push(`Missing ${contract.name} top-level port '${portName}'.`);
      continue;
    }

    if (parsed.direction !== expected.direction) {
      issues.push(
        `Top-level port '${portName}' must be declared as ${expected.direction}, found ${parsed.direction} in '${parsed.declaration}'.`,
      );
    }

    const expectedWidth = evaluateContractPortWidthBits(expected.width_expr, expected.width_bits, layer);
    if (parsed.width_bits !== null && parsed.width_bits !== expectedWidth) {
      issues.push(
        `Top-level port '${portName}' declares width ${parsed.width_bits} bits, expected ${expectedWidth} bits from LayerIR.`,
      );
    }
  }

  return issues;
}

export async function ensureLayerIr(
  checkpointPath: string,
  runtime: OrchestratorRuntime = createOrchestratorRuntime(),
): Promise<{
  pipelineIr: PipelineIR;
  bootstrapUsage?: {
    total_cost_usd: number;
    modelUsage: Record<string, ModelUsageEntry>;
  };
}> {
  const layerIrPath = resolvePipelineConfigPath(PIPELINE_CONFIG.layer_ir_path);
  const layerIrFingerprintPath = `${layerIrPath}.checkpoint`;
  const checkpointAbs = resolveInputPathForCurrentHost(checkpointPath);

  if (await pathExists(layerIrPath)) {
    // Only reuse layer_ir.json if it was generated from the same checkpoint
    // the user is asking about now. A mismatch means a stale artifact from a
    // previous run and silently compiling it would yield nonsense.
    let fingerprintMatches = false;
    try {
      const prior = (await readFile(layerIrFingerprintPath, "utf8")).trim();
      fingerprintMatches = pathFingerprintKey(prior) === pathFingerprintKey(checkpointAbs);
    } catch {
      fingerprintMatches = false;
    }
    if (fingerprintMatches) {
      const pipelineIr = normalizePipelineIrForCurrentHost(
        await readJsonFile<PipelineIR>(layerIrPath, pipelineIrZod),
      );
      validateAddModulePacking(pipelineIr);
      return { pipelineIr };
    }
    throw new Error(
      `Stale output/layer_ir.json found (not tied to checkpoint '${checkpointAbs}'). ` +
        `Delete output/layer_ir.json (and output/pipeline_state.json) to rebuild from the new checkpoint.`,
    );
  }

  const payload = {
    checkpoint_path: checkpointPath,
    quantization_config: {
      quantization: "int8_symmetric_per_tensor",
    },
    output_path: layerIrPath,
  };

  await appendRunLog(
    {
      event: "action",
      action: "read_weights_deterministic",
      payload,
    },
    runtime,
  );

  const pipelineIr = normalizePipelineIrForCurrentHost(
    await runtime.readWeightsFn(checkpointAbs, payload.quantization_config),
  );

  await appendRunLog(
    {
      event: "cartographer_bypassed_deterministic_read_weights",
      checkpoint_path: checkpointAbs,
      layer_count: pipelineIr.layers.length,
    },
    runtime,
  );

  validateAddModulePacking(pipelineIr);
  await writeJsonFile(layerIrPath, pipelineIr);
  await writeFile(layerIrFingerprintPath, `${checkpointAbs}\n`, "utf8");
  return {
    pipelineIr,
  };
}

// Deterministic, LLM-free Vivado invocation. The previous design routed synth
// through query() with an allowedTool and let Claude mediate
// the tool call; that mediator could refuse for content-filter reasons and
// produced "I cannot comply" responses on modules with absolute host paths
// in $readmemh. Vivado is pure infrastructure — no reasoning needed — so it
// goes through the MCP tool impl directly, validated against the same
// synthesisReportSchema the SDK path used.
// Resolved as a runtime string so tsc does not analyze the target module
// (it lives in sibling package `mcp/`, outside this package's rootDir).
// When `sdk/` is compiled to `sdk/dist/`, we need the sibling `mcp/dist/`
// build; when running straight from source via tsx, we target the .ts file.
const MCP_TOOLS_MODULE_PATH = path.basename(__dirname) === "dist"
  ? pathToFileURL(path.resolve(repoRoot, "mcp", "dist", "tools.js")).href
  : pathToFileURL(path.resolve(repoRoot, "mcp", "tools.ts")).href;

function expectedLatencyCyclesForContract(layer: LayerIR, sidecarFields: Record<string, unknown>): number {
  const contractId = currentContractId(layer);
  const params = layer.contract_params ?? {};
  switch (contractId) {
    case "tiled-streaming":
      return layer.pipeline_latency_cycles + Math.max(1, Number(sidecarFields.beats_per_input_sample) || 1) - 1;
    case "dram-backed-weights":
      return (
        layer.pipeline_latency_cycles +
        (Number(params.weight_prefetch_latency_cycles) || 0) +
        (Number(params.prefetch_underrun_slack_cycles) || 0)
      );
    case "activation-double-buffering":
      return layer.pipeline_latency_cycles + (Number(params.activation_buffer_fill_cycles) || 0);
    case "weight-tiling": {
      const tileCount = Math.max(1, Number(params.weight_tile_count) || 1);
      const tileLoadLatency = Number(params.weight_tile_load_cycles) || 0;
      return layer.pipeline_latency_cycles * tileCount + tileLoadLatency * tileCount;
    }
    case "depthwise-conv":
      // Same public-interface latency contract as flat-bus: pipeline_latency_cycles
      // is the LayerIR-pinned time-to-first-output.
      return layer.pipeline_latency_cycles;
    case "flat-bus":
      return layer.pipeline_latency_cycles;
  }
}

export async function loadRetrospectorKnowledgeDoc(layer: LayerIR): Promise<RtlKnowledgeDoc> {
  const mcpTools = (await import(MCP_TOOLS_MODULE_PATH)) as {
    get_rtl_patterns: (
      op_type: string,
      kernel_h?: number,
      kernel_w?: number,
      contract_id?: ContractId,
      signature_hash?: string,
      exact_reference_key?: string | null,
      runtime_layer_signature?: Record<string, unknown>,
    ) => Promise<RtlKnowledgeDoc>;
  };
  const kernelH = layer.op_type === "conv2d" ? layer.weight_shape[2] : undefined;
  const kernelW = layer.op_type === "conv2d" ? layer.weight_shape[3] : undefined;
  const signatures = signatureMetadataForLayer(layer, layer);
  return mcpTools.get_rtl_patterns(
    layer.op_type,
    kernelH,
    kernelW,
    currentContractId(layer),
    signatures.signature_hash,
    signatures.exact_reference_key,
    signatures.runtime_layer_signature,
  );
}

async function invokeVivado(module: VerilogModule, layer: LayerIR): Promise<SynthesisReport> {
  const mcpTools = (await import(MCP_TOOLS_MODULE_PATH)) as {
    run_vivado: (
      verilog_source: string,
      module_name: string,
      clock_period_ns: number,
    ) => Promise<SynthesisReport>;
  };
  const raw = await mcpTools.run_vivado(
    module.verilog_source,
    module.module_id,
    layer.clock_period_ns,
  );
  const parsed = synthesisReportZod.safeParse(raw);
  if (!parsed.success) {
    throw new Error(
      `run_vivado returned invalid output:\n${JSON.stringify(parsed.error.issues, null, 2)}`,
    );
  }
  return parsed.data;
}

async function processSynthesisOutcome(
  manager: PipelineStateManager,
  moduleId: string,
  module: VerilogModule,
  layer: LayerIR,
  verifiedResult: VerifResult,
  statePath: string,
  runtime: OrchestratorRuntime,
  selfImproveEnabled = false,
): Promise<void> {
  const synthPreflightIssues = synthesisPreflightViolations(module, layer);
  if (synthPreflightIssues.length > 0) {
    const report = synthesisPreflightReport(module, layer, synthPreflightIssues);
    await writeJsonFile(reportPath(`${moduleId}.vivado.json`), report);
    const preflightFailure = synthesisPreflightAsVerifResult(
      moduleId,
      verifiedResult,
      synthPreflightIssues,
    );
    await appendRunLog(
      {
        event: "synthesis_preflight_failed",
        module_id: moduleId,
        rules: synthPreflightIssues.map((issue) => issue.rule),
        violations: synthPreflightIssues,
      },
      runtime,
    );
    await recordFailureAttempt(
      layer,
      "synthesis_preflight",
      preflightFailure,
      module,
      runtime,
      {
        synthesis_report: report.report,
        synthesis_metrics: {
          success: report.success,
          lut_count: report.lut_count,
          ff_count: report.ff_count,
          dsp_count: report.dsp_count,
          bram18_count: report.bram18_count,
          bram36_count: report.bram36_count,
          bram18_equiv: report.bram18_equiv,
          wns_ns: report.wns_ns,
          setup_wns_ns: report.setup_wns_ns,
          hold_wns_ns: report.hold_wns_ns,
          timing_met: report.timing_met,
          fmax_mhz: report.fmax_mhz,
        },
      },
    );
    if (selfImproveEnabled) {
      await archiveProbationaryDocsForFailure(
        layer,
        moduleId,
        "synthesis_preflight",
        runtime,
      );
    }
    const before = manager.getState().modules[moduleId];
    manager.applyVerifResult(moduleId, preflightFailure);
    const after = manager.getState().modules[moduleId];
    await logStateTransition(manager, moduleId, before, after, "synthesis_preflight_failed", runtime);
    await manager.saveState(statePath);
    return;
  }

  let report: SynthesisReport;
  try {
    report = await runtime.synthesisFn(module, layer);
  } catch (error: unknown) {
    // Tool itself crashed before producing a structured report. This is
    // infrastructure, not an RTL repair task, so fail-abort via tb_setup_error.
    const setupFailure: VerifResult = {
      ...verifiedResult,
      module_id: moduleId,
      status: "fail",
      status_class: "tb_setup_error",
      failure_class: null,
      fix_hint: [
        "Vivado failed before producing a structured synthesis report.",
        "This is infrastructure, not a module-local RTL bug. Check NN2RTL_VIVADO_BIN, PATH, and Windows/WSL path access.",
        error instanceof Error ? error.message : String(error),
      ].join("\n\n"),
    };
    const classifiedSetupFailure = await classifyFailedModule(
      manager,
      setupFailure,
      layer,
      module,
      runtime,
      { synthesis_tool_error: error instanceof Error ? error.stack ?? error.message : String(error) },
    );
    await recordFailureAttempt(
      layer,
      "vivado_tool_error",
      classifiedSetupFailure,
      module,
      runtime,
      { synthesis_tool_error: error instanceof Error ? error.stack ?? error.message : String(error) },
    );
    if (selfImproveEnabled) {
      await archiveProbationaryDocsForFailure(
        layer,
        moduleId,
        "vivado_tool_error",
        runtime,
      );
    }
    const before = manager.getState().modules[moduleId];
    manager.applyVerifResult(moduleId, classifiedSetupFailure);
    await logStateTransition(manager, moduleId, before, manager.getState().modules[moduleId], "vivado_tool_error", runtime);
    await manager.saveState(statePath);
    return;
  }

  await writeJsonFile(reportPath(`${moduleId}.vivado.json`), report);

  const synthesisFailure = evaluateSynthesis(moduleId, verifiedResult, report);
  if (!synthesisFailure) {
    // Genuine pass — RTL simulates correctly, synthesizes, and hits the PPA gates.
    await appendRunLog(
      {
        event: "vivado_pass",
        module_id: moduleId,
        lut_count: report.lut_count,
        ff_count: report.ff_count,
        dsp_count: report.dsp_count,
        bram18_equiv: report.bram18_equiv,
        fmax_mhz: report.fmax_mhz,
        // Surface both setup and hold WNS for visibility. Hold is not a
        // pass/fail signal at synth-only stage but a negative value tells
        // you the design will need P&R-stage attention (place_design /
        // opt_design hold-fixing) before bitstream.
        setup_wns_ns: report.setup_wns_ns ?? report.wns_ns,
        hold_wns_ns: report.hold_wns_ns,
      },
      runtime,
    );
    // Best-effort knowledge promotion: passes on contracts with pattern-doc
    // coverage but no protected reference seed a probationary reference for
    // future runs to crib from. Wrapped in try/catch so a promotion failure
    // never blocks a real pass from being recorded.
    try {
      await autoPromotePassingReference(layer, module, verifiedResult, report, runtime);
    } catch (err) {
      await appendRunLog(
        {
          event: "auto_promote_reference_error",
          module_id: moduleId,
          error: err instanceof Error ? err.message : String(err),
        },
        runtime,
      );
    }
    await manager.saveState(statePath);
    return;
  }

  const classifiedSynthesisFailure = await classifyFailedModule(
    manager,
    synthesisFailure,
    layer,
    module,
    runtime,
    {
      synthesis_report: report.report,
      synthesis_metrics: {
        success: report.success,
        lut_count: report.lut_count,
        ff_count: report.ff_count,
        dsp_count: report.dsp_count,
        bram18_count: report.bram18_count,
        bram36_count: report.bram36_count,
        bram18_equiv: report.bram18_equiv,
        wns_ns: report.wns_ns,
        setup_wns_ns: report.setup_wns_ns ?? report.wns_ns,
        hold_wns_ns: report.hold_wns_ns,
        timing_met: report.timing_met,
        fmax_mhz: report.fmax_mhz,
      },
    },
  );
  await recordFailureAttempt(
    layer,
    "vivado_synthesis",
    classifiedSynthesisFailure,
    module,
    runtime,
    {
      synthesis_report: report.report,
      synthesis_metrics: {
        success: report.success,
        lut_count: report.lut_count,
        ff_count: report.ff_count,
        dsp_count: report.dsp_count,
        bram18_count: report.bram18_count,
        bram36_count: report.bram36_count,
        bram18_equiv: report.bram18_equiv,
        wns_ns: report.wns_ns,
        setup_wns_ns: report.setup_wns_ns ?? report.wns_ns,
        hold_wns_ns: report.hold_wns_ns,
        timing_met: report.timing_met,
        fmax_mhz: report.fmax_mhz,
      },
    },
  );
  if (selfImproveEnabled) {
    await archiveProbationaryDocsForFailure(
      layer,
      moduleId,
      `vivado_${classifiedSynthesisFailure.failure_class ?? "fail"}`,
      runtime,
    );
  }
  const statusBeforeApply = manager.getState().modules[moduleId];
  manager.applyVerifResult(moduleId, classifiedSynthesisFailure);
  const statusAfterApply = manager.getState().modules[moduleId];
  await logStateTransition(
    manager,
    moduleId,
    statusBeforeApply,
    statusAfterApply,
    `vivado_${classifiedSynthesisFailure.failure_class ?? "fail"}`,
    runtime,
  );
  await manager.saveState(statePath);

  if (statusAfterApply === "fail_abort") {
    await appendRunLog(
      {
        event: "module_fail_abort",
        module_id: moduleId,
        result: classifiedSynthesisFailure,
      },
      runtime,
    );
  }
}

/**
 * Detect a Surgeon functional regression by comparing its VerifResult to
 * the prior one. "Regression" means: the prior result was better on at
 * least one dimension and the new result is not measurably better on any
 * dimension that matters. Specifically:
 *   - Timing: if prior had exact timing (actual == expected) and Surgeon's
 *     timing is now off, that's a clear regression. Surgeon is told
 *     explicitly never to change pipeline latency.
 *   - Error magnitude: a meaningful increase in max_error or mean_error
 *     (more than 10% or more than 8 INT8 LSBs) with no compensating
 *     improvement anywhere else.
 *   - Sample count drop: fewer outputs emitted than before means Surgeon
 *     broke an output path that was already producing.
 *   - First-mismatch index moved backward: prior had a correct prefix of
 *     length N before diverging; new result diverges earlier. Surgeon
 *     broke outputs that were already numerically correct (e.g. by
 *     reverting Foundry's rounding so pixels that were exact now drift).
 * Any single one of these is sufficient to treat as regression — the
 * prior module is at least as good on every dimension we care about.
 */
function isSurgeonRegression(prior: VerifResult, next: VerifResult): boolean {
  const priorTimingActual = prior.timing_actual_cycles ?? -1;
  const priorTimingExpected = prior.timing_expected_cycles ?? -1;
  const nextTimingActual = next.timing_actual_cycles ?? -1;
  const nextTimingExpected = next.timing_expected_cycles ?? priorTimingExpected;

  // Timing regression: prior was exact, new isn't.
  if (
    priorTimingActual >= 0 &&
    priorTimingExpected >= 0 &&
    priorTimingActual === priorTimingExpected &&
    nextTimingActual >= 0 &&
    nextTimingActual !== nextTimingExpected
  ) {
    return true;
  }

  // Error-magnitude regression. Compare only when both sides report a real
  // number (the testbench uses -1 as a sentinel for "no measurement").
  const priorMax = prior.max_error ?? -1;
  const nextMax = next.max_error ?? -1;
  const priorMean = prior.mean_error ?? -1;
  const nextMean = next.mean_error ?? -1;
  const meaningfulMaxIncrease =
    priorMax >= 0 && nextMax > priorMax && nextMax - priorMax >= 8;
  const meaningfulMeanIncrease =
    priorMean >= 0 && nextMean > priorMean * 1.1 && nextMean - priorMean >= 1.0;
  if (meaningfulMaxIncrease || meaningfulMeanIncrease) {
    return true;
  }

  // Sample-count drop (fewer outputs emitted). A non-stall run has a
  // fixed full sample_count; a smaller value means the run stalled earlier
  // than before.
  const priorSamples = prior.sample_count ?? -1;
  const nextSamples = next.sample_count ?? -1;
  if (priorSamples >= 0 && nextSamples >= 0 && nextSamples < priorSamples) {
    return true;
  }

  // First-mismatch index regression. If the prior had a meaningful correct
  // prefix (>=16 outputs exact before diverging) and Surgeon's new output
  // diverges meaningfully earlier (>=16 indices), Surgeon broke outputs
  // that were already correct. Observed in the wild: Foundry emits correct
  // scale-rounding so pixels 0..360 match exactly (first_mismatch=361);
  // Surgeon "fixes" the tail gap but reverts the rounding, so the same
  // first 361 pixels now drift by ±1 (first_mismatch=0). Mean_error barely
  // changes, so the error-magnitude check alone cannot catch this.
  const priorFirstMis = prior.first_mismatch_index ?? -1;
  const nextFirstMis = next.first_mismatch_index ?? -1;
  if (
    priorFirstMis >= 16 &&
    nextFirstMis >= 0 &&
    priorFirstMis - nextFirstMis >= 16
  ) {
    return true;
  }

  return false;
}

function summarizeVerifForLog(r: VerifResult): Record<string, unknown> {
  return {
    status: r.status,
    timing_pass: r.timing_pass,
    timing_actual_cycles: r.timing_actual_cycles,
    timing_expected_cycles: r.timing_expected_cycles,
    max_error: r.max_error,
    mean_error: r.mean_error,
    sample_count: r.sample_count,
    failure_class: r.failure_class,
  };
}

async function tryRecoverVerilogModuleFromDisk(
  layerIr: LayerIR,
  generatedBy: "Foundry" | "Surgeon",
  attempt: number,
): Promise<VerilogModule | null> {
  // Foundry/Surgeon sometimes return a bare path or plain text as their final
  // message even though they already wrote the .v via the write_verilog MCP
  // tool.  If the file is on disk we can reconstruct a VerilogModule and keep
  // the pipeline moving instead of failing the whole run.
  const rtlDir = resolvePipelineConfigPath(PIPELINE_CONFIG.rtl_dir);
  const verilogPath = path.join(rtlDir, `${layerIr.module_id}.v`);
  try {
    const source = await readFile(verilogPath, "utf8");
    if (!source.trim()) return null;
    return {
      module_id: layerIr.module_id,
      spec_hash: computeExpectedSpecHash(layerIr),
      verilog_source: source,
      generated_by: generatedBy,
      attempt,
    };
  } catch {
    return null;
  }
}

// Delete the on-disk Verilog + meta for a module. Called before a fresh
// Foundry attempt for a (module, contract) pair (attempt 1, including after
// resetModuleForContractRetry()) so that tryRecoverVerilogModuleFromDisk()
// cannot resurrect a stale .v from a prior contract or process. Reports and
// debug bundles in output/reports/ and output/debug/ are evidence and are
// NOT touched here.
async function clearGeneratedRtlArtifacts(
  moduleId: string,
  reason: string,
  runtime: OrchestratorRuntime,
): Promise<void> {
  const rtlDir = resolvePipelineConfigPath(PIPELINE_CONFIG.rtl_dir);
  const verilogPath = path.join(rtlDir, `${moduleId}.v`);
  const metaPath = path.join(rtlDir, `${moduleId}.meta.json`);
  const removed: string[] = [];
  for (const target of [verilogPath, metaPath]) {
    try {
      await unlink(target);
      removed.push(path.basename(target));
    } catch (err: unknown) {
      const code = (err as NodeJS.ErrnoException).code;
      if (code === "ENOENT") continue;
      await appendRunLog(
        {
          event: "rtl_artifact_clear_warning",
          module_id: moduleId,
          path: target,
          reason,
          error: err instanceof Error ? err.message : String(err),
        },
        runtime,
      );
    }
  }
  if (removed.length > 0) {
    await appendRunLog(
      {
        event: "rtl_artifacts_cleared",
        module_id: moduleId,
        removed,
        reason,
      },
      runtime,
    );
  }
}

async function persistVerilogModule(module: VerilogModule): Promise<void> {
  // Agents are supposed to call the write_verilog MCP tool themselves, but
  // Sonnet/Opus under outputFormat: json_schema sometimes skip tool calls
  // to save turns. Orchestrator owns disk state, so ensure the .v and
  // .meta.json files exist regardless of whether the agent persisted them.
  const rtlDir = resolvePipelineConfigPath(PIPELINE_CONFIG.rtl_dir);
  const verilogPath = path.join(rtlDir, `${module.module_id}.v`);
  const metaPath = path.join(rtlDir, `${module.module_id}.meta.json`);
  await mkdir(rtlDir, { recursive: true });
  await writeFile(verilogPath, module.verilog_source, "utf8");
  await writeFile(metaPath, `${JSON.stringify(module, null, 2)}\n`, "utf8");
}

async function invokeFoundry(
  layerIr: LayerIR,
  runtime: OrchestratorRuntime,
  options: {
    resumeSessionId?: string;
    retrospectorAdvice?: RetrospectorAdvice;
    selfImproveEnabled?: boolean;
    replacementForDocIds?: string[];
    newDocFailureContext?: NewDocFailureContext;
    // Continuation context. When any of these is set, invokeFoundry resumes
    // the latest Foundry session for this (module, contract) and sends only
    // a focused user-turn (no system-prompt re-injection, no full LayerIR
    // dump). Surgeon's attempt and the Retrospector advice — when present —
    // arrive in the resumed conversation as a single orchestrator user
    // message, never as a fake assistant turn.
    priorVerifResult?: VerifResult;
    surgeonAttempt?: { module: VerilogModule; verifResult: VerifResult };
    isFinalAttempt?: boolean;
  } = {},
): Promise<RtlAgentRunResult> {
  // If the caller didn't pin a specific session but a prior Foundry call has
  // already happened on this (module, contract), auto-resume it. This is
  // mainly used by the post-Retrospector final Foundry retry.
  const autoResumeSessionId = options.resumeSessionId ?? latestFoundrySessionId(layerIr) ?? undefined;
  const isContinuation =
    autoResumeSessionId !== undefined &&
    (options.priorVerifResult !== undefined ||
      options.surgeonAttempt !== undefined ||
      options.retrospectorAdvice !== undefined);
  await appendRunLog(
    {
      event: "action",
      action: options.retrospectorAdvice
        ? "invoke_foundry_after_retrospector"
        : isContinuation
          ? "invoke_foundry_continuation"
          : "invoke_foundry",
      module_id: layerIr.module_id,
      ...(autoResumeSessionId ? { resume_session_id: autoResumeSessionId } : {}),
    },
    runtime,
  );

  // Doc-coverage guard. When self-improve is enabled we only ask Foundry to
  // emit a `draft_doc` if no existing pattern doc already covers this
  // layer's (contract_id, op_type, kernel) tuple. This prevents the
  // probationary tier from accumulating redundant timestamped duplicates
  // every time a covered contract runs successfully, and lets the wrapper
  // schema (`{module, draft_doc}`) be replaced by the simpler `{module}`
  // shape on covered runs — which materially reduces Foundry's
  // malformed-final-message rate.
  const docLifecycleStateForGuard = options.selfImproveEnabled
    ? await loadDocLifecycleState()
    : null;
  const coveringDocForLayer = docLifecycleStateForGuard
    ? findCoveringDoc(docLifecycleStateForGuard, layerIr)
    : null;
  const selfImproveDocRequest =
    options.selfImproveEnabled && coveringDocForLayer === null
      ? {
          enabled: true,
          destination_tier: "probationary",
          promotion_successes_required: PIPELINE_CONFIG.doc_promotion_success_threshold,
          replacement_for_doc_ids: options.replacementForDocIds ?? [],
          contract_id: currentContractId(layerIr),
          contract_key: contractStateKeyForLayer(layerIr),
        }
      : undefined;
  if (options.selfImproveEnabled && coveringDocForLayer !== null) {
    await appendRunLog(
      {
        event: "self_improve_doc_request_skipped",
        module_id: layerIr.module_id,
        contract_id: currentContractId(layerIr),
        contract_key: contractStateKeyForLayer(layerIr),
        covering_doc_tier: coveringDocForLayer.tier,
        covering_doc_path: coveringDocForLayer.path,
        ...(coveringDocForLayer.doc_id ? { covering_doc_id: coveringDocForLayer.doc_id } : {}),
        reason: "existing_doc_covers_contract_op_kernel",
      },
      runtime,
    );
  }
  const createNewDocRequest = options.selfImproveEnabled
    ? await maybeBuildCreateNewDocRequest(layerIr, options.newDocFailureContext, runtime)
    : null;
  const preloadedRtlPatterns = await loadRetrospectorKnowledgeDoc(layerIr);
  const failureMemory = await failureMemoryForLayer(layerIr);
  const foundryPayload = {
    layer_ir: layerIr,
    expected_spec_hash: computeExpectedSpecHash(layerIr),
    preloaded_rtl_patterns: preloadedRtlPatterns,
    ...(failureMemory.length > 0 ? { failure_memory: failureMemory } : {}),
    contract_options: {
      selected_contract: contractPlanForLayer(layerIr),
      ordered_contracts: CONTRACT_PLANS,
      expected_latency_cycles: expectedLatencyCyclesForContract(layerIr, contractSidecarFields(layerIr)),
      covered_by_existing_doc: createNewDocRequest === null,
    },
    write_verilog_output_dir: resolvePipelineConfigPath(PIPELINE_CONFIG.rtl_dir),
    ...(selfImproveDocRequest ? { self_improve_doc_request: selfImproveDocRequest } : {}),
    ...(createNewDocRequest ? { create_new_doc_request: createNewDocRequest } : {}),
  };
  // Three prompt modes for Foundry:
  //   1. Fresh call (no resume, no continuation context) → full delegation
  //      prompt with system prompt + LayerIR + closest-family docs. ~20K tok.
  //   2. Continuation (resumed session + prior failure / surgeon attempt /
  //      retrospector advice) → focused user-turn appended to the resumed
  //      session. The agent already has the LayerIR / docs in its context;
  //      we just hand it the new evidence. ~1-3K tok.
  //   3. Retrospector-final-retry (legacy specialised path; preserved for
  //      back-compat) → the original injection prompt. Same shape as
  //      continuation but written before this refactor; kept as the path
  //      `maybeRunRetrospectorFinalAttempt` already calls.
  let resumedPrompt: string | undefined;
  if (isContinuation) {
    resumedPrompt = buildFoundryContinuationPrompt({
      expected_spec_hash: foundryPayload.expected_spec_hash,
      write_verilog_output_dir: foundryPayload.write_verilog_output_dir,
      attempt_index: foundryVersionsFor(layerIr).length + 1,
      prior_verif_result: options.priorVerifResult,
      surgeon_attempt: options.surgeonAttempt
        ? { module: options.surgeonAttempt.module, verif_result: options.surgeonAttempt.verifResult }
        : undefined,
      retrospector_advice: options.retrospectorAdvice,
      is_final_attempt: options.isFinalAttempt ?? !!options.retrospectorAdvice,
      failure_memory: failureMemory,
      self_improve_doc_request: selfImproveDocRequest,
      create_new_doc_request: createNewDocRequest ?? undefined,
    });
  } else if (options.retrospectorAdvice) {
    // Legacy retrospector-injection path: kept so any existing call-site that
    // passes retrospectorAdvice without surgeonAttempt still works.
    resumedPrompt = buildFoundryRetrospectorInjectionPrompt({
      ...foundryPayload,
      retrospector_advice: options.retrospectorAdvice,
      final_attempt: foundryVersionsFor(layerIr).length + 1,
      self_improve_doc_request: selfImproveDocRequest,
      create_new_doc_request: createNewDocRequest ?? undefined,
    });
  }

  // Foundry occasionally returns a path or bare text as its final message
  // instead of the VerilogModule JSON, even though it correctly called
  // write_verilog.  Recover from disk when the JSON parse / schema validation
  // fails but the .v file is present.
  // Use the `{module, draft_doc}` wrapper schema only when we are actually
  // asking Foundry for a draft. On self-improve runs whose contract+kernel
  // is already covered (selfImproveDocRequest === undefined), the wrapper is
  // unnecessary surface area and noticeably more likely to come back
  // malformed — fall back to the plain `{VerilogModule}` schema then.
  const useDraftDocWrapperSchema = selfImproveDocRequest !== undefined;
  let result: RtlAgentRunResult;
  try {
    if (useDraftDocWrapperSchema) {
      const withDoc = await runDelegatedAgent<RtlAgentWithDoc>(
        "foundry",
        foundryPayload,
        rtlAgentWithDocOutputFormat,
        rtlAgentWithDocZod,
        runtime,
        {
          prompt: resumedPrompt,
          resumeSessionId: autoResumeSessionId,
        },
      );
      const hydratedModule = await hydrateVerilogModuleFromDisk(
        withDoc.payload.module,
        layerIr,
      );
      result = {
        payload: hydratedModule,
        draft_doc: withDoc.payload.draft_doc,
        doc_request: createNewDocRequest,
        result: withDoc.result,
        messages: withDoc.messages,
      };
    } else {
      const plainResult = await runDelegatedAgent<VerilogModuleAgentOutput>(
        "foundry",
        foundryPayload,
        verilogModuleAgentOutputFormat,
        verilogModuleAgentOutputZod,
        runtime,
        {
          prompt: resumedPrompt,
          resumeSessionId: autoResumeSessionId,
        },
      );
      const hydratedModule = await hydrateVerilogModuleFromDisk(
        plainResult.payload,
        layerIr,
      );
      result = {
        payload: hydratedModule,
        result: plainResult.result,
        messages: plainResult.messages,
      };
    }
  } catch (err) {
    if (err instanceof SpecHashMismatchError) {
      throw err;
    }
    const recovered = await tryRecoverVerilogModuleFromDisk(
      layerIr,
      "Foundry",
      /* attempt */ 1,
    );
    if (!recovered) {
      throw err;
    }
    // Preserve cost / session / modelUsage / messages from the SDK turn
    // even when the final structured output was unparseable. Without this,
    // pipeline_state.json reports $0 spent on multi-dollar Foundry runs and
    // tool-use audit goes empty.
    const carried =
      err instanceof StructuredOutputParseError
        ? { result: err.result, messages: err.messages }
        : {
            result: {
              type: "result",
              subtype: "success",
              result: "",
              total_cost_usd: 0,
              modelUsage: {},
            } as unknown as SDKResultMessage,
            messages: [] as SDKMessage[],
          };
    // Best-effort scrape of `draft_doc` so a working RTL on a previously-
    // uncovered contract still grows the probationary tier even when the
    // wrapper JSON parse failed (Phase 4 must not silently drop a real win).
    let scrapedDraft: DocDraft | null = null;
    if (useDraftDocWrapperSchema && err instanceof StructuredOutputParseError) {
      const rawText =
        err.result.subtype === "success" && typeof err.result.result === "string"
          ? err.result.result
          : "";
      const scraped = scrapeDraftDocFromText(rawText);
      if (scraped !== null) {
        const parsed = docDraftZod.safeParse(scraped);
        if (parsed.success) {
          scrapedDraft = parsed.data;
        }
      }
    }
    await appendRunLog(
      {
        event: "agent_result_recovered",
        agent: "Foundry",
        module_id: layerIr.module_id,
        reason: err instanceof Error ? err.message : String(err),
        carried_cost_usd:
          (carried.result as { total_cost_usd?: number }).total_cost_usd ?? 0,
        carried_session_id:
          (carried.result as { session_id?: string }).session_id ?? null,
        carried_message_count: carried.messages.length,
        scraped_draft_doc: scrapedDraft !== null,
      },
      runtime,
    );
    result = {
      payload: recovered,
      ...(scrapedDraft !== null ? { draft_doc: scrapedDraft } : {}),
      doc_request: createNewDocRequest,
      result: carried.result,
      messages: carried.messages,
    };
  }

  await persistVerilogModule(result.payload);

  const foundryAudits = extractToolUseAudits(result.messages, {
    agent: "Foundry",
    module_id: layerIr.module_id,
    nowIso: runtime.now().toISOString(),
  });
  recordFoundryVersion(layerIr, result, foundryAudits);
  if (options.selfImproveEnabled) {
    await recordDocUsageForAgent(layerIr, layerIr.module_id, foundryAudits, runtime);
  }
  await appendToolUseAudits(foundryAudits);
  await appendForeignMcpToolWarnings(foundryAudits, runtime);
  await appendRunLog(
    {
      event: "agent_tool_use_summary",
      agent: "Foundry",
      module_id: layerIr.module_id,
      ...summarizeToolUse(foundryAudits),
    },
    runtime,
  );

  await appendRunLog(
    {
      event: "agent_result",
      agent: "Foundry",
      module_id: layerIr.module_id,
      total_cost_usd: result.result.total_cost_usd,
      modelUsage: result.result.modelUsage,
      session_id: extractSessionId(result.messages, result.result),
      payload: result.payload,
    },
    runtime,
  );

  return result;
}

// Deterministic verification. The orchestrator writes the sidecar from the
// LayerIR (all fields are either fixed literals — canonical signal names —
// or LayerIR values), then invokes the MCP run_iverilog lint pass followed
// by run_verilator. The Verilator testbench itself produces a VerifResult
// JSON that we validate via Zod. Previously this path went through a Haiku
// "Assayer" LLM that repeatedly hallucinated VerifResults instead of calling
// the tools. There is no language reasoning involved: the pipeline has a
// VerifResult iff Verilator produced one.
export async function runAssayerDeterministic(
  module: VerilogModule,
  layer: LayerIR,
): Promise<VerifResult> {
  assayerLayerBusContractZod.parse(layer);

  // Truncation/stub guard. Foundry/Surgeon agents occasionally hit their
  // output token cap mid-module and emit a placeholder ("// See output/rtl/
  // ...v on disk for the full source") or a self-truncated stub. Catch
  // those deterministically before iverilog gets a chance to fail with
  // STATUS_NOT_IMPLEMENTED. The check is conservative: it only fires when
  // the source contains an explicit truncation marker AND is far below
  // the smallest plausible canonical RTL we've seen for any contract.
  // Patterns the agent uses when it hits its output cap mid-module and
  // tries to "outsource" the body. Each one is the agent's confession that
  // the file does NOT contain real RTL — even if line count is plausible.
  // Add new variants here as they're observed in failure_corpus.
  const truncationMarkers = [
    /See\s+output\/rtl\//i,
    /persisted\s+to\s+output\/rtl\//i,
    /written\s+to\s+output\/rtl\//i,
    /full\s+(FSM|module|source|datapath|body)\s+persisted/i,
    /full\s+(FSM|module|source|datapath|body)\s+(in|on)\s+disk/i,
    /(write_verilog|mcp__nn2rtl-tools__write_verilog).*unavailable/i,
    /truncated\s+here/i,
    /See\s+full\s+source/i,
    /the\s+full\s+\d+-line\s+source/i,
    /full\s+source\s+(in|on)\s+disk/i,
    /\(adapted\s+from\b.*\bsee\s+file\s+on\s+disk/i,
  ];
  const sourceLines = module.verilog_source.split("\n").length;
  const hasTruncationMarker = truncationMarkers.some((re) => re.test(module.verilog_source));
  if (hasTruncationMarker || sourceLines < 30) {
    const stubMessage = [
      "Foundry/Surgeon output is a truncated stub or placeholder, not real Verilog.",
      `Detected ${sourceLines} lines${hasTruncationMarker ? " and a truncation marker" : ""}.`,
      "Treat as agent_max_turns_exhausted and force a fresh attempt or template clone.",
      "Do NOT proceed to iverilog/Verilator — they will produce noise that hides the agent failure.",
    ].join("\n");
    return {
      module_id: module.module_id,
      status: "fail",
      timing_pass: false,
      timing_actual_cycles: 0,
      timing_expected_cycles: layer.pipeline_latency_cycles,
      failure_class: "agent_max_turns_exhausted",
      failure_category: "code_bug",
      fix_hint: stubMessage,
      iverilog_stderr: stubMessage,
    };
  }

  const preflightIssues = preflightVerilogModule(module, layer);
  if (preflightIssues.length > 0) {
    const preflightMessage = [
      "Deterministic preflight rejected the RTL before iverilog/Verilator.",
      "Repair the canonical top-level interface so it matches the Assayer contract.",
      "Preflight findings:",
      ...preflightIssues.map((issue) => `- ${issue}`),
    ].join("\n");
    return {
      module_id: module.module_id,
      status: "fail",
      timing_pass: false,
      timing_actual_cycles: 0,
      timing_expected_cycles: layer.pipeline_latency_cycles,
      failure_class: "port_width_mismatch",
      fix_hint: preflightMessage,
      iverilog_stderr: preflightMessage,
    };
  }

  const structuralIssues = structuralPreflightViolations(module, layer);
  if (structuralIssues.length > 0) {
    const rules = structuralIssues.map((v) => v.rule).join(", ");
    const structuralMessage = [
      "Deterministic structural preflight rejected the RTL before iverilog/Verilator.",
      `Violated rule(s): ${rules}.`,
      "Repair the indicted construct exactly; do not touch unrelated logic.",
      "Violations:",
      ...structuralIssues.map((v) => `- [${v.rule}] ${v.detail}`),
    ].join("\n");
    return {
      module_id: module.module_id,
      status: "fail",
      timing_pass: false,
      timing_actual_cycles: 0,
      timing_expected_cycles: layer.pipeline_latency_cycles,
      failure_class: "structural_preflight_failed",
      fix_hint: structuralMessage,
      iverilog_stderr: structuralMessage,
    };
  }

  const contractIssues = contractConformanceViolations(module, layer);
  if (contractIssues.length > 0) {
    const rules = contractIssues.map((v) => v.rule).join(", ");
    const contractMessage = [
      "Deterministic contract conformance gate rejected the RTL before iverilog/Verilator.",
      `Selected contract: ${currentContractId(layer)}.`,
      `Violated rule(s): ${rules}.`,
      "Repair the contract-level architecture exactly; do not paper over this with comments or unused placeholder ports.",
      "Violations:",
      ...contractIssues.map((v) => `- [${v.rule}] ${v.detail}`),
    ].join("\n");
    return {
      module_id: module.module_id,
      status: "fail",
      timing_pass: false,
      timing_actual_cycles: 0,
      timing_expected_cycles: layer.pipeline_latency_cycles,
      failure_class: "structural_preflight_failed",
      violated_constraint: `contract_conformance:${rules}`,
      fix_hint: contractMessage,
      iverilog_stderr: contractMessage,
    };
  }

  const mcpTools = (await import(MCP_TOOLS_MODULE_PATH)) as {
    run_iverilog: (
      verilog_source: string,
      module_name: string,
    ) => Promise<{ success: boolean; stderr: string }>;
    run_verilator: (
      verilog_source: string,
      module_name: string,
      sidecar_path: string,
    ) => Promise<VerifResult>;
  };

  // Canonical signal names are fixed; every LayerIR carries them as literals
  // (enforced by the schema) so we just pass them through. The sidecar lives
  // under output/tb/ next to any future manually authored test sidecars.
  const sidecarPath = buildSidecarPath(module.module_id);
  const resultsPath = path.join(
    resolvePipelineConfigPath(PIPELINE_CONFIG.reports_dir),
    `${module.module_id}.results.json`,
  );
  const contractFields = contractSidecarFields(layer);
  const contractGoldens = await materializeContractGoldens(layer);
  const expectedLatencyCycles = expectedLatencyCyclesForContract(layer, contractFields);
  const sidecar = {
    module_name: module.module_id,
    module_id: module.module_id,
    clock_signal: "clk" as const,
    reset_signal: "rst_n" as const,
    valid_in_signal: "valid_in" as const,
    valid_out_signal: "valid_out" as const,
    ready_in_signal: "ready_in" as const,
    data_in_signal: "data_in" as const,
    data_out_signal: "data_out" as const,
    bus_bytes_per_sample: layer.input_width_bits / 8,
    input_width_bits: layer.input_width_bits,
    output_width_bits: layer.output_width_bits,
    pipeline_latency_cycles: expectedLatencyCycles,
    clock_period_ns: layer.clock_period_ns,
    golden_inputs_path: contractGoldens.goldenInputsPath,
    golden_outputs_path: contractGoldens.goldenOutputsPath,
    results_path: resultsPath,
    testbench_template_path: contractTestbenchTemplatePath(resolveLayerContractId(layer)),
    ...contractFields,
  };
  await mkdir(path.dirname(sidecarPath), { recursive: true });
  await mkdir(path.dirname(resultsPath), { recursive: true });
  await writeFile(sidecarPath, `${JSON.stringify(sidecar, null, 2)}\n`, "utf8");

  // Lint first — iverilog catches most obvious Verilog mistakes faster than
  // Verilator's multi-minute build, and a lint failure is always a syntax
  // error (not a numerical/timing issue).
  const iverilog = await mcpTools.run_iverilog(module.verilog_source, module.module_id);
  if (!iverilog.success) {
    const noDiagnosticIverilogFailure =
      /iverilog exited non-zero without diagnostic output|exit_code=3221225794|0xC0000002|STATUS_NOT_IMPLEMENTED/i.test(
        iverilog.stderr,
      );
    if (noDiagnosticIverilogFailure) {
      return {
        module_id: module.module_id,
        status: "fail",
        status_class: "tb_setup_error",
        timing_pass: false,
        timing_actual_cycles: -1,
        timing_expected_cycles: expectedLatencyCycles,
        iverilog_stderr: iverilog.stderr,
        fix_hint: [
          "iverilog exited non-zero without compiler diagnostics.",
          "This is treated as a deterministic verification/toolchain setup failure, not an RTL syntax bug.",
          "Check the pipeline process environment, especially PATH, YOSYSHQ_ROOT, TMPDIR/TEMP, and the exact iverilog binary.",
          "Replay the embedded verilog_source before spending a Surgeon attempt.",
          "iverilog diagnostic:",
          iverilog.stderr,
        ].join("\n\n"),
      };
    }

    return {
      module_id: module.module_id,
      status: "syntax_error",
      timing_pass: false,
      timing_actual_cycles: 0,
      timing_expected_cycles: expectedLatencyCycles,
      iverilog_stderr: iverilog.stderr,
      fix_hint: [
        "iverilog lint rejected the RTL before Verilator could run.",
        "Repair the Verilog so `iverilog -g2012` accepts it.",
        "iverilog stderr:",
        iverilog.stderr,
      ].join("\n\n"),
    };
  }

  // Full Verilator run — builds the DUT, runs the handwritten C++ bench,
  // reads the structured results JSON written by the bench, validates it
  // via verifResultSchema inside run_verilator, and returns a VerifResult.
  return mcpTools.run_verilator(module.verilog_source, module.module_id, sidecarPath);
}

async function invokeAssayer(
  module: VerilogModule,
  layerIr: LayerIR,
  runtime: OrchestratorRuntime,
): Promise<VerifResult> {
  await appendRunLog(
    {
      event: "action",
      action: "invoke_assayer",
      module_id: module.module_id,
    },
    runtime,
  );

  let payload: VerifResult;
  try {
    const raw = await runtime.assayerFn(module, layerIr);
    const parsed = verifResultZod.safeParse(raw);
    if (!parsed.success) {
      throw new Error(
        `assayerFn returned invalid VerifResult:\n${JSON.stringify(parsed.error.issues, null, 2)}`,
      );
    }
    payload = parsed.data;
  } catch (error: unknown) {
    // Tool crashed before producing a structured VerifResult. That means the
    // RTL never received a real deterministic verdict, so stop the retry loop
    // as a testbench/toolchain setup failure instead of spending Surgeon turns
    // on a module-local repair that has no evidence.
    payload = {
      module_id: module.module_id,
      status: "fail",
      status_class: "tb_setup_error",
      timing_pass: false,
      timing_actual_cycles: -1,
      timing_expected_cycles: layerIr.pipeline_latency_cycles,
      fix_hint: `Assayer runner crashed before producing a VerifResult: ${error instanceof Error ? error.message : String(error)}`,
    };
  }

  await appendRunLog(
    {
      event: "assayer_result",
      module_id: module.module_id,
      diagnostic_summary: verifDiagnosticSummary(payload),
      score: verifScore(payload),
      payload,
    },
    runtime,
  );

  return payload;
}

/**
 * Compact record of one completed repair cycle. The orchestrator keeps a
 * ring buffer of these per module and hands the last N to each fresh
 * Surgeon invocation so it can see which approaches have already been
 * tried and why they didn't work. This is the mechanism that breaks the
 * "oscillate between the same failed edits" cycle — Surgeon now has
 * memory of prior attempts.
 *
 * Architecture-neutral: every field is a fact about simulation behaviour
 * or a syntactic diff of Verilog text. Works for any layer, any bug class.
 */
type SurgeonAttemptRecord = {
  attempt_index: number;
  outcome:
    | "accepted_still_failing"   // attempt compiled and ran, verif still fail
    | "reverted_preflight"       // Surgeon broke the port contract
    | "reverted_functional"      // Surgeon broke the simulation (regression guard)
    | "reverted_recovered";      // Surgeon's LLM dispatch crashed; disk recovery
  verif_summary: Record<string, unknown>;
  // Unified diff of the Surgeon-produced RTL against the module that was
  // handed to Surgeon at the start of the attempt. Truncated to at most
  // ~6k chars so a run of 2-3 prior attempts stays under ~20k tokens.
  rtl_diff_unified: string;
};

/** Per-module ring buffer of prior Surgeon attempts. Kept in-memory only;
 *  lost on resume-from-disk, which is acceptable for the first iteration. */
const SURGEON_HISTORY = new Map<string, SurgeonAttemptRecord[]>();
const SURGEON_HISTORY_DEPTH = 3;  // last 3 attempts surfaced in the next prompt

/**
 * Reset all in-memory agent-history Maps. Called at the start of every
 * `runPipeline` so test-suite ordering can't leak session ids or prior
 * attempt records across tests, and so a fresh top-level invocation never
 * inherits state from a prior run in the same process. The Maps are not
 * persisted to disk, so this only affects in-process state.
 */
export function clearAgentHistories(): void {
  FOUNDRY_HISTORY.clear();
  FAILURE_ATTEMPT_HISTORY.clear();
  ATTEMPT_ARTIFACT_HISTORY.clear();
  SURGEON_HISTORY.clear();
}

/**
 * Reseed FOUNDRY_HISTORY from the on-disk run_log so a process restart can
 * resume the previous Foundry conversation (continuation prompt + Anthropic
 * server-side cache reuse) instead of starting a fresh session and losing
 * the cached system prompt + retrieved patterns.
 *
 * Scoped narrowly: only seeds entries for modules currently in `fail_retry`,
 * which is the only state where the next tick is a Foundry retry. Modules
 * that already passed, fail-aborted, or are mid-generation do not need a
 * resumable session; reseeding them would risk grafting a stale id onto a
 * different (module, contract) pair after a contract walk.
 */
async function reseedFoundryHistoryFromRunLog(
  pipelineIr: PipelineIR,
  manager: PipelineStateManager,
  runtime: OrchestratorRuntime,
): Promise<void> {
  const runLogPath = reportPath("run_log.jsonl");
  if (!(await pathExists(runLogPath))) return;
  let raw: string;
  try {
    raw = await readFile(runLogPath, "utf8");
  } catch {
    return;
  }
  const layersByModuleId = new Map<string, LayerIR>();
  for (const layer of pipelineIr.layers) {
    layersByModuleId.set(layer.module_id, layer);
  }
  const state = manager.getState();
  let seeded = 0;
  const seededModules = new Set<string>();
  for (const line of raw.split(/\r?\n/)) {
    if (!line.trim()) continue;
    let event: Record<string, unknown>;
    try {
      event = JSON.parse(line) as Record<string, unknown>;
    } catch {
      continue;
    }
    if (event.event !== "agent_result") continue;
    if (event.agent !== "Foundry") continue;
    const moduleId = typeof event.module_id === "string" ? event.module_id : null;
    const sessionId = typeof event.session_id === "string" ? event.session_id : null;
    if (!moduleId || !sessionId) continue;
    if (state.modules[moduleId] !== "fail_retry") continue;
    const layer = layersByModuleId.get(moduleId);
    if (!layer) continue;
    const parsedPayload = verilogModuleZod.safeParse(event.payload);
    if (!parsedPayload.success) continue;
    // Build the key from the payload's spec_hash, NOT from
    // computeExpectedSpecHash(layer): the prior run's spec_hash reflects
    // whatever contract the contract walker had selected at the time
    // (e.g. dram-backed-weights), but pipelineIr.layers carries the base
    // LayerIR before applyContractPlan(), so recomputing from `layer` here
    // produces the base-contract key and misses the seeded session.
    const key = `${moduleId}:${parsedPayload.data.spec_hash}`;
    const history = FOUNDRY_HISTORY.get(key) ?? [];
    history.push({
      version_index: history.length + 1,
      module: jsonClone(parsedPayload.data),
      session_id: sessionId,
      // run_log doesn't preserve tool-use audits or document-retrieval calls,
      // so the reseeded record only carries what's needed for session resume.
      // Retrospector-input shaping (foundry_versions field) sees an empty
      // tool_use_summary on the reseeded record; that's correct because the
      // tool calls are now in the resumed Anthropic session, not in our log.
      tool_use_summary: {},
      documents_used: [],
    });
    FOUNDRY_HISTORY.set(key, history);
    seeded += 1;
    seededModules.add(moduleId);
  }
  if (seeded > 0) {
    await appendRunLog(
      {
        event: "foundry_history_reseeded_from_run_log",
        seeded_versions: seeded,
        seeded_modules: [...seededModules],
        source: runLogPath,
      },
      runtime,
    );
  }
}

function recordSurgeonAttempt(moduleId: string, record: SurgeonAttemptRecord): void {
  const history = SURGEON_HISTORY.get(moduleId) ?? [];
  history.push(record);
  if (history.length > SURGEON_HISTORY_DEPTH) {
    history.splice(0, history.length - SURGEON_HISTORY_DEPTH);
  }
  SURGEON_HISTORY.set(moduleId, history);
}

function priorSurgeonAttempts(moduleId: string): SurgeonAttemptRecord[] {
  return SURGEON_HISTORY.get(moduleId) ?? [];
}

/** Unified-diff-ish line-by-line comparison. Not a full Myers diff — just
 *  marks lines unique to the prior source with "-" and lines unique to the
 *  new source with "+". Context is the shortest run of equal lines. This
 *  is cheap, architecture-neutral, and good enough for Surgeon to see
 *  "you edited these lines last time." */
function unifiedishDiff(priorSource: string, nextSource: string, maxChars = 6000): string {
  const priorLines = priorSource.split(/\r?\n/);
  const nextLines = nextSource.split(/\r?\n/);
  const priorSet = new Set(priorLines);
  const nextSet = new Set(nextLines);
  const out: string[] = [];
  let i = 0;
  let j = 0;
  let contextCarry = 0;
  while (i < priorLines.length || j < nextLines.length) {
    const a = priorLines[i];
    const b = nextLines[j];
    if (a !== undefined && b !== undefined && a === b) {
      // Equal line — include a few as context, then skip consecutive runs.
      if (contextCarry < 2) {
        out.push(`  ${a}`);
        contextCarry += 1;
      } else if (contextCarry === 2) {
        out.push("  ...");
        contextCarry += 1;
      }
      i += 1; j += 1;
      continue;
    }
    contextCarry = 0;
    if (a !== undefined && !nextSet.has(a)) {
      out.push(`- ${a}`);
      i += 1;
      continue;
    }
    if (b !== undefined && !priorSet.has(b)) {
      out.push(`+ ${b}`);
      j += 1;
      continue;
    }
    // Lines present in both but out of order — treat as equal, advance.
    if (a !== undefined) i += 1;
    if (b !== undefined) j += 1;
  }
  const joined = out.join("\n");
  if (joined.length <= maxChars) return joined;
  return `${joined.slice(0, maxChars)}\n[... diff truncated at ${maxChars} chars ...]`;
}

// Window size (samples before and after) kept around first_mismatch_index when
// trimming expected/got arrays for the Surgeon payload. 64 samples gives enough
// local context to see the pattern (e.g. an off-by-one, sign flip, saturation)
// without dumping the full 100k-element vector that dominated cache-creation.
const SURGEON_MISMATCH_WINDOW = 64;

// Strip large noisy arrays from a VerifResult before embedding it in the
// Surgeon prompt. Previously the full 100,352-element expected/got arrays were
// always passed through, costing ~100k cache-creation tokens per Surgeon call
// for conv layers. They are only diagnostic when there's a sim mismatch, and
// even then only a window around `first_mismatch_index` is load-bearing —
// beyond that window the arrays are just token bloat.
function trimVerifResultForSurgeon(verif: VerifResult): VerifResult {
  const hasArrays = Array.isArray(verif.expected) && Array.isArray(verif.got);
  if (!hasArrays) return verif;

  // Synthesis-only failure: sim passed, arrays match. They contribute nothing
  // and waste ~100k tokens of cache. Drop them entirely.
  if (verif.status_class === "sim_passed") {
    return { ...verif, expected: [], got: [] };
  }

  const idx = verif.first_mismatch_index ?? -1;
  if (idx < 0) return verif;

  const expected = verif.expected as number[];
  const got = verif.got as number[];
  const start = Math.max(0, idx - SURGEON_MISMATCH_WINDOW);
  const end = Math.min(expected.length, idx + SURGEON_MISMATCH_WINDOW + 1);
  // Note: `first_mismatch_index` stays absolute; Surgeon can compute the
  // local offset within the trimmed arrays as (first_mismatch_index - start).
  // We don't add window-start fields because verifResultSchema is strict.
  return {
    ...verif,
    expected: expected.slice(start, end),
    got: got.slice(start, end),
  };
}

type LayerReferenceEvidence = {
  source: "compute_layer_reference";
  reason: "first_mismatch";
  request: {
    module_id: string;
    vector_idx: number;
    output_pixel_oy: number;
    output_pixel_ox: number;
    oc_start: number;
    oc_end: number;
    include_intermediates: boolean;
    caller_role: "assayer";
  };
  observed: {
    first_mismatch_index: number | null;
    first_mismatch_vector_index: number;
    first_mismatch_output_index: number;
    first_mismatch_channel_index: number;
    first_mismatch_expected: number | null;
    first_mismatch_got: number | null;
  };
  result: Record<string, unknown>;
};

function nonnegativeInteger(value: unknown): value is number {
  return Number.isInteger(value) && Number(value) >= 0;
}

function computeReferenceRequestForFirstMismatch(
  layer: LayerIR,
  verif: VerifResult,
): LayerReferenceEvidence["request"] | null {
  if (layer.op_type !== "conv2d") return null;
  if ((layer.groups ?? 1) !== 1) return null;
  if (layer.dilation?.some((value) => value !== 1)) return null;
  const [, oc, oh, ow] = layer.output_shape;
  if (![oc, oh, ow].every((value) => Number.isInteger(value) && value > 0)) return null;

  const vectorIdx = verif.first_mismatch_vector_index;
  const outputIdx = verif.first_mismatch_output_index;
  const channelIdx = verif.first_mismatch_channel_index;
  if (!nonnegativeInteger(vectorIdx) || !nonnegativeInteger(outputIdx) || !nonnegativeInteger(channelIdx)) {
    return null;
  }

  // The testbench's first_mismatch_output_index is the emitted output beat
  // within the current golden vector. A flat-bus conv emits one beat per
  // output pixel; channel-tiled contracts emit ceil(OC/channels_per_beat)
  // beats per pixel. Derive the pixel and absolute output channel from the
  // LayerIR bus width so the oracle request is contract-neutral.
  const channelsPerBeat = Math.min(oc, Math.max(1, Math.floor(layer.output_width_bits / 8)));
  const beatsPerPixel = Math.max(1, Math.ceil(oc / channelsPerBeat));
  const pixelIndex = Math.floor(outputIdx / beatsPerPixel);
  if (pixelIndex < 0 || pixelIndex >= oh * ow) return null;
  const beatIndex = outputIdx % beatsPerPixel;
  const trueOc = beatIndex * channelsPerBeat + channelIdx;
  if (trueOc < 0 || trueOc >= oc) return null;

  return {
    module_id: layer.module_id,
    vector_idx: vectorIdx,
    output_pixel_oy: Math.floor(pixelIndex / ow),
    output_pixel_ox: pixelIndex % ow,
    oc_start: trueOc,
    oc_end: trueOc + 1,
    include_intermediates: true,
    caller_role: "assayer",
  };
}

async function maybeComputeLayerReferenceEvidence(
  layer: LayerIR,
  verif: VerifResult,
  runtime: OrchestratorRuntime,
): Promise<LayerReferenceEvidence | null> {
  const request = computeReferenceRequestForFirstMismatch(layer, verif);
  if (!request) return null;
  try {
    const mcpTools = (await import(MCP_TOOLS_MODULE_PATH)) as {
      compute_layer_reference: (input: typeof request) => Promise<Record<string, unknown>>;
    };
    const result = await mcpTools.compute_layer_reference(request);
    await appendRunLog(
      {
        event: "layer_reference_evidence_preloaded",
        module_id: layer.module_id,
        source: "compute_layer_reference",
        request,
        output_fingerprint: typeof result.output_fingerprint === "string" ? result.output_fingerprint : null,
      },
      runtime,
    );
    return {
      source: "compute_layer_reference",
      reason: "first_mismatch",
      request,
      observed: {
        first_mismatch_index: verif.first_mismatch_index ?? null,
        first_mismatch_vector_index: verif.first_mismatch_vector_index ?? request.vector_idx,
        first_mismatch_output_index: verif.first_mismatch_output_index ?? 0,
        first_mismatch_channel_index: verif.first_mismatch_channel_index ?? request.oc_start,
        first_mismatch_expected: verif.first_mismatch_expected ?? null,
        first_mismatch_got: verif.first_mismatch_got ?? null,
      },
      result,
    };
  } catch (error: unknown) {
    await appendRunLog(
      {
        event: "layer_reference_evidence_unavailable",
        module_id: layer.module_id,
        source: "compute_layer_reference",
        request,
        reason: error instanceof Error ? error.message : String(error),
      },
      runtime,
    );
    return null;
  }
}

async function invokeSurgeon(
  brokenModule: VerilogModule,
  verifResult: VerifResult,
  layerIr: LayerIR,
  runtime: OrchestratorRuntime,
  options: {
    selfImproveEnabled?: boolean;
    // Set when this Surgeon dispatch is the post-Retrospector final attempt
    // (the orchestrator already exhausted ordinary Foundry/Surgeon retries
    // and Retrospector chose `next_actor: "surgeon"`). The advice block is
    // forwarded verbatim into the Surgeon payload so the agent can frame
    // its repair around the architectural / scope verdict.
    retrospectorAdvice?: RetrospectorAdvice;
    isFinalAttempt?: boolean;
  } = {},
): Promise<RtlAgentRunResult> {
  await appendRunLog(
    {
      event: "action",
      action: "invoke_surgeon",
      module_id: brokenModule.module_id,
      ...(options.isFinalAttempt ? { final_attempt: true } : {}),
      ...(options.retrospectorAdvice ? { retrospector_advice: options.retrospectorAdvice } : {}),
    },
    runtime,
  );

  const prior_attempts = priorSurgeonAttempts(brokenModule.module_id);
  const retrySeed = [
    brokenModule.module_id,
    brokenModule.spec_hash,
    prior_attempts.length + 1,
    runtime.now().toISOString(),
  ].join(":");
  const trimmedVerif = trimVerifResultForSurgeon(verifResult);

  // Mirror the doc-coverage guard from `invokeFoundry`: only ask Surgeon for
  // a `draft_doc` when no existing pattern doc already covers this layer's
  // contract+op+kernel. See `findCoveringDoc`.
  const surgeonDocLifecycleState = options.selfImproveEnabled
    ? await loadDocLifecycleState()
    : null;
  const surgeonCoveringDoc = surgeonDocLifecycleState
    ? findCoveringDoc(surgeonDocLifecycleState, layerIr)
    : null;
  const surgeonAsksForDraftDoc =
    options.selfImproveEnabled === true && surgeonCoveringDoc === null;
  if (options.selfImproveEnabled && surgeonCoveringDoc !== null) {
    await appendRunLog(
      {
        event: "self_improve_doc_request_skipped",
        agent: "Surgeon",
        module_id: layerIr.module_id,
        contract_id: currentContractId(layerIr),
        contract_key: contractStateKeyForLayer(layerIr),
        covering_doc_tier: surgeonCoveringDoc.tier,
        covering_doc_path: surgeonCoveringDoc.path,
        ...(surgeonCoveringDoc.doc_id ? { covering_doc_id: surgeonCoveringDoc.doc_id } : {}),
        reason: "existing_doc_covers_contract_op_kernel",
      },
      runtime,
    );
  }

  const preloadedRtlPatterns = await loadRetrospectorKnowledgeDoc(layerIr);
  // Foundry's prior attempts on this same (module, contract). Surgeon needs
  // to see *what Foundry already tried and how each attempt failed* so it
  // doesn't repeat a fix shape that's already proven not to work. These come
  // from a separate agent (Foundry); Surgeon receives them as user-provided
  // evidence in its own conversation.
  const priorFoundryAttempts = foundryVersionsFor(layerIr).map((version, idx) => {
    const failureForVersion = failureAttemptsFor(layerIr).find(
      (entry) => entry.module?.attempt === version.module.attempt,
    );
    return {
      foundry_attempt_index: idx + 1,
      module_attempt: version.module.attempt,
      generated_by: version.module.generated_by,
      verilog_source: version.module.verilog_source,
      tool_use_summary: version.tool_use_summary,
      failure_summary: failureForVersion
        ? trimVerifResultForSurgeon(failureForVersion.result)
        : undefined,
    };
  });
  const failureMemory = await failureMemoryForLayer(layerIr);
  const referenceEvidence = await maybeComputeLayerReferenceEvidence(layerIr, verifResult, runtime);
  const surgeonPayload = {
    broken_module: brokenModule,
    verif_result: trimmedVerif,
    layer_ir: layerIr,
    expected_spec_hash: computeExpectedSpecHash(layerIr),
    preloaded_rtl_patterns: preloadedRtlPatterns,
    prior_attempts,
    prior_foundry_attempts: priorFoundryAttempts,
    ...(failureMemory.length > 0 ? { failure_memory: failureMemory } : {}),
    ...(referenceEvidence ? { reference_evidence: referenceEvidence } : {}),
    ...(options.retrospectorAdvice ? { retrospector_advice: options.retrospectorAdvice } : {}),
    ...(options.isFinalAttempt ? { is_final_attempt: true } : {}),
    retry_seed: retrySeed,
    write_verilog_output_dir: resolvePipelineConfigPath(PIPELINE_CONFIG.rtl_dir),
    ...(surgeonAsksForDraftDoc
      ? {
          self_improve_doc_request: {
            enabled: true,
            destination_tier: "probationary",
            promotion_successes_required: PIPELINE_CONFIG.doc_promotion_success_threshold,
            replacement_for_doc_ids: [],
          },
        }
      : {}),
  };

  let result: RtlAgentRunResult;
  try {
    if (surgeonAsksForDraftDoc) {
      const withDoc = await runDelegatedAgent<RtlAgentWithDoc>(
        "surgeon",
        surgeonPayload,
        rtlAgentWithDocOutputFormat,
        rtlAgentWithDocZod,
        runtime,
      );
      const hydratedModule = await hydrateVerilogModuleFromDisk(
        withDoc.payload.module,
        layerIr,
      );
      result = {
        payload: hydratedModule,
        draft_doc: withDoc.payload.draft_doc,
        result: withDoc.result,
        messages: withDoc.messages,
      };
    } else {
      const plainResult = await runDelegatedAgent<VerilogModuleAgentOutput>(
        "surgeon",
        surgeonPayload,
        verilogModuleAgentOutputFormat,
        verilogModuleAgentOutputZod,
        runtime,
      );
      const hydratedModule = await hydrateVerilogModuleFromDisk(
        plainResult.payload,
        layerIr,
      );
      result = {
        payload: hydratedModule,
        result: plainResult.result,
        messages: plainResult.messages,
      };
    }
  } catch (err) {
    if (err instanceof SpecHashMismatchError) {
      throw err;
    }
    const recovered = await tryRecoverVerilogModuleFromDisk(
      layerIr,
      "Surgeon",
      /* attempt */ Math.max(brokenModule.attempt + 1, 2),
    );
    if (!recovered) {
      throw err;
    }
    // Same metadata-preservation as invokeFoundry's recovery path: keep the
    // SDK turn's cost / session / modelUsage / messages even when the final
    // structured output failed to parse. See `StructuredOutputParseError`.
    const carried =
      err instanceof StructuredOutputParseError
        ? { result: err.result, messages: err.messages }
        : {
            result: {
              type: "result",
              subtype: "success",
              result: "",
              total_cost_usd: 0,
              modelUsage: {},
            } as unknown as SDKResultMessage,
            messages: [] as SDKMessage[],
          };
    let scrapedDraft: DocDraft | null = null;
    if (surgeonAsksForDraftDoc && err instanceof StructuredOutputParseError) {
      const rawText =
        err.result.subtype === "success" && typeof err.result.result === "string"
          ? err.result.result
          : "";
      const scraped = scrapeDraftDocFromText(rawText);
      if (scraped !== null) {
        const parsed = docDraftZod.safeParse(scraped);
        if (parsed.success) {
          scrapedDraft = parsed.data;
        }
      }
    }
    await appendRunLog(
      {
        event: "agent_result_recovered",
        agent: "Surgeon",
        module_id: brokenModule.module_id,
        reason: err instanceof Error ? err.message : String(err),
        carried_cost_usd:
          (carried.result as { total_cost_usd?: number }).total_cost_usd ?? 0,
        carried_session_id:
          (carried.result as { session_id?: string }).session_id ?? null,
        carried_message_count: carried.messages.length,
        scraped_draft_doc: scrapedDraft !== null,
      },
      runtime,
    );
    result = {
      payload: recovered,
      ...(scrapedDraft !== null ? { draft_doc: scrapedDraft } : {}),
      result: carried.result,
      messages: carried.messages,
    };
  }

  await persistVerilogModule(result.payload);

  const surgeonAudits = extractToolUseAudits(result.messages, {
    agent: "Surgeon",
    module_id: brokenModule.module_id,
    nowIso: runtime.now().toISOString(),
  });
  if (options.selfImproveEnabled) {
    await recordDocUsageForAgent(layerIr, brokenModule.module_id, surgeonAudits, runtime);
  }
  await appendToolUseAudits(surgeonAudits);
  await appendForeignMcpToolWarnings(surgeonAudits, runtime);
  await appendRunLog(
    {
      event: "agent_tool_use_summary",
      agent: "Surgeon",
      module_id: brokenModule.module_id,
      ...summarizeToolUse(surgeonAudits),
    },
    runtime,
  );

  await appendRunLog(
    {
      event: "agent_result",
      agent: "Surgeon",
      module_id: brokenModule.module_id,
      total_cost_usd: result.result.total_cost_usd,
      modelUsage: result.result.modelUsage,
      payload: result.payload,
    },
    runtime,
  );

  return result;
}

async function attemptNextContractOrEscalate(input: {
  manager: PipelineStateManager;
  moduleId: string;
  baseLayer: LayerIR;
  currentLayer: LayerIR;
  pipelineIr: PipelineIR;
  activeLayers: Map<string, LayerIR>;
  newDocFailureContexts: Map<string, NewDocFailureContext>;
  contractState: ContractResponseState;
  statePath: string;
  runtime: OrchestratorRuntime;
  result: VerifResult;
  reason: string;
}): Promise<boolean> {
  const currentPlan = contractPlanForLayer(input.currentLayer);
  if (!shouldPersistContractManualCorrection(input.reason, input.result)) {
    const before = input.manager.getState().modules[input.moduleId];
    const nonContractResult: VerifResult = {
      ...input.result,
      module_id: input.moduleId,
      status: "fail",
      failure_category:
        input.result.failure_category ??
        (isToolchainInfrastructureFailure(input.result) ? "toolchain_infra" : "unknown"),
      violated_constraint: input.result.violated_constraint ?? input.reason,
      classifier_reason:
        input.result.classifier_reason ??
        "Failure was not persisted as contract manual-correction state because it was classified as infrastructure, tool/API availability, or quota-related.",
    };
    input.manager.applyVerifResult(input.moduleId, nonContractResult);
    await logStateTransition(
      input.manager,
      input.moduleId,
      before,
      input.manager.getState().modules[input.moduleId],
      "contract_manual_correction_suppressed",
      input.runtime,
    );
    await input.manager.saveState(input.statePath);
    await appendRunLog(
      {
        event: "contract_manual_correction_suppressed",
        module_id: input.moduleId,
        contract_id: currentPlan.id,
        contract_key: contractStateKeyForLayer(input.currentLayer),
        reason: input.reason,
        result: nonContractResult,
      },
      input.runtime,
    );
    return true;
  }

  // Code-bug failures don't justify a contract walk. Walking to a different
  // contract on a code_bug failure (agent forgot to register a window, broke
  // a bus width while fixing rounding, etc.) just gives the agent a fresh
  // chance to make new bugs on a different interface — observed on
  // node_conv_292 (2026-05-06): all 8 dispatches across dram-backed-weights
  // → weight-tiling were code_bug failures, walking from one to the other
  // wasted ~$5–6 chasing the same agent-side mistakes and discarded the
  // failure-corpus signal we'd built up on the original contract.
  //
  // Escalate to manual review without flagging the contract — the contract
  // isn't the problem, the agents' RTL is. The human-review flow can decide
  // whether to retry under a different model / give the agent more
  // protected docs / accept the module as unsynthesizable.
  if (input.result.failure_category === "code_bug") {
    const manualResult: VerifResult = {
      ...input.result,
      module_id: input.moduleId,
      status: "fail",
      failure_class: "manual_correction_needed",
      violated_constraint:
        input.result.violated_constraint ?? "code_bug_after_retrospector_exhausted",
      classifier_reason:
        "Failure category is code_bug. Walking to another contract would not address agent-level RTL mistakes; escalating to manual review without flagging the current contract.",
    };
    const before = input.manager.getState().modules[input.moduleId];
    input.manager.applyVerifResult(input.moduleId, manualResult);
    await logStateTransition(
      input.manager,
      input.moduleId,
      before,
      input.manager.getState().modules[input.moduleId],
      "code_bug_no_contract_walk",
      input.runtime,
    );
    await input.manager.saveState(input.statePath);
    await appendRunLog(
      {
        event: "human_escalation_required",
        module_id: input.moduleId,
        contract_key: contractStateKeyForLayer(input.currentLayer),
        reason: "code_bug_after_retrospector_exhausted",
        result: manualResult,
      },
      input.runtime,
    );
    return true;
  }

  await flagContractForManualCorrection(
    input.contractState,
    input.currentLayer,
    input.moduleId,
    input.reason,
    input.result,
    input.runtime,
  );

  const next = selectAvailableContract(
    input.baseLayer,
    input.contractState,
    currentPlan.id,
  );
  await saveContractResponseState(input.contractState);

  if (next) {
    const before = input.manager.getState().modules[input.moduleId];
    const nextLayer = withSignatureMetadata(input.baseLayer, next.layer, input.pipelineIr.quantization);
    setActiveLayerForModule(
      input.pipelineIr,
      input.activeLayers,
      input.moduleId,
      nextLayer,
    );
    input.manager.resetModuleForContractRetry(input.moduleId);
    input.newDocFailureContexts.set(input.moduleId, {
      reason: "contract_switch_after_failure",
      previous_contract_id: currentPlan.id,
      previous_contract_key: contractStateKeyForLayer(input.currentLayer),
      failure_result: input.result,
    });
    await logStateTransition(
      input.manager,
      input.moduleId,
      before,
      "pending",
      `contract_switch_${currentPlan.id}_to_${next.plan.id}`,
      input.runtime,
    );
    await appendRunLog(
      {
        event: "contract_alternative_selected",
        module_id: input.moduleId,
        previous_contract_id: currentPlan.id,
        next_contract_id: next.plan.id,
        previous_contract_key: contractStateKeyForLayer(input.currentLayer),
        next_contract_key: contractStateKeyForLayer(nextLayer),
        next_layer_ir: nextLayer,
        available_contracts: CONTRACT_PLANS,
      },
      input.runtime,
    );
    await input.manager.saveState(input.statePath);
    return true;
  }

  const manualResult: VerifResult = {
    ...input.result,
    module_id: input.moduleId,
    status: "fail",
    failure_class: "manual_correction_needed",
    failure_category: "unknown",
    violated_constraint: "all_available_contracts_exhausted",
    classifier_reason:
      "All available contracts for this structural spec are flagged manual_correction_needed. Manual intervention is required before this contract can run again.",
  };
  const before = input.manager.getState().modules[input.moduleId];
  input.manager.applyVerifResult(input.moduleId, manualResult);
  await logStateTransition(
    input.manager,
    input.moduleId,
    before,
    input.manager.getState().modules[input.moduleId],
    "all_contracts_exhausted",
    input.runtime,
  );
  await input.manager.saveState(input.statePath);
  await appendRunLog(
    {
      event: "human_escalation_required",
      module_id: input.moduleId,
      contract_key: contractStateKeyForLayer(input.currentLayer),
      reason: "all_available_contracts_exhausted",
      result: manualResult,
    },
    input.runtime,
  );
  return true;
}

/**
 * Surgeon-led variant of the post-Retrospector final attempt. Picks a base
 * artifact according to Retrospector's `base_artifact` choice (or the
 * default `best_known`), then dispatches Surgeon with the full advice
 * block. Replaces the unconditional Foundry-resume that previously ran on
 * every retrospector verdict — Surgeon-shaped repair work was being
 * thrown away because Foundry would regenerate from scratch.
 */
async function runRetrospectorSurgeonFinalAttempt(input: {
  manager: PipelineStateManager;
  moduleId: string;
  layer: LayerIR;
  baseLayer: LayerIR;
  pipelineIr: PipelineIR;
  activeLayers: Map<string, LayerIR>;
  newDocFailureContexts: Map<string, NewDocFailureContext>;
  contractState: ContractResponseState;
  terminalResult: VerifResult;
  statePath: string;
  runtime: OrchestratorRuntime;
  passedModules: Map<string, { module: VerilogModule; layer: LayerIR }>;
  selfImproveEnabled: boolean;
  advice: RetrospectorAdvice;
  baseArtifactChoice: RetrospectorBaseArtifact;
  replacementForDocIds: string[];
}): Promise<boolean> {
  const {
    manager,
    moduleId,
    layer,
    baseLayer,
    pipelineIr,
    activeLayers,
    newDocFailureContexts,
    contractState,
    terminalResult,
    statePath,
    runtime,
    passedModules,
    selfImproveEnabled,
    advice,
    baseArtifactChoice,
    replacementForDocIds,
  } = input;

  // Resolve the base artifact Surgeon will repair. `fresh` falls back to
  // the latest persisted module — Surgeon cannot work without a starting
  // module — and the orchestrator logs the downgrade so the policy choice
  // remains auditable.
  let baseModule: VerilogModule | null = null;
  let baseVerif: VerifResult = terminalResult;
  let baseSource: "best_known" | "latest" | "fresh_downgraded_to_latest" = "latest";

  if (baseArtifactChoice === "best_known") {
    const best = pickBestKnownAttempt(layer);
    if (best) {
      baseModule = best.module;
      baseVerif = best.result;
      baseSource = "best_known";
    }
  }
  if (baseModule === null) {
    try {
      baseModule = await loadPersistedVerilogModule(moduleId);
      if (baseArtifactChoice === "fresh") {
        baseSource = "fresh_downgraded_to_latest";
      } else if (baseArtifactChoice !== "best_known") {
        baseSource = "latest";
      }
    } catch (error: unknown) {
      await appendRunLog(
        {
          event: "retrospector_surgeon_dispatch_failed",
          module_id: moduleId,
          stage: "load_base_artifact",
          base_artifact: baseArtifactChoice,
          reason: error instanceof Error ? error.message : String(error),
        },
        runtime,
      );
      return attemptNextContractOrEscalate({
        manager,
        moduleId,
        baseLayer,
        currentLayer: layer,
        pipelineIr,
        activeLayers,
        newDocFailureContexts,
        contractState,
        statePath,
        runtime,
        result: terminalResult,
        reason: "retrospector_surgeon_no_base_artifact",
      });
    }
  }

  await appendRunLog(
    {
      event: "retrospector_surgeon_base_artifact_selected",
      module_id: moduleId,
      base_artifact_choice: baseArtifactChoice,
      base_artifact_source: baseSource,
      base_module_attempt: baseModule.attempt,
      base_module_generated_by: baseModule.generated_by,
      base_verif_status: baseVerif.status,
      base_verif_exact_match_count: baseVerif.exact_match_count ?? null,
      base_verif_max_error: baseVerif.max_error ?? null,
    },
    runtime,
  );

  const beforeGenerate = manager.getState().modules[moduleId];
  manager.setStatus(moduleId, "generating");
  await logStateTransition(
    manager,
    moduleId,
    beforeGenerate,
    "generating",
    "retrospector_surgeon_dispatch",
    runtime,
  );
  await manager.saveState(statePath);

  let surgeonResult: RtlAgentRunResult;
  try {
    surgeonResult = await invokeSurgeon(baseModule, baseVerif, layer, runtime, {
      selfImproveEnabled,
      retrospectorAdvice: advice,
      isFinalAttempt: true,
    });
    recordUsageFromResult(manager, surgeonResult.result);
  } catch (error: unknown) {
    const finalFailure: VerifResult = {
      module_id: moduleId,
      status: "fail",
      timing_pass: false,
      timing_actual_cycles: 0,
      timing_expected_cycles: layer.pipeline_latency_cycles,
      failure_category: "unknown",
      violated_constraint: "retrospector_surgeon_dispatch_failed",
      classifier_reason: "Surgeon's final post-retrospector attempt failed before producing RTL.",
      fix_hint: error instanceof Error ? error.message : String(error),
    };
    await recordFailureAttempt(layer, "retrospector_surgeon_dispatch", finalFailure, null, runtime);
    await archiveProbationaryDocsForFailure(
      layer,
      moduleId,
      "retrospector_surgeon_dispatch_failed",
      runtime,
    );
    const beforeApply = manager.getState().modules[moduleId];
    manager.applyVerifResult(moduleId, finalFailure);
    await logStateTransition(
      manager,
      moduleId,
      beforeApply,
      manager.getState().modules[moduleId],
      "retrospector_surgeon_dispatch_failed",
      runtime,
    );
    await manager.saveState(statePath);
    return attemptNextContractOrEscalate({
      manager,
      moduleId,
      baseLayer,
      currentLayer: layer,
      pipelineIr,
      activeLayers,
      newDocFailureContexts,
      contractState,
      statePath,
      runtime,
      result: finalFailure,
      reason: "retrospector_surgeon_dispatch_failed",
    });
  }

  await manager.saveState(statePath);
  const beforeVerify = manager.getState().modules[moduleId];
  manager.setStatus(moduleId, "verifying");
  await logStateTransition(
    manager,
    moduleId,
    beforeVerify,
    "verifying",
    "retrospector_surgeon_completed",
    runtime,
  );
  await manager.saveState(statePath);

  const rawVerif = await invokeAssayer(surgeonResult.payload, layer, runtime);
  const finalVerif = await classifyFailedModule(
    manager,
    rawVerif,
    layer,
    surgeonResult.payload,
    runtime,
  );
  if (finalVerif.status !== "pass") {
    await recordFailureAttempt(
      layer,
      "retrospector_surgeon_assayer",
      finalVerif,
      surgeonResult.payload,
      runtime,
    );
    await archiveProbationaryDocsForFailure(
      layer,
      moduleId,
      `retrospector_surgeon_assayer_${finalVerif.status}`,
      runtime,
    );
  }

  const beforeApply = manager.getState().modules[moduleId];
  manager.applyVerifResult(moduleId, finalVerif);
  const afterApply = manager.getState().modules[moduleId];
  await logStateTransition(
    manager,
    moduleId,
    beforeApply,
    afterApply,
    `retrospector_surgeon_assayer_${finalVerif.status}`,
    runtime,
  );
  await manager.saveState(statePath);

  if (afterApply === "pass") {
    await processSynthesisOutcome(
      manager,
      moduleId,
      surgeonResult.payload,
      layer,
      finalVerif,
      statePath,
      runtime,
      selfImproveEnabled,
    );
    if (manager.getState().modules[moduleId] === "pass") {
      passedModules.set(moduleId, { module: surgeonResult.payload, layer });
      await finalizeSuccessfulRtlDocs(
        surgeonResult.payload,
        layer,
        surgeonResult.draft_doc,
        runtime,
        selfImproveEnabled,
        replacementForDocIds,
        surgeonResult.doc_request ?? null,
      );
    } else {
      const synthesisResult = manager.getState().results[moduleId] ?? finalVerif;
      return attemptNextContractOrEscalate({
        manager,
        moduleId,
        baseLayer,
        currentLayer: layer,
        pipelineIr,
        activeLayers,
        newDocFailureContexts,
        contractState,
        statePath,
        runtime,
        result: synthesisResult,
        reason: "retrospector_surgeon_synthesis_failed",
      });
    }
    return true;
  }

  return attemptNextContractOrEscalate({
    manager,
    moduleId,
    baseLayer,
    currentLayer: layer,
    pipelineIr,
    activeLayers,
    newDocFailureContexts,
    contractState,
    statePath,
    runtime,
    result: finalVerif,
    reason: "retrospector_surgeon_final_attempt_failed",
  });
}

async function maybeRunRetrospectorFinalAttempt(
  manager: PipelineStateManager,
  moduleId: string,
  layer: LayerIR,
  baseLayer: LayerIR,
  pipelineIr: PipelineIR,
  activeLayers: Map<string, LayerIR>,
  newDocFailureContexts: Map<string, NewDocFailureContext>,
  contractState: ContractResponseState,
  terminalResult: VerifResult,
  statePath: string,
  runtime: OrchestratorRuntime,
  passedModules: Map<string, { module: VerilogModule; layer: LayerIR }>,
  selfImproveEnabled: boolean,
): Promise<boolean> {
  if (!selfImproveEnabled) return false;
  if (terminalResult.status === "pass") return false;

  const state = manager.getState();
  if (state.modules[moduleId] !== "fail_abort") return false;
  const retryBudgetExhausted =
    terminalResult.failure_category === "code_bug" &&
    (state.attempts[moduleId] ?? 0) >= state.max_retries;
  const architecturalFit = terminalResult.failure_category === "architectural_fit";
  if (!retryBudgetExhausted && !architecturalFit) return false;

  const contractKey = moduleContractKey(layer);
  if (manager.retrospectorCallCount(contractKey) >= 1) {
    await appendRunLog(
      {
        event: "retrospector_skipped",
        module_id: moduleId,
        contract_key: contractKey,
        reason: "retrospector_already_used_for_module_contract",
      },
      runtime,
    );
    return attemptNextContractOrEscalate({
      manager,
      moduleId,
      baseLayer,
      currentLayer: layer,
      pipelineIr,
      activeLayers,
      newDocFailureContexts,
      contractState,
      statePath,
      runtime,
      result: terminalResult,
      reason: "retrospector_already_used_for_module_contract",
    });
  }

  const resumeSessionId = latestFoundrySessionId(layer);

  let input: RetrospectorInput;
  try {
    const docState = await loadDocLifecycleState();
    input = {
      original_spec: layer,
      contract: buildModuleContractSummary(layer),
      current_contract: contractPlanForLayer(layer),
      available_contracts: CONTRACT_PLANS,
      doc_used: await loadRetrospectorKnowledgeDoc(layer),
      knowledge_docs_used: lifecycleDocsForLayer(docState, layer, ["active", "probationary"]),
      foundry_versions: foundryVersionsFor(layer),
      failure_attempts: failureAttemptsFor(layer),
      failure_corpus: await failureMemoryForLayer(layer, 10),
    };
  } catch (error: unknown) {
    await appendRunLog(
      {
        event: "retrospector_failed",
        module_id: moduleId,
        contract_key: contractKey,
        stage: "load_doc",
        reason: error instanceof Error ? error.message : String(error),
      },
      runtime,
    );
    // Retrospector died on infrastructure (doc load), not on a real
    // architectural finding. Don't strand the module — the underlying
    // terminal failure is still real, so let the contract walk proceed
    // (next contract or `manual_correction_needed` if all flagged).
    return attemptNextContractOrEscalate({
      manager,
      moduleId,
      baseLayer,
      currentLayer: layer,
      pipelineIr,
      activeLayers,
      newDocFailureContexts,
      contractState,
      statePath,
      runtime,
      result: terminalResult,
      reason: "retrospector_failed_load_doc",
    });
  }

  manager.recordRetrospectorCall(contractKey);
  await manager.saveState(statePath);

  let adviceRun: AgentRunResult<RetrospectorAdvice>;
  try {
    adviceRun = await invokeRetrospector(input, runtime);
    recordUsageFromResult(manager, adviceRun.result);
  } catch (error: unknown) {
    await appendRunLog(
      {
        event: "retrospector_failed",
        module_id: moduleId,
        contract_key: contractKey,
        stage: "invoke_retrospector",
        reason: error instanceof Error ? error.message : String(error),
      },
      runtime,
    );
    await manager.saveState(statePath);
    // Same reasoning as the load-doc fail above: a retrospector outage
    // (API limit, dispatch crash, classifier unavailable) is not evidence
    // that the contract is impossible — fall through to the contract walk
    // so the module either picks a new contract or correctly escalates to
    // `manual_correction_needed` if all are flagged.
    return attemptNextContractOrEscalate({
      manager,
      moduleId,
      baseLayer,
      currentLayer: layer,
      pipelineIr,
      activeLayers,
      newDocFailureContexts,
      contractState,
      statePath,
      runtime,
      result: terminalResult,
      reason: "retrospector_failed_invoke",
    });
  }

  await appendRunLog(
    {
      event: "retrospector_result",
      module_id: moduleId,
      contract_key: contractKey,
      resume_session_id: resumeSessionId,
      total_cost_usd: adviceRun.result.total_cost_usd,
      modelUsage: adviceRun.result.modelUsage,
      advice: adviceRun.payload,
    },
    runtime,
  );
  await manager.saveState(statePath);

  const replacementForDocIds = await archiveActiveDocsConfirmedByRetrospector(
    layer,
    moduleId,
    adviceRun.payload,
    runtime,
  );

  // Decide who runs the post-Retrospector final attempt. Default policy:
  // **Surgeon** preserves a near-passing artifact and applies a localized
  // fix; Retrospector picks Foundry only when the architecture or contract
  // itself must change. Older Retrospector outputs that pre-date the
  // `next_actor` field default to "surgeon" too — the previous always-
  // Foundry behaviour discarded mostly-good RTL on every final attempt
  // (see node_conv_288 case study: 98%-correct module rebuilt from scratch
  // and regressed). To force the legacy Foundry path, Retrospector must
  // emit `next_actor: "foundry"` explicitly with an architecture-level
  // `repair_scope` (`architecture_replacement` / `interface_or_contract_fix`).
  const advice = adviceRun.payload;
  const baseArtifactDefault: RetrospectorBaseArtifact =
    advice.next_actor === "foundry" ? "fresh" : "best_known";
  const baseArtifactChoice: RetrospectorBaseArtifact =
    advice.base_artifact ?? baseArtifactDefault;
  const nextActor: RetrospectorNextActor = advice.next_actor ?? "surgeon";

  await appendRunLog(
    {
      event: "retrospector_final_attempt_routing",
      module_id: moduleId,
      contract_key: contractKey,
      next_actor: nextActor,
      base_artifact: baseArtifactChoice,
      repair_scope: advice.repair_scope ?? null,
      retrospector_specified_actor: advice.next_actor ?? null,
    },
    runtime,
  );

  if (nextActor === "surgeon") {
    return runRetrospectorSurgeonFinalAttempt({
      manager,
      moduleId,
      layer,
      baseLayer,
      pipelineIr,
      activeLayers,
      newDocFailureContexts,
      contractState,
      terminalResult,
      statePath,
      runtime,
      passedModules,
      selfImproveEnabled,
      advice,
      baseArtifactChoice,
      replacementForDocIds,
    });
  }

  if (!resumeSessionId) {
    await appendRunLog(
      {
        event: "retrospector_no_resumable_foundry_session",
        module_id: moduleId,
        contract_key: contractKey,
        reason: "Retrospector routed the final retry to Foundry, but no Foundry session exists for this contract. The orchestrator will try the next available contract instead.",
      },
      runtime,
    );
    return attemptNextContractOrEscalate({
      manager,
      moduleId,
      baseLayer,
      currentLayer: layer,
      pipelineIr,
      activeLayers,
      newDocFailureContexts,
      contractState,
      statePath,
      runtime,
      result: terminalResult,
      reason: "retrospector_no_resumable_foundry_session",
    });
  }

  const beforeGenerate = manager.getState().modules[moduleId];
  manager.setStatus(moduleId, "generating");
  await logStateTransition(
    manager,
    moduleId,
    beforeGenerate,
    "generating",
    "retrospector_advice_injected",
    runtime,
  );
  await manager.saveState(statePath);

  // Load Surgeon's last attempt (the one that just failed and triggered the
  // retrospector). Pass it to Foundry as user-message evidence in the resumed
  // session so Foundry sees the Surgeon's repaired RTL and the verifier's
  // verdict on it — never as a fake assistant turn from Foundry itself.
  let surgeonAttempt: { module: VerilogModule; verifResult: VerifResult } | undefined;
  try {
    const surgeonModule = await loadPersistedVerilogModule(moduleId);
    if (surgeonModule.generated_by === "Surgeon") {
      surgeonAttempt = { module: surgeonModule, verifResult: terminalResult };
    }
  } catch {
    // No persisted module → no surgeon evidence to forward.
  }

  let foundryResult: RtlAgentRunResult;
  try {
    foundryResult = await invokeFoundry(layer, runtime, {
      resumeSessionId,
      retrospectorAdvice: advice,
      surgeonAttempt,
      isFinalAttempt: true,
      selfImproveEnabled,
      replacementForDocIds,
      newDocFailureContext: newDocFailureContexts.get(moduleId),
    });
    recordUsageFromResult(manager, foundryResult.result);
  } catch (error: unknown) {
    const finalFailure: VerifResult = {
      module_id: moduleId,
      status: "fail",
      timing_pass: false,
      timing_actual_cycles: 0,
      timing_expected_cycles: layer.pipeline_latency_cycles,
      failure_category: "unknown",
      violated_constraint: "retrospector_foundry_dispatch_failed",
      classifier_reason: "Foundry's final post-retrospector attempt failed before producing RTL.",
      fix_hint: error instanceof Error ? error.message : String(error),
    };
    await recordFailureAttempt(layer, "retrospector_foundry_dispatch", finalFailure, null, runtime);
    await archiveProbationaryDocsForFailure(
      layer,
      moduleId,
      "retrospector_foundry_dispatch_failed",
      runtime,
    );
    const beforeApply = manager.getState().modules[moduleId];
    manager.applyVerifResult(moduleId, finalFailure);
    await logStateTransition(
      manager,
      moduleId,
      beforeApply,
      manager.getState().modules[moduleId],
      "retrospector_foundry_dispatch_failed",
      runtime,
    );
    await manager.saveState(statePath);
    return attemptNextContractOrEscalate({
      manager,
      moduleId,
      baseLayer,
      currentLayer: layer,
      pipelineIr,
      activeLayers,
      newDocFailureContexts,
      contractState,
      statePath,
      runtime,
      result: finalFailure,
      reason: "retrospector_foundry_dispatch_failed",
    });
  }

  await manager.saveState(statePath);
  const beforeVerify = manager.getState().modules[moduleId];
  manager.setStatus(moduleId, "verifying");
  await logStateTransition(
    manager,
    moduleId,
    beforeVerify,
    "verifying",
    "retrospector_foundry_completed",
    runtime,
  );
  await manager.saveState(statePath);

  const rawVerif = await invokeAssayer(foundryResult.payload, layer, runtime);
  const finalVerif = await classifyFailedModule(
    manager,
    rawVerif,
    layer,
    foundryResult.payload,
    runtime,
  );
  if (finalVerif.status !== "pass") {
    await recordFailureAttempt(
      layer,
      "retrospector_foundry_assayer",
      finalVerif,
      foundryResult.payload,
      runtime,
    );
    await archiveProbationaryDocsForFailure(
      layer,
      moduleId,
      `retrospector_assayer_${finalVerif.status}`,
      runtime,
    );
  }

  const beforeApply = manager.getState().modules[moduleId];
  manager.applyVerifResult(moduleId, finalVerif);
  const afterApply = manager.getState().modules[moduleId];
  await logStateTransition(
    manager,
    moduleId,
    beforeApply,
    afterApply,
    `retrospector_assayer_${finalVerif.status}`,
    runtime,
  );
  await manager.saveState(statePath);

  if (afterApply === "pass") {
    await processSynthesisOutcome(
      manager,
      moduleId,
      foundryResult.payload,
      layer,
      finalVerif,
      statePath,
      runtime,
      selfImproveEnabled,
    );
    if (manager.getState().modules[moduleId] === "pass") {
      passedModules.set(moduleId, { module: foundryResult.payload, layer });
      await finalizeSuccessfulRtlDocs(
        foundryResult.payload,
        layer,
        foundryResult.draft_doc,
        runtime,
        selfImproveEnabled,
        replacementForDocIds,
        foundryResult.doc_request ?? null,
      );
    } else {
      const synthesisResult = manager.getState().results[moduleId] ?? finalVerif;
      return attemptNextContractOrEscalate({
        manager,
        moduleId,
        baseLayer,
        currentLayer: layer,
        pipelineIr,
        activeLayers,
        newDocFailureContexts,
        contractState,
        statePath,
        runtime,
        result: synthesisResult,
        reason: "retrospector_foundry_synthesis_failed",
      });
    }
    return true;
  }

  return attemptNextContractOrEscalate({
    manager,
    moduleId,
    baseLayer,
    currentLayer: layer,
    pipelineIr,
    activeLayers,
    newDocFailureContexts,
    contractState,
    statePath,
    runtime,
    result: finalVerif,
    reason: "retrospector_final_attempt_failed",
  });
}

// ---------------------------------------------------------------------------
// Spec-hash template reuse — skip Foundry entirely when a structurally
// identical module has already passed verification.
//
// Two modules share a spec_hash when they have the same op_type, channel
// counts, kernel dimensions, and bus widths. Their RTL structure is identical;
// only the module name, $readmemh paths, and scale-factor constants differ.
// We clone the passing module's Verilog, substitute those three things, then
// run the assayer directly. This saves ~60 % of Foundry LLM calls on layer1.
// ---------------------------------------------------------------------------

/** Compute the structural spec_hash from LayerIR fields (no scale factor).
 *
 * Two modules share a spec_hash when an identical Verilog template can be
 * reused for both — the line-buffer / window-shift datapath of a spatial
 * conv depends on `IH`/`IW`, so a 3×3 conv on 112×112 must NOT be cloned
 * for a 3×3 conv on 56×56. Include input spatial dims for every op; include
 * MaxPool kernel/stride/padding as well since those also parameterise its
 * datapath.
 */
export function computeExpectedSpecHash(layer: LayerIR): string {
  const ic = layer.input_shape.length >= 2 ? layer.input_shape[1] : 0;
  const oc = layer.output_shape.length >= 2 ? layer.output_shape[1] : 0;
  const ih = layer.input_shape.length >= 3 ? layer.input_shape[2] : 0;
  const iw = layer.input_shape.length >= 4 ? layer.input_shape[3] : 0;
  const spatial = `s${ih}x${iw}`;
  const contractSuffix =
    currentContractId(layer) === "flat-bus"
      ? ""
      : `_io${currentContractId(layer)}_tile${layer.channel_tile ?? "auto"}`;
  if (layer.op_type === "conv2d" && layer.weight_shape.length >= 4) {
    const kh = layer.weight_shape[2];
    const kw = layer.weight_shape[3];
    const stride = layer.stride && layer.stride.length >= 2 ? `_st${layer.stride[0]}x${layer.stride[1]}` : "";
    const padding =
      layer.padding && layer.padding.length >= 2 ? `_p${layer.padding[0]}x${layer.padding[1]}` : "";
    const dilation =
      layer.dilation && layer.dilation.length >= 2 ? `_d${layer.dilation[0]}x${layer.dilation[1]}` : "";
    const groups = layer.groups ? `_g${layer.groups}` : "";
    // mac_parallelism affects the FSM's OC-group iteration, so two layers
    // with identical geometry but different mac_parallelism have structurally
    // different RTL and MUST NOT be clone-substituted for each other.
    const mp = layer.mac_parallelism ? `_mp${layer.mac_parallelism}` : "";
    return `conv2d_${ic}x${oc}x${kh}x${kw}_${spatial}${stride}${padding}${dilation}${groups}${mp}_i${layer.input_width_bits}_o${layer.output_width_bits}${contractSuffix}`;
  }
  if (layer.op_type === "maxpool") {
    // The schema's superRefine guarantees these three arrays exist and are
    // at least 2-long for maxpool layers. A null-safe fallback here would
    // mask schema regressions silently.
    if (!layer.kernel_size || !layer.pool_stride || !layer.pool_padding) {
      throw new Error(
        `spec_hash for maxpool layer '${layer.module_id}' requires kernel_size, pool_stride, pool_padding — schema should have rejected this upstream.`,
      );
    }
    const ks = layer.kernel_size.join("x");
    const st = layer.pool_stride.join("x");
    const pd = layer.pool_padding.join("x");
    return `maxpool_${ic}x${oc}_k${ks}_s${st}_p${pd}_${spatial}_i${layer.input_width_bits}_o${layer.output_width_bits}${contractSuffix}`;
  }
  return `${layer.op_type}_${ic}x${oc}_${spatial}_i${layer.input_width_bits}_o${layer.output_width_bits}${contractSuffix}`;
}

/** Choose SCALE_MULT/SCALE_SHIFT that minimise the relative approximation error.
 *
 * Shift range is 0..23. The 0-end is needed for deep-network layers whose
 * scale_factor exceeds ~128 (mult @ shift=8 would overflow INT16). The
 * loop still picks the LARGEST shift that fits — bigger shift = more
 * fractional precision — so layers with small scale_factors continue to
 * pick high shifts as before. Hit on node_relu_14 (scale=283.33) which
 * required shift<=6 to fit mult<32768. */
function computeScaleApprox(scaleFactor: number): { mult: number; shift: number } {
  if (scaleFactor <= 0) {
    throw new Error(`Scale factor must be positive; got ${scaleFactor}.`);
  }
  let best = { mult: 1, shift: 0, err: Infinity };
  for (let shift = 0; shift <= 23; shift++) {
    const mult = Math.round(scaleFactor * Math.pow(2, shift));
    if (mult >= 1 && mult < 32768) {
      const err = Math.abs(mult / Math.pow(2, shift) - scaleFactor) / scaleFactor;
      if (err < best.err) {
        best = { mult, shift, err };
      }
    }
  }
  if (!Number.isFinite(best.err)) {
    throw new Error(
      `Scale factor ${scaleFactor} is outside the representable SCALE_MULT/SCALE_SHIFT range.`,
    );
  }
  return { mult: best.mult, shift: best.shift };
}

/**
 * Quantized add uses fused constants:
 *
 *   out = round((lhs * lhs_scale + rhs * rhs_scale) / output_scale)
 *
 * The ratios can be greater than 1, so unlike the conv scale helper this does
 * not cap the multiplier at INT16 range. The serialized add RTL only spends
 * two DSPs, so wider constants are acceptable.
 */
function computeAddFusedScaleApprox(
  inputScaleFactor: number,
  outputScaleFactor: number,
): { mult: number; shift: number } {
  if (inputScaleFactor <= 0 || outputScaleFactor <= 0) {
    throw new Error(
      `Add scale factors must be positive; got input=${inputScaleFactor}, output=${outputScaleFactor}.`,
    );
  }
  const ratio = inputScaleFactor / outputScaleFactor;
  let best = { mult: 1, shift: 0, err: Infinity };
  // Same lower-bound extension rationale as computeScaleApprox: deep-network
  // layers can have ratios > 128, which need shift<8 to fit the 23-bit cap.
  for (let shift = 0; shift <= 23; shift++) {
    const mult = Math.round(ratio * Math.pow(2, shift));
    if (mult >= 1 && mult < Math.pow(2, 23)) {
      const err = Math.abs(mult / Math.pow(2, shift) - ratio) / ratio;
      if (err < best.err) {
        best = { mult, shift, err };
      }
    }
  }
  if (!Number.isFinite(best.err)) {
    throw new Error(
      `Add scale ratio ${ratio} is outside the representable fused multiplier range.`,
    );
  }
  return { mult: best.mult, shift: best.shift };
}

function addScaleConstWidth(...mults: number[]): number {
  const maxMult = Math.max(...mults, 1);
  return Math.max(2, Math.ceil(Math.log2(maxMult + 1)) + 1);
}

function computeSerializedAddLatencyCycles(layer: LayerIR): number {
  const outputChannels = getShapeChannels(layer.output_shape, "output_shape", layer.module_id);
  // One cycle to capture the packed pixel, then one channel per cycle through
  // the three registered arithmetic stages. The testbench observes valid_out
  // one cycle after the final register update, so the contract is OC + 3.
  return outputChannels + 3;
}

export function buildSerializedAddModule(layer: LayerIR): VerilogModule {
  if (layer.op_type !== "add") {
    throw new Error(`buildSerializedAddModule called for non-add layer '${layer.module_id}'.`);
  }
  if (layer.lhs_scale_factor === undefined || layer.rhs_scale_factor === undefined) {
    throw new Error(`Add layer '${layer.module_id}' is missing lhs_scale_factor/rhs_scale_factor.`);
  }
  const oc = getShapeChannels(layer.output_shape, "output_shape", layer.module_id);
  const w = layer.output_width_bits;
  if (layer.input_width_bits !== 2 * w) {
    throw new Error(
      `Add layer '${layer.module_id}' has input_width_bits=${layer.input_width_bits}; expected ${2 * w}.`,
    );
  }

  const lhs = computeAddFusedScaleApprox(layer.lhs_scale_factor, layer.scale_factor);
  const rhs = computeAddFusedScaleApprox(layer.rhs_scale_factor, layer.scale_factor);
  const fusedShift = Math.max(lhs.shift, rhs.shift);
  const lhsMult = lhs.mult * Math.pow(2, fusedShift - lhs.shift);
  const rhsMult = rhs.mult * Math.pow(2, fusedShift - rhs.shift);
  const constWidth = addScaleConstWidth(lhsMult, rhsMult);
  const chIdxW = Math.max(1, Math.ceil(Math.log2(oc + 1)));
  const prodW = 8 + constWidth;
  const sumW = prodW + 2;
  const expectedLatency = computeSerializedAddLatencyCycles(layer);

  if (layer.pipeline_latency_cycles !== expectedLatency) {
    throw new Error(
      `Add layer '${layer.module_id}' latency=${layer.pipeline_latency_cycles}, but serialized add expects ${expectedLatency}. ` +
        `Regenerate LayerIR/goldens with the current add latency contract.`,
    );
  }

  const source = `module ${layer.module_id} (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                valid_in,
    output reg                 ready_in,
    input  wire [${layer.input_width_bits - 1}:0]       data_in,
    output reg                 valid_out,
    output reg  [${layer.output_width_bits - 1}:0]       data_out
);

    localparam integer OC            = ${oc};
    localparam integer W             = ${w};
    localparam integer CH_IDX_W      = (OC <= 1) ? 1 : $clog2(OC + 1);
    localparam integer SCALE_CONST_W = ${constWidth};
    localparam integer FUSED_SHIFT   = ${fusedShift};
    localparam integer PROD_W        = ${prodW};
    localparam integer SUM_W         = ${sumW};
    localparam [CH_IDX_W-1:0] OC_IDX      = ${chIdxW}'d${oc};
    localparam [CH_IDX_W-1:0] LAST_CH_IDX = ${chIdxW}'d${oc - 1};

    localparam signed [SCALE_CONST_W-1:0] LHS_FUSED_MULT = ${constWidth}'sd${lhsMult};
    localparam signed [SCALE_CONST_W-1:0] RHS_FUSED_MULT = ${constWidth}'sd${rhsMult};
    localparam signed [SUM_W-1:0]         FUSED_ROUND_BIAS =
        {{(SUM_W-1){1'b0}}, 1'b1} <<< (FUSED_SHIFT - 1);
    localparam signed [SUM_W-1:0]         SAT_HI =  127;
    localparam signed [SUM_W-1:0]         SAT_LO = -128;

    localparam [1:0] ST_IDLE = 2'd0;
    localparam [1:0] ST_RUN  = 2'd1;

    reg [1:0] state;
    reg [${layer.input_width_bits - 1}:0] input_buf;
    reg [CH_IDX_W-1:0] ch_idx;
    reg [CH_IDX_W-1:0] stage1_idx, stage2_idx;
    reg stage1_valid, stage2_valid;

    // 3-stage pipeline (OC + 3 latency): stage 1 multiplies one channel
    // per cycle into (lhs_term, rhs_term); stage 2 sums + adds the round
    // bias one cycle later, reading stage 1's products DIRECTLY (NBA
    // semantics make the BEFORE-edge value of lhs_term = the channel
    // stage 1 just issued); stage 3 (saturate + write) fires the cycle
    // after stage 2 commits, indexed by stage2_idx.
    //
    // The intermediate "wait state" (stage2_idx <= stage1_idx; stage3_idx
    // <= stage2_idx; sum_term <= lhs_term) that an earlier version had
    // is REMOVED here -- it caused a one-channel index drift between
    // sum_term and stage3_idx because stage 1's lhs_term was overwritten
    // before the stage that read it actually committed. Verilator
    // simulation catches that drift; lint does not. Do not re-introduce
    // it without re-verifying with a real Verilator run.
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] lhs_term;
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] rhs_term;
    reg signed [SUM_W-1:0] sum_term;

    wire [CH_IDX_W-1:0] safe_ch_idx = (ch_idx < OC_IDX) ? ch_idx : {CH_IDX_W{1'b0}};
    wire signed [7:0] lhs_ch = $signed(input_buf[safe_ch_idx*8 +: 8]);
    wire signed [7:0] rhs_ch = $signed(input_buf[W + safe_ch_idx*8 +: 8]);
    wire signed [SUM_W-1:0] lhs_term_ext = {{(SUM_W-PROD_W){lhs_term[PROD_W-1]}}, lhs_term};
    wire signed [SUM_W-1:0] rhs_term_ext = {{(SUM_W-PROD_W){rhs_term[PROD_W-1]}}, rhs_term};
    wire signed [SUM_W-1:0] shifted_w = (sum_term >>> FUSED_SHIFT); // [INVARIANT:ROUNDING]

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ready_in     <= 1'b0;
            valid_out    <= 1'b0;
            data_out     <= {W{1'b0}};
            input_buf    <= {${layer.input_width_bits}{1'b0}};
            state        <= ST_IDLE;
            ch_idx       <= {CH_IDX_W{1'b0}};
            stage1_idx   <= {CH_IDX_W{1'b0}};
            stage2_idx   <= {CH_IDX_W{1'b0}};
            stage1_valid <= 1'b0;
            stage2_valid <= 1'b0;
            lhs_term     <= {PROD_W{1'b0}};
            rhs_term     <= {PROD_W{1'b0}};
            sum_term     <= {SUM_W{1'b0}};
        end else begin
            valid_out <= 1'b0;

            case (state)
                ST_IDLE: begin
                    ready_in     <= 1'b1; // [INVARIANT:READY_IN_GATING]
                    stage1_valid <= 1'b0;
                    stage2_valid <= 1'b0;
                    ch_idx       <= {CH_IDX_W{1'b0}};

                    if (valid_in) begin
                        input_buf <= data_in;
                        ready_in  <= 1'b0;
                        state     <= ST_RUN;
                    end
                end

                ST_RUN: begin
                    ready_in <= 1'b0;

                    // Stage 3: saturate + write data_out for the channel
                    // whose sum stage 2 committed last cycle.
                    if (stage2_valid) begin
                        if (shifted_w > SAT_HI)
                            data_out[stage2_idx*8 +: 8] <= 8'h7F;
                        else if (shifted_w < SAT_LO)
                            data_out[stage2_idx*8 +: 8] <= 8'h80;
                        else
                            data_out[stage2_idx*8 +: 8] <= shifted_w[7:0];

                        if (stage2_idx == LAST_CH_IDX) begin
                            valid_out    <= 1'b1; // [INVARIANT:VALID_OUT_LATENCY]
                            ready_in     <= 1'b1;
                            state        <= ST_IDLE;
                            stage1_valid <= 1'b0;
                            stage2_valid <= 1'b0;
                        end
                    end

                    // Stage 2: sum stage 1's products + round bias.
                    // Reads lhs_term / rhs_term BEFORE the edge -- those
                    // are the ch=stage1_idx products that stage 1 just
                    // committed at the previous edge. Forwards stage1_idx
                    // to stage2_idx so the sat stage indexes correctly.
                    if (stage1_valid) begin
                        sum_term     <= lhs_term_ext + rhs_term_ext + FUSED_ROUND_BIAS;
                        stage2_idx   <= stage1_idx;
                        stage2_valid <= 1'b1;
                    end else begin
                        stage2_valid <= 1'b0;
                    end

                    // Stage 1: issue one channel per cycle until ch_idx
                    // reaches OC. Multiplies are registered and DSP-tagged.
                    if (ch_idx < OC_IDX) begin
                        lhs_term     <= lhs_ch * LHS_FUSED_MULT;
                        rhs_term     <= rhs_ch * RHS_FUSED_MULT;
                        stage1_idx   <= ch_idx;
                        stage1_valid <= 1'b1;
                        ch_idx       <= ch_idx + 1'b1;
                    end else begin
                        stage1_valid <= 1'b0;
                    end
                end

                default: begin
                    state        <= ST_IDLE;
                    ready_in     <= 1'b1;
                    stage1_valid <= 1'b0;
                    stage2_valid <= 1'b0;
                end
            endcase
        end
    end

endmodule
`;

  return {
    module_id: layer.module_id,
    spec_hash: computeExpectedSpecHash(layer),
    verilog_source: source,
    generated_by: "Foundry",
    attempt: 1,
  };
}

/**
 * Clone a passing VerilogModule for a new module_id, substituting:
 *   1. The `module <name>` declaration
 *   2. The $readmemh weight/bias paths
 *   3. SCALE_MULT and SCALE_SHIFT localparam values
 *
 * Falls back to null if any substitution produces a collision (meaning the
 * source RTL structure doesn't match the expected naming convention — let
 * Foundry regenerate in that case).
 */
function instantiateTemplateModule(
  source: VerilogModule,
  sourceLayer: LayerIR,
  targetLayer: LayerIR,
): VerilogModule | null {
  let src = source.verilog_source;

  // 1. Rename `module <old_id>` → `module <new_id>`
  const moduleRe = new RegExp(`\\bmodule\\s+${escapeRegex(source.module_id)}\\b`, "g");
  const renamed = src.replace(moduleRe, `module ${targetLayer.module_id}`);
  if (renamed === src) return null; // pattern not found — bail
  src = renamed;

  // 2. Substitute $readmemh paths (exact string match; paths are unique)
  if (sourceLayer.weights_path && targetLayer.weights_path) {
    src = src.split(sourceLayer.weights_path).join(targetLayer.weights_path);
  }
  if (sourceLayer.bias_path && targetLayer.bias_path) {
    src = src.split(sourceLayer.bias_path).join(targetLayer.bias_path);
  }

  // 3. Substitute SCALE_MULT and SCALE_SHIFT localparam values.
  //    Pattern covers both old-style (32'd784) and new-style (just integers)
  //    so templates generated by any Foundry prompt version are handled.
  const newScale = computeScaleApprox(targetLayer.scale_factor);
  src = src.replace(
    /\bSCALE_MULT\b(\s*=\s*)\d*'?d?(\d+)/g,
    (_m, eq) => `SCALE_MULT${eq}32'd${newScale.mult}`,
  );
  src = src.replace(
    /\bSCALE_SHIFT\b(\s*=\s*)\d*'?d?(\d+)/g,
    (_m, eq) => `SCALE_SHIFT${eq}5'd${newScale.shift}`,
  );
  // 3a. If the source uses a DSP-banked HI*256 + LO split for SCALE_MULT,
  //     recompute HI/LO from the new mult. Without this, a clone keeps the
  //     source layer's HI/LO encoding the old SCALE_MULT — making the
  //     effective multiplier wrong by a factor of (old_mult / new_mult).
  //     Surfaced as systematic INT8 drift on node_conv_256 (got 0 instead
  //     of -1, max_error=5, signed-error skew toward zero), which only
  //     showed up because the cloned RTL embedded SCALE_HI=113 / SCALE_LO=83
  //     from a 29011 split alongside the substituted SCALE_MULT=6427.
  if (/\bSCALE_HI\b\s*=/.test(src) && /\bSCALE_LO\b\s*=/.test(src)) {
    const hi = Math.floor(newScale.mult / 256);
    const lo = newScale.mult % 256;
    src = src.replace(
      /\bSCALE_HI\b(\s*=\s*)\d*'?d?(\d+)/g,
      (_m, eq) => `SCALE_HI${eq}${hi}`,
    );
    src = src.replace(
      /\bSCALE_LO\b(\s*=\s*)\d*'?d?(\d+)/g,
      (_m, eq) => `SCALE_LO${eq}${lo}`,
    );
  }

  if (
    targetLayer.op_type === "add" &&
    sourceLayer.op_type === "add" &&
    targetLayer.lhs_scale_factor !== undefined &&
    targetLayer.rhs_scale_factor !== undefined
  ) {
    const lhs = computeAddFusedScaleApprox(targetLayer.lhs_scale_factor, targetLayer.scale_factor);
    const rhs = computeAddFusedScaleApprox(targetLayer.rhs_scale_factor, targetLayer.scale_factor);
    const fusedShift = Math.max(lhs.shift, rhs.shift);
    const lhsMult = lhs.mult * Math.pow(2, fusedShift - lhs.shift);
    const rhsMult = rhs.mult * Math.pow(2, fusedShift - rhs.shift);
    const constWidth = addScaleConstWidth(lhsMult, rhsMult);

    src = src.replace(
      /\bSCALE_CONST_W\b(\s*=\s*)\d+/g,
      (_m, eq) => `SCALE_CONST_W${eq}${constWidth}`,
    );
    src = src.replace(
      /\bFUSED_SHIFT\b(\s*=\s*)\d+/g,
      (_m, eq) => `FUSED_SHIFT${eq}${fusedShift}`,
    );
    src = src.replace(
      /\bLHS_FUSED_MULT\b(\s*=\s*)\d+'sd\d+/g,
      (_m, eq) => `LHS_FUSED_MULT${eq}${constWidth}'sd${lhsMult}`,
    );
    src = src.replace(
      /\bRHS_FUSED_MULT\b(\s*=\s*)\d+'sd\d+/g,
      (_m, eq) => `RHS_FUSED_MULT${eq}${constWidth}'sd${rhsMult}`,
    );
  }

  const targetSpecHash = computeExpectedSpecHash(targetLayer);
  return {
    module_id: targetLayer.module_id,
    spec_hash: targetSpecHash,
    verilog_source: src,
    generated_by: "Foundry",
    attempt: 1,
  };
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export async function writePipelineSummary(
  manager: PipelineStateManager,
  pipelineIr: PipelineIR,
  runtime: OrchestratorRuntime = createOrchestratorRuntime(),
): Promise<void> {
  const summaryPath = reportPath("pipeline_summary.json");
  const summaryPayload = {
    run_id: manager.getState().run_id,
    completed_at: runtime.now().toISOString(),
    is_done: manager.isDone(),
    model_name: pipelineIr.model_name,
    modules_total: pipelineIr.layers.length,
    total_cost_usd: manager.getState().total_cost_usd,
    model_usage: manager.getState().model_usage,
    summary_table: manager.summary(),
    state: manager.getState(),
  };

  await writeJsonFile(summaryPath, summaryPayload);
  await appendRunLog(
    {
      event: "pipeline_summary_written",
      path: summaryPath,
      payload: summaryPayload,
    },
    runtime,
  );
}

export async function runPipeline(
  checkpointPath: string,
  options: RunPipelineOptions = {},
): Promise<void> {
  if (options.networkId) {
    setActiveNetwork(options.networkId);
  }
  const resume = options.resume ?? false;
  // Module-level agent-history Maps are not disk-persisted; clear them at
  // every entry so back-to-back runPipeline calls in the same process (tests,
  // dashboard background jobs, REPLs) start from a clean slate. Resuming a
  // pipeline still works because the disk state — pipeline_state.json,
  // canonical RTL, reports — is the source of truth; the in-memory maps are
  // only consulted within a single process lifetime.
  clearAgentHistories();
  const runtime = createOrchestratorRuntime(options.runtime);
  await ensureOutputLayout();

  const runLogPath = reportPath("run_log.jsonl");
  if (!resume) {
    await writeFile(runLogPath, "", "utf8");
  }

  await appendRunLog(
    {
      event: "pipeline_start",
      network_id: getActiveNetworkId(),
      output_root: getPipelineOutputRoot(),
      checkpoint_path: checkpointPath,
      resume,
    },
    runtime,
  );

  const layerIrBootstrap = await ensureLayerIr(checkpointPath, runtime);
  const pipelineIr = layerIrBootstrap.pipelineIr;
  const allModuleIds = pipelineIr.layers.map((layer) => layer.module_id);
  if (options.only && !allModuleIds.includes(options.only)) {
    throw new Error(
      `--only '${options.only}' is not a module in the current LayerIR. ` +
        `Valid module_ids: [${allModuleIds.join(", ")}].`,
    );
  }
  const exceptSet = new Set(options.except ?? []);
  for (const id of exceptSet) {
    if (!allModuleIds.includes(id)) {
      throw new Error(
        `--except '${id}' is not a module in the current LayerIR. ` +
          `Valid module_ids: [${allModuleIds.join(", ")}].`,
      );
    }
  }
  const moduleIds = options.only
    ? [options.only]
    : allModuleIds.filter((id) => !exceptSet.has(id));
  if (moduleIds.length === 0) {
    throw new Error("--except excluded every module; nothing to run.");
  }
  const baseLayersByModule = new Map(
    pipelineIr.layers.map((layer) => [layer.module_id, jsonClone(layer)] as const),
  );
  const statePath = resolvePipelineConfigPath(PIPELINE_CONFIG.pipeline_state_path);
  const maxRetries = options.maxRetries ?? PIPELINE_CONFIG.max_retries;
  const selfImproveEnabled = options.selfImprove ?? PIPELINE_CONFIG.self_improve;
  const manager = new PipelineStateManager(moduleIds, maxRetries);

  if (resume && (await pathExists(statePath))) {
    await manager.loadState(statePath);
    manager.requireModuleIdsMatch(moduleIds);
    await appendRunLog(
      {
        event: "pipeline_resume_loaded",
        state_path: statePath,
        state: manager.getState(),
      },
      runtime,
    );
    // FOUNDRY_HISTORY is in-memory only and was just cleared by
    // clearAgentHistories(). On a process-restart resume, reseed it from
    // the persistent run_log so any module sitting in fail_retry can hit
    // its prior Foundry session id and produce an
    // `invoke_foundry_continuation` instead of paying for a fresh session.
    await reseedFoundryHistoryFromRunLog(pipelineIr, manager, runtime);
  } else {
    if (layerIrBootstrap.bootstrapUsage) {
      manager.recordAgentUsage(
        layerIrBootstrap.bootstrapUsage.total_cost_usd,
        layerIrBootstrap.bootstrapUsage.modelUsage,
      );
    }
    await manager.saveState(statePath);
    await appendRunLog(
      {
        event: "pipeline_state_initialized",
        state_path: statePath,
        state: manager.getState(),
      },
      runtime,
    );
  }

  const contractState = selfImproveEnabled
    ? await loadContractResponseState()
    : emptyContractResponseState();
  const activeLayers = new Map<string, LayerIR>();
  const newDocFailureContexts = new Map<string, NewDocFailureContext>();
  for (const moduleId of moduleIds) {
    const baseLayer = baseLayersByModule.get(moduleId);
    if (!baseLayer) {
      throw new Error(`LayerIR for module '${moduleId}' was not found while selecting contracts.`);
    }
    if (!selfImproveEnabled) {
      setActiveLayerForModule(
        pipelineIr,
        activeLayers,
        moduleId,
        withSignatureMetadata(baseLayer, baseLayer, pipelineIr.quantization),
      );
      continue;
    }

    const selected = selectAvailableContract(baseLayer, contractState);
    if (!selected) {
      const manualResult: VerifResult = {
        module_id: moduleId,
        status: "fail",
        timing_pass: false,
        timing_actual_cycles: 0,
        timing_expected_cycles: baseLayer.pipeline_latency_cycles,
        failure_class: "manual_correction_needed",
        failure_category: "unknown",
        violated_constraint: "all_available_contracts_flagged",
        classifier_reason:
          "Every contract variant for this structural spec is flagged manual_correction_needed. Delete or edit output/contract_state.json after manual correction to re-enable it.",
      };
      const before = manager.getState().modules[moduleId];
      manager.applyVerifResult(moduleId, manualResult);
      await logStateTransition(
        manager,
        moduleId,
        before,
        manager.getState().modules[moduleId],
        "contract_skipped_manual_correction_needed",
        runtime,
      );
      await appendRunLog(
        {
          event: "contract_skipped_manual_correction_needed",
          module_id: moduleId,
          available_contracts: CONTRACT_PLANS,
          contract_state_path: contractStatePath(),
          result: manualResult,
        },
        runtime,
      );
      setActiveLayerForModule(
        pipelineIr,
        activeLayers,
        moduleId,
        withSignatureMetadata(baseLayer, baseLayer, pipelineIr.quantization),
      );
      continue;
    }

    const selectedLayer = withSignatureMetadata(baseLayer, selected.layer, pipelineIr.quantization);
    setActiveLayerForModule(pipelineIr, activeLayers, moduleId, selectedLayer);
    if (selected.plan.id !== "flat-bus") {
      const flaggedContracts = Object.values(contractState.contracts).filter(
        (flag) => flag.op_type === selectedLayer.op_type && flag.status === "manual_correction_needed",
      );
      newDocFailureContexts.set(moduleId, {
        reason: "selected_after_skipping_flagged_contracts",
        flagged_contracts: flaggedContracts,
      });
      await appendRunLog(
        {
          event: "contract_selected_after_skipping_flagged",
          module_id: moduleId,
          selected_contract_id: selected.plan.id,
          selected_contract_key: contractStateKeyForLayer(selectedLayer),
          selected_layer_ir: selectedLayer,
          contract_state_path: contractStatePath(),
        },
        runtime,
      );
    }
  }
  await manager.saveState(statePath);

  // Spec-hash cache: module_id → {module, layer} for every module that passed
  // verification. Keyed by module_id so we can look up the source LayerIR when
  // instantiating a clone. The spec_hash on the module itself is the lookup key
  // for clone eligibility.
  const passedModules = new Map<string, { module: VerilogModule; layer: LayerIR }>();

  // On resume: pre-populate the cache with any modules already in pass state
  // so cloning works even when earlier modules passed in a previous run.
  for (const layer of pipelineIr.layers) {
    if (manager.getState().modules[layer.module_id] === "pass") {
      try {
        const existing = await loadPersistedVerilogModule(layer.module_id);
        passedModules.set(layer.module_id, { module: existing, layer });
      } catch {
        // RTL file missing from a prior interrupted run — safe to skip;
        // the module stays "pass" and won't be re-generated anyway.
      }
    }
  }

  while (!manager.isDone()) {
    const beforeTickState = manager.getState();
    const nextAction = manager.tick();

    if (nextAction.action === "done") {
      break;
    }

    const afterTickState = manager.getState();
    const tickModuleId = "module_id" in nextAction ? nextAction.module_id : undefined;

    if (tickModuleId) {
      const beforeStatus = beforeTickState.modules[tickModuleId];
      const afterStatus = afterTickState.modules[tickModuleId];
      if (beforeStatus !== afterStatus) {
        await logStateTransition(
          manager,
          tickModuleId,
          beforeStatus,
          afterStatus,
          nextAction.action,
          runtime,
        );
      }
    }

    await manager.saveState(statePath);

    if (nextAction.action === "invoke_foundry") {
      const layer = activeLayers.get(nextAction.module_id) ?? findLayer(pipelineIr, nextAction.module_id);
      const baseLayer = baseLayersByModule.get(nextAction.module_id) ?? layer;

      // --- Bus-width capability gate -----------------------------------------
      // Fail-fast on layers whose bus widths exceed the pipeline's current
      // capability. Cheaper than burning Foundry+Surgeon calls on a layer we
      // know we cannot correctly generate. Routes directly to fail_abort via
      // failure_class=architectural_unsupported (pipeline.ts skips Surgeon
      // for this class).
      const unsupportedReason = checkBusWidthCapability(layer);
      if (unsupportedReason) {
        const archFail: VerifResult = {
          module_id: nextAction.module_id,
          status: "fail",
          timing_pass: false,
          timing_actual_cycles: 0,
          timing_expected_cycles: layer.pipeline_latency_cycles,
          failure_class: "architectural_unsupported",
          fix_hint: unsupportedReason,
        };
        const classifiedArchFail = await classifyFailedModule(
          manager,
          archFail,
          layer,
          null,
          runtime,
          { capability_gate: unsupportedReason },
        );
        await recordFailureAttempt(
          layer,
          "capability_gate",
          classifiedArchFail,
          null,
          runtime,
          { capability_gate: unsupportedReason },
        );
        const statusBeforeApply = manager.getState().modules[nextAction.module_id];
        manager.applyVerifResult(nextAction.module_id, classifiedArchFail);
        const statusAfterApply = manager.getState().modules[nextAction.module_id];
        await logStateTransition(
          manager,
          nextAction.module_id,
          statusBeforeApply,
          statusAfterApply,
          "architectural_unsupported",
          runtime,
        );
        await manager.saveState(statePath);
        const handledByRetrospector = await maybeRunRetrospectorFinalAttempt(
          manager,
          nextAction.module_id,
          layer,
          baseLayer,
          pipelineIr,
          activeLayers,
          newDocFailureContexts,
          contractState,
          classifiedArchFail,
          statePath,
          runtime,
          passedModules,
          selfImproveEnabled,
        );
        if (!handledByRetrospector) {
          await appendRunLog(
            {
              event: "module_fail_abort",
              module_id: nextAction.module_id,
              result: classifiedArchFail,
            },
            runtime,
          );
        }
        continue;
      }

      // The deterministic serialized-add template (`buildSerializedAddModule`)
      // is OFF by default. Pipeline runs (`--only layer*_add` or full
      // pipeline) go through the normal Foundry + Surgeon path so add
      // layers exercise the LLM contract documented in
      // `knowledge/patterns/protected/05_add_quantized.md` end-to-end.
      //
      // Set `NN2RTL_DETERMINISTIC_ADD=1` to short-circuit Foundry and
      // emit the deterministic template instead. This is intended ONLY
      // for testing the template (e.g. via `scripts/regen_add_smoke.ts`)
      // and for golden-result comparisons; it must NOT be set during
      // milestone or thesis-evaluation runs.
      if (
        layer.op_type === "add" &&
        process.env.NN2RTL_DETERMINISTIC_ADD === "1"
      ) {
        const addModule = buildSerializedAddModule(layer);
        await appendRunLog(
          {
            event: "action",
            action: "invoke_deterministic_add_template",
            module_id: nextAction.module_id,
            spec_hash: addModule.spec_hash,
          },
          runtime,
        );

        await persistVerilogModule(addModule);

        const statusBeforeVerify = manager.getState().modules[nextAction.module_id];
        manager.setStatus(nextAction.module_id, "verifying");
        await logStateTransition(
          manager,
          nextAction.module_id,
          statusBeforeVerify,
          "verifying",
          "deterministic_add_template_completed",
          runtime,
        );
        await manager.saveState(statePath);

        const rawAddVerif = await invokeAssayer(addModule, layer, runtime);
        const addVerif = await classifyFailedModule(
          manager,
          rawAddVerif,
          layer,
          addModule,
          runtime,
        );
        if (addVerif.status !== "pass") {
          await recordFailureAttempt(layer, "deterministic_add_assayer", addVerif, addModule, runtime);
        }
        const statusBeforeApply = manager.getState().modules[nextAction.module_id];
        manager.applyVerifResult(nextAction.module_id, addVerif);
        const statusAfterApply = manager.getState().modules[nextAction.module_id];
        await logStateTransition(
          manager,
          nextAction.module_id,
          statusBeforeApply,
          statusAfterApply,
          `assayer_${addVerif.status}`,
          runtime,
        );
        await manager.saveState(statePath);

        if (statusAfterApply === "pass") {
          await processSynthesisOutcome(
            manager,
            nextAction.module_id,
            addModule,
            layer,
            addVerif,
            statePath,
            runtime,
            selfImproveEnabled,
          );
          if (manager.getState().modules[nextAction.module_id] === "pass") {
            passedModules.set(nextAction.module_id, { module: addModule, layer });
          }
        }

        if (statusAfterApply === "fail_abort") {
          await appendRunLog({ event: "module_fail_abort", module_id: nextAction.module_id, result: addVerif }, runtime);
        }

        continue;
      }

      // --- Spec-hash template reuse -------------------------------------------
      // Check whether a structurally identical module has already passed.
      // If so, clone it (substituting module name, weight paths, scale constants)
      // and run the assayer directly — no Foundry LLM call needed.
      const expectedSpecHash = computeExpectedSpecHash(layer);
      const templateEntry = [...passedModules.values()].find(
        (entry) => entry.module.spec_hash === expectedSpecHash,
      );

      if (templateEntry) {
        const cloned = instantiateTemplateModule(
          templateEntry.module,
          templateEntry.layer,
          layer,
        );

        if (cloned !== null) {
          await appendRunLog(
            {
              event: "action",
              action: "invoke_foundry_template_clone",
              module_id: nextAction.module_id,
              source_module_id: templateEntry.module.module_id,
              spec_hash: expectedSpecHash,
            },
            runtime,
          );

          await persistVerilogModule(cloned);

          const statusBeforeVerify = manager.getState().modules[nextAction.module_id];
          manager.setStatus(nextAction.module_id, "verifying");
          await logStateTransition(manager, nextAction.module_id, statusBeforeVerify, "verifying", "template_clone_completed", runtime);
          await manager.saveState(statePath);

          const rawCloneVerif = await invokeAssayer(cloned, layer, runtime);
          const cloneVerif = await classifyFailedModule(
            manager,
            rawCloneVerif,
            layer,
            cloned,
            runtime,
          );
          if (cloneVerif.status !== "pass") {
            await recordFailureAttempt(layer, "template_clone_assayer", cloneVerif, cloned, runtime);
          }
          const statusBeforeApply = manager.getState().modules[nextAction.module_id];
          manager.applyVerifResult(nextAction.module_id, cloneVerif);
          const statusAfterApply = manager.getState().modules[nextAction.module_id];
          await logStateTransition(manager, nextAction.module_id, statusBeforeApply, statusAfterApply, `assayer_${cloneVerif.status}`, runtime);
          await manager.saveState(statePath);

          if (statusAfterApply === "pass") {
            await processSynthesisOutcome(manager, nextAction.module_id, cloned, layer, cloneVerif, statePath, runtime, selfImproveEnabled);
            if (manager.getState().modules[nextAction.module_id] === "pass") {
              passedModules.set(nextAction.module_id, { module: cloned, layer });
            }
          }

          if (statusAfterApply === "fail_abort") {
            await appendRunLog({ event: "module_fail_abort", module_id: nextAction.module_id, result: cloneVerif }, runtime);
          }

          // Even if the clone failed verification, fall through to Surgeon via
          // the normal fail_retry path — do NOT fall through to Foundry here.
          continue;
        }
      }
      // ------------------------------------------------------------------------

      // tick() increments attempts BEFORE dispatching. Under the default
      // budget there is only one normal Foundry call; this continuation
      // plumbing remains for explicit larger budgets and the self-improve
      // final Foundry retry.
      const foundryAttemptIndex = manager.getState().attempts[nextAction.module_id] ?? 1;
      const foundryPriorResult =
        foundryAttemptIndex > 1
          ? manager.getState().results[nextAction.module_id]
          : undefined;
      // Attempt 1 means a fresh Foundry call for this (module, contract) —
      // either a non-resume cold start or a post-resetModuleForContractRetry
      // contract switch. Clear stale .v / .meta.json so a no-write or
      // malformed-write Foundry response cannot be silently rescued by
      // tryRecoverVerilogModuleFromDisk() reading a prior contract's RTL.
      // Attempt 2+ deliberately preserves Surgeon's last attempt on disk —
      // that file is the input the resumed Foundry session is reviewing.
      if (foundryAttemptIndex === 1) {
        await clearGeneratedRtlArtifacts(
          nextAction.module_id,
          "fresh_foundry_attempt",
          runtime,
        );
      }
      let foundryResult: RtlAgentRunResult;
      try {
        foundryResult = await invokeFoundry(layer, runtime, {
          selfImproveEnabled,
          newDocFailureContext: newDocFailureContexts.get(nextAction.module_id),
          priorVerifResult: foundryPriorResult,
        });
      } catch (err) {
        // SpecHashMismatchError used to crash the whole run; treat it as a
        // retryable code-bug VerifResult so the per-(module, contract) retry
        // budget handles it the same way as any other Foundry mistake.
        if (err instanceof SpecHashMismatchError) {
          const specFail = specHashMismatchAsVerifResult(err, layer);
          await recordFailureAttempt(layer, "foundry_spec_hash_mismatch", specFail, null, runtime);
          const statusBeforeApply = manager.getState().modules[nextAction.module_id];
          manager.applyVerifResult(nextAction.module_id, specFail);
          const statusAfterApply = manager.getState().modules[nextAction.module_id];
          await logStateTransition(
            manager,
            nextAction.module_id,
            statusBeforeApply,
            statusAfterApply,
            "foundry_spec_hash_mismatch",
            runtime,
          );
          await manager.saveState(statePath);
          continue;
        }
        // Reached-maximum-turns errors from the Anthropic SDK used to crash
        // the run via handlePipelineError. Convert to a retryable code-bug
        // VerifResult so the per-(module, contract) attempt budget gets a
        // chance to converge on the resumed Foundry session.
        if (isAgentMaxTurnsError(err)) {
          const turnsFail = agentMaxTurnsAsVerifResult(err, layer, "Foundry");
          await recordFailureAttempt(layer, "foundry_max_turns", turnsFail, null, runtime);
          const statusBeforeApply = manager.getState().modules[nextAction.module_id];
          manager.applyVerifResult(nextAction.module_id, turnsFail);
          const statusAfterApply = manager.getState().modules[nextAction.module_id];
          await logStateTransition(
            manager,
            nextAction.module_id,
            statusBeforeApply,
            statusAfterApply,
            "foundry_max_turns",
            runtime,
          );
          await manager.saveState(statePath);
          continue;
        }
        throw err;
      }
      recordUsageFromResult(manager, foundryResult.result);
      await manager.saveState(statePath);

      const statusBeforeVerify = manager.getState().modules[nextAction.module_id];
      manager.setStatus(nextAction.module_id, "verifying");
      await logStateTransition(
        manager,
        nextAction.module_id,
        statusBeforeVerify,
        "verifying",
        "foundry_completed",
        runtime,
      );
      await manager.saveState(statePath);

      const rawAssayerVerif = await invokeAssayer(foundryResult.payload, layer, runtime);
      const assayerVerif = await classifyFailedModule(
        manager,
        rawAssayerVerif,
        layer,
        foundryResult.payload,
        runtime,
      );
      if (assayerVerif.status !== "pass") {
        await recordFailureAttempt(layer, "foundry_assayer", assayerVerif, foundryResult.payload, runtime);
        if (selfImproveEnabled) {
          await archiveProbationaryDocsForFailure(
            layer,
            nextAction.module_id,
            `foundry_assayer_${assayerVerif.status}`,
            runtime,
          );
        }
      }
      const statusBeforeApply = manager.getState().modules[nextAction.module_id];
      manager.applyVerifResult(nextAction.module_id, assayerVerif);
      const statusAfterApply = manager.getState().modules[nextAction.module_id];
      await logStateTransition(
        manager,
        nextAction.module_id,
        statusBeforeApply,
        statusAfterApply,
        `assayer_${assayerVerif.status}`,
        runtime,
      );
      await manager.saveState(statePath);

      if (statusAfterApply === "pass") {
        await processSynthesisOutcome(
          manager,
          nextAction.module_id,
          foundryResult.payload,
          layer,
          assayerVerif,
          statePath,
          runtime,
          selfImproveEnabled,
        );
        if (manager.getState().modules[nextAction.module_id] === "pass") {
          passedModules.set(nextAction.module_id, { module: foundryResult.payload, layer });
          await finalizeSuccessfulRtlDocs(
            foundryResult.payload,
            layer,
            foundryResult.draft_doc,
            runtime,
            selfImproveEnabled,
            [],
            foundryResult.doc_request ?? null,
          );
        }
        const synthesisResult = manager.getState().results[nextAction.module_id];
        if (manager.getState().modules[nextAction.module_id] === "fail_abort" && synthesisResult) {
          await maybeRunRetrospectorFinalAttempt(
            manager,
            nextAction.module_id,
            layer,
            baseLayer,
            pipelineIr,
            activeLayers,
            newDocFailureContexts,
            contractState,
            synthesisResult,
            statePath,
            runtime,
            passedModules,
            selfImproveEnabled,
          );
        }
      }

      if (statusAfterApply === "fail_abort") {
        const handledByRetrospector = await maybeRunRetrospectorFinalAttempt(
          manager,
          nextAction.module_id,
          layer,
          baseLayer,
          pipelineIr,
          activeLayers,
          newDocFailureContexts,
          contractState,
          assayerVerif,
          statePath,
          runtime,
          passedModules,
          selfImproveEnabled,
        );
        if (!handledByRetrospector) {
          await appendRunLog(
            {
              event: "module_fail_abort",
              module_id: nextAction.module_id,
              result: assayerVerif,
            },
            runtime,
          );
        }
      }

      continue;
    }

    if (nextAction.action === "invoke_surgeon") {
      const layer = activeLayers.get(nextAction.module_id) ?? findLayer(pipelineIr, nextAction.module_id);
      const baseLayer = baseLayersByModule.get(nextAction.module_id) ?? layer;

      // If Foundry exhausted maxTurns (or otherwise crashed) without ever
      // calling write_verilog, there's no RTL on disk for Surgeon to repair.
      // Hit on node_add_7 (2026-05-08): Foundry max-turns'd in the
      // create_new_doc_request flow for a fresh contract, never persisted
      // RTL, then the next-action loop scheduled Surgeon. Without this
      // guard the pipeline crashes with ENOENT mid-run. Instead: surface
      // the existing foundry_max_turns failure as a clean fail_abort and
      // let the orchestrator route through Retrospector / contract walker
      // / human escalation per the standard error-flow taxonomy.
      let brokenModule: VerilogModule;
      try {
        brokenModule = await loadPersistedVerilogModule(nextAction.module_id);
      } catch (loadErr) {
        const isMissing =
          typeof loadErr === "object" &&
          loadErr !== null &&
          "code" in loadErr &&
          (loadErr as { code?: string }).code === "ENOENT";
        if (!isMissing) throw loadErr;
        const priorResult = manager.getState().results[nextAction.module_id];
        const escalated: VerifResult = {
          ...(priorResult ?? {
            module_id: nextAction.module_id,
            status: "fail",
            timing_pass: false,
            timing_actual_cycles: 0,
            timing_expected_cycles: layer.pipeline_latency_cycles,
            failure_class: "agent_max_turns_exhausted",
          }),
          status: "fail",
          failure_category:
            (priorResult?.failure_category as VerifResult["failure_category"]) ?? "code_bug",
          violated_constraint:
            priorResult?.violated_constraint ?? "foundry_produced_no_rtl",
          classifier_reason:
            priorResult?.classifier_reason ??
            "Foundry exhausted maxTurns without persisting any Verilog via write_verilog; Surgeon cannot repair a non-existent module. Escalating to fail_abort.",
        };
        const before = manager.getState().modules[nextAction.module_id];
        manager.applyVerifResult(nextAction.module_id, escalated);
        await logStateTransition(
          manager,
          nextAction.module_id,
          before,
          manager.getState().modules[nextAction.module_id],
          "surgeon_skipped_no_rtl_on_disk",
          runtime,
        );
        await appendRunLog(
          {
            event: "surgeon_skipped_no_rtl_on_disk",
            module_id: nextAction.module_id,
            reason: "foundry_produced_no_rtl",
            result: escalated,
          },
          runtime,
        );
        await manager.saveState(statePath);
        continue;
      }
      const verifResult = manager.getState().results[nextAction.module_id];

      if (!verifResult) {
        throw new Error(
          `Cannot invoke Surgeon for module '${nextAction.module_id}' without a previous VerifResult.`,
        );
      }

      let surgeonResult: RtlAgentRunResult;
      try {
        surgeonResult = await invokeSurgeon(brokenModule, verifResult, layer, runtime, {
          selfImproveEnabled,
        });
      } catch (err) {
        // Same SpecHashMismatch handling as the Foundry call site: convert to
        // a retryable VerifResult so a stale-contract response from Surgeon
        // doesn't crash the run. The fail_retry budget catches it.
        if (err instanceof SpecHashMismatchError) {
          const specFail = specHashMismatchAsVerifResult(err, layer);
          await recordFailureAttempt(layer, "surgeon_spec_hash_mismatch", specFail, brokenModule, runtime);
          const statusBeforeApply = manager.getState().modules[nextAction.module_id];
          manager.applyVerifResult(nextAction.module_id, specFail);
          const statusAfterApply = manager.getState().modules[nextAction.module_id];
          await logStateTransition(
            manager,
            nextAction.module_id,
            statusBeforeApply,
            statusAfterApply,
            "surgeon_spec_hash_mismatch",
            runtime,
          );
          await manager.saveState(statePath);
          continue;
        }
        // Same max-turns rescue as Foundry: a Surgeon turn-cap blow-up should
        // be a recoverable code-bug, not a process-killing error.
        if (isAgentMaxTurnsError(err)) {
          const turnsFail = agentMaxTurnsAsVerifResult(err, layer, "Surgeon");
          await recordFailureAttempt(layer, "surgeon_max_turns", turnsFail, brokenModule, runtime);
          const statusBeforeApply = manager.getState().modules[nextAction.module_id];
          manager.applyVerifResult(nextAction.module_id, turnsFail);
          const statusAfterApply = manager.getState().modules[nextAction.module_id];
          await logStateTransition(
            manager,
            nextAction.module_id,
            statusBeforeApply,
            statusAfterApply,
            "surgeon_max_turns",
            runtime,
          );
          await manager.saveState(statePath);
          continue;
        }
        throw err;
      }
      recordUsageFromResult(manager, surgeonResult.result);

      // Surgeon regression guard. Without this guard, a Surgeon turn that
      // rewrites the module and breaks the preflight contract (missing
      // canonical ports, wrong port widths, malformed module header) gets
      // persisted to disk, and the NEXT fail_retry loads THAT broken module
      // as the new "broken_module" input to Surgeon. The damage compounds
      // across retries instead of being bounded to one attempt.
      //
      // When the incoming brokenModule already passed preflight (so the bug
      // was in the datapath, not the interface) and Surgeon's output fails
      // preflight, we treat that as a pure regression: revert to the prior
      // module on disk and hand it back as the verification input too. The
      // attempt still counts against max_retries — the budget's job is to
      // cap churn, not to measure whether each attempt was productive.
      // Convergence is addressed by giving Surgeon better context (history
      // of prior attempts), not by inflating the retry budget.
      // Preserve Surgeon's original attempted source *before* any revert
      // so we can record a faithful diff in attempt history. Also detect
      // the "recovery" case where the LLM dispatch crashed and the payload
      // came from disk-read — in that case the attempted source equals
      // the broken module verbatim.
      const attemptedSource = surgeonResult.payload.verilog_source;
      const attemptedEqualsBroken = attemptedSource === brokenModule.verilog_source;

      let attemptOutcome: SurgeonAttemptRecord["outcome"] =
        attemptedEqualsBroken ? "reverted_recovered" : "accepted_still_failing";

      const surgeonPreflightIssues = preflightVerilogModule(surgeonResult.payload, layer);
      const brokenPreflightIssues = preflightVerilogModule(brokenModule, layer);
      if (
        surgeonPreflightIssues.length > 0 &&
        brokenPreflightIssues.length === 0
      ) {
        await appendRunLog(
          {
            event: "surgeon_regression_reverted",
            module_id: nextAction.module_id,
            surgeon_preflight_issues: surgeonPreflightIssues,
            reason:
              "Surgeon output regressed on the preflight contract while the prior module satisfied it. Reverted to the prior module.",
          },
          runtime,
        );
        await persistVerilogModule(brokenModule);
        surgeonResult.payload = brokenModule;
        attemptOutcome = "reverted_preflight";
      }

      await manager.saveState(statePath);

      const statusBeforeVerify = manager.getState().modules[nextAction.module_id];
      manager.setStatus(nextAction.module_id, "verifying");
      await logStateTransition(
        manager,
        nextAction.module_id,
        statusBeforeVerify,
        "verifying",
        "surgeon_completed",
        runtime,
      );
      await manager.saveState(statePath);

      let assayerVerif = await invokeAssayer(surgeonResult.payload, layer, runtime);

      // Functional regression guard. If Surgeon's output verifies *worse*
      // than what it started with — e.g. broke previously-exact timing,
      // or significantly raised mean/max error — revert to the prior
      // module on disk and re-attribute the failing state to the prior
      // VerifResult. Without this, a Surgeon pass that "partially
      // addresses" the reported bug while destroying working logic ends
      // up as the next iteration's starting point and compounds damage.
      if (isSurgeonRegression(verifResult, assayerVerif)) {
        await appendRunLog(
          {
            event: "surgeon_regression_reverted",
            module_id: nextAction.module_id,
            reason: "Surgeon output is functionally worse than the prior module. Reverted to prior module and prior VerifResult.",
            prior_summary: summarizeVerifForLog(verifResult),
            surgeon_summary: summarizeVerifForLog(assayerVerif),
          },
          runtime,
        );
        await persistVerilogModule(brokenModule);
        surgeonResult.payload = brokenModule;
        assayerVerif = verifResult; // carry forward the prior result
        attemptOutcome = "reverted_functional";
      }

      assayerVerif = await classifyFailedModule(
        manager,
        assayerVerif,
        layer,
        surgeonResult.payload,
        runtime,
      );
      if (assayerVerif.status !== "pass") {
        await recordFailureAttempt(layer, "surgeon_assayer", assayerVerif, surgeonResult.payload, runtime);
        if (selfImproveEnabled) {
          await archiveProbationaryDocsForFailure(
            layer,
            nextAction.module_id,
            `surgeon_assayer_${assayerVerif.status}`,
            runtime,
          );
        }
      }

      // Record this Surgeon attempt in the per-module history ring buffer
      // BEFORE applyVerifResult so the next invocation of Surgeon (if
      // another fail_retry is queued) sees the full trajectory.
      recordSurgeonAttempt(nextAction.module_id, {
        attempt_index: manager.getState().attempts[nextAction.module_id] ?? 0,
        outcome: attemptOutcome,
        verif_summary: summarizeVerifForLog(assayerVerif),
        rtl_diff_unified:
          attemptOutcome === "reverted_recovered"
            ? "(LLM dispatch crashed; no new RTL was produced on this attempt.)"
            : unifiedishDiff(brokenModule.verilog_source, attemptedSource),
      });

      const statusBeforeApply = manager.getState().modules[nextAction.module_id];
      manager.applyVerifResult(nextAction.module_id, assayerVerif);
      const statusAfterApply = manager.getState().modules[nextAction.module_id];
      await logStateTransition(
        manager,
        nextAction.module_id,
        statusBeforeApply,
        statusAfterApply,
        `assayer_${assayerVerif.status}`,
        runtime,
      );
      await manager.saveState(statePath);

      if (statusAfterApply === "pass") {
        await processSynthesisOutcome(
          manager,
          nextAction.module_id,
          surgeonResult.payload,
          layer,
          assayerVerif,
          statePath,
          runtime,
          selfImproveEnabled,
        );
        if (manager.getState().modules[nextAction.module_id] === "pass") {
          passedModules.set(nextAction.module_id, { module: surgeonResult.payload, layer });
          await finalizeSuccessfulRtlDocs(
            surgeonResult.payload,
            layer,
            surgeonResult.draft_doc,
            runtime,
            selfImproveEnabled,
          );
        }
        const synthesisResult = manager.getState().results[nextAction.module_id];
        if (manager.getState().modules[nextAction.module_id] === "fail_abort" && synthesisResult) {
          await maybeRunRetrospectorFinalAttempt(
            manager,
            nextAction.module_id,
            layer,
            baseLayer,
            pipelineIr,
            activeLayers,
            newDocFailureContexts,
            contractState,
            synthesisResult,
            statePath,
            runtime,
            passedModules,
            selfImproveEnabled,
          );
        }
      }

      if (statusAfterApply === "fail_abort") {
        const handledByRetrospector = await maybeRunRetrospectorFinalAttempt(
          manager,
          nextAction.module_id,
          layer,
          baseLayer,
          pipelineIr,
          activeLayers,
          newDocFailureContexts,
          contractState,
          assayerVerif,
          statePath,
          runtime,
          passedModules,
          selfImproveEnabled,
        );
        if (!handledByRetrospector) {
          await appendRunLog(
            {
              event: "module_fail_abort",
              module_id: nextAction.module_id,
              result: assayerVerif,
            },
            runtime,
          );
        }
      }

      continue;
    }

    // tick() only emits invoke_foundry / invoke_surgeon; Assayer is always run
    // inline after a generation step above. Reaching here means PipelineStateManager
    // added a new action type that runPipeline was not updated to handle.
    throw new Error(`Unhandled pipeline action '${JSON.stringify(nextAction)}'.`);
  }

  await writePipelineSummary(manager, pipelineIr, runtime);
  await appendRunLog(
    {
      event: "pipeline_complete",
      run_id: manager.getState().run_id,
      summary: manager.summary(),
    },
    runtime,
  );
}

export function parseCliArgs(argv: string[]): {
  checkpointPath: string;
  networkId: string;
  resume: boolean;
  maxRetries: number | undefined;
  only: string | undefined;
  except: string[];
} {
  let networkId = defaultNetworkId();
  let resume = false;
  let maxRetries: number | undefined;
  let only: string | undefined;
  const except: string[] = [];
  const positional: string[] = [];

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--resume") {
      resume = true;
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
    } else if (arg === "--max-retries") {
      const next = argv[++i];
      if (next === undefined) {
        throw new Error("--max-retries requires a non-negative integer value.");
      }
      const parsed = Number(next);
      if (!Number.isInteger(parsed) || parsed < 0) {
        throw new Error(`--max-retries must be a non-negative integer, got '${next}'.`);
      }
      maxRetries = parsed;
    } else if (arg.startsWith("--max-retries=")) {
      const raw = arg.slice("--max-retries=".length);
      const parsed = Number(raw);
      if (!Number.isInteger(parsed) || parsed < 0) {
        throw new Error(`--max-retries must be a non-negative integer, got '${raw}'.`);
      }
      maxRetries = parsed;
    } else if (arg === "--only") {
      const next = argv[++i];
      if (next === undefined || next.startsWith("--")) {
        throw new Error("--only requires a module_id argument.");
      }
      only = next;
    } else if (arg.startsWith("--only=")) {
      only = arg.slice("--only=".length);
      if (!only) {
        throw new Error("--only= requires a non-empty module_id.");
      }
    } else if (arg === "--except") {
      const next = argv[++i];
      if (next === undefined || next.startsWith("--")) {
        throw new Error("--except requires a comma-separated module_id argument.");
      }
      for (const id of next.split(",").map((s) => s.trim()).filter(Boolean)) {
        except.push(id);
      }
    } else if (arg.startsWith("--except=")) {
      const raw = arg.slice("--except=".length);
      for (const id of raw.split(",").map((s) => s.trim()).filter(Boolean)) {
        except.push(id);
      }
    } else if (arg.startsWith("--")) {
      throw new Error(`Unknown flag '${arg}'.`);
    } else {
      positional.push(arg);
    }
  }

  if (positional.length < 1) {
    throw new Error("Usage: tsx main.ts <checkpoint-path> [--resume] [--max-retries N] [--only MODULE_ID | --except MODULE_ID[,MODULE_ID...]]");
  }

  if (only && except.length > 0) {
    throw new Error("--only and --except are mutually exclusive.");
  }

  return {
    checkpointPath: positional[0],
    networkId,
    resume,
    maxRetries,
    only,
    except,
  };
}

export async function runCli(argv: string[] = process.argv.slice(2)): Promise<void> {
  const cli = parseCliArgs(argv);
  setActiveNetwork(cli.networkId);
  // Validate the checkpoint path at the CLI boundary so a typo fails fast and
  // with a useful message instead of being routed through the Python frontend
  // (which produces a noisier error after doing real work).
  if (!(await pathExists(cli.checkpointPath))) {
    throw new Error(
      `Checkpoint not found: '${cli.checkpointPath}'. Pass a valid path relative to the repo root or an absolute path.`,
    );
  }
  await runPipeline(cli.checkpointPath, {
    networkId: cli.networkId,
    resume: cli.resume,
    maxRetries: cli.maxRetries,
    only: cli.only,
    except: cli.except,
  });
}

export async function handlePipelineError(
  error: unknown,
  runtime: Partial<OrchestratorRuntime> = {},
): Promise<void> {
  const resolvedRuntime = createOrchestratorRuntime(runtime);
  const message = error instanceof Error ? error.message : String(error);
  console.error(message);
  // Recovery-side failures (disk full, permission denied on output/, bad
  // runtime) used to be swallowed, which hid the real root cause when the
  // failure mode was "cannot write to output/ at all". Log them to stderr so
  // postmortems see every failure in the chain.
  try {
    await ensureOutputLayout();
  } catch (layoutErr: unknown) {
    const m = layoutErr instanceof Error ? layoutErr.message : String(layoutErr);
    console.error(`handlePipelineError: ensureOutputLayout failed: ${m}`);
  }
  try {
    await appendRunLog(
      {
        event: "pipeline_error",
        error: message,
      },
      resolvedRuntime,
    );
  } catch (logErr: unknown) {
    const m = logErr instanceof Error ? logErr.message : String(logErr);
    console.error(`handlePipelineError: appendRunLog failed: ${m}`);
  }
  process.exitCode = 1;
}
