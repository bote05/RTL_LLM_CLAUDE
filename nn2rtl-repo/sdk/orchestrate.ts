import { existsSync } from "node:fs";
import { appendFile, access, mkdir, readFile, rename, writeFile } from "node:fs/promises";
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
  type AgentName,
} from "./config.js";
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
  ],
  surgeon: [
    "mcp__nn2rtl-tools__write_verilog",
    "mcp__nn2rtl-tools__get_rtl_patterns",
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

const pipelineIrOutputFormat = toOutputFormat(pipelineIrZod);
const verifResultOutputFormat = toOutputFormat(verifResultZod);
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

  const rtlDir = resolveFromSdk(PIPELINE_CONFIG.rtl_dir);
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
  return path.join(resolveFromSdk(PIPELINE_CONFIG.reports_dir), fileName);
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

async function materializeContractGoldens(layer: LayerIR): Promise<{
  goldenInputsPath: string;
  goldenOutputsPath: string;
}> {
  const key = sanitizePathPart(`${layer.module_id}_${contractStateKeyForLayer(layer)}`);
  const dir = path.join(resolveFromSdk(PIPELINE_CONFIG.output_dir), "goldens", "contracts", key);
  const goldenInputsPath = await materializeContractGoldenFile(
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
    mkdir(resolveFromSdk(PIPELINE_CONFIG.output_dir), { recursive: true }),
    mkdir(resolveFromSdk(PIPELINE_CONFIG.rtl_dir), { recursive: true }),
    mkdir(resolveFromSdk(PIPELINE_CONFIG.tb_dir), { recursive: true }),
    mkdir(resolveFromSdk(PIPELINE_CONFIG.weights_dir), { recursive: true }),
    mkdir(resolveFromSdk(PIPELINE_CONFIG.reports_dir), { recursive: true }),
  ]);
}

function buildSidecarPath(moduleId: string): string {
  return path.join(resolveFromSdk(PIPELINE_CONFIG.tb_dir), `${moduleId}.sidecar.json`);
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
  if (retrySeed) {
    lines.push(`- retry seed: ${retrySeed}. Use this as a fresh-attempt discriminator; do not repeat a prior unsuccessful patch shape.`);
  }
  if (Array.isArray(payload.prior_foundry_attempts) && payload.prior_foundry_attempts.length > 0) {
    lines.push(
      `- foundry tried this contract ${payload.prior_foundry_attempts.length} time(s) before you. Their RTL and the verifier's verdict on each are in payload.prior_foundry_attempts. Read them — every entry tells you a fix shape that ALREADY didn't work. Pick a different lever.`,
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
};
type DocLifecycleEntry = {
  id: string;
  op_type: LayerIR["op_type"];
  contract_id?: ContractId;
  contract_key?: string;
  spec_hash: string;
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
];

const CONTRACT_STATE_PATH = resolveFromSdk(PIPELINE_CONFIG.contract_state_path);

function emptyContractResponseState(): ContractResponseState {
  return { version: 1, contracts: {} };
}

async function loadContractResponseState(): Promise<ContractResponseState> {
  try {
    const raw = await readFile(CONTRACT_STATE_PATH, "utf8");
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
  await writeJsonFile(CONTRACT_STATE_PATH, state);
}

function currentContractId(layer: LayerIR): ContractId {
  if (layer.contract_id) return layer.contract_id;
  if (layer.io_mode === "channel_tiled") return "tiled-streaming";
  if (layer.io_mode === "dram_backed_weights") return "dram-backed-weights";
  if (layer.io_mode === "activation_double_buffered") return "activation-double-buffering";
  if (layer.io_mode === "weight_tiled") return "weight-tiling";
  return "flat-bus";
}

function contractStateKeyForLayer(layer: LayerIR): string {
  return `${currentContractId(layer)}:${computeExpectedSpecHash(layer)}`;
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
  } else if (layer.op_type === "maxpoo