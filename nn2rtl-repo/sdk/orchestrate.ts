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
const verilogModuleOutputFormat = toOutputFormat(verilogModuleZod);
const verifResultOutputFormat = toOutputFormat(verifResultZod);
const failureClassificationOutputFormat = toOutputFormat(failureClassificationZod);
const retrospectorAdviceOutputFormat = toOutputFormat(retrospectorAdviceZod);

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
    module: verilogModuleZod,
    draft_doc: docDraftZod,
  })
  .strict();
type RtlAgentWithDoc = z.infer<typeof rtlAgentWithDocZod>;
const rtlAgentWithDocOutputFormat = toOutputFormat(rtlAgentWithDocZod);

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

type ContractPlan = {
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

const CONTRACT_PLANS: ContractPlan[] = [
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

function applyContractPlan(baseLayer: LayerIR, plan: ContractPlan): LayerIR {
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

function deterministicFailureClassification(result: VerifResult): FailureClassification | null {
  if (isToolchainInfrastructureFailure(result)) {
    return {
      category: "toolchain_infra",
      violated_resource: null,
      violated_constraint: result.violated_constraint ?? result.failure_class ?? "toolchain_or_testbench_setup",
      rationale:
        "Deterministic evidence says the RTL did not receive a trustworthy compiler/simulation verdict because the local toolchain or testbench setup failed.",
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
  if (isTransientAgentOrQuotaFailure(reason, result)) return false;
  return true;
}

const FOUNDRY_HISTORY = new Map<string, FoundryVersionRecord[]>();
const FAILURE_ATTEMPT_HISTORY = new Map<string, FailureAttemptRecord[]>();

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

function recordFailureAttempt(
  layer: LayerIR,
  stage: string,
  result: VerifResult,
  module: VerilogModule | null,
  extraLogs: Record<string, unknown> = {},
): void {
  if (result.status === "pass") return;
  const key = moduleContractKey(layer);
  const history = FAILURE_ATTEMPT_HISTORY.get(key) ?? [];
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
}

function failureAttemptsFor(layer: LayerIR): FailureAttemptRecord[] {
  return FAILURE_ATTEMPT_HISTORY.get(moduleContractKey(layer)) ?? [];
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

function docAppliesToLayer(doc: Pick<DocLifecycleEntry, "op_type" | "contract_id">, layer: LayerIR): boolean {
  return doc.op_type === layer.op_type && docContractId(doc) === currentContractId(layer);
}

function protectedPatternCoveragePath(layer: LayerIR): string | null {
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
  };
  const file = byOp[layer.op_type];
  return file ? `knowledge/patterns/protected/${file}` : null;
}

function protectedPatternCandidatesForLayer(layer: LayerIR): Array<{ id: string; op_type: LayerIR["op_type"] | "shared"; relPath: string }> {
  const common = [
    { id: "protected_context", op_type: "shared" as const, relPath: "knowledge/patterns/protected/01_context.md" },
    { id: "protected_common_bugs", op_type: "shared" as const, relPath: "knowledge/patterns/protected/08_common_bugs.md" },
  ];
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
    .filter(
      (doc) =>
        doc.op_type === layer.op_type &&
        (doc.status === "active" || doc.status === "probationary"),
    )
    .sort((a, b) => {
      const aExact = docContractId(a) === currentContractId(layer) ? 0 : 1;
      const bExact = docContractId(b) === currentContractId(layer) ? 0 : 1;
      if (aExact !== bExact) return aExact - bExact;
      const aTier = a.status === "active" ? 0 : 1;
      const bTier = b.status === "active" ? 0 : 1;
      return aTier !== bTier ? aTier - bTier : a.id.localeCompare(b.id);
    })
    .slice(0, 4);
  for (const doc of generated) {
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
  // Tier 1: protected. Flat-bus is the only contract with file-based
  // protected coverage today (`02_conv1x1.md`, `03_conv3x3_pad1.md`,
  // `04_conv7x7_pad3.md`, `05_add_quantized.md`, `06_relu.md`, `07_maxpool.md`).
  // Other contracts inherit no protected coverage by design.
  if (currentContractId(layer) === "flat-bus") {
    const coveragePath = protectedPatternCoveragePath(layer);
    if (coveragePath !== null && existsSync(absFromRepo(coveragePath))) {
      return { tier: "protected", path: coveragePath };
    }
  }

  // Tier 2 + 3: active / probationary. Match by op_type + contract_id, with
  // an extra kernel-signature gate for flat-bus convs so a probationary 1x1
  // doc cannot claim to cover a 5x5 layer. Non-flat-bus contracts are
  // themselves the discriminator so a single doc per (op, contract) is the
  // granularity the lifecycle has historically tracked there.
  const layerKernelSignature =
    layer.op_type === "conv2d" && layer.weight_shape.length >= 4
      ? `${layer.weight_shape[2]}x${layer.weight_shape[3]}`
      : null;
  const requireFlatBusKernelMatch =
    currentContractId(layer) === "flat-bus" && layerKernelSignature !== null;

  const tierRank: Record<"active" | "probationary", number> = { active: 0, probationary: 1 };
  const candidates = Object.values(state.docs)
    .filter((doc) => {
      if (doc.status !== "active" && doc.status !== "probationary") return false;
      if (!docAppliesToLayer(doc, layer)) return false;
      if (!requireFlatBusKernelMatch) return true;
      // Flat-bus + conv2d: require the doc's spec_hash to mention the same
      // kernel signature. Spec hashes follow `conv2d_<ic>x<oc>x<kh>x<kw>_...`
      // so we look for both the canonical infix and the maxpool-style
      // `_k<kh>x<kw>_` form. Seeded docs with synthetic hashes (test
      // fixtures) fail this check, which is the desired behaviour — they
      // should not silently claim coverage for kernels they were never
      // tagged with.
      if (layerKernelSignature === null) return true;
      return (
        doc.spec_hash.includes(`x${layerKernelSignature}_`) ||
        doc.spec_hash.includes(`_k${layerKernelSignature}_`)
      );
    })
    .sort((a, b) => {
      const ta = tierRank[a.status as "active" | "probationary"];
      const tb = tierRank[b.status as "active" | "probationary"];
      return ta !== tb ? ta - tb : a.id.localeCompare(b.id);
    });

  if (candidates.length === 0) return null;
  const match = candidates[0];
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
    .filter((doc) => docAppliesToLayer(doc, layer) && tiers.includes(doc.status as "active" | "probationary"))
    .sort((a, b) => {
      const tierDelta =
        tierRank[a.status as "active" | "probationary"] -
        tierRank[b.status as "active" | "probationary"];
      return tierDelta !== 0 ? tierDelta : a.id.localeCompare(b.id);
    })
    .flatMap((doc): KnowledgeDocRecord[] => [
      {
        id: doc.id,
        tier: doc.status === "active" ? "active" : "probationary",
        kind: "pattern",
        op_type: doc.op_type,
        contract_id: doc.contract_id,
        path: absFromRepo(doc.pattern_path),
        relative_path: doc.pattern_path,
      },
      {
        id: doc.id,
        tier: doc.status === "active" ? "active" : "probationary",
        kind: "reference",
        op_type: doc.op_type,
        contract_id: doc.contract_id,
        path: absFromRepo(doc.reference_path),
        relative_path: doc.reference_path,
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
    if (!docAppliesToLayer(doc, layer)) continue;
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
        .filter((doc) => docAppliesToLayer(doc, layer) && doc.used_by_modules.includes(moduleId))
        .map((doc) => ({ id: doc.id, status: doc.status, contract_id: docContractId(doc) })),
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
  state: DocLifecycleState,
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
      docAppliesToLayer(doc, layer) &&
      doc.status === "probationary" &&
      doc.used_by_modules.includes(moduleId),
  );
  if (docs.length === 0) return;
  for (const doc of docs) {
    if (!doc.failed_modules.includes(moduleId)) {
      doc.failed_modules.push(moduleId);
    }
    await archiveDocEntry(state, doc, reason, runtime);
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
    if (!docAppliesToLayer(doc, layer) || doc.status !== "active") return false;
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
      state,
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
    if (!docAppliesToLayer(doc, layer)) continue;
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

  const deterministic = deterministicFailureClassification(result);
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
      deterministicFailureClassification(result) ??
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
    "Retries for this module/contract have failed. Analyze the evidence and return advisory JSON only.",
    "",
    "Your job:",
    "- Read the original LayerIR spec, the contract, the exact RTL knowledge doc used for this op, every Foundry RTL version, and the failure logs from every attempt.",
    "- Explain why the attempts are likely failing.",
    "- Suggest one concrete strategy for Foundry's final resumed attempt.",
    "- Set `doc_fault: true` only when an active/probationary generated doc used by this module is the likely root cause, rather than an implementation mistake in the RTL attempt.",
    "- If `doc_fault` is true, set `faulty_doc_paths` to the exact lifecycle doc ids or paths from `knowledge_docs_used` that should be archived. Leave it empty if the specific doc cannot be isolated.",
    "- Do not write Verilog. Do not ask for more retries. If the current contract appears exhausted or architecturally wrong, say so clearly; the orchestrator owns contract switching.",
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
    "Keep the public contract unchanged: canonical ports, bus widths, latency, quantization, and spec_hash remain authoritative.",
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
  return firstBrace > 0 ? candidate.slice(firstBrace) : candidate;
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

  return {
    payload: requireStructuredOutput<T>(finalResult, slug, resultSchema),
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
  const metaPath = path.join(resolveFromSdk(PIPELINE_CONFIG.rtl_dir), `${moduleId}.meta.json`);
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

const CANONICAL_TOP_PORTS = {
  clk: { direction: "input" as const, width_bits: 1 },
  rst_n: { direction: "input" as const, width_bits: 1 },
  valid_in: { direction: "input" as const, width_bits: 1 },
  ready_in: { direction: "output" as const, width_bits: 1 },
  valid_out: { direction: "output" as const, width_bits: 1 },
  data_in: { direction: "input" as const, width_key: "input_width_bits" as const },
  data_out: { direction: "output" as const, width_key: "output_width_bits" as const },
};

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

function expectedTopPortWidthBits(
  portName: keyof typeof CANONICAL_TOP_PORTS,
  layer: LayerIR,
): number {
  const spec = CANONICAL_TOP_PORTS[portName];
  return "width_key" in spec ? layer[spec.width_key] : spec.width_bits;
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
  const usesCoordScheduler = /\bcoord_scheduler\b/.test(source);
  const clockedAlwaysBlocks = extractClockedAlwaysBlocks(source);

  // Rule 1: spatial conv requires a `line_buf` array-of-arrays / memory decl.
  // Skipped when the top-level instantiates `line_buf_window` (the library
  // module owns the line buffer).
  if (isSpatialConv && !usesLineBufWindow) {
    const lineBufRe = /\breg\s+(?:signed\s+)?(?:\[[^\]]+\]\s+)?line_buf\s*\[/;
    if (!lineBufRe.test(source)) {
      violations.push({
        rule: "line_buffer_missing",
        detail:
          `Spatial conv2d (kernel=${layer.weight_shape[2]}x${layer.weight_shape[3]}) must ` +
          `either instantiate line_buf_window or declare a line buffer 'line_buf' ` +
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

  // Rule 4: weights and biases must use $readmemh. Skipped when the
  // top-level instantiates `conv_datapath`: that library module owns the
  // weight/bias arrays and their $readmemh loaders, driven by WEIGHTS_PATH /
  // BIAS_PATH module parameters. Accept the current flat single-array form
  // `$readmemh(..., weights)` and the future banked form
  // `$readmemh(..., weights_bank<N>)`.
  if (layer.op_type === "conv2d" && !usesConvDatapath) {
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

export function preflightVerilogModule(module: VerilogModule, layer: LayerIR): string[] {
  const issues: string[] = [];

  if (module.module_id !== layer.module_id) {
    issues.push(
      `VerilogModule.module_id='${module.module_id}' does not match LayerIR.module_id='${layer.module_id}'.`,
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
  const layerIrPath = resolveFromSdk(PIPELINE_CONFIG.layer_ir_path);
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
    case "flat-bus":
      return layer.pipeline_latency_cycles;
  }
}

async function loadRetrospectorKnowledgeDoc(layer: LayerIR): Promise<RtlKnowledgeDoc> {
  const mcpTools = (await import(MCP_TOOLS_MODULE_PATH)) as {
    get_rtl_patterns: (
      op_type: string,
      kernel_h?: number,
      kernel_w?: number,
      contract_id?: ContractId,
    ) => Promise<RtlKnowledgeDoc>;
  };
  const kernelH = layer.op_type === "conv2d" ? layer.weight_shape[2] : undefined;
  const kernelW = layer.op_type === "conv2d" ? layer.weight_shape[3] : undefined;
  return mcpTools.get_rtl_patterns(layer.op_type, kernelH, kernelW, currentContractId(layer));
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
    recordFailureAttempt(
      layer,
      "vivado_tool_error",
      classifiedSetupFailure,
      module,
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
  recordFailureAttempt(
    layer,
    "vivado_synthesis",
    classifiedSynthesisFailure,
    module,
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
  const rtlDir = resolveFromSdk(PIPELINE_CONFIG.rtl_dir);
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

async function persistVerilogModule(module: VerilogModule): Promise<void> {
  // Agents are supposed to call the write_verilog MCP tool themselves, but
  // Sonnet/Opus under outputFormat: json_schema sometimes skip tool calls
  // to save turns. Orchestrator owns disk state, so ensure the .v and
  // .meta.json files exist regardless of whether the agent persisted them.
  const rtlDir = resolveFromSdk(PIPELINE_CONFIG.rtl_dir);
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
  } = {},
): Promise<RtlAgentRunResult> {
  await appendRunLog(
    {
      event: "action",
      action: options.retrospectorAdvice
        ? "invoke_foundry_after_retrospector"
        : "invoke_foundry",
      module_id: layerIr.module_id,
      ...(options.resumeSessionId ? { resume_session_id: options.resumeSessionId } : {}),
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
  const foundryPayload = {
    layer_ir: layerIr,
    expected_spec_hash: computeExpectedSpecHash(layerIr),
    preloaded_rtl_patterns: preloadedRtlPatterns,
    contract_options: {
      selected_contract: contractPlanForLayer(layerIr),
      ordered_contracts: CONTRACT_PLANS,
      expected_latency_cycles: expectedLatencyCyclesForContract(layerIr, contractSidecarFields(layerIr)),
      covered_by_existing_doc: createNewDocRequest === null,
    },
    write_verilog_output_dir: resolveFromSdk(PIPELINE_CONFIG.rtl_dir),
    ...(selfImproveDocRequest ? { self_improve_doc_request: selfImproveDocRequest } : {}),
    ...(createNewDocRequest ? { create_new_doc_request: createNewDocRequest } : {}),
  };
  const resumedPrompt = options.retrospectorAdvice
    ? buildFoundryRetrospectorInjectionPrompt({
        ...foundryPayload,
        retrospector_advice: options.retrospectorAdvice,
        final_attempt: foundryVersionsFor(layerIr).length + 1,
        self_improve_doc_request: selfImproveDocRequest,
        create_new_doc_request: createNewDocRequest ?? undefined,
      })
    : undefined;

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
          resumeSessionId: options.resumeSessionId,
        },
      );
      result = {
        payload: withDoc.payload.module,
        draft_doc: withDoc.payload.draft_doc,
        doc_request: createNewDocRequest,
        result: withDoc.result,
        messages: withDoc.messages,
      };
    } else {
      result = await runDelegatedAgent<VerilogModule>(
        "foundry",
        foundryPayload,
        verilogModuleOutputFormat,
        verilogModuleZod,
        runtime,
        {
          prompt: resumedPrompt,
          resumeSessionId: options.resumeSessionId,
        },
      );
    }
  } catch (err) {
    const recovered = await tryRecoverVerilogModuleFromDisk(
      layerIr,
      "Foundry",
      /* attempt */ 1,
    );
    if (!recovered) {
      throw err;
    }
    await appendRunLog(
      {
        event: "agent_result_recovered",
        agent: "Foundry",
        module_id: layerIr.module_id,
        reason: err instanceof Error ? err.message : String(err),
      },
      runtime,
    );
    // Build a minimal AgentRunResult stub.  cost/messages are unknown on the
    // recovery path but the pipeline doesn't need them for state transition.
    result = {
      payload: recovered,
      doc_request: createNewDocRequest,
      result: {
        type: "result",
        subtype: "success",
        result: "",
        total_cost_usd: 0,
        modelUsage: {},
      } as unknown as SDKResultMessage,
      messages: [],
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
async function runAssayerDeterministic(
  module: VerilogModule,
  layer: LayerIR,
): Promise<VerifResult> {
  assayerLayerBusContractZod.parse(layer);

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
    resolveFromSdk(PIPELINE_CONFIG.reports_dir),
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

async function invokeSurgeon(
  brokenModule: VerilogModule,
  verifResult: VerifResult,
  layerIr: LayerIR,
  runtime: OrchestratorRuntime,
  options: {
    selfImproveEnabled?: boolean;
  } = {},
): Promise<RtlAgentRunResult> {
  await appendRunLog(
    {
      event: "action",
      action: "invoke_surgeon",
      module_id: brokenModule.module_id,
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
  const surgeonPayload = {
    broken_module: brokenModule,
    verif_result: trimmedVerif,
    layer_ir: layerIr,
    preloaded_rtl_patterns: preloadedRtlPatterns,
    prior_attempts,
    retry_seed: retrySeed,
    write_verilog_output_dir: resolveFromSdk(PIPELINE_CONFIG.rtl_dir),
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
      result = {
        payload: withDoc.payload.module,
        draft_doc: withDoc.payload.draft_doc,
        result: withDoc.result,
        messages: withDoc.messages,
      };
    } else {
      result = await runDelegatedAgent<VerilogModule>(
        "surgeon",
        surgeonPayload,
        verilogModuleOutputFormat,
        verilogModuleZod,
        runtime,
      );
    }
  } catch (err) {
    const recovered = await tryRecoverVerilogModuleFromDisk(
      layerIr,
      "Surgeon",
      /* attempt */ Math.max(brokenModule.attempt + 1, 2),
    );
    if (!recovered) {
      throw err;
    }
    await appendRunLog(
      {
        event: "agent_result_recovered",
        agent: "Surgeon",
        module_id: brokenModule.module_id,
        reason: err instanceof Error ? err.message : String(err),
      },
      runtime,
    );
    result = {
      payload: recovered,
      result: {
        type: "result",
        subtype: "success",
        result: "",
        total_cost_usd: 0,
        modelUsage: {},
      } as unknown as SDKResultMessage,
      messages: [],
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
    setActiveLayerForModule(
      input.pipelineIr,
      input.activeLayers,
      input.moduleId,
      next.layer,
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
        next_contract_key: contractStateKeyForLayer(next.layer),
        next_layer_ir: next.layer,
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
    return false;
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
    return false;
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

  if (!resumeSessionId) {
    await appendRunLog(
      {
        event: "retrospector_no_resumable_foundry_session",
        module_id: moduleId,
        contract_key: contractKey,
        reason: "No Foundry session exists for this contract, so the orchestrator will try the next available contract instead of a same-contract final retry.",
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

  let foundryResult: RtlAgentRunResult;
  try {
    foundryResult = await invokeFoundry(layer, runtime, {
      resumeSessionId,
      retrospectorAdvice: adviceRun.payload,
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
    recordFailureAttempt(layer, "retrospector_foundry_dispatch", finalFailure, null);
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
    recordFailureAttempt(
      layer,
      "retrospector_foundry_assayer",
      finalVerif,
      foundryResult.payload,
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
function computeExpectedSpecHash(layer: LayerIR): string {
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

/** Choose SCALE_MULT/SCALE_SHIFT that minimise the relative approximation error. */
function computeScaleApprox(scaleFactor: number): { mult: number; shift: number } {
  if (scaleFactor <= 0) {
    throw new Error(`Scale factor must be positive; got ${scaleFactor}.`);
  }
  let best = { mult: 1, shift: 8, err: Infinity };
  for (let shift = 8; shift <= 23; shift++) {
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
  let best = { mult: 1, shift: 8, err: Infinity };
  for (let shift = 8; shift <= 23; shift++) {
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
  const resume = options.resume ?? false;
  const runtime = createOrchestratorRuntime(options.runtime);
  await ensureOutputLayout();

  const runLogPath = reportPath("run_log.jsonl");
  if (!resume) {
    await writeFile(runLogPath, "", "utf8");
  }

  await appendRunLog(
    {
      event: "pipeline_start",
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
  const statePath = resolveFromSdk(PIPELINE_CONFIG.pipeline_state_path);
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
      setActiveLayerForModule(pipelineIr, activeLayers, moduleId, baseLayer);
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
          contract_state_path: CONTRACT_STATE_PATH,
          result: manualResult,
        },
        runtime,
      );
      setActiveLayerForModule(pipelineIr, activeLayers, moduleId, baseLayer);
      continue;
    }

    setActiveLayerForModule(pipelineIr, activeLayers, moduleId, selected.layer);
    if (selected.plan.id !== "flat-bus") {
      const flaggedContracts = Object.values(contractState.contracts).filter(
        (flag) => flag.op_type === selected.layer.op_type && flag.status === "manual_correction_needed",
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
          selected_contract_key: contractStateKeyForLayer(selected.layer),
          selected_layer_ir: selected.layer,
          contract_state_path: CONTRACT_STATE_PATH,
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
        recordFailureAttempt(
          layer,
          "capability_gate",
          classifiedArchFail,
          null,
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
          recordFailureAttempt(layer, "deterministic_add_assayer", addVerif, addModule);
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
            recordFailureAttempt(layer, "template_clone_assayer", cloneVerif, cloned);
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

      const foundryResult = await invokeFoundry(layer, runtime, {
        selfImproveEnabled,
        newDocFailureContext: newDocFailureContexts.get(nextAction.module_id),
      });
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
        recordFailureAttempt(layer, "foundry_assayer", assayerVerif, foundryResult.payload);
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
      const brokenModule = await loadPersistedVerilogModule(nextAction.module_id);
      const verifResult = manager.getState().results[nextAction.module_id];

      if (!verifResult) {
        throw new Error(
          `Cannot invoke Surgeon for module '${nextAction.module_id}' without a previous VerifResult.`,
        );
      }

      const surgeonResult = await invokeSurgeon(brokenModule, verifResult, layer, runtime, {
        selfImproveEnabled,
      });
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
        recordFailureAttempt(layer, "surgeon_assayer", assayerVerif, surgeonResult.payload);
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
  resume: boolean;
  maxRetries: number | undefined;
  only: string | undefined;
  except: string[];
} {
  let resume = false;
  let maxRetries: number | undefined;
  let only: string | undefined;
  const except: string[] = [];
  const positional: string[] = [];

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--resume") {
      resume = true;
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
    resume,
    maxRetries,
    only,
    except,
  };
}

export async function runCli(argv: string[] = process.argv.slice(2)): Promise<void> {
  const cli = parseCliArgs(argv);
  // Validate the checkpoint path at the CLI boundary so a typo fails fast and
  // with a useful message instead of being routed through the Python frontend
  // (which produces a noisier error after doing real work).
  if (!(await pathExists(cli.checkpointPath))) {
    throw new Error(
      `Checkpoint not found: '${cli.checkpointPath}'. Pass a valid path relative to the repo root or an absolute path.`,
    );
  }
  await runPipeline(cli.checkpointPath, {
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
