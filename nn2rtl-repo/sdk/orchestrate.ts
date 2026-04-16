import { appendFile, access, mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { parse as parseYaml } from "yaml";
import { z } from "zod";

import {
  query,
  type AgentDefinition,
  type OutputFormat,
  type SDKMessage,
  type SDKResultMessage,
} from "./claude-agent-sdk-compat.js";
import { AGENT_CONFIG, PIPELINE_CONFIG, type AgentName } from "./config.js";
import { PipelineStateManager } from "./pipeline.js";
import {
  layerIrSchema as layerIrZod,
  pipelineIrSchema as pipelineIrZod,
  synthesisReportSchema as synthesisReportZod,
  verifResultSchema as verifResultZod,
  verilogModuleSchema as verilogModuleZod,
} from "./schemas.js";
import type {
  LayerIR,
  ModelUsageEntry,
  PipelineIR,
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
  foundry: ["mcp__nn2rtl-tools__write_verilog"],
  surgeon: ["mcp__nn2rtl-tools__write_verilog"],
} as const;

const GLOBAL_ALLOWED_TOOLS = [
  "Agent",
  ...new Set(Object.values(AGENT_MCP_TOOLS).flat()),
];

type SynthesisReport = z.infer<typeof synthesisReportZod>;
type AgentSlug = (typeof AGENT_SLUGS)[AgentName];
type AgentRunResult<T> = {
  payload: T;
  result: SDKResultMessage;
  messages: SDKMessage[];
};
type FrontmatterRecord = Record<string, unknown>;

type PortDirection = "input" | "output" | "inout";
type ParsedTopPort = {
  declaration: string;
  direction: PortDirection;
  width_bits: number | null;
};

export type YosysFn = (module: VerilogModule, layer: LayerIR) => Promise<SynthesisReport>;
export type AssayerFn = (
  module: VerilogModule,
  layer: LayerIR,
) => Promise<VerifResult>;

export type OrchestratorRuntime = {
  now: () => Date;
  queryFn: typeof query;
  yosysFn: YosysFn;
  assayerFn: AssayerFn;
};

export type RunPipelineOptions = {
  resume?: boolean;
  runtime?: Partial<OrchestratorRuntime>;
  maxRetries?: number;
};

const DEFAULT_ORCHESTRATOR_RUNTIME: OrchestratorRuntime = {
  now: () => new Date(),
  queryFn: query,
  yosysFn: (module, layer) => invokeYosys(module, layer),
  assayerFn: (module, layer) => runAssayerDeterministic(module, layer),
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

// PPA gates — a module that passes functional verification but fails these
// is treated as a real hardware failure and routed back to Surgeon via a
// synthesized VerifResult. Thresholds come from the README's FPGA targets.
const FMAX_TARGET_MHZ = 50;
const MAX_LUT_COUNT_PER_MODULE = 5000;

// Maps a Yosys outcome to either a pass (null) or a synthesized VerifResult
// with the correct failure_class. The classification matters because Surgeon
// uses it to pick the repair strategy — "add a pipeline register" is very
// different from "remove a non-synthesizable construct."
// Yosys reports on large residual blocks can be tens of MB (mostly
// repetitive warnings like "No latch inferred"). Tail-only truncation hid
// the real fatal error in the head — Surgeon was repairing blindly on a
// diet of warning spam. Summarize as head + ERROR/error lines + tail so
// every fatal diagnostic survives even when the middle is huge noise.
const YOSYS_REPORT_HEAD_BYTES = 2_500;
const YOSYS_REPORT_TAIL_BYTES = 3_500;
const YOSYS_REPORT_ERRORS_BYTES = 4_000;
function capYosysReport(report: string): string {
  if (report.length <= YOSYS_REPORT_HEAD_BYTES + YOSYS_REPORT_TAIL_BYTES) {
    return report;
  }
  const head = report.slice(0, YOSYS_REPORT_HEAD_BYTES);
  const tail = report.slice(-YOSYS_REPORT_TAIL_BYTES);
  const errorLines = report
    .split(/\r?\n/)
    .filter((line) => /ERROR|error:|Error:/.test(line))
    .join("\n");
  const errorBlock =
    errorLines.length > YOSYS_REPORT_ERRORS_BYTES
      ? errorLines.slice(0, YOSYS_REPORT_ERRORS_BYTES) +
        `\n...[${errorLines.length - YOSYS_REPORT_ERRORS_BYTES} more error-line bytes elided]...`
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
    // Yosys crashed, emitted a syntax/elaboration error, or hit a construct
    // iverilog's linter accepted but Yosys refuses. Fix strategy: rewrite
    // the offending construct so the Sky130 synth flow accepts it.
    return {
      ...verifiedResult,
      module_id: moduleId,
      status: "fail",
      failure_class: "synthesis_failed",
      fix_hint: [
        "Yosys synthesis failed after functional verification passed.",
        "Repair the RTL so `synth; dfflibmap -liberty sky130.lib; abc -liberty sky130.lib; stat -liberty sky130.lib` succeeds.",
        "Look at the HEAD and ERRORS sections below for the root cause; the TAIL is usually noise (e.g. repeated 'No latch inferred' warnings).",
        "Yosys output summary (head + errors + tail):",
        capYosysReport(report.report),
      ].join("\n\n"),
    };
  }

  if (report.fmax_mhz <= 0) {
    return {
      ...verifiedResult,
      module_id: moduleId,
      status: "fail",
      failure_class: "synthesis_failed",
      fix_hint: [
        "Yosys synthesis succeeded but did not emit a measurable Sky130 timing result.",
        "Repair the RTL or synthesis flow so constrained `abc -constr ... -D ...` reports a critical-path delay.",
        "Yosys output summary (head + errors + tail):",
        capYosysReport(report.report),
      ].join("\n\n"),
    };
  }

  if (report.fmax_mhz < FMAX_TARGET_MHZ) {
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
        "Insert a pipeline register to break the critical path, and update pipeline_latency_cycles to match.",
        "Yosys output summary (head + errors + tail):",
        capYosysReport(report.report),
      ].join("\n\n"),
    };
  }

  // `lut_count` is a real LUT count only on the old FPGA/iCE40 flow. Under
  // the current Sky130 `stat -liberty` flow it is a total standard-cell count
  // proxy, so the old 5k LUT ceiling is not comparable. Keep reporting it for
  // observability, but only enforce the LUT gate when no standard-cell area
  // metric is present.
  if (report.area_um2 === 0 && report.lut_count > MAX_LUT_COUNT_PER_MODULE) {
    // Design synthesizes but burns absurd area. Fix strategy: simplify /
    // factor shared terms; this is not the same bug as a timing failure.
    return {
      ...verifiedResult,
      module_id: moduleId,
      status: "fail",
      failure_class: "synthesis_failed",
      fix_hint: [
        `Synthesis passed but LUT count ${report.lut_count} exceeds the ${MAX_LUT_COUNT_PER_MODULE} per-module ceiling.`,
        "Rewrite the module to share arithmetic or collapse redundant logic.",
        "Yosys output summary (head + errors + tail):",
        capYosysReport(report.report),
      ].join("\n\n"),
    };
  }

  return null;
}

function reportPath(fileName: string): string {
  return path.join(resolveFromSdk(PIPELINE_CONFIG.reports_dir), fileName);
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
  const prompt = parsedSkill
    ? `${body}\n\nSupplemental skill reference:\n\n${parsedSkill.body}`
    : body;

  return {
    description: AGENT_CONFIG[agentName].description,
    prompt,
    tools: combinedTools.length > 0 ? combinedTools : undefined,
    disallowedTools,
    model: AGENT_CONFIG[agentName].model,
    maxTurns: AGENT_CONFIG[agentName].maxTurns,
    skills,
  };
}

export async function loadAllAgentDefinitions(): Promise<Record<string, AgentDefinition>> {
  const entries = await Promise.all(
    Object.values(AGENT_SLUGS).map(
      async (slug) => [slug, await loadPluginAgentDefinition(slug)] as const,
    ),
  );

  return Object.fromEntries(entries);
}

export function buildDelegationPrompt(slug: AgentSlug, payload: unknown): string {
  const lines = [
    `Invoke the \`${slug}\` subagent immediately.`,
    "Do not solve the task yourself.",
    "Do not use any other subagent.",
    "Return only the subagent's final JSON object.",
    "The only data channel into the subagent is this prompt string, so the payload is embedded below as JSON.",
  ];

  if (slug === "foundry" || slug === "surgeon") {
    lines.push(
      "",
      "HARD CONTRACT for this subagent — do not accept any other output:",
      "1. The subagent MUST call the mcp__nn2rtl-tools__write_verilog tool exactly once to persist the RTL before returning.",
      "2. The subagent's final message MUST be a single JSON object with exactly these five fields and NOTHING else:",
      '   { "module_id": string, "spec_hash": string, "verilog_source": string, "generated_by": "Foundry"|"Surgeon", "attempt": integer >= 1 }',
      "3. `verilog_source` MUST be the full Verilog source code as a single string (the same string passed to write_verilog).",
      "4. Do NOT invent other keys (no `source_path`, no `port_list`, no `module_name`). Do NOT wrap the JSON in markdown fences.",
      "5. If the subagent cannot comply, it must still return the five-field JSON with a best-effort `verilog_source`.",
    );
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
): Promise<AgentRunResult<T>> {
  const agentName = normalizeAgentName(slug);
  const agents = await loadAllAgentDefinitions();
  const messages: SDKMessage[] = [];
  let finalResult: SDKResultMessage | null = null;

  for await (const message of runtime.queryFn({
    prompt: buildDelegationPrompt(slug, payload),
    options: {
      cwd: repoRoot,
      tools: ["Agent"],
      allowedTools: ["Agent", ...AGENT_MCP_TOOLS[slug]],
      plugins: [{ type: "local", path: pluginPath }],
      agents,
      outputFormat,
      maxTurns: AGENT_CONFIG[agentName].maxTurns,
    },
  })) {
    messages.push(message);

    if (isResultMessage(message)) {
      finalResult = message;
    }
  }

  if (!finalResult) {
    throw new Error(`No final result message was received for subagent '${slug}'.`);
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
  const inputChannels = getShapeChannels(layer.input_shape, "input_shape", layer.module_id);
  return layer.op_type === "add" ? inputChannels * 16 : inputChannels * 8;
}

function expectedOutputBusWidthBits(layer: LayerIR): number {
  const outputChannels = getShapeChannels(layer.output_shape, "output_shape", layer.module_id);
  return outputChannels * 8;
}

const assayerLayerBusContractZod = layerIrZod.superRefine((layer, ctx) => {
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

  const expectedInput = expectedInputBusWidthBits(layer);
  if (layer.input_width_bits !== expectedInput) {
    ctx.addIssue({
      code: "custom",
      path: ["input_width_bits"],
      message:
        `input_width_bits=${layer.input_width_bits} does not match the LayerIR channel contract ` +
        `for op_type='${layer.op_type}' (expected ${expectedInput}).`,
    });
  }

  const expectedOutput = expectedOutputBusWidthBits(layer);
  if (layer.output_width_bits !== expectedOutput) {
    ctx.addIssue({
      code: "custom",
      path: ["output_width_bits"],
      message:
        `output_width_bits=${layer.output_width_bits} does not match the LayerIR channel contract ` +
        `for op_type='${layer.op_type}' (expected ${expectedOutput}).`,
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

  for (const rawEntry of splitTopLevelCommaList(portBlock)) {
    const declaration = stripVerilogComments(rawEntry).replace(/\s+/g, " ").trim();
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
  for (const [portName, expected] of Object.entries(CANONICAL_TOP_PORTS)) {
    const parsed = ports.get(portName);
    if (!parsed) {
      issues.push(`Missing canonical top-level port '${portName}'.`);
      continue;
    }

    if (parsed.direction !== expected.direction) {
      issues.push(
        `Top-level port '${portName}' must be declared as ${expected.direction}, found ${parsed.direction} in '${parsed.declaration}'.`,
      );
    }

    const expectedWidth = expectedTopPortWidthBits(
      portName as keyof typeof CANONICAL_TOP_PORTS,
      layer,
    );
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
  const checkpointAbs = path.resolve(checkpointPath);

  if (await pathExists(layerIrPath)) {
    // Only reuse layer_ir.json if it was generated from the same checkpoint
    // the user is asking about now. A mismatch means a stale artifact from a
    // previous run and silently compiling it would yield nonsense.
    let fingerprintMatches = false;
    try {
      const prior = (await readFile(layerIrFingerprintPath, "utf8")).trim();
      fingerprintMatches = prior === checkpointAbs;
    } catch {
      fingerprintMatches = false;
    }
    if (fingerprintMatches) {
      const pipelineIr = await readJsonFile<PipelineIR>(layerIrPath, pipelineIrZod);
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
      action: "invoke_cartographer",
      payload,
    },
    runtime,
  );

  const result = await runDelegatedAgent<PipelineIR>(
    "cartographer",
    payload,
    pipelineIrOutputFormat,
    pipelineIrZod,
    runtime,
  );

  await appendRunLog(
    {
      event: "agent_result",
      agent: "Cartographer",
      total_cost_usd: result.result.total_cost_usd,
      modelUsage: result.result.modelUsage,
      payload: result.payload,
    },
    runtime,
  );

  validateAddModulePacking(result.payload);
  await writeJsonFile(layerIrPath, result.payload);
  await writeFile(layerIrFingerprintPath, `${checkpointAbs}\n`, "utf8");
  return {
    pipelineIr: result.payload,
    bootstrapUsage: {
      total_cost_usd: result.result.total_cost_usd,
      modelUsage: result.result.modelUsage as Record<string, ModelUsageEntry>,
    },
  };
}

// Deterministic, LLM-free Yosys invocation. The previous design routed this
// through query() with an allowedTool of run_yosys and let Claude mediate
// the tool call; that mediator could refuse for content-filter reasons and
// produced "I cannot comply" responses on modules with absolute host paths
// in $readmemh. Yosys is pure infrastructure — no reasoning needed — so it
// goes through the MCP tool impl directly, validated against the same
// synthesisReportSchema the SDK path used.
// Resolved as a runtime string so tsc does not analyze the target module
// (it lives in sibling package `mcp/`, outside this package's rootDir).
// When `sdk/` is compiled to `sdk/dist/`, we need the sibling `mcp/dist/`
// build; when running straight from source via tsx, we target the .ts file.
const MCP_TOOLS_MODULE_PATH = path.basename(__dirname) === "dist"
  ? pathToFileURL(path.resolve(repoRoot, "mcp", "dist", "tools.js")).href
  : pathToFileURL(path.resolve(repoRoot, "mcp", "tools.ts")).href;

async function invokeYosys(module: VerilogModule, layer: LayerIR): Promise<SynthesisReport> {
  const mcpTools = (await import(MCP_TOOLS_MODULE_PATH)) as {
    run_yosys: (
      verilog_source: string,
      module_name: string,
      clock_period_ns: number,
    ) => Promise<SynthesisReport>;
  };
  const raw = await mcpTools.run_yosys(
    module.verilog_source,
    module.module_id,
    layer.clock_period_ns,
  );
  const parsed = synthesisReportZod.safeParse(raw);
  if (!parsed.success) {
    throw new Error(
      `run_yosys returned invalid output:\n${JSON.stringify(parsed.error.issues, null, 2)}`,
    );
  }
  return parsed.data;
}

async function processYosysOutcome(
  manager: PipelineStateManager,
  moduleId: string,
  module: VerilogModule,
  layer: LayerIR,
  verifiedResult: VerifResult,
  statePath: string,
  runtime: OrchestratorRuntime,
): Promise<void> {
  let report: SynthesisReport;
  try {
    report = await runtime.yosysFn(module, layer);
  } catch (error: unknown) {
    // Tool itself crashed before producing a structured report. Treat as a
    // synthesis failure so Surgeon gets a chance to repair; the fix_hint
    // carries whatever error message the runner surfaced.
    report = {
      success: false,
      lut_count: 0,
      fmax_mhz: 0,
      area_um2: 0,
      report: error instanceof Error ? error.message : String(error),
    };
  }

  await writeJsonFile(reportPath(`${moduleId}.yosys.json`), report);

  const synthesisFailure = evaluateSynthesis(moduleId, verifiedResult, report);
  if (!synthesisFailure) {
    // Genuine pass — RTL simulates correctly, synthesizes, and hits the PPA gates.
    await appendRunLog(
      {
        event: "yosys_pass",
        module_id: moduleId,
        lut_count: report.lut_count,
        fmax_mhz: report.fmax_mhz,
      },
      runtime,
    );
    await manager.saveState(statePath);
    return;
  }

  const statusBeforeApply = manager.getState().modules[moduleId];
  manager.applyVerifResult(moduleId, synthesisFailure);
  const statusAfterApply = manager.getState().modules[moduleId];
  await logStateTransition(
    manager,
    moduleId,
    statusBeforeApply,
    statusAfterApply,
    `yosys_${synthesisFailure.failure_class ?? "fail"}`,
    runtime,
  );
  await manager.saveState(statePath);

  if (statusAfterApply === "fail_abort") {
    await appendRunLog(
      {
        event: "module_fail_abort",
        module_id: moduleId,
        result: synthesisFailure,
      },
      runtime,
    );
  }
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
): Promise<AgentRunResult<VerilogModule>> {
  await appendRunLog(
    {
      event: "action",
      action: "invoke_foundry",
      module_id: layerIr.module_id,
    },
    runtime,
  );

  // Foundry occasionally returns a path or bare text as its final message
  // instead of the VerilogModule JSON, even though it correctly called
  // write_verilog.  Recover from disk when the JSON parse / schema validation
  // fails but the .v file is present.
  let result: AgentRunResult<VerilogModule>;
  try {
    result = await runDelegatedAgent<VerilogModule>(
      "foundry",
      {
        layer_ir: layerIr,
        write_verilog_output_dir: resolveFromSdk(PIPELINE_CONFIG.rtl_dir),
      },
      verilogModuleOutputFormat,
      verilogModuleZod,
      runtime,
    );
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

  await appendRunLog(
    {
      event: "agent_result",
      agent: "Foundry",
      module_id: layerIr.module_id,
      total_cost_usd: result.result.total_cost_usd,
      modelUsage: result.result.modelUsage,
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
    pipeline_latency_cycles: layer.pipeline_latency_cycles,
    clock_period_ns: layer.clock_period_ns,
    golden_inputs_path: layer.golden_inputs_path,
    golden_outputs_path: layer.golden_outputs_path,
    results_path: resultsPath,
    testbench_template_path: resolveFromSdk(PIPELINE_CONFIG.static_testbench_path),
  };
  await mkdir(path.dirname(sidecarPath), { recursive: true });
  await mkdir(path.dirname(resultsPath), { recursive: true });
  await writeFile(sidecarPath, `${JSON.stringify(sidecar, null, 2)}\n`, "utf8");

  // Lint first — iverilog catches most obvious Verilog mistakes faster than
  // Verilator's multi-minute build, and a lint failure is always a syntax
  // error (not a numerical/timing issue).
  const iverilog = await mcpTools.run_iverilog(module.verilog_source, module.module_id);
  if (!iverilog.success) {
    return {
      module_id: module.module_id,
      status: "syntax_error",
      timing_pass: false,
      timing_actual_cycles: 0,
      timing_expected_cycles: layer.pipeline_latency_cycles,
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
    // Tool crashed before producing a structured VerifResult. Synthesize a
    // fail so Surgeon gets a chance to look at the broken RTL; the fix_hint
    // carries whatever the runner surfaced. This mirrors processYosysOutcome.
    payload = {
      module_id: module.module_id,
      status: "fail",
      timing_pass: false,
      timing_actual_cycles: 0,
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

async function invokeSurgeon(
  brokenModule: VerilogModule,
  verifResult: VerifResult,
  layerIr: LayerIR,
  runtime: OrchestratorRuntime,
): Promise<AgentRunResult<VerilogModule>> {
  await appendRunLog(
    {
      event: "action",
      action: "invoke_surgeon",
      module_id: brokenModule.module_id,
    },
    runtime,
  );

  let result: AgentRunResult<VerilogModule>;
  try {
    result = await runDelegatedAgent<VerilogModule>(
      "surgeon",
      {
        broken_module: brokenModule,
        verif_result: verifResult,
        layer_ir: layerIr,
        write_verilog_output_dir: resolveFromSdk(PIPELINE_CONFIG.rtl_dir),
      },
      verilogModuleOutputFormat,
      verilogModuleZod,
      runtime,
    );
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

/** Compute the structural spec_hash from LayerIR fields (no scale factor). */
function computeExpectedSpecHash(layer: LayerIR): string {
  const ic = layer.input_shape.length >= 2 ? layer.input_shape[1] : 0;
  const oc = layer.output_shape.length >= 2 ? layer.output_shape[1] : 0;
  if (layer.op_type === "conv2d" && layer.weight_shape.length >= 4) {
    const kh = layer.weight_shape[2];
    const kw = layer.weight_shape[3];
    return `conv2d_${ic}x${oc}x${kh}x${kw}_i${layer.input_width_bits}_o${layer.output_width_bits}`;
  }
  return `${layer.op_type}_${ic}x${oc}_i${layer.input_width_bits}_o${layer.output_width_bits}`;
}

/** Choose SCALE_MULT/SCALE_SHIFT that minimise the relative approximation error. */
function computeScaleApprox(scaleFactor: number): { mult: number; shift: number } {
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
  return { mult: best.mult, shift: best.shift };
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
  const moduleIds = pipelineIr.layers.map((layer) => layer.module_id);
  const statePath = resolveFromSdk(PIPELINE_CONFIG.pipeline_state_path);
  const maxRetries = options.maxRetries ?? PIPELINE_CONFIG.max_retries;
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
      const layer = findLayer(pipelineIr, nextAction.module_id);

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

          const cloneVerif = await invokeAssayer(cloned, layer, runtime);
          const statusBeforeApply = manager.getState().modules[nextAction.module_id];
          manager.applyVerifResult(nextAction.module_id, cloneVerif);
          const statusAfterApply = manager.getState().modules[nextAction.module_id];
          await logStateTransition(manager, nextAction.module_id, statusBeforeApply, statusAfterApply, `assayer_${cloneVerif.status}`, runtime);
          await manager.saveState(statePath);

          if (statusAfterApply === "pass") {
            passedModules.set(nextAction.module_id, { module: cloned, layer });
            await processYosysOutcome(manager, nextAction.module_id, cloned, layer, cloneVerif, statePath, runtime);
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

      const foundryResult = await invokeFoundry(layer, runtime);
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

      const assayerVerif = await invokeAssayer(foundryResult.payload, layer, runtime);
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
        passedModules.set(nextAction.module_id, { module: foundryResult.payload, layer });
        await processYosysOutcome(
          manager,
          nextAction.module_id,
          foundryResult.payload,
          layer,
          assayerVerif,
          statePath,
          runtime,
        );
      }

      if (statusAfterApply === "fail_abort") {
        await appendRunLog(
          {
            event: "module_fail_abort",
            module_id: nextAction.module_id,
            result: assayerVerif,
          },
          runtime,
        );
      }

      continue;
    }

    if (nextAction.action === "invoke_surgeon") {
      const layer = findLayer(pipelineIr, nextAction.module_id);
      const brokenModule = await loadPersistedVerilogModule(nextAction.module_id);
      const verifResult = manager.getState().results[nextAction.module_id];

      if (!verifResult) {
        throw new Error(
          `Cannot invoke Surgeon for module '${nextAction.module_id}' without a previous VerifResult.`,
        );
      }

      const surgeonResult = await invokeSurgeon(brokenModule, verifResult, layer, runtime);
      recordUsageFromResult(manager, surgeonResult.result);
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

      const assayerVerif = await invokeAssayer(surgeonResult.payload, layer, runtime);
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
        await processYosysOutcome(
          manager,
          nextAction.module_id,
          surgeonResult.payload,
          layer,
          assayerVerif,
          statePath,
          runtime,
        );
      }

      if (statusAfterApply === "fail_abort") {
        await appendRunLog(
          {
            event: "module_fail_abort",
            module_id: nextAction.module_id,
            result: assayerVerif,
          },
          runtime,
        );
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
} {
  let resume = false;
  let maxRetries: number | undefined;
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
    } else if (arg.startsWith("--")) {
      throw new Error(`Unknown flag '${arg}'.`);
    } else {
      positional.push(arg);
    }
  }

  if (positional.length < 1) {
    throw new Error("Usage: tsx main.ts <checkpoint-path> [--resume] [--max-retries N]");
  }

  return {
    checkpointPath: positional[0],
    resume,
    maxRetries,
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
