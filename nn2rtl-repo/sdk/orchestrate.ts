import { appendFile, access, mkdir, readFile, writeFile } from "node:fs/promises";
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
  const lines = [
    "Compact generation brief:",
    `- op_type=${layer.op_type}; module_id=${layer.module_id}; return spec_hash=${expectedSpecHash} exactly.`,
    `- bus contract: data_in=${layer.input_width_bits} bits, data_out=${layer.output_width_bits} bits.`,
    `- pipeline_latency_cycles=${layer.pipeline_latency_cycles} from LayerIR is authoritative; do not override it with a hand-derived formula.`,
  ];

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
    "- invariant markers: when the mechanism exists, annotate ROUNDING, DRAIN_EXIT, INTER_VECTOR_RESET, READY_IN_GATING, and VALID_OUT_LATENCY with [INVARIANT:*] comments.",
  );
  return lines.join("\n");
}

function buildSurgeonRepairBrief(payload: unknown): string | null {
  if (!isRecord(payload) || !isRecord(payload.layer_ir) || !isRecord(payload.verif_result)) {
    return null;
  }
  const layer = payload.layer_ir as unknown as LayerIR;
  const verif = payload.verif_result as unknown as VerifResult;

  // Distinguish the sim-passed / synth-only failure from a full functional
  // failure. When sim passed, the datapath is already correct by evidence —
  // Surgeon must NOT rewrite the numerical logic, only the constructs that
  // upset Yosys (wide unrolled blocks, non-synthesizable $signed patterns,
  // deep combinational cones, etc.). Framing this narrowly prevents Surgeon
  // from regressing sim in the process of "fixing" synth.
  const isSynthOnlyFailure =
    verif.status_class === "sim_passed" && verif.failure_class === "synthesis_failed";

  const lines = [
    "Compact repair brief:",
    `- op_type=${layer.op_type}; bus contract=data_in ${layer.input_width_bits} bits, data_out ${layer.output_width_bits} bits; preserve the public interface exactly.`,
    `- authoritative latency contract: pipeline_latency_cycles=${layer.pipeline_latency_cycles}.`,
    `- current failure: status=${verif.status}; status_class=${verif.status_class ?? "n/a"}; failure_class=${verif.failure_class ?? "n/a"}.`,
    "- compiler-first rule: if status=syntax_error or compiler stderr is populated, read iverilog/verilator stderr before touching datapath logic.",
    "- setup-failure rule: if evidence points only to static_verilator_tb.cpp, sidecar JSON, or toolchain glue, do not rewrite the RTL datapath in response to it.",
  ];

  if (isSynthOnlyFailure) {
    lines.push(
      "- SYNTHESIS-ONLY FAILURE: simulation passed with correct outputs and exact timing." +
      " The datapath is proven correct — DO NOT rewrite numerical logic, MAC ordering," +
      " requantisation, ready/valid handshaking, or state transitions." +
      " Your ONLY job is to make the existing logic synthesizable." +
      " Typical synth-hostile patterns to target: deep combinational cones that abc can't map," +
      " unsynthesizable constructs (non-constant array indices into large regs, dynamic $signed," +
      " latch inference from incomplete case statements), or register/wire width issues." +
      " Read the Yosys error output below carefully and make the minimum change that addresses it.",
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

export function buildDelegationPrompt(slug: AgentSlug, payload: unknown): string {
  // The outer query() IS the agent (see runDelegatedAgent: no Task/Agent
  // dispatch, agent body is attached via systemPrompt.append). This prompt is
  // therefore a direct task instruction, not a "dispatch to subagent" message.
  const lines = [
    `You are the \`${slug}\` agent. Execute the task described below.`,
    "The payload is embedded as JSON at the end of this message.",
  ];

  if (slug === "foundry" || slug === "surgeon") {
    lines.push(
      "",
      "HARD CONTRACT — do not accept any other output:",
      "1. You MUST call the mcp__nn2rtl-tools__write_verilog tool exactly once to persist the RTL before returning.",
      "2. Your final message MUST be a single JSON object with exactly these five fields and NOTHING else:",
      '   { "module_id": string, "spec_hash": string, "verilog_source": string, "generated_by": "Foundry"|"Surgeon", "attempt": integer >= 1 }',
      "3. `verilog_source` MUST be the full Verilog source code as a single string (the same string passed to write_verilog).",
      "4. Do NOT invent other keys (no `source_path`, no `port_list`, no `module_name`). Do NOT wrap the JSON in markdown fences.",
      "5. If you cannot comply, still return the five-field JSON with a best-effort `verilog_source`.",
    );
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
    prompt: buildDelegationPrompt(slug, payload),
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
        expected_spec_hash: computeExpectedSpecHash(layerIr),
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
): Promise<AgentRunResult<VerilogModule>> {
  await appendRunLog(
    {
      event: "action",
      action: "invoke_surgeon",
      module_id: brokenModule.module_id,
    },
    runtime,
  );

  const prior_attempts = priorSurgeonAttempts(brokenModule.module_id);
  const trimmedVerif = trimVerifResultForSurgeon(verifResult);

  let result: AgentRunResult<VerilogModule>;
  try {
    result = await runDelegatedAgent<VerilogModule>(
      "surgeon",
      {
        broken_module: brokenModule,
        verif_result: trimmedVerif,
        layer_ir: layerIr,
        prior_attempts,
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
  if (layer.op_type === "conv2d" && layer.weight_shape.length >= 4) {
    const kh = layer.weight_shape[2];
    const kw = layer.weight_shape[3];
    const stride = layer.stride && layer.stride.length >= 2 ? `_st${layer.stride[0]}x${layer.stride[1]}` : "";
    const padding =
      layer.padding && layer.padding.length >= 2 ? `_p${layer.padding[0]}x${layer.padding[1]}` : "";
    // mac_parallelism affects the FSM's OC-group iteration, so two layers
    // with identical geometry but different mac_parallelism have structurally
    // different RTL and MUST NOT be clone-substituted for each other.
    const mp = layer.mac_parallelism ? `_mp${layer.mac_parallelism}` : "";
    return `conv2d_${ic}x${oc}x${kh}x${kw}_${spatial}${stride}${padding}${mp}_i${layer.input_width_bits}_o${layer.output_width_bits}`;
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
    return `maxpool_${ic}x${oc}_k${ks}_s${st}_p${pd}_${spatial}_i${layer.input_width_bits}_o${layer.output_width_bits}`;
  }
  return `${layer.op_type}_${ic}x${oc}_${spatial}_i${layer.input_width_bits}_o${layer.output_width_bits}`;
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
