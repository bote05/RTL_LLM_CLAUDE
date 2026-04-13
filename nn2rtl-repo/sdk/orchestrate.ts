import { randomUUID } from "node:crypto";
import { access, appendFile, mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  query,
  type AgentDefinition,
  type OutputFormat,
  type SDKMessage,
  type SDKResultMessage,
} from "./claude-agent-sdk-compat.js";

import { AGENT_CONFIG, PIPELINE_CONFIG, type AgentName } from "./config.js";
import { PipelineStateManager } from "./pipeline.js";
import type {
  FailureClass,
  LayerIR,
  ModelUsageEntry,
  PipelineIR,
  PipelineState,
  VerifResult,
  VerilogModule,
} from "./types.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const pluginPath = path.resolve(__dirname, "../nn2rtl-plugin");

const ACTIVE_CLI_OPTIONS = {
  resume: false,
};

const AGENT_SLUGS = {
  Conductor: "conductor",
  Cartographer: "cartographer",
  Foundry: "foundry",
  Assayer: "assayer",
  Surgeon: "surgeon",
} as const satisfies Record<AgentName, string>;

const AGENT_MCP_TOOLS = {
  conductor: [] as string[],
  cartographer: ["mcp__nn2rtl-tools__read_weights"],
  foundry: ["mcp__nn2rtl-tools__write_verilog"],
  assayer: [
    "mcp__nn2rtl-tools__run_iverilog",
    "mcp__nn2rtl-tools__run_verilator",
  ],
  surgeon: ["mcp__nn2rtl-tools__write_verilog"],
} as const;

const DIRECT_MCP_TOOLS = {
  run_yosys: "mcp__nn2rtl-tools__run_yosys",
} as const;

const GLOBAL_ALLOWED_TOOLS = [
  "Agent",
  ...new Set(Object.values(AGENT_MCP_TOOLS).flat()),
];

const FAILURE_CLASSES = [
  "integer_overflow",
  "sign_extension_error",
  "bit_shift_wrong",
  "rounding_mode_wrong",
  "saturation_missing",
  "loop_bounds_incorrect",
  "array_indexing_error",
  "port_width_mismatch",
  "residual_addition_overflow",
  "missing_pipeline_register",
  "pipeline_latency_wrong",
  "reset_logic_broken",
  "enable_signal_ignored",
  "scale_factor_misapplied",
  "bias_term_missing",
  "batch_norm_not_folded",
] as const satisfies readonly FailureClass[];

type SynthesisReport = {
  success: boolean;
  lut_count: number;
  fmax_mhz: number;
  report: string;
};

const verifResultSchema = {
  type: "object",
  additionalProperties: false,
  required: ["module_id", "status"],
  properties: {
    module_id: { type: "string" },
    status: { type: "string", enum: ["pass", "fail", "syntax_error"] },
    timing_pass: { type: "boolean" },
    timing_actual_cycles: { type: "number" },
    timing_expected_cycles: { type: "number" },
    mismatch_layer: { type: "string" },
    expected: { type: "array", items: { type: "number" } },
    got: { type: "array", items: { type: "number" } },
    max_error: { type: "number" },
    mean_error: { type: "number" },
    failure_class: {
      anyOf: [
        { type: "string", enum: [...FAILURE_CLASSES] },
        { type: "null" },
      ],
    },
    fix_hint: { type: "string" },
    iverilog_stderr: { type: "string" },
    verilator_stderr: { type: "string" },
  },
} as const;

const layerIrSchema = {
  type: "object",
  additionalProperties: false,
  required: [
    "module_id",
    "op_type",
    "input_shape",
    "output_shape",
    "weights_path",
    "bias_path",
    "weight_shape",
    "num_weights",
    "scale_factor",
    "zero_point",
    "pipeline_latency_cycles",
    "clock_period_ns",
    "input_width_bits",
    "output_width_bits",
    "valid_in_signal",
    "valid_out_signal",
    "clock_signal",
    "reset_signal",
    "golden_inputs",
    "golden_outputs",
  ],
  properties: {
    module_id: { type: "string" },
    op_type: { type: "string", enum: ["conv2d", "relu", "add"] },
    input_shape: { type: "array", items: { type: "number" } },
    output_shape: { type: "array", items: { type: "number" } },
    weights_path: { type: "string" },
    bias_path: {
      anyOf: [
        { type: "string" },
        { type: "null" },
      ],
    },
    weight_shape: { type: "array", items: { type: "number" } },
    num_weights: { type: "number" },
    scale_factor: { type: "number" },
    zero_point: { type: "number" },
    pipeline_latency_cycles: { type: "number" },
    clock_period_ns: { type: "number" },
    input_width_bits: { type: "number" },
    output_width_bits: { type: "number" },
    valid_in_signal: { type: "string" },
    valid_out_signal: { type: "string" },
    clock_signal: { type: "string" },
    reset_signal: { type: "string" },
    golden_inputs: {
      type: "array",
      items: {
        type: "array",
        items: { type: "number" },
      },
    },
    golden_outputs: {
      type: "array",
      items: {
        type: "array",
        items: { type: "number" },
      },
    },
  },
} as const;

const pipelineIrSchema = {
  type: "object",
  additionalProperties: false,
  required: ["model_name", "quantization", "generated_at", "layers"],
  properties: {
    model_name: { type: "string" },
    quantization: { type: "string", const: "int8_symmetric_per_tensor" },
    generated_at: { type: "string" },
    layers: {
      type: "array",
      items: layerIrSchema,
    },
  },
} as const;

const verilogModuleSchema = {
  type: "object",
  additionalProperties: false,
  required: [
    "module_id",
    "spec_hash",
    "verilog_source",
    "generated_by",
    "attempt",
  ],
  properties: {
    module_id: { type: "string" },
    spec_hash: { type: "string" },
    verilog_source: { type: "string" },
    generated_by: { type: "string", enum: ["Foundry", "Surgeon"] },
    attempt: { type: "number", minimum: 1 },
  },
} as const;

const synthesisReportSchema = {
  type: "object",
  additionalProperties: false,
  required: ["success", "lut_count", "fmax_mhz", "report"],
  properties: {
    success: { type: "boolean" },
    lut_count: { type: "number" },
    fmax_mhz: { type: "number" },
    report: { type: "string" },
  },
} as const;

const pipelineIrOutputFormat: OutputFormat = {
  type: "json_schema",
  schema: pipelineIrSchema,
};

const verilogModuleOutputFormat: OutputFormat = {
  type: "json_schema",
  schema: verilogModuleSchema,
};

const verifResultOutputFormat: OutputFormat = {
  type: "json_schema",
  schema: verifResultSchema,
};

const synthesisReportOutputFormat: OutputFormat = {
  type: "json_schema",
  schema: synthesisReportSchema,
};

type AgentSlug = (typeof AGENT_SLUGS)[AgentName];

type AgentRunResult<T> = {
  payload: T;
  result: SDKResultMessage;
  messages: SDKMessage[];
};

type FrontmatterRecord = Record<string, string>;

function resolveFromSdk(relativePath: string): string {
  return path.resolve(__dirname, relativePath);
}

function reportPath(fileName: string): string {
  return path.join(resolveFromSdk(PIPELINE_CONFIG.reports_dir), fileName);
}

function normalizeAgentName(slug: AgentSlug): AgentName {
  const match = Object.entries(AGENT_SLUGS).find(([, value]) => value === slug);
  if (!match) {
    throw new Error(`No AgentName mapping found for slug '${slug}'.`);
  }

  return match[0] as AgentName;
}

function parseFrontmatter(markdown: string): { frontmatter: FrontmatterRecord; body: string } {
  const match = markdown.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/);
  if (!match) {
    throw new Error("Expected agent markdown to start with YAML frontmatter.");
  }

  const [, rawFrontmatter, body] = match;
  const frontmatter: FrontmatterRecord = {};

  for (const line of rawFrontmatter.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }

    const separatorIndex = trimmed.indexOf(":");
    if (separatorIndex === -1) {
      throw new Error(`Invalid frontmatter line '${trimmed}'.`);
    }

    const key = trimmed.slice(0, separatorIndex).trim();
    const value = trimmed.slice(separatorIndex + 1).trim();
    frontmatter[key] = value;
  }

  return { frontmatter, body: body.trim() };
}

function splitCsvField(value: string | undefined): string[] | undefined {
  if (!value) {
    return undefined;
  }

  const parts = value
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);

  return parts.length > 0 ? parts : undefined;
}

function isResultMessage(message: SDKMessage): message is SDKResultMessage {
  return message.type === "result" && "modelUsage" in message;
}

async function readText(filePath: string): Promise<string> {
  return readFile(filePath, "utf8");
}

async function pathExists(filePath: string): Promise<boolean> {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function readJsonFile<T>(filePath: string): Promise<T> {
  const raw = await readFile(filePath, "utf8");
  const parsed = JSON.parse(raw) as T;

  // TODO: Replace this cast-only helper with schema-backed validation so resume and artifact loading fail deterministically on malformed JSON.
  return parsed;
}

async function writeJsonFile(filePath: string, value: unknown): Promise<void> {
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

async function appendRunLog(entry: Record<string, unknown>): Promise<void> {
  const logPath = reportPath("run_log.jsonl");
  await mkdir(path.dirname(logPath), { recursive: true });
  await appendFile(logPath, `${JSON.stringify({ timestamp: new Date().toISOString(), ...entry })}\n`, "utf8");
}

async function ensureOutputLayout(): Promise<void> {
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

async function loadPluginAgentDefinition(slug: AgentSlug): Promise<AgentDefinition> {
  const agentName = normalizeAgentName(slug);
  const markdownPath = path.join(pluginPath, "agents", `${slug}.md`);
  const markdown = await readText(markdownPath);
  const { frontmatter, body } = parseFrontmatter(markdown);
  const skillMarkdownPath = path.join(pluginPath, "skills", slug, "SKILL.md");
  const skillMarkdown = (await pathExists(skillMarkdownPath))
    ? await readText(skillMarkdownPath)
    : "";
  const parsedSkill = skillMarkdown ? parseFrontmatter(skillMarkdown) : null;

  // TODO: If the plugin frontmatter becomes more expressive, replace this hand-rolled parser with a real YAML parser and shared schema validation.
  const builtInTools = splitCsvField(frontmatter.tools) ?? [];
  const disallowedTools = splitCsvField(frontmatter.disallowedTools);
  const mcpTools = [...AGENT_MCP_TOOLS[slug]];
  const combinedTools = [...new Set([...builtInTools, ...mcpTools])];
  const prompt = parsedSkill
    ? `${body}\n\nSupplemental skill reference:\n\n${parsedSkill.body}`
    : body;

  // TODO: Switch this back to AgentDefinition.skills once the published SDK typings expose the documented field and the installable package typechecks cleanly.
  // TODO: Restore AgentDefinition.maxTurns once the published SDK typings match the documented field; for now the parent query's maxTurns acts as the guardrail.

  return {
    description: AGENT_CONFIG[agentName].description,
    prompt,
    tools: combinedTools.length > 0 ? combinedTools : undefined,
    disallowedTools,
    model: AGENT_CONFIG[agentName].model,
  };
}

async function loadAllAgentDefinitions(): Promise<Record<string, AgentDefinition>> {
  const entries = await Promise.all(
    Object.values(AGENT_SLUGS).map(async (slug) => [slug, await loadPluginAgentDefinition(slug)] as const),
  );

  return Object.fromEntries(entries);
}

function buildDelegationPrompt(slug: AgentSlug, payload: unknown): string {
  return [
    `Invoke the \`${slug}\` subagent immediately.`,
    "Do not solve the task yourself.",
    "Do not use any other subagent.",
    "Return only the subagent's final JSON object.",
    "The only data channel into the subagent is this prompt string, so the payload is embedded below as JSON.",
    "",
    "Payload JSON:",
    JSON.stringify(payload, null, 2),
  ].join("\n");
}

function requireStructuredOutput<T>(result: SDKResultMessage, label: string): T {
  if (result.subtype !== "success") {
    throw new Error(`${label} query did not succeed: ${result.subtype}`);
  }

  if (result.structured_output !== undefined) {
    // TODO: Validate structured_output with AJV or Zod against the canonical schema so SDK validation and local validation cannot drift silently.
    return result.structured_output as T;
  }

  // TODO: Keep this JSON.parse fallback only until every agent call is enforced with outputFormat and the result path is covered by tests.
  return JSON.parse(result.result) as T;
}

async function runDelegatedAgent<T>(
  slug: AgentSlug,
  payload: unknown,
  outputFormat: OutputFormat,
): Promise<AgentRunResult<T>> {
  const agents = await loadAllAgentDefinitions();
  const messages: SDKMessage[] = [];
  let finalResult: SDKResultMessage | null = null;

  for await (const message of query({
    prompt: buildDelegationPrompt(slug, payload),
    options: {
      cwd: repoRoot,
      tools: ["Agent"],
      allowedTools: GLOBAL_ALLOWED_TOOLS,
      plugins: [{ type: "local", path: pluginPath }],
      agents,
      outputFormat,
      maxTurns: 6,
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
    payload: requireStructuredOutput<T>(finalResult, slug),
    result: finalResult,
    messages,
  };
}

function findLayer(pipelineIr: PipelineIR, moduleId: string): LayerIR {
  const layer = pipelineIr.layers.find((candidate) => candidate.module_id === moduleId);
  if (!layer) {
    throw new Error(`LayerIR for module '${moduleId}' was not found in output/layer_ir.json.`);
  }

  return layer;
}

async function loadPersistedVerilogModule(moduleId: string): Promise<VerilogModule> {
  const metaPath = path.join(resolveFromSdk(PIPELINE_CONFIG.rtl_dir), `${moduleId}.meta.json`);
  return readJsonFile<VerilogModule>(metaPath);
}

async function logStateTransition(
  manager: PipelineStateManager,
  moduleId: string,
  from: string,
  to: string,
  reason: string,
): Promise<void> {
  await appendRunLog({
    event: "state_transition",
    module_id: moduleId,
    from,
    to,
    reason,
    pipeline_state: manager.getState(),
  });
}

async function ensureLayerIr(
  checkpointPath: string,
): Promise<{
  pipelineIr: PipelineIR;
  bootstrapUsage?: {
    total_cost_usd: number;
    modelUsage: Record<string, ModelUsageEntry>;
  };
}> {
  const layerIrPath = resolveFromSdk(PIPELINE_CONFIG.layer_ir_path);

  if (await pathExists(layerIrPath)) {
    return {
      pipelineIr: await readJsonFile<PipelineIR>(layerIrPath),
    };
  }

  const payload = {
    checkpoint_path: checkpointPath,
    quantization_config: {
      quantization: "int8_symmetric_per_tensor",
    },
    output_path: layerIrPath,
  };

  await appendRunLog({
    event: "action",
    action: "invoke_cartographer",
    payload,
  });

  const result = await runDelegatedAgent<PipelineIR>("cartographer", payload, pipelineIrOutputFormat);

  await appendRunLog({
    event: "agent_result",
    agent: "Cartographer",
    total_cost_usd: result.result.total_cost_usd,
    modelUsage: result.result.modelUsage,
    payload: result.payload,
  });

  await writeJsonFile(layerIrPath, result.payload);
  return {
    pipelineIr: result.payload,
    bootstrapUsage: {
      total_cost_usd: result.result.total_cost_usd,
      modelUsage: result.result.modelUsage as Record<string, ModelUsageEntry>,
    },
  };
}

async function invokeYosys(module: VerilogModule): Promise<AgentRunResult<SynthesisReport>> {
  const messages: SDKMessage[] = [];
  let finalResult: SDKResultMessage | null = null;

  for await (const message of query({
    prompt: [
      "Call the run_yosys MCP tool exactly once with the payload below.",
      "Do not use built-in tools.",
      "Return only the tool result as JSON.",
      "",
      "Payload JSON:",
      JSON.stringify(
        {
          verilog_source: module.verilog_source,
          module_name: module.module_id,
        },
        null,
        2,
      ),
    ].join("\n"),
    options: {
      cwd: repoRoot,
      tools: [],
      allowedTools: [DIRECT_MCP_TOOLS.run_yosys],
      plugins: [{ type: "local", path: pluginPath }],
      outputFormat: synthesisReportOutputFormat,
      maxTurns: 3,
    },
  })) {
    messages.push(message);

    if (isResultMessage(message)) {
      finalResult = message;
    }
  }

  if (!finalResult) {
    throw new Error(`No final result message was received for run_yosys on '${module.module_id}'.`);
  }

  return {
    payload: requireStructuredOutput<SynthesisReport>(finalResult, "run_yosys"),
    result: finalResult,
    messages,
  };
}

async function invokeFoundry(layerIr: LayerIR): Promise<AgentRunResult<VerilogModule>> {
  await appendRunLog({
    event: "action",
    action: "invoke_foundry",
    module_id: layerIr.module_id,
  });

  const result = await runDelegatedAgent<VerilogModule>(
    "foundry",
    { layer_ir: layerIr },
    verilogModuleOutputFormat,
  );

  await appendRunLog({
    event: "agent_result",
    agent: "Foundry",
    module_id: layerIr.module_id,
    total_cost_usd: result.result.total_cost_usd,
    modelUsage: result.result.modelUsage,
    payload: result.payload,
  });

  return result;
}

async function invokeAssayer(
  module: VerilogModule,
  layerIr: LayerIR,
): Promise<AgentRunResult<VerifResult>> {
  await appendRunLog({
    event: "action",
    action: "invoke_assayer",
    module_id: module.module_id,
  });

  const result = await runDelegatedAgent<VerifResult>(
    "assayer",
    {
      module,
      layer_ir: layerIr,
      verilog_path: path.join(resolveFromSdk(PIPELINE_CONFIG.rtl_dir), `${module.module_id}.v`),
      sidecar_path: buildSidecarPath(module.module_id),
      testbench_template_path: resolveFromSdk(PIPELINE_CONFIG.static_testbench_path),
    },
    verifResultOutputFormat,
  );

  await appendRunLog({
    event: "agent_result",
    agent: "Assayer",
    module_id: module.module_id,
    total_cost_usd: result.result.total_cost_usd,
    modelUsage: result.result.modelUsage,
    payload: result.payload,
  });

  return result;
}

async function invokeSurgeon(
  brokenModule: VerilogModule,
  verifResult: VerifResult,
  layerIr: LayerIR,
): Promise<AgentRunResult<VerilogModule>> {
  await appendRunLog({
    event: "action",
    action: "invoke_surgeon",
    module_id: brokenModule.module_id,
  });

  const result = await runDelegatedAgent<VerilogModule>(
    "surgeon",
    {
      broken_module: brokenModule,
      verif_result: verifResult,
      layer_ir: layerIr,
    },
    verilogModuleOutputFormat,
  );

  await appendRunLog({
    event: "agent_result",
    agent: "Surgeon",
    module_id: brokenModule.module_id,
    total_cost_usd: result.result.total_cost_usd,
    modelUsage: result.result.modelUsage,
    payload: result.payload,
  });

  return result;
}

async function writePipelineSummary(
  manager: PipelineStateManager,
  pipelineIr: PipelineIR,
): Promise<void> {
  const summaryPath = reportPath("pipeline_summary.json");
  const summaryPayload = {
    run_id: manager.getState().run_id,
    completed_at: new Date().toISOString(),
    is_done: manager.isDone(),
    model_name: pipelineIr.model_name,
    modules_total: pipelineIr.layers.length,
    total_cost_usd: manager.getState().total_cost_usd,
    model_usage: manager.getState().model_usage,
    summary_table: manager.summary(),
    state: manager.getState(),
  };

  await writeJsonFile(summaryPath, summaryPayload);
  await appendRunLog({
    event: "pipeline_summary_written",
    path: summaryPath,
    payload: summaryPayload,
  });
}

export async function runPipeline(checkpointPath: string): Promise<void> {
  await ensureOutputLayout();

  const runLogPath = reportPath("run_log.jsonl");
  if (!ACTIVE_CLI_OPTIONS.resume) {
    await writeFile(runLogPath, "", "utf8");
  }

  await appendRunLog({
    event: "pipeline_start",
    checkpoint_path: checkpointPath,
    resume: ACTIVE_CLI_OPTIONS.resume,
  });

  const layerIrBootstrap = await ensureLayerIr(checkpointPath);
  const pipelineIr = layerIrBootstrap.pipelineIr;
  const moduleIds = pipelineIr.layers.map((layer) => layer.module_id);
  const statePath = resolveFromSdk(PIPELINE_CONFIG.pipeline_state_path);
  const manager = new PipelineStateManager(moduleIds, PIPELINE_CONFIG.max_retries);

  if (ACTIVE_CLI_OPTIONS.resume && (await pathExists(statePath))) {
    await manager.loadState(statePath);
    await appendRunLog({
      event: "pipeline_resume_loaded",
      state_path: statePath,
      state: manager.getState(),
    });
  } else {
    if (layerIrBootstrap.bootstrapUsage) {
      manager.recordAgentUsage(
        layerIrBootstrap.bootstrapUsage.total_cost_usd,
        layerIrBootstrap.bootstrapUsage.modelUsage,
      );
    }
    await manager.saveState(statePath);
    await appendRunLog({
      event: "pipeline_state_initialized",
      state_path: statePath,
      state: manager.getState(),
    });
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
        await logStateTransition(manager, tickModuleId, beforeStatus, afterStatus, nextAction.action);
      }
    }

    await manager.saveState(statePath);

    if (nextAction.action === "invoke_foundry") {
      const layer = findLayer(pipelineIr, nextAction.module_id);
      const foundryResult = await invokeFoundry(layer);
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
      );
      await manager.saveState(statePath);

      const assayerResult = await invokeAssayer(foundryResult.payload, layer);
      recordUsageFromResult(manager, assayerResult.result);
      const statusBeforeApply = manager.getState().modules[nextAction.module_id];
      manager.applyVerifResult(nextAction.module_id, assayerResult.payload);
      const statusAfterApply = manager.getState().modules[nextAction.module_id];
      await logStateTransition(
        manager,
        nextAction.module_id,
        statusBeforeApply,
        statusAfterApply,
        `assayer_${assayerResult.payload.status}`,
      );
      await manager.saveState(statePath);

      if (statusAfterApply === "pass") {
        // TODO: If synthesis failures should trigger retries, extend PipelineState and the retry logic to include post-verification Yosys outcomes.
        const yosysResult = await invokeYosys(foundryResult.payload);
        recordUsageFromResult(manager, yosysResult.result);
        await writeJsonFile(reportPath(`${nextAction.module_id}.yosys.json`), yosysResult.payload);
        await manager.saveState(statePath);
      }

      if (statusAfterApply === "fail_abort") {
        await appendRunLog({
          event: "module_fail_abort",
          module_id: nextAction.module_id,
          result: assayerResult.payload,
        });
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

      const surgeonResult = await invokeSurgeon(brokenModule, verifResult, layer);
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
      );
      await manager.saveState(statePath);

      const assayerResult = await invokeAssayer(surgeonResult.payload, layer);
      recordUsageFromResult(manager, assayerResult.result);
      const statusBeforeApply = manager.getState().modules[nextAction.module_id];
      manager.applyVerifResult(nextAction.module_id, assayerResult.payload);
      const statusAfterApply = manager.getState().modules[nextAction.module_id];
      await logStateTransition(
        manager,
        nextAction.module_id,
        statusBeforeApply,
        statusAfterApply,
        `assayer_${assayerResult.payload.status}`,
      );
      await manager.saveState(statePath);

      if (statusAfterApply === "pass") {
        // TODO: If synthesis failures should trigger retries, extend PipelineState and the retry logic to include post-verification Yosys outcomes.
        const yosysResult = await invokeYosys(surgeonResult.payload);
        recordUsageFromResult(manager, yosysResult.result);
        await writeJsonFile(reportPath(`${nextAction.module_id}.yosys.json`), yosysResult.payload);
        await manager.saveState(statePath);
      }

      if (statusAfterApply === "fail_abort") {
        await appendRunLog({
          event: "module_fail_abort",
          module_id: nextAction.module_id,
          result: assayerResult.payload,
        });
      }

      continue;
    }

    // TODO: If the state machine is expanded to schedule Assayer as a first-class tick() action, implement that path here instead of throwing.
    throw new Error(`Unhandled pipeline action '${JSON.stringify(nextAction)}'.`);
  }

  await writePipelineSummary(manager, pipelineIr);
  await appendRunLog({
    event: "pipeline_complete",
    run_id: manager.getState().run_id,
    summary: manager.summary(),
  });
}

function parseCliArgs(argv: string[]): { checkpointPath: string; resume: boolean } {
  const resume = argv.includes("--resume");
  const positional = argv.filter((arg) => !arg.startsWith("--"));

  if (positional.length < 1) {
    throw new Error("Usage: tsx orchestrate.ts <checkpoint-path> [--resume]");
  }

  // TODO: Extend CLI parsing if the pipeline grows knobs like alternate plugins, custom output roots, or per-run retry budgets.
  return {
    checkpointPath: positional[0],
    resume,
  };
}

async function main(): Promise<void> {
  const cli = parseCliArgs(process.argv.slice(2));
  ACTIVE_CLI_OPTIONS.resume = cli.resume;
  await runPipeline(cli.checkpointPath);
}

main().catch(async (error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  await ensureOutputLayout().catch(() => undefined);
  await appendRunLog({
    event: "pipeline_error",
    error: message,
  }).catch(() => undefined);
  console.error(message);
  process.exitCode = 1;
});
