import { cp, mkdir, mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import {
  createOrchestratorRuntime,
  runPipeline,
} from "../orchestrate.js";
import type { SDKMessage, SDKResultMessage } from "../claude-agent-sdk-compat.js";
import type { PipelineIR } from "../types.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "../..");
const outputRoot = path.join(repoRoot, "output");
const reportsDir = path.join(outputRoot, "reports");
const rtlDir = path.join(outputRoot, "rtl");
const knowledgeRoot = path.join(repoRoot, "knowledge");
const outputResetTargets = [
  "reports",
  "rtl",
  "tb",
  "weights",
  "layer_ir.json",
  "layer_ir.json.checkpoint",
  "pipeline_state.json",
  "contract_state.json",
  "golden_vectors.json",
] as const;
const knowledgeResetTargets = [
  "knowledge/doc_lifecycle.json",
  "knowledge/patterns/active",
  "knowledge/patterns/probationary",
  "knowledge/patterns/archive",
  "knowledge/references/active",
  "knowledge/references/probationary",
  "knowledge/references/archive",
] as const;
let outputBackupRoot: string | null = null;
let knowledgeBackupRoot: string | null = null;

type MockStepContext = {
  prompt: string;
  options?: { resume?: string };
};
type MockStep = SDKResultMessage | ((context: MockStepContext) => Promise<SDKResultMessage> | SDKResultMessage);
type VivadoReport = {
  success: boolean;
  lut_count: number;
  fmax_mhz: number;
  timing_met?: boolean;
  wns_ns?: number | null;
  ff_count?: number;
  dsp_count?: number;
  bram18_count?: number;
  bram36_count?: number;
  bram18_equiv?: number;
  report: string;
};
type VivadoStep = VivadoReport | (() => Promise<VivadoReport> | VivadoReport);
type VerifLike = Record<string, unknown>;
type AssayerStep = VerifLike | (() => Promise<VerifLike> | VerifLike);

const fixedNow = () => new Date("2026-04-14T00:00:00Z");

function createVivadoMock(steps: VivadoStep[]): ReturnType<typeof vi.fn> {
  return vi.fn(async () => {
    const next = steps.shift();
    if (!next) {
      throw new Error("No mock result queued for synthesisFn.");
    }
    const report = typeof next === "function" ? await next() : next;
    return {
      tool: "vivado",
      part: "xczu9eg-ffvb1156-2-e",
      stage: "synth",
      ff_count: 0,
      dsp_count: 0,
      bram18_count: 0,
      bram36_count: 0,
      bram18_equiv: 0,
      wns_ns: report.fmax_mhz > 0 ? 1 : null,
      timing_met: report.fmax_mhz >= 50,
      ...report,
    };
  });
}

// Assayer is now deterministic in production (orchestrator calls run_iverilog
// + run_verilator directly), so tests inject a VerifResult stream here.
function createAssayerMock(steps: AssayerStep[]): ReturnType<typeof vi.fn> {
  return vi.fn(async () => {
    const next = steps.shift();
    if (!next) {
      throw new Error("No mock result queued for assayerFn.");
    }
    return typeof next === "function" ? await next() : next;
  });
}

async function resetOutput(): Promise<void> {
  for (const dir of ["reports", "rtl", "tb", "weights"]) {
    const fullDir = path.join(outputRoot, dir);
    for (const entry of await readdir(fullDir)) {
      if (entry === ".gitkeep") {
        continue;
      }
      await rm(path.join(fullDir, entry), { recursive: true, force: true });
    }
  }

  for (const fileName of ["layer_ir.json", "layer_ir.json.checkpoint", "pipeline_state.json", "contract_state.json", "golden_vectors.json"]) {
    await rm(path.join(outputRoot, fileName), { force: true });
  }
}

function isEnoent(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    (error as { code?: string }).code === "ENOENT"
  );
}

async function copyPathIfPresent(source: string, destination: string): Promise<void> {
  await mkdir(path.dirname(destination), { recursive: true });
  try {
    await cp(source, destination, { recursive: true, force: true });
  } catch (error: unknown) {
    if (!isEnoent(error)) throw error;
  }
}

async function writeFixture(relativeFixturePath: string, destinationPath: string): Promise<unknown> {
  const raw = await readFile(path.join(repoRoot, "test", "fixtures", relativeFixturePath), "utf8");
  await writeFile(destinationPath, raw, "utf8");
  return JSON.parse(raw);
}

async function writePipelineIrFixture(): Promise<unknown> {
  const pipelineIr = await writeFixture("pipeline_ir.json", path.join(outputRoot, "layer_ir.json"));
  await writeFile(
    path.join(outputRoot, "layer_ir.json.checkpoint"),
    `${path.resolve("checkpoint.pth")}\n`,
    "utf8",
  );
  return pipelineIr;
}

/**
 * Like `writePipelineIrFixture` but forces the unit_module to a 5x5 conv —
 * a kernel with no protected/active/probationary doc coverage in the seeded
 * knowledge tree. Use this from tests that exercise the self-improve doc
 * lifecycle (probationary creation, promotion, archive) so the doc-coverage
 * guard does not suppress the wrapper schema and the test mocks' wrapper
 * `{module, draft_doc}` payload remains valid.
 */
async function writeUncoveredPipelineIrFixture(): Promise<{ layers: Array<Record<string, unknown>> }> {
  const pipelineIr = (await writePipelineIrFixture()) as { layers: Array<Record<string, unknown>> };
  pipelineIr.layers[0].weight_shape = [1, 1, 5, 5];
  pipelineIr.layers[0].num_weights = 25;
  await writeFile(
    path.join(outputRoot, "layer_ir.json"),
    `${JSON.stringify(pipelineIr, null, 2)}\n`,
    "utf8",
  );
  return pipelineIr;
}

/**
 * Strip `verilog_source` from any module-shaped object found at the top level
 * or nested under `module:` so test fixtures don't have to care about the
 * agent-output-schema split (verilog_source moved out of the agent contract;
 * the orchestrator hydrates it from disk after the agent returns). This
 * keeps existing fixtures (`successResult(module)`,
 * `successRtlWithDoc(module)`) authoring full modules while letting the
 * SDK validation see the metadata-only shape.
 */
function stripVerilogSourceForAgentSchema(payload: unknown): unknown {
  if (payload === null || typeof payload !== "object") return payload;
  const obj = payload as Record<string, unknown>;
  if ("verilog_source" in obj && "module_id" in obj && "spec_hash" in obj) {
    const { verilog_source: _omit, ...rest } = obj;
    void _omit;
    return rest;
  }
  if ("module" in obj && obj.module && typeof obj.module === "object") {
    return { ...obj, module: stripVerilogSourceForAgentSchema(obj.module) };
  }
  return obj;
}

function successResult(structured_output: unknown, sessionId?: string): SDKResultMessage {
  const stripped = stripVerilogSourceForAgentSchema(structured_output);
  return {
    type: "result",
    subtype: "success",
    result: JSON.stringify(stripped),
    structured_output: stripped,
    total_cost_usd: 1,
    modelUsage: { fixture: { input_tokens: 1, output_tokens: 1 } },
    ...(sessionId ? { session_id: sessionId } : {}),
  };
}

let activeMockPrompt: string | null = null;

function extractPayloadFromPrompt(prompt: string): Record<string, unknown> | null {
  const markers = ["Final-attempt payload JSON:", "Payload JSON:"];
  for (const marker of markers) {
    const markerIndex = prompt.lastIndexOf(marker);
    if (markerIndex < 0) continue;
    const raw = prompt.slice(markerIndex + marker.length).trim();
    const jsonStart = raw.indexOf("{");
    if (jsonStart < 0) continue;
    try {
      const parsed = JSON.parse(raw.slice(jsonStart));
      return parsed && typeof parsed === "object" && !Array.isArray(parsed)
        ? (parsed as Record<string, unknown>)
        : null;
    } catch {
      return null;
    }
  }
  return null;
}

function activePromptExpectedSpecHash(): string | null {
  if (!activeMockPrompt) return null;
  const payload = extractPayloadFromPrompt(activeMockPrompt);
  return typeof payload?.expected_spec_hash === "string"
    ? payload.expected_spec_hash
    : null;
}

/**
 * Writes a mock Foundry/Surgeon RTL deliverable to disk before the agent's
 * structured output is parsed. The orchestrator now hydrates VerilogModule
 * from the .v on disk rather than from `verilog_source` in the agent's
 * final JSON, so mocks must materialize both `<module_id>.v` (the source
 * orchestrator reads) and `<module_id>.meta.json` (the historical
 * pipeline meta record). Returns the SDKResultMessage with verilog_source
 * stripped (mirroring the new agent-output schema).
 */
async function persistMockRtlDeliverable(
  module: { module_id: string; verilog_source?: string; [key: string]: unknown },
): Promise<void> {
  const expectedSpecHash = activePromptExpectedSpecHash();
  if (expectedSpecHash) {
    module.spec_hash = expectedSpecHash;
  }
  const verilogSource =
    typeof module.verilog_source === "string"
      ? module.verilog_source
      : `module ${module.module_id}; endmodule\n`;
  await writeFile(path.join(rtlDir, `${module.module_id}.v`), verilogSource, "utf8");
  await writeFile(
    path.join(rtlDir, `${module.module_id}.meta.json`),
    `${JSON.stringify(module, null, 2)}\n`,
    "utf8",
  );
}

function docDraft(title = "Generated RTL note"): Record<string, string> {
  return {
    title,
    pattern_markdown: "Use the same registered pipeline shape and keep LayerIR latency authoritative.",
    reference_verilog: "module reference_generated; endmodule",
  };
}

function successRtlWithDoc(module: unknown, sessionId?: string): SDKResultMessage {
  return successResult({ module, draft_doc: docDraft() }, sessionId);
}

// Probationary / archive tiers and doc_lifecycle.json are growth artifacts
// from real LLM-driven pipeline runs (e.g. node_conv_248's tiled-streaming
// win). Real runs legitimately leave files in
// `knowledge/patterns/probationary/`,
// `knowledge/references/probationary/`, and entries in
// `knowledge/doc_lifecycle.json`, but those leak into test execution and
// silently provide "covering doc" matches against the unit_module fixture
// — which suppresses `create_new_doc_request` in tests that depend on it.
// Always reset these to empty during the test suite, regardless of what
// the on-disk backup snapshotted. Tests that need fixture docs use
// `seedLifecycleDoc` to write a known doc into a known-empty baseline.
const knowledgeAlwaysEmptyTiers: ReadonlyArray<string> = [
  "knowledge/doc_lifecycle.json",
  "knowledge/patterns/probationary",
  "knowledge/patterns/archive",
  "knowledge/references/probationary",
  "knowledge/references/archive",
];

async function resetKnowledgeLifecycle(): Promise<void> {
  if (!knowledgeBackupRoot) return;
  for (const target of knowledgeResetTargets) {
    const originalPath = path.join(repoRoot, target);
    await rm(originalPath, { recursive: true, force: true });
    if (knowledgeAlwaysEmptyTiers.includes(target)) {
      // Recreate empty for directory targets so subsequent test code that
      // copies fixture docs into the tier doesn't fail on a missing
      // directory. For file targets (`doc_lifecycle.json`) we leave the
      // path absent — `loadDocLifecycleState` returns an empty state when
      // the file is missing.
      if (!target.endsWith(".json")) {
        await mkdir(originalPath, { recursive: true });
      }
      continue;
    }
    await copyPathIfPresent(path.join(knowledgeBackupRoot, target), originalPath);
  }
}

async function seedLifecycleDoc(args: {
  id: string;
  status: "probationary" | "active";
  op_type?: string;
  contract_id?: "flat-bus" | "tiled-streaming" | "dram-backed";
  used_by_modules?: string[];
  successful_modules?: string[];
}): Promise<void> {
  const opType = args.op_type ?? "conv2d";
  const contractId = args.contract_id ?? "flat-bus";
  const patternRel = `knowledge/patterns/${args.status}/${args.id}.md`;
  const referenceRel = `knowledge/references/${args.status}/${args.id}.v`;
  await mkdir(path.dirname(path.join(repoRoot, patternRel)), { recursive: true });
  await mkdir(path.dirname(path.join(repoRoot, referenceRel)), { recursive: true });
  await writeFile(path.join(repoRoot, patternRel), `# ${args.id}\n\nseed pattern\n`, "utf8");
  await writeFile(path.join(repoRoot, referenceRel), `module ${args.id}; endmodule\n`, "utf8");
  await writeFile(
    path.join(knowledgeRoot, "doc_lifecycle.json"),
    `${JSON.stringify({
      version: 1,
      docs: {
        [args.id]: {
          id: args.id,
          op_type: opType,
          contract_id: contractId,
          contract_key: `${contractId}:seed_hash`,
          spec_hash: "seed_hash",
          status: args.status,
          pattern_path: patternRel,
          reference_path: referenceRel,
          created_by_module: "seed",
          created_by_agent: "Foundry",
          created_at: fixedNow().toISOString(),
          used_by_modules: args.used_by_modules ?? [],
          successful_modules: args.successful_modules ?? [],
          failed_modules: [],
        },
      },
    }, null, 2)}\n`,
    "utf8",
  );
}

const fixtureContractKeys = {
  flat: "flat-bus:conv2d_1x1x1x1_s1x1_i8_o8",
  tiled: "tiled-streaming:conv2d_1x1x1x1_s1x1_i8_o8_iotiled-streaming_tile1",
  dram: "dram-backed-weights:conv2d_1x1x1x1_s1x1_i8_o8_iodram-backed-weights_tile1",
} as const;

async function seedContractFlags(keys: Array<keyof typeof fixtureContractKeys>): Promise<void> {
  const docs = Object.fromEntries(
    keys.map((keyName) => {
      const key = fixtureContractKeys[keyName];
      const [contractId, specHash] = key.split(":", 2);
      return [
        key,
        {
          key,
          contract_id: contractId,
          spec_hash: specHash,
          op_type: "conv2d",
          status: "manual_correction_needed",
          flagged_at: fixedNow().toISOString(),
          updated_at: fixedNow().toISOString(),
          module_ids: ["unit_module"],
          reason: "seeded test flag",
        },
      ];
    }),
  );
  await writeFile(
    path.join(outputRoot, "contract_state.json"),
    `${JSON.stringify({ version: 1, contracts: docs }, null, 2)}\n`,
    "utf8",
  );
}

function createQueryMock(
  steps: Partial<Record<"cartographer" | "foundry" | "surgeon" | "failure_classifier" | "retrospector", MockStep[]>>,
): ReturnType<typeof vi.fn> {
  return vi.fn(async function* ({
    prompt,
    options,
  }: {
    prompt: string;
    options?: { resume?: string };
  }): AsyncGenerator<SDKMessage, void> {
    // The collapsed single-layer dispatch (Fix B) puts the prompt in the
    // form "You are the `<slug>` agent. Execute the task described below."
    const key =
      prompt.includes("You are the `cartographer`") ? "cartographer"
        : prompt.includes("You are the `foundry`") ? "foundry"
        : prompt.includes("existing `foundry` agent conversation") || options?.resume ? "foundry"
        : prompt.includes("You are the `surgeon`") ? "surgeon"
        : prompt.includes("You are the `failure_classifier`") ? "failure_classifier"
        : prompt.includes("You are the `retrospector`") ? "retrospector"
        : null;

    if (!key) {
      throw new Error(`Unhandled prompt: ${prompt}`);
    }

    const queue = steps[key];
    if (key === "failure_classifier" && (!queue || queue.length === 0)) {
      const evidence = prompt.split("Evidence JSON:").pop() ?? prompt;
      const architectural =
        /"failure_class":\s*"architectural_unsupported"|"capability_gate":|requires tiled channel streaming|DSP48 exhausted|resource utilization exceeds available/i.test(evidence);
      yield successResult(
        architectural
          ? {
              category: "architectural_fit",
              violated_resource: null,
              violated_constraint:
                evidence.includes("MAX_SUPPORTED_BUS_BITS") ||
                evidence.includes("max_bus_width_bits") ||
                evidence.includes("max_supported_bus_bits")
                  ? "MAX_SUPPORTED_BUS_BITS"
                  : "resource_or_constraint_from_logs",
              rationale: "Mock classifier found a contract-fit indicator in the prompt.",
            }
          : {
              category: "code_bug",
              violated_resource: null,
              violated_constraint: null,
              rationale: "Mock classifier defaulted retryable failures to code_bug.",
            },
      );
      return;
    }

    if (!queue || queue.length === 0) {
      throw new Error(`No mock result queued for ${key}.`);
    }

    const next = queue.shift();
    const previousPrompt = activeMockPrompt;
    activeMockPrompt = prompt;
    try {
      const message = typeof next === "function" ? await next({ prompt, options }) : next;
      yield message;
    } finally {
      activeMockPrompt = previousPrompt;
    }
  });
}

beforeAll(async () => {
  outputBackupRoot = await mkdtemp(path.join(os.tmpdir(), "nn2rtl-sdk-output-backup-"));
  for (const target of outputResetTargets) {
    await copyPathIfPresent(path.join(outputRoot, target), path.join(outputBackupRoot, target));
  }
  knowledgeBackupRoot = await mkdtemp(path.join(os.tmpdir(), "nn2rtl-sdk-knowledge-backup-"));
  for (const target of knowledgeResetTargets) {
    await copyPathIfPresent(path.join(repoRoot, target), path.join(knowledgeBackupRoot, target));
  }
});

beforeEach(async () => {
  await resetOutput();
  await resetKnowledgeLifecycle();
});

afterEach(async () => {
  await resetOutput();
  await resetKnowledgeLifecycle();
});

afterAll(async () => {
  if (outputBackupRoot) {
    for (const target of outputResetTargets) {
      const originalPath = path.join(outputRoot, target);
      await rm(originalPath, { recursive: true, force: true });
      await copyPathIfPresent(path.join(outputBackupRoot, target), originalPath);
    }
    await rm(outputBackupRoot, { recursive: true, force: true });
    outputBackupRoot = null;
  }
  if (knowledgeBackupRoot) {
    for (const target of knowledgeResetTargets) {
      const originalPath = path.join(repoRoot, target);
      await rm(originalPath, { recursive: true, force: true });
      await copyPathIfPresent(path.join(knowledgeBackupRoot, target), originalPath);
    }
    await rm(knowledgeBackupRoot, { recursive: true, force: true });
    knowledgeBackupRoot = null;
  }
});

describe("runPipeline", () => {
  it("uses an existing layer_ir.json without invoking Cartographer", async () => {
    const pipelineIr = await writePipelineIrFixture();
    const module = await writeFixture("verilog_module.json", path.join(rtlDir, "unit_module.meta.json"));
    await writeFile(path.join(rtlDir, "unit_module.v"), (module as { verilog_source: string }).verilog_source, "utf8");
    const verifPass = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verif_pass.json"), "utf8"),
    );

    const queryFn = createQueryMock({
      foundry: [async () => {
        await persistMockRtlDeliverable(module);
        return successResult(module);
      }],
    });
    const assayerFn = createAssayerMock([verifPass]);
    const synthesisFn = createVivadoMock([{ success: true, lut_count: 1, fmax_mhz: 75, report: "fixture" }]);

    await runPipeline("checkpoint.pth", {
      runtime: createOrchestratorRuntime({ now: fixedNow, queryFn, synthesisFn, assayerFn }),
    });

    expect(queryFn.mock.calls.some(([call]) => (call as { prompt: string }).prompt.includes("cartographer"))).toBe(false);
    // Single-layer dispatch (Fix B) passes allowedTools equal to the merged
    // agent tools list; Agent/Task are in disallowedTools, not allowedTools.
    expect(queryFn.mock.calls[0]?.[0]).toMatchObject({
      options: {
        maxTurns: 40,
        allowedTools: expect.arrayContaining(["Bash", "mcp__nn2rtl-tools__write_verilog"]),
      },
    });
    expect(JSON.parse(await readFile(path.join(reportsDir, "pipeline_summary.json"), "utf8"))).toMatchObject({
      model_name: (pipelineIr as { model_name: string }).model_name,
      is_done: true,
    });
  });

  it("calls deterministic read_weights when layer_ir.json is missing (Cartographer LLM bypassed)", async () => {
    // SYSTEM_REVIEW_FINDINGS #8: production extraction now calls
    // deterministic `read_weights` directly. The Cartographer LLM agent is
    // no longer dispatched for layer_ir.json bootstrap; this test pins that
    // contract by injecting a `readWeightsFn` runtime stub and asserting
    // (a) the stub fires, (b) no Cartographer prompt is sent.
    const pipelineIr = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "pipeline_ir.json"), "utf8"),
    );
    const module = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verilog_module.json"), "utf8"),
    );
    const verifPass = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verif_pass.json"), "utf8"),
    );

    const queryFn = createQueryMock({
      foundry: [async () => {
        await persistMockRtlDeliverable(module);
        return successResult(module);
      }],
    });
    const assayerFn = createAssayerMock([verifPass]);
    const synthesisFn = createVivadoMock([{ success: true, lut_count: 2, fmax_mhz: 75, report: "fixture" }]);
    const readWeightsFn = vi.fn(async () => pipelineIr as PipelineIR);

    await runPipeline("checkpoint.pth", {
      runtime: createOrchestratorRuntime({ now: fixedNow, queryFn, synthesisFn, assayerFn, readWeightsFn }),
    });

    expect(readWeightsFn).toHaveBeenCalledOnce();
    expect(queryFn.mock.calls.some(([call]) => (call as { prompt: string }).prompt.includes("cartographer"))).toBe(false);
    expect(JSON.parse(await readFile(path.join(outputRoot, "layer_ir.json"), "utf8"))).toEqual(pipelineIr);

    const log = await readFile(path.join(reportsDir, "run_log.jsonl"), "utf8");
    expect(log).toContain('"event":"cartographer_bypassed_deterministic_read_weights"');
  });

  it("runs the Surgeon repair path after one Foundry attempt fails verification", async () => {
    // New flow: 1 Foundry call then 1 Surgeon repair.
    await writePipelineIrFixture();
    const originalModule = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verilog_module.json"), "utf8"),
    );
    const repairedModule = {
      ...originalModule,
      generated_by: "Surgeon",
      attempt: 2,
    };
    const verifFail = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verif_fail.json"), "utf8"),
    );
    const verifPass = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verif_pass.json"), "utf8"),
    );

    const queryFn = createQueryMock({
      foundry: [async () => {
        await persistMockRtlDeliverable(originalModule);
        return successResult(originalModule, "foundry-session-1");
      }],
      surgeon: [async () => {
        await persistMockRtlDeliverable(repairedModule);
        return successResult(repairedModule);
      }],
    });
    const assayerFn = createAssayerMock([verifFail, verifPass]);
    const synthesisFn = createVivadoMock([{ success: true, lut_count: 3, fmax_mhz: 75, report: "fixture" }]);

    await runPipeline("checkpoint.pth", {
      runtime: createOrchestratorRuntime({ now: fixedNow, queryFn, synthesisFn, assayerFn }),
    });

    const prompts = queryFn.mock.calls.map(([call]) => (call as { prompt: string }).prompt);
    expect(prompts.some((prompt) => prompt.includes("You are the `surgeon`"))).toBe(true);
    const state = JSON.parse(await readFile(path.join(outputRoot, "pipeline_state.json"), "utf8"));
    expect(state.modules.unit_module).toBe("pass");
    expect(state.attempts.unit_module).toBe(2);
  });

  it("reverts Surgeon output and logs surgeon_regression_reverted when first_mismatch_index goes backward", async () => {
    await writePipelineIrFixture();
    const originalModule = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verilog_module.json"), "utf8"),
    );
    const repairedModule = { ...originalModule, generated_by: "Surgeon", attempt: 2 };

    // Foundry verif: correct timing, first 361 outputs are exact.
    const foundryVerif = {
      module_id: "unit_module",
      status: "fail",
      status_class: "sim_completed_mismatch",
      timing_pass: true,
      timing_actual_cycles: 1,
      timing_expected_cycles: 1,
      max_error: 50,
      mean_error: 33.0,
      sample_count: 12433,
      first_mismatch_index: 361,
      failure_class: "loop_bounds_incorrect",
    };
    // Surgeon verif: first_mismatch regressed to 0 — Surgeon broke what Foundry got right.
    const surgeonVerif = {
      module_id: "unit_module",
      status: "fail",
      status_class: "sim_completed_mismatch",
      timing_pass: true,
      timing_actual_cycles: 1,
      timing_expected_cycles: 1,
      max_error: 50,
      mean_error: 33.5,
      sample_count: 12433,
      first_mismatch_index: 0,
      failure_class: "loop_bounds_incorrect",
    };

    const queryFn = createQueryMock({
      foundry: [async () => {
        await persistMockRtlDeliverable(originalModule);
        return successResult(originalModule);
      }],
      surgeon: [async () => {
        await persistMockRtlDeliverable(repairedModule);
        return successResult(repairedModule);
      }],
    });
    // New flow: Foundry fails, then Surgeon repairs but regresses first_mismatch_index.
    const assayerFn = createAssayerMock([foundryVerif, surgeonVerif]);
    const synthesisFn = createVivadoMock([]);

    await runPipeline("checkpoint.pth", {
      maxRetries: 2,
      // Pin self_improve off so the contract walker doesn't activate — this
      // test exercises the Surgeon regression-revert path, not the contract
      // walk.
      selfImprove: false,
      runtime: createOrchestratorRuntime({ now: fixedNow, queryFn, synthesisFn, assayerFn }),
    });

    const log = await readFile(path.join(reportsDir, "run_log.jsonl"), "utf8");
    expect(log).toContain('"event":"surgeon_regression_reverted"');

    const state = JSON.parse(await readFile(path.join(outputRoot, "pipeline_state.json"), "utf8"));
    expect(state.modules.unit_module).toBe("fail_abort");

    // Reverted module on disk must match the original (Foundry) source, not Surgeon's.
    const diskMeta = JSON.parse(await readFile(path.join(rtlDir, "unit_module.meta.json"), "utf8"));
    expect(diskMeta.generated_by).toBe("Foundry");
  });

  it("runs one retrospector pass after retry exhaustion and resumes the Foundry session", async () => {
    // Uncovered geometry so the doc-coverage guard keeps the wrapper schema
    // active for both Foundry and Surgeon — the mocks return
    // `{module, draft_doc}` payloads.
    await writeUncoveredPipelineIrFixture();
    await seedLifecycleDoc({
      id: "auto_test_active_doc_fault",
      status: "active",
      used_by_modules: ["unit_module"],
    });
    const originalModule = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verilog_module.json"), "utf8"),
    );
    const surgeonModule = { ...originalModule, generated_by: "Surgeon", attempt: 2 };
    const finalFoundryModule = {
      ...originalModule,
      verilog_source: `${originalModule.verilog_source}\n// post-retrospector final attempt\n`,
      attempt: 2,
    };
    const verifFail = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verif_fail.json"), "utf8"),
    );
    const verifPass = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verif_pass.json"), "utf8"),
    );

    const queryFn = createQueryMock({
      foundry: [
        // Call 1: fresh session.
        async () => {
          await persistMockRtlDeliverable(originalModule);
          return successRtlWithDoc(originalModule, "foundry-session-1");
        },
        // Call 2 (post-retrospector final): same resumed session + advice.
        async () => {
          await persistMockRtlDeliverable(finalFoundryModule);
          return successRtlWithDoc(finalFoundryModule, "foundry-session-1");
        },
      ],
      surgeon: [
        async () => {
          await persistMockRtlDeliverable(surgeonModule);
          return successRtlWithDoc(surgeonModule);
        },
      ],
      retrospector: [
        successResult({
          analysis: "Both attempts preserved the same bad output schedule.",
          suggestion: "Keep the contract and rebuild the output counter around the documented latency.",
          doc_fault: true,
          faulty_doc_paths: ["auto_test_active_doc_fault"],
        }),
      ],
    });
    // Foundry fail, Surgeon fail (fail_abort), retrospector advisory,
    // final Foundry passes. 3 verifications total.
    const assayerFn = createAssayerMock([verifFail, verifFail, verifPass]);
    const synthesisFn = createVivadoMock([{ success: true, lut_count: 3, fmax_mhz: 75, report: "fixture" }]);

    await runPipeline("checkpoint.pth", {
      maxRetries: 2,
      selfImprove: true,
      runtime: createOrchestratorRuntime({ now: fixedNow, queryFn, synthesisFn, assayerFn }),
    });

    const prompts = queryFn.mock.calls.map(([call]) => (call as { prompt: string }).prompt);
    expect(prompts.filter((prompt) => prompt.includes("You are the `retrospector`"))).toHaveLength(1);
    expect(
      queryFn.mock.calls.some(([call]) => (call as { options?: { resume?: string } }).options?.resume === "foundry-session-1"),
    ).toBe(true);
    expect(prompts.some((prompt) => prompt.includes("continuing the same `foundry` agent conversation"))).toBe(true);

    const state = JSON.parse(await readFile(path.join(outputRoot, "pipeline_state.json"), "utf8"));
    expect(state.modules.unit_module).toBe("pass");
    expect(state.attempts.unit_module).toBe(2);
    expect(Object.values(state.retrospector_calls)).toEqual([1]);
    expect(synthesisFn).toHaveBeenCalledTimes(1);

    const lifecycle = JSON.parse(await readFile(path.join(knowledgeRoot, "doc_lifecycle.json"), "utf8"));
    expect(lifecycle.docs.auto_test_active_doc_fault.status).toBe("archived");
    expect(lifecycle.docs.auto_test_active_doc_fault.archive_reason).toContain("retrospector_doc_fault");
    expect(
      Object.values(lifecycle.docs).some(
        (doc) =>
          typeof doc === "object" &&
          doc !== null &&
          (doc as { replacement_for?: string[] }).replacement_for?.includes("auto_test_active_doc_fault"),
      ),
    ).toBe(true);

    const log = await readFile(path.join(reportsDir, "run_log.jsonl"), "utf8");
    expect(log).toContain('"event":"retrospector_result"');
    expect(log).toContain('"action":"invoke_foundry_after_retrospector"');
  });

  it("flags an exhausted contract and switches to the next available contract", async () => {
    // 1x1 fixture so the contract_state keys (`fixtureContractKeys.flat`,
    // `fixtureContractKeys.tiled`) — derived from the 1x1 spec_hash — line up
    // with what the orchestrator actually flags. The doc-coverage guard
    // suppresses the wrapper schema on the initial flat-bus pass (covered by
    // `02_conv1x1.md`); the mocks return plain `{VerilogModule}` for both
    // flat-bus attempts and the wrapper for the uncovered tiled-streaming
    // pass.
    await writePipelineIrFixture();
    const module = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verilog_module.json"), "utf8"),
    );
    const tiledModule = {
      ...module,
      spec_hash: "conv2d_1x1x1x1_s1x1_i8_o8_iotiled-streaming_tile1",
      verilog_source: `${module.verilog_source}\n// tiled contract attempt\n`,
    };
    const verifFail = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verif_fail.json"), "utf8"),
    );
    const verifPass = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verif_pass.json"), "utf8"),
    );

    const queryFn = createQueryMock({
      foundry: [
        // Flat-bus 1x1 is covered by `02_conv1x1.md` → guard suppresses
        // wrapper schema → plain `{VerilogModule}` is what Foundry returns.
        async () => {
          await persistMockRtlDeliverable(module);
          return successResult(module, "flat-session");
        },
        async () => {
          await persistMockRtlDeliverable(module);
          return successResult(module, "flat-session");
        },
        // Tiled-streaming has no doc coverage seeded for this test → wrapper
        // schema active → `{module, draft_doc}` is required.
        async () => {
          await persistMockRtlDeliverable(tiledModule);
          return successRtlWithDoc(tiledModule, "tiled-session");
        },
      ],
      retrospector: [
        successResult({
          analysis: "The flat-bus contract keeps reproducing the same failure.",
          suggestion: "Try a channel-tiled interface so the datapath can be rebuilt under a simpler stream width.",
        }),
      ],
    });
    const assayerFn = createAssayerMock([verifFail, verifFail, verifPass]);
    const synthesisFn = createVivadoMock([{ success: true, lut_count: 1, fmax_mhz: 75, report: "fixture" }]);

    await runPipeline("checkpoint.pth", {
      maxRetries: 0,
      selfImprove: true,
      runtime: createOrchestratorRuntime({ now: fixedNow, queryFn, synthesisFn, assayerFn }),
    });

    const prompts = queryFn.mock.calls.map(([call]) => (call as { prompt: string }).prompt);
    expect(prompts.some((prompt) => prompt.includes("contract variant: tiled-streaming"))).toBe(true);
    expect(prompts.some((prompt) => prompt.includes("create_new_doc_request"))).toBe(true);
    expect(prompts.some((prompt) => prompt.includes("Do NOT use web search"))).toBe(true);

    const contractState = JSON.parse(await readFile(path.join(outputRoot, "contract_state.json"), "utf8"));
    expect(contractState.contracts[fixtureContractKeys.flat].status).toBe("manual_correction_needed");
    expect(contractState.contracts[fixtureContractKeys.tiled]).toBeUndefined();

    const state = JSON.parse(await readFile(path.join(outputRoot, "pipeline_state.json"), "utf8"));
    expect(state.modules.unit_module).toBe("pass");

    const lifecycle = JSON.parse(await readFile(path.join(knowledgeRoot, "doc_lifecycle.json"), "utf8"));
    const createdDocs = Object.values(lifecycle.docs) as Array<{
      contract_id?: string;
      creation_reason?: string;
      source_doc_ids?: string[];
      pattern_path: string;
    }>;
    const createdTiledDoc = createdDocs.find((doc) => doc.contract_id === "tiled-streaming");
    expect(createdTiledDoc).toBeDefined();
    expect(createdTiledDoc?.creation_reason).toBe("create_new_doc");
    expect(createdTiledDoc?.source_doc_ids?.some((id) => id.includes("protected"))).toBe(true);
    await expect(readFile(path.join(repoRoot, createdTiledDoc!.pattern_path), "utf8")).resolves.toContain("contract_id: tiled-streaming");

    const log = await readFile(path.join(reportsDir, "run_log.jsonl"), "utf8");
    expect(log).toContain('"event":"contract_alternative_selected"');
    expect(log).toContain('"event":"create_new_doc_requested"');
  });

  it("skips persisted flagged contracts on subsequent self-improve runs", async () => {
    await writePipelineIrFixture();
    await seedContractFlags(["flat"]);
    const module = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verilog_module.json"), "utf8"),
    );
    const verifPass = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verif_pass.json"), "utf8"),
    );

    const queryFn = createQueryMock({
      foundry: [async () => {
        await persistMockRtlDeliverable(module);
        return successRtlWithDoc(module, "tiled-session");
      }],
    });
    const assayerFn = createAssayerMock([verifPass]);
    const synthesisFn = createVivadoMock([{ success: true, lut_count: 1, fmax_mhz: 75, report: "fixture" }]);

    await runPipeline("checkpoint.pth", {
      selfImprove: true,
      runtime: createOrchestratorRuntime({ now: fixedNow, queryFn, synthesisFn, assayerFn }),
    });

    const firstFoundryPrompt = queryFn.mock.calls
      .map(([call]) => (call as { prompt: string }).prompt)
      .find((prompt) => prompt.includes("You are the `foundry`")) ?? "";
    expect(firstFoundryPrompt).toContain("contract variant: tiled-streaming");

    const state = JSON.parse(await readFile(path.join(outputRoot, "pipeline_state.json"), "utf8"));
    expect(state.modules.unit_module).toBe("pass");
    const log = await readFile(path.join(reportsDir, "run_log.jsonl"), "utf8");
    expect(log).toContain('"event":"contract_selected_after_skipping_flagged"');
  });

  it("reuses contract-tagged lifecycle docs instead of creating a duplicate new-doc request", async () => {
    // 1x1 fixture so `seedContractFlags(["flat"])` matches the canonical
    // contract_state key for unit_module. After the flat-bus flag fires the
    // orchestrator switches to tiled-streaming, where the seeded
    // `auto_tiled_existing` active doc covers the (op_type, contract_id)
    // tuple — the doc-coverage guard suppresses the wrapper schema, so the
    // mock returns plain `{VerilogModule}`.
    await writePipelineIrFixture();
    await seedContractFlags(["flat"]);
    await seedLifecycleDoc({
      id: "auto_tiled_existing",
      status: "active",
      contract_id: "tiled-streaming",
    });
    const module = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verilog_module.json"), "utf8"),
    );
    const verifPass = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verif_pass.json"), "utf8"),
    );

    const queryFn = createQueryMock({
      // Plain `{VerilogModule}` — coverage exists for tiled-streaming via
      // `auto_tiled_existing`, so the guard suppresses the wrapper schema.
      foundry: [async () => {
        await persistMockRtlDeliverable(module);
        return successResult(module, "tiled-session");
      }],
    });
    const assayerFn = createAssayerMock([verifPass]);
    const synthesisFn = createVivadoMock([{ success: true, lut_count: 1, fmax_mhz: 75, report: "fixture" }]);

    await runPipeline("checkpoint.pth", {
      selfImprove: true,
      runtime: createOrchestratorRuntime({ now: fixedNow, queryFn, synthesisFn, assayerFn }),
    });

    const prompts = queryFn.mock.calls.map(([call]) => (call as { prompt: string }).prompt);
    expect(prompts.some((prompt) => prompt.includes("contract variant: tiled-streaming"))).toBe(true);
    expect(prompts.some((prompt) => prompt.includes("create_new_doc_request"))).toBe(false);

    const log = await readFile(path.join(reportsDir, "run_log.jsonl"), "utf8");
    expect(log).not.toContain('"event":"create_new_doc_requested"');
    expect(log).toContain('"event":"self_improve_doc_request_skipped"');
  });

  it("requests a new doc when flat-bus has same-family docs but no exact pattern", async () => {
    const pipelineIr = await writePipelineIrFixture() as { layers: Array<Record<string, unknown>> };
    pipelineIr.layers[0].weight_shape = [1, 1, 5, 5];
    pipelineIr.layers[0].num_weights = 25;
    await writeFile(path.join(outputRoot, "layer_ir.json"), `${JSON.stringify(pipelineIr, null, 2)}\n`, "utf8");
    const module = {
      ...JSON.parse(await readFile(path.join(repoRoot, "test", "fixtures", "verilog_module.json"), "utf8")),
      spec_hash: "conv2d_1x1x5x5_s1x1_i8_o8",
    };
    const verifPass = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verif_pass.json"), "utf8"),
    );

    const queryFn = createQueryMock({
      foundry: [async () => {
        await persistMockRtlDeliverable(module);
        return successRtlWithDoc(module, "flat-session");
      }],
    });
    const assayerFn = createAssayerMock([verifPass]);
    const synthesisFn = createVivadoMock([{ success: true, lut_count: 1, fmax_mhz: 75, report: "fixture" }]);

    await runPipeline("checkpoint.pth", {
      selfImprove: true,
      runtime: createOrchestratorRuntime({ now: fixedNow, queryFn, synthesisFn, assayerFn }),
    });

    const prompts = queryFn.mock.calls.map(([call]) => (call as { prompt: string }).prompt);
    expect(prompts.some((prompt) => prompt.includes("create_new_doc_request"))).toBe(true);
    expect(prompts.some((prompt) => prompt.includes("protected_02_conv1x1_md"))).toBe(true);

    const lifecycle = JSON.parse(await readFile(path.join(knowledgeRoot, "doc_lifecycle.json"), "utf8"));
    const createdDocs = Object.values(lifecycle.docs) as Array<{
      contract_id?: string;
      creation_reason?: string;
      source_doc_ids?: string[];
    }>;
    const createdFlatDoc = createdDocs.find((doc) => doc.contract_id === "flat-bus");
    expect(createdFlatDoc?.creation_reason).toBe("create_new_doc");
    expect(createdFlatDoc?.source_doc_ids?.some((id) => id.includes("protected"))).toBe(true);
  });

  it("escalates when all available contracts are exhausted", async () => {
    await writePipelineIrFixture();
    await seedContractFlags(["flat", "tiled"]);
    const module = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verilog_module.json"), "utf8"),
    );
    const verifFail = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verif_fail.json"), "utf8"),
    );

    const foundryStep = async () => {
      await persistMockRtlDeliverable(module);
      return successRtlWithDoc(module, "dram-session");
    };
    const queryFn = createQueryMock({
      // After clearGeneratedRtlArtifacts started running before each fresh
      // Foundry attempt, the orchestrator can no longer paper over a missing
      // mock by recovering stale RTL from disk. With dram-backed flagged
      // here, the walk reaches activation-double-buffering and weight-tiling,
      // each of which dispatches a fresh Foundry attempt before being
      // flagged in turn. Queue enough mocks for: dram-backed attempt 1,
      // dram-backed final retry after retrospector, then one each for the
      // two remaining contracts.
      foundry: [foundryStep, foundryStep, foundryStep, foundryStep],
      retrospector: [
        successResult({
          analysis: "The final contract variant still fails.",
          suggestion: "Escalate with the full failure history for manual correction.",
        }),
      ],
    });
    // Defensive queue depth: one verifFail per Foundry call.
    const assayerFn = createAssayerMock([verifFail, verifFail, verifFail, verifFail]);
    const synthesisFn = createVivadoMock([]);

    await runPipeline("checkpoint.pth", {
      maxRetries: 0,
      selfImprove: true,
      runtime: createOrchestratorRuntime({ now: fixedNow, queryFn, synthesisFn, assayerFn }),
    });

    const state = JSON.parse(await readFile(path.join(outputRoot, "pipeline_state.json"), "utf8"));
    expect(state.modules.unit_module).toBe("fail_abort");
    expect(state.results.unit_module.failure_class).toBe("manual_correction_needed");

    const contractState = JSON.parse(await readFile(path.join(outputRoot, "contract_state.json"), "utf8"));
    expect(contractState.contracts[fixtureContractKeys.dram].status).toBe("manual_correction_needed");
    const reportPath = path.join(repoRoot, contractState.contracts[fixtureContractKeys.dram].report_path);
    await expect(readFile(reportPath, "utf8")).resolves.toContain("manual_correction_needed");

    const log = await readFile(path.join(reportsDir, "run_log.jsonl"), "utf8");
    expect(log).toContain('"event":"human_escalation_required"');
  });

  it("writes successful self-improve draft docs to probationary", async () => {
    // 5x5 conv has no protected/active/probationary doc coverage in the
    // seeded knowledge tree, so the doc-coverage guard does not suppress
    // self_improve_doc_request — Foundry is asked for a draft_doc and the
    // probationary doc lifecycle path runs end-to-end.
    await writeUncoveredPipelineIrFixture();
    const module = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verilog_module.json"), "utf8"),
    );
    const verifPass = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verif_pass.json"), "utf8"),
    );

    const queryFn = createQueryMock({
      foundry: [async () => {
        await persistMockRtlDeliverable(module);
        return successRtlWithDoc(module);
      }],
    });
    const assayerFn = createAssayerMock([verifPass]);
    const synthesisFn = createVivadoMock([{ success: true, lut_count: 1, fmax_mhz: 75, report: "fixture" }]);

    await runPipeline("checkpoint.pth", {
      selfImprove: true,
      runtime: createOrchestratorRuntime({ now: fixedNow, queryFn, synthesisFn, assayerFn }),
    });

    const lifecycle = JSON.parse(await readFile(path.join(knowledgeRoot, "doc_lifecycle.json"), "utf8"));
    const docs = Object.values(lifecycle.docs) as Array<{
      status: string;
      created_by_module: string;
      pattern_path: string;
      reference_path: string;
    }>;
    const created = docs.find((doc) => doc.created_by_module === "unit_module");
    expect(created).toBeDefined();
    expect(created?.status).toBe("probationary");
    expect(created?.pattern_path).toContain("knowledge/patterns/probationary/");
    expect(created?.reference_path).toContain("knowledge/references/probationary/");
    await expect(readFile(path.join(repoRoot, created!.pattern_path), "utf8")).resolves.toContain("Generated RTL note");
    await expect(readFile(path.join(repoRoot, created!.reference_path), "utf8")).resolves.toContain("reference_generated");
  });

  it("promotes probationary docs after enough successful users", async () => {
    // Use uncovered geometry so the guard does not suppress the wrapper
    // schema; the seeded probationary doc is what the test verifies gets
    // promoted to active after this success.
    await writeUncoveredPipelineIrFixture();
    await seedLifecycleDoc({
      id: "auto_test_promote",
      status: "probationary",
      used_by_modules: ["unit_module"],
      successful_modules: ["prior_a", "prior_b"],
    });
    const module = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verilog_module.json"), "utf8"),
    );
    const verifPass = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verif_pass.json"), "utf8"),
    );

    const queryFn = createQueryMock({
      foundry: [async () => {
        await persistMockRtlDeliverable(module);
        return successRtlWithDoc(module);
      }],
    });
    const assayerFn = createAssayerMock([verifPass]);
    const synthesisFn = createVivadoMock([{ success: true, lut_count: 1, fmax_mhz: 75, report: "fixture" }]);

    await runPipeline("checkpoint.pth", {
      selfImprove: true,
      runtime: createOrchestratorRuntime({ now: fixedNow, queryFn, synthesisFn, assayerFn }),
    });

    const lifecycle = JSON.parse(await readFile(path.join(knowledgeRoot, "doc_lifecycle.json"), "utf8"));
    expect(lifecycle.docs.auto_test_promote.status).toBe("active");
    expect(lifecycle.docs.auto_test_promote.successful_modules).toContain("unit_module");
    expect(lifecycle.docs.auto_test_promote.pattern_path).toContain("knowledge/patterns/active/");
    await expect(readFile(path.join(repoRoot, "knowledge/patterns/active/auto_test_promote.md"), "utf8")).resolves.toContain("seed pattern");
  });

  it("archives probationary docs immediately when a module using them fails", async () => {
    // Use uncovered geometry so the seeded probationary doc is genuinely
    // the only matching coverage — when the failing module triggers
    // archival, the guard's coverage check correctly transitions empty.
    await writeUncoveredPipelineIrFixture();
    await seedLifecycleDoc({
      id: "auto_test_archive",
      status: "probationary",
      used_by_modules: ["unit_module"],
    });
    const module = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verilog_module.json"), "utf8"),
    );
    const verifFail = {
      module_id: "unit_module",
      status: "fail",
      timing_pass: false,
      timing_actual_cycles: 0,
      timing_expected_cycles: 1,
      failure_category: "architectural_fit",
      classifier_reason: "Seeded failure for lifecycle test.",
    };

    const foundryStep = async () => {
      await persistMockRtlDeliverable(module);
      return successRtlWithDoc(module);
    };
    // After clearGeneratedRtlArtifacts started running before each fresh
    // Foundry attempt, missing mocks no longer get rescued by stale RTL on
    // disk. With no contracts pre-flagged here the walk visits every
    // contract in CONTRACT_PLANS — queue enough mocks to satisfy each
    // (the retrospector throws on the first walk because no mock is
    // queued, which short-circuits the same-contract final retry).
    const queryFn = createQueryMock({
      foundry: [foundryStep, foundryStep, foundryStep, foundryStep, foundryStep],
    });
    const assayerFn = createAssayerMock([verifFail, verifFail, verifFail, verifFail, verifFail]);
    const synthesisFn = createVivadoMock([]);

    await runPipeline("checkpoint.pth", {
      selfImprove: true,
      runtime: createOrchestratorRuntime({ now: fixedNow, queryFn, synthesisFn, assayerFn }),
    });

    const lifecycle = JSON.parse(await readFile(path.join(knowledgeRoot, "doc_lifecycle.json"), "utf8"));
    expect(lifecycle.docs.auto_test_archive.status).toBe("archived");
    expect(lifecycle.docs.auto_test_archive.failed_modules).toContain("unit_module");
    expect(lifecycle.docs.auto_test_archive.archived_pattern_path).toContain("knowledge/patterns/archive/");
    await expect(readFile(path.join(repoRoot, lifecycle.docs.auto_test_archive.archived_pattern_path), "utf8")).resolves.toContain("seed pattern");
    await expect(readFile(path.join(repoRoot, "knowledge/patterns/probationary/auto_test_archive.md"), "utf8")).rejects.toThrow();
  });

  it("fail-aborts a layer whose bus width exceeds MAX_SUPPORTED_BUS_BITS without invoking Foundry or Surgeon", async () => {
    // Build a custom pipeline IR whose sole layer has a bus width way over
    // the 4096-bit cap. Everything else mirrors the standard fixture so the
    // pipeline otherwise boots cleanly.
    const pipelineIr = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "pipeline_ir.json"), "utf8"),
    );
    const layer = pipelineIr.layers[0];
    // 1024 channels * 8 = 8192 bits input. 16384 bits output. Both over 4096.
    layer.input_shape = [1, 1024, 1, 1];
    layer.output_shape = [1, 2048, 1, 1];
    layer.input_width_bits = 8192;
    layer.output_width_bits = 16384;
    layer.weight_shape = [2048, 1024, 1, 1];
    layer.num_weights = 2048 * 1024;
    await writeFile(path.join(outputRoot, "layer_ir.json"), JSON.stringify(pipelineIr, null, 2), "utf8");
    await writeFile(
      path.join(outputRoot, "layer_ir.json.checkpoint"),
      `${path.resolve("checkpoint.pth")}\n`,
      "utf8",
    );

    const queryFn = createQueryMock({});
    const assayerFn = createAssayerMock([]);
    const synthesisFn = createVivadoMock([]);

    await runPipeline("checkpoint.pth", {
      // Pin self_improve off so the contract walker doesn't try alternate
      // contracts before the bus-width capability gate fires — this test
      // asserts the gate produces fail_abort with zero LLM calls.
      selfImprove: false,
      runtime: createOrchestratorRuntime({ now: fixedNow, queryFn, synthesisFn, assayerFn }),
    });

    const prompts = queryFn.mock.calls.map(([call]) => (call as { prompt: string }).prompt);
    expect(prompts.some((prompt) => prompt.includes("You are the `foundry`"))).toBe(false);
    expect(prompts.some((prompt) => prompt.includes("You are the `surgeon`"))).toBe(false);
    // SYSTEM_REVIEW_FINDINGS #2: bus-width capability gate now classifies
    // architectural_unsupported deterministically — no LLM classifier call,
    // no money spent on a verdict the orchestrator already knows.
    expect(prompts.some((prompt) => prompt.includes("You are the `failure_classifier`"))).toBe(false);
    expect(synthesisFn).not.toHaveBeenCalled();
    expect(assayerFn).not.toHaveBeenCalled();

    const state = JSON.parse(await readFile(path.join(outputRoot, "pipeline_state.json"), "utf8"));
    expect(state.modules[layer.module_id]).toBe("fail_abort");
    expect(state.results[layer.module_id].failure_class).toBe("architectural_unsupported");
    expect(state.results[layer.module_id].failure_category).toBe("architectural_fit");
    // Per SYSTEM_REVIEW_FINDINGS #2 the deterministic classifier writes the
    // generalized constraint name (the bus-width gate is one of several
    // architectural-fit triggers); MAX_SUPPORTED_BUS_BITS would be too
    // narrow to reuse for the contract-specific cap variants.
    expect(state.results[layer.module_id].violated_constraint).toBe("architectural_unsupported");

    const log = await readFile(path.join(reportsDir, "run_log.jsonl"), "utf8");
    expect(log).toContain('"reason":"architectural_unsupported"');
  });

  it("aborts on tb_setup_error without invoking Surgeon", async () => {
    await writePipelineIrFixture();
    const originalModule = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verilog_module.json"), "utf8"),
    );
    const tbSetupFail = {
      module_id: "unit_module",
      status: "fail",
      status_class: "tb_setup_error",
      timing_pass: false,
      timing_actual_cycles: -1,
      timing_expected_cycles: 1,
      expected: [],
      got: [],
      failure_class: null,
      verilator_stderr: "static_verilator_tb.cpp:91: error: bad bus width",
      fix_hint: "Static testbench did not produce results JSON.",
    };

    const queryFn = createQueryMock({
      foundry: [async () => {
        await persistMockRtlDeliverable(originalModule);
        return successResult(originalModule);
      }],
    });
    const assayerFn = createAssayerMock([tbSetupFail]);
    const synthesisFn = createVivadoMock([]);

    await runPipeline("checkpoint.pth", {
      runtime: createOrchestratorRuntime({ now: fixedNow, queryFn, synthesisFn, assayerFn }),
    });

    const prompts = queryFn.mock.calls.map(([call]) => (call as { prompt: string }).prompt);
    expect(prompts.some((prompt) => prompt.includes("You are the `surgeon`"))).toBe(false);
    expect(synthesisFn).not.toHaveBeenCalled();

    const state = JSON.parse(await readFile(path.join(outputRoot, "pipeline_state.json"), "utf8"));
    expect(state.modules.unit_module).toBe("fail_abort");
    expect(state.results.unit_module.status_class).toBe("tb_setup_error");
  });

  it("aborts when deterministic assayer infrastructure crashes", async () => {
    await writePipelineIrFixture();
    const originalModule = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verilog_module.json"), "utf8"),
    );

    const queryFn = createQueryMock({
      foundry: [async () => {
        await persistMockRtlDeliverable(originalModule);
        return successResult(originalModule);
      }],
    });
    const assayerFn = createAssayerMock([
      async () => {
        throw new Error("iverilog exited non-zero without diagnostic output");
      },
    ]);
    const synthesisFn = createVivadoMock([]);

    await runPipeline("checkpoint.pth", {
      runtime: createOrchestratorRuntime({ now: fixedNow, queryFn, synthesisFn, assayerFn }),
    });

    const prompts = queryFn.mock.calls.map(([call]) => (call as { prompt: string }).prompt);
    expect(prompts.some((prompt) => prompt.includes("You are the `surgeon`"))).toBe(false);
    expect(synthesisFn).not.toHaveBeenCalled();

    const state = JSON.parse(await readFile(path.join(outputRoot, "pipeline_state.json"), "utf8"));
    expect(state.modules.unit_module).toBe("fail_abort");
    expect(state.results.unit_module.status_class).toBe("tb_setup_error");
    expect(state.results.unit_module.fix_hint).toContain("Assayer runner crashed");
  });

  it("routes verilator_timeout VerifResult through Surgeon the same as other sim failures", async () => {
    await writePipelineIrFixture();
    const originalModule = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verilog_module.json"), "utf8"),
    );
    const repairedModule = { ...originalModule, generated_by: "Surgeon", attempt: 2 };
    const verifPass = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verif_pass.json"), "utf8"),
    );
    // A Verilator-side timeout: the binary was killed after the wall-clock cap.
    const timeoutVerif = {
      module_id: "unit_module",
      status: "fail",
      status_class: "sim_stalled",
      timing_pass: false,
      timing_actual_cycles: -1,
      timing_expected_cycles: 1,
      expected: [],
      got: [],
      failure_class: "verilator_timeout",
      verilator_stderr: "child process killed by SIGTERM after timeout",
      fix_hint: "Verilator simulation exceeded the 600s wall-clock cap.",
    };

    const queryFn = createQueryMock({
      foundry: [async () => {
        await persistMockRtlDeliverable(originalModule);
        return successResult(originalModule);
      }],
      surgeon: [async () => {
        await persistMockRtlDeliverable(repairedModule);
        return successResult(repairedModule);
      }],
    });
    // Foundry times out; Surgeon then succeeds.
    const assayerFn = createAssayerMock([timeoutVerif, verifPass]);
    const synthesisFn = createVivadoMock([{ success: true, lut_count: 3, fmax_mhz: 75, report: "fixture" }]);

    await runPipeline("checkpoint.pth", {
      runtime: createOrchestratorRuntime({ now: fixedNow, queryFn, synthesisFn, assayerFn }),
    });

    // Surgeon was invoked (not fail_abort like tb_setup_error).
    const prompts = queryFn.mock.calls.map(([call]) => (call as { prompt: string }).prompt);
    expect(prompts.some((prompt) => prompt.includes("You are the `surgeon`"))).toBe(true);
    // Surgeon prompt carries the verilator_timeout rubric.
    const surgeonPrompt = prompts.find((p) => p.includes("You are the `surgeon`")) ?? "";
    expect(surgeonPrompt).toContain("VERILATOR TIMEOUT");

    const state = JSON.parse(await readFile(path.join(outputRoot, "pipeline_state.json"), "utf8"));
    expect(state.modules.unit_module).toBe("pass");
  });

  it("runs the Surgeon repair path after a failed vivado synthesis report", async () => {
    await writePipelineIrFixture();
    const originalModule = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verilog_module.json"), "utf8"),
    );
    const repairedModule = {
      ...originalModule,
      generated_by: "Surgeon",
      attempt: 2,
    };
    const verifPass = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verif_pass.json"), "utf8"),
    );

    const queryFn = createQueryMock({
      foundry: [async () => {
        await persistMockRtlDeliverable(originalModule);
        return successResult(originalModule);
      }],
      surgeon: [async () => {
        await persistMockRtlDeliverable(repairedModule);
        return successResult(repairedModule);
      }],
    });
    // Foundry passes sim but fails Vivado, then Surgeon succeeds.
    const assayerFn = createAssayerMock([verifPass, verifPass]);
    const synthesisFn = createVivadoMock([
      { success: false, lut_count: 0, fmax_mhz: 0, report: "synth failed" },
      { success: true, lut_count: 3, fmax_mhz: 75, report: "fixture" },
    ]);

    await runPipeline("checkpoint.pth", {
      runtime: createOrchestratorRuntime({ now: fixedNow, queryFn, synthesisFn, assayerFn }),
    });

    const prompts = queryFn.mock.calls.map(([call]) => (call as { prompt: string }).prompt);
    expect(synthesisFn).toHaveBeenCalledTimes(2);
    expect(prompts.some((prompt) => prompt.includes("You are the `surgeon`"))).toBe(true);

    const state = JSON.parse(await readFile(path.join(outputRoot, "pipeline_state.json"), "utf8"));
    expect(state.modules.unit_module).toBe("pass");
    expect(state.attempts.unit_module).toBe(2);
    expect(await readFile(path.join(reportsDir, "run_log.jsonl"), "utf8")).toContain('"reason":"vivado_synthesis_failed"');
  });

  it("routes missing Vivado timing measurement to Surgeon as synthesis_failed", async () => {
    await writePipelineIrFixture();
    const originalModule = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verilog_module.json"), "utf8"),
    );
    const repairedModule = { ...originalModule, generated_by: "Surgeon", attempt: 2 };
    const verifPass = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verif_pass.json"), "utf8"),
    );

    const queryFn = createQueryMock({
      foundry: [async () => {
        await persistMockRtlDeliverable(originalModule);
        return successResult(originalModule);
      }],
      surgeon: [async () => {
        await persistMockRtlDeliverable(repairedModule);
        return successResult(repairedModule);
      }],
    });
    // Foundry misses timing, then Surgeon succeeds.
    const assayerFn = createAssayerMock([verifPass, verifPass]);
    const synthesisFn = createVivadoMock([
      {
        success: true,
        lut_count: 1479,
        fmax_mhz: 0,
        wns_ns: null,
        timing_met: false,
        report: "| Slice LUTs* | 1479 |",
      },
      {
        success: true,
        lut_count: 1479,
        fmax_mhz: 75,
        wns_ns: 1,
        timing_met: true,
        report: "WNS(ns): 1.000\n| Slice LUTs* | 1479 |",
      },
    ]);

    await runPipeline("checkpoint.pth", {
      runtime: createOrchestratorRuntime({ now: fixedNow, queryFn, synthesisFn, assayerFn }),
    });

    const prompts = queryFn.mock.calls.map(([call]) => (call as { prompt: string }).prompt);
    expect(synthesisFn).toHaveBeenCalledTimes(2);
    expect(prompts.some((prompt) => prompt.includes("You are the `surgeon`"))).toBe(true);
    const state = JSON.parse(await readFile(path.join(outputRoot, "pipeline_state.json"), "utf8"));
    expect(state.modules.unit_module).toBe("pass");
    expect(await readFile(path.join(reportsDir, "run_log.jsonl"), "utf8")).toContain('"reason":"vivado_synthesis_failed"');
  });

  it("routes Fmax-below-target to Surgeon with missing_pipeline_register failure class", async () => {
    await writePipelineIrFixture();
    const originalModule = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verilog_module.json"), "utf8"),
    );
    const repairedModule = { ...originalModule, generated_by: "Surgeon", attempt: 2 };
    const verifPass = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verif_pass.json"), "utf8"),
    );

    const queryFn = createQueryMock({
      foundry: [async () => {
        await persistMockRtlDeliverable(originalModule);
        return successResult(originalModule);
      }],
      surgeon: [async () => {
        await persistMockRtlDeliverable(repairedModule);
        return successResult(repairedModule);
      }],
    });
    const assayerFn = createAssayerMock([verifPass, verifPass]);
    const synthesisFn = createVivadoMock([
      { success: true, lut_count: 4, fmax_mhz: 35, timing_met: false, wns_ns: -8.571, report: "WNS(ns): -8.571" },
      { success: true, lut_count: 4, fmax_mhz: 75, timing_met: true, wns_ns: 6.667, report: "WNS(ns): 6.667" },
    ]);

    await runPipeline("checkpoint.pth", {
      runtime: createOrchestratorRuntime({ now: fixedNow, queryFn, synthesisFn, assayerFn }),
    });

    expect(synthesisFn).toHaveBeenCalledTimes(2);
    const state = JSON.parse(await readFile(path.join(outputRoot, "pipeline_state.json"), "utf8"));
    expect(state.modules.unit_module).toBe("pass");
    expect(await readFile(path.join(reportsDir, "run_log.jsonl"), "utf8")).toContain('"reason":"vivado_missing_pipeline_register"');
  });

  it("blocks resume when fail_retry has no prior verification result", async () => {
    await writePipelineIrFixture();
    await writeFile(
      path.join(outputRoot, "pipeline_state.json"),
      JSON.stringify({
        run_id: "run-1",
        started_at: "2026-04-14T00:00:00Z",
        modules: { unit_module: "fail_retry" },
        attempts: { unit_module: 1 },
        results: {},
        max_retries: 3,
        total_cost_usd: 0,
        model_usage: {},
      }),
      "utf8",
    );

    const queryFn = createQueryMock({});
    await expect(
      runPipeline("checkpoint.pth", {
        resume: true,
        runtime: createOrchestratorRuntime({ now: fixedNow, queryFn }),
      }),
    ).rejects.toThrow("fail_retry");
  });

  it("reloads resume state before completing", async () => {
    await writePipelineIrFixture();
    await writeFile(
      path.join(outputRoot, "pipeline_state.json"),
      JSON.stringify({
        run_id: "run-1",
        started_at: "2026-04-14T00:00:00Z",
        modules: { unit_module: "pass" },
        attempts: { unit_module: 0 },
        results: { unit_module: { module_id: "unit_module", status: "pass" } },
        max_retries: 3,
        total_cost_usd: 0,
        model_usage: {},
      }),
      "utf8",
    );

    const queryFn = createQueryMock({});
    await runPipeline("checkpoint.pth", {
      resume: true,
      runtime: createOrchestratorRuntime({ now: fixedNow, queryFn }),
    });

    expect(queryFn).not.toHaveBeenCalled();
    expect(await readFile(path.join(reportsDir, "run_log.jsonl"), "utf8")).toContain(
      '"event":"pipeline_resume_loaded"',
    );
  });
});
