// Docs: https://platform.claude.com/docs/en/agent-sdk/typescript
//
// Model selection is INTENTIONAL, not tier-based. We pass full model IDs so
// the pick is reproducible regardless of the user's global ~/.claude/settings
// default model. Tier strings ("sonnet" / "opus") resolve in ways that
// depend on both the installed SDK version and the user's global default —
// when we used to say `model: "sonnet"` we were actually getting whatever
// the global settings pinned (most recently Opus 4.6[1m]), which was
// undetected for weeks of runs.
//
// Why each pick:
//   - Cartographer: legacy fallback agent for PipelineIR formatting. The
//     production orchestrator now calls deterministic read_weights directly
//     so extraction does not depend on an LLM rewriting tool output.
//   - Foundry: one-shot Verilog codegen from a 25 KB spec with correctness
//     requirements (line buffers, padding drain, sign extension, scale-
//     factor derivation). Opus 4.7 is the current coding-best model
//     (released 2026-04-16) and first-shot quality is what matters here —
//     a failed Foundry output costs a Surgeon pass which is strictly more
//     expensive than the Opus differential vs Sonnet.
//   - Surgeon: targeted repair with rich diagnostic signal, doing minimal
//     rewrites. Opus 4.7 also — repair is the highest-stakes call in the
//     pipeline (a regression here corrupts the on-disk module for the next
//     iteration).
//
// `maxTurns` caps the agentic turn count per subagent call; the outer
// query() also sets a parent cap that applies on top of these.
export const AGENT_CONFIG = {
  Cartographer: { model: "claude-sonnet-4-6" as const, maxTurns: 30, description: "Legacy PipelineIR formatter fallback; production extraction calls read_weights directly." },
  Foundry:      { model: "claude-opus-4-7"  as const, maxTurns: 20, description: "Verilog codegen. Receives one LayerIR, produces one VerilogModule." },
  Surgeon:      { model: "claude-opus-4-7"  as const, maxTurns: 20, description: "Targeted repair. Receives broken Verilog + VerifResult + LayerIR. Classifies the failure and performs minimal rewrite." },
} as const;

export type AgentName = keyof typeof AGENT_CONFIG;

export const FAILURE_CLASSIFIER_CONFIG = {
  model: "claude-sonnet-4-6" as const,
  maxTurns: 4,
  description: "Classifies failed module evidence as code_bug, architectural_fit, toolchain_infra, verification_env, or unknown.",
} as const;

export const RETROSPECTOR_CONFIG = {
  model: "claude-opus-4-7" as const,
  maxTurns: 4,
  description: "Advises Foundry after the normal retry budget is exhausted.",
} as const;

export function parseBooleanEnv(
  env: NodeJS.ProcessEnv,
  name: string,
  fallback: boolean,
): boolean {
  const raw = env[name];
  if (raw === undefined || raw.trim() === "") {
    return fallback;
  }
  const normalized = raw.trim().toLowerCase();
  if (["1", "true", "yes", "on"].includes(normalized)) {
    return true;
  }
  if (["0", "false", "no", "off"].includes(normalized)) {
    return false;
  }
  return fallback;
}

export function parsePositiveIntEnv(
  env: NodeJS.ProcessEnv,
  name: string,
  fallback: number,
): number {
  const raw = env[name];
  if (raw === undefined || raw.trim() === "") {
    return fallback;
  }
  const parsed = Number.parseInt(raw.trim(), 10);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

export const PIPELINE_CONFIG = {
  // Foundation switch for the self-improving documentation flow. Default OFF
  // keeps today's pipeline behavior unchanged; later phases should gate any
  // doc-writing / promotion logic on this flag. The knowledge reader still
  // loads protected + active + probationary docs in both modes so validated
  // generated guidance remains available even when new self-improvement writes
  // are disabled.
  self_improve: parseBooleanEnv(process.env, "NN2RTL_SELF_IMPROVE", false),
  doc_promotion_success_threshold: parsePositiveIntEnv(
    process.env,
    "NN2RTL_DOC_PROMOTION_SUCCESSES",
    3,
  ),
  max_retries: 3,
  // Cap on the number of accumulator lanes in each conv output-channel
  // group. Per-layer mac_parallelism = min(OC, MAX_PARALLEL_MACS). The
  // current verified FSM still issues one lane's weight read / MAC per
  // cycle; MP controls OC grouping and amortizes bias/scale/output overhead,
  // but it is not yet MP cycle-parallel throughput. 4 is the current
  // frontend value and keeps the serialized weight-memory structure small
  // enough for ZCU102/Vivado while leaving a clean migration path to future
  // banked BRAM datapaths. Python
  // frontends must read this same value when computing mac_parallelism and
  // pipeline_latency_cycles.
  MAX_PARALLEL_MACS: 4,
  // Flat-bus capability ceiling. Layers above this should be manually tagged
  // with a heavier contract (`tiled-streaming`, `dram-backed-weights`,
  // `activation-double-buffering`, or `weight-tiling`) so the orchestrator
  // selects the matching contract infrastructure instead of spending retries
  // on an impossible full-width bus.
  MAX_SUPPORTED_BUS_BITS: 4096,
  output_dir: "../output",
  rtl_dir: "../output/rtl",
  tb_dir: "../output/tb",
  weights_dir: "../output/weights",
  reports_dir: "../output/reports",
  golden_vectors_path: "../output/golden_vectors.json",
  layer_ir_path: "../output/layer_ir.json",
  pipeline_state_path: "../output/pipeline_state.json",
  contract_state_path: "../output/contract_state.json",
  static_testbench_path: "../tb/static_verilator_tb.cpp",
} as const;
