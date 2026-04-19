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
//   - Cartographer: runs once per pipeline to produce layer_ir.json from a
//     PyTorch checkpoint (it is bypassed entirely on the ONNX path). Pure
//     extraction + formatting, no complex reasoning. Sonnet 4.6 is cheaper
//     and plenty. Running Opus here is waste.
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
  Cartographer: { model: "claude-sonnet-4-6" as const, maxTurns: 30, description: "Model extractor. Runs once at pipeline start. Emits output/layer_ir.json." },
  Foundry:      { model: "claude-opus-4-7"  as const, maxTurns: 20, description: "Verilog codegen. Receives one LayerIR, produces one VerilogModule." },
  Surgeon:      { model: "claude-opus-4-7"  as const, maxTurns: 20, description: "Targeted repair. Receives broken Verilog + VerifResult + LayerIR. Classifies the failure and performs minimal rewrite." },
} as const;

export type AgentName = keyof typeof AGENT_CONFIG;

export const PIPELINE_CONFIG = {
  max_retries: 3,
  // Cap on the number of parallel MAC lanes Foundry instantiates per conv
  // layer. Per-layer mac_parallelism = min(OC, MAX_PARALLEL_MACS). The FSM
  // iterates OC in groups of mac_parallelism — this keeps the combinational
  // cone small enough for Sky130 / ABC to map inside YOSYS_TIMEOUT_MS. 8 is
  // the current sweet spot: 8×8-bit INT8 multipliers feeding a 3-level adder
  // tree maps in seconds on Sky130. Raising it trades synth time for
  // throughput; dropping it trades throughput for synth time. Python
  // frontends must read this same value when computing mac_parallelism and
  // pipeline_latency_cycles.
  MAX_PARALLEL_MACS: 8,
  output_dir: "../output",
  rtl_dir: "../output/rtl",
  tb_dir: "../output/tb",
  weights_dir: "../output/weights",
  reports_dir: "../output/reports",
  golden_vectors_path: "../output/golden_vectors.json",
  layer_ir_path: "../output/layer_ir.json",
  pipeline_state_path: "../output/pipeline_state.json",
  static_testbench_path: "../tb/static_verilator_tb.cpp",
} as const;
