// Docs: https://platform.claude.com/docs/en/agent-sdk/typescript
// AgentDefinition model field accepts: "sonnet" | "opus" | "haiku" | "inherit"
// effort is only supported in .claude/agents/ file-based agents, not in AgentDefinition
// For SDK programmatic agents, model tier is the only override available

// The deterministic TypeScript orchestrator plays both the Conductor and
// Assayer roles itself. Assayer used to be a Haiku "simulation runner" LLM
// but had zero real reasoning to do (run iverilog, run Verilator, parse
// JSON, return) and repeatedly hallucinated VerifResults instead of calling
// the tools. Verification now goes through runtime.assayerFn — same pattern
// as runtime.yosysFn for synthesis. The three remaining LLM agents below
// are the only ones dispatched via the SDK's query() path.
// `maxTurns` caps the agentic turn count per subagent call; the outer
// query() also sets a parent cap that applies on top of these.
export const AGENT_CONFIG = {
  Cartographer: { model: "sonnet" as const, maxTurns: 30, description: "Model extractor. Runs once at pipeline start. Emits output/layer_ir.json." },
  Foundry:      { model: "sonnet" as const, maxTurns: 20, description: "Verilog codegen. Receives one LayerIR, produces one VerilogModule." },
  Surgeon:      { model: "opus"   as const, maxTurns: 8, description: "Targeted repair. Receives broken Verilog + VerifResult + LayerIR. Classifies the failure and performs minimal rewrite." },
} as const;

export type AgentName = keyof typeof AGENT_CONFIG;

export const PIPELINE_CONFIG = {
  max_retries: 3,
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
