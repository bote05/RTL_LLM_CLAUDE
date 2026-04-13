// Docs: https://platform.claude.com/docs/en/agent-sdk/typescript
// AgentDefinition model field accepts: "sonnet" | "opus" | "haiku" | "inherit"
// effort is only supported in .claude/agents/ file-based agents, not in AgentDefinition
// For SDK programmatic agents, model tier is the only override available

export const AGENT_CONFIG = {
  Conductor:    { model: "opus"   as const, description: "Pipeline orchestrator. Decides which agent to invoke next. Never generates Verilog." },
  Cartographer: { model: "sonnet" as const, description: "Model extractor. Runs once at pipeline start. Emits output/layer_ir.json." },
  Foundry:      { model: "sonnet" as const, description: "Verilog codegen. Receives one LayerIR, produces one VerilogModule." },
  Assayer:      { model: "haiku"  as const, description: "Simulation runner. Runs iverilog and Verilator, returns VerifResult JSON." },
  Surgeon:      { model: "opus"   as const, description: "Targeted repair. Receives broken Verilog + VerifResult + LayerIR. Minimal rewrite only." },
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
