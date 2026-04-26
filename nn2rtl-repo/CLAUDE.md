# nn2rtl — Neural Network to RTL compiler

Read this file before touching the repo.

## Core Rules

- Never write `.v` files directly. Always persist Verilog through the `write_verilog` MCP tool.
- Before running the pipeline, run `npm run typecheck` in both `sdk/` and `mcp/`.
- Before the first pipeline run, execute:
  - `python3 scripts/quantize_model.py`
  - `python3 scripts/generate_golden.py checkpoints/resnet50_int8.pth`
- The SDK package is `@anthropic-ai/claude-agent-sdk`, not `@anthropic-ai/claude-code`.
- The static Verilator C++ testbench lives at `tb/static_verilator_tb.cpp` and is handwritten infrastructure, not agent-generated code.

## Output Conventions

- Verilog modules go to `output/rtl/`
- Testbenches go to `output/tb/`
- Weight and bias hex files go to `output/weights/`
- Reports and logs go to `output/reports/`

## Agents

The deterministic TypeScript orchestrator in `sdk/orchestrate.ts` plays both
the pipeline-coordinator role and the verification (Assayer) role itself.
Verification goes through `runAssayerDeterministic` — a deterministic function
that writes the sidecar from LayerIR fields, runs `run_iverilog` then
`run_verilator` via direct MCP import, and returns a Zod-validated `VerifResult`.
No LLM is involved in verification; there is no Assayer agent.

The three LLM agents below are the only ones dispatched via the SDK's
`query()` path.

- `cartographer`
  - Role: PyTorch checkpoint and layer IR extractor (bypassed on the ONNX path)
  - Model: `claude-sonnet-4-6` (simple extraction — Opus is waste here)
  - Defined in `nn2rtl-plugin/agents/cartographer.md`
- `foundry`
  - Role: synthesizable Verilog generator for one `LayerIR`
  - Model: `claude-opus-4-7` (coding best — first-shot correctness matters)
  - Defined in `nn2rtl-plugin/agents/foundry.md`
- `surgeon`
  - Role: targeted Verilog repair specialist
  - Model: `claude-opus-4-7`
  - Defined in `nn2rtl-plugin/agents/surgeon.md`

Models are pinned to full IDs (not tier aliases like `"sonnet"`) in
`sdk/config.ts` so the pick is reproducible regardless of the user's global
`~/.claude/settings.json` default model.

## Architecture

- `nn2rtl-plugin/`
  - Claude Code plugin root containing plugin manifest, plugin agents, plugin skills, and `.mcp.json`
- `sdk/`
  - TypeScript orchestrator using `@anthropic-ai/claude-agent-sdk`
  - Owns the pipeline state machine and agent dispatch loop
- `mcp/`
  - TypeScript MCP server exposing:
    - `run_iverilog`
    - `run_verilator`
    - `run_vivado`
    - `read_weights`
    - `write_verilog`
    - `get_rtl_patterns`

## Working Style

- Keep the pipeline resumable by updating `output/pipeline_state.json` after each state transition.
- Treat `output/layer_ir.json` and `output/reports/*.json*` as runtime artifacts, not hand-authored source files.
- Treat `output/weights/*.hex` and `output/tb/*.sidecar.json` as generated runtime artifacts.
- Preserve the plugin layout exactly: only `plugin.json` lives inside `nn2rtl-plugin/.claude-plugin/`; agents, skills, hooks, and `.mcp.json` live at the plugin root.
