# nn2rtl — Architecture Reference

Last updated: April 14, 2026

This document is a file-by-file tour of the codebase. It complements [README.md](./README.md) (the design spec — *what the project is and why*) by describing *where each decision lives in the code* and *what is still outstanding*.

If you are new to the repo, read the README first. Then use this document when you need to find the code that implements a specific part of the spec, or when you want to know what is still a placeholder.

---

## Top-Level Layout

```
nn2rtl-repo/
├── README.md               # Canonical design specification
├── CLAUDE.md               # Operational rules for working in the repo
├── ARCHITECTURE.md         # This file — file-level reference
├── .gitignore              # Ignores node_modules, dist, runtime outputs, checkpoints
├── .claude/settings.json   # Claude Code workspace permissions
│
├── nn2rtl-plugin/          # Layer 1: Claude Code plugin (agent roles + skills)
├── sdk/                    # Layer 2: TypeScript orchestrator
├── mcp/                    # Layer 3: MCP server exposing hardware toolchain
├── scripts/                # Python pre-processing (quantization + golden vectors)
├── tb/                     # Static C++ Verilator testbench
└── output/                 # Runtime artifacts (git-ignored, stubs committed)
```

The three layers map directly to the README's architecture section. Agents are defined in the plugin layer, the orchestrator lives in `sdk/`, and the MCP server in `mcp/` is the bridge between agents and the external Verilog toolchain.

---

## Root-Level Files

### `README.md`
The canonical specification. Describes the research thesis, scope decisions, architectural choices (weights via `$readmemh`, mandatory pipelined modules with timing contracts, static testbench), the five-agent roster, pipeline flow, data contracts, verification strategy, failure-mode taxonomy, and known risks. When in doubt, this file is the source of truth.

### `CLAUDE.md`
Tactical operational rules surfaced to Claude Code on every session start. Core rules: never write `.v` files directly; always run `npm run typecheck` before the pipeline; run the Python pre-processing scripts once before the first run; the SDK package is `@anthropic-ai/claude-agent-sdk`. Output conventions, agent registry, and plugin layout rules are also here.

### `ARCHITECTURE.md`
This file.

### `.gitignore`
Ignores dependency directories (`node_modules/`), build outputs (`dist/`), all runtime artifacts (`output/rtl/`, `output/tb/`, `output/reports/`, `output/weights/`, generated JSON files), Python bytecode, `.env*`, and OS junk files. Runtime output subdirectories are kept in git via `.gitkeep` where required.

### `.claude/settings.json`
Workspace permissions for Claude Code. Permits Bash invocations for `node`, `npm`, `python3`, `iverilog`, `verilator`, and `yosys`.

---

## Layer 1 — Claude Code Plugin (`nn2rtl-plugin/`)

Defines the five specialized agents and their supporting skill documentation. Loaded by the SDK orchestrator at runtime via the `plugins: [{ type: "local", path: pluginPath }]` option.

### `.claude-plugin/plugin.json`
Plugin manifest. Declares name, version, and paths to agent, skill, and MCP config directories.

### `.mcp.json`
MCP server registration. Points to `../mcp/dist/server.js` with `OUTPUT_DIR=../output` and registers the server under the name `nn2rtl-tools`. Every MCP tool name is therefore prefixed `mcp__nn2rtl-tools__` in `allowedTools` configuration.

### `agents/conductor.md`
Pipeline orchestrator definition. Model: `opus`. Describes an agent that would own `output/pipeline_state.json`, dispatch Foundry / Assayer / Surgeon, and enforce the 3-retry ceiling. **In the current codebase the Conductor agent is loaded into the agent registry but never dispatched** — the real orchestration is implemented deterministically in TypeScript by `PipelineStateManager.tick()` in `sdk/pipeline.ts` and `runPipeline()` in `sdk/orchestrate.ts`. The agent file is kept so a future agentic orchestration mode can be enabled without restructuring the plugin.

### `agents/cartographer.md`
Model extractor. Model: `sonnet`. Runs once. Loads the quantized PyTorch checkpoint, traces via `torch.fx`, folds batch normalization into convolutions, writes weight/bias `.hex` files to `output/weights/`, emits `output/layer_ir.json`. Knows PyTorch, not Verilog.

### `agents/foundry.md`
Primary Verilog generator. Model: `sonnet`. Receives one `LayerIR`, produces one synthesizable `VerilogModule`. Hard rules: INT8 fixed-point, `8×8 → 16-bit` multipliers, signed datapath, saturating residual adds, `$readmemh` for weights, exact `valid_in` / `valid_out` timing, no simulation-only constructs. Must call `write_verilog` to persist.

### `agents/assayer.md`
Simulation runner. Model: `haiku`. Generates the JSON sidecar consumed by the static testbench, runs `iverilog` for syntax, runs the `run_verilator` MCP tool, returns a structured `VerifResult` with timing fields and a classified `failure_class` on functional failure. Never writes source files (`disallowedTools: Write, Edit`).

### `agents/surgeon.md`
Targeted repair specialist. Model: `opus`. Activated on failure. Receives the broken module, the `VerifResult`, and the original `LayerIR`. Must classify the failure into one of the 16 taxonomy classes, locate the exact faulty lines, and rewrite only those lines while preserving the module's port interface. Capped at 3 retries per module.

### `skills/{conductor,cartographer,foundry,assayer,surgeon}/SKILL.md`
Supplemental skill reference material loaded alongside the matching agent prompt. Contains schema reminders, RTL patterns, and domain heuristics. The orchestrator concatenates the skill markdown body onto the agent prompt at load time (see `loadPluginAgentDefinition` in `sdk/orchestrate.ts`).

---

## Layer 2 — TypeScript Orchestrator (`sdk/`)

The deterministic control plane. Not a prompt — a real state machine that reads state from disk, decides the next action, dispatches agents via the Claude Agent SDK's `query()`, validates their structured outputs, and updates state on disk after every transition. This is what makes the pipeline resumable, auditable, and measurable.

### `package.json`
Pinned dependencies: `@anthropic-ai/claude-agent-sdk` and `zod`. Scripts: `build`, `typecheck`, `start`, `dev`, `pipeline`. Node `>=20`, ES modules.

### `package-lock.json`
npm lockfile for the SDK package. Committed to pin the exact dependency graph for reproducible builds; not hand-edited.

### `tsconfig.json`
Strict TypeScript config targeting ES2022 modules.

### `types.ts`
Canonical TypeScript interfaces for the pipeline data contracts:

- `LayerIR` — the master per-module spec. Contains `module_id`, `op_type`, input/output shapes, `weights_path` / `bias_path` / `weight_shape` / `num_weights` (weights are on disk, not inline), `scale_factor`, `zero_point`, timing contract (`pipeline_latency_cycles`, `clock_period_ns`), port widths, signal names (`valid_in_signal`, `valid_out_signal`, `clock_signal`, `reset_signal`), and golden input/output vectors.
- `PipelineIR` — container for all LayerIRs plus model metadata.
- `VerilogModule` — what Foundry and Surgeon produce.
- `VerifResult` — what Assayer returns. Includes timing fields and `failure_class`.
- `VerificationSidecar` — the JSON blob Assayer writes for the static Verilator testbench to consume.
- `PipelineState` — the authoritative run state. Includes `total_cost_usd` and per-model token usage.
- `ModuleStatus` and `NextAction` — discriminated unions for the state machine.
- `FailureClass` — the 16-category taxonomy from the README.

### `schemas.ts` *(new — added for runtime validation)*
Zod runtime schemas that mirror the types in `types.ts` used on SDK-side trust boundaries — agent outputs, disk artifacts, resume state. Centralising the Zod definitions here keeps SDK-side validation aligned with the types. Note that the JSON-Schema blobs passed to the SDK `outputFormat` field are *still* declared inline in `sdk/orchestrate.ts` and again in `mcp/server.ts`; deduplicating those against this file is outstanding work.

Exports: `failureClassSchema`, `moduleStatusSchema`, `layerIrSchema`, `pipelineIrSchema`, `verilogModuleSchema`, `verifResultSchema`, `synthesisReportSchema`, `modelUsageEntrySchema`, `pipelineStateSchema`.

### `config.ts`
Central agent and pipeline configuration. `AGENT_CONFIG` maps each of the five agents to its model tier and description. `PIPELINE_CONFIG` pins `max_retries`, all output paths, and the path to the static testbench. This is the only place where model assignments or paths should change.

### `claude-agent-sdk-compat.ts`
Compatibility shim around `@anthropic-ai/claude-agent-sdk`. Re-exports the types and `query()` function needed by the orchestrator. Required because the published SDK package currently ships without a usable root declaration file; this file is the single point that will be replaced when the SDK's typings are fixed.

### `pipeline.ts`
`PipelineStateManager` — the state machine. Constructor seeds all modules as `pending` with zero attempts. Core methods:

- `tick()` — scans modules in order, returns the next action (`invoke_foundry`, `invoke_surgeon`, or `done`). Mutates the in-memory status.
- `applyVerifResult()` — transitions a module to `pass`, `fail_retry`, or `fail_abort` based on the verification outcome and the retry count.
- `recordAgentUsage()` — accumulates `total_cost_usd` and merges per-model token usage entries so cost tracking survives resume.
- `saveState()` / `loadState()` — persist to `output/pipeline_state.json`. **`loadState` validates the JSON against `pipelineStateSchema` before accepting it**; a corrupted resume file throws with field-level errors rather than silently mutating in-memory state.
- `summary()` — renders a plain-text table used by the orchestrator when writing the pipeline summary.

### `orchestrate.ts`
The entry point. `runPipeline(checkpointPath)` is the whole autonomous loop. Responsibilities:

1. Ensure output directory layout exists.
2. If `output/layer_ir.json` is missing, dispatch Cartographer and persist its result.
3. Initialize or resume `PipelineStateManager` from disk.
4. Loop: `tick()` → dispatch agent → validate structured output → apply `VerifResult` → save state → log transition.
5. On a passing module, invoke `run_yosys` via a direct SDK call and write the synthesis report to `output/reports/<module_id>.yosys.json`.
6. On terminal state, write `output/reports/pipeline_summary.json`.

Agent dispatch goes through `runDelegatedAgent(slug, payload, outputFormat, resultSchema)`. Both the SDK `outputFormat` (JSON Schema) and the local `resultSchema` (Zod) must agree; drift is detected because Zod re-validates the structured output before it is returned.

Helper layers:

- `loadPluginAgentDefinition(slug)` — parses agent `.md` frontmatter plus optional skill body into an `AgentDefinition` for the SDK. The frontmatter parser is hand-rolled and intentionally minimal.
- `requireStructuredOutput(result, label, schema)` — unwraps the SDK's `structured_output`, falls back to `JSON.parse(result.result)` if missing, validates through Zod, and throws a field-level error on mismatch.
- `readJsonFile(path, schema?)` — generic JSON loader. When a Zod schema is provided, the loaded content is validated and the typed value is returned; without a schema it is a plain cast (reserved for untyped internal artifacts).
- `appendRunLog()` — JSONL event log written to `output/reports/run_log.jsonl`.
- `invokeYosys(module)` — direct `run_yosys` MCP call (no subagent), validated against `synthesisReportSchema`.

Cost tracking accumulates over every agent call including Cartographer's bootstrap, Foundry, Assayer, Surgeon, and direct Yosys invocations.

---

## Layer 3 — MCP Server (`mcp/`)

The boundary between agents and the external Verilog toolchain. Five tools, a stdio transport, and strict input validation.

### `package.json`
Depends on `@modelcontextprotocol/sdk` and `zod`. Same script shape as the SDK package. Compiled output (`dist/server.js`) is what `.mcp.json` points to.

### `package-lock.json`
npm lockfile for the MCP package. Committed for the same reproducibility reason as the SDK lockfile.

### `tsconfig.json`
Mirrors the SDK config.

### `types.ts`
An exact copy of `sdk/types.ts`. Kept in sync manually so the MCP server can be built and typed without depending on `sdk/`. If the two diverge, local validation will drift silently, so treat them as one logical file that happens to live in two places.

### `schemas.ts` *(new)*
Zod schemas for MCP tool **inputs** (`runIverilogInput`, `runVerilatorInput`, `runYosysInput`, `readWeightsInput`, `writeVerilogInput`) and for the shared data contracts. Each tool handler in `server.ts` calls `.parse()` on incoming arguments before doing any work, so malformed MCP calls fail immediately with field-level errors rather than being silently cast.

### `tools.ts`
Tool implementations:

- `run_iverilog(verilog_source, module_name)` — writes the source to a temporary file, runs `iverilog -o /dev/null -g2012`, returns `{ success, stderr }`. This tool is fully implemented.
- `run_verilator(verilog_source, module_name, sidecar_path)` — currently runs only `verilator --lint-only`. The full implementation (copy static testbench, compile with `verilator --cc --exe --build`, execute, parse results JSON) is outstanding.
- `run_yosys(verilog_source, module_name)` — runs `yosys -p "synth_ice40 -abc9; stat"`, parses LUT count and an `MHz` figure from the report, returns `{ success, lut_count, fmax_mhz, report }`. Implemented.
- `read_weights(checkpoint_path, quantization_config)` — spawns `python3 scripts/generate_golden.py`, reads `output/golden_vectors.json`, returns it as `PipelineIR`. **Currently cannot complete successfully**: `scripts/generate_golden.py` writes a placeholder `golden_vectors.json` and then unconditionally raises `NotImplementedError`, so the child-process `await` throws. Even if the error were suppressed, the placeholder payload contains extra fields (`checkpoint_fingerprint`, `note`) that fail strict `pipelineIrSchema` validation on the SDK side. The raw JSON is also cast without schema validation at the MCP-tool level.
- `write_verilog(module, output_dir)` — the only way Verilog reaches disk. Writes `<output_dir>/rtl/<module_id>.v` and `<module_id>.meta.json`, returns the absolute `.v` path. Implemented.

### `server.ts`
The MCP stdio server. Registers the five tools with full JSON Schema input and output definitions, routes incoming tool calls through a handler per tool. Every handler parses its arguments through the matching Zod schema from `schemas.ts` before invoking the tool implementation.

---

## Python Pre-Processing (`scripts/`)

Single-use utilities the human runs once before the autonomous pipeline. These produce the ground truth that every agent consumes.

### `quantize_model.py`
**Scaffolded, not implemented.** Intended to load torchvision ResNet-50, perform INT8 symmetric per-tensor post-training static quantization with calibration, save the checkpoint to `checkpoints/resnet50_int8.pth`, and print per-layer scale factors as JSON.

### `generate_golden.py`
**Scaffolded, not implemented.** Intended to load the quantized checkpoint, trace through `torch.fx`, capture per-layer INT8 activation tensors for a fixed 224×224 test image, fold batch normalization into the preceding convolution, serialize weights and biases to `$readmemh`-compatible `.hex` files in `output/weights/`, and emit `output/golden_vectors.json` in `PipelineIR` shape (with `weights_path` / `bias_path` / timing / width / signal metadata).

---

## Static Testbench (`tb/`)

Handwritten C++ Verilator driver. Intentionally *not* agent-generated — the README flags this as an architectural decision to avoid the two-bug problem (wrong RTL + wrong testbench).

### `static_verilator_tb.cpp`
Loads the sidecar JSON (path passed as `argv[1]`), reads golden input/output vectors from the paths it references, applies reset for five cycles, streams inputs through `data_in` with `valid_in=1`, measures the cycle delta to the first `valid_out` assertion, samples `data_out` for each expected value, computes `max_error` / `mean_error`, and writes a structured results JSON to `results_path`. Hang detection: a vector whose `valid_out` never asserts within `2 × pipeline_latency_cycles + 8` cycles throws with a clear error.

Build-time contract: the DUT is selected via preprocessor macros `VMODEL_HEADER` (path to the generated Verilator header) and `VMODEL_CLASS` (the Verilator-generated class name). `run_verilator` sets both when it invokes `verilator --cc --exe --build`. Run-time contract: the DUT must use canonical port names `clk`, `rst_n`, `valid_in`, `valid_out`, `data_in`, `data_out`. The sidecar's signal-name fields are validated against these canonical names before simulation starts; a mismatch throws immediately.

Numerical pass threshold is `max_error <= 3` (from the README). Timing pass requires `timing_actual_cycles == pipeline_latency_cycles` exactly. `failure_class` is emitted as `null` — Assayer classifies failures after reading the results file.

### `third_party/json.hpp`
Vendored copy of [nlohmann/json v3.11.3](https://github.com/nlohmann/json), single-header JSON library. Used by the testbench for sidecar, golden-vector, and results I/O. Kept in-tree so the C++ build is self-contained and reproducible without a package manager.

### `README.md`
Per-directory reference documenting the build-time macros, run-time contract, sidecar schema, and results schema in one place.

---

## Runtime Output Layout (`output/`)

All runtime artifacts live here. The directory layout is committed via empty `.gitkeep` files (one per runtime subdirectory) while the contents themselves are git-ignored. These `.gitkeep` files are the only source-controlled files under `output/` and exist purely so the layout survives a fresh clone.

- `output/rtl/<module_id>.v` — Foundry / Surgeon output, written exclusively through `write_verilog`.
- `output/rtl/<module_id>.meta.json` — sidecar metadata for each generated module.
- `output/tb/<module_id>.sidecar.json` — Assayer's per-run sidecar for the static testbench.
- `output/weights/<module_id>_weights.hex`, `<module_id>_bias.hex` — Cartographer's hex-format tensors.
- `output/reports/run_log.jsonl` — JSONL event stream for the full run.
- `output/reports/<module_id>.yosys.json` — Yosys synthesis report per passing module.
- `output/reports/pipeline_summary.json` — final summary including total cost and model usage.
- `output/layer_ir.json` — Cartographer's canonical `PipelineIR`.
- `output/pipeline_state.json` — authoritative `PipelineState`, updated after every transition, validated through Zod on resume.
- `output/golden_vectors.json` — raw Python output before it is promoted to `layer_ir.json`.

---

## Runtime Validation Policy

Runtime Zod validation was added for the main trust boundaries. The policy:

| Boundary | Where validated | Schema |
|---|---|---|
| MCP tool arguments | `mcp/server.ts` handlers | `mcp/schemas.ts` per-tool input schemas |
| Agent `structured_output` | `sdk/orchestrate.ts` `requireStructuredOutput` | `sdk/schemas.ts` (per agent) |
| `output/layer_ir.json` on load | `sdk/orchestrate.ts` `ensureLayerIr` | `pipelineIrSchema` |
| `output/rtl/<id>.meta.json` on load | `sdk/orchestrate.ts` `loadPersistedVerilogModule` | `verilogModuleSchema` |
| `output/pipeline_state.json` on resume | `sdk/pipeline.ts` `loadState` | `pipelineStateSchema` |
| Yosys direct MCP result | `sdk/orchestrate.ts` `invokeYosys` | `synthesisReportSchema` |

Validation failures throw with field-level error paths so corrupted artifacts and malformed agent outputs fail loudly instead of silently propagating bad data.

Known gaps in this policy (to be closed):

- `run_verilator` in `mcp/tools.ts` reads the Assayer-authored sidecar via an unchecked `readJsonFileIfPresent<VerificationSidecar>` cast; there is no `VerificationSidecar` Zod schema in `mcp/schemas.ts`.
- `read_weights` in `mcp/tools.ts` parses `output/golden_vectors.json` with `JSON.parse(raw) as PipelineIR`, bypassing schema validation at the MCP-tool level. The SDK-side `ensureLayerIr` loader does validate, but only after the unchecked value has been returned across the MCP boundary.

---

## Outstanding TODOs

Categorized list. Entries reference exact file and line where the work lives today. In future revisions these entries should be replaced with descriptions of the finished implementation.

### Critical Path (blocks a real pipeline run)

These require external tools and ML libraries; they cannot be completed in pure TypeScript.

- **`mcp/tools.ts:71` — `run_verilator`**: write the generated source to a temp file, load the JSON sidecar, invoke `verilator --cc --exe --build` with `-DVMODEL_HEADER` / `-DVMODEL_CLASS` pointing at the generated DUT, run the produced binary with the sidecar path, parse the structured results JSON written to `results_path`, and return a complete `VerifResult` including timing checks. Today the function only runs `--lint-only` and returns a placeholder failure. The testbench itself is now implemented at `tb/static_verilator_tb.cpp`.
- **Foundry prompt — canonical signal names**: Foundry's system prompt and schema must be updated so every generated module uses exactly `clk`, `rst_n`, `valid_in`, `valid_out`, `data_in`, `data_out`. The static testbench validates these names at run time and refuses anything else. Today the prompt lets the LayerIR's signal-name fields pick arbitrary names.
- **`mcp/tools.ts:147` — `read_weights`**: spawn `python3 scripts/generate_golden.py`, wait for completion, verify that `.hex` weight and bias files were emitted under `output/weights/`, and parse `output/golden_vectors.json`. The Python script still raises `NotImplementedError` (see `scripts/generate_golden.py:82`), which makes the current tool invocation throw.
- **`scripts/quantize_model.py` — PTQ pipeline**: load torchvision ResNet-50, run calibration, convert with `torch.quantization`, persist the INT8 checkpoint, print per-layer scale factors as JSON.
- **`scripts/generate_golden.py:36–40` — golden-vector capture**: load the quantized checkpoint, trace via `torch.fx`, fold batch normalization into preceding convolutions, serialize weights and biases to `$readmemh`-compatible hex files, and emit a full `PipelineIR` with `weights_path` / `bias_path` / latency / clock / width / signal metadata. Also remove the final `NotImplementedError` at line 82 once real output is emitted.

### Blocked on External Dependencies

- **`sdk/claude-agent-sdk-compat.ts:1`**: remove the compatibility shim once `@anthropic-ai/claude-agent-sdk` ships a valid root declaration file.
- **`sdk/orchestrate.ts:415–416`**: restore `AgentDefinition.skills` and `AgentDefinition.maxTurns` once the published SDK typings expose them; today the parent query's `maxTurns` is the only guardrail.

### Design Decisions Deferred

These are intentional "maybe later" notes, not bugs.

- **`sdk/orchestrate.ts:406`**: replace the hand-rolled frontmatter parser with a real YAML parser (e.g. `yaml` or `gray-matter`) if plugin frontmatter grows more expressive.
- **`sdk/orchestrate.ts:861`, `:920`**: decide whether Yosys synthesis failures should trigger the retry loop. Today a module that passes simulation but fails synthesis is still marked `pass` and a degraded Yosys report lands in `output/reports/`.
- **`sdk/orchestrate.ts:938`**: promote Assayer into a first-class `tick()` action if the state machine grows beyond the current Foundry-or-Surgeon binary choice.
- **`sdk/orchestrate.ts:958`**: extend CLI argument parsing when the pipeline grows knobs — alternate plugins, custom output roots, per-run retry budgets.
- **Deduplicate JSON-Schema blobs**: `sdk/orchestrate.ts:96` and `mcp/server.ts:63` each declare the same JSON-Schema shapes (LayerIR, PipelineIR, VerilogModule, VerifResult, synthesis report). These could be generated from the Zod schemas in `sdk/schemas.ts` / `mcp/schemas.ts` to eliminate the duplication, or at least cross-linked so drift is caught at build time.
- **Dispatch the Conductor agent (or remove it)**: `nn2rtl-plugin/agents/conductor.md` is loaded by `loadAllAgentDefinitions` but never invoked; either wire it into the orchestration path as an optional agentic-mode switch, or drop the agent file so the plugin matches the deterministic TypeScript orchestration that actually runs.

### Done (removed from the outstanding list)

For the record, these items from earlier revisions are complete:

- Inline `weight_int8` tensor in `LayerIR` replaced with `weights_path` / `bias_path` plus hex-file convention.
- `LayerIR` extended with timing, width, and signal-name metadata.
- `VerifResult` extended with `timing_pass`, `timing_actual_cycles`, `timing_expected_cycles`, `failure_class`.
- `PipelineState` extended with `total_cost_usd` and `model_usage`, both tracked across agent calls.
- Foundry system prompt rewritten to require `$readmemh`, valid/ready streaming, and exact pipeline latency.
- Assayer system prompt rewritten to generate the JSON sidecar and surface timing results.
- Stale TODO comments on `run_iverilog`, `run_yosys`, and `write_verilog` removed — those tools are implemented.
- Zod runtime validation added across MCP tool inputs, agent structured outputs, and all disk-loaded artifacts including the resume state file.
- Static Verilator testbench (`tb/static_verilator_tb.cpp`) implemented: sidecar parsing, canonical-signal enforcement, reset + streaming drive, hang detection, cycle-accurate timing measurement, numerical tolerance check, structured results JSON, best-effort error propagation. nlohmann/json vendored at `tb/third_party/json.hpp`.

---

## Working Rules Summary

From [CLAUDE.md](./CLAUDE.md), reiterated here for convenience:

- Never write `.v` files directly — always use `write_verilog`.
- Before any pipeline run: `npm run typecheck` in both `sdk/` and `mcp/`.
- Before the first pipeline run ever: run `scripts/quantize_model.py`, then `scripts/generate_golden.py`.
- The SDK package is `@anthropic-ai/claude-agent-sdk`, not `@anthropic-ai/claude-code`.
- The static Verilator testbench at `tb/static_verilator_tb.cpp` is handwritten infrastructure; never let an agent regenerate it.
- Keep `output/pipeline_state.json` updated after every state transition so the pipeline is resumable.
- Treat everything under `output/` as runtime output, not source.
- Preserve plugin layout: only `plugin.json` lives inside `nn2rtl-plugin/.claude-plugin/`; agents, skills, hooks, and `.mcp.json` live at the plugin root.
