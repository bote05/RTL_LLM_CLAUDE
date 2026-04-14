# nn2rtl — Architecture Reference

Last updated: April 14, 2026

This document is a file-by-file tour of the codebase. It complements [README.md](./README.md) (the design spec — *what the project is and why*) by describing *where each decision lives in the code* and *what is still outstanding*.

Read the README first. Then use this document to find the code that implements a specific part of the spec, or to see what is still a placeholder.

---

## Top-Level Layout

```
nn2rtl-repo/
├── README.md                  # Canonical design specification
├── CLAUDE.md                  # Operational rules for working in the repo
├── ARCHITECTURE.md            # This file — file-level reference
├── package.json               # Monorepo test scripts (test:fast / test:full / coverage)
├── pytest.ini                 # Python test config (markers, default flags)
├── .gitignore
├── .claude/settings.json      # Claude Code workspace permissions
│
├── nn2rtl-plugin/             # Layer 1: Claude Code plugin (agent roles + skills)
├── sdk/                       # Layer 2: TypeScript orchestrator
├── mcp/                       # Layer 3: MCP server exposing hardware toolchain
├── scripts/                   # Python pre-processing (quantization + golden vectors)
├── tb/                        # Static C++ Verilator testbench
├── test/fixtures/             # Cross-language fixtures used by Python & TS tests
└── output/                    # Runtime artifacts (git-ignored; empty subdirs kept)
```

The three layers map directly to the README's architecture section: agents in the plugin layer, deterministic orchestration in `sdk/`, hardware-tool bridge in `mcp/`.

---

## Root-Level Files

### `README.md`
The canonical specification. Describes the research thesis, scope decisions, architectural choices (weights via `$readmemh`, pipelined modules with timing contracts, static testbench), the five-agent roster, pipeline flow, data contracts, verification strategy, failure-mode taxonomy, and known risks. Source of truth when the code and docs disagree.

### `CLAUDE.md`
Tactical operational rules surfaced to Claude Code on every session start. Core rules: never write `.v` files directly; always run `npm run typecheck` before the pipeline; run the Python pre-processing scripts once before the first run; the SDK package is `@anthropic-ai/claude-agent-sdk`.

### `ARCHITECTURE.md`
This file.

### `package.json` *(root)*
Thin monorepo aggregator. No dependencies of its own. Exposes three scripts that fan out into both TypeScript packages and the Python suite:

- `test:fast` — vitest in `sdk/` and `mcp/` plus `pytest -m "not full"` (skip heavy markers).
- `test:full` — vitest in both packages plus `pytest -m "not manual"` (includes heavy tests, skips only opt-in manual smoke tests).
- `coverage` — per-package vitest coverage followed by `pytest -m "not manual" --cov=scripts --cov-branch --cov-fail-under=90`.

### `pytest.ini`
Default pytest options (`-ra` report) and registered markers: `full` (heavy / slow tests) and `manual` (opt-in smoke tests that require user-supplied external artifacts).

### `.gitignore`
Ignores `node_modules/`, `dist/`, `*.tsbuildinfo`, log files, IDE dirs, simulator artifacts (`obj_dir/`, `*.vcd`, `*.fst`, `*.vvp`, `*.blif`), Python virtualenvs (`.venv/`, `venv/`, `*.egg-info/`), runtime outputs under `output/`, `checkpoints/`, `__pycache__/`, `.env*`, and OS junk.

### `.claude/settings.json`
Workspace permissions for Claude Code. Permits Bash invocations for `node`, `npm`, `python3`, `iverilog`, `verilator`, and `yosys`.

---

## Layer 1 — Claude Code Plugin (`nn2rtl-plugin/`)

Defines the five specialized agents and their supporting skill documentation. Loaded by the SDK orchestrator via `plugins: [{ type: "local", path: pluginPath }]`.

### `.claude-plugin/plugin.json`
Plugin manifest. Declares name, version, and paths to agent, skill, and MCP config directories.

### `.mcp.json`
MCP server registration. Points to `../mcp/dist/server.js` with `OUTPUT_DIR=../output` and registers the server as `nn2rtl-tools`. Every MCP tool name is therefore prefixed `mcp__nn2rtl-tools__` in `allowedTools`.

### `agents/conductor.md`
Pipeline orchestrator definition. Model: `opus`. Describes the agent that would own `output/pipeline_state.json`, dispatch Foundry / Assayer / Surgeon, and enforce the 3-retry ceiling. **In the current codebase the Conductor agent is loaded into the agent registry but never dispatched** — orchestration is implemented deterministically in TypeScript by `PipelineStateManager.tick()` in `sdk/pipeline.ts` and `runPipeline()` in `sdk/orchestrate.ts`. The agent file is retained so a future agentic orchestration mode can be enabled without restructuring the plugin.

### `agents/cartographer.md`
Model extractor. Model: `sonnet`. Runs once. Loads the quantized PyTorch checkpoint, traces via `torch.fx`, folds batch normalization into convolutions, writes weight/bias `.hex` files to `output/weights/`, emits `output/layer_ir.json`. The JSON schema enforces that signal-name fields are emitted as the canonical literals (`"clk"`, `"rst_n"`, `"valid_in"`, `"valid_out"`, `"ready_in"`, `"data_in"`, `"data_out"`).

### `agents/foundry.md`
Primary Verilog generator. Model: `sonnet`. Receives one `LayerIR`, produces one synthesizable `VerilogModule`. Hard rules: INT8 fixed-point, `8×8 → 16-bit` multipliers, signed datapath, saturating residual adds, `$readmemh` for weights, exact `pipeline_latency_cycles` from first `valid_in` to first `valid_out`, no simulation-only constructs. Canonical port names are mandatory (`clk`, `rst_n`, `valid_in`, `ready_in`, `data_in`, `valid_out`, `data_out`); `ready_in` is a module **output** (upstream backpressure) that may be tied high if stalling is not needed. Must call `write_verilog` to persist.

### `agents/assayer.md`
Simulation runner. Model: `haiku`. Generates the JSON sidecar consumed by the static testbench, runs `iverilog` for syntax, calls the `run_verilator` MCP tool, returns a structured `VerifResult` with timing fields and a classified `failure_class` on functional failure. Never writes source files (`disallowedTools: Write, Edit`).

### `agents/surgeon.md`
Targeted repair specialist. Model: `opus`. Activated on failure. Receives the broken module, the `VerifResult`, and the original `LayerIR`. Must classify the failure into one of the 16 taxonomy classes, locate the exact faulty lines, and rewrite only those lines while preserving the module's port interface. Capped at 3 retries per module.

### `skills/{conductor,cartographer,foundry,assayer,surgeon}/SKILL.md`
Supplemental skill reference material loaded alongside the matching agent prompt. The orchestrator concatenates the skill markdown body onto the agent prompt at load time (see `loadPluginAgentDefinition` in `sdk/orchestrate.ts`). Contains schema reminders, RTL patterns, and canonical-signal-name reminders.

---

## Layer 2 — TypeScript Orchestrator (`sdk/`)

The deterministic control plane. Not a prompt — a real state machine that reads state from disk, decides the next action, dispatches agents via the Claude Agent SDK's `query()`, validates their structured outputs through Zod, and updates state on disk after every transition. Resumable, auditable, measurable.

### `package.json`
Dependencies: `@anthropic-ai/claude-agent-sdk`, `zod`. Dev: `tsx`, `typescript`, `vitest`, `@vitest/coverage-v8`. Scripts: `build`, `typecheck`, `start` (runs `dist/main.js`), `dev` (watches `main.ts`), `pipeline`, `test`, `test:fast`, `test:full`, `coverage`.

### `package-lock.json`
npm lockfile; pins the dependency graph. Not hand-edited.

### `tsconfig.json`
Strict TypeScript targeting ES2022 modules.

### `vitest.config.ts`
Vitest test config: includes `test/**/*.test.ts`, coverage via the v8 provider, with 95% branch / 100% line / 100% function / 100% statement thresholds. Coverage includes `config.ts`, `orchestrate.ts`, `pipeline.ts`, `schemas.ts`; excludes `main.ts`, `types.ts`, and build output.

### `main.ts` *(entry point)*
Tiny CLI wrapper. Calls `runCli()` and routes uncaught errors through `handlePipelineError()` exported from `orchestrate.ts`. Kept separate from the library code so `orchestrate.ts` can be imported by tests without triggering `process.argv` parsing.

### `types.ts`
Canonical TypeScript interfaces for the pipeline data contracts:

- `LayerIR` — master per-module spec. `module_id`, `op_type`, input/output shapes, `weights_path` / `bias_path` / `weight_shape` / `num_weights` (weights on disk, not inline), `scale_factor`, `zero_point`, timing contract (`pipeline_latency_cycles`, `clock_period_ns`), port widths, **seven canonical signal names typed as string literals** (`"clk"`, `"rst_n"`, `"valid_in"`, `"valid_out"`, `"ready_in"`, `"data_in"`, `"data_out"`), and golden input/output vectors.
- `PipelineIR` — container for all LayerIRs plus model metadata.
- `VerilogModule` — what Foundry and Surgeon produce.
- `VerifResult` — what Assayer returns. Includes timing fields and `failure_class`.
- `VerificationSidecar` — the JSON blob Assayer writes for the static testbench to consume; includes all seven signal names.
- `PipelineState` — authoritative run state. Includes `total_cost_usd` and per-model `model_usage`.
- `ModuleStatus` and `NextAction` — discriminated unions for the state machine.
- `FailureClass` — the README's 16-category repair taxonomy plus `synthesis_failed`, used when a module passes simulation but fails the post-pass Yosys synthesis step.

### `schemas.ts`
Zod 4 runtime schemas mirroring `types.ts`. Single source of truth for every JSON Schema the SDK or MCP server advertises — those are now derived with `z.toJSONSchema(...)` rather than hand-written. Exports: `failureClassSchema`, `moduleStatusSchema`, `layerIrSchema`, `pipelineIrSchema`, `verilogModuleSchema`, `verifResultSchema`, `synthesisReportSchema`, `modelUsageEntrySchema`, `pipelineStateSchema`.

Constraint highlights:

- `layerIrSchema`: all seven signal-name fields are `z.literal(...)`; `attempt`, widths, latency are `int().positive()`.
- `verilogModuleSchema.attempt`: `z.number().int().positive()` (Foundry starts at 1).
- `pipelineStateSchema`: `max_retries`, `attempts.value` are `int().nonnegative()`; `total_cost_usd` is `nonnegative()`. A `superRefine` enforces cross-field invariants:
  - Every `modules` key has an `attempts` entry and vice versa.
  - Every `results` key refers to a known module.
  - `fail_retry` requires a prior `VerifResult` and `attempts < max_retries`.
  - `fail_abort` requires a prior `VerifResult` and `attempts >= max_retries`.
  - `pass` requires a `VerifResult` whose own `status === "pass"`.
  - Every `results[id].module_id` must equal `id`.

### `config.ts`
`AGENT_CONFIG` maps each of the five agents to its model tier and description. `PIPELINE_CONFIG` pins `max_retries`, all output paths, and the path to the static testbench. Single point of change for model assignments and paths.

### `claude-agent-sdk-compat.ts`
Compatibility shim around `@anthropic-ai/claude-agent-sdk`. Re-exports the types and `query()` function. Required because the published SDK currently ships without a usable root declaration file; this file is the single point that will be replaced when the SDK's typings are fixed.

### `pipeline.ts`
`PipelineStateManager` — the state machine. Constructor seeds all modules as `pending` with zero attempts.

- `tick()` — scans modules in order, returns the next action (`invoke_foundry`, `invoke_surgeon`, or `done`). Mutates in-memory status.
- `applyVerifResult()` — transitions to `pass`, `fail_retry`, or `fail_abort` based on outcome and retry count.
- `recordAgentUsage()` — accumulates `total_cost_usd` and merges per-model token usage so cost tracking survives resume.
- `saveState()` / `loadState()` — persist to `output/pipeline_state.json`. `loadState` validates through `pipelineStateSchema` and then **repairs transient statuses** from a crashed prior run:
  - `generating` + no prior result → `pending` (Foundry crashed; re-run from scratch).
  - `generating` + prior result → `fail_retry`, `attempts--` (Surgeon crashed mid-repair; re-run Surgeon without double-billing the retry).
  - `verifying` + no prior result → `pending` (Assayer crashed after Foundry; re-run Foundry).
  - `verifying` + prior result → `fail_retry`, `attempts--` (Assayer crashed after Surgeon; re-run Surgeon).
  The per-crash-point reasoning is documented inline.
- `summary()` — plain-text table used when writing the pipeline summary.

### `orchestrate.ts`
Library module. Exports `runPipeline`, `runCli`, `handlePipelineError`, plus many helpers for tests (`AGENT_SLUGS`, `createOrchestratorRuntime`, `resolveFromSdk`, `normalizeAgentName`, `parseFrontmatter`, `splitCsvField`, `readText`, `pathExists`, `readJsonFile`, `writeJsonFile`, `appendRunLog`, `ensureOutputLayout`, `loadPluginAgentDefinition`, `loadAllAgentDefinitions`, `buildDelegationPrompt`, `requireStructuredOutput`, `findLayer`, `loadPersistedVerilogModule`, `ensureLayerIr`, `writePipelineSummary`, `parseCliArgs`).

`runPipeline(checkpointPath, options)` is the whole autonomous loop:

1. Ensure output directory layout exists.
2. If `output/layer_ir.json` is missing, dispatch Cartographer and persist its result.
3. Initialize `PipelineStateManager`, or resume from `output/pipeline_state.json` if `options.resume`.
4. Loop: `tick()` → dispatch agent → validate structured output through Zod → apply `VerifResult` → save state → log transition.
5. On a passing module, invoke `run_yosys` via a direct SDK call and write `output/reports/<module_id>.yosys.json`. If Yosys reports `success: false`, synthesize a `VerifResult` with `failure_class: "synthesis_failed"` and feed it back through `PipelineStateManager.applyVerifResult()`, so the module enters the same `fail_retry` / `fail_abort` path as any other failure.
6. On terminal state, write `output/reports/pipeline_summary.json`.

**Runtime injection** via `OrchestratorRuntime = { now, queryFn }`: every helper accepts either a full or partial runtime so tests can supply deterministic clocks and mock `query()` implementations. The default is `{ now: () => new Date(), queryFn: query }`.

Agent dispatch goes through `runDelegatedAgent(slug, payload, outputFormat, resultSchema, runtime)`. Both the SDK `outputFormat` (JSON Schema, generated from Zod via `z.toJSONSchema`) and the local `resultSchema` (Zod) are derived from the same schema export, so drift is structurally impossible.

`requireStructuredOutput` unwraps the SDK's `structured_output`, falls back to `JSON.parse(result.result)` if missing, validates through the supplied Zod schema, and throws a field-level error on mismatch.

Cost tracking accumulates over every agent call including Cartographer's bootstrap, Foundry, Assayer, Surgeon, and direct Yosys invocations.

---

## Layer 3 — MCP Server (`mcp/`)

The boundary between agents and the external Verilog toolchain. Five tools, a stdio transport, strict input validation.

### `package.json`
Dependencies: `@modelcontextprotocol/sdk`, `zod`. Dev: `tsx`, `typescript`, `vitest`, `@vitest/coverage-v8`. Scripts match the SDK package. Compiled output `dist/main.js` is the CLI entry; `dist/server.js` is what `.mcp.json` points to.

### `package-lock.json`
npm lockfile; committed for reproducibility.

### `tsconfig.json`
Mirrors the SDK config.

### `vitest.config.ts`
Same shape as the SDK's. Coverage includes `schemas.ts`, `server.ts`, `tools.ts`; excludes `main.ts`, `types.ts`, build output. Thresholds: 95% branches, 100% lines/functions/statements.

### `main.ts` *(entry point)*
Tiny wrapper that calls `startServer()` from `server.ts` and logs fatal errors. Separation of concerns: keeps `server.ts` testable without side effects.

### `types.ts`
Mirror of `sdk/types.ts`. Kept in sync manually so the MCP server can build and type-check without depending on `sdk/`. If the two diverge, local validation drifts silently — treat them as one logical file in two places.

### `schemas.ts`
Single source of truth for MCP-side schemas. Exports:

- Shared data contracts: `failureClassSchema`, `layerIrSchema` (with all seven canonical signal literals), `pipelineIrSchema`, `verilogModuleSchema`, `verifResultSchema`, `verificationSidecarSchema` (with all seven canonical signal literals).
- Per-tool input schemas: `runIverilogInput`, `runVerilatorInput`, `runYosysInput`, `readWeightsInput`, `writeVerilogInput`.
- Per-tool output schemas: `runIverilogOutput`, `runYosysOutput`, `writeVerilogOutput`.

`server.ts` advertises each tool's `inputSchema` and `outputSchema` via `z.toJSONSchema(...)` from these Zod definitions — no hand-written JSON Schema.

### `tools.ts`
Tool implementations. Each exposes a `ToolsRuntime` override parameter so tests can supply a mock `commandRunner`, `cwd`, `env`, and `tmpDirRoot`. The default runtime uses `execFile`, `process.cwd()`, `process.env`, and a writable system temp root (`/tmp` on non-Windows, `os.tmpdir()` on Windows). It does not set an explicit child-process timeout; that is intentional because real Verilator builds can take minutes. If a timeout is added later, production runs should use a generous ceiling (roughly 10 minutes, not smoke-test scale).

Exports: `CommandRunner`, `ToolsRuntime`, `createToolsRuntime`, `withTempDir`, `stderrFromUnknown`, `parseYosysReport`, `resolveOutputRoot`, `TB_SOURCE_PATH`, `TB_JSON_HPP_PATH`, `run_iverilog`, `run_verilator`, `run_yosys`, `read_weights`, `write_verilog`, `readSidecarIfPresent`.

- `run_iverilog(verilog_source, module_name)` — writes the source to a temp file, runs `iverilog -o /dev/null -g2012`, returns `{ success, stderr }`. **Implemented.**
- `run_verilator(verilog_source, module_name, sidecar_path)` — loads and validates the sidecar via `readSidecarIfPresent` (Zod-checked), rejects relative `golden_inputs_path` / `golden_outputs_path` / `results_path`, copies the static testbench plus vendored `third_party/json.hpp` into a temp build dir, invokes `verilator --cc --exe --build` with `VMODEL_HEADER` / `VMODEL_CLASS`, runs the produced binary with the sidecar path, reads `sidecar.results_path`, validates it through `verifResultSchema`, and maps build / execution failures into well-formed `VerifResult` payloads. **Implemented.**
- `run_yosys(verilog_source, module_name)` — runs `yosys -p "synth_ice40 -abc9; stat"`, uses the exported `parseYosysReport` helper to extract LUT count and an `MHz` figure, returns `{ success, lut_count, fmax_mhz, report }`. **Implemented.**
- `read_weights(checkpoint_path, quantization_config)` — spawns `python3 scripts/generate_golden.py`, reads `output/golden_vectors.json`, and validates it against `pipelineIrSchema` before returning. The current implementation is deterministic and local-first: it drives the toy-model checkpoint/golden-vector flow under `scripts/`, not the final ResNet-50 extraction pipeline.
- `write_verilog(module, output_dir)` — the only way Verilog reaches disk. Writes `<output_dir>/rtl/<module_id>.v` and `<module_id>.meta.json`, returns the absolute `.v` path. **Implemented.**
- `readSidecarIfPresent(filePath)` — returns the parsed `VerificationSidecar` or `null` if the file is missing (ENOENT). Any other error, or a schema mismatch, throws with a field-level message.

### `server.ts`
MCP stdio server factory. Exports:

- `ToolImplementations` / `DEFAULT_TOOL_IMPLEMENTATIONS` — the tool-impl injection seam for tests.
- `toolDefinitions` — the five tool schemas (input + output), each derived from Zod via `toJsonSchema()`.
- `handleToolCall(name, args, impls?)` — routes a single tool call. Used by both the real server and unit tests.
- `createServer(impls?)` — constructs a `Server` with the stdio handlers wired up.
- `startServer(impls?)` — same plus `connect(new StdioServerTransport())`. Called by `main.ts`.

Each handler parses its arguments through the matching Zod schema before invoking the (possibly injected) tool implementation, so malformed MCP calls fail immediately with field-level errors.

---

## Python Pre-Processing (`scripts/`)

Single-use utilities the human runs once before the autonomous pipeline. Split into thin CLI wrappers plus importable `*_impl.py` modules so pytest can exercise the helpers without depending on the eventual full ResNet-50 extraction path.

### `__init__.py`
Makes `scripts/` an importable package so `pytest` (and future tests) can `from scripts.golden_impl import ...`.

### `paths.py`
`detect_repo_root(current_file)` — resolves the repo root, honoring a `NN2RTL_REPO_ROOT` env override so tests can point the scripts at a temp dir.

### `quantize_impl.py`
Importable helpers for the deterministic automated test flow. Core pieces:

- `resolve_checkpoint_path(...)`, `get_quantized_checkpoint_path(...)` — canonical checkpoint path resolution.
- `build_toy_quantized_checkpoint(...)`, `write_quantized_checkpoint(...)`, `load_quantized_checkpoint(...)` — create, persist, and validate the toy checkpoint format.
- `ToyPointwiseModel`, `create_toy_model(...)`, `run_toy_model(...)` — a tiny deterministic INT8-friendly model used by the automated suite.
- `build_quantization_summary(...)` — machine-readable summary emitted by the CLI.

The helper layer validates checkpoint structure eagerly and raises `CheckpointValidationError` for malformed metadata.

### `quantize_model.py`
Thin CLI wrapper over `quantize_impl.py`. Today it writes a deterministic toy checkpoint at `checkpoints/resnet50_int8.pth` and prints a stable summary JSON for local testing. This is intentionally a local-first stand-in for the eventual real PTQ flow.

### `golden_impl.py`
Importable helpers for the deterministic golden-vector flow. Core pieces:

- `get_output_paths(...)`, `get_weight_artifact_paths(...)` — canonical output/weights layout.
- `int8_to_hex(...)`, `write_signed_int8_hex(...)` — `$readmemh`-compatible signed INT8 serialization.
- `fold_batch_norm_into_conv(...)` — folds BN parameters into the toy convolution weights/bias.
- `build_pipeline_ir_payload(...)`, `write_pipeline_ir(...)` — produce a valid one-layer `PipelineIR` plus emitted `.hex` weight/bias artifacts.

### `generate_golden.py`
Thin CLI wrapper over `golden_impl.py`. Today it reads the toy checkpoint, emits a valid deterministic `output/golden_vectors.json`, and writes matching `.hex` weight and bias files under `output/weights/`. This is intentionally a local-first stand-in for the eventual real `torch.fx` extraction path.

---

## Static Testbench (`tb/`)

Handwritten C++ Verilator driver. Intentionally *not* agent-generated — the README flags this as an architectural decision to avoid the two-bug problem (wrong RTL + wrong testbench).

### `static_verilator_tb.cpp`
**Implemented.** Loads the sidecar JSON (`argv[1]`), reads golden input/output vectors from the paths it references, applies reset for five cycles, and then runs a single unified interleaved drive/sample loop per vector:

- On each tick: first, if `valid_out` is asserted, sample `data_out` as the current expected output index and advance; second, if the DUT's `ready_in` is high and inputs remain, drive `data_in` / `valid_in`; finally, `tickClock()` unconditionally — including on the cycle that just sampled a vector's final output, to avoid bleeding `valid_out` / `data_out` state into the next vector.
- Per-vector state (`input_idx`, `output_idx`, `idle_cycles`) is scoped to the vector, so every vector samples starting at output 0.
- Hang budget is `pipeline_latency_cycles * 4 + 16`. Exceeding it throws with the stuck vector / output index and the cycle count.

Handshake semantics documented in the file header and enforced against the sidecar: `valid_in` / `data_in` are bench-driven inputs; `ready_in` is a DUT output (upstream backpressure); `valid_out` / `data_out` are DUT outputs and `data_out` is sampled only when `valid_out == 1`.

Build-time contract: the DUT is selected via preprocessor macros `VMODEL_HEADER` (path to the generated Verilator header) and `VMODEL_CLASS` (the Verilator-generated class name). `run_verilator` sets both when it invokes `verilator --cc --exe --build`.

Run-time contract: the DUT must use canonical port names `clk`, `rst_n`, `valid_in`, `ready_in`, `data_in`, `valid_out`, `data_out`. `requireCanonicalSignals()` validates all seven sidecar fields against the literal names before the first tick; a mismatch throws immediately.

Numerical pass threshold is `max_error <= 3`. Timing pass requires `timing_actual_cycles == pipeline_latency_cycles` exactly. `failure_class` is emitted as `null` — Assayer classifies after reading the results file.

Best-effort error path: if the testbench throws before finishing, a fallback writes a `fail`-shaped results JSON to the sidecar's `results_path` so `run_verilator` always has a result file to parse.

### `third_party/json.hpp`
Vendored [nlohmann/json v3.11.3](https://github.com/nlohmann/json), single-header. Keeps the C++ build self-contained.

### `README.md`
Per-directory reference documenting the build-time macros, run-time contract, sidecar schema, and results schema.

---

## Cross-Language Fixtures (`test/fixtures/`)

Shared fixtures used by both vitest suites and pytest. Today:

- `passthrough.v` — minimal Verilog DUT that honors the canonical port contract (`clk`, `rst_n`, `valid_in`, `ready_in` tied high, `data_in`, `valid_out`, `data_out`). Round-trips inputs unchanged so tests can verify the bench's happy path.
- `broken_module.v` — a deliberately broken module used to exercise the failure path.
- `pipeline_ir.json` — a minimal `PipelineIR` snapshot that satisfies `pipelineIrSchema` (one conv2d layer, all seven canonical signals, 1×1×1×1 tensors).
- `verilog_module.json` — a minimal `VerilogModule` JSON.
- `verif_pass.json`, `verif_fail.json` — sample `VerifResult` payloads for both outcomes.
- `verilator/stream_passthrough.v`, `stream_offset.v`, `stream_latency2.v`, `stream_stall.v`, `stream_bubble.v` — real DUT fixtures for the full MCP integration suite, covering happy-path streaming, numerical mismatch, exact-latency mismatch, `ready_in` backpressure, and `valid_out` bubbles.

Actual suites now live in `sdk/test/`, `mcp/test/`, and `scripts/test_*.py`. The fast path is mostly mocked and deterministic; the full path exercises real `iverilog`, `verilator`, `yosys`, and the toy Python extraction flow.

---

## Runtime Output Layout (`output/`)

All runtime artifacts live here. The four subdirectories (`rtl/`, `tb/`, `reports/`, `weights/`) exist on disk but are empty and git-ignored. `ensureOutputLayout()` in `sdk/orchestrate.ts` re-creates them at every pipeline start; nothing under `output/` is source-controlled.

- `output/rtl/<module_id>.v` — Foundry / Surgeon output, written exclusively through `write_verilog`.
- `output/rtl/<module_id>.meta.json` — sidecar metadata for each generated module.
- `output/tb/<module_id>.sidecar.json` — Assayer's per-run sidecar for the static testbench.
- `output/weights/<module_id>_weights.hex`, `<module_id>_bias.hex` — Cartographer's hex-format tensors.
- `output/reports/run_log.jsonl` — JSONL event stream for the full run, produced by `appendRunLog`.
- `output/reports/<module_id>.yosys.json` — Yosys synthesis report per passing module.
- `output/reports/pipeline_summary.json` — final summary including total cost and model usage.
- `output/layer_ir.json` — Cartographer's canonical `PipelineIR`, Zod-validated on load.
- `output/pipeline_state.json` — authoritative `PipelineState`, updated after every transition, Zod-validated (with `superRefine` cross-field checks) on resume.
- `output/golden_vectors.json` — raw Python output before it is promoted to `layer_ir.json`.

---

## Runtime Validation Policy

Every JSON trust boundary is Zod-validated. The same schemas drive both local validation and the JSON Schema advertised to the Claude Agent SDK (`outputFormat`) and the MCP server (`inputSchema` / `outputSchema`), via `z.toJSONSchema()`.

| Boundary | Where validated | Schema |
|---|---|---|
| MCP tool arguments | `mcp/server.ts` handlers | `mcp/schemas.ts` per-tool input schemas |
| MCP tool outputs advertised to clients | `mcp/server.ts` tool definitions | `mcp/schemas.ts` (generated JSON Schema) |
| Agent `structured_output` | `sdk/orchestrate.ts` `requireStructuredOutput` | `sdk/schemas.ts` (per agent) |
| `run_verilator` sidecar read | `mcp/tools.ts` `readSidecarIfPresent` | `verificationSidecarSchema` |
| `read_weights` PipelineIR read | `mcp/tools.ts` `read_weights` | `pipelineIrSchema` |
| `output/layer_ir.json` on load | `sdk/orchestrate.ts` `ensureLayerIr` | `pipelineIrSchema` |
| `output/rtl/<id>.meta.json` on load | `sdk/orchestrate.ts` `loadPersistedVerilogModule` | `verilogModuleSchema` |
| `output/pipeline_state.json` on resume | `sdk/pipeline.ts` `loadState` | `pipelineStateSchema` (incl. `superRefine`) |
| Yosys direct MCP result | `sdk/orchestrate.ts` `invokeYosys` | `synthesisReportSchema` |

Validation failures throw with field-level error paths so corrupted artifacts and malformed agent outputs fail loudly instead of silently propagating.

---

## Outstanding TODOs

Entries reference exact file and line of the current work. In future revisions, completed entries should move to **Done** with a description of the implementation.

### Critical Path (blocks the intended thesis-grade ResNet-50 pipeline)

These require external tools and ML libraries; they cannot be completed in pure TypeScript.

- **`scripts/quantize_model.py` — replace the toy checkpoint flow with the real PTQ path**: load torchvision ResNet-50, run calibration, convert with `torch.quantization`, persist the INT8 checkpoint, and print real per-layer scale factors as JSON. Today the script intentionally writes a deterministic toy checkpoint so the automated suite stays local and reproducible.
- **`scripts/generate_golden.py` — replace the toy golden-vector flow with real residual-block capture**: load the quantized checkpoint, trace via `torch.fx`, fold batch normalization into preceding convolutions, serialize real weights and biases to `$readmemh`-compatible hex files, and emit a full `PipelineIR` for the target model. Today the script intentionally emits a deterministic one-layer toy `PipelineIR`.
- **Manual smoke test on the intended checkpoint**: the automated suite covers the toy model end-to-end, but the final thesis path still needs an opt-in smoke command against the actual ResNet-50 checkpoint and artifact set.

### Blocked on External Dependencies

- **`sdk/claude-agent-sdk-compat.ts`**: remove the compatibility shim once `@anthropic-ai/claude-agent-sdk` ships a valid root declaration file.
- **`sdk/orchestrate.ts:267–268`**: restore `AgentDefinition.skills` and `AgentDefinition.maxTurns` once the published SDK typings expose them; today the parent query's `maxTurns: 6` is the only guardrail.

### Design Decisions Deferred

Intentional "maybe later" notes, not bugs.

- **`sdk/orchestrate.ts:258`**: replace the hand-rolled frontmatter parser with a real YAML parser (e.g. `yaml` or `gray-matter`) if plugin frontmatter grows more expressive.
- **`sdk/orchestrate.ts:872`**: promote Assayer into a first-class `tick()` action if the state machine grows beyond the current Foundry-or-Surgeon binary choice.
- **`sdk/orchestrate.ts:895`**: extend CLI argument parsing when the pipeline grows knobs — alternate plugins, custom output roots, per-run retry budgets.
- **Dispatch the Conductor agent (or remove it)**: `nn2rtl-plugin/agents/conductor.md` is loaded by `loadAllAgentDefinitions` but never invoked. Either wire it into the orchestration path as an optional agentic-mode switch, or drop the agent file so the plugin matches the deterministic TypeScript orchestration that actually runs.

### Done (removed from the outstanding list)

For the record, these items from earlier revisions are complete:

- Inline `weight_int8` tensor in `LayerIR` replaced with `weights_path` / `bias_path` plus hex-file convention.
- `LayerIR` extended with timing, width, and signal-name metadata; all seven signal names typed as string literals.
- `VerifResult` extended with `timing_pass`, `timing_actual_cycles`, `timing_expected_cycles`, `failure_class`.
- `PipelineState` extended with `total_cost_usd` and `model_usage`, both tracked across agent calls, with cross-field invariants enforced via `superRefine`.
- Foundry and Cartographer system prompts rewritten to require `$readmemh`, valid/ready streaming, exact pipeline latency, and canonical signal-name literal constants.
- Assayer system prompt rewritten to generate the JSON sidecar and surface timing results.
- Stale TODO comments on `run_iverilog`, `run_yosys`, and `write_verilog` removed — those tools are implemented.
- Zod runtime validation added at every JSON trust boundary, including MCP tool inputs/outputs, agent structured outputs, `read_weights` PipelineIR read, `run_verilator` sidecar read, and the resume state file.
- JSON-Schema blobs deduplicated: `sdk/orchestrate.ts` and `mcp/server.ts` now derive every advertised schema from the Zod exports via `z.toJSONSchema()`.
- Static Verilator testbench (`tb/static_verilator_tb.cpp`) implemented: sidecar parsing, seven-name canonical enforcement, ready/backpressure-aware drive loop, per-vector sampling that correctly ticks past the final output, hang detection, cycle-accurate timing, numerical tolerance check, structured results JSON, best-effort error propagation. `nlohmann/json` vendored at `tb/third_party/json.hpp`.
- `run_verilator` fully wired end-to-end: copies the static testbench into a tempdir with preserved `third_party/json.hpp` layout, invokes `verilator --cc --exe --build` with `VMODEL_HEADER` / `VMODEL_CLASS` macros, runs the produced binary against the sidecar, reads and Zod-validates the results JSON, and maps build / execution failures to properly shaped `VerifResult`s.
- `PipelineStateManager.loadState` now recovers transient `generating` / `verifying` statuses from a crashed prior run, correctly rolling back the attempts counter for Surgeon-path crashes so the retry budget is not over-billed.
- Entry points separated: `sdk/main.ts` and `mcp/main.ts` are the runnable CLIs, leaving `orchestrate.ts` / `server.ts` as library code importable by tests.
- Dependency injection seams in place: `OrchestratorRuntime` (`now`, `queryFn`) in the SDK, `ToolsRuntime` (`commandRunner`, `cwd`, `env`, `tmpDirRoot`) in the MCP tools, `ToolImplementations` in the MCP server — all testable without touching the real toolchain, real clock, or real Claude API.
- Test infrastructure implemented: root `package.json` with `test:fast` / `test:full` / `coverage`, `pytest.ini` with markers, per-package `vitest.config.ts` with coverage thresholds, SDK and MCP test suites, Python helper tests, and shared fixtures under `test/fixtures/`.
- Python preprocessing split into importable `*_impl.py` helpers (`paths.py`, `quantize_impl.py`, `golden_impl.py`), with a deterministic toy checkpoint / golden-vector flow that exercises checkpoint validation, BN folding, hex serialization, and `PipelineIR` emission end-to-end.
- Yosys policy decided and implemented: a failed post-pass Yosys synthesis report now feeds back into the retry loop as `failure_class: "synthesis_failed"` instead of being recorded as a degraded-but-still-passing module.

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
