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
├── scripts/                   # Python frontend prep (quantization + layer-IR / smoke harness)
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
Ignores `node_modules/`, `dist/`, `*.tsbuildinfo`, log files, IDE dirs, simulator artifacts (`obj_dir/`, `*.vcd`, `*.fst`, `*.vvp`, `*.blif`), Python virtualenvs (`.venv/`, `venv/`, `*.egg-info/`), test/coverage caches (`.pytest_cache/`, `.coverage*`, `coverage/`, `htmlcov/`, `.mypy_cache/`, `.ruff_cache/`), local Codex state, runtime outputs under `output/`, `checkpoints/`, `__pycache__/`, `.env*`, and OS junk.

### `.claude/settings.json`
Workspace permissions for Claude Code. Permits Bash invocations for `node`, `npm`, `python3`, `iverilog`, `verilator`, and `yosys`.

---

## Layer 1 — Claude Code Plugin (`nn2rtl-plugin/`)

Defines the four specialized agents (Cartographer, Foundry, Assayer, Surgeon) and their supporting skill documentation. The pipeline-coordinator role is played by the deterministic TypeScript orchestrator in `sdk/orchestrate.ts`, not by an LLM agent. Loaded by the SDK orchestrator via `plugins: [{ type: "local", path: pluginPath }]`.

### `.claude-plugin/plugin.json`
Plugin manifest. Declares name, version, and paths to agent, skill, and MCP config directories.

### `.mcp.json`
MCP server registration. Points to `../mcp/dist/server.js` with `OUTPUT_DIR=../output` and registers the server as `nn2rtl-tools`. Every MCP tool name is therefore prefixed `mcp__nn2rtl-tools__` in `allowedTools`.

### `agents/cartographer.md`
Model extractor. Model: `sonnet`. Runs once. Loads the quantized PyTorch checkpoint, traces via `torch.fx`, folds batch normalization into convolutions, writes weight/bias `.hex` files to `output/weights/`, emits `output/layer_ir.json`. The JSON schema enforces that signal-name fields are emitted as the canonical literals (`"clk"`, `"rst_n"`, `"valid_in"`, `"valid_out"`, `"ready_in"`, `"data_in"`, `"data_out"`).

### `agents/foundry.md`
Primary Verilog generator. Model: `sonnet`. Receives one `LayerIR`, produces one synthesizable `VerilogModule`. Hard rules: INT8 fixed-point, `8×8 → 16-bit` multipliers, signed datapath, saturating residual adds, `$readmemh` for weights, exact `pipeline_latency_cycles` from first `valid_in` to first `valid_out`, no simulation-only constructs. Canonical port names are mandatory (`clk`, `rst_n`, `valid_in`, `ready_in`, `data_in`, `valid_out`, `data_out`); `ready_in` is a module **output** (upstream backpressure) that may be tied high if stalling is not needed. Must call `write_verilog` to persist.

### `agents/assayer.md`
Simulation runner. Model: `haiku`. Generates the JSON sidecar consumed by the static testbench, runs `iverilog` for syntax, calls the `run_verilator` MCP tool, returns a structured `VerifResult` with timing fields and a classified `failure_class` on functional failure. Never writes source files (`disallowedTools: Write, Edit`).

### `agents/surgeon.md`
Targeted repair specialist. Model: `opus`. Activated on failure. Receives the broken module, the `VerifResult`, and the original `LayerIR`. Must classify the failure into one of the 16 taxonomy classes, locate the exact faulty lines, and rewrite only those lines while preserving the module's port interface. Capped at 3 retries per module.

### `skills/{cartographer,foundry,assayer,surgeon}/SKILL.md`
Supplemental skill reference material loaded alongside the matching agent prompt. The orchestrator concatenates the skill markdown body onto the agent prompt at load time (see `loadPluginAgentDefinition` in `sdk/orchestrate.ts`). Contains schema reminders, RTL patterns, and canonical-signal-name reminders.

---

## Layer 2 — TypeScript Orchestrator (`sdk/`)

The deterministic control plane. Not a prompt — a real state machine that reads state from disk, decides the next action, dispatches agents via the Claude Agent SDK's `query()`, validates their structured outputs through Zod, and updates state on disk after every transition. Resumable, auditable, measurable.

### `package.json`
Dependencies: `@anthropic-ai/claude-agent-sdk`, `zod`. Dev: `@types/node`, `tsx`, `typescript`, `vitest`, `@vitest/coverage-v8`. Scripts: `build`, `typecheck`, `start` (runs `dist/main.js`), `dev` (watches `main.ts`), `pipeline`, `test`, `test:fast`, `test:full`, `coverage`.

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
`AGENT_CONFIG` maps each of the four LLM subagents (Cartographer, Foundry, Assayer, Surgeon) to its model tier, per-agent `maxTurns`, and description. `PIPELINE_CONFIG` pins `max_retries`, all output paths, and the path to the static testbench. Single point of change for model assignments and paths.

### `claude-agent-sdk-compat.ts`
Deliberately narrowed facade over `@anthropic-ai/claude-agent-sdk`. The published SDK's `AgentDefinition`, `SDKMessage`, and `query()` option types are large, include fields we do not use (`mcpServers`, `source`, `criticalSystemReminder_EXPERIMENTAL`, various policy-settings knobs), and `model` is typed as `string` so any model name compiles. This shim re-exports only the fields the orchestrator actually touches (`description`, `prompt`, `tools`, `disallowedTools`, `model`, `skills`, `maxTurns` on `AgentDefinition`; `cwd`, `tools`, `allowedTools`, `plugins`, `agents`, `outputFormat`, `maxTurns` on the `query()` options) and narrows `model` to the `"sonnet" | "opus" | "haiku" | "inherit"` union we support. The runtime `query` function is cast back to this narrower signature via `as unknown as ...` — since the SDK runtime accepts any superset of our shape, the cast is sound. Upside: adding a field to the SDK's types does not silently change our API, and unsupported fields fail at compile time instead of at runtime. The SDK itself is now fully typed, so the shim is no longer a declarations workaround; removing it would mean accepting the SDK's wider surface, which we do not want.

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

**Runtime injection** via `OrchestratorRuntime = { now, queryFn, yosysFn }`: every helper accepts either a full or partial runtime so tests can supply deterministic clocks, a mock `query()` implementation, and a mock Yosys invocation. The default is `{ now: () => new Date(), queryFn: query, yosysFn: invokeYosys }` where `invokeYosys` dynamically imports `run_yosys` from the MCP package and validates the report against `synthesisReportSchema`.

Agent dispatch goes through `runDelegatedAgent(slug, payload, outputFormat, resultSchema, runtime)`. Both the SDK `outputFormat` (JSON Schema, generated from Zod via `z.toJSONSchema`) and the local `resultSchema` (Zod) are derived from the same schema export, so drift is structurally impossible.

`invokeFoundry()` and `invokeSurgeon()` also call `persistVerilogModule()` after a successful structured return. This is a defensive fallback: agents are still instructed to use `write_verilog`, but the orchestrator force-persists the returned `{ module_id, verilog_source, ... }` payload so disk state is not lost if a model skips the MCP tool call to save turns.

`requireStructuredOutput` unwraps the SDK's `structured_output`, falls back to `JSON.parse(result.result)` if missing, validates through the supplied Zod schema, and throws a field-level error on mismatch.

Cost tracking accumulates over every agent call including Cartographer's bootstrap, Foundry, Assayer, Surgeon, and direct Yosys invocations.

---

## Layer 3 — MCP Server (`mcp/`)

The boundary between agents and the external Verilog toolchain. Five tools, a stdio transport, strict input validation.

### `package.json`
Dependencies: `@modelcontextprotocol/sdk`, `zod`. Dev: `@types/node`, `tsx`, `typescript`, `vitest`, `@vitest/coverage-v8`. Script shape mirrors the SDK package, with temp-dir environment overrides on the Vitest commands used in this workspace. Compiled output `dist/main.js` is the CLI entry; `dist/server.js` is what `.mcp.json` points to.

### `package-lock.json`
npm lockfile; committed for reproducibility.

### `tsconfig.json`
Mirrors the SDK config.

### `vitest.config.ts`
Same shape as the SDK's. Coverage includes `schemas.ts`, `server.ts`, `tools.ts`; excludes `main.ts`, `types.ts`, build output. Thresholds: 95% branches, 100% lines/functions/statements.

### `main.ts` *(entry point)*
Tiny wrapper that calls `startServer()` from `server.ts` and logs fatal errors. Separation of concerns: keeps `server.ts` testable without side effects.

### `types.ts`
Byte-for-byte mirror of `sdk/types.ts`. The two packages have separate `rootDir`s so neither can import from the other; each keeps a local copy of the shared data contracts. Drift is prevented by `scripts/check-twins.mjs`, which runs as the first step of `npm run test:fast` / `test:full` / `coverage` and exits non-zero if the files diverge (also enforces the shared Zod exports between the two `schemas.ts` files). Treat these files as one logical file in two places, and rely on the twin check to catch accidental single-sided edits.

### `schemas.ts`
Single source of truth for MCP-side schemas. Exports:

- Shared data contracts: `failureClassSchema`, `layerIrSchema` (with all seven canonical signal literals), `pipelineIrSchema`, `verilogModuleSchema`, `verifResultSchema`, `verificationSidecarSchema` (with all seven canonical signal literals).
- Per-tool input schemas: `runIverilogInput`, `runVerilatorInput`, `runYosysInput`, `readWeightsInput`, `writeVerilogInput`.
- Per-tool output schemas: `runIverilogOutput`, `runYosysOutput`, `writeVerilogOutput`.

`server.ts` advertises each tool's `inputSchema` and `outputSchema` via `z.toJSONSchema(...)` from these Zod definitions — no hand-written JSON Schema.

### `tools.ts`
Tool implementations. Each exposes a `ToolsRuntime` override parameter so tests can supply a mock `commandRunner`, `cwd`, `env`, and `tmpDirRoot`. The default runtime uses `execFile`, the repo root as `cwd`, `process.env`, and a writable temp root chosen by `resolveTmpDirRoot()` (`os.tmpdir()` on Windows, otherwise an absolute `TMPDIR` or `/tmp`). It also exposes platform-aware `VERILATOR_COMMAND` and `PYTHON_COMMAND` constants so Windows callers can avoid the broken Perl/MS Store launcher paths that frequently show up in default installs. It does not set an explicit child-process timeout; that is intentional because real Verilator builds can take minutes. If a timeout is added later, production runs should use a generous ceiling (roughly 10 minutes, not smoke-test scale).

Exports: `CommandRunner`, `ToolsRuntime`, `createToolsRuntime`, `withTempDir`, `stderrFromUnknown`, `parseYosysReport`, `resolveOutputRoot`, `resolveRepoRootFromEnv`, `TB_SOURCE_PATH`, `TB_JSON_HPP_PATH`, `VERILATOR_COMMAND`, `PYTHON_COMMAND`, `run_iverilog`, `run_verilator`, `run_yosys`, `read_weights`, `write_verilog`, `readSidecarIfPresent`.

- `run_iverilog(verilog_source, module_name)` — writes the source to a temp file, runs `iverilog -o <os.devNull> -g2012`, returns `{ success, stderr }`. **Implemented.**
- `run_verilator(verilog_source, module_name, sidecar_path)` — loads and validates the sidecar via `readSidecarIfPresent` (Zod-checked), rejects relative `golden_inputs_path` / `golden_outputs_path` / `results_path`, copies the static testbench plus vendored `third_party/json.hpp` into a temp build dir, invokes `VERILATOR_COMMAND --cc --exe --build` with `VMODEL_HEADER` / `VMODEL_CLASS`, runs the produced binary with the sidecar path, reads `sidecar.results_path`, validates it through `verifResultSchema`, and maps build / execution failures into well-formed `VerifResult` payloads. **Implemented.**
- `run_yosys(verilog_source, module_name)` — runs `yosys -p "synth_ice40 -abc9; stat"`, uses the exported `parseYosysReport` helper to extract LUT count and an `MHz` figure, returns `{ success, lut_count, fmax_mhz, report }`. **Implemented.**
- `read_weights(checkpoint_path, quantization_config)` — spawns `PYTHON_COMMAND scripts/generate_golden.py`, reads `output/golden_vectors.json`, and validates it against `pipelineIrSchema` before returning. `generate_golden.py` now writes the canonical artifact to `output/layer_ir.json` and mirrors the same JSON to `output/golden_vectors.json` for MCP compatibility, so `read_weights` still sees a valid `PipelineIR` without any TypeScript changes.
- `write_verilog(module, output_dir)` — the persistence path agents are expected to use. Writes `<output_dir>/rtl/<module_id>.v` and `<module_id>.meta.json`, returns the absolute `.v` path. The orchestrator also has a `persistVerilogModule()` safety net in `sdk/orchestrate.ts` for cases where an agent returns valid structured RTL but skipped the MCP tool call. **Implemented.**
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

Single-use utilities the human runs once before the autonomous pipeline. Split into thin CLI wrappers plus importable `*_impl.py` modules so pytest can exercise the helpers while keeping the local test flow deterministic. `prepare_pipeline.py` is the top-level no-argument smoke harness for the full frontend.

### `__init__.py`
Makes `scripts/` an importable package so `pytest` (and future tests) can `from scripts.golden_impl import ...`.

### `paths.py`
`detect_repo_root(current_file)` — resolves the repo root, honoring a `NN2RTL_REPO_ROOT` env override so tests can point the scripts at a temp dir.

### `quantize_impl.py`
Importable helpers for the ResNet-50 PTQ checkpoint flow. Core pieces:

- `resolve_checkpoint_path(...)`, `get_quantized_checkpoint_path(...)` — canonical checkpoint path resolution.
- `build_resnet50_quantized_checkpoint(...)` — loads torchvision ResNet-50, seeds PyTorch with `0`, runs deterministic synthetic calibration on 32 random `1×3×224×224` tensors, folds batch norm into conv parameters, and exports the fused stem conv plus the three `layer1` bottlenecks as a `format_version: 2` checkpoint with stable `layer0_0_conv1` / `layer1_<block>_<op>` module IDs.
- Quantization is symmetric INT8 per tensor (`zero_point = 0`, `scale = max(|w|) / 127`) with INT32 bias export derived from the observed activation range during calibration.
- `write_quantized_checkpoint(...)`, `load_quantized_checkpoint(...)` — persist and validate the new flattened v2 checkpoint schema. Validation is eager and raises `CheckpointValidationError` on malformed metadata.
- Legacy helpers (`build_toy_quantized_checkpoint(...)`, `ToyPointwiseModel`, `create_toy_model(...)`, `run_toy_model(...)`) are still present for older local fixtures and compatibility tests.
- `build_quantization_summary(...)` — machine-readable summary emitted by the CLI.

The current export scope is intentionally constrained to the initial stem convolution plus `layer1` (16 modules total) so the downstream RTL path can be exercised incrementally. The CLI and summary text explicitly note that real users should swap the synthetic calibration tensors for ImageNet samples.

### `quantize_model.py`
Thin CLI wrapper over `quantize_impl.py`. The CLI shape is unchanged: `python scripts/quantize_model.py [checkpoint_path]`. It now writes a real ResNet-50 INT8 checkpoint to `checkpoints/resnet50_int8.pth` by default and prints a JSON summary describing the constrained export scope plus per-layer scale metadata.

### `golden_impl.py`
Importable helpers for the golden-vector / layer-IR extraction flow. Core pieces:

- `get_output_paths(...)`, `get_weight_artifact_paths(...)` — canonical output/weights layout.
- `int8_to_hex(...)`, `int32_to_hex(...)`, `write_signed_int8_hex(...)`, `write_signed_int32_hex(...)` — `$readmemh`-compatible signed INT8/INT32 serialization.
- `fold_batch_norm_into_conv(...)` — folds BN parameters into convolution weights/bias.
- `CheckpointResidualStack`, `ResidualStackTracer`, `ActivationCaptureInterpreter` — rebuild or load the residual stack, preserve module IDs through `torch.fx`, and capture per-node activations.
- `build_deterministic_input_stream(...)`, `capture_golden_outputs(...)` — generate the required 8 seeded INT8 vectors and record golden inputs/outputs per traced node in topological order.
- `build_pipeline_ir_payload(...)`, `validate_pipeline_ir_payload(...)`, `write_pipeline_ir(...)` — produce a schema-shaped `PipelineIR`, write canonical `output/layer_ir.json`, and mirror it to `output/golden_vectors.json` for existing MCP consumers.
- Format-version-2 checkpoints now have two paths:
  - Flat PTQ checkpoints from `quantize_model.py` are bridged directly into `PipelineIR` by writing the stored INT8/INT32 tensors back out to `output/weights/`.
  - Richer residual-stack checkpoints that carry graph/module information still go through the existing `torch.fx` activation-capture path.
- Legacy format-version-1 toy checkpoints are still supported so the older local tests and fixtures continue to work.

### `generate_golden.py`
Thin CLI wrapper over `golden_impl.py`. It resolves the checkpoint path, generates `output/layer_ir.json`, mirrors the same JSON to `output/golden_vectors.json`, emits per-module weight/bias hex files under `output/weights/`, and prints a compact JSON summary (`status`, `model_name`, `num_layers`, `checkpoint_path`, `pipeline_ir_path`). The current path supports both the flat PTQ ResNet-50 checkpoint written by `quantize_model.py` and the richer format-version-2 residual-stack checkpoints used by the `torch.fx` tests, while retaining the legacy toy checkpoint fallback.

### `prepare_pipeline.py`
No-argument frontend smoke harness. It runs `quantize_model.py`, runs `generate_golden.py`, validates the resulting `output/layer_ir.json` against `pipelineIrSchema` from `mcp/schemas.ts` by shelling out to Node with `--experimental-strip-types`, prints a fixed-width summary table (`module_id | op_type | shape | num_weights | pipeline_latency_cycles`), and exits non-zero on the first failed step. The helper surface is intentionally small: subprocess chaining, TypeScript-schema validation, and table rendering.

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

Actual suites now live in `sdk/test/`, `mcp/test/`, and `scripts/test_*.py`. The fast path is mostly mocked and deterministic; the full path exercises real `iverilog`, `verilator`, `yosys`, and the Python frontend scripts, including the `prepare_pipeline.py` smoke harness and its failure-path coverage in `scripts/test_prepare_pipeline.py`.

---

## Runtime Output Layout (`output/`)

All runtime artifacts live here. The four subdirectories (`rtl/`, `tb/`, `reports/`, `weights/`) are kept in the repo with `.gitkeep` files so the layout survives a fresh clone; the generated contents inside them are git-ignored and may be present locally after any run. `ensureOutputLayout()` in `sdk/orchestrate.ts` re-creates the directory layout at every pipeline start.

- `output/rtl/<module_id>.v` — Foundry / Surgeon output, normally written through `write_verilog` and force-persisted by `persistVerilogModule()` as a safety net.
- `output/rtl/<module_id>.meta.json` — sidecar metadata for each generated module.
- `output/tb/<module_id>.sidecar.json` — Assayer's per-run sidecar for the static testbench.
- `output/weights/<module_id>_weights.hex`, `<module_id>_bias.hex` — Cartographer's hex-format tensors.
- `output/reports/run_log.jsonl` — JSONL event stream for the full run, produced by `appendRunLog`.
- `output/reports/<module_id>.yosys.json` — Yosys synthesis report per passing module.
- `output/reports/pipeline_summary.json` — final summary including total cost and model usage.
- `output/layer_ir.json` — Cartographer's canonical `PipelineIR`, Zod-validated on load. `scripts/generate_golden.py` now writes this file directly.
- `output/pipeline_state.json` — authoritative `PipelineState`, updated after every transition, Zod-validated (with `superRefine` cross-field checks) on resume.
- `output/golden_vectors.json` — compatibility mirror of `output/layer_ir.json` kept because `mcp/tools.ts` `read_weights` still reads this filename.

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

- **Extend the current PTQ export beyond stem + `layer1`**: the real torchvision ResNet-50 PTQ path is now implemented, but the checkpoint intentionally stops after the fused stem conv and first residual stack so the downstream RTL pipeline can expand in controlled increments.
- **Emit real activation traces for the flat v2 PTQ bridge**: `generate_golden.py` can already turn the flattened ResNet-50 checkpoint into a valid `PipelineIR`, but the direct bridge currently focuses on weight/bias artifact emission rather than per-layer captured `golden_inputs` / `golden_outputs`.
- **Manual smoke test on the intended checkpoint**: the automated suite now covers the deterministic frontend harness end-to-end, but the final thesis path still needs an opt-in smoke command against the actual ResNet-50 checkpoint and artifact set.

### Blocked on External Dependencies

- *(none currently)* — the earlier `claude-agent-sdk-compat.ts` entry has been removed. The shim is no longer a workaround; it is load-bearing by design (see the `claude-agent-sdk-compat.ts` section above for what it narrows and why), so it does not belong on a TODO list.

### Design Decisions Deferred

Intentional "maybe later" notes, not bugs.

- **Custom output roots / alternate plugin paths in the CLI**: `parseCliArgs` now accepts `--resume` and `--max-retries`; adding `--output-dir` or `--plugin` is still deferred because both currently come from `PIPELINE_CONFIG` / the plugin path constant in `orchestrate.ts`.

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
- Python preprocessing split into importable `*_impl.py` helpers (`paths.py`, `quantize_impl.py`, `golden_impl.py`), with a real ResNet-50 PTQ export path, a flat format-version-2 `PipelineIR` bridge, the richer `torch.fx` golden-capture path, and the older deterministic toy compatibility path.
- **Real PTQ path implemented**: `scripts/quantize_model.py` / `scripts/quantize_impl.py` now load torchvision ResNet-50, run deterministic synthetic calibration, fold BN into conv weights, quantize to symmetric INT8 per tensor, and persist the flattened `format_version: 2` checkpoint schema validated by `load_quantized_checkpoint`.
- Yosys policy decided and implemented: a failed post-pass Yosys synthesis report now feeds back into the retry loop as `failure_class: "synthesis_failed"` instead of being recorded as a degraded-but-still-passing module.
- **Fmax gate tightened**: `parseYosysReport` now extracts abc9's `Delay = X ns` / `Delay = X ps` lines in addition to explicit `MHz` numbers, and the Yosys invocation is `synth_ice40 -abc9 -top <module>; stat; tee -o /dev/stdout ltp -noff`. `evaluateSynthesis` no longer silently skips the timing gate when `fmax_mhz === 0` — that path now emits a `synthesis_failed` VerifResult with `fix_hint` explaining that timing could not be measured, so Surgeon is forced to address the issue instead of the gate being quietly bypassed.
- **Surgeon retry loop tested**: `test/orchestrate-flow.test.ts` now exercises two forced-failure yosys paths end-to-end — `fmax_mhz === 0` (synthesis_failed) and `fmax_mhz` below the PPA target (missing_pipeline_register). Both route through `applyVerifResult` into a Surgeon dispatch and a second Assayer + Yosys round. A new `yosysFn` runtime seam on `OrchestratorRuntime` makes these tests possible without invoking the real toolchain.
- **Frontmatter parser replaced**: the hand-rolled `---`-delimited parser in `orchestrate.ts` is gone; frontmatter now goes through the `yaml` package. A new `toStringList` helper accepts both CSV strings (`tools: Bash, Read`) and YAML lists (`tools: [Bash, Read]`), and the legacy `splitCsvField` export was deleted.
- **SDK typing hacks resolved**: `@anthropic-ai/claude-agent-sdk@0.2.107` exposes `AgentDefinition.skills` and `AgentDefinition.maxTurns`, so the local compat shim now advertises both fields and `loadPluginAgentDefinition` propagates them from `AGENT_CONFIG` (per-agent `maxTurns`) and from agent-markdown frontmatter (`skills`). The parent `query()` call keeps its own `maxTurns: 6` as an outer safety cap.
- **Conductor agent removed**: there is no LLM "Conductor" anymore. The deterministic TypeScript orchestrator in `sdk/orchestrate.ts` owns the role directly, and `nn2rtl-plugin/agents/conductor.md` plus `nn2rtl-plugin/skills/conductor/` were deleted. `AGENT_CONFIG` and `AGENT_SLUGS` now list only the four real subagents (Cartographer, Foundry, Assayer, Surgeon).
- **CLI knobs extended**: `parseCliArgs` accepts `--max-retries N` (with both `--max-retries N` and `--max-retries=N` forms) and rejects unknown `--flag` arguments with a clear error. `runPipeline` / `RunPipelineOptions` gained an optional `maxRetries` field that overrides `PIPELINE_CONFIG.max_retries`. Unknown flags now error instead of being silently ignored.
- **Unreachable Assayer action comment cleaned up**: `runPipeline`'s fall-through `throw` no longer carries a stale "TODO: first-class Assayer action" note; the comment now correctly documents that the branch is a defensive guard for a future `PipelineStateManager.tick()` action type.
- **Golden generation widened for v2 checkpoints**: `scripts/golden_impl.py` and `scripts/generate_golden.py` now support both flattened PTQ checkpoints from `quantize_model.py` and richer format-version-2 residual-stack checkpoints. The flat bridge writes stored INT8/INT32 tensors back out to hex and emits strict `PipelineIR` under `output/layer_ir.json`, while the `torch.fx` path still handles graph-backed activation capture. `output/golden_vectors.json` remains a compatibility mirror for MCP `read_weights`.
- **Frontend smoke harness added**: `scripts/prepare_pipeline.py` now chains quantization, golden generation, TypeScript-schema validation of `output/layer_ir.json`, and a human-readable pipeline summary table. `scripts/test_prepare_pipeline.py` covers the success path, missing-checkpoint failure, and schema-invalid revalidation path.

---

## Working Rules Summary

From [CLAUDE.md](./CLAUDE.md), reiterated here for convenience:

- Never write `.v` files directly — always use `write_verilog`.
- Before any pipeline run: `npm run typecheck` in both `sdk/` and `mcp/`.
- Before the first pipeline run ever: run `scripts/prepare_pipeline.py` (or equivalently `scripts/quantize_model.py`, then `scripts/generate_golden.py`).
- The SDK package is `@anthropic-ai/claude-agent-sdk`, not `@anthropic-ai/claude-code`.
- The static Verilator testbench at `tb/static_verilator_tb.cpp` is handwritten infrastructure; never let an agent regenerate it.
- Keep `output/pipeline_state.json` updated after every state transition so the pipeline is resumable.
- Treat everything under `output/` as runtime output, not source.
- Preserve plugin layout: only `plugin.json` lives inside `nn2rtl-plugin/.claude-plugin/`; agents, skills, hooks, and `.mcp.json` live at the plugin root.
