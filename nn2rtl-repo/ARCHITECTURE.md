# nn2rtl — Architecture Reference

Last updated: April 16, 2026

This document is a file-by-file tour of the codebase. It complements [README.md](./README.md) (the design spec — *what the project is and why*) by describing *where each decision lives in the code* and *what is still outstanding*.

Read the README first. Then use this document to find the code that implements a specific part of the spec, or to see what is still a placeholder.

---

## Top-Level Layout

```
nn2rtl-repo/
├── README.md                  # Canonical design specification
├── CLAUDE.md                  # Operational rules for working in the repo
├── ARCHITECTURE.md            # This file — file-level reference
├── package.json               # Monorepo contract-check, test, and coverage scripts
├── pytest.ini                 # Python test config (markers, default flags)
├── .gitignore
├── .claude/settings.json      # Claude Code workspace permissions
│
├── nn2rtl-plugin/             # Layer 1: Claude Code plugin (agent roles + skills)
├── sdk/                       # Layer 2: TypeScript orchestrator
├── mcp/                       # Layer 3: MCP server exposing hardware toolchain
├── scripts/                   # Frontend prep plus shared maintenance checks
├── tb/                        # Static C++ Verilator testbench
├── vendor/sky130/             # Sky130 standard-cell library + download script
├── test/fixtures/             # Cross-language fixtures used by Python & TS tests
└── output/                    # Runtime artifacts (git-ignored; empty subdirs kept)
```

The three layers map directly to the README's architecture section: agents in the plugin layer, deterministic orchestration in `sdk/`, hardware-tool bridge in `mcp/`.

---

## Root-Level Files

### `README.md`
The design-spec document. Describes the research thesis, intended scope, architectural choices (weights via `$readmemh`, pipelined modules with timing contracts, static testbench), the agent roster, pipeline flow, data contracts, verification strategy, failure-mode taxonomy, and known risks. Treat it as the target architecture; use this file for the current implementation status when the code is ahead of or behind the original spec wording.

### `CLAUDE.md`
Tactical operational rules surfaced to Claude Code on every session start. Core rules: never write `.v` files directly; always run `npm run typecheck` before the pipeline; run the Python pre-processing scripts once before the first run; the SDK package is `@anthropic-ai/claude-agent-sdk`.

### `ARCHITECTURE.md`
This file.

### `package.json` *(root)*
Thin monorepo aggregator. No dependencies of its own. Exposes one shared contract check plus the cross-package test/coverage entrypoints:

- `check:twins` — runs `scripts/check-twins.mjs` to enforce that shared SDK/MCP types and shared Zod schema exports stay in sync.
- `test:fast` — `check:twins`, then vitest in `sdk/` and `mcp/`, then `pytest -m "not full"` (skip heavy markers).
- `test:full` — `check:twins`, then vitest in both packages, then `pytest -m "not manual"` (includes heavy tests, skips only opt-in manual smoke tests).
- `coverage` — `check:twins`, per-package vitest coverage, then `pytest -m "not manual" --cov=scripts --cov-branch --cov-fail-under=90`.

### `pytest.ini`
Default pytest options (`-ra` report) and registered markers: `full` (heavy / slow tests) and `manual` (opt-in smoke tests that require user-supplied external artifacts).

### `.gitignore`
Ignores `node_modules/`, `dist/`, `*.tsbuildinfo`, log files, IDE dirs, simulator artifacts (`obj_dir/`, `*.vcd`, `*.fst`, `*.vvp`, `*.blif`), Python virtualenvs (`.venv/`, `venv/`, `*.egg-info/`), test/coverage caches (`.pytest_cache/`, `.coverage*`, `coverage/`, `htmlcov/`, `.mypy_cache/`, `.ruff_cache/`), local Codex state, runtime outputs under `output/`, `checkpoints/`, `__pycache__/`, `.env*`, and OS junk.

### `.claude/settings.json`
Workspace permissions for Claude Code. Permits Bash invocations for `node`, `npm`, `python3`, `iverilog`, `verilator`, and `yosys`.

---

## Layer 1 — Claude Code Plugin (`nn2rtl-plugin/`)

Defines the three LLM agents (Cartographer, Foundry, Surgeon) and their supporting skill documentation. The pipeline-coordinator role and the verification (Assayer) role are both played by the deterministic TypeScript orchestrator in `sdk/orchestrate.ts`, not by LLM agents. Loaded by the SDK orchestrator via `plugins: [{ type: "local", path: pluginPath }]`.

### `.claude-plugin/plugin.json`
Plugin manifest. Declares name, version, and paths to agent, skill, and MCP config directories.

### `.mcp.json`
MCP server registration. Points to `../mcp/dist/server.js` with `OUTPUT_DIR=../output` and registers the server as `nn2rtl-tools`. Every MCP tool name is therefore prefixed `mcp__nn2rtl-tools__` in `allowedTools`.

### `agents/cartographer.md`
Model extractor. Model: `sonnet`. Runs once. The prompt tells Cartographer to invoke the `read_weights` MCP tool, and that tool delegates into the Python frontend (`scripts/generate_golden.py`) to load the quantized checkpoint, fold batch norm into convolutions, rebuild/trace the graph when available, write weight/bias `.hex` files to `output/weights/`, and emit a schema-valid `PipelineIR`. Cartographer then returns that `PipelineIR`, which the orchestrator persists to `output/layer_ir.json`. The JSON schema enforces that signal-name fields are emitted as the canonical literals (`"clk"`, `"rst_n"`, `"valid_in"`, `"valid_out"`, `"ready_in"`, `"data_in"`, `"data_out"`).

### `agents/foundry.md`
Primary Verilog generator. Model: `sonnet`. Receives one `LayerIR`, produces one synthesizable `VerilogModule`. Hard rules: INT8 fixed-point, `8×8 → 16-bit` multipliers, signed datapath, saturating residual adds, `$readmemh` for weights, exact `pipeline_latency_cycles` from first `valid_in` to first `valid_out`, no simulation-only constructs. Canonical port names are mandatory (`clk`, `rst_n`, `valid_in`, `ready_in`, `data_in`, `valid_out`, `data_out`); `ready_in` is a module **output** (upstream backpressure) that may be tied high if stalling is not needed. Must call `write_verilog` to persist.

**Wide-bus packed-channel interface.** `data_in` is a packed channel bus, not a scalar 8-bit port. For conv/relu, `data_in[i*8 +: 8]` is channel `i` of the current pixel and the port width must be `IC*8` bits. `data_out[i*8 +: 8]` is channel `i` of the emitted output and the port width must be `OC*8` bits. All channels of a pixel arrive in a single clock cycle. For `op_type=add`, `data_in[W-1:0] = lhs` and `data_in[2W-1:W] = rhs` where `W = input_width_bits / 2`.

**Output-stationary MAC array mandate.** Conv modules must instantiate `OC` parallel signed 8x8 MAC lanes, one accumulator per output channel, reused across input-channel × kernel-position cycles. `pipeline_latency_cycles = IC * KH * KW + 3` (fetch, multiply, accumulate, output buffer). Single-MAC designs are explicitly rejected. `ready_in` must deassert while the MAC array is running.

**Disallowed tools**: `Agent`, `Task` (no implicit Opus subagent spawning). `maxTurns: 20`.

### `agents/surgeon.md`
Targeted repair specialist. Model: `opus`. Activated on failure. Receives the broken module, the `VerifResult`, and the original `LayerIR`. Must classify the failure into one of the 17 taxonomy classes (the Verilator bench produces `status`, `timing_pass`, `expected`, `got`, `max_error`, and stderrs but does NOT emit `failure_class` — Surgeon owns that classification), locate the exact faulty lines, and rewrite only those lines while preserving the module's port interface. The prompt mandates terse output: "Output immediately. Do not explain." `maxTurns: 8`. Capped at 3 retries per module (via `PIPELINE_CONFIG.max_retries`).

**Disallowed tools**: `Agent`, `Task`.

### `skills/{cartographer,foundry,surgeon}/SKILL.md`
Supplemental skill reference material loaded alongside the matching agent prompt. The orchestrator concatenates the skill markdown body onto the agent prompt at load time (see `loadPluginAgentDefinition` in `sdk/orchestrate.ts`). Contains schema reminders, RTL patterns, and canonical-signal-name reminders.

**Note:** There is no `agents/assayer.md` and no `skills/assayer/` directory. Assayer was removed as an LLM agent and replaced with deterministic verification logic in `sdk/orchestrate.ts` (see `runAssayerDeterministic`).

---

## Layer 2 — TypeScript Orchestrator (`sdk/`)

The deterministic control plane. Not a prompt — a real state machine that reads state from disk, decides the next action, dispatches agents via the Claude Agent SDK's `query()`, validates their structured outputs through Zod, and updates state on disk after every transition. Resumable, auditable, measurable.

### `package.json`
Dependencies: `@anthropic-ai/claude-agent-sdk`, `yaml`, `zod`. Dev: `@types/node`, `cross-env`, `tsx`, `typescript`, `vitest`, `@vitest/coverage-v8`. Scripts: `build`, `typecheck`, `start` (runs `dist/main.js`), `dev` (watches `main.ts`), `pipeline`, `test`, `test:fast`, `test:full`, `coverage`. In this workspace, `test:fast` and `coverage` use `cross-env` to pin temp-dir environment variables for deterministic cross-platform test runs.

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

- `LayerIR` — master per-module spec. `module_id`, `op_type`, input/output shapes, `weights_path` / `bias_path` / `weight_shape` / `num_weights` (weights on disk, not inline), `scale_factor`, `zero_point`, timing contract (`pipeline_latency_cycles`, `clock_period_ns`), port widths (`input_width_bits = in_channels * 8` for conv/relu, `= in_channels * 16` for add; `output_width_bits = out_channels * 8`), **seven canonical signal names typed as string literals** (`"clk"`, `"rst_n"`, `"valid_in"`, `"valid_out"`, `"ready_in"`, `"data_in"`, `"data_out"`), and golden input/output vector file paths.
- `PipelineIR` — container for all LayerIRs plus model metadata.
- `VerilogModule` — what Foundry and Surgeon produce.
- `VerifResult` — what the deterministic Assayer function returns. Includes timing fields, numerical diagnostics, and optional `failure_class`.
- `VerificationSidecar` — the JSON blob the orchestrator writes for the static testbench to consume; includes all seven signal names and `bus_bytes_per_sample`.
- `PipelineState` — authoritative run state. Includes `total_cost_usd` and per-model `model_usage`.
- `ModuleStatus` and `NextAction` — discriminated unions for the state machine.
- `FailureClass` — the README's 16-category repair taxonomy plus `synthesis_failed`, used when a module passes simulation but fails the post-pass Yosys synthesis step.

### `schemas.ts`
Zod 4 runtime schemas mirroring `types.ts`. Single source of truth for every JSON Schema the SDK or MCP server advertises — those are now derived with `z.toJSONSchema(...)` rather than hand-written. Exports: `failureClassSchema`, `moduleStatusSchema`, `layerIrSchema`, `pipelineIrSchema`, `verilogModuleSchema`, `verifResultSchema`, `synthesisReportSchema`, `modelUsageEntrySchema`, `pipelineStateSchema`.

Constraint highlights:

- `layerIrSchema`: all seven signal-name fields are `z.literal(...)`; `attempt`, widths, latency are `int().positive()`.
- `synthesisReportSchema`: includes `area_um2` (defaults to 0 for non-Sky130 reports).
- `verilogModuleSchema.attempt`: `z.number().int().positive()` (Foundry starts at 1).
- `pipelineStateSchema`: `max_retries`, `attempts.value` are `int().nonnegative()`; `total_cost_usd` is `nonnegative()`. A `superRefine` enforces cross-field invariants:
  - Every `modules` key has an `attempts` entry and vice versa.
  - Every `results` key refers to a known module.
  - `fail_retry` requires a prior `VerifResult` and `attempts < max_retries`.
  - `fail_abort` requires a prior `VerifResult` and `attempts >= max_retries`.
  - `pass` requires a `VerifResult` whose own `status === "pass"`.
  - Every `results[id].module_id` must equal `id`.

### `config.ts`
`AGENT_CONFIG` maps each of the three LLM subagents (Cartographer, Foundry, Surgeon) to its model tier, per-agent `maxTurns`, and description. There is no Assayer entry — the orchestrator plays both the Conductor and Assayer roles itself via deterministic code. `PIPELINE_CONFIG` pins `max_retries`, all output paths, and the path to the static testbench. Single point of change for model assignments and paths.

Agent configuration:
- **Cartographer**: model `sonnet`, maxTurns 30
- **Foundry**: model `sonnet`, maxTurns 20
- **Surgeon**: model `opus`, maxTurns 8

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
Library module. Exports `runPipeline`, `runCli`, `handlePipelineError`, plus many helpers for tests (`AGENT_SLUGS`, `createOrchestratorRuntime`, `resolveFromSdk`, `normalizeAgentName`, `parseFrontmatter`, `toStringList`, `readText`, `pathExists`, `readJsonFile`, `writeJsonFile`, `appendRunLog`, `ensureOutputLayout`, `loadPluginAgentDefinition`, `buildDelegationPrompt`, `requireStructuredOutput`, `findLayer`, `loadPersistedVerilogModule`, `ensureLayerIr`, `writePipelineSummary`, `parseCliArgs`, `preflightVerilogModule`).

`AGENT_SLUGS` lists only three agents: `Cartographer`, `Foundry`, `Surgeon`.

`runPipeline(checkpointPath, options)` is the whole autonomous loop:

1. Ensure output directory layout exists.
2. If `output/layer_ir.json` is missing, dispatch Cartographer and persist its result.
3. Initialize `PipelineStateManager`, or resume from `output/pipeline_state.json` if `options.resume`.
4. Loop on `tick()`:
   - `invoke_foundry` → call Foundry for the current `LayerIR`, validate the returned `VerilogModule`, persist it, then invoke the deterministic Assayer on that module.
   - `invoke_surgeon` → load the persisted broken module plus the prior `VerifResult`, call Surgeon, validate/persist the repaired `VerilogModule`, then invoke the deterministic Assayer on the repaired module.
   - Every structured agent return is validated through Zod before it is trusted.
5. Feed the Assayer's `VerifResult` into `PipelineStateManager.applyVerifResult()`, save state, and append run-log events after each transition.
6. On a passing module, invoke `run_yosys` via direct MCP import (Sky130 flow) and write `output/reports/<module_id>.yosys.json`. If Yosys reports `success: false`, synthesize a `VerifResult` with `failure_class: "synthesis_failed"` and feed it back through `PipelineStateManager.applyVerifResult()`, so the module enters the same `fail_retry` / `fail_abort` path as any other failure.
7. On terminal state, write `output/reports/pipeline_summary.json`.

**Deterministic Assayer (`runAssayerDeterministic`).** Verification is no longer an LLM agent. The orchestrator writes the sidecar JSON from LayerIR fields (all signal names are fixed literals, widths and paths copied from the LayerIR), runs a deterministic RTL preflight that parses the ANSI port list to catch port-direction, port-width, and missing-port errors before invoking the toolchain, then calls `run_iverilog` for a fast lint pass followed by `run_verilator` for full simulation — both imported directly from the MCP package. The Verilator testbench produces a structured `VerifResult` JSON validated by Zod. No Haiku, no hallucination — a module has a `VerifResult` iff Verilator produced one.

The sidecar includes `bus_bytes_per_sample` (= `input_width_bits / 8`), which the testbench cross-checks against the NN2V binary header's `bytes_per_sample` field.

**Deterministic Yosys invocation (`invokeYosys`).** Yosys is also invoked via direct MCP import, not through an LLM mediator. The result is validated against `synthesisReportSchema`.

**RTL Preflight (`preflightVerilogModule`).** Before running any toolchain, the orchestrator parses the generated Verilog's ANSI port list and checks: (1) all seven canonical ports are present, (2) directions match the spec (e.g. `ready_in` must be `output`), (3) bus widths match `input_width_bits` / `output_width_bits` from the LayerIR. Preflight failures return a `VerifResult` with `failure_class: port_width_mismatch` and skip the iverilog/Verilator invocations entirely, saving minutes of build time.

**PPA gates (`evaluateSynthesis`).** After Yosys succeeds, `evaluateSynthesis` checks: (1) Fmax >= 50 MHz target (else `failure_class: missing_pipeline_register`), (2) `fmax_mhz > 0` (else `synthesis_failed` — timing could not be measured), (3) when no standard-cell area metric is present, LUT count <= 5000 per module. Under the current Sky130 flow, `lut_count` is a total standard-cell count proxy (from `stat -liberty`), not a real LUT count, so the 5k LUT ceiling is only enforced as a fallback. Reports larger than ~6 KB are summarized as head + ERROR lines + tail so Surgeon sees the actual diagnostics instead of warning spam.

**Runtime injection** via `OrchestratorRuntime = { now, queryFn, yosysFn, assayerFn }`: every helper accepts either a full or partial runtime so tests can supply deterministic clocks, a mock `query()` implementation, a mock Yosys invocation, and a mock Assayer function. The default is `{ now: () => new Date(), queryFn: query, yosysFn: invokeYosys, assayerFn: runAssayerDeterministic }`.

Agent dispatch goes through `runDelegatedAgent(slug, payload, outputFormat, resultSchema, runtime)`. Both the SDK `outputFormat` (JSON Schema, generated from Zod via `z.toJSONSchema`) and the local `resultSchema` (Zod) are derived from the same schema export, so drift is structurally impossible.

`invokeFoundry()` and `invokeSurgeon()` also call `persistVerilogModule()` after a successful structured return. This is a defensive fallback: agents are still instructed to use `write_verilog`, but the orchestrator force-persists the returned `{ module_id, verilog_source, ... }` payload so disk state is not lost if a model skips the MCP tool call to save turns.

`requireStructuredOutput` unwraps the SDK's `structured_output`, falls back to `JSON.parse(result.result)` if missing, validates through the supplied Zod schema, and throws a field-level error on mismatch.

Cost tracking accumulates over every agent call including Cartographer's bootstrap, Foundry, Surgeon, and direct Yosys invocations.

---

## Layer 3 — MCP Server (`mcp/`)

The boundary between agents and the external Verilog toolchain. Five tools, a stdio transport, strict input validation.

### `package.json`
Dependencies: `@modelcontextprotocol/sdk`, `zod`. Dev: `@types/node`, `cross-env`, `tsx`, `typescript`, `vitest`, `@vitest/coverage-v8`. Script shape mirrors the SDK package, with temp-dir environment overrides on the Vitest commands used in this workspace. Compiled output `dist/main.js` is the CLI entry; `dist/server.js` is what `.mcp.json` points to.

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

- Shared data contracts: `failureClassSchema`, `layerIrSchema` (with all seven canonical signal literals), `pipelineIrSchema`, `verilogModuleSchema`, `verifResultSchema`, `verificationSidecarSchema` (with all seven canonical signal literals plus `bus_bytes_per_sample`).
- Per-tool input schemas: `runIverilogInput`, `runVerilatorInput`, `runYosysInput` (includes optional `clock_period_ns`), `readWeightsInput`, `writeVerilogInput`.
- Per-tool output schemas: `runIverilogOutput`, `runYosysOutput` (includes `area_um2`), `writeVerilogOutput`.

`server.ts` advertises each tool's `inputSchema` and `outputSchema` via `z.toJSONSchema(...)` from these Zod definitions — no hand-written JSON Schema.

### `tools.ts`
Tool implementations. Each exposes a `ToolsRuntime` override parameter so tests can supply a mock `commandRunner`, `cwd`, `env`, and `tmpDirRoot`. The default runtime uses `execFile`, the repo root as `cwd`, `process.env`, and a writable temp root chosen by `resolveTmpDirRoot()` (`os.tmpdir()` on Windows, otherwise an absolute `TMPDIR` or `/tmp`). It also exposes platform-aware `VERILATOR_COMMAND` and `PYTHON_COMMAND` constants so Windows callers can avoid the broken Perl/MS Store launcher paths that frequently show up in default installs.

`isSystemSpawnError` distinguishes Node-level spawn failures (ENOENT, EACCES, EPERM, ENOMEM, ETIMEDOUT, EMFILE, ENFILE, `ERR_CHILD_PROCESS_STDIO_MAXBUFFER`, killed-by-signal) from tool-level exit-code errors. System spawn errors are thrown rather than laundered into Verilog syntax/synthesis failures, so Surgeon does not try to "fix" an out-of-memory error by rewriting correct code.

Exports: `CommandRunner`, `ToolsRuntime`, `createToolsRuntime`, `withTempDir`, `stderrFromUnknown`, `isSystemSpawnError`, `parseYosysReport`, `resolveOutputRoot`, `resolveRepoRootFromEnv`, `TB_SOURCE_PATH`, `TB_JSON_HPP_PATH`, `SKY130_LIB_PATH`, `VERILATOR_COMMAND`, `PYTHON_COMMAND`, `YOSYS_TIMEOUT_MS`, `YOSYS_MAX_BUFFER_BYTES`, `augmentEnvForOssCadSuite`, `augmentEnvForOssCadSuiteLibOnly`, `augmentEnvForVerilatorCxx`, `run_iverilog`, `run_verilator`, `run_yosys`, `read_weights`, `write_verilog`, `readSidecarIfPresent`.

- `run_iverilog(verilog_source, module_name)` — writes the source to a temp file, runs `iverilog -o <os.devNull> -g2012`, returns `{ success, stderr }`. System spawn errors are re-thrown. **Implemented.**
- `run_verilator(verilog_source, module_name, sidecar_path)` — loads and validates the sidecar via `readSidecarIfPresent` (Zod-checked), rejects relative `golden_inputs_path` / `golden_outputs_path` / `results_path`, copies the static testbench plus vendored `third_party/json.hpp` into a temp build dir, invokes `VERILATOR_COMMAND --cc --exe --build` with `VMODEL_HEADER` / `VMODEL_CLASS`, runs the produced binary with the sidecar path, reads `sidecar.results_path`, validates it through `verifResultSchema`, and maps build / execution failures into well-formed `VerifResult` payloads. **Implemented.**
- `run_yosys(verilog_source, module_name, clock_period_ns)` — **Sky130 standard-cell flow.** Runs `yosys -p "synth -top <module_name>; dfflibmap -liberty sky130.lib; abc -liberty sky130.lib [-constr ... -D <period_ps>]; stat -liberty sky130.lib"`. When `clock_period_ns > 0`, generates an ABC constraint file setting `set_driving_cell sky130_fd_sc_hd__buf_1` and `set_load 10.0`, and passes `-constr <file> -D <period_ps>` to `abc` to enable timing-aware mapping and `stime -p` critical-path reporting. The lib is `vendor/sky130/sky130_fd_sc_hd__tt_025C_1v80.lib` (downloaded via `vendor/sky130/download.sh`). `parseYosysReport` extracts `Chip area for module` (um^2), `Number of cells` (or `N cells` summary lines), and ABC delay lines (ns/ps → MHz). Has a **120-second timeout** (`YOSYS_TIMEOUT_MS`) and a **64 MiB maxBuffer** (`YOSYS_MAX_BUFFER_BYTES`). Timeout errors include a diagnostic explaining the likely cause (deep combinational blob that abc cannot map) and advising the registered output-stationary MAC-array structure. **Implemented.**
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

## Vendor Libraries (`vendor/`)

### `vendor/sky130/`
Contains the Sky130 open standard-cell library used for ASIC synthesis.

- `sky130_fd_sc_hd__tt_025C_1v80.lib` — the Liberty timing library for the SkyWater Sky130 high-density standard-cell family, typical-typical corner at 25C and 1.80V. Used by `run_yosys` for `dfflibmap -liberty`, `abc -liberty`, and `stat -liberty`.
- `download.sh` — script to fetch the Liberty file from the SkyWater PDK repository.
- `README.md` — provenance and license notes.

---

## Scripts (`scripts/`)

Mostly frontend-prep utilities plus one repo-maintenance contract check. The Python side is split into thin CLI wrappers plus importable `*_impl.py` modules so pytest can exercise the helpers while keeping the local test flow deterministic. `prepare_pipeline.py` is the top-level no-argument smoke harness for the full frontend.

### `__init__.py`
Makes `scripts/` an importable package so `pytest` (and future tests) can `from scripts.golden_impl import ...`.

### `paths.py`
`detect_repo_root(current_file)` — resolves the repo root, honoring a `NN2RTL_REPO_ROOT` env override so tests can point the scripts at a temp dir.

### `quantize_impl.py`
Importable helpers for the ResNet-50 PTQ checkpoint flow. Core pieces:

- `resolve_checkpoint_path(...)`, `get_quantized_checkpoint_path(...)` — canonical checkpoint path resolution.
- `build_resnet50_quantized_checkpoint(...)` — loads torchvision ResNet-50, seeds PyTorch with `0`, runs deterministic synthetic calibration on 32 random `1×3×224×224` tensors, folds batch norm into conv parameters, and exports **17 modules** as a `format_version: 2` checkpoint:
  - `layer0_0_conv1` — the stem conv module. The current legacy `.pth` path does **not** fold MaxPool into this layer; always trust the emitted `LayerIR` + goldens over prose when deciding whether extra fused stages are present.
  - For each `block_index ∈ {0, 1, 2}`: `layer1_<block>_conv1`, `layer1_<block>_conv2`, `layer1_<block>_conv3`, `layer1_<block>_add`, `layer1_<block>_post_add_relu`. The post-add ReLU is a first-class module so Foundry generates it as its own pipeline stage.
  - `layer1_0_downsample` — the 1×1 projection conv on block 0 that feeds the `rhs` of `layer1_0_add`. Serialized through the same BN-folded conv path as the main-line convs.
- `_build_residual_stack_spec(...)` emits a `residual_stack_spec` alongside `layers`. Each operation carries its `module_id`, `op_type`, and wiring (`input` for unary ops, `lhs` / `rhs` for adds). This is the topology information `golden_impl.py`'s fx path needs to rebuild the network — previously missing, which is why the fx bridge used to fail loudly for v2 checkpoints.
- Add modules also get `lhs_scale_factor` and `rhs_scale_factor` on the LayerIR so Foundry can implement the quantized-add formula `out = saturate(round((lhs · S_lhs + rhs · S_rhs) / S_out))` — a naive byte-wise add would be numerically wrong because the two operands live at different scales.
- Quantization is symmetric INT8 per tensor (`zero_point = 0`, `scale = max(|w|) / 127`) with INT32 bias export derived from the observed activation range during calibration.
- `write_quantized_checkpoint(...)`, `load_quantized_checkpoint(...)` — persist and validate the new flattened v2 checkpoint schema. Validation is eager and raises `CheckpointValidationError` on malformed metadata, and the residual_stack_spec integrity check (`_validate_residual_stack_spec`) confirms every wired `module_id` exists and every add has both `lhs` and `rhs`.
- Legacy helpers (`build_toy_quantized_checkpoint(...)`, `ToyPointwiseModel`, `create_toy_model(...)`, `run_toy_model(...)`) are still present for older local fixtures and compatibility tests.
- `build_quantization_summary(...)` — machine-readable summary emitted by the CLI.

The current export scope is intentionally constrained to the initial stem convolution plus `layer1` (17 modules total) so the downstream RTL path can be exercised incrementally. The CLI and summary text explicitly note that real users should swap the synthetic calibration tensors for ImageNet samples.

### `quantize_model.py`
Thin CLI wrapper over `quantize_impl.py`. The CLI shape is unchanged: `python scripts/quantize_model.py [checkpoint_path]`. It now writes a real ResNet-50 INT8 checkpoint to `checkpoints/resnet50_int8.pth` by default and prints a JSON summary describing the constrained export scope plus per-layer scale metadata.

### `golden_impl.py`
Importable helpers for the golden-vector / layer-IR extraction flow. Core pieces:

- `get_output_paths(...)`, `get_weight_artifact_paths(...)`, `get_goldens_dir(...)`, `get_golden_artifact_paths(...)` — canonical output/weights/goldens layout.
- `int8_to_hex(...)`, `int32_to_hex(...)`, `write_signed_int8_hex(...)`, `write_signed_int32_hex(...)` — `$readmemh`-compatible signed INT8/INT32 weight/bias serialization.
- `write_golden_vector_file(...)`, `read_golden_vector_file(...)` — per-module binary activation streams in **NN2V v2 format**. File layout: 20-byte header — 4-byte ASCII magic `NN2V`, uint32 LE version (=2), uint32 LE `num_vectors`, uint32 LE `samples_per_vector`, uint32 LE `bytes_per_sample` — then `num_vectors × samples_per_vector × ceil(bytes_per_sample/4)` int32 LE words. Each logical sample is a packed bus value for one cycle; bytes are little-endian within each word. The C++ testbench's `loadVectorFile` reads the same format directly from disk.
- `requantize_tensor_with_scale(tensor, scale_factor)` — applies the requantization multiplier and clamps to INT8 range: `clamp(round(tensor * scale_factor), -128, 127)`. Used by `Int8Conv2d` and `Int8FusedStemConv2d` for conv modules (multiply by `scale_factor`). `Int8Add` converts both operands to the real domain (`lhs * lhs_scale_factor + rhs * rhs_scale_factor`) then divides by `output_scale_factor` to get back to INT8.
- `compute_conv2d_latency_cycles(weight_shape)` — returns `IC * KH * KW + CONV_PIPELINE_STAGES` (where `CONV_PIPELINE_STAGES = 3`) for the output-stationary MAC-array pipeline latency.
- `fold_batch_norm_into_conv(...)` — folds BN parameters into convolution weights/bias.
- `CheckpointResidualStack`, `ResidualStackTracer`, `ActivationCaptureInterpreter` — rebuild or load the residual stack, preserve module IDs through `torch.fx`, and capture per-node activations.
- `build_deterministic_input_stream(...)`, `capture_golden_outputs(...)` — generate the required 8 seeded INT8 vectors and record golden inputs/outputs per traced node in topological order.
- `build_pipeline_ir_payload(...)`, `validate_pipeline_ir_payload(...)`, `write_pipeline_ir(...)` — produce a schema-shaped `PipelineIR`, write canonical `output/layer_ir.json`, and mirror it to `output/golden_vectors.json` for existing MCP consumers.
- Format-version-2 checkpoints go through a single path: `CheckpointResidualStack` is built from the `residual_stack_spec` embedded in the checkpoint, `ResidualStackTracer` produces an `fx.GraphModule`, and `capture_golden_outputs` runs 8 deterministic input vectors through it to produce per-node int8 activation streams. Weights come from the flat `layers` dict's `weight_int8` / `bias_int32` fields (`resolve_layer_parameters` understands both the v2 names and the legacy `weights` / `bias` keys used by hand-crafted fixtures).
- **Wide-bus packing**: `input_width_bits = in_channels * 8` for conv/relu, `= in_channels * 16` for add (lhs + rhs packed into a single `data_in` bus). `output_width_bits = out_channels * 8`. Golden vectors are packed at the channel granularity — `data_in[i*8 +: 8]` corresponds to channel `i`.
- **Add module packing**: `build_fx_pipeline_ir_payload` drives add modules with `input_width_bits = 2 × operand_width` and writes the per-vector samples as packed 16-bit values where `bits[W-1:0] = lhs` and `bits[2W-1:W] = rhs` into the binary `.goldin` file. This matches the canonical single-`data_in` port contract without needing multi-port modules.
- **Binary golden-vector breakout**: real torchvision ResNet-50 activations would otherwise inflate `output/layer_ir.json` into a multi-GB JSON file that exceeds Node's 512 MB `readFileSync` string cap and the MCP argument-size limit. To keep the LayerIR small (tens of KB), per-module golden vectors now live in separate binary files under `output/goldens/<module_id>.goldin` and `<module_id>.goldout`, referenced by absolute-POSIX `golden_inputs_path` / `golden_outputs_path` fields on each LayerIR entry.
- **Stem semantics**: do not infer `layer0_0_conv1` behaviour from documentation alone. The current legacy `.pth` path and the ONNX path may differ; Foundry and Surgeon must treat the emitted `LayerIR` + golden vectors as authoritative for whether ReLU / MaxPool are fused.
- **Scale-aware add**: `Int8Add` uses `lhs_scale_factor` and `rhs_scale_factor` from the LayerIR to requantize both operands before adding, matching the math Foundry must implement in RTL.
- Format-version-2 checkpoints that lack a `residual_stack_spec` (or equivalent `model` / `model_spec` / `graph`) raise `GoldenGenerationError` rather than silently emitting empty golden arrays — this was the caveat that motivated emitting the spec from `quantize_impl.py`.
- Legacy format-version-1 toy checkpoints are still supported so the older local tests and fixtures continue to work.

### `generate_golden.py`
Thin CLI wrapper over `golden_impl.py`. It resolves the checkpoint path, generates `output/layer_ir.json`, mirrors the same JSON to `output/golden_vectors.json`, emits per-module weight/bias hex files under `output/weights/`, and prints a compact JSON summary (`status`, `model_name`, `num_layers`, `checkpoint_path`, `pipeline_ir_path`). The current path supports both the flat PTQ ResNet-50 checkpoint written by `quantize_model.py` and the richer format-version-2 residual-stack checkpoints used by the `torch.fx` tests, while retaining the legacy toy checkpoint fallback.

### `prepare_pipeline.py`
No-argument frontend smoke harness. It runs `quantize_model.py`, runs `generate_golden.py`, validates the resulting `output/layer_ir.json` against `pipelineIrSchema` from `mcp/schemas.ts` by shelling out to Node with `--experimental-strip-types`, prints a fixed-width summary table (`module_id | op_type | shape | num_weights | pipeline_latency_cycles`), and exits non-zero on the first failed step. The helper surface is intentionally small: subprocess chaining, TypeScript-schema validation, and table rendering.

### `check-twins.mjs`
Small Node maintenance script invoked by the root `check:twins` script before the repo-wide test and coverage commands. It enforces the "twin file" contract between `sdk/` and `mcp/`: `types.ts` must remain byte-identical, while the shared schema exports in `schemas.ts` (`failureClassSchema`, `layerIrSchema`, `pipelineIrSchema`, `verilogModuleSchema`, `verifResultSchema`) must remain byte-identical even though each package also has local-only schemas.

---

## Static Testbench (`tb/`)

Handwritten C++ Verilator driver. Intentionally *not* agent-generated — the README flags this as an architectural decision to avoid the two-bug problem (wrong RTL + wrong testbench).

### `static_verilator_tb.cpp`
**Implemented.** Loads the sidecar JSON (`argv[1]`), reads golden input/output vectors from the NN2V v2 binary files at the paths the sidecar references, applies reset for five cycles, and then runs a single unified interleaved drive/sample loop per vector:

- On each tick: first, if `valid_out` is asserted, sample `data_out` as the current expected output index and advance; second, if the DUT's `ready_in` is high and inputs remain, drive `data_in` / `valid_in`; finally, `tickClock()` unconditionally — including on the cycle that just sampled a vector's final output, to avoid bleeding `valid_out` / `data_out` state into the next vector.
- Per-vector state (`input_idx`, `output_idx`, `idle_cycles`) is scoped to the vector, so every vector samples starting at output 0.
- Hang budget is `pipeline_latency_cycles * 4 + 16`. Exceeding it throws with the stuck vector / output index and the cycle count.

**Wide-bus handling.** The testbench uses a three-way template dispatch to handle buses of any width:
- `IData`/`QData` (scalar types, up to 64 bits) — packed into a single integer.
- `VlWide<N>` — Verilator's wide-signal class for ports > 64 bits.
- `WData[N]` — raw array form for the same.

`assignPackedWords(signal, words)` writes a vector of uint32 words into the DUT's `data_in` port. `readPackedWords(signal, words_per_sample)` reads the DUT's `data_out` port into a vector of uint32 words. `unpackInt8Channels(words, bytes_per_sample)` extracts per-channel INT8 values with proper sign extension from the packed representation.

**Sidecar cross-checks.** `requireCanonicalSignals()` validates all seven signal names. `requireConsistentBusWidths()` checks that `bus_bytes_per_sample == input_width_bits / 8`.

**NN2V v2 binary reader (`loadVectorFile`).** Reads the 20-byte header (magic "NN2V", version=2, num_vectors, samples_per_vector, bytes_per_sample), then reads `num_vectors × samples_per_vector × ceil(bytes_per_sample/4)` int32 LE words.

Handshake semantics documented in the file header and enforced against the sidecar: `valid_in` / `data_in` are bench-driven inputs; `ready_in` is a DUT output (upstream backpressure); `valid_out` / `data_out` are DUT outputs and `data_out` is sampled only when `valid_out == 1`.

Build-time contract: the DUT is selected via preprocessor macros `VMODEL_HEADER` (path to the generated Verilator header) and `VMODEL_CLASS` (the Verilator-generated class name). `run_verilator` sets both when it invokes `verilator --cc --exe --build`.

Numerical pass threshold is `max_error <= 3`. Timing pass requires `timing_actual_cycles == pipeline_latency_cycles` exactly. `failure_class` is emitted as `null` — Surgeon classifies after reading the results file.

**Diagnostic cap.** When a results JSON is emitted, the `expected` and `got` arrays are capped at 1000 samples (500 head + 500 tail) to keep the JSON small enough for LLM context windows.

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

Actual suites now live in `sdk/test/`, `mcp/test/`, and the Python files under `scripts/` (`test_paths.py`, `test_quantize_impl.py`, `test_golden_impl.py`, `test_cli.py`, `test_prepare_pipeline.py`). The fast path is mostly mocked and deterministic; the full path exercises real `iverilog`, `verilator`, `yosys`, and the Python frontend scripts, including the `prepare_pipeline.py` smoke harness and its failure-path coverage in `scripts/test_prepare_pipeline.py`.

---

## Runtime Output Layout (`output/`)

All runtime artifacts live here. The four subdirectories (`rtl/`, `tb/`, `reports/`, `weights/`) are kept in the repo with `.gitkeep` files so the layout survives a fresh clone; the generated contents inside them are git-ignored and may be present locally after any run. `ensureOutputLayout()` in `sdk/orchestrate.ts` re-creates the directory layout at every pipeline start.

- `output/rtl/<module_id>.v` — Foundry / Surgeon output, normally written through `write_verilog` and force-persisted by `persistVerilogModule()` as a safety net.
- `output/rtl/<module_id>.meta.json` — sidecar metadata for each generated module.
- `output/tb/<module_id>.sidecar.json` — orchestrator's per-run sidecar for the static testbench (written by `runAssayerDeterministic`).
- `output/weights/<module_id>_weights.hex`, `<module_id>_bias.hex` — Cartographer's hex-format tensors.
- `output/goldens/<module_id>.goldin`, `<module_id>.goldout` — per-module binary golden-activation streams in NN2V v2 format, written by `scripts/generate_golden.py` and referenced by the LayerIR's `golden_inputs_path` / `golden_outputs_path` fields (see `golden_impl.py` → `write_golden_vector_file` for the format).
- `output/reports/run_log.jsonl` — JSONL event stream for the full run, produced by `appendRunLog`.
- `output/reports/<module_id>.yosys.json` — Yosys synthesis report per passing module.
- `output/reports/<module_id>.results.json` — Verilator testbench results per verification run.
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
| Deterministic Assayer sidecar write | `sdk/orchestrate.ts` `runAssayerDeterministic` | `assayerLayerBusContractZod` (bus-width cross-checks) |
| Deterministic Assayer preflight | `sdk/orchestrate.ts` `preflightVerilogModule` | ANSI port-list parser + `CANONICAL_TOP_PORTS` |
| `run_verilator` sidecar read | `mcp/tools.ts` `readSidecarIfPresent` | `verificationSidecarSchema` |
| `read_weights` PipelineIR read | `mcp/tools.ts` `read_weights` | `pipelineIrSchema` |
| `output/layer_ir.json` on load | `sdk/orchestrate.ts` `ensureLayerIr` | `pipelineIrSchema` + `validateAddModulePacking` |
| `output/rtl/<id>.meta.json` on load | `sdk/orchestrate.ts` `loadPersistedVerilogModule` | `verilogModuleSchema` |
| `output/pipeline_state.json` on resume | `sdk/pipeline.ts` `loadState` | `pipelineStateSchema` (incl. `superRefine`) |
| Yosys direct MCP result | `sdk/orchestrate.ts` `invokeYosys` | `synthesisReportSchema` |

Validation failures throw with field-level error paths so corrupted artifacts and malformed agent outputs fail loudly instead of silently propagating.

---

## Outstanding TODOs

Entries reference exact file and line of the current work. In future revisions, completed entries should move to **Done** with a description of the implementation.

### Critical Path (blocks the intended thesis-grade ResNet-50 pipeline)

These require external tools and ML libraries; they cannot be completed in pure TypeScript.

- **Extend the current PTQ export beyond stem + `layer1`**: the real torchvision ResNet-50 PTQ path now produces a 17-module checkpoint covering the fused stem and all three `layer1` bottlenecks, but `layer2` / `layer3` / `layer4` / `avgpool` / `fc` still need to land before a full-network run.
- **Manual smoke test on the intended checkpoint**: the automated suite now covers the deterministic frontend harness end-to-end (quantize → generate_golden → prepare_pipeline → 17-layer LayerIR) on a shrunk-channel torchvision shim, but the final thesis path still needs an opt-in smoke command against the real torchvision ResNet-50 weights and artifact set.

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
- Stale TODO comments on `run_iverilog`, `run_yosys`, and `write_verilog` removed — those tools are implemented.
- Zod runtime validation added at every JSON trust boundary, including MCP tool inputs/outputs, agent structured outputs, `read_weights` PipelineIR read, `run_verilator` sidecar read, and the resume state file.
- JSON-Schema blobs deduplicated: `sdk/orchestrate.ts` and `mcp/server.ts` now derive every advertised schema from the Zod exports via `z.toJSONSchema()`.
- Static Verilator testbench (`tb/static_verilator_tb.cpp`) implemented: sidecar parsing, seven-name canonical enforcement, ready/backpressure-aware drive loop, per-vector sampling that correctly ticks past the final output, hang detection, cycle-accurate timing, numerical tolerance check, structured results JSON, best-effort error propagation. `nlohmann/json` vendored at `tb/third_party/json.hpp`.
- `run_verilator` fully wired end-to-end: copies the static testbench into a tempdir with preserved `third_party/json.hpp` layout, invokes `verilator --cc --exe --build` with `VMODEL_HEADER` / `VMODEL_CLASS` macros, runs the produced binary against the sidecar, reads and Zod-validates the results JSON, and maps build / execution failures to properly shaped `VerifResult`s.
- `PipelineStateManager.loadState` now recovers transient `generating` / `verifying` statuses from a crashed prior run, correctly rolling back the attempts counter for Surgeon-path crashes so the retry budget is not over-billed.
- Entry points separated: `sdk/main.ts` and `mcp/main.ts` are the runnable CLIs, leaving `orchestrate.ts` / `server.ts` as library code importable by tests.
- Dependency injection seams in place: `OrchestratorRuntime` (`now`, `queryFn`, `yosysFn`, `assayerFn`) in the SDK, `ToolsRuntime` (`commandRunner`, `cwd`, `env`, `tmpDirRoot`) in the MCP tools, `ToolImplementations` in the MCP server — all testable without touching the real toolchain, real clock, or real Claude API.
- Test infrastructure implemented: root `package.json` with `test:fast` / `test:full` / `coverage`, `pytest.ini` with markers, per-package `vitest.config.ts` with coverage thresholds, SDK and MCP test suites, Python helper tests, and shared fixtures under `test/fixtures/`.
- Python preprocessing split into importable `*_impl.py` helpers (`paths.py`, `quantize_impl.py`, `golden_impl.py`), with a real ResNet-50 PTQ export path, the `torch.fx` golden-capture path driven by an embedded `residual_stack_spec`, and the older deterministic toy compatibility path.
- **Real PTQ path implemented**: `scripts/quantize_model.py` / `scripts/quantize_impl.py` now load torchvision ResNet-50, run deterministic synthetic calibration, fold BN into conv weights, quantize to symmetric INT8 per tensor, and persist the flattened `format_version: 2` checkpoint schema validated by `load_quantized_checkpoint`.
- **ResNet-50 frontend correctness pass landed**: the PTQ export emits all 17 layer1 modules (stem fused with MaxPool, three bottlenecks × {conv1, conv2, conv3, add, post_add_relu}, plus `layer1_0_downsample`), embeds a `residual_stack_spec` describing the DAG with explicit `lhs` / `rhs` wiring for every residual add, and populates `lhs_scale_factor` / `rhs_scale_factor` on add modules so the int8 quantized-add math is implementable. `scripts/golden_impl.py`'s fx path reads the spec, scales both add operands via `Int8Add`, and emits `golden_inputs` as packed 16-bit values matching the single-`data_in`-port contract. Foundry's agent and skill files document both the packed-add convention (PATTERN A) and the stem fusion rule (PATTERN B); no twin-protected schema change beyond two new optional `lhs_scale_factor` / `rhs_scale_factor` fields on `layerIrSchema`.
- **Twin-file check protects the add-scale fields**: `scripts/check-twins.mjs` covers the `layerIrSchema` shared export, so the two new optional add-scale fields cannot drift between `sdk/` and `mcp/`. The twin check runs first in every root test target.
- **MCP `run_verilator` test no longer leaks `mcp/relative-inputs.json`**: `writeSidecar` in `mcp/test/tools.test.ts` now materializes the golden input/output vector files only when the sidecar paths are absolute, so the "rejects relative path" test case no longer writes into the package root cwd on Windows runs.
- Yosys policy decided and implemented: a failed post-pass Yosys synthesis report now feeds back into the retry loop as `failure_class: "synthesis_failed"` instead of being recorded as a degraded-but-still-passing module.
- **Fmax gate tightened**: `parseYosysReport` now extracts ABC delay lines (`Delay = X ns`, `Delay = X ps`, `Current delay (X ns/ps)`) in addition to explicit `MHz` numbers. `evaluateSynthesis` no longer silently skips the timing gate when `fmax_mhz === 0` — that path now emits a `synthesis_failed` VerifResult with `fix_hint` explaining that timing could not be measured, so Surgeon is forced to address the issue instead of the gate being quietly bypassed.
- **Surgeon retry loop tested**: `test/orchestrate-flow.test.ts` now exercises two forced-failure yosys paths end-to-end — `fmax_mhz === 0` (synthesis_failed) and `fmax_mhz` below the PPA target (missing_pipeline_register). Both route through `applyVerifResult` into a Surgeon dispatch and a second Assayer + Yosys round. A new `yosysFn` runtime seam on `OrchestratorRuntime` makes these tests possible without invoking the real toolchain.
- **Frontmatter parser replaced**: the hand-rolled `---`-delimited parser in `orchestrate.ts` is gone; frontmatter now goes through the `yaml` package. A new `toStringList` helper accepts both CSV strings (`tools: Bash, Read`) and YAML lists (`tools: [Bash, Read]`), and the legacy `splitCsvField` export was deleted.
- **SDK typing hacks resolved**: `@anthropic-ai/claude-agent-sdk@0.2.107` exposes `AgentDefinition.skills` and `AgentDefinition.maxTurns`, so the local compat shim now advertises both fields and `loadPluginAgentDefinition` propagates them from `AGENT_CONFIG` (per-agent `maxTurns`) and from agent-markdown frontmatter (`skills`). The parent `query()` call keeps its own `maxTurns: 6` as an outer safety cap.
- **Conductor agent removed**: there is no LLM "Conductor" anymore. The deterministic TypeScript orchestrator in `sdk/orchestrate.ts` owns the role directly, and `nn2rtl-plugin/agents/conductor.md` plus `nn2rtl-plugin/skills/conductor/` were deleted. `AGENT_CONFIG` and `AGENT_SLUGS` now list only the three real subagents (Cartographer, Foundry, Surgeon).
- **Assayer agent removed**: there is no LLM "Assayer" anymore. The Haiku-based simulation runner repeatedly hallucinated `VerifResult` payloads instead of calling the MCP tools, and had zero real reasoning to do (run iverilog, run Verilator, parse JSON, return). Verification is now handled by `runAssayerDeterministic` in `sdk/orchestrate.ts`, which writes the sidecar from LayerIR fields, runs `run_iverilog` then `run_verilator` directly via MCP import, and returns a Zod-validated `VerifResult`. `nn2rtl-plugin/agents/assayer.md` and `nn2rtl-plugin/skills/assayer/` are deleted. `AGENT_CONFIG` and `AGENT_SLUGS` list only three agents: Cartographer, Foundry, Surgeon. The `assayerFn` runtime seam on `OrchestratorRuntime` allows tests to mock this path.
- **Sky130 replaces iCE40 for synthesis**: `run_yosys` now targets the Sky130 open standard-cell library instead of iCE40. The Yosys script is `synth -top X; dfflibmap -liberty sky130.lib; abc -liberty sky130.lib [-constr ... -D <period_ps>]; stat -liberty sky130.lib`. `parseYosysReport` extracts `Chip area for module` (um^2), total cell count, and ABC critical-path delay. The lib is vendored at `vendor/sky130/sky130_fd_sc_hd__tt_025C_1v80.lib`. Yosys has a 120s timeout and 64 MiB maxBuffer to prevent hung synthesis and buffer overflow. `isSystemSpawnError` distinguishes infra failures from synth failures.
- **Wide-bus packed-channel interface**: `data_in` is `IC*8` bits wide (all channels of a pixel arrive in one clock), `data_out` is `OC*8` bits wide. LayerIR carries `input_width_bits = in_channels * 8` and `output_width_bits = out_channels * 8`. For add modules, `input_width_bits = in_channels * 16` (lhs + rhs packed). The golden model, Foundry prompt, deterministic verification path, and C++ testbench all implement this contract.
- **Output-stationary MAC array**: Foundry prompt mandates OC parallel MAC lanes with one accumulator per output channel. `pipeline_latency_cycles = IC * KH * KW + 3` (from `compute_conv2d_latency_cycles`). Single-MAC designs are rejected. Foundry prompt includes a Verilog pseudo-template for the MAC-array state machine.
- **Golden model applies scale_factor**: `requantize_tensor_with_scale(tensor, sf)` does `clamp(round(tensor * sf), -128, 127)` for conv (multiply). `Int8Add` divides by `output_scale_factor` (real → int domain).
- **NN2V v2 binary format**: 20-byte header: magic "NN2V", version=2, num_vectors, samples_per_vector, bytes_per_sample. Each sample is `ceil(bytes_per_sample/4)` int32 LE words. Both the Python writer and C++ reader implement the same format.
- **Testbench handles wide buses**: three-way template dispatch for IData/QData (up to 64b), VlWide<N>, WData[N]. `assignPackedWords`/`readPackedWords`/`unpackInt8Channels` handle pack/unpack. Per-channel INT8 sign-extension. Diagnostic cap: 1000 samples in results JSON.
- **Sidecar has `bus_bytes_per_sample`**: cross-checks against NN2V header. Sidecar is written by `runAssayerDeterministic`, not by an LLM agent.
- **Surgeon capped**: maxTurns=8, prompt says "Output immediately. Do not explain."
- **All agents disallow Agent/Task tools**: `disallowedTools: Agent, Task` in every agent markdown. No implicit Opus subagent spawning.
- **Deterministic RTL preflight**: `preflightVerilogModule` in `orchestrate.ts` parses ANSI port list, checks canonical port names, directions, and bus widths before running Verilator.
- **CLI knobs extended**: `parseCliArgs` accepts `--max-retries N` (with both `--max-retries N` and `--max-retries=N` forms) and rejects unknown `--flag` arguments with a clear error. `runPipeline` / `RunPipelineOptions` gained an optional `maxRetries` field that overrides `PIPELINE_CONFIG.max_retries`. Unknown flags now error instead of being silently ignored.
- **`invokeYosys` compiled-path fix**: the dynamic import specifier for the MCP `run_yosys` helper used to be the relative string `"../mcp/tools.js"`, which resolves correctly from `sdk/orchestrate.ts` source but points at the non-existent `sdk/mcp/tools.js` when running `node sdk/dist/main.js`. Fixed by computing the specifier at module load via `pathToFileURL(path.resolve(repoRoot, "mcp", ...))` with a branch on whether `__dirname` is `dist` (→ `mcp/dist/tools.js`) or source (→ `mcp/tools.ts`).
- **Proven end-to-end result**: `layer1_0_conv1` (1x1 conv, IC=64, OC=64) passes end-to-end: max_error=1, timing=67/67 cycles (matching `IC*KH*KW + 3 = 64*1*1 + 3 = 67`), Fmax=77.4 MHz on Sky130, area=669K um^2, cost=$0.35, zero Surgeon retries.

---

## Known Bottleneck — Spatial Convolutions Do Not Close Reliably

**What passes and what doesn't, as of 2026-04-19:**

| Layer | Shape | Pipeline result | Notes |
|---|---|---|---|
| `layer1_0_conv1` | 1×1, IC=OC=64, no padding | ✅ **PASS** end-to-end | Foundry first shot, 0 Surgeon attempts, Yosys 12s. Fmax 116.3 MHz, lut 12,015, area ~100k µm². Reproducible across two separate runs. |
| `layer0_0_conv1` | 7×7 stride-2, IC=3 OC=64, padding=3 | ❌ `fail_abort` in pipeline; ✅ passes when Yosys is invoked manually with no timeout | Sim+timing pass; synth times out at pipeline's 25-min cap. Manual Yosys at ~16 min: Fmax 61.4 MHz, lut 144,159, area 1.92 mm². RTL is synthesizable — tooling budget was tight. |
| `layer1_0_conv2` | 3×3, IC=OC=64, padding=1 | ❌ Sim never closes | Three separate Surgeon attempts converge on `sim_stalled` without reaching a passing state. Spent ~$50 on Surgeon retries across the run before aborting. |

### What we tried on the spatial convs, chronologically

1. **Original OC-parallel design (MP = OC = 64).** Each ST_RUNNING cycle fired 64 parallel MACs and 64 parallel reads from a flat `weights[0..OC*K_TOTAL-1]` register array. Synth correct in principle, but the combinational cone at Yosys after OPT was ~300k cells, ABC on Sky130 couldn't map in 600s. Pipeline reported `fail_abort: synthesis_failed`. Tried across ~5 runs, each ~$6–$9.

2. **Introduced `mac_parallelism` (MP=8).** Replaced OC-parallel with OC-group iteration: 8 parallel MAC lanes, `OC_PASSES = ceil(OC/MP)` passes per output pixel, each pass doing `K_TOTAL` MAC cycles across 8 lanes. Post-OPT cell count dropped ~7× (299k → 43k). Synth still timed out at 600s, the new hot spot was 8 parallel reads from a ~9k-element weight array. Verified end-to-end once by running Yosys manually with no timeout (the 61.4 MHz result above). Ran ~4 pipeline attempts, $5–$9 each.

3. **Raised `YOSYS_TIMEOUT_MS` to 1800s then 1500s (25 min).** Manual run was 16 min. One pipeline run came within the budget, but Surgeon output varied — sometimes the particular RTL that won the manual run wasn't what Foundry/Surgeon produced next time, and the next RTL's synth still exceeded budget. Surgeon also began introducing "clever" memory-packing optimizations (`weights_packed`) to fight the cone, which Yosys's `OPT_MEM` pass rejected as non-constant `$meminit`. 3 runs, $6–$10 each.

4. **Introduced registered shift-register `window`.** Foundry had been rebuilding `window[kh][kw][ic]` combinationally every cycle from `line_buf` + `cur_row` + `data_in` with per-tap bounds checks. Replaced with a shift-left + single-column load pattern. Moved line-buffer promotion from start-of-next-row to end-of-current-row to keep window loads ordered correctly. Fixed the combinational-window cone specifically. 2 runs, $4–$6 each — sim started passing more often but synth still timed out on a different axis (parallel weight reads).

5. **Serialized weight reads (MP=4, one read per cycle).** Added `lane_counter` register that rotates 0..MP-1 across cycles; per cycle: one weight read, one multiply, one accumulate into `acc[lane_counter]`. Per OC pass: `MP*K_TOTAL + 3` cycles instead of `K_TOTAL + 3`. Latency for `layer0_0_conv1` went 1885 → 10,141. Synth side of the problem solved in principle. Locked in with `[INVARIANT:WEIGHT_ARRAY]` protection so Surgeon can't re-pack. 1 run: Foundry's first output stalled sim at `fmm=0`, three Surgeon attempts each burning ~20 turns of Opus converged on the same `fmm=7122` sim stall at the right-edge output column. Cost $9.95.

6. **Ran `layer1_0_conv2` (3×3 pad=1, smaller geometry) in isolation to see if the spatial-specific issue reproduces on a less-extreme shape.** It does. First Foundry output: `sim_stalled fmm=0`. Two Surgeon attempts ($2.22 + unknown second attempt) made no progress before the Verilator simulation itself hung (no timeout on Verilator) and had to be killed manually.

### Why it still doesn't close

Each fix above is individually correct and measurable. The remaining failure modes are not single bugs — they're **LLM correctness variance at a problem size that keeps growing**:

1. **Foundry emits a new FSM every time.** Even with the template pinned in `foundry.md`, Opus-generated RTL varies run-to-run. One run's right-edge drain exit condition is correct; the next run's isn't. The pinned template reduces variance but does not eliminate it, because Foundry still interprets/instantiates the template against a specific LayerIR.

2. **Surgeon's repair surface grows with architecture.** Each added rule (shift-register window, serialized weights, INVARIANT:WEIGHT_ARRAY, partial-group gating, ready_in freeze, line-buffer promotion timing) narrows what Surgeon may touch. The bugs Foundry introduces still live inside that narrowed space but are harder to diagnose from the evidence because the evidence (`first_mismatch_index`, `output_gap_histogram`) looks the same for many different root causes.

3. **Tooling has no Verilator-simulation timeout.** `run_yosys` has `YOSYS_TIMEOUT_MS`; `run_verilator` has no equivalent cap on the simulation binary. A Surgeon edit that produces an FSM which occasionally-but-not-always fires `valid_out` keeps the simulation alive forever — the testbench's `hang_budget` only fires on total silence. Observed once, hung for 50+ min before manual kill.

4. **Weight-array size scales with the network.** At `layer0_0_conv1` (7×7, 64×3), `weights` is 9,408 INT8 bytes. At `layer1_0_conv2` (3×3, 64×64), it's 36,864. The user's projection for `layer4_0_conv2` (3×3, 512×512) is **2,359,296 INT8 weights = 18.9 Mbit** of state. No PDK-BRAM inference is possible on Sky130. Even correct RTL becomes a 2.4M-flop register file with a long mux tree for indexed access.

5. **Bus widths scale too.** `data_out` at `layer4_0_conv3` (OC=2048) is 16,384 bits. Foundry has to emit correct bit-slicing (`data_out[global_oc*8 +: 8]`) across that entire width. LLM accuracy at this scale is the load-bearing risk; template doesn't help if the LLM skips one range of indices.

### What the current pipeline actually proves

- **Architecture is correct**: verified latency formula matches TB measurement to the cycle; shift-register window + serialized MAC + OC-group iteration produce synthesizable, functionally-correct RTL when the LLM gets the FSM right.
- **1×1 pointwise convs work cleanly**: no window, no padding, no right-edge drain. `layer1_0_conv1` passes reproducibly, first-shot, under $1.
- **Synth characterization is real**: `manual_yosys` run on `layer0_0_conv1` produced genuine Sky130 PPA (61.4 MHz, 1.92 mm², 144K gates).
- **Spec-hash template reuse is implemented** but never exercised end-to-end, because no two spatial convs have both passed in the same run.

### What would make it bulletproof (not done yet)

The pipeline's bottleneck is generative RTL for spatial convs at scale, which is an open problem. The proposed fix, documented but not implemented:

1. **Parameterized handwritten operator library** (`rtl_library/conv2d_pointwise.v`, `conv2d_spatial_k3p1.v`, `conv2d_spatial_k7p3.v`, `add_quantized.v`, `relu.v`, `maxpool.v`). Each handwritten once, verified once, parameterized on `OC, IC, IH, IW, MP, scale constants, weights_path`.
2. **Foundry collapses to instantiation**: instead of generating RTL, Foundry picks a library module and emits an instantiation wrapper with the LayerIR's parameters. ~30 lines of LLM output instead of ~300. Zero structural variance.
3. **Surgeon becomes optional** for conv/add/relu/maxpool. It stays available for new operators not in the library.
4. **`VERILATOR_SIM_TIMEOUT_MS` added** to `mcp/tools.ts` mirroring the Yosys timeout. Currently missing and has burned ~1 hour of wall-clock on hung sims.
5. **Weight-array size ceiling + BRAM-inferrable memory flow** before the pipeline accepts layers beyond ~1M weights. L2+ shapes will hit this.

This conversion is the honest path to scaling past `layer1_0_conv1`. The `LLM-generates-RTL-from-scratch` loop is fine as a research demo and produces the one passing 1×1 module; the hybrid `LLM-picks-parts-from-verified-library` flow is what would actually cover the 17-module ResNet-50 pipeline, let alone L4.

### Pattern-library + cross-cutting infrastructure — landed 2026-04-19

The `knowledge/IMPLEMENTATION_PLAN.md` plan has shipped. What's now active
in the pipeline (not yet re-measured against the 17-module run):

- **`VERILATOR_SIM_TIMEOUT_MS = 10 min`** (`mcp/tools.ts`) — simulation
  timeouts classify as `failure_class: verilator_timeout` (routes to
  Surgeon with a timeout-specific rubric).
- **Bus-width capability gate** (`sdk/config.ts::PIPELINE_CONFIG.MAX_SUPPORTED_BUS_BITS = 4096`) —
  layers above the cap fail-abort with `failure_class: architectural_unsupported`;
  Surgeon is NOT invoked on these. Reports separately in the pipeline summary.
- **Five structural preflight rules** (`structuralPreflightViolations` in
  `sdk/orchestrate.ts`): `line_buffer_missing`, `window_not_registered`,
  `weights_packed_forbidden`, `readmemh_missing`, `output_counter_missing`,
  plus a sixth `coord_scheduler_missing` check for spatial conv / maxpool.
  Violations surface as `failure_class: structural_preflight_failed`
  with the specific rule name in `fix_hint`.
- **`coord_scheduler.v`** (`rtl_library/coord_scheduler.v`) — handwritten,
  parameterized on `IH/IW/OH/OW/KH/KW/SH/SW/PH/PW`. Terminates on
  `outputs_emitted == OH*OW` (never on `in_row > IH-1+PH`). Spatial conv
  and maxpool modules must instantiate it; rolling their own coordinate
  logic is structurally rejected.
- **`get_rtl_patterns` MCP tool** — dispatches on `op_type` + kernel
  dimensions to concatenated `knowledge/patterns/*.md` context files,
  and returns `conv1x1_passing_reference.v` verbatim for 1×1 convs.
  Foundry and Surgeon call this before emitting / repairing RTL.
- **Eight pattern markdown files** in `knowledge/patterns/`:
  01_context, 02_conv1x1, 03_conv3x3_pad1, 04_conv7x7_pad3,
  05_add_quantized, 06_relu, 07_maxpool, 08_common_bugs.
- **Reference file**: `knowledge/references/conv1x1_passing_reference.v`
  — the one proven-passing 1×1 module, adapted by Foundry on pointwise
  convs. No external (out-of-repo) RTL is currently load-bearing.

None of these are ResNet-specific; all apply to any network-any-layer
composition. The 17-module re-measurement was deferred to a future
session — the new infrastructure changes the `fail_abort` taxonomy and
the PPA results will look different.

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
