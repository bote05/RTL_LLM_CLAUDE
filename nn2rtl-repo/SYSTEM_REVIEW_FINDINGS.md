# nn2rtl System Review Findings

Date: 2026-05-02

Scope: static code and artifact review only. I did not run the existing tests, per request. I inspected the architecture, orchestration logic, MCP tools, contracts, agents, and current `output/` artifacts. The environment matters here: this repo is being used from Windows/WSL paths, and several findings are specifically about that split.

## Short Version

The system has a good overall direction: deterministic extraction, contract-based IO plans, autonomous repair, and lifecycle docs are the right ingredients for supporting many architectures and neural networks. The biggest problems are not "one bad RTL bug"; they are feedback-loop problems where the system can learn from, cache, or permanently react to the wrong signal.

Highest priority fixes:

1. Fix Windows/Icarus failures being classified as RTL or contract bugs.
2. Make contract switching real by transforming golden vectors, latency expectations, and testbench semantics per contract.
3. Prevent auto-improve from permanently flagging contracts because of API limits, tool crashes, or classifier outages.
4. Preserve best-known RTL artifacts so a later bad repair does not overwrite an earlier better result.
5. Expand LayerIR semantics for common ONNX Conv variants before claiming broad NN support.

## Fix Pass Update

Implemented after review:

- Windows/Icarus: `run_iverilog` now writes to a real temporary `.ivvp` file instead of `os.devNull`, supports `NN2RTL_IVERILOG_BIN`, and no-diagnostic Windows crashes such as `3221225794` classify as `toolchain_infra`.
- Failure taxonomy: added `toolchain_infra`; classifier outages now get deterministic retryable fallbacks instead of permanent `unknown`, and quota/tool/API failures are not persisted as contract manual-correction state.
- Contracts: all five advertised contracts now have executable orchestrator plans; selected contracts get distinct `io_mode`; contract-specific golden files are materialized by retile/repack; sidecar latency now uses contract latency semantics.
- Self-improve contract selection: verified this was real, not just cosmetic. Startup self-improve selection happened before any failure and always started at plan index 0; additionally, applying `flat-bus` did not clear an existing non-flat `contract_id`. Selection now starts from the layer's selected contract and `flat-bus` explicitly tags itself.
- Best-known RTL/cache: verified the artifact history showed a sim-passed/synth-failed attempt followed by a worse syntax result. The reusable clone cache now only receives modules after Vivado/PPA still leaves them in `pass`, and Surgeon successes are cached only after that same final gate.
- Windows/WSL paths: checkpoint fingerprints and runtime sidecar/golden paths now normalize `C:/...`, `C:\...`, and `/mnt/c/...` at tool boundaries.
- Cartographer/ONNX: production extraction now calls deterministic `read_weights` directly; Cartographer docs were updated for `maxpool`; LayerIR now carries Conv `dilation` and `groups`; asymmetric ONNX Conv padding is rejected instead of silently collapsed.
- Knowledge lookup: `get_rtl_patterns` now accepts `contract_id`, filters lifecycle docs by contract, and the orchestrator preloads contract-filtered pattern context for Foundry and Surgeon.
- Quantization constants: scale approximation now fails loudly when a scale or fused add ratio is outside the representable multiplier/shift range.

Explicitly not changed:

- Probationary docs remain available to generation. That was a deliberate design choice, not a bug. Promotion still gates long-term trust and bloat; this report no longer recommends removing probationary docs from generation by default.

## Current Run Signal

The current artifacts show a pipeline that is blocked very early:

- `output/reports/pipeline_summary.json` reports `modules_total: 17`, but only `layer1_0_conv1` appears in the summary table, likely because the run was scoped with `--only`.
- The same artifacts show repeated `iverilog` failures with exit code `3221225794`, which is Windows `0xC0000002` / `STATUS_NOT_IMPLEMENTED`.
- Several later decisions appear to treat this as a generated-code or contract problem rather than a toolchain/process problem.

That means the most urgent issue is not improving the neural-network architecture coverage. It is making sure the system can tell the difference between "the RTL is bad" and "the local Windows tool invocation crashed before diagnostics existed."

## Blocking Findings

### 1. Windows `iverilog` infrastructure crashes are treated like RTL/code bugs

Evidence:

- `mcp/tools.ts` builds the `iverilog` command with `-o os.devNull`.
- On Windows, `os.devNull` becomes a `\\.\nul` style path. The current artifacts show commands like `iverilog -o \\.\\nul ...`.
- `output/contract_state.json` and `output/reports/pipeline_summary.json` show `iverilog` failing with exit code `3221225794` and no useful compiler diagnostics.
- `sdk/orchestrate.ts` only detects no-diagnostic setup failures when stderr matches a narrow literal string like `iverilog exited non-zero without diagnostic output`.
- The actual failure text comes through as a generic Node/process failure, so the classifier path misses the infrastructure classification.

Why this is a real problem:

The pipeline can burn Foundry and Surgeon calls trying to repair RTL that may never have been compiled. Worse, the system can flag a contract for manual correction because a Windows process crashed with no diagnostics.

Recommended direction:

- Do not write Icarus output to `os.devNull` on Windows. Use a real temporary output file and delete it afterward.
- Classify `3221225794`, `0xC0000002`, empty stderr, and no-diagnostic compiler exits as infrastructure/toolchain failures.
- Add an explicit `NN2RTL_IVERILOG_BIN` or toolchain config field so Windows users can point at the intended executable.
- Do not route no-diagnostic compiler crashes to Surgeon or contract learning.

### 2. Transient classifier/API-limit failures become terminal system state

Evidence:

- In `sdk/orchestrate.ts`, classifier failures fall back to `category: "unknown"` with `violated_constraint: "failure_classifier_unavailable"`.
- In `sdk/pipeline.ts`, `unknown` maps to `fail_abort`.
- Current artifacts show a classifier failure caused by usage/rate limit text: `You've hit your limit`.
- `output/contract_state.json` shows a contract marked for manual correction after a `retrospector_foundry_dispatch_failed` style event.

Why this is a real problem:

Auto-improve should not permanently learn from quota limits, unavailable classifier calls, or agent dispatch failures. Those are transient orchestration failures, not architecture failures.

Recommended direction:

- Separate "tool/API unavailable" from "RTL failed".
- Do not set `manual_correction_needed` for model quota, API limit, timeout, or missing-tool failures.
- Add deterministic fallback classification for common failure classes. For example, syntax diagnostics can be `code_bug`; missing executable/no diagnostics/process crash should be `toolchain_infra`.
- Make persisted contract state require evidence from an actual Assayer or Vivado diagnostic, not from classifier unavailability.

### 3. Non-flat contracts are selected, but their golden vectors and latency semantics are not actually enforced

Evidence:

- `contracts/*/golden.py` and `contracts/*/latency.ts` exist.
- Runtime code appears to compute beat metadata, but the static Verilator sidecar still uses the original `golden_inputs_path` and `golden_outputs_path`.
- `sdk/orchestrate.ts` changes sidecar fields such as `bus_bytes_per_sample`, but does not appear to generate contract-specific `.goldin` or `.goldout` files.
- `contracts/tiled-streaming/latency.ts` defines different latency behavior, but that logic does not appear to be consumed by Assayer.

Why this is a real problem:

Changing a layer from a flat 512-bit bus to a 256-bit tiled stream without transforming the golden stream changes the meaning of the bytes. The testbench may reject the data shape or verify the wrong protocol. This makes contract switching partly conceptual rather than truly verified.

Recommended direction:

- When a contract is applied, generate contract-specific golden input/output files using the selected contract adapter.
- Feed contract-specific latency expectations into the testbench.
- Treat each contract as a real protocol with explicit stream order, valid timing, byte packing, and output schedule.
- Make Assayer consume those semantics directly rather than only changing width fields.

### 4. The contract registry advertises more contracts than the orchestrator can execute

Evidence:

- `sdk/contracts.ts` lists five contracts: `flat-bus`, `tiled-streaming`, `dram-backed-weights`, `activation-double-buffering`, and `weight-tiling`.
- `sdk/orchestrate.ts` only defines executable plans for three of them.
- If a current contract is not in the executable plan list, the orchestrator throws an error.
- The apply logic treats all non-flat, non-tiled plans like `dram_backed_weights`, which is too broad for the missing contracts.

Why this is a real problem:

The system claims broader architectural coverage than it can safely run. A model or user that selects `activation-double-buffering` or `weight-tiling` can hit an orchestrator-level failure before Foundry even gets a meaningful chance.

Recommended direction:

- Make one authoritative contract registry that includes metadata, executable plan, adapter, latency semantics, and testbench support.
- Only advertise contracts that are executable end to end.
- Give each contract its own `applyContractPlan` behavior instead of routing all unknown non-flat plans into `dram_backed_weights`.

### 5. Self-improve mode starts from `flat-bus` and can ignore a layer's selected contract

Evidence:

- In self-improve mode, the orchestrator calls `selectAvailableContract(baseLayer, contractState)` for each module.
- The selector starts from the first contract unless given an `afterContractId`.
- `applyContractPlan` overwrites fields such as `io_mode`, `contract_id`, and widths.

Why this is a real problem:

If the frontend or a previous stage already selected a non-flat contract for a layer, self-improve can erase that intent and spend attempts failing `flat-bus` first. That works against the goal of supporting many architectures.

Recommended direction:

- Start contract selection from the layer's existing `contract_id` when present.
- Only fall back to alternatives after the selected contract has a real diagnostic failure.
- Record whether a contract was selected by the frontend, by the user, or by self-improve.

### 6. Best-known RTL is not protected from later regressions

Evidence:

- Several paths add generated modules to `passedModules` after static simulation or template validation, before final Vivado/PPA success.
- Current reports show an earlier attempt that passed simulation but failed synthesis, followed by a later attempt ending in syntax failure.

Why this is a real problem:

The system can overwrite a better artifact with a worse repair attempt. In a self-improving loop, that is dangerous because a later bad edit can become the remembered state while the earlier "at least simulated" design is lost.

Recommended direction:

- Track best-known artifact by stage: generated, compiled, simulated, synthesized, PPA-pass.
- Only promote a module into reusable/passed caches after the final required stage passes.
- If a later attempt regresses, preserve and report the best previous artifact.
- Use best-known RTL as the starting point for Surgeon instead of only the most recent generated text.

### 7. Windows, WSL, and native tool paths are mixed without enough normalization

Evidence:

- Current `output/layer_ir.json` uses paths like `C:/Users/User/...`.
- Current `output/layer_ir.json.checkpoint` uses backslash paths like `C:\Users\...`.
- The active shell path is WSL-style `/mnt/c/Users/User/...`.
- Some path checks use raw `path.resolve` comparisons.
- POSIX Node does not treat `C:/...` as absolute in the same way Windows Node does.
- Verilator or C++ sidecars running in WSL generally cannot open `C:/...` paths unless translated.

Why this is a real problem:

The same model checkpoint and golden files can look stale, missing, or invalid depending on whether the command is run through Windows Node, WSL Node, Vivado, Verilator, or Python.

Recommended direction:

- Pick one canonical internal path form per process.
- Normalize `C:/...` to `/mnt/c/...` for WSL/Linux tools.
- Normalize `/mnt/c/...` to `C:/...` only at Windows-native tool boundaries such as Vivado.
- Fingerprint input files by content hash and normalized absolute identity, not raw path string.

### 8. The deterministic ONNX frontend is wrapped by an LLM step that can corrupt or omit valid layers

Evidence:

- `read_weights` already runs deterministic extraction and validates the resulting PipelineIR.
- Cartographer is still asked to emit the final PipelineIR through an LLM wrapper.
- The Cartographer prompt lists supported `op_type` values as `conv2d`, `relu`, and `add`.
- The actual schema and frontend also support `maxpool`.

Why this is a real problem:

An ONNX model with MaxPool can be extracted by the deterministic tool but then omitted, rewritten, or rejected by stale agent instructions. This is the wrong place to put model judgment.

Recommended direction:

- Let the orchestrator call deterministic `read_weights` directly.
- If Cartographer remains, make it return exactly the tool output unless it can prove a schema issue.
- Update agent docs to include every supported op type.
- Treat LLM changes to PipelineIR as suspicious unless they pass a structural diff policy.

### 9. LayerIR cannot represent several common Conv semantics

Evidence:

- `scripts/onnx_frontend.py` reads Conv attributes such as `dilations` and `group`.
- The final emitted LayerIR does not appear to include `groups`, `dilation`, or full asymmetric padding semantics.
- `sdk/types.ts` and schema definitions also lack those fields.

Why this is a real problem:

Many real networks use grouped convolution, depthwise convolution, dilation, or asymmetric padding. MobileNet-style and EfficientNet-style models are especially affected. If these are silently flattened into ordinary conv fields, the generated RTL can verify against the wrong mathematical operation or fail later in confusing ways.

Recommended direction:

- Add explicit LayerIR fields for `groups`, `dilation`, and full padding.
- Until RTL generation supports them, reject unsupported variants loudly during extraction.
- Add architecture capability metadata so the contract/Foundry layer knows which op variants are legal.

## Auto-Improve And Learning Findings

### 10. Probationary knowledge is intentionally available before promotion

Evidence:

- Knowledge read tiers include `protected`, `active`, and `probationary`.
- Coverage checks can treat probationary docs as existing coverage.
- Promotion thresholds exist, but probationary docs are still available to generation before promotion.

Why this is not automatically a bug:

This was explicitly designed behavior: probationary docs are allowed to influence nearby generation early, while promotion gates long-term trust and controls bloat. The risk is real, but it is a policy tradeoff rather than broken logic.

Guardrails to keep:

- Keep promotion thresholds meaningful.
- Track which modules used each probationary doc.
- Archive probationary docs after failures tied to their contract/op/kernel.
- Revisit this only if probationary docs empirically create repeated bad generations or suppress better docs.

### 11. Knowledge lookup is not contract-filtered

Evidence:

- Lifecycle docs are filtered by `op_type`, but not reliably by `contract_id` or contract key.
- `get_rtl_patterns` takes op/kernel style arguments, but not the selected contract.

Why this is a real problem:

Flat-bus generation can receive tiled-streaming or DRAM-backed advice, and vice versa. That cross-contamination is exactly the kind of subtle learning error that makes auto-improve unreliable.

Recommended direction:

- Add `contract_id` to knowledge lookup.
- Label every returned pattern with contract, tier, source module, and promotion status.
- Do not use docs from a different contract unless explicitly requested as analogical reference.

### 12. Mandatory pattern-tool usage is requested, but not enforced

Evidence:

- Foundry instructions say pattern lookup is mandatory.
- Runtime reports show Foundry using shell/file tools while `documents_used` stays empty.
- The orchestrator only records documents if the specific MCP pattern tool was called.

Why this is a real problem:

The lifecycle system cannot know which learned docs influenced a module. If agents bypass the tool, auto-improve loses attribution and cannot measure whether a pattern helped or harmed.

Recommended direction:

- Feed selected patterns deterministically into the Foundry prompt before generation.
- Hard-fail or retry a Foundry result if no required pattern lookup occurred for a covered op.
- Remove alternate paths that let the agent read knowledge files without attribution.

## Additional Improvements

### 13. Documentation has drifted from the implementation

Evidence:

- README and architecture docs describe a four-agent LLM flow, but Assayer is mostly deterministic in the current implementation.
- Some docs mention Haiku-style Assayer usage while config and code use other models and deterministic tools.
- Foundry and Surgeon skill append behavior in config appears disabled, while docs imply those skills are central.

Why this matters:

The project thesis is strong, but stale docs make it harder to evaluate what is actually implemented versus intended. They also make debugging architecture coverage harder.

Recommended direction:

- Split docs into "implemented today" and "planned direction".
- Keep the README aligned with actual model assignments, deterministic stages, and contract support.
- Include a short Windows/WSL setup section because this environment has special failure modes.

### 14. CLI checkpoint paths are more cwd-sensitive than the docs imply

Evidence:

- The CLI appears to check `cli.checkpointPath` directly before some repo-root resolution paths.
- Current artifacts contain a mixture of absolute Windows and WSL paths.

Why this matters:

On Windows/WSL, the same command can work or fail depending on whether it is launched from the repo root, parent directory, PowerShell, CMD, or WSL bash.

Recommended direction:

- Resolve relative checkpoint paths against the repo root consistently.
- Print the resolved checkpoint path at startup.
- Store normalized paths in the checkpoint metadata.

### 15. Quantization scale approximation can silently produce bad constants outside expected ranges

Evidence:

- Scale approximation searches bounded multiplier/shift ranges.
- If the target scale is outside the representable range, the fallback can keep a default/best value rather than failing loudly.

Why this matters:

For broader neural-network support, calibration ranges will vary. Silent approximation failure is much worse than a clear unsupported-scale error because it can produce RTL that looks structurally valid but computes the wrong values.

Recommended direction:

- Fail loudly when approximation error exceeds a configured threshold.
- Emit the chosen multiplier, shift, error, and target scale into module reports.
- Consider wider constants or per-layer quantization constraints for models outside the current range.

## Suggested Priority Order

1. Stabilize the Windows toolchain path first: Icarus output path, no-diagnostic crash classification, and Windows/WSL path normalization.
2. Make the failure taxonomy safe: API limits, classifier outages, and tool crashes must not become permanent contract or architecture knowledge.
3. Make contracts executable end to end: plan, adapter, latency, golden transform, testbench behavior, and Vivado expectations must all line up.
4. Protect best-known artifacts so auto-repair cannot make the final state worse than an earlier attempt.
5. Tighten auto-improve knowledge: contract-filter docs and enforce attributed pattern usage while keeping probationary availability as an intentional policy.
6. Expand or loudly reject unsupported ONNX semantics, especially grouped/depthwise/dilated conv and full padding.

## Bottom Line

The system is not fundamentally broken, but the current auto-improve loop can learn from the wrong failures. That is the dangerous part. The architecture will get much stronger if the pipeline first becomes strict about evidence: distinguish infrastructure from RTL bugs, verify each contract with real transformed data, and only persist learned knowledge after trustworthy end-to-end success.
