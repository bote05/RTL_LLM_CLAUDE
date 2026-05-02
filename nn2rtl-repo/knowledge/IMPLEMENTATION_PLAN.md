# Implementation Plan — Pattern-Library MCP Tool for Foundry/Surgeon

**Goal:** give Foundry and Surgeon a lookup tool that returns op-type-specific RTL patterns + reference implementations, so they adapt proven structures instead of hallucinating them. Covers Tiers 0, 1, 2 — deliberately stops short of handwritten parameterized library (Tier 3).

**Self-improvement status:**
- Phase 1 foundation is implemented: `PIPELINE_CONFIG.self_improve` / `NN2RTL_SELF_IMPROVE`, and `patterns/` + `references/` are tiered into `protected/`, `active/`, `probationary/`, and `archive/`.
- Phase 2 failure classifier is implemented: every failed module gets a retry-policy category (`code_bug`, `architectural_fit`, `unknown`) plus violated resource/constraint extraction for architectural-fit failures.
- Phase 3 Retrospector is implemented behind self-improve mode: after retry exhaustion on a `code_bug`, it reads the full recorded failure history, Foundry RTL versions, the active knowledge doc, and the original spec, then injects advisory JSON back into the same Foundry session via SDK `resume` for one final attempt. One call max per module/contract.
- Phase 4 doc lifecycle is implemented behind self-improve mode: generated docs and references are written to `probationary/`, promoted after the configured success threshold, archived on probationary failure, and replaced by archive+new-doc rather than in-place edits.
- Phase 5 failure response is implemented behind self-improve mode: terminal `code_bug` and `architectural_fit` failures run Retrospector, exhausted contracts are flagged in `output/contract_state.json`, alternatives are tried in `flat-bus -> tiled-streaming -> dram-backed` order, and all-contract exhaustion writes a manual-correction report plus human-escalation log.
- Phase 6 new doc/reference creation is implemented behind self-improve mode: when the selected contract or technique has no doc coverage, Foundry receives `create_new_doc_request` with the spec, closest same-family local docs/references, and failure context; external retrieval is disallowed, and the successful doc/reference enters `probationary/` tagged with `contract_id` / `contract_key`.

**Phase 7 status:** contract-specific infrastructure is now split into
`contracts/<contract_name>/` folders for `flat-bus`, `tiled-streaming`,
`dram-backed-weights`, `activation-double-buffering`, and `weight-tiling`.
Each folder owns metadata, a testbench template, a golden-vector adapter, and
a latency checker. `sdk/contracts.ts` is the shared selection logic.

**Phase 8 status:** Phase 8.1 doc-coverage guard is implemented. Foundry and
Surgeon are now only asked to emit `draft_doc` (the `{module, draft_doc}`
wrapper schema) when no existing pattern doc covers the layer's
`(contract_id, op_type, kernel)` tuple. The shared lookup is
`findCoveringDoc(state, layer)` in `sdk/orchestrate.ts`, used by both the
guard and `maybeBuildCreateNewDocRequest` so the two stay in lockstep.
On covered runs the orchestrator logs `self_improve_doc_request_skipped`
with the covering doc's tier and path. Side effects: (a) probationary docs
no longer accumulate one timestamp-suffixed duplicate per successful run on
a covered contract+kernel — so promotion-by-N-uses can finally trigger;
(b) Foundry's malformed-final-message rate drops on covered runs because
the wrapper schema is not requested. Phase 8.2 (full-pipeline regression
across all 17 layer-1 modules with self-improve ON vs OFF) is still
outstanding; the 3-module spike on `auto-approve` post-merge produced
identical Vivado synthesis numbers (LUT/FF/DSP/BRAM/Fmax) on both passes.

**Context to read before starting:**
- `ARCHITECTURE.md` — especially the "Known Bottleneck — Spatial Convolutions Do Not Close Reliably" section (near the end). Documents what we've tried and the exact failure modes.
- `nn2rtl-plugin/agents/foundry.md` — current Foundry system prompt and pinned template.
- `nn2rtl-plugin/agents/surgeon.md` — Surgeon's repair rules, including the retired invariant-tag policy.
- `knowledge/references/protected/conv1x1_passing_reference.v` — the historical pointwise reference that passed the earlier flow.
- `scripts/golden_impl.py::compute_conv2d_latency_cycles` — the authoritative latency formula. Do not change; pattern files must be consistent with it.
- `mcp/tools.ts` + `mcp/server.ts` — existing MCP tools for reference. New tool lives here.

**What NOT to do:**
- Do not write Verilog `.v` files directly under `output/rtl/` — always via `write_verilog` MCP tool (the Foundry/Surgeon path handles this).
- Do not modify `tb/static_verilator_tb.cpp` or `scripts/golden_impl.py` semantics without updating the contract infrastructure tests. LayerIR schema changes must stay mirrored between `sdk/` and `mcp/`.
- Do not add handwritten parameterized operator library modules (that would be Tier 3; deferred).

---

## Tier 0 — Sanity check: does Foundry use a tool when we give it one? (~0.5 day)

**Goal:** prove the MCP-tool integration path works end-to-end before investing in pattern content. If Foundry ignores the tool, the rest of the plan is wasted.

### Steps

1. **Create MCP tool `get_rtl_patterns`** in `mcp/tools.ts`:
   - Signature: `async function get_rtl_patterns(op_type: string, kernel_h?: number, kernel_w?: number): Promise<{ pattern_markdown: string; reference_verilog: string | null; license_notice: string | null }>`
   - For Tier 0, hardcode **one** branch: when `op_type === "conv2d"` and `kernel_h === 1 && kernel_w === 1`, read and return `knowledge/references/protected/conv1x1_passing_reference.v` plus a short markdown preamble explaining it's a proven-passing 1×1 pointwise reference.
   - All other op_types return `{ pattern_markdown: "No pattern available for this op_type yet. Proceed with foundry.md rules.", reference_verilog: null, license_notice: null }`.

2. **Register the tool in `mcp/server.ts`** with the name `mcp__nn2rtl-tools__get_rtl_patterns`, a clear JSON-schema description, and Zod validation for the return shape (add a schema to `mcp/schemas.ts`).

3. **Add the tool to Foundry's allowedTools** in `sdk/orchestrate.ts`:
   - `AGENT_MCP_TOOLS.foundry` currently has `["mcp__nn2rtl-tools__write_verilog"]`. Add `"mcp__nn2rtl-tools__get_rtl_patterns"`.
   - Also add it to `AGENT_MCP_TOOLS.surgeon` — Surgeon benefits too.

4. **Add an instruction to `foundry.md`** near the top rules:
   > Before emitting any Verilog, call `mcp__nn2rtl-tools__get_rtl_patterns` with the `op_type` from the `LayerIR` and the kernel dimensions from `weight_shape[2]` / `weight_shape[3]`. Use the returned `pattern_markdown` as architectural guidance and the returned `reference_verilog` (when non-null) as a structural starting point — adapt its parameter values to this specific `LayerIR` rather than regenerating from scratch.

   Same instruction in `surgeon.md` near the top: Surgeon should call the tool when diagnosing synth or sim failures, to see the canonical pattern for that op.

5. **Typecheck both packages:** `cd sdk && npm run typecheck && cd ../mcp && npm run typecheck`.

6. **Run the measurement:**
   - Clean pipeline state: `rm -f output/pipeline_state.json output/reports/run_log.jsonl output/reports/pipeline_summary.json output/rtl/layer1_0_conv1.* output/tb/layer1_0_conv1.* output/reports/layer1_0_conv1.*`
   - Regenerate goldens: `python scripts/generate_golden.py checkpoints/resnet50_int8.pth`
   - Write checkpoint fingerprint: `output/layer_ir.json.checkpoint` content = `C:\Users\User\Desktop\RTL_LLM_CLAUDE\nn2rtl-repo\checkpoints\resnet50_int8.pth\n`
   - Launch: `cd sdk && npm run pipeline -- ../checkpoints/resnet50_int8.pth --only layer1_0_conv1` (run in background)

7. **Measurement criteria** (read `output/reports/run_log.jsonl` after completion):
   - **Did Foundry call `get_rtl_patterns`?** Check the `agent_result` payload's `modelUsage` for tool-use traces or add a side-log in the MCP tool's implementation that writes to `output/reports/tool_calls.jsonl` on every invocation.
   - **Did Foundry's output pass sim + synth first shot (0 Surgeon attempts)?**
   - **Cost ≤ $1.00?**

### Go/no-go

- **Go to Tier 1** if Foundry called the tool at least once AND first-shot output passed sim. The reference Verilog doesn't need to be used verbatim — just observed in the tool-call log.
- **No-go** if Foundry ignored the tool despite the instruction. Redesign: either make the tool call mandatory via the HARD CONTRACT in `buildDelegationPrompt`, or try a different prompting pattern. Do not proceed to Tier 1 content work until this is resolved.

### Tier 0 deliverables

- `mcp/tools.ts` exports `get_rtl_patterns`
- `mcp/server.ts` registers it under the `mcp__nn2rtl-tools__get_rtl_patterns` name
- `mcp/schemas.ts` has a Zod schema for the return value
- `sdk/orchestrate.ts` `AGENT_MCP_TOOLS` lists the tool for Foundry and Surgeon
- `foundry.md` and `surgeon.md` have a single added paragraph instructing tool use
- One pipeline run measurement recorded (pass/fail, tool-call count, cost)

---

## Tier 1 — Write pattern markdown files from session knowledge (2–3 days)

**Goal:** build the full protected `knowledge/patterns/` library without depending on external sources. Use what this repo already knows: `ARCHITECTURE.md`'s failure-mode catalog, `foundry.md`'s pinned FSM template, `conv1x1_passing_reference.v`, and the `golden_impl.py` latency formula.

### File list

Create under `knowledge/patterns/protected/`:

```
01_context.md               Interface contract, canonical 7-signal ports, packed buses,
                            INT8 quantization formula, $readmemh rules, write_verilog
                            path convention. Referenced by every other pattern file.
02_conv1x1.md               Pointwise conv. Uses conv1x1_passing_reference.v as the
                            canonical example. Covers MP/OC_PASSES, serialized weight
                            reads, no line buffer / no window / no drain.
03_conv3x3_pad1.md          Spatial 3×3 with padding=1. Covers line buffer + shift-
                            register window, right-edge padding in ST_STREAM (wrap at
                            IW-1+PW), bottom-edge ST_DRAIN, fmm=0 sim-stall failure mode.
04_conv7x7_pad3.md          Stem 7×7 stride=2 pad=3. Same structure as 03 with bigger
                            kernel. Documents the fmm=7122 right-edge bug we never
                            closed — so Foundry knows to pay extra attention there.
05_add_quantized.md         INT8 quantized add: data_in has lhs+rhs packed, output is
                            lhs*lhs_scale + rhs*rhs_scale, re-quantized to INT8.
06_relu.md                  Trivial: data_out = max(data_in, 0) saturated to INT8.
07_maxpool.md               Line buffer + KH×KW compare tree, similar FSM to conv but
                            no MAC / no weights.
08_common_bugs.md           Union of observed failure modes: drain-exit bug,
                            right-edge padding off-by-one, MAC window indexing
                            (window[ic][kh][kw] swap), weights_packed OPT_MEM rejection,
                            non-constant meminit, verilator hang from partial outputs.
                            Each entry: symptom → diagnosis → fix.
```

### Required structure for each pattern file

Every `02-07_*.md` file must contain, in this order:

1. **When to use** — exact `op_type` + kernel match criteria
2. **Latency contract** — the formula branch from `golden_impl.py::compute_conv2d_latency_cycles` that applies
3. **Required FSM states** — list + allowed transitions
4. **Required register declarations** — `acc`/`biased`/`scaled` (sized `[0:MP-1]`), `window`/`line_buf`/`cur_row` if applicable, `k_counter`/`lane_counter`/`oc_group` counters
5. **Weight / bias convention** — `$readmemh`, Vivado-friendly ROM reads, optional `weight_bank_paths`, layout `weights[oc*K_TOTAL + k]`
6. **Known failure modes** — cross-reference to `08_common_bugs.md`
7. **Reference Verilog skeleton** — ~50 lines of canonical structure with placeholders

### MCP tool upgrade

Expand `get_rtl_patterns` to dispatch by `op_type` + kernel dimensions:

```typescript
function resolvePatternPath(op_type: string, kh?: number, kw?: number): string[] {
  const tiers = ["protected", "active", "probationary"];
  const tiered = (fileName: string) =>
    tiers.map((tier) => `knowledge/patterns/${tier}/${fileName}`);
  // Always include context + common_bugs
  const base = [...tiered("01_context.md"), ...tiered("08_common_bugs.md")];
  if (op_type === "conv2d") {
    if (kh === 1 && kw === 1) base.push(...tiered("02_conv1x1.md"));
    else if (kh === 3 && kw === 3) base.push(...tiered("03_conv3x3_pad1.md"));
    else if (kh === 7 && kw === 7) base.push(...tiered("04_conv7x7_pad3.md"));
  } else if (op_type === "add")      base.push(...tiered("05_add_quantized.md"));
  else if (op_type === "relu")       base.push(...tiered("06_relu.md"));
  else if (op_type === "maxpool")    base.push(...tiered("07_maxpool.md"));
  return base;
}
```

Tool returns the concatenated contents of the matched files plus `conv1x1_passing_reference.v` when the 1×1 branch hits.

### Measurement protocol for Tier 1

Run the pipeline on **4 different LayerIRs** (at least one of each: pointwise, spatial-pad1, add, relu) and record:
- First-shot Foundry pass rate
- Mean Surgeon attempts per layer
- Total cost per passing module

Compare against the pre-Tier-1 baseline documented in `ARCHITECTURE.md`:
- Pre-Tier-1: 1×1 passes first-shot, 3×3 never closes in 3 Surgeon attempts
- Target: ≥80% first-shot pass rate across the 4 test layers

### Tier 1 deliverables

- 8 markdown pattern files under `knowledge/patterns/protected/`
- Expanded `get_rtl_patterns` dispatching to them
- Measurement report under `knowledge/measurements/tier1_results.md` with per-layer cost, attempts, pass/fail

### Go/no-go for Tier 2

- **Skip Tier 2** if Tier 1 hits ≥90% first-shot pass rate. Diminishing returns from external references.
- **Do Tier 2** if Tier 1 is at 50–80%. External references can close the gap on spatial convs specifically.
- **Reconsider architecture** if Tier 1 is below 50%. The tool mechanism isn't enough; may need Tier 3 (handwritten parameterized library).

---

## Tier 2 — Extract reference patterns from NVDLA / SAURIA / Efficient-FPGA (1 week)

**Goal:** upgrade the Tier 1 pattern files with real-world references from production accelerators. Use WebFetch to pull specific files from GitHub raw URLs.

### Sources (in priority order)

| Repo | File | URL template | License |
|------|------|--------------|---------|
| **SAURIA** (BSC) | `rtl/sauria_core/sauria_top.sv` and conv engine | `raw.githubusercontent.com/bsc-loca/sauria/main/<path>` | Confirm first (Apache 2.0 / Solderpad expected) |
| **Efficient-FPGA-CNN** | `convolver.v` and `line_buffer.v` | `raw.githubusercontent.com/Mattjesc/Efficient-FPGA-CNN-Accelerator/main/<path>` | Confirm first (MIT expected) |
| **NVDLA** | `vmod/nvdla/cmac/NV_NVDLA_cmac.v`, `NV_NVDLA_cmac_core.v`, and `cdp/` top files | `raw.githubusercontent.com/nvdla/hw/nvdlav1/<path>` | NVIDIA Open NVDLA License |

**Procedure for each source:**

1. **Verify license** — fetch `LICENSE` or `LICENSE.md` from the repo root first. Record terms in `knowledge/references/protected/LICENSES.md`.
2. **Fetch the target files** via WebFetch. Save raw content to `knowledge/references/probationary/<source>_<filename>_raw.v`.
3. **Distill** — write a cleaned-up Verilog file at `knowledge/references/probationary/<source>_<op>_distilled.v` with:
   - Project-specific macros (`NV_NVDLA_*`, `SAURIA_*`) replaced with their effective values
   - Bus protocol wrappers stripped (we use our 7-signal canonical interface)
   - INT8 quantization path adapted to our `$readmemh` + scale-factor convention
   - Aggressive comments explaining adaptations
4. **Integrate into pattern files** — each `03_*.md` / `04_*.md` / `07_*.md` gets a new "Real-world reference" section citing the distilled file and line ranges, plus a 20–40-line annotated snippet showing the key mechanism.

### What each source contributes

- **SAURIA** → `02_conv1x1.md` (GeMM systolic has clean pointwise pattern), `08_common_bugs.md` (their TB catches bugs ours doesn't)
- **Efficient-FPGA convolver.v** → `03_conv3x3_pad1.md` (line buffer + window shift, readable)
- **NVDLA cmac** → `04_conv7x7_pad3.md` (stride-2 + padding handling in production silicon)

### Skip Gemmini for Tier 2

Gemmini is Chisel (Scala). Readable for algorithms, not extractable as Verilog without running their elaboration flow. Defer unless Tier 1+2 combined still hits issues that specifically need systolic-array patterns.

### Tier 2 deliverables

- `knowledge/references/protected/LICENSES.md` with source licenses recorded
- Raw fetched files under `knowledge/references/probationary/*_raw.v` (kept for audit/provenance)
- Distilled adapted files under `knowledge/references/probationary/*_distilled.v`
- Updated pattern markdown files (03, 04, 07 minimum) with real-world reference sections
- MCP tool updated to return `reference_verilog` arrays (multiple references per op when available), not just a single file

### Final measurement

Run the pipeline on the same 4 LayerIRs from Tier 1. Report the delta.

---

## Cross-cutting infrastructure (apply during any tier)

All four items below are universal, not ResNet-specific. They encode lessons learned from the failures documented in `ARCHITECTURE.md`. Do them alongside Tier 0 — they're small but load-bearing.

### CX-1 — Verilator simulation timeout

**Problem observed:** a Surgeon edit produced an FSM that partially-but-not-reliably fired `valid_out`. The TB's `hang_budget` only fires on *complete* silence, so a partially-functioning DUT hangs forever. One occurrence burned 50+ minutes of wall-clock before manual kill.

**Fix, applied across every layer of the stack:**

1. **`mcp/tools.ts`:**
   - Add `export const VERILATOR_SIM_TIMEOUT_MS = 10 * 60 * 1000;` near the external-tool timeout constants.
   - Pass `timeout: VERILATOR_SIM_TIMEOUT_MS` to the `runtime.commandRunner(binaryPath, [sidecar_path], { ... })` call inside `run_verilator`.
   - On timeout, return a `VerifResult` with `status: "fail"`, `status_class: "sim_stalled"`, `failure_class: "verilator_timeout"`, and a `fix_hint` that names the timeout duration and asks Surgeon to look for output-loop bugs.

2. **Twin-protected schemas:** add `"verilator_timeout"` to the `failureClassSchema` enum in both `sdk/schemas.ts` and `mcp/schemas.ts`. Run `scripts/check-twins.mjs` to confirm they match.

3. **Types:** add `"verilator_timeout"` to the `FailureClass` union in `sdk/types.ts` and `mcp/types.ts`.

4. **Surgeon taxonomy:** `nn2rtl-plugin/agents/surgeon.md` already has a failure-class diagnosis section. Add `verilator_timeout` with the rubric: "DUT compiled but simulation never terminated. Look for states that fire `valid_out` intermittently — the hang_budget only catches total silence. Check FSM exit conditions."

5. **Repair brief:** `buildSurgeonRepairBrief` in `sdk/orchestrate.ts` already pattern-matches on `failure_class`. Add a branch for `verilator_timeout` that tells Surgeon: "Do not assume RTL is partially correct — a timeout means the FSM is structurally wrong enough that it can never reach the end of the output stream."

6. **Tests:** add a unit test in `sdk/test/orchestrate-flow.test.ts` that fakes a timeout `VerifResult` and confirms the orchestrator routes it to Surgeon the same as other sim failures.

**Not hardcoded. Applies to any network any layer.**

### CX-2 — Bus-width capability check (fail-fast before Foundry)

**Problem anticipated:** at ResNet-50 L3/L4, `input_width_bits` and `output_width_bits` scale to 8 192 / 16 384 bits. Foundry generating correct bit-slicing across that width is an LLM accuracy problem we have no evidence it can solve. Burning Foundry+Surgeon attempts on a layer beyond pipeline capability is pure waste.

**Fix:**

1. **`sdk/config.ts`:** add `MAX_SUPPORTED_BUS_BITS: 4096` to `PIPELINE_CONFIG`. The exact constant is a judgment call — 4096 bits = 512 channels INT8, which covers ResNet-50 conv/relu outputs up to and including L2. For add layers, the gate must check each operand width (`input_width_bits / 2`) plus `output_width_bits`, not the concatenated lhs+rhs `data_in` width. Layers beyond that need tiled channel streaming (deferred).

2. **`sdk/orchestrate.ts`:** before dispatching Foundry (i.e. at the top of `tick()` or inside `invoke_foundry`'s handler), check `layer.output_width_bits > MAX_SUPPORTED_BUS_BITS` and the effective input stream width (`layer.input_width_bits` except `layer.input_width_bits / 2` for add). On exceed:
   - Emit a `VerifResult` with `status: "fail"`, `failure_class: "architectural_unsupported"`, and a `fix_hint` reading exactly `"Layer <module_id> has bus width <N> bits exceeding MAX_SUPPORTED_BUS_BITS=<M>. Tiled channel streaming is not yet implemented. Skip or extend the pipeline before retrying."`
   - Transition the module directly to `fail_abort` — **do not route to Surgeon**. Surgeon cannot fix a capability gap.

3. **Schema work:** add `"architectural_unsupported"` to `failureClassSchema` in both twin schemas + types.

4. **CLI reporting:** the pipeline summary should separate architectural_unsupported modules from actual RTL failures, so the operator sees the gap as a tooling limit, not a broken module.

5. **Tests:** a unit test that constructs a LayerIR with `output_width_bits = 8192` and confirms the module fail_aborts with the right failure_class and without Surgeon ever being invoked.

**Not hardcoded to ResNet. Any network with wide layers hits this. Change `MAX_SUPPORTED_BUS_BITS` when tiled streaming ships.**

### CX-3 — Structural preflight extensions

**Problem observed:** existing `preflightVerilogModule` only checks port declarations. Surgeon has introduced RTL that parses cleanly but is structurally wrong (e.g. `weights_packed` memory that Vivado cannot infer as ROM, purely combinational window rebuilds that blow up synth cones, missing output-counter guards that cause Verilator hangs).

**Fix — extend `preflightVerilogModule` with five checks derivable purely from LayerIR fields:**

1. **Spatial conv requires line buffer.** If `op_type === "conv2d" && weight_shape[2] * weight_shape[3] > 1`, the RTL must contain a declaration matching `/reg\s+(?:signed\s+)?\[[^\]]+\]\s+line_buf\s*\[/` (accept any width, require the array-of-arrays shape). Violation → `structural_preflight_failed: line_buffer_missing`.

2. **Spatial conv requires registered window.** If `op_type === "conv2d" && weight_shape[2] * weight_shape[3] > 1`, the RTL must contain both:
   - a `reg ... window ...` declaration
   - at least one `window[...] <=` assignment inside a `always @(posedge clk...)` block (not a `wire` + `assign`, not a purely combinational `always @*`).
   
   Parsing note: this requires scanning the `always @(posedge clk*)` block bodies for `window[...]` on the LHS of `<=`. Doable with the existing ANSI-port-parse infrastructure plus a second-pass regex on always-block contents. Violation → `structural_preflight_failed: window_not_registered`.

3. **No packed weight array initializers.** Any `$meminit`-style pattern where the initial value of `weights` is derived from anything other than a direct `$readmemh` call is rejected. Specifically: reject any `weights_packed`, `initial weights[...] = <expression>`, or `assign weights[...] = ...` constructs. Violation → `structural_preflight_failed: weights_packed_forbidden`. Cross-reference the memory-inference rule in `foundry.md`.

4. **Weight and bias must use `$readmemh`.** Require at least one `$readmemh("...", weights)` and at least one `$readmemh("...", biases)` inside an `initial` block. Violation → `structural_preflight_failed: readmemh_missing`.

5. **Output counter / completion guard must exist for frame-traversing ops.** Spatial conv and maxpool must have a state or condition that strictly bounds the total number of `valid_out` pulses. Approximated via a regex scan for `out_row|out_col|outputs_emitted` registers declared and updated, or by recognizing `coord_scheduler`. Pointwise 1x1 conv, ReLU, and add are intentionally excluded because they are one-output-per-accepted-input and a frame-level counter breaks back-to-back frames. Violation → `structural_preflight_failed: output_counter_missing`.

**On any violation:** return a `VerifResult` immediately with the exact violation name in `failure_class` and the offending source line range in `fix_hint`. This feeds into Surgeon the same as sim/synth failures, so Surgeon knows exactly what structural rule it broke.

**None of these reference ResNet or specific network shapes.** All derive from LayerIR fields (`op_type`, `weight_shape`) + general RTL safety rules.

### CX-4 — License tracking

Maintain `knowledge/references/protected/LICENSES.md` from the start. Record every external source's license and any restrictions relevant to commercial product use.

---

## Handwritten infrastructure — `coord_scheduler` component

**Why this is in-scope (and why it's not Tier 3):** the coordinate FSM — row counter, column counter with `IW-1+PW` wrap, output-fires predicate with stride/padding divisibility, termination by `outputs_emitted == OH*OW` — is the single most-bug-prone component in the spatial-conv path. Our 6 architectural iterations (documented in `ARCHITECTURE.md`) all touched this logic; the unresolved `fmm=7122` right-edge bug lives here. Every Foundry attempt reinvents it, with variable correctness.

This module is *infrastructure*, analogous to `tb/static_verilator_tb.cpp` — handwritten plumbing that every operator uses, not a per-operator datapath. That's why it belongs in 0/1/2 scope while the full parameterized operator library is deferred to Tier 3.

### Specification

**File:** `rtl_library/coord_scheduler.v`

**Parameters (all sourced from LayerIR):**
- `IH, IW` — input spatial dimensions
- `OH, OW` — output spatial dimensions
- `KH, KW` — kernel dimensions
- `SH, SW` — stride
- `PH, PW` — padding

**Ports:**
- `clk, rst_n` — standard
- `start` — one-shot pulse to begin a new input frame
- `stall_in` — external backpressure (raised while MAC pipeline is busy for the current output pixel)
- `in_row, in_col` — current input position, both outputs
- `output_fires` — combinational signal, high on cycles where an output pixel completes
- `in_frame_done` — high exactly one cycle after the last input has been absorbed
- `out_frame_done` — high exactly one cycle after `outputs_emitted == OH*OW`
- `outputs_emitted` — output counter, bounded to `OH*OW`

**Termination rule (critical):**
- `out_frame_done` goes high when `outputs_emitted == OH*OW`. Not when `in_row > IH-1+PH`. The old drain-exit condition has been the single worst source of bugs across every spatial conv iteration. The scheduler fires `output_fires` exactly `OH*OW` times per frame and then stops, independent of how the input counters end up.

**Internal structure:**
- `in_row` / `in_col` counters with `IW-1+PW` wrap (matches canonical FSM in `foundry.md`).
- Combinational `row_num = in_row + PH - (KH-1)`, `col_num = in_col + PW - (KW-1)`.
- `row_trigger = (row_num >= 0) && (row_num % SH == 0)`.
- `col_trigger = (col_num >= 0) && (col_num % SW == 0)`.
- `output_fires = row_trigger && col_trigger && outputs_emitted < OH*OW`.
- `outputs_emitted` increments on every `output_fires` cycle, saturates at `OH*OW`.

### Integration into `foundry.md`

Add a rule: **for any `op_type == "conv2d"` with `KH*KW > 1`, the generated module must instantiate `coord_scheduler` and wire its outputs into the MAC/window FSM. Do not invent coordinate logic in the top-level always block.**

Also: the RTL preflight (CX-3) can be extended with a sixth check — "spatial conv must instantiate `coord_scheduler`" — once this module lands, making the rule enforceable, not just documented.

### Verification

Write a minimal Verilator testbench under `rtl_library/test/coord_scheduler_tb.cpp` that drives a few parameterizations (`layer0_0_conv1`'s 7×7-s2-p3 + `layer1_0_conv2`'s 3×3-s1-p1 + `layer1_0_conv3`'s 1×1) and asserts:
- Exact number of `output_fires` pulses per frame
- Correct ordering (row-major, column-within-row)
- Terminates cleanly — `out_frame_done` exactly once per frame

This testbench runs in CI the same way the static testbench does. Any change to `coord_scheduler.v` must keep it passing.

### Why this is universal

- Works for 1×1 (trivially, every input cycle is an output).
- Works for 3×3, 5×5, 7×7 — any KH/KW.
- Works for stride 1, 2, 4 — any SH/SW.
- Works for padding 0, 1, 2, 3 — any PH/PW.
- No network-specific knowledge embedded. ResNet, MobileNet, YOLO, custom — all instantiate the same module with different parameters.

### Scheduling

Do this **between Tier 1 and Tier 2**, after pattern files are in place but before extracting external references. Rationale: once Foundry is instantiating `coord_scheduler`, many of the spatial-conv pattern files' "known failure modes" sections become obsolete (those bugs literally cannot occur). Starting Tier 2 without the scheduler means the extracted NVDLA/SAURIA references are still carrying obsolete drain-exit logic.

---

## Non-goals (deferred)

- **Tier 3** — parameterized handwritten operator library. Not needed if Tier 1+2 closes the gap.
- **RAG / embeddings / vector search** — explicitly rejected. Lookup table is sufficient.
- **Gemmini extraction** — Chisel source, defer unless Tier 2 proves insufficient.
- **Handling novel operators** (BatchNorm, GroupConv, DepthwiseConv) — outside current scope. Current LayerIR only has conv2d / add / relu / maxpool.

---

## Final deliverable

After all three tiers + cross-cutting + coord_scheduler are complete:

1. `knowledge/patterns/protected/*.md` plus readable generated tiers — 8 protected pattern files and any active/probationary successors
2. `knowledge/references/{protected,active,probationary}/*` — protected references plus generated raw/distilled external references with license attribution
3. `knowledge/measurements/*.md` — tier-by-tier measurement results
4. `rtl_library/coord_scheduler.v` + `rtl_library/test/coord_scheduler_tb.cpp` — handwritten coordinate FSM with its own testbench
5. `mcp/tools.ts::get_rtl_patterns` — production-ready lookup tool
6. `mcp/tools.ts` also has `VERILATOR_SIM_TIMEOUT_MS` (CX-1)
7. `sdk/config.ts` has `MAX_SUPPORTED_BUS_BITS` + orchestrator gate (CX-2)
8. Extended `preflightVerilogModule` with the five structural checks (CX-3)
9. `failure_class` enum extended with `verilator_timeout`, `architectural_unsupported`, `structural_preflight_failed` in both twin schemas
10. Updated `foundry.md` and `surgeon.md` with tool-use instructions + coord_scheduler instantiation rule
11. A short write-up in `ARCHITECTURE.md` replacing the current "Known Bottleneck" section with measured results across the new pipeline

**Success criterion:** the 17-module ResNet-50 run, with everything active, completes with ≥14 modules in `pass` state (82%) and total cost ≤ $40. Layers exceeding `MAX_SUPPORTED_BUS_BITS` report as `architectural_unsupported` and are counted separately, not as RTL failures.

---

## Execution order (reference)

A single sequential ordering that respects dependencies:

1. **CX-1** (Verilator timeout) — do first, blocks nothing, prevents wall-clock loss during measurement
2. **CX-2** (bus-width gate) — do second, prevents expense waste on out-of-scope layers
3. **CX-3** (structural preflight) — do third, makes Foundry/Surgeon failures actionable rather than mysterious
4. **CX-4** (license doc) — ongoing, start a stub during Tier 0
5. **Tier 0** — sanity check that Foundry calls tools
6. **Tier 1** — pattern markdown files
7. **coord_scheduler** — handwritten module + testbench + foundry.md integration + preflight rule #6
8. **Tier 2** — external reference extraction, pattern files updated with real-world citations
9. **Final 17-module ResNet-50 measurement** — records the success-criterion result
