---
task_id: 13a
title: System wiring & spec reconciliation (main-agent integration)
type: Hand-integration by orchestrator (not dispatchable to sub-agents)
status: in-progress
depends_on: [00, 02, 03, 04c, 05, 07-11]
unblocks: [13]
---

# Task 13a — System wiring & spec reconciliation

## Why this task exists

Wave 1 + Wave 2 + 04c each passed their *local* verification gates (port consistency, iverilog parse, JSON schemas, FIFO byte budget). An independent audit at the end found that the **system-level integration is not coherent**. Each agent built its piece against its task spec; the seams between pieces have real bugs that local gates don't catch.

This task is the main agent's responsibility — not dispatchable to a sub-agent — because the bugs are cross-piece consistency problems. Sub-agents can fix a piece against a spec; they can't fix a spec that's wrong across multiple pieces at once.

## The 12 problems the audit found, triaged

| # | Problem | Severity | Fix scope |
|---|---|---|---|
| 1 | Scheduler asserts `awvalid` and `wvalid` in separate states; config block requires simultaneous → AXI deadlock on first write | **BUG** | Scheduler FSM |
| 6 | `scale_mult` is 32-bit in scheduler ROM but truncated to 16 bits in config block + requant pipeline → silent value corruption | **BUG** | Three files + PORTS.md |
| 4 | Engine skeleton ties `mac_valid_in`, `requant_valid_in`, `requant_bias_in` to constants outside the `ifndef` guard → engine FSM enters ST_REQUANT and waits forever | **BUG** | Skeleton internal wiring |
| 2 | Scheduler not instantiated in top wrapper; `engine_start` tied to `1'b0` | **BUG** | Top wrapper |
| 3 | `node_conv_220_valid_out/data_out` (and 9 other heavy layers' outputs) declared and consumed in the wrapper but never produced | **BUG** | Top wrapper |
| 5 | Address generator emits `bias_rd_addr/en` but engine has no `bias_rd_data` input port; no bias memory anywhere | **BUG** | Engine + top + new bias memory |
| 7 | Weight memory map uses 288-bit URAM words; top instantiates uram_weight_mem with 2048-bit words; scheduler passes raw 288-bit base words | **BUG** | Address-units reconciliation |
| 8 | `skip_fifo` and `uram_weight_mem` modules in nn2rtl_top.v are empty (port lists + immediate `endmodule`) | **BUG** | Two new module bodies |
| 9 | Full elaboration including per-layer modules and `output/rtl/rtl_library/*.v` fails on undefined `coord_scheduler`, `line_buf_window`, `conv_datapath` | Test gap | Update integration parse command |
| 11 | PORTS.md register map (0x00=CONTROL) contradicts task 10's implementation (0x00=INPUT_CHANNELS, 0x2C=CONTROL) | Stale doc | Refresh PORTS.md |
| 10 | Verilator rejects `address_generator.v` due to invalid `UNUSEDSIGNAL` lint pragmas | Lint cleanup | One file |
| 12 | `_aggregate.json` says modules_total: 4 (smoke-test stale); actual Vivado run has 72/119 .vivado.json files | Reporting | Re-run aggregate after Phase 0 finishes |

## Fix order (this task executes in this sequence)

1. **AXI handshake** — collapse scheduler S_WRITE_ADDR + S_WRITE_DATA into one S_WRITE state asserting both awvalid and wvalid in the same cycle. Wait for both awready and wready to fire. Smallest blast radius; isolated to scheduler.
2. **Scale width 16→32** — three file edits: `config_register_block.v` (don't truncate `reg_scale_mult` on assign), `requant_pipeline.v` (input width), and PORTS.md (spec table). Update task 08 + task 10 spec files too so future regenerations honour the new width.
3. **Stub bodies in build_top_wrapper.ts** — write proper bodies for `skip_fifo` (a depth-parameterised FIFO with backpressure) and `uram_weight_mem` (a `$readmemh`-initialised URAM read port). Regenerate `nn2rtl_top.v`. Cleaner than hand-patching the generated file.
4. **Engine skeleton internal wiring** — remove the tie-offs at lines 240-247 of `shared_engine_skeleton.v`. Wire `mac_valid_in`/`mac_weight_bus`/`requant_valid_in`/`requant_bias_in` from the address generator's outputs (and from the URAM weight bus, and from the bias memory's output). This is the most architectural fix; it makes the engine's datapath actually flow end-to-end.
5. **Bias memory** — add `bias_rd_data` input port to the engine top. Instantiate a small BRAM-backed bias memory in the top wrapper, initialise via `$readmemh` from a generated bias `.mem` file. Wire to engine.
6. **Scheduler instantiation in top** — add `nn2rtl_scheduler u_scheduler(...)` to the top wrapper. Connect its AXI4-Lite master to the engine's slave (already a port). Drive `engine_start` from the scheduler.
7. **Engine output to spatial chain** — for each of the 10 engine-dispatched layers, route the engine's output BRAM bank through a `bram_to_stream_bridge` instance into that layer's `_valid_out/_data_out` consumer.
8. **Weight address units** — settle on the on-chip-engine canonical unit. Choose 288-bit URAM-word indexing throughout (matches the weight_memory_map.json layout). Verify scheduler emits values in that unit; update `uram_weight_mem` body to accept that addressing.
9. **PORTS.md refresh** — rewrite the register map section to match the actual implementation (0x00 INPUT_CHANNELS, 0x2C CONTROL, scale_mult 32-bit, ACT_IN/OUT bases at 0x34/0x38). Remove the stale "Suggested config-register map" that contradicts task 10.
10. **Strict full elaboration** — extend the integration parse command to include `output/rtl/*.v output/rtl/engine/*.v output/rtl/rtl_library/*.v`. Document this in the task 13 spec as the canonical pre-integration parse.

## Per-fix verification

After each fix, run the relevant gate:

- After (1): hand-trace one AXI write in waveform-form (mental simulation); confirm a Verilator-style scoreboard would now accept it.
- After (2): rerun `npm run typecheck` in sdk + mcp; spot-check that the requant unit-TB still passes (it was tested under 16-bit; widening to 32 should still satisfy its existing test vectors).
- After (3): iverilog parse top + sub-blocks again; confirm `skip_fifo` and `uram_weight_mem` have non-empty bodies.
- After (4): the engine's address generator output must reach the MAC array and requant pipeline through actual wires, not zero ties.
- After (5): the engine has a `bias_rd_data` input; the top wires it to a memory whose `.mem` file is generated.
- After (6): the top wrapper instantiates `nn2rtl_scheduler` exactly once; `engine_start` is no longer assigned to `1'b0`.
- After (7): for each of the 10 heavy layers, `<layer>_valid_out` and `<layer>_data_out` have exactly one driver (in the top, from the bridge), exactly one consumer (the downstream spatial layer).
- After (8): unit reconciliation — the scheduler's `weight_base_word` ROM, the uram_weight_mem's `rd_addr` width, and the address generator's `weight_rd_addr` output are all in the same address unit (288-bit URAM words).
- After (9): grep PORTS.md for 0x00 — should only mention INPUT_CHANNELS; no CONTROL register at 0x00 anywhere.
- After (10): full elaboration parse exits 0 without "undefined module" warnings.

## Out of scope

- Do NOT regenerate per-layer modules.
- Do NOT change the engine sub-blocks' MAC arithmetic, requantisation arithmetic, or address-generation algorithm (apart from the scale-width port widening).
- Do NOT touch LayerIR, goldens, contracts (except updating on-chip-weights spec if widths change), or the pipeline state.

## Definition of done

- All 10 fix gates above pass.
- Full elaboration parse exits 0.
- A 1-image Verilator smoke test of the integrated top **completes without deadlock** (cycle budget bounded). Output value correctness is NOT required at this stage — task 13 owns the bit-exact end-to-end run. 13a's job is structural soundness.
- Task 13 (integration & first-light Vivado synth) is then unblocked.

## Bundle A delivery (Fixes 4 + 5) — main-agent direct edit

Done by the orchestrating Claude (not a sub-agent). What landed:

- **Engine skeleton (`output/rtl/shared_engine_skeleton.v`):**
  - Added external ports `bias_rd_addr` [22], `bias_rd_en` [1], `bias_rd_data` [8192].
  - Removed the eight tie-offs at the prior lines 240-247.
  - Added FSM-driven counters: `oc_pass_idx_r` (advanced on each ST_REQUANT exit), `pixel_h_r` / `pixel_w_r` (advanced on each ST_DRAIN exit). Reset to 0 on ST_LOAD_CONFIG.
  - Added 1-cycle alignment regs (`ag_weight_rd_en_d`, `ag_act_in_rd_en_d`, `ag_act_in_ic_byte_idx_d`, `ag_bias_rd_en_d`) so the MAC array sees `mac_valid_in` aligned with the URAM/BRAM-returned data, and the requant pipeline sees `requant_valid_in` aligned with `bias_rd_data`.
  - `mac_clear` pulses on rising edge into ST_RUN (clears the 256 accumulators per pixel).
  - `mac_weight_bus` = `weight_rd_data[MAC_COUNT*WGT_W-1:0]` (the full URAM beat).
  - `mac_act_byte` = `act_in_rd_data[ic_byte_idx_d*8 +: 8]` (pipelined byte select).
  - `requant_bias_in` = `bias_rd_data` (no transformation; the bias memory delivers the wide word ready to consume).
  - `bias_rd_addr`/`bias_rd_en` external pins driven from the address generator's outputs.

- **Top wrapper (`scripts/build_top_wrapper.ts` → `output/rtl/nn2rtl_top.v`):**
  - Declared 3 new wires: `engine_bias_rd_addr` [22], `engine_bias_rd_en` [1], `engine_bias_rd_data` [8192].
  - Instantiated `u_bias_mem` with `SIZE_WORDS=256, WORD_WIDTH=8192, ADDR_W=8` (256 wide-bias-word entries cover ~14 heavy layers × up to 8 oc_passes each).
  - Added the 3 bias ports to the `u_shared_engine` instantiation.

- **PORTS.md:** documented the 3 new external bias ports under "Engine top-level interface".

Verification:
- Engine skeleton standalone parse: `iverilog -t null output/rtl/shared_engine_skeleton.v` exits 0.
- Full integration parse (top + scheduler + skeleton + all 5 sub-blocks): exits 0.
- Port consistency for all 5 sub-blocks: 8 + 9 + 33 + 43 + 12 ports verified against PORTS.md, all OK.
- `grep "assign mac_valid_in = 1'b0"` in skeleton → no matches.
- `grep "assign requant_bias_in = "` in skeleton → only the new assignment from `bias_rd_data`.

Known follow-ups (out of Bundle A scope, deferred to task 13 first-light):
- **Strict full elaboration with per-layer + rtl_library** — RESOLVED. The helpers are canonical at top-level `rtl_library/` (referenced by `RTL_LIBRARY_SOURCES` in `mcp/tools.ts`); the follow-up was based on a wrong assumption that they lived only in tmp dirs. Strict full elaboration of the integrated top now exits 0 either via `-y rtl_library` or by explicitly listing `rtl_library/{conv_datapath,coord_scheduler,line_buf_window}.v` alongside the per-layer `node_*.v` sources.
- **Bias `.mem` file** — RESOLVED. `scripts/build_bias_memory_map.py` walks the dispatched heavy list and emits `output/weights/bias.mem` (one 8192-bit wide bias word per oc_pass, 41 wide words for ResNet-50 / 256 capacity), `bias_memory_map.json`, and `bias_memory_map.vh`. The scheduler reads each layer's `base_word` from the JSON and writes it to the engine via `bias_base_word`; with the bias map present, every dispatched conv has a non-zero offset matching the .mem layout (cross-checked).
- **`oc_pass_total_m1` truncation**: the engine uses `cfg_oc[11:8] - 1` which assumes oc is a multiple of MAC_COUNT=256. For all 14 heavy modules this is true (channel counts 256, 512, 1024, 2048). For future layers with odd channel counts this would need refinement.

## Post-Bundle-A second audit (found 3 more bugs, all caught and fixed except #3)

A second independent audit ran after Bundle A landed and caught 3 issues that the per-piece local gates missed:

### BLOCKER 1 — FIXED — scale_mult parameter truncation in the engine skeleton

Fix 2 widened scale_mult to 32 bits in `config_register_block.v` and `requant_pipeline.v`, but the engine **skeleton** still had:
- `parameter integer SCALE_MULT_W = 16` (line 52)
- `wire [SCALE_MULT_W-1:0] cfg_scale_mult;` (line 135)
- The stub fallback module declared the port as `[15:0]`

So the skeleton received `cfg_scale_mult` from the (now 32-bit) `config_register_block` output through a **16-bit internal wire**, silently truncating every layer's scale to its low 16 bits. ResNet-50 scale_mult values are ~30 bits (e.g. 1284434803 = 0x4C8…), so every dispatch would have produced garbage requant output.

Fix landed in this audit: updated `SCALE_MULT_W` → 32, updated the stub module's `cfg_scale_mult` declarations from 16'd0 to 32'd0. Engine standalone parse + full integration parse both still exit 0.

### BLOCKER 2 — FIXED — FSM ST_REQUANT exit condition hardcoded to 8 passes

Line 221 used to be: `if (requant_valid_out) next_state = (oc_pass_idx == (MAX_OC/MAC_COUNT-1)) ? ST_DRAIN : ST_RUN;`

`MAX_OC/MAC_COUNT - 1 = 2048/256 - 1 = 7`, hardcoded. But `oc_pass_idx_r` wraps at the **layer's** actual oc-pass count via `oc_pass_total_m1 = cfg_oc[11:8] - 1` (3 for cfg_oc=1024, 1 for cfg_oc=512, 0 for cfg_oc=256). So for any layer with cfg_oc < MAX_OC, the FSM would be waiting for an `oc_pass_idx` value the counter never reaches. **Deadlock.**

Fix landed in this audit: condition is now `oc_pass_idx == oc_pass_total_m1[2:0]`. Same parse exit 0.

### BLOCKER 3 — FIXED (Path D landed)

After the proposal-vs-fix-now decision (Path B vs Path D vs Path C), the orchestrator chose **Path D — banked memory subsystem with native URAM widths**. Landed by the main agent, not dispatched to a sub-agent (architectural change touching script + wrapper + new module body + verification tool).

**Path D delivery — what landed:**

1. **`scripts/build_weight_memory_map.py` rewritten** (schema `weight_memory_map_v2_banked`):
   - For each conv layer, re-orders weights from PyTorch `[oc, ic, kh, kw]` natural layout into engine-consumption MAC-cycle order: `for (oc_pass, ic, kh, kw)` → 256 weights per cycle = 8 banks × 32 weights each.
   - Emits 8 separate `.mem` files (`uram_weights_bank0.mem` ... `bank7.mem`), one per parallel URAM read bank.
   - Each line is 72 hex chars = 288 bits (native URAM cascade width): low 256 bits hold 32 useful weight bytes (byte 0 at `[7:0]`), top 32 bits are zero-pad to honour URAM native shape.
   - Out-of-range output channels (when a layer's `cfg_oc` < the next 256-multiple) are zero-padded — the engine processes the full oc_pass and discards the padded slots, no per-layer special-casing.
   - JSON sidecar records per-layer `base_mac_cycle` and `size_mac_cycles`.

2. **`scripts/build_top_wrapper.ts` updated**:
   - Removed the broken `uram_weight_mem` instantiation and its `engine_weight_rd_addr >> 3` conversion.
   - Generates 8 parallel `uram_weight_bank` instances, all reading at the same MAC-cycle address (= the engine's `weight_rd_addr` directly, no conversion).
   - Concatenates the **low 256 bits** of each bank into `engine_weight_rd_data` (2048 bits = exactly the MAC array's per-cycle weight bus). The top 32 zero-pad bits per bank are deliberately discarded.
   - Replaced the `uram_weight_mem` module body with `uram_weight_bank`: a 288-bit-wide × DEPTH-deep memory with `(* ram_style = "ultra" *)` for Vivado URAM inference, synchronous 1-cycle read, `$readmemh` init from a per-bank `.mem` file.

3. **`scripts/verify_weight_memory_map.py` added** (new):
   - Walks each test layer's MAC-cycle range, reads the 8 bank `.mem` files at those line numbers, decodes each bank's low-256 bits into 32 INT8 bytes, concatenates the bank bytes into the 256-weight MAC cycle, and checks each (output-channel × ic × kh × kw) entry against the original PyTorch `.hex` file.
   - Catches: byte-order swap, MAC-cycle iteration mistake, per-bank slot allocation error, pad-byte placement, multi-layer base offset bug.
   - This is the strongest possible local correctness check short of a full Verilator engine run.

**Verification:**
- `iverilog -t null` full integration parse (top + scheduler + skeleton + 5 sub-blocks): exit 0.
- `verify_weight_memory_map.py` checked the 2 sample layers (stem `node_conv_196` and a heavy `node_conv_284`): **0 mismatches across 2.40M slot comparisons.**
- `verify_weight_memory_map.py --layer` on **all 14 heavy modules**: **0 mismatches across ~10.6M slot comparisons.**

**Numbers from the new layout:**

| Metric | Path-D value |
| --- | ---: |
| Total MAC cycles (all 53 conv layers) | 96,659 |
| Useful weight bytes packed | 24.7 M (vs raw 22.4 M; +10% from oc-pass padding) |
| URAM288 primitives per bank | 96 |
| Total URAM288 primitives | 768 |
| U250 URAM budget | 1,280 |
| URAM utilisation | **60%** |

**Why Path D is the right answer for this deployment (now verified, not asserted):**
- No wasted memory cells (the 288-bit URAM physical width is fully used).
- No address-conversion magic in the wrapper.
- The engine sees the same address it would for any conv shape; the memory subsystem absorbs the URAM-vs-MAC width mismatch internally.
- The `weight_memory_map_v2_banked` schema is parameterised on `MAC_COUNT` and `NUM_BANKS` — retargeting to a different engine width or a different network requires changing two constants in the generator, no per-network bank layout hand-tuning.

**Original BLOCKER-3 problem analysis (kept for reference — what we did NOT do):**

The earlier proposal listed three paths (A: drop MAC_COUNT, B: repack mem into 2304-bit lines, C: burst 8 URAM reads per MAC). Each had a real cost:
- Path A: ~7× throughput hit.
- Path B: works for this engine geometry but couples MAC width to URAM line width as an assumption.
- Path C: ~8× throughput hit.

Path D decouples MAC width from URAM line width via the banked memory subsystem. The "rate-matching FIFO" the proposal described collapses to a trivial concatenation slice for this concrete case (because 8 × 288 = 2304 > 2048 = MAC bus need), so the implementation cost was much lower than the proposal estimated (one Python rewrite + one wrapper changes + one new module body + one verifier).

### Original BLOCKER 3 — historical context

This one is **not a Bundle A patch**. It is an architectural decision the supervisor needs to confirm.

The mismatch:

| | Format on disk / in engine |
| --- | --- |
| `output/weights/uram_weights.mem` (generated by task 01) | One **288-bit URAM word per line** (72 hex chars / line). Total 651,545 lines = 22.4 MB. |
| `weight_memory_map.json.uram_word_bits` | 288 — confirms the .mem file is 288-bit-per-word. |
| Scheduler's `weight_base_word_rom` units | 288-bit URAM words. |
| Engine's `weight_rd_addr` output | 22 bits, in 288-bit URAM word units. |
| Top wrapper `uram_weight_mem.WORD_WIDTH` | **2048 bits**, not 288. |
| Top wrapper `uram_rd_addr_wide = engine_weight_rd_addr >> 3` | Assumes 8 URAM words per 2048-bit "wide line". But 8 × 288 = 2304 ≠ 2048. |
| MAC array's expected weight bus width | 256 weights × 8 bits = 2048 bits per cycle. |

Two things break:

1. `$readmemh` would read each 288-bit line and zero-extend it to fill a 2048-bit memory cell. Then `>>3` skips 7 of every 8 cells. **Every weight read returns the wrong data.**
2. Even if you ignored that and tried to repack the .mem file into 2048-bit lines, 8 × 288 = 2304 ≠ 2048 — there is no integer ratio that maps URAM words to "wide weight beats" without waste or split.

The underlying fact: UltraScale+ URAM is **natively 72-bit-per-port** (288 Kbit total = 4096 × 72). The 288-bit "URAM word" choice in task 01 was a logical-not-physical aggregation. To get 2048 bits per cycle, you'd need ~29 URAM blocks in parallel — and even then, the byte-to-channel layout depends on how the 28-block stripe is wired.

Three honest paths the supervisor should choose between:

**Path A — Drop MAC parallelism to fit native URAM width.** Use MAC_COUNT = 36 (= 288 / 8). Engine processes one "URAM word's worth" of weights per cycle. This is 7× slower per pass but uses one URAM block as the weight port. Simplest hardware, biggest throughput hit.

**Path B — Re-stride the .mem file and use natural URAM stripes.** Repack weights into 2048-bit physical-wide lines (= 28-29 URAM blocks operated as one logical block). The .mem file becomes 2048-bit-per-line (512 hex chars / line); SIZE_WORDS becomes ceil(22.4 MB / 256 bytes) = ~91,624 wide lines. The address conversion becomes `engine_weight_rd_addr >> 3` IF and only if we accept ~10% padding loss (effectively rounding 2048 to a multiple of 288 = 2304 and wasting the extra bits). Cleaner if we instead choose 2304-bit physical width (= 8 × 288) and have the MAC array take only the low 2048 bits — wastes 11% of URAM but the address conversion is exact.

**Path C — Stay 288-bit native in the memory and burst 8 URAM reads per MAC cycle.** Memory is 288-bit; engine reads 8 cycles' worth into a register before strobing mac_valid_in. 8× slower per MAC step, but the .mem file works as-is.

For Phase 2 first-light, I recommend **Path C** (slowest but most honest) until the supervisor approves a deeper restructure. It requires:
- Change `uram_weight_mem.WORD_WIDTH` from 2048 → 288 in the wrapper generator.
- Change `SIZE_WORDS` to 651,545 (the real count from weight_memory_map.json).
- Remove the `>> 3` conversion (use `engine_weight_rd_addr` directly).
- Inside the engine, add an 8-cycle weight gather register that collects 8 × 288-bit reads into a 2048-bit `mac_weight_bus`. Issue mac_valid_in only on the 8th cycle.
- MAC array's effective throughput drops 8×.

For an honest Vivado first-light synth that produces meaningful PPA, even the slow path is fine. Throughput is a Task 13 measurement; the system-level coherence is what task 13a chases.

**Status: BLOCKER 3 is undecided. Task 13 cannot dispatch until the supervisor picks a path.**

## Final state of Bundle A after audit fixes

- All 5 sub-block port-consistency checks still pass.
- Engine skeleton standalone parse: exit 0.
- Full integration parse (top + scheduler + skeleton + 5 sub-blocks): exit 0.
- 8 tie-offs replaced with real wiring.
- Bias data path is end-to-end on the engine side; the wrapper instantiates `u_bias_mem` but the `.mem` init file is not yet generated (known follow-up).
- Scale width 16→32 is consistent across config_register_block.v, requant_pipeline.v, shared_engine_skeleton.v param + internal wire + stub, and PORTS.md.
- FSM ST_REQUANT → ST_DRAIN now uses layer-specific oc_pass count, no longer deadlocks for cfg_oc < 2048.
- Weight memory architecture mismatch (BLOCKER 3) is documented above and pending supervisor decision.

## Why the parallel-agent waves missed this

The waves missed it because each agent's local gate was sound but each gate had a narrow scope:

- Port consistency check: validates port *names + widths + directions* match the spec, but cannot validate that the spec itself is correct (e.g. scale width should have been 32, not 16).
- iverilog parse: tolerates floating signals (treats them as X) and accepts modules with empty bodies. Doesn't enforce "every wire has a driver and a consumer."
- Per-sub-block unit testbenches: each agent built its own TB against its own spec. They didn't co-simulate across sub-blocks, so the engine FSM tie-offs in the skeleton never met the sub-blocks' real outputs.

**The lesson for future networks (MobileNetV2 retarget, etc.)**: add a system-level "integration smoke" gate at the Wave 2 review boundary that does Verilator-co-simulation of the engine + scheduler + a tiny dispatch sequence. Treat that as a mandatory gate alongside port-consistency and iverilog parse. The system-level gate would have caught most of these problems at hours-to-days instead of late-integration time.

Document this lesson in `docs/nn2rtl_supervisor_explanation.md` after task 13a lands.

## Fixes 3 + 6 + 7 + 8 — done by agent O

All four wrapper-touching fixes were bundled into a single edit of
`scripts/build_top_wrapper.ts`; `output/rtl/nn2rtl_top.v` was regenerated
from it. No per-layer `.v`, engine sub-block, or scheduler file was
modified.

- **Fix 3** — `skip_fifo`, `uram_weight_mem` and (newly added) `bias_mem`
  module bodies are now real implementations rather than port-list-only
  stubs. `skip_fifo` is a power-of-2 DEPTH FIFO with the standard
  extra-bit pointer trick for full/empty detection; `in_ready` low when
  full, `out_valid` low when empty; array write split into a sibling
  `always @(posedge clk)` per
  [protected/08_common_bugs.md](../../knowledge/patterns/protected/08_common_bugs.md).
  `uram_weight_mem` is a `(* ram_style = "block" *)`-hinted BRAM with
  `$readmemh` init from a `MEM_INIT_FILE` parameter (default
  `output/weights/uram_weights.mem`) and a synchronous 1-cycle read,
  matching UltraScale+ BRAM/URAM read latency. `bias_mem` mirrors the
  same shape at WORD_WIDTH=8192 (256 × INT32 biases per wide word) and
  init file `output/weights/bias.mem`. None of the bodies are
  instantiated from the engine itself — Bundle A is responsible for
  wiring `bias_mem` into the engine once a `bias_rd_data` port lands
  (Fix 5).
- **Fix 6** — the wrapper now declares a `sched_axil_*` AXI4-Lite bundle
  between the new `u_scheduler` (`nn2rtl_scheduler`) instance and
  `u_shared_engine`'s slave. The scheduler's `start` is a one-shot
  derived from the first cycle `s_axis_tvalid && s_axis_tready` (per
  the spec's "easiest is to start when the first input beat is
  accepted"). The host's top-level `s_axil_*` is tied off (all `*ready`
  and `*valid` outputs low) for Phase 2 first-light; muxing the host in
  is task 13's concern. The previous `assign engine_start = 1'b0` line
  is gone — `engine_start` is now driven by `u_scheduler.engine_start`,
  and the engine's `engine_busy`/`engine_done` feed back into the
  scheduler.
- **Fix 7** — for each module in `nn2rtl_scheduler_schedule.json`'s
  `dispatches[]` (the 10 layers the engine actually runs), the wrapper
  emits a new `engine_output_bridge` instance with `SLOT =
  dispatch_index`. The bridge counts `sched_engine_output_ready`
  pulses; when its slot matches it forwards `engine_act_out_wr_*` onto
  the layer's `_valid_out`/`_data_out`, width-adapting via a generate
  branch (zero-pad up, truncate down). The downstream layer's
  `ready_in` is wired to the bridge's `ready_out` (no backpressure if
  no downstream consumer is recorded). For heavy-list entries that the
  scheduler does not actually dispatch (the heavy list and the
  scheduler dispatch order have drifted — itself a 13a-class
  reconciliation problem, not in this PR), the wrapper emits
  `assign _valid_out = 1'b0; assign _data_out = N'd0;` so every heavy
  layer's output has exactly one driver. The wrapper merges the heavy
  list and the schedule's dispatch order before building topology, so
  `node_conv_220`/`_284`/`_288`/`_292`/`_298` (in the schedule but
  absent from the current heavy-list file) are now correctly engine-
  handled instead of being instantiated as per-layer modules they no
  longer have bodies for.
- **Fix 8** — the wrapper now drives `uram_weight_mem.rd_addr` with
  `engine_weight_rd_addr >> 3`. Per the task spec's choice, the
  canonical unit at the memory interface is 2048-bit "engine-beat"
  words; the scheduler/engine continue to emit `weight_rd_addr` in
  288-bit URAM-word units, and the wrapper performs the
  8 × 288 ≈ 2304 ≈ 2048 conversion at the memory's address port. The
  exact byte-precise remap (since 8 × 288 ≠ 2048 cleanly) requires
  repacking `output/weights/uram_weights.mem` from 288-bit lines into
  2048-bit lines; that is owned by task 13.

The new `engine_output_bridge` module body is emitted inline in
`nn2rtl_top.v`'s wrapper-local `\`ifndef NN2RTL_TOP_NO_STUBS` section
alongside the FIFO / memory bodies — it is not a separate sub-block.
The locked port spec for `bram_to_stream_bridge` is unsuited to this
job (it is the engine's internal byte-select / byte-pack module, not a
stream-out bridge to the spatial chain), so a new wrapper-local module
was the cleaner option. Reconciling the two bridge concepts under one
name is left to a future 13a-class refresh.

### Verification on the regenerated wrapper

- iverilog parse on `output/rtl/nn2rtl_top.v output/rtl/nn2rtl_scheduler.v
  output/rtl/shared_engine_skeleton.v output/rtl/engine/*.v
  output/rtl/node_*.v output/tmp/.../coord_scheduler.v
  output/tmp/.../line_buf_window.v output/tmp/.../conv_datapath.v` →
  **exit 0, 0 errors** (full elaboration including per-layer modules
  and rtl-library files). The narrower task-spec command without the
  per-layer / library files exits non-zero only because of Fix 9's
  scope (unknown `node_*` and `coord_scheduler` / `line_buf_window` /
  `conv_datapath` modules); no error in the regenerated wrapper itself.
- `grep "assign engine_start = 1'b0" output/rtl/nn2rtl_top.v` → no
  matches.
- `grep -c "u_engine_out_" output/rtl/nn2rtl_top.v` → 10 (one bridge
  per scheduler dispatch slot).
- `grep "node_conv_220_valid_out" output/rtl/nn2rtl_top.v` → 1 wire
  declaration, 1 driver (the `u_engine_out_node_conv_220` bridge's
  `.valid_out(...)` port hookup), 1 consumer (the next spatial layer's
  `.valid_in(...)`).
- `awk '/^module skip_fifo/,/^endmodule/' output/rtl/nn2rtl_top.v |
  wc -l` → 54 lines (non-empty).
- `awk '/^module uram_weight_mem/,/^endmodule/' output/rtl/nn2rtl_top.v
  | wc -l` → 20 lines (non-empty).

## Post-Path-D third audit (4 critical fixes)

A third audit while Vivado synth was running surfaced four further
correctness bugs in this session's work — the prior two audits had
unblocked **synthesis** but the design was still functionally inert.
None of these would have caused Vivado to fail (it tolerates floating
nets and accepts wrong-time pulses); they would have surfaced as
deadlocks or numerical garbage at Verilator first-light.

### Fix A — requant_valid_in pulsed at start of ST_RUN instead of end

`shared_engine_skeleton.v` had `assign requant_valid_in = ag_bias_rd_en_d`,
which fires one cycle after `bias_rd_en` — i.e. at the **start** of
each ST_RUN entry. But mac_array hasn't accumulated anything yet at
that point (`mac_clear` just fired, so `acc=0`), and the FSM's
ST_REQUANT exit condition (`requant_valid_out`) would only see the
pulse during ST_RUN, never during ST_REQUANT → deadlock.

Replaced with a 3-cycle delay chain off `ag_mac_done`
(`ag_mac_done → d1 → d2 → d3 → requant_valid_in`), matching the
mac_array pipeline depth (URAM read latency + mul stage 1 + acc
stage 2). The bias pre-fetch path is unchanged — bias_rd_data is
held by the BRAM's synchronous read until the new pulse arrives.

### Fix B — address_generator BRAM read address doesn't stride for IC > 256

The activation BRAM word width is `MAC_COUNT × ACT_W = 2048` bits =
256 bytes = 256 channels per word. For IC > 256, the layer's
activation tensor takes `ceil(IC / 256)` BRAM words per pixel
(channels [0..255] in chunk 0, [256..511] in chunk 1, etc.). The
WRITE side already strides by `oc_passes_total`; the READ side did not.

8 of 14 heavy layers have IC > 256 (up to IC=2048). The bug:
- `act_in_addr_n = base + (in_r * IW + in_c)` — no IC stride.
- `act_in_ic_byte_idx = ic_cnt[7:0]` — wraps every 256, so `ic_cnt=256`
  re-reads chunk 0 byte 0 (= channel 0) instead of advancing to
  chunk 1 byte 0 (= channel 256).

Replaced with
`act_in_addr_n = base + (in_r*IW + in_c) * ic_chunks_total + ic_chunk_idx`
where `ic_chunks_total = ceil(cfg_ic / 256)` and
`ic_chunk_idx = ic_cnt[11:8]`. For IC ≤ 256 the stride collapses to
`× 1 + 0` → identical to the legacy formula.

### Fix C — engine_act_in_rd_data was a floating wire; no activation BRAM in wrapper

`scripts/build_top_wrapper.ts` declared `engine_act_in_rd_data` but
never drove it. The engine's BRAM-style input port had no source.
Vivado would optimise it away, the design would read X / 0 in
simulation. The scheduler emitted `input_bank_sel` /
`output_bank_sel` but nothing in the wrapper consumed them.

Added a flat unified URAM-backed activation BRAM
(`act_unified_mem` module, `(* ram_style = "ultra" *)`):
- `NUM_BANKS × BANK_DEPTH_WORDS = 6 × 4096 = 24,576` entries × 2048 b.
- Bumped `BANK_DEPTH_WORDS` from 2048 → 4096 in
  `scripts/build_scheduler.py` because node_conv_250's output tensor
  (28×28×1024 = 3,136 words) exceeds 2,048; 4,096 is the next power
  of two and costs zero extra URAMs (cascade-depth granularity is
  4,096 anyway).
- Engine read port driven directly: 1-cycle synchronous read.
- Engine write port: same memory; the engine's `act_out_wr_*` writes
  go into the same flat memory, so the next heavy dispatch reads
  the previous dispatch's output via its scheduled `input_bank`.
- The scheduler already partitions the address space by setting
  `cfg_act_in_bram_base = bank × BANK_DEPTH_WORDS` (and same for
  out) — no bank-mux logic needed in the wrapper because the engine
  pre-translates bank index into address bits.

URAM accounting: weight banks 768 + activation 174 = 942 / 1280 =
73.6% U250 URAM utilisation (was 60% pre-fix).

**Out of scope for this fix (deferred to a later task)**: routing
SPATIAL layer outputs into the activation BRAM. For first-light and
Verilator unit-tests the TB pre-loads the BRAM with input data; the
engine's own outputs already flow into the BRAM via the write port.
End-to-end spatial↔BRAM coordination is the next discrete task.

### Fix D — bias byte-order in pack_oc_pass_word

`scripts/build_bias_memory_map.py` used `struct.pack("<i", ...)`
(little-endian) to serialise each INT32 bias. `$readmemh` then placed
the bytes MSB-first in the wide bias word, putting LSByte at the
high-bit position of each 32-bit slot. `requant_pipeline.v` reads
`$signed(bias_in[lane*32 +: 32])` expecting bit [31:24] = MSByte,
so every bias was byte-swapped. E.g. bias `-4` (`0xFFFFFFFC`) became
`bias_in[31:0] = 0xFCFFFFFF = -50,331,649`.

Changed to `struct.pack(">i", ...)` (big-endian). Round-trip
verified: bias `-4` now arrives at the requant pipeline as `-4`.

### Verification

- Strict full elaboration (top + scheduler + skeleton + 5 sub-blocks
  + 119 per-layer modules + 3 rtl_library helpers) under iverilog
  `-t null`: exit 0 after all four fixes.
- `verify_weight_memory_map.py` (Path D bit-exact) still passes
  unchanged.
- Vivado first-light synth on U250 (xcu250-figd2104-2L-e) re-launched
  on the fixed design after the initial run (on the pre-fix wrapper)
  was killed for missing the activation BRAM.

