# K1 — FDCE→FDRE datapath recode: safety analysis (2026-06-09)

**Applier:** `scripts/apply_k1_fdce_recode.py` (anchor-asserted, idempotent,
`--dry-run`, `.prek1` backups, two-phase validate-then-commit — NO file is
written if ANY anchor drifts).

**Goal.** ~73.5% of the design's ~1.3M FFs are async-reset FDCE
(`always @(posedge clk or negedge rst_n)`) clearing DATAPATH registers whose
reset value is dead. On UltraScale+ this (a) fans `rst_n` out to ~960K loads,
(b) fragments control sets (each {clk, rst, ce} combo is its own set → poor
slice packing at the observed 96.7% CLB / ~67% LUT), (c) blocks SRL and
LUTRAM/BRAM inference, and (d) folds sync-clear emulation LUTs (est. 60–150K
LUT). K1 moves the proven-safe datapath registers into separate no-reset
`always @(posedge clk)` blocks — the repo's established **"Block A: array/data
writes (sync-only)" vs "Block B: control FSM"** pattern (precedents already in
tree: `node_relu.v`, `node_add_1.v`, `node_add_5.v`, `act_unified_mem`,
`engine_output_fifo` mem-write, per
`knowledge/patterns/protected/08_common_bugs.md` §"Array memory write in
async-reset block").

ALL control/handshake state — FSM `state`, `valid_*`/`ready_*`, beat/pixel
counters, FIFO pointers, skid `*_valid`, scheduler coordinates, `loaded`,
`drain_complete`, `mac_valid_q*`, `oc_group`/`k_group` — KEEPS its async reset,
untouched.

---

## 1. The universal byte-exactness argument

The e2e gate runs Verilator `--x-initial 0`, i.e. FPGA power-on zeros. For a
register moved out of the reset clause:

1. **Power-on value is unchanged.** No-reset `reg`s start at 0 under
   `--x-initial 0`; on the FPGA an FDRE without an `initial` gets `INIT=0`.
   That is exactly the value the old async reset produced.
2. **No writes can fire during the (t=0, single) `rst_n` assertion window.**
   Every Block-A write-enable traces exclusively to control registers that are
   STILL async-reset-held: upstream `valid_out` regs (→ `valid_in`,
   `beat_fire`), FSM `state`, `sched_advance`/`start_pulse` (wrapper FSM +
   `coord_scheduler` both keep reset), `mac_valid_q1/q2`, `skid_valid`,
   `buf_valid`, FIFO pointers, `do_rd`/`load_skid`. The only Block-A writes
   that fire unconditionally during reset write **values identical to the old
   reset value** (e.g. `partial_q <= sum_lane_w` where `sum_lane_w` is 0
   because the reset-held/zero-power-on taps multiply to 0; `scale_q1 <=
   scale_in` where the scale ROM's `rd_data` reg is still its power-on 0).
3. Therefore the FULL machine state at reset release is bit-identical to the
   pre-K1 design; by induction every later cycle (and every output byte) is
   identical. **Byte-exact by construction, not just "should be".**
4. **Defense in depth — write-before-read per class** (sections below): even
   if a future TB re-asserted reset mid-run, each recoded register is either
   rewritten before its first post-reset read in every frame/pixel/OC-pass, or
   only sampled under a reset-kept valid bit.

A residual, deliberate behavioural delta exists ONLY for a hypothetical
mid-operation re-reset (datapath bits would hold stale values instead of 0
while control resets); the deployed flow (single power-on reset, TB asserts
reset once at t=0) never exercises it, and even then no stale value can reach
an output because all sampling is valid-gated.

---

## 2. Register classes converted (FF moved off rst_n)

Counts computed from the live RTL parameters of the **instantiated** modules
(36 spatial convs incl. stem, 49 relus, 16 adds, maxpool, engine, top
plumbing; the 17 engine-dispatched `node_conv_*.v` files are not instantiated
in `nn2rtl_top.v` and don't contribute FFs).

| # | Class (files) | Registers moved | FFs |
|---|---|---|---|
| P1 | `rtl_library/line_buf_window.v` (38 insts) | `window[KH][KW-1][IC]`, `bypass_reg[IC]` | 128,520 |
| P2 | `rtl_library/conv_datapath_mp_k.v` (36 insts) | `data_out[OC*8]`, `acc[MP]`, `biased[MP]`, `scaled[MP]`, `partial_q[MP]` | 187,624 |
| P7 | 35 conv wrappers `output/rtl/node_conv_*.v` | `in_lo[IC*8-256]`, `out_pix[OC*8]` | 196,864 |
| P8 | 48 relus `output/rtl/node_relu_*.v` | `beat_buf[BEATS][256]` | 181,248 |
| P3 | `output/rtl/engine/requant_pipeline.v` | `scale_q1/q2[8192]`, 256× lane pipe (`biased_q1`, `scaled_q2`, `sat_hi/lo_q3a`, `v_low_q3a`, `data_out_q4`) | 46,080 |
| P4 | `output/rtl/engine/mac_array.v` | 256× `acc[32]` | 8,192 |
| P5 | `output/rtl/shared_engine_skeleton.v` | `act_in_rd_data_d[2048]` | 2,048 |
| P6 | `output/rtl/nn2rtl_top.v` helpers | `skip_fifo.out_data_r` (33 URAM-deep insts), `engine_output_fifo.out_data` (×2), `stream_to_act_bram_bridge.{wr_data, skid_data, accumulator, beat_buf}` (×17), `engine_output_bridge.{beat_buf, data_out}` (×17) | 125,696 |
| P9/P10 | `node_add_14.v`, `node_add_15.v` | `lhs_buf[2048]`, `rhs_buf[2048]`, `out_beats[64][256]` (+`v_tmp` temp) | 98,304* |
| | **TOTAL** | | **~974,576** |

\* FF-equivalent: with the array writes inside the async-reset block Vivado
cannot infer RAM for them (the known
`activation_memory_in_async_reset_block` bug class), so today they dissolve
to FF/LUT; after K1 they are RAM-inference-eligible — either way the rst_n
loads disappear.

Expected synthesis effect: `rst_n` fanout drops from ~960K loads to roughly
the surviving ~300K control bits' worth; FDCE→FDRE collapses the control-set
count on the datapath (better slice packing at 96.7% CLB); SRL inference
becomes possible for shift pipes (e.g. `scale_q1→q2`, requant lane stages);
the folded sync-clear LUT cones on reset-defaulted datapath disappear
(~60–150K LUT recovery per the K1 context analysis — placement/packing
relief, not arithmetic change).

---

## 3. Per-class write-before-read proofs

### P1 `line_buf_window` — `window`, `bypass_reg`
The old block was `if (!rst_n) clear; else if (frame_start) clear; else if
(sched_advance && !sched_output_fires) shift`. K1 deletes only the `!rst_n`
arm. Every node wrapper enters `ST_ARM` on the first post-reset cycle and
pulses `start_pulse` (= `frame_start`), so the **identical sync clear runs
before any pixel is shifted in** — the reset arm was a strict duplicate.
During reset, both enables are 0 (`start_pulse` async-reset in the wrapper;
`sched_advance` derives from the async-reset `coord_scheduler`'s `running`).
The datapath reads `window_flat`/`chan_window_flat` only on `output_fires`,
which the scheduler can only assert after `start`. `q_reg`/`mem`/`row_valid`/
`oldest_slot`/`tcnt`/`burst_active` untouched (q_reg+mem already no-reset;
the rest is control).

### P2 `conv_datapath_mp_k` — `partial_q`, `acc`, `biased`, `scaled`, `data_out`
* `partial_q`: rewritten EVERY cycle; read only under `mac_valid_q2`
  (control, reset-kept). During reset it loads `sum_lane_w` = Σ(w×tap) with
  taps at power-on 0 → writes 0, same as the old reset value.
* `acc`: read only (a) by the gated accumulate itself and (b) in `ST_BIAS`.
  Both happen strictly after a sync clear: `ST_IDLE && start_mac` clears all
  lanes before the first `mac_valid_q1/q2` of the pass can rise (the valid
  pipe is still reset-held 0 at that point), and the `ST_OUTPUT`
  oc-advance clears before the next pass. The Block-A clears are placed
  AFTER the accumulate to preserve the original single-block NBA
  last-write-wins ordering (clear overrides accumulate on a shared edge —
  which in practice never coincides, but order parity is kept anyway).
* `biased`/`scaled`: written in `ST_BIAS`/`ST_SCALE`, read one state later —
  write-before-read within every OC pass.
* `data_out[oc*8+:8]`: every consumed slice is written during that pass's
  `ST_OUTPUT` before `valid_out` (reset-kept) pulses on the LAST pass; the
  wrapper latches `out_pix <= lib_data_out_w` only on `lib_valid_out_w`.
* Shared-variable race check (the file's own documented bug class): after the
  move, `fsm_i`, `fsm_lane_i`, `bias_oc`, `sc_oc`, `out_oc`, `out_shift`,
  `out_round`, `v_tmp` are referenced ONLY by Block A — no loop variable is
  shared across `always` blocks.
* `mac_oc_group_q1/q2/q1b`, `mac_valid_*`, `k_group`, `oc_group`, `state`,
  `valid_out` keep reset (control / lane-address gating).

### P7 conv wrappers — `in_lo`, `out_pix`
* `in_lo` slice *i* (i < IN_BEATS−1) is written when `in_beat_idx == i` during
  the gather; the only read is `{data_in, in_lo}` consumed at
  `last_beat_fire` (slice IN_BEATS−1). `in_beat_idx` is reset to 0 and walks
  up, so for EVERY pixel (including the first after power-on) all consumed
  slices are written first.
* `out_pix` is written exactly when `lib_valid_out_w && !out_busy`, the same
  edge that sets `out_busy`; `data_out = out_pix[out_idx*TILE..]` is sampled
  downstream only while `valid_out = out_busy` (reset-kept).
* Two template variants handled (33× `frame_state` streamer, 2×
  `irow/icol` decimator: `node_conv_224`, `node_conv_288`). In variant A the
  dangling `if (!is_last_in_beat)` guard is removed together with the moved
  write (it guarded ONLY the `in_lo` line; `in_beat_idx` keeps its
  unconditional ternary update — verified by anchor).

### P8 relus — `beat_buf`
Byte-for-byte the `node_relu.v` precedent: sync-only write block with the
replicated guard `!sending && valid_in && ready_in` (exactly the original
nested condition; `ready_in` is read pre-edge in both forms). All
`BEATS_PER_PIXEL ≥ 2` beats are written before `sending` rises;
the data_out cone reads `beat_buf[0]` (written ≥1 cycle earlier) and
`beat_buf[out_beat_count]` during `sending` when no write can occur. Also
unblocks LUTRAM inference for the 16Kb-class stage-4 relus.

### P3 `requant_pipeline` — `scale_q1/q2` + 256-lane pipes
Feed-forward only. The 4-deep `valid_q1..valid_out` chain KEEPS its reset and
is the sole sampling gate for `data_out` (engine captures under
`requant_valid_out`). Lane registers at cycle N+4 are pure functions of
inputs at N..N+3; the engine asserts `valid_in` long after reset, so consumed
values never depend on power-on contents. `scale_q1/q2` rewritten every cycle.

### P4 `mac_array` — `acc`
The engine FSM pulses `mac_clear` (= `run_entered`) on EVERY `ST_RUN` entry —
verified in `shared_engine_skeleton.v` (`state_run_d` one-shot). The first
gated accumulate (`mac_valid_q1`, reset-kept) of every dot product is
therefore always preceded by the sync clear; the old `!rst_n` arm was a
strict duplicate of `mac_clear`'s effect. `mul_q1` was already no-reset.

### P5 `shared_engine_skeleton` — `act_in_rd_data_d`
2048b hold register rewritten every cycle; consumed (`mac_act_byte`) only
when `ag_act_in_rd_en_d2`/`mac_valid_in` (reset-kept) are high. The
en/idx delay bits (`ag_*_d/_d2`) stay async-reset (they ARE the gates).

### P6 top helpers
* `skip_fifo` (URAM branch) `out_data_r`: written under `do_rd`
  (pointer-derived, pointers reset-kept); sampled only under `out_valid_r`
  (reset-kept). Same for `engine_output_fifo.out_data` / `load_skid` /
  `out_valid`. The LUT-branch `mem` write block was already sync-only.
* `stream_to_act_bram_bridge` (all 3 generate branches): `skid_data` consumed
  only while `skid_valid`; `wr_data` consumed (BRAM write mux) only while
  `wr_req` pending; `accumulator`'s consumed slices are all rewritten each
  word before `wr_data` is formed; `beat_buf` consumed only while
  `buf_active`. All four controls keep reset. g_w_gt's two `wr_data` writes
  keep their textual order in Block A (drain overrides continue-slice on a
  shared edge, as before). `wr_addr` (15b) deliberately kept reset
  (address/control).
* `engine_output_bridge`: `beat_buf` consumed (`current_tile`) only while
  `buf_valid`; `data_out` sampled only under `valid_out`; both controls keep
  reset, as do `tile_idx`/`tiles_emitted`/`drain_complete`/`dispatch_count`.

### P9/P10 `node_add_14/15` — `lhs_buf`, `rhs_buf`, `out_beats`
The ONLY two adds whose array writes still lived inside the async-reset block
(all other adds already use the Block-A pattern). lhs/rhs are fully rewritten
during each pixel's 64-beat gather before `ST_COMPUTE` reads them
(`in_beat_count` resets to 0/1 and covers 0..63); every `out_beats` byte is
written by the 3-stage pipe (`stage3_valid` covers ch 0..2047) before
`ST_STREAM` presents beats under `valid_out`. `node_add_15`'s `v_tmp` is a
blocking temp that moves with the `out_beats` write (its dead NBA reset is
dropped to avoid a multi-driver). Gather guards replicated exactly
(`valid_in` for _14, `valid_in && ready_in` for _15 — they differ in the
originals). MAC pipes (`lhs_term`, `sum_term`, `stage*_valid`, `ch_s*`) keep
reset (small; `stage*_valid`/`ch_s*` are control/address).

---

## 4. Deliberately SKIPPED (and why)

| Item | FFs | Reason |
|---|---|---|
| `node_conv_196.v` wrapper (`held_beat1` etc.) | ~260 | SPECIAL stem wrapper (fixed 48-cyc shift-reg streamer, known MP-fragile per in-file warning). Its lbw + datapath still benefit via P1/P2. |
| relu/add/`node_max_pool2d` `data_out` output regs | ~17K | Written at two+ sites interleaved with `valid_out` control and blocking temps (`tmp_byte`/`rs_*`); replicating those cones per 49+17 files is drift-prone for ~1.7% extra. Candidate for K1b. |
| add `s1/s2/s3` MAC pipes, `lhs_term/rhs_term/sum_term` (adds 1–13) | ~2K | Tiny; heterogeneous templates across add generations. |
| `mac_oc_group_q1/q2/q1b` (P2), `wr_addr` (P6), `ag_act_in_ic_byte_idx_d/d2` (P5) | <1K | Address/lane-select gating — treated as control (conservative). |
| `coord_scheduler`, `nn2rtl_scheduler`, `config_register_block`, `address_generator`, engine FSM regs, all `skip_fifo` pointers/`peak_occ` | — | Control by definition. |
| `bram_to_stream_bridge.v` (engine dir) | — | Not instantiated in the integrated top (task-11 bridges in `nn2rtl_top.v` superseded it). |
| The 17 engine-dispatched `node_conv_*.v` (250/264/282/...) + `archive/`, `*.preimprove`, `nn2rtl_top_iv.v` | — | Not instantiated / legacy snapshots; zero synth contribution. |
| `dbg_*` instrumentation regs | — | Sim-only. |

---

## 5. Verification performed (2026-06-09, worktree sandbox `_k1_sandbox/`)

1. **Anchor audit on the live tree**: every anchor string verified to occur
   exactly once per target file (33 variant-A wrappers, 2 variant-B, 48
   relus, adds, library/engine/top blocks) — the dry run validates 91/91
   files with zero drift; encoding (utf-8 vs cp1252) and EOL (CRLF vs LF) are
   auto-detected per file and preserved on write.
2. **Applied** to a full sandbox copy; re-run reports `0 to patch, 91 already
   applied` (idempotency).
3. **Lint** (`verilator_bin --lint-only -Wno-fatal --top-module nn2rtl_top`,
   full integrated top + all nodes + engine + library): **0 errors** before
   and after. Warning histogram identical except the exactly-predicted
   deltas: +6 `WIDTHEXPAND` from the new Block-A `oc_group != OC_PASSES-1`
   guard (same width-expansion class as the pre-existing `==` compare), and
   −1 `WIDTHCONCAT` because the removed `data_out <= {OC*8{1'b0}}` 16384-bit
   reset replication is gone.

## 6. Required gates BEFORE Vivado (HARD RULE)

K1 is *argued* byte-exact, not yet *measured*. Per
`feedback_vivado_only_when_proven`: after applying to the real tree, run the
e2e Verilator `--x-initial 0` full-frame check against the FRESH golden
(expect `result=PASS mismatch=0` and the identical 13,348,787-cycle count —
K1 changes no handshake/latency, so even the cycle count must match
exactly); only then re-synth. No `.mem`/golden regeneration is needed (no
weights/scales/contracts touched).

Apply with:
```
python scripts/apply_k1_fdce_recode.py --dry-run   # inspect plan
python scripts/apply_k1_fdce_recode.py             # writes .prek1 backups
```
Rollback: restore `*.prek1` (every patched file has one).
