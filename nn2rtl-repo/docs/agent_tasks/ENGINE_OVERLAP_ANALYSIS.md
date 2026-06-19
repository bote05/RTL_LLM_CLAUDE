# ENGINE_OVERLAP_ANALYSIS — adversarial verification + build record

**Lever**: let the spatial chain keep streaming while the shared engine runs its
17 dispatches (previously: scheduler `S_WAIT_DONE` + top `spatial_throttle =
engine_busy | sched_spatial_stall` froze the whole chain for every engine run).

**Baseline (measured, vec1 PASS run, main repo
`output/reports_integrated/verilator_nn2rtl_top_value/run.log`)**:
frame = **12,813,738 cycles**, e2e PASS 0/100352. The run's own
`[dbg-partition]` counters (cumulative, printed each 1M cycles):

| cyc snapshot | eng_busy (cum) | spat_run (cum) | disp_idx |
|---:|---:|---:|---:|
| 1M | 0 | 999,961 | 0 |
| 3M | 0 | 2,999,961 | 0 |
| 4M | 863,580 | 3,136,337 | 1 |
| 6M | 1,771,456 | 4,228,337 | 4 |
| 8M | 2,866,067 | 5,133,640 | 6 |
| 10M | 3,825,979 | 6,173,602 | 9 |
| 12M | 5,413,961 | 6,585,410 | 14 |
| end 12.814M | ~6,118,495 | ~6,694,500 | done |

So: **T_pre ≈ 3.137M** (zero engine activity — stem→stage2 until ldr0/ldr1
loaded; input stream done at ~3.1M), **engine phase ≈ 9.65M** containing
eng_busy **6.118M** (= the 47.75%) + engine-phase spatial-run ≈ **3.56M** +
scheduler stall ≈ 0.6K.

---

## Phase-1 verdict: HOLDS, with two corrections

### (1) Is the engine's act-BRAM write dead for ALL 17 dispatches? — YES (proven)

* `act_unified_mem` (24576×2048b) has exactly ONE read port, wired to
  `engine_act_in_rd_addr` (engine act_in). Nothing else reads the act BRAM.
* Every dispatch's act_in read region **exactly equals** its loader's region
  (verified all 17: base & word count; checker C1), and the scheduler's
  `S_WAIT_LOAD`/`current_loaded` gate guarantees the loader finished before
  `engine_start`. The engine therefore NEVER consumes engine-written BRAM data.
* The schedule JSON's `reserved_banks` / `skip_bank_mask` entries are a
  *scheduling abstraction only*: in the RTL, residual/skip data rides
  `skip_fifo` instances (`u_skip_node_add_7..15`, `u_skid_node_conv_288`, …),
  never the act BRAM. The suspicious case — d1 (conv_250) `feeds_skip_of
  node_add_7` — resolves to `u_engine_out_node_conv_250` (FIFO bridge) feeding
  add_7's rhs directly; the parked lhs (conv_248) sits in a skip_fifo.
  `sched_input_bank_sel/output_bank_sel/skip_bank_reserved_mask` are tied off
  as `_unused_` in the top.
* Real outputs leave via `engine_output_fifo` (4096 deep), which taps
  `engine_act_out_wr_en/wr_data` directly — independent of the BRAM write.

**Correction → implementation (B)**: instead of *remapping* the act_out base to
a scratch region (proposal (b)), the engine is **removed from the act-write
arbiter entirely** (BRAM writes dropped; FIFO tap untouched). Remapping would
have kept the engine's absolute write-port priority, and during overlapped runs
its writes (~1/cycle bursts) would starve loader grants beyond the loaders'
1-deep skid → B20-class beat loss. Removal eliminates both the address hazard
AND the contention hazard.

### (2) Per-dispatch hazard table

Chain topology (from `nn2rtl_top.v` loader `in_valid` producers): during
dispatch d's overlapped run, the only loader(s) receiving beats are the next
dispatch's (the stream stops at d+1's conv because its output-bridge slot is
inactive until `dispatch_count == SLOT`). d0 feeds NO loader during its run
(relu_23→conv_248 parks into add_7's skip fifo; ldr1 was filled pre-phase by
relu_21). Full successor map + regions: see `scripts/check_act_region_hazards.py`
output (asserted against the schedule JSON).

Hazards found (= why the freeze was load-bearing):

* **Read-vs-fill (C3)**: all 12 bank-2 loaders share base 8192. During d2/d5/
  d6/d9/d12/d15's runs, the successor loader (d3/d6/d7/d10/d13/d16) fills
  8192.. while the engine reads 8192.. → 6 violations.
* **Engine-write-vs-fill (C2)**: d1 writes act_out [8192,8976) while ldr(d2)
  fills [8192,8388); d10 writes [4096,4488) while ldr(d11) fills the same;
  d13 writes [0,392) while ldr(d14) fills the same → 3 violations.

Fix (A): remap the INPUT regions of **d3→12288, d6→12544, d10→12800,
d13→12928, d16→13056** (196/196/98/98/98 words — all inside bank 3, words
12288..16383, which the schedule shows EMPTY along with banks 4-5). This is
exactly the proposal's remap list; note d7's loader still fills 8192 during
d6's run — safe because d6's READ moved to 12544. Fix (B) (above) clears C2.
Post-patch the checker proves C1–C5 clean for all 17 dispatches; loader
lifetime analysis (C4: a fill must never clobber parked-not-yet-consumed
data of a region-sharing loader) also passes — region sharers' fill windows
all start at/after the earlier sharer's consuming run.

### (3) Does S_WAIT_LOAD still serialize correctly? — YES

The ungate touches ONLY `S_WAIT_DONE` (stall 1→0). `S_WAIT_LOAD` still blocks
`S_PULSE_START` on `current_loaded`; `S_WAIT_DRAIN` still blocks `S_NEXT_DISP`
on `current_drain_complete` (a sticky latch — no missed-pulse race if drain
finishes during the run); the engine remains strictly serialized (one run at a
time). Loaders finishing EARLY (during d's run) just make S_WAIT_LOAD exit in
1 cycle. Config-write windows (13 AXI writes ≈ 52 cyc/dispatch) stay frozen —
total stall ≈ 0.9K/frame, irrelevant.

### (4) The claimed −3.3M — OVERSTATED; corrected ceiling ≈ −3.0M

The recoverable time is **bounded by the spatial work available during engine
runs** = engine-phase spatial-run ≈ **3.56M cycles** (measured above), NOT by
the 6.118M of freeze. T_pre (3.137M: stem phase, no engine) and the post-d16
tail cannot overlap anything. Savings = Σ_d min(run_d, gap_d) per window:

* Engine run model (validated: MACs/254 reproduces the measured 6.118M total):
  9 3×3 dispatches ≈ 455K each; conv_250/282 ≈ 405K; the 1×1s ≈ 202K.
* Big gaps (stage-3 expand/reduce convs at ~51.4M MACs / (MP16·MP_K8) ≈ 402K):
  d1→d2 (conv_252), d2→d3 (conv_256+258), d3→d4 (conv_262), d5→d6 (268+270),
  d6→d7 (274+276), d7→d8 (conv_280): each pairs with a 405-455K run →
  ~0.4M each ≈ **2.4M**. Small gaps (d0→d1, d4→d5, d8→d9, d9→d10, stage-4
  relu/loader hops) ≈ 0.15M. conv_288 (~0.8M of work, parked-skip fed,
  overlappable across d8-d10 windows) ≈ +0.4-0.5M.
* **Expected: −2.8 to −3.2M → frame ≈ 9.6-10.0M** (the claimed 9.49M is the
  optimistic edge). Well above the 1.5M kill threshold → BUILD.

### (5) engine_output_fifo / cadence risks

* FIFO (4096 deep) never backpressures the engine (`in_ready` unused); safe
  because per-dispatch beats ≤ 784 and `S_WAIT_DRAIN` forces empty between
  dispatches. Overlap strictly REDUCES peak occupancy (bridges drain during
  the run instead of after).
* The top's own comment says the engine_busy freeze bounds skip-FIFO growth.
  Checked against the PASS run's `[fifo-peak]` audit: the thin-margin FIFOs
  (add_0..3, 447-479/512) are all STAGE-1/2 — they complete inside T_pre,
  where nothing changes. The engine-phase skip FIFOs are bounded by their
  FULL PARKED MAP, already reached today: add_10/11/12 peak 6271 < 8192,
  add_13 1983 < 4096, add_14/15 3135 < 4096 → cannot overflow under ANY
  cadence. `u_skid_node_conv_288` (4096 < relu_39's 6272-beat map) is
  backpressured by design (relu_39's combined ready) — lossless.
* B20 narrow-relu last-beat class: overlap changes engine-phase cadence;
  the structural mitigations (deep fork receivers; loaders MORE available
  since the engine left the arbiter) plus the e2e byte-exact gate arbitrate.

---

## Build (what changed)

`scripts/apply_engine_overlap.py` (anchor-asserted, idempotent, --dry-run,
`.pre_overlap` backups) applies, in the generated RTL:

1. `output/rtl/nn2rtl_scheduler.v`: `act_in_base_word_rom` d3/d6/d10/d13/d16
   → 12288/12544/12800/12928/13056; `S_WAIT_DONE` drives `spatial_stall=0`.
2. `output/rtl/nn2rtl_top.v`: matching loader `BRAM_BASE_ADDR` for
   u_ldr_node_conv_260/272/286/294/300; act-write arbiter drops
   `engine_act_out_wr_en` (17 grant terms + en/addr/data muxes; FIFO tap
   untouched); `spatial_throttle = sched_spatial_stall` (engine_busy dropped).

`scripts/check_act_region_hazards.py`: static prover (C1-C5 above) over the
schedule JSON + the live RTL text. Pre-patch it reports exactly the 9
violations (negative test); post-patch PASS. It hard-fails if the dispatch
order in the schedule JSON ever changes (the successor map must be re-derived).

## Found during bring-up: conv output-streamer pixel-drop race (FIXED)

First overlap e2e DEADLOCKED in S_WAIT_DRAIN(d1) (engine froze after d0+d1 =
863,580 busy cycles — d0 454K + d1 410K both ran fully and correctly; the
50-cycle dispatch turnaround worked). Forensics ([DBG-OVL] block in the
worktree top, log `output/reports_integrated/ovl_forensics_vec0.log`):

```
[dbg-ovl @4250000] sc248_in=1568 sc248_out=1568 beat=1568 pxin=196 pxout=196
                   emit=6240 add7_fire=6240 ...
```

conv_248's datapath produced all **196** pixels but the output streamer
emitted only **6240 = 195×32** beats — exactly one pixel dropped. The
apply_3x3_backpressure.py streamer template latches a completed pixel with
`if (lib_valid_out_w && !out_busy)` and has NO else: a pixel that completes
while the previous one is still streaming is silently discarded. The conv
MAC is NOT spatial_run-gated but the streamer is (ready_out carries
spatial_run), so any chain freeze that (a) pins out_busy=1 mid-pixel and
(b) lasts past the in-flight MAC's completion eats one pixel. The trigger
here: d1's 13 config-writes freeze (~55 cycles at cyc 3,515,79x) while
conv_248 was mid-backlog. The class is LATENT in the serialized baseline
(equal-length config freezes exist; its fixed schedule happens never to
land a completion inside one) — overlap re-rolls the dice at every dispatch
with a busy chain. Downstream effect: add_7 starved at 6240/6272 → bridge
slot-1 never finished → drain deadlock.

**Fix**: `scripts/apply_conv_output_pend.py` — one-deep pending slot in all
35 instantiated spatial conv nodes (engine-dispatched node files 284/292/298
are never instantiated and are skipped): a pixel completing while the
streamer is busy is parked in `pend_pix` and reloaded the moment the current
pixel finishes. One slot is provably sufficient (`stall_in = mac_busy ||
out_busy` blocks the next MAC start until the streamer frees, so at most one
completed-unstreamed pixel can exist). Value-preserving and cycle-identical
in any execution where the drop never fired (the passing baseline).

## Results

* Hazard checker: **PASS** (post-patch, C1-C5, all 17 dispatches).
* e2e value gate **vec0: PASS 0/100352, e2e_cycles = 9,622,057**
  (baseline 12,813,738 → **−3,191,681 = −24.9%**; log:
  `output/reports_integrated/RUN_overlap_vec0_PASS.log`).
* Engine-phase duty: 6,117,860 engine-busy cycles inside a 6.56M-cycle engine
  phase = **93.3%** (was 63.4% serialized). T_pre = 3.062M, tail ≈ 2.2K.
* Per-dispatch inter-run gaps collapsed from ~65-540K to: 51-71 cycles after
  every reduce/expand 1×1 (d0,d4,d8,d9,d11,d12,d14,d15), ~2.2K after
  d10/d13, ~64-66K after each 3×3 window (d2,d3,d5,d6,d7 — the residual =
  the downstream expand-conv's last-pixels pipeline tail, which cannot start
  before the engine's final output beats), 107K after d1 (conv_248 backlog
  tail + conv_252 tail).
* All remapped dispatches ran with their bank-3 act_in bases
  (12288/12544/12800/12928/13056) and correct configs (per-dispatch
  [dbg-ovl] START lines: cfg ic/oc/k/s/ih/oh/wbase all match the schedule).
* FIFO peaks: stage-1/2 thin-margin FIFOs unchanged vs baseline (add_0..2 =
  463/479/479 of 512 — pre-engine phase untouched); no new near-full FIFOs.
* e2e value gate **vec1: PASS 0/100352, e2e_cycles = 9,622,057** — identical
  to vec0 (handshake timing is data-independent), RUNONLY on the same exe.
  Confirmed in-run: conv_248 emit = 6272/6272 (the pend fix delivering every
  pixel; the broken build dropped to 6240).

## Fallback

If the e2e ever regresses on a future schedule: SAFE-SUBSET = keep
`spatial_stall=1` in S_WAIT_DONE for the hazardous dispatches only
(d2/d5/d6/d9/d12/d15 per the C3 table) via a per-dispatch stall ROM —
worth ~1.8-1.9M instead of ~3M. Not needed for the current schedule.
