# ENG-PIPE — Pipelined (pixel, oc_pass) Issue for the MBV2 Shared Engine

**Date:** 2026-06-10 · **Param:** `ENG_PIPE` (default **0** = verbatim legacy elaboration)
**Files:** `output/rtl/shared_engine_skeleton.v` (only RTL file touched),
`output/mobilenet-v2/rtl/nn2rtl_top_engine.v` (`.ENG_PIPE(1)`),
`tb/engine_iso_wrap_mbv2.v` (`-DENG_PIPE` / `-DTHROTTLE` hooks)
**Apply script:** `scripts/apply_mbv2_engpipe.py` (anchor-asserted, idempotent, `.preengpipe` backups)
**Gate runner:** `scripts/run_mbv2_engine_iso_engpipe.sh`

## 1. The bubble being removed

Legacy FSM round-trips `ST_RUN -> ST_REQUANT -> ST_DRAIN -> ST_RUN` per
(pixel, oc_pass) with the address generator PARKED (`run_active =
(state==ST_RUN)`). Measured against `D` = the cycle `ag_mac_done` is visible
(= the pass's **last weight-issue cycle**):

* requant_valid_in at `D+5` (mac_done_d5), requant_valid_out at `D+9`
* intermediate pass: `ST_REQUANT -> ST_RUN` at `D+10`, first issue of the
  next pass visible at `D+11` → **10 idle issue cycles**
* pixel boundary: `+ ST_DRAIN` (bridge write) → first issue at `D+13`
  → **12 idle issue cycles**

With KPAR8 the dense walks are only `N = IC/8` issue cycles (N as low as 2),
so the bubble dominates: hundreds of K cycles per frame.

## 2. THE SCHEDULE (ENG_PIPE=1, unstalled)

`D` = mac_done cycle of pass P. `S = D+3` = restart (first `ST_RUN` cycle of
pass P+1). All "edge" events are the posedge **ending** the named cycle.

| cycle | ISSUE side                                            | RETIRE side (pass P)                                          |
|-------|-------------------------------------------------------|---------------------------------------------------------------|
| D     | last weight issue of P; FSM arc `ST_RUN -> ST_GAP`    | pend push (pend++)                                             |
| D+1   | ST_GAP(1): **oc_pass/pixel advance** (`mac_done_d1`)  | `addr_cap <= ag_act_out_wr_addr` (live, stable until S+1)      |
| D+2   | ST_GAP(2): exit decision (`pixel_done`→DRAIN; `pend<=1`→RUN) |                                                          |
| D+3   | **ST_RUN**: AG rising edge → counters reset, bias read issued (new oc_pass) | —                                        |
| D+4   | first new weight issue visible; bias_rd_en visible    | `bias_cap/scale_cap <= live bus` (`mac_done_d4`) — new bias word lands only at D+5 |
| D+5   | new pass's bias data on live bus                      | `acc_cap <= mac_acc_out` (`mac_done_d5`; acc final since D+4); head-ready tick |
| D+6   | **mac_clear** (`run_entered_d3` = S+3)                | earliest **FIRE** (`requant_valid_in` from caps)               |
| D+7   | first accumulate of P+1 (edge ends D+7)               | beat in requant pipe                                           |
| D+10  |                                                       | `requant_valid_out` (fire+4)                                   |
| D+11  |                                                       | bridge write presented (held while `!out_ready`)               |

**Bubble: 12 (pixel) / 10 (intermediate) → 3 + 3** (better than the sweep's
4/3 model — pixel boundaries pipeline identically to intermediate passes
because the counters advance at `mac_done_d1`). Retire latency per pass is +1
cycle vs legacy (fire at d6 vs capture at d5) — pure latency, paid once per
**layer** (the END flush), not per pass.

### Acc-overlap exactness proof (risk 2)

* P's last accumulate edge ends `D+3` (last mac_valid_in at `D+2`, q1 at
  `D+3`) → `mac_acc_out` final from `D+4`.
* `acc_cap` captured at edge ending `D+5` — strictly after final-acc.
* `mac_clear = run_entered_d3` = edge ending `S+3 = D+6` — **strictly after**
  the `D+5` capture (≥1 cycle margin even when the gap is held longer,
  because the clear is keyed to the restart, which is keyed to the gap exit).
* P+1's first `mac_valid_q1` is `S+4 = D+7` → first accumulate edge ends
  `D+7` — **strictly after** the clear. No product is dropped (clear has
  priority over accumulate in `mac_array`, so they must never coincide — and
  they cannot: `S+3 < S+4` by construction for every restart including the
  first pass after `ST_LOAD_CONFIG`).

### Per-pass params under overlap (risk 3)

The restarted walk re-fires the AG's existing rising-edge init, so the next
pass's **bias read is issued at S+1 = D+4 and its data lands on the live bus
at D+5** — exactly the cycle legacy's requant capture would read it. Hence
the capture registers:

* `bias_cap`/`scale_cap` latch the live buses at `mac_done_d4` (**D+4**, one
  cycle before the clobber); `requant_bias_in`/`requant_scale_in` are driven
  from the caps, and the requant pipe's own `scale_q1` pipelining samples
  them at the fire edge — per-OC indexing is unchanged (the caps are the
  same 8192b words the live buses carried).
* `addr_cap` latches the AG's write address at `mac_done_d1` (**D+1**; the
  live register is overwritten at S+1 = D+4 by the restarted walk). At FIRE
  it transfers to `addr_inflight`, which drives the `act_out_wr_addr` port
  and is stable through the entire (possibly held) bridge-write window — the
  next fire is gated on this beat's acceptance.

## 3. Backpressure / throttled out_ready (risk 1)

The retire path is **event-driven and self-throttling**:

* **FIRE gate:** `fire = pend!=0 && rdy_head && !in_pipe && !act_out_wr_en`.
  A beat enters the rigid 4-stage requant pipe only when the pipe is empty
  AND the bridge write register is empty. A held beat (`act_out_wr_en &&
  !out_ready`, the bridge's existing hold) therefore blocks the next fire —
  beats can never collide or drop. Unstalled, this gating is free for every
  walk with N≥3 issues (the previous beat clears the bridge before this
  beat's earliest fire); N=2 walks (IC=16: conv_816's dispatch) retire at
  period 6 instead of 5.
* **pend queue (max 2 by construction):** the gap exit requires `pend<=1`,
  so at most one finished-unfired pass + one walking pass exist. If the walk
  finishes while the head is still unfired (long stall), `pend==2` **holds
  the gap**: no new walk starts, so every live source (AG write addr, bias/
  scale buses, the accumulators) is frozen.
* **fire_recap:** a pass that finished behind an unfired head skipped its
  d-chain captures (`ev_is_head` was false). When the head fires, the
  successor's frozen live values are staged into the freed caps at the same
  edge (NBA: the firing beat reads the old cap values). If that staging ran
  before the successor's acc was final (fire earlier than its `D+4`), the
  successor — now the head — refreshes `acc_cap` at its own `d5`.
* **Ready bits:** `rdy_head`/`rdy_tail` shift on fire; the `mac_done_d5`
  tick marks the oldest un-ready pend. A promoted tail can never fire before
  its acc is final: promotion happens at a fire, and the next fire is gated
  behind that beat's pipe+bridge occupancy (≥5 cycles).
* **End of layer:** `ST_GAP -> ST_DRAIN` when `ag_pixel_done`; `ST_DRAIN ->
  ST_DONE` once `pend==0 && !in_pipe && !bridge_busy` (all beats written).

Legacy behavior with `ENG_PIPE=0` is untouched — kept VERBATIM inside
`generate if (ENG_PIPE == 0)` — **except one explicitly-marked rider**:

### B-class finding: pre-existing legacy backpressure pass-skip (FIXED)

The new LFSR-throttled iso harness exposed a latent bug in the legacy
`ENABLE_OUTPUT_BACKPRESSURE` hold (shipped 2026-06, dormant until now):
at the `requant_valid_out` cycle `oc_pass_idx` ALREADY advances, and the
held re-evaluation (`req_done_pending`) re-entered the shared arm

```
ST_REQUANT: if (requant_valid_out || req_done_pending)
                if (oc_pass_idx == oc_pass_total_m1) next = ST_DRAIN; ...
```

with the NEXT pass's index — so a hold landing on the **second-to-last**
oc_pass jumped to `ST_DRAIN` and **skipped the final oc_pass** (stale
output channels; observed: legacy+throttle DW 896 mismatch=14114,
took=2825 < unthrottled 3824 — the skip makes it *faster*). REPRODUCED
byte-identically on the unpatched `.preengpipe` skeleton
(`reports/engpipe/iso_pre_thr_896_v0.log`) → pre-existing, not an
ENG-PIPE regression. Dormant in deployment: MBV2's e2e FIFO never stalled
at the vulnerable cycle, and ResNet has backpressure disabled →
`req_done_pending` is identically 0 → the changed arm is dead code there
(provably bit/cycle-identical; re-verified by gate d). Fix: split the arm —
`requant_valid_out` keeps the original (pre-increment-correct) decision; a
pending re-evaluation can only ever resume `ST_RUN` (a held LAST pass went
to `ST_DRAIN` at valid_out and never re-enters). After the fix,
legacy+throttle DW 896 is byte-exact. With `ENG_PIPE=1` the path doesn't
elaborate at all (no ST_REQUANT).

## 4. Modes (risk 4)

The retire machinery keys only on `ag_mac_done` and the FSM-owned counters —
it is mode-agnostic. **Dense fast-walk (KPAR8), depthwise serial (the 12 DW
dispatches), and FC all take the same pipelined path**; no per-mode gating
was needed. DW specifics re-checked: `ic_chunk_idx = oc_pass_idx` is only
consumed during the walk (counters advance at D+1, walk restarts at D+3);
the serial walk's N=9 issue cycles satisfy every constraint above.

## 5. Param-gating & inertness (risk 5)

All legacy logic is wrapped `generate if (ENG_PIPE == 0)` **textually
verbatim**; the only shared-text changes are wire indirections
(`requant_acc_in` for the requant `acc_in` port; `requant_scale_in` loses
its inline initializer) that are assign-identical when `ENG_PIPE==0`.
ResNet's top never sets `ENG_PIPE` → default 0 → bit- and cycle-identical
(gate d: vec0 PASS 0/100352 @ EXACTLY 5,664,715). Only
`nn2rtl_top_engine.v` (MBV2) sets `.ENG_PIPE(1)`.

## 6. Atomic rule (risk 6)

No `scripts/` Python/JS formula models the shared-engine FSM cycle count.
`compute_conv2d_latency_cycles` (`scripts/golden_impl.py`) models the
per-module **spatial** datapath (lane_counter/MP), which ENG-PIPE does not
touch. Goldens are value-streams (unaffected). Nothing to co-update.

## 6b. e2e integration findings (TWO MORE pre-existing B-class bugs)

The first full-chain run (EP=1) hit `mismatch_bytes≈500-670/vec` (±1-class
logit deltas) at the predicted 1,957,391 cycles while every ISO case
passed. Localization chain (all instrumentation temporary, removed):

1. **Per-dispatch accepted-beat checksums** at the engine FIFO push:
   dispatches 0-5 byte-identical to EP0, **first divergence at dispatch 6**
   (conv_828 expand, IC=24) with identical beat counts.
2. **Input-region snapshot checksum** at each engine_start: dispatch 6's
   region [0,3136) ALREADY differed at its start → the ENGINE was computing
   correctly; its loaded input was corrupt.
3. **Stream taps around `node_add_198`** (the spatial residual join feeding
   dispatch 6's loader): lhs stream == dispatch 5's output (byte-exact),
   skip stream == dispatch 2's output (byte-exact), counts 3136/3136/3136 —
   but the add's accepted OUTPUT stream csum diverged. Held-beat-overwrite
   detector: 0 events.

**Root cause [ADD-JOIN FIX]:** in every MBV2 `node_add_*.v` the two input
skid-FIFOs pop on the add's REGISTERED `ready_in` while the add ACCEPTS on
the COMBINATIONAL `valid_in && !skid_block` — one cycle of staleness apart.
When the downstream loader's `in_ready` toggles (act-BRAM write-port grant
denials — made frequent by ENG_PIPE's ~6-cycle engine write cadence vs
legacy ~16), the add can accept a pair the FIFOs did not pop → the pair is
processed twice and every later pixel shifts one position (counts
preserved!). The final GAP (global average pool) averages the spatial
shift away — hence the deceptive ±1 logit signature. Fix (all 10 adds,
uniform): `ready_in` becomes the combinational truth
`(state==ST_IDLE) && !skid_block`, making pop ≡ accept BY CONSTRUCTION;
the old registered writes retarget a dead shadow reg. Cycle-identical for
`ENABLE_BACKPRESSURE==0` (1 in IDLE / 0 in RUN, same edges); the accept's
`!skid_block` gate is unchanged so the parked-beat overwrite protection is
preserved. This bug is LATENT IN THE KPAR8 BASELINE — it simply never saw
a ready-toggle at the vulnerable cycle.

**Hardening rider [ARB-COMMIT]:** the engine's held `act_out_wr_en`
(backpressure hold) used to occupy the act-BRAM write arbiter for the
whole hold, starving the 1-deep-skid loaders during FIFO-full stretches.
The arbiter now sees the engine only on the FIFO-ACCEPT cycle
(`wr_en && eofifo_in_ready`) — value-identical (each beat still lands
exactly once, before engine_done) and removes the hold-hogging regime
ENG_PIPE would otherwise introduce. (In the vec0 frame the FIFO never
fills, so this rider is currently latent margin.)

## 7. Gates & results

| gate | command | result |
|------|---------|--------|
| (a) lint | verilator --lint-only, 6 configs (kp8×{ep0,ep1,ep1_thr,ep0_thr}, kp1, kp4) | **0 errors, 0 warnings** (`reports/engpipe/lint_*.log`) |
| (b) ISO A/B | `bash scripts/run_mbv2_engine_iso_engpipe.sh` — 816 (dense IC=16), 898 (dense IC=960), linear (FC), 896 (DW); vec0+vec1; each also with LFSR-throttled out_ready vs legacy-throttled reference | see `reports/engpipe/iso_gate_summary.log` |
| (c) MBV2 e2e | `bash scripts/run_mbv2_e2e_parallel.sh` | see §8 |
| (d) ResNet inertness | `NN2RTL_VALUE_THREADS=1 NN2RTL_VALUE_XINIT=0 npx tsx scripts/run_nn2rtl_top_value.ts 0` | see §8 |

## 8. Measured results

* **(a) lint**: 0 errors, 0 warnings — all 6 configs
  (`reports/engpipe/lint_*.log`).
* **(b) ISO** (`reports/engpipe/iso_gate_summary.log`): **PASS, 21/21
  byte-exact** — ep1 {816, 898, linear, 896} × vec0+vec1; ep0 references;
  ep0_thr + ep1_thr (LFSR ~50% out_ready throttle) all mismatch=0.
  Cycle deltas (vec0): 816 175,618→75,275 (−57%); 898 6,470→6,039
  (−6.7%); linear 684→664 (−3%); 896 (DW) 3,824→2,364 (−38%).
* **(c) MBV2 e2e 8/8**: **PASS (8/8 byte-exact, TOTAL mismatch 0)**,
  frame = **1,957,391 cycles on ALL 8 vectors** (from 2,264,013 KPAR8
  baseline = **−306,622, −13.5%**; baseline re-verified PASS @ exactly
  2,264,013 in this tree before the fixes).
* **(d) ResNet inertness**: **PASS 0/100352 @ e2e_cycles EXACTLY
  5,664,715** (ENG_PIPE unset → legacy elaboration; run AFTER the B-fix,
  proving the changed ST_REQUANT arm is dead code without backpressure).
* Apply-script reproduction test: pristine `.preengpipe` tree + one
  `python scripts/apply_mbv2_engpipe.py` reproduces every patched file
  byte-EXACTLY (`reports/engpipe/applytest`).

## 9. Promotion notes

* Apply order on a pristine KPAR8 (d52edab) tree:
  `python scripts/apply_mbv2_engpipe.py` → rerun gates. No weight/mem/map
  regeneration needed (ENG-PIPE is pure control timing; no data layout
  change).
* The `-DTHROTTLE` iso hook arms `ENABLE_OUTPUT_BACKPRESSURE(1)` in the iso
  wrap — useful for any future engine-output work, independent of ENG_PIPE.
* Follow-on lever (not taken): the 3-cycle gap could shrink to 2 by clearing
  at `D+5` (same edge as the acc capture, NBA-safe) — rejected for margin;
  the remaining ~3-cycle bubble × ~190K passes ≈ 0.57M is already mostly
  fundamental (bias-read latency + URAM flight time).
