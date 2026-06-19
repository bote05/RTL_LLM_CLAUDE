# TAIL_PIPE2 — requant-tail pipelining of conv_datapath_mp_k (2026-06-09)

Re-engineering of the failed 2026-06-05 TAIL_PIPE attempt
(`scripts/apply_resnet_tail_pipe.py`) against the current K1-era code.
New applier: `scripts/apply_resnet_tail_pipe2.py` (anchor-asserted, idempotent,
`--dry-run`, timestamped backups).

---

## 1. Root cause of the 2026-06-05 deadlock (B23)

**The old TAIL_PIPE RTL was functionally and protocol-correct. The deadlock was
a SYSTEM-LEVEL latency-budget violation, not an FSM bug.**

Evidence trail (e2e timeout, `in=50176` consumed / `out=0` forever; skip-FIFO
forensics: `add_0..6` skips filled 463–511, `add_7+` peak 0; recorded in
`memory/project_resnet_route_logic_bound.md` "TAIL_PIPE FAILED + REVERTED" and
`docs/TSCIT2026_FINDINGS.md` bug **B23**):

1. **The suspects in the FSM-mechanics class are all CLEARED.** Walking the
   06-05 hunks against the pre-K1 file they patched
   (`rtl_library/conv_datapath_mp_k.v.prek1` + `backups/resnet_tail_pipe_20260605_022454/`):
   - the split states `ST_OUT_ROUND(5) → ST_OUT_SHIFT(6) → ST_OUT_SAT(7)`
     advance **unconditionally** — no wait condition exists, so the datapath FSM
     cannot hang internally;
   - `mac_busy = (state != ST_IDLE)` stays asserted through states 5–7, so the
     wrapper's `stall_in = mac_busy || out_busy` freeze of `coord_scheduler` is
     preserved (there is no separate per-conv `spatial_stall` input to honor —
     the top-level `spatial_run` gates only the stream handshakes, which are
     latency-elastic);
   - the `oc_group/k_group` advance moved `ST_OUTPUT → ST_OUT_SAT`, which is
     still the cycle immediately preceding `ST_MAC` — the stage-1 weight/tap
     prefetch alignment at the pass boundary is identical, and the stale
     `weight_word_q`/`tap_q` values loaded during the tail states are never
     consumed (`mac_valid_q1/q1b/q2` are fully drained before leaving `ST_MAC`);
   - `valid_out` pulses 1 cycle in `ST_OUT_SAT`, the same edge the last
     `data_out` bytes land, with `state → ST_IDLE` — the exact `ST_OUTPUT`
     contract the wrapper's `lib_valid_out_w && !out_busy` capture expects;
   - the operator chain was value-identical (the e2e never got far enough to
     check values, but the arithmetic hunks are a faithful 3-way split).

2. **What actually broke: a ONE-SIDED rate/latency change between the two
   compute paths feeding the stage-3 residual joins.** The applier enabled
   TAIL_PIPE on every spatial conv (all 45 `.DSP_INPUT_PIPE(1)` files, stem
   included), adding **+2 cycles per oc_pass** (= `+2*OC_PASSES` per output
   pixel, a ~3% per-pixel rate slowdown compounding *cumulatively* down the
   chain). The **engine-dispatched** convs (the stage-3+ heavies, not
   instantiated in the top) got **+0**. Stage-3 residual joins pair an
   engine-fed arm against a spatial/skip-FIFO-fed arm, and the skip FIFOs had
   been **empirically right-sized to the baseline's peak occupancy**
   (`size_skip_fifos.py`, −89% BRAM) with only a small margin. The de-synced
   arrival phase pushed occupancy past a right-sized depth at the stage-2/3
   boundary: the full FIFO backpressures its fork, the fork can then no longer
   deliver beats to the **engine input loader** (`u_ldr_node_conv_246`,
   combined-ready fork off `node_relu_22`), `current_loaded` never asserts, the
   scheduler parks in `S_WAIT_LOAD` forever (with `spatial_stall=0`, so this is
   not a freeze-gating bug), and the join that would drain the FIFO waits on
   the engine that never starts — a circular wait. Upstream everything backs
   up (`add_0..6` skips pinned at 463–511 of their 512/1024 depths), downstream
   never sees a beat (`add_7+` peak 0), `out=0` forever.

3. **Why DSP_INPUT_PIPE (+1) survived and TAIL_PIPE (+2 more) did not:** the
   FIFO-margin slack absorbed a +1/oc_pass uniform slowdown but not +3/oc_pass
   cumulative (B23: "+1 DSP delay tolerated, +3 not"). The lesson recorded in
   B23: *latency changes must be applied consistently across both compute
   paths* — or the elastic buffering between them must have margin for the
   skew.

**1b. DEEPER ROOT CAUSE, ESTABLISHED THIS SESSION (e2e forensics, runs 1-4,
2026-06-09/10): the "latency desync" is only the TRIGGER. The latent defect
that turns a cadence change into a permanent wedge is a LOSSY one-cycle
last-beat offer in the narrow-relu output streamer (the B20 bug class,
believed fixed but only fixed for the WIDE relus).**

`output/rtl/node_relu_*.v` streamer FSM: while `sending`, beats are held
under backpressure (correct); but when the second-to-last beat is accepted
the FSM loads the LAST beat, sets `sending <= 0`, and the `!sending` branch
then drops `valid_out` **unconditionally** one cycle later. The final beat
of every pixel is therefore presented for **exactly one cycle**: if the
relu's `ready_out` — at the post-add relus a **combined dual-ready fork
gate** like `relu_9_dual_ready = skid218_ready & skid224_ready` (and at
relu_21/22 a gate that also ANDs `spatial_run`) — happens to be 0 on that
precise cycle, the beat is silently DISCARDED. One lost beat ⇒ the lockstep
residual-add join downstream can never pair again ⇒ permanent starvation
cascade = every observed "wedge". The baseline (12,670,107 cycles) passes
only because its deterministic cadence happens to never land a last-beat
offer on a not-ready cycle; ANY rate perturbation re-rolls every one of
these dice — which retroactively explains B22 ("one beat slips" on
MP-increase), B23 (this bug on tail-pipe), and likely B29 (MBV2 MP=16).

Forensic chain that established this (each step falsifying the previous
hypothesis):
- **Run 1** (35 convs tail-piped): froze at cyc ~3M; every cumulative counter
  static; `r9_emit = r9_to_skid218 = r9_to_skid224 = 25087/25088`;
  `ldr0_loaded=0`, scheduler parked in `S_WAIT_LOAD`; `spatial_run` high
  99.999% (not a gating bug). First reading (B23-style): skid224 bypass full.
- **Run 2** (skid224 512→1024): froze **bit-identically** — falsified the
  skid224-full hypothesis (with +512 slots not even one extra beat moved).
- **Run 3** (region taps `[dbg-tp2]`): every hop around the suspect region is
  LOSSLESS and COMPLETE — `add2f=25088` (relu_9 RECEIVED its full frame),
  all conv in/out counts consistent, all wrapper capture-miss detectors 0,
  `occ(skip_add_1/2)` peaked ~471 of 512 and drained to 0. Yet relu_9 fired
  only 25087 ⇒ the loss is INSIDE relu_9's output hop ⇒ the one-cycle
  last-beat offer met a not-ready fork (skid218, DEPTH 128, transiently full
  against the tail-piped/slower conv_218) and the frame's FINAL beat
  vanished.
- **Run 4** (last-beat-loss detectors on relus 3/6/9/12/15/18/21/22 +
  skid218 128→1024): **zero drops chain-wide**, the former wedge point
  cleared (stage-2 completes, `relu22=6272`, `ldr0_loaded=1`, dispatches
  advance on the baseline timeline) → full e2e gate (§5).

**Consequences for TAIL_PIPE2** (what we do differently, beyond the K1 rebase):
- **Do not touch the stem** (`node_conv_196`): it is the special fixed-timing
  wrapper (2-beat combinational output splitter, no `ready_out`) and it heads
  the entire chain, so its slowdown shifts every downstream phase for zero
  Fmax benefit (the stem is not on the critical path).
- **Enable only the 35 live spatial mp_k convs** actually instantiated in
  `nn2rtl_top.v` — not the 9 engine-dispatched files
  (246/254/260/266/272/278 + K5's 284/292/298), whose `.v` files still exist
  but are dead.
- **Restore last-beat-offer margin at the tripped fork**: deepen relu_9's two
  fork receivers (`skid_node_conv_218` 128→1024, `skid_node_conv_224`
  512→1024) so neither can be full on a last-beat-offer cycle (the
  end-of-frame burst through add_2 is bounded by `skip_node_add_2`'s 512
  + in-flight). Byte-exact by construction; DEPTH≥512 maps to URAM (16%
  used), ~+1.5 URAM equivalent.
- **RECOMMENDED DURABLE FIX (follow-up, not in this change)**: retrofit the
  narrow-relu streamer to HOLD the last beat until accepted (elastic
  producer, the MBV2 `ENABLE_BACKPRESSURE` pattern / the wide-relu B20 fix
  generalized). Until then every relu→fork hop remains an alignment dice-roll
  under ANY future cadence change — the depth bumps fix THIS deterministic
  frame provably (e2e byte-exact), not the class.

---

## 2. New design (K1-era) — states and write conditions

Target file: `rtl_library/conv_datapath_mp_k.v` at commit `be16f61` ("K1:
FDCE→FDRE recode"). K1 moved all datapath writes (`biased[]/scaled[]/
data_out[]/acc[]/partial_q[]`) out of the async-reset FSM block into the
sync-only **Block A**, whose write conditions replicate the FSM state. Any new
state must therefore extend **both** blocks coherently. The 06-05 hunks
predate this split and no longer anchor.

Pipeline when `TAIL_PIPE=1` (per oc_pass; `S` = `scale_rom[oc]`):

| state        | Block A writes (sync, no reset)                                                | FSM-block control (async reset)                                  | binding comb. path in the cycle |
|--------------|--------------------------------------------------------------------------------|------------------------------------------------------------------|---------------------------------|
| `ST_BIAS`    | `biased[l] <= acc[l]+biases[oc]`; **`sc_mult_q[l] <= S[15:0]`, `sc_shift_q[l] <= S[21:16]`** | `state <= ST_SCALE`                                              | bias-ROM read + 33b add (unchanged); scale-ROM read → reg |
| `ST_SCALE`   | `scaled[l] <= $signed(biased[l]) * $signed(sc_mult_q[l])` (**registered** operand) | `state <= TAIL_PIPE ? ST_OUT_ROUND : ST_OUTPUT` (elab constant)  | pure reg×reg 33×16 mult (was ROM-read→mult, ~8–11ns) |
| `ST_OUT_ROUND` | `out_round_q[l] <= (sc_shift_q[l]==0) ? 0 : 1 <<< (sc_shift_q[l]-1)`         | `state <= ST_OUT_SHIFT`                                          | one barrel shift from a 6b reg |
| `ST_OUT_SHIFT` | `v_tmp_q[l] <= (scaled[l] + out_round_q[l]) >>> sc_shift_q[l]`               | `state <= ST_OUT_SAT`                                            | SCALED_W add + one arithmetic barrel shift |
| `ST_OUT_SAT` | `data_out[oc*8+:8] <= clip(v_tmp_q[l])`; `acc[l] <= 0` (non-final pass, clears stay textually LAST) | final pass: `valid_out <= 1`, `state <= ST_IDLE`; else `oc_group++`, `k_group <= 0`, `state <= ST_MAC` | 2 compares + byte mux |

(The legacy `ST_OUTPUT` did round-bias + add + barrel shift + clip in **one**
cycle, ~12–16ns — the #1 post-DSP-pipe combinational wall.)

Cost: **+2 cycles per oc_pass** (BIAS,SCALE,OUTPUT → BIAS,SCALE,ROUND,SHIFT,SAT),
i.e. `+2*OC_PASSES` per output pixel. FF-only cost
(`MP*(16+6+2*SCALED_W)` ≈ 1.9k FF/conv, ~66k total — FF is at 37.8%); **zero
BRAM/URAM/DSP delta**, so fit is preserved.

K1-coherence details:
- New tail registers (`sc_mult_q/sc_shift_q/out_round_q/v_tmp_q`) are
  **Block-A-written, reset-free (FDRE)** like all K1 datapath regs. Safety:
  every read is preceded by a same-pass write (`sc_*_q` in `ST_BIAS` before
  any `ST_SCALE/ST_OUT_*` read; `out_round_q` in ROUND before SHIFT;
  `v_tmp_q` in SHIFT before SAT), so no power-on value is observable.
- **NBA last-write-wins ordering preserved**: the new Block-A arms are
  inserted *before* the accumulator clears, and the new
  `ST_OUT_SAT && oc_group != OC_PASSES-1` acc-clear is appended *after* the
  `mac_valid_q2` accumulate and next to the existing `ST_OUTPUT` clear (the
  two are mutually exclusive — `ST_OUTPUT` is unreachable when `TAIL_PIPE=1`
  and vice versa). No register has two writers in any cycle.
- The FSM block gains **control only** (`state`, `valid_out`, `oc_group`,
  `k_group`) — the K1 invariant that Block A owns every datapath write and the
  FSM block owns every async-reset control reg is intact. Loop vars
  (`fsm_lane_i`, `out_oc`) stay Block-A-exclusive (no shared-var race).

Protocol invariants (wrapper/coord_scheduler/top all unchanged):
- `mac_busy = (state != ST_IDLE)` covers states 5–7 → the wrapper's
  `stall_in` freeze and the `frame_state` re-arm logic see one longer, but
  otherwise identical, busy window.
- `valid_out` is a 1-cycle pulse raised the same edge the last `data_out`
  bytes are written, with `state → ST_IDLE` — identical capture contract.
- `start_mac` is only sampled in `ST_IDLE` (unchanged).

## 3. Byte-exactness argument (`TAIL_PIPE=1` vs `0`)

Identical operator chain on identical operand values; only register
boundaries move:

1. `sc_mult_q[l]` is loaded in `ST_BIAS` from `scale_rom[bias_oc][15:0]`;
   the legacy path reads `scale_rom[sc_oc][15:0]` in `ST_SCALE` with
   `sc_oc == bias_oc` (same `oc_group`, advanced only in SAT/OUTPUT) and the
   ROM is read-only after `initial` — **same 16-bit value**. Both multiply
   operands are 16-bit `$signed` in the same context → `scaled[]` identical.
2. `sc_shift_q[l]` similarly equals `scale_rom[out_oc][21:16]`. The ROUND
   expression is textually the legacy `out_round` expression with
   `out_shift → sc_shift_q[l]` (`6'd0` compare, `{(SCALED_W-1){1'b0},1'b1}
   <<< (shift-1)`, same widths/signedness) → `out_round_q` identical.
3. SHIFT computes the legacy `v_tmp` expression on `scaled[]` (stable since
   `ST_SCALE`) and `out_round_q` → identical `SCALED_W` signed value.
4. SAT applies the identical clip and writes the identical `data_out` byte
   lanes; `data_out` is sampled downstream only under `valid_out`, so the
   2-cycle-later write time is unobservable.
5. Pass-boundary state: `acc` clear moved `ST_OUTPUT → ST_OUT_SAT`, still
   strictly before the next pass's first gated accumulate (≥4 cycles into
   `ST_MAC`); `k_group=0/oc_group++` still happen on the single cycle
   preceding `ST_MAC`, so the stage-1 prefetch sees the same address sequence;
   the MAC issue/drain logic (incl. DSP_INPUT_PIPE q1/q1b/q2) is untouched.

`TAIL_PIPE=0` (default — MobileNetV2 + every unpatched instantiation):
every new Block-A arm and the ST_SCALE operand select are guarded by the
**elaboration constant** `TAIL_PIPE != 0` → constant-folded away; states 5–7
are never written (`ST_SCALE → ST_OUTPUT`), and their case arms collapse to
`state <= ST_IDLE`, the exact pre-patch `default:` recovery for those
encodings. Byte- AND latency-identical elaboration.

**Module-level proof** (`verify/tail_pipe2_equiv/tb_equiv.sv`): two instances
(TAIL_PIPE=0 vs 1) with the real `node_conv_244` weight/bias/scale mems
(IC=512, OC=256, MP=16, MP_K=8, DSP_INPUT_PIPE=1), 64 random windows,
`--x-initial 0`:
`pixels=64 mismatch_pixels=0 bad_latency_deltas=0 expected_delta=32 result=PASS`
(+32 = exactly `2*OC_PASSES`).

## 4. What the applier changes

`scripts/apply_resnet_tail_pipe2.py` (idempotent via `[TAIL-PIPE2]` /
`.TAIL_PIPE(` markers; every hunk anchor must match exactly once; `--dry-run`;
backups → `backups/resnet_tail_pipe2_<ts>/`):

- `rtl_library/conv_datapath_mp_k.v` — 8 hunks: param (A), state encodings
  (B), tail regs (C), Block-A ST_BIAS prefetch (D), Block-A ST_SCALE
  registered operand (E), Block-A tail-state write arms + SAT acc-clear (F),
  FSM ST_SCALE select (G), FSM tail arms (H).
- `.TAIL_PIPE(1)` on the **35 live** spatial wrappers (hard-asserted set):
  198 200 202 204 206 208 210 212 214 216 218 220 222 224 226 228 230 232
  234 236 238 240 242 244 248 252 256 258 262 268 270 274 276 280 288.
- **Untouched**: `node_conv_196` (stem) and the 9 engine-dispatched files
  (246/254/260/266/272/278/284/292/298) — the applier fails loudly if the
  live/dead split drifts from this expectation.
- `output/rtl/nn2rtl_top.v`: TWO skip-FIFO depth bumps
  (`[TAIL-PIPE2-FIFO]` marker): `u_skid_node_conv_218` 128→1024 and
  `u_skid_node_conv_224` 512→1024 — relu_9's fork receivers, so neither can
  be full on a one-cycle last-beat offer (§1b). Byte-exact by construction
  (lossless order-preserving FIFOs; only elasticity changes); fit-safe
  (DEPTH≥512 ⇒ URAM branch, URAM at 16%).
- NOT in the applier (worktree debug only, consistent with the existing
  unguarded DEBUG_E2E instrumentation in the top): the `[dbg-tp2]` /
  `[dbg-tp2-drop]` forensic taps (fire counters, wrapper capture-miss
  detectors, per-relu last-beat-loss detectors). Remove by deleting the
  `[dbg-tp2]` block if undesired.
- Latency-formula hygiene (`compute_conv2d_latency_cycles` /
  `golden_impl.py`): deliberately **not** changed, same precedent as
  DSP_INPUT_PIPE (2026-06-09 review: contract TBs are value-only, and
  regenerating goldens would corrupt the hand-edited MP16/MP_K8 sidecars).

## 5. Verification results

- **Verilator lint**: clean (exit 0) on `conv_datapath_mp_k.v` standalone for
  TAIL_PIPE ∈ {0,1} × {1x1 MP_K=8, 3x3 MP_K=9 chan-window, 3x3 INT3
  chan-window} (6 configs).
- **Module equivalence TB**: PASS (see §3).
- **Full e2e value gate** (`NN2RTL_VALUE_THREADS=1 npx tsx
  scripts/run_nn2rtl_top_value.ts 0`, worktree):

  **Run 1 (35 convs tail-piped, no FIFO change): WEDGED — the predicted B23
  failure class, reproduced and localized.** All 50176 input beats consumed
  by cyc ~3.1M; every cumulative counter frozen from ~3M onward; `out=0`;
  `spatial_run` high 99.999% of cycles (so NOT a gating bug);
  `ldr0_loaded=0`, `disp_idx=0` forever (scheduler parked in `S_WAIT_LOAD`,
  baseline loads dispatch 0 by cyc ~4M). Apex pinned by the fork counters:
  `r9_emit = r9_to_skid218 = r9_to_skid224 = 25087` of 25088 — relu_9's
  **combined-ready** fork (`relu_9_dual_ready = skid218_ready &
  skid224_ready`) never delivered the LAST beat of its frame. `c218`
  consumed all 25087 forked beats (so skid218 was empty) ⇒ **skid224 was
  full**. Circular wait: skid224(512) full → fork blocked → main arm
  starved (`c218_in=25087`, `c220_out=3132/3136`) → `add_3` join (lhs main
  arm dead, skip arm valid) can never fire → `conv_224`/`skip_node_add_3`
  never drain → skid224 stays full. Downstream the starvation cascades
  (r12/r15/r18/r21 frozen at 464-beat offsets, relu_22 stuck at 5568 of the
  6272 beats ldr0 needs) — identical physiognomy to the 06-05 forensics.
  (Run 1's initial "skid224 full" reading and Runs 2-4's falsification +
  re-localization are detailed in §1b — the actual defect is the lossy
  one-cycle last-beat offer of the narrow-relu streamer, tripped at relu_9's
  fork when `skid_node_conv_218` [DEPTH 128] filled.)

  **Run 4 — FINAL GATE: PASS.**
  `result=PASS beats=3136/3136 mismatch_bytes=0 first_mismatch_beat=-1`
  (vector 0, sim 502 s). **e2e_cycles = 12,813,738** vs baseline 12,670,107
  = **+143,631 = +1.13%** — the throughput cost of +2 cycles/oc_pass across
  the 35 spatial convs (engine eras dominate the rest of the frame and are
  untouched). Last-beat-loss detectors on relus 3/6/9/12/15/18/21/22: **0
  drops chain-wide**; wrapper capture-miss detectors: 0.

  **Quantitative confirmation from the end-of-run `[fifo-peak]` audit**
  (high-water occupancy per skip_fifo):
  - `u_skid_node_conv_218 peak=144` — **exceeds its old DEPTH of 128**: under
    tail-pipe timing this FIFO genuinely saturated, creating the not-ready
    window relu_9's one-cycle final-beat offer landed in. The 1024 depth
    (need ≥144) closes it with ample margin.
  - `u_skid_node_conv_224 peak=488` < its ORIGINAL 512 — proof the run-1
    "skid224 full" hypothesis was wrong (and why run 2's bump changed
    nothing); the bump is kept as margin (old headroom was only 24 beats).
  - `u_skip_node_add_1/2 peak=479` of 512, `add_3 peak=447` of 512 — held,
    but with thin (~33-beat) margins: the next cadence perturbation should
    re-check these (or land the durable relu fix first).

Bottom line: **byte-exact PASS at +1.13% cycles** with the requant tail fully
register-bounded on all 35 live spatial convs.

## 6. Expected Fmax effect

Post-DSP-pipe the binding requant-tail stages were `ST_OUTPUT`
(round+add+barrel-shift+clip, ~12–16ns) and `ST_SCALE` (scale_rom read into
33×16 mult, ~8–11ns) → ~62–83 MHz ceiling. After the split every tail stage is
≤ one add + one barrel shift (or one reg×reg mult) ≈ 6–8 ns → tail no longer
binds below ~85–95 MHz (the 2026-06-05 STEP-2 roadmap band). Caveats from the
06-06 investigation stand: the design is ROUTE/congestion-bound
(fo=11487 csel broadcast, 97% CLB), so this raises the **timing ceiling**, not
necessarily the routed Fmax — it must be confirmed by a legal route, and adding
~66k FF on a 97%-CLB die carries the known congestion-regression risk.
