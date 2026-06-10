# 812-PAIR — paired-channel MAC walk on node_conv_812 (MBV2 final cycle lever)

Date: 2026-06-10 · Base: quartet `bc57baa` (MBV2 frame 1,380,155; 8/8 byte-exact)
Patch: `scripts/apply_mbv2_812pair.py` (single file: `output/mobilenet-v2/rtl/node_conv_812.v`)

## Verdict

| Gate | Result |
|---|---|
| (a) Verilator lint (full engine-top file set, gate-standard flags) | **PASS — 0 errors**, 8 warnings, all pre-existing benign classes (1 DEFOVERRIDE + 7 TIMESCALEMOD on rtl_library files) — identical classes to quartet `lint_stage2.log` |
| (b) MBV2 e2e 8/8 | **PASS — mismatch_bytes=0 on all 8 vectors**; `e2e_cycles=1,184,731` (all 8 identical) |
| (c) FIFO-peak / cadence audit (front zone) | **CLEAN** — see §Cadence audit |
| (d) ResNet vec0 | **N/A — zero shared files touched** (`git status`: only `node_conv_812.v` modified; `rtl_library/line_buf_window.v` byte-identical to HEAD) |

**Frame: 1,380,155 → 1,184,731 = −195,424 cycles (−14.16%).**
Sweep prediction was 1,179,451 (−200,704); measured captures **97.4%** of it. The
+5,280 residual is front-zone fringe (per-pixel spatial-advance overlap is not
perfectly 0 and the stem/ldr0 edges of the zone shave a little of the ideal),
consistent with the DW lane-serial model's `<0.1%`-per-conv calibration band.

Gate logs: `output/mobilenet-v2/reports/pair812/{lint_pair812.log,e2e_pair812_result.txt}`.

## The lever

Post-DW-QUARTET, `node_conv_812` (depthwise 3×3 s1 p1, C=32, 112×112) is the
ONLY spatial depthwise conv left and paces the entire FRONT zone, with the
engine 100% idle under it:

- before: 12,544 px × ceil(32/16)=2 passes × (16 lane-issues + 6) = **44 cyc/px** → 551,936 cyc
- after: 12,544 px × 2 passes × (8 pair-issues + 6) = **28 cyc/px** → 351,232 cyc

Per-pass anatomy (unchanged FSM): issues + 3 (q1/q2 drain) + 3 (BIAS/SCALE/OUTPUT).
Only the ST_MAC issue count changes (16 → 8); drain and tail are identical.

## Design

`lane_counter` becomes a pair-STEP counter (0..7); step *s* issues channels
`{oc_group*16 + 2s, oc_group*16 + 2s+1}` in the same cycle:

- **Lane A (even)** keeps the entire legacy path under its legacy names:
  `current_global_oc` still drives `lbw.channel_select` → `chan_window_flat`,
  `weight_base_addr`, `weight_q/tap_q → prod_q → sum_comb → acc[even]`.
- **Lane B (odd)** is a disjoint twin: `current_global_oc_b = oc_a|1`,
  `weight_base_addr_b`, `weight_qb/tap_qb → prod_qb → sum_comb_b → acc[odd]`.
- The q1/q2 valid/lane/oc pipeline carries the EVEN lane-A indices; the
  accumulate stage derives lane B as `|1` (lanes are even/odd by construction).
  Two `acc[]` writes per cycle hit DISJOINT elements (even vs odd index).
- BIAS / SCALE (per-OC `scale_rom`, DW-CONSTSHIFT mult′ form) / OUTPUT were
  **already 16-lane parallel per pass** — the per-channel requant lane for the
  odd channel always existed; no requant duplication was needed. The
  `[DW-CONSTSHIFT]` constant-shift requant and the `[K1-MBV2]` Block-A
  (sync-only, no-reset datapath regs) structure are preserved; the new lane-B
  regs (`weight_qb/tap_qb/prod_qb`) follow the same K1 class (unconditional
  rewrite, consumed only under reset-kept `mac_valid_q2`).

### The second window port — ZERO shared-file changes

The sweep suggested a second `chan_window_flat` select port on
`rtl_library/line_buf_window.v` (shared with ALL ResNet spatial convs → would
have made the ResNet vec0 gate mandatory). Avoided entirely:

- `EXPOSE_FULL_WINDOW(1)` on the 812 instantiation re-enables lbw's
  full-window output, which inside lbw is a **pure assign-only flatten** of the
  same `window` / `window_kwm1_wire` / `bypass_reg` sources the
  `chan_window_flat` mux reads — no regs, no behavioral change, and for C=32 it
  is only 2,304 wires (not the C≥192 congestion class the 0-setting was built
  for; post-quartet, 812 is the only lbw user in MBV2 anyway).
- Lane B's 9 tap bytes are extracted inside `node_conv_812` using lbw's
  documented identity
  `chan_window_flat[(k)*8 +: 8] == window_flat[(k*IC + channel_select)*8 +: 8]`
  — one C-way byte mux per tap, i.e. exactly the logic a second select port
  would have instantiated, but module-locally.

### Byte-exact by construction

Depthwise channel lanes are fully independent: disjoint weights (`oc*9..oc*9+8`),
disjoint window bytes, disjoint `acc[]`, disjoint per-OC requant slot. With
K_GROUPS=1 each `acc[]` receives exactly ONE accumulate per pass (single-shot
9-tap tree sum), so there is no accumulation-order freedom to perturb — pairing
changes only WHEN each channel's one accumulate lands, not its operands or
ordering. The 8/8 byte-exact gate confirms empirically.

## Cadence audit (gate c)

812 finishes pixels faster (44→28), changing arrival cadence downstream. Class:
rate-change-only (no residual joins in the front zone). Verified:

- **Consumer chain**: 812 → `u_n4_2` (ReLU6, `ENABLE_BACKPRESSURE(1)` true
  elastic skid: holds `out_full` until taken — NOT a one-cycle offer) →
  `u_ldr_node_conv_814` (ldr0, `stream_to_act_bram_bridge`,
  `TOTAL_BRAM_WORDS=12544` = full-frame window, capacity-aware `in_ready`).
  812 itself is BP=1 with `skid_block` freezing scheduler+rearm while parked —
  handshake-based, rate-agnostic.
- **B20 class (narrow-relu last-beat one-cycle offer)**: grep audit of
  `nn2rtl_top_engine.v` — **all 49 node/relu instantiations carry
  `ENABLE_BACKPRESSURE(1)`**; zero legacy push-only hops remain anywhere on the
  spatial chain, so the B20 drop class has no instance on (or off) 812's path.
- **FIFO peaks**: the front zone contains NO `skip_fifo` (first residual-add
  FIFOs sit at `node_add_828`+, all engine-side and paced by the unchanged
  engine dispatch order). Front-zone buffering = 1-deep skids + ldr0's
  full-frame BRAM window behind the act-write arbiter (grant denial → `in_ready`
  low → elastic stall, never a drop). Faster fill of ldr0 only ADVANCES
  `ldr0_loaded`; the scheduler still serializes on it before `engine_start`.
- **Empirical**: 8/8 `mismatch_bytes=0` with `in_beats=50176/50176` consumed and
  `out_beats=32` correct on every vector — any drop/dup/overflow anywhere would
  shift the stream and fail byte-exactness; all 8 vectors also report the
  IDENTICAL cycle count (1,184,731), i.e. fully deterministic cadence.

## Cost (synthesis-facing, not gated here)

- +9 DSP-class 8×8 multipliers (`prod_qb`) + one more 9-input ACC_W tree.
- Weights ROM now needs 18 byte-reads/cycle (was 9) — at 288 bytes total it is
  LUTROM-replication noise.
- The 2,304-wire flatten + a second 9× 32-way byte mux ≈ the area of the
  channel_select mux it mirrors. No new control sets ([K1-MBV2] discipline kept).

## Files / reproduction

- `scripts/apply_mbv2_812pair.py` — 12 anchored replacements; idempotent
  (marker `[812-PAIR`), `--check` proves live == `.prepair` + patch
  (byte-identical, verified), `--revert` restores.
- Backup: `output/mobilenet-v2/rtl/node_conv_812.v.prepair` (untracked).
- Gates: `bash scripts/run_mbv2_e2e_parallel.sh` (PASS 8/8, 62 s wall);
  full-design `verilator_bin --lint-only` with the harness's exact file list +
  `-Wno` set (see `reports/pair812/lint_pair812.log`).

## Promotion notes

1. Promote `output/mobilenet-v2/rtl/node_conv_812.v` + `scripts/apply_mbv2_812pair.py`
   (or run the script in the main checkout — anchors assert against drift).
   No weights / goldens / scale.mem / engine-map changes — RTL-only, no regen
   checklist needed ([[feedback_regen_must_rebuild_engine_maps]] not triggered).
2. Analytic DW model for 812 becomes `OH*OW*ceil(C/MP)*(MP/2+6)`; the in-file
   stale latency comment (MP=4-era "452") was refreshed in the same patch.
3. Combines cleanly with the queued MBV2 wave (FC-on-engine, K-parallel P4,
   DW-on-engine P1): 812 stays spatial, so this lever is orthogonal to all
   engine-side moves. If 812 itself ever moves on-engine, this patch is
   superseded — until then it is the front-zone floor (~351K).
4. New MBV2 cycle baseline for budgeting: **1,184,731**.
