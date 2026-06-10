# DW-ENGINE STRIDE-1 EXTENSION — moving the 9 remaining stride-1 depthwise convs onto the engine

**Date:** 2026-06-10 · **Base:** DW-ENGINE P1 (commit `bc67c94`) · **Status:** gates in `output/mobilenet-v2/reports/dw_ext_iso/`

## 1. Goal

P1 put a param-gated DEPTHWISE mode into the shared engine (config reg `0x3C`,
`address_generator` k_total=KH*KW / loop_ic=1 / act chunk=oc_pass,
`mac_array` per-lane act mux) and moved the 3 wide DW convs (896/902/908,
C=960, 7x7) onto it. This extension moves the **9 remaining STRIDE-1 DW
convs** onto the same machinery — worth ~1.6M cycles of spatial time vs the
7,415,501-cycle P1 frame. The stride-2 quartet (818/830/848/890 — verified
from wrapper localparams; 890's header comment says "STRIDE 1" but its
`SH=2, IH=14, OH=7` localparams prove stride-2) is NOT coverable (the engine
DW walk is shape-generic but the act layout for a stride-2 consumer needs new
windowing) and stays spatial. 812 (C=32, 112x112) stays spatial too: its
in+out regions (2x12544 words) cannot coexist with the stem regions.

## 2. Candidates — verified shapes and dispatch configs

All 9 verified from wrapper localparams (`C/IH/IW/OH/OW/SH/SW/KH/KW/PH/PW`,
asserted by the applier preflight) — every one is 3x3, stride 1, pad 1,
IH=OH, MP=16 spatial:

| conv | C | HxW | px | oc_passes | act wpp | in/out words | dispatch (of 46) | wgt base | b/s base | loader kind | bridge kind |
|------|-----|-------|------|---|---|------|----|-------|----|------------------------|---------------------|
| 824 | 144 | 56x56 | 3136 | 1 | 1 | 3136 | 4 | 13260 | 70 | flat BUS_W=1152 | OUT_KIND=0 low-slice |
| 836 | 192 | 28x28 | 784 | 1 | 1 | 784 | 9 | 13269 | 71 | flat BUS_W=1536 | OUT_KIND=0 low-slice |
| 842 | 192 | 28x28 | 784 | 1 | 1 | 784 | 12 | 13278 | 72 | flat BUS_W=1536 | OUT_KIND=0 low-slice |
| 854 | 384 | 14x14 | 196 | 2 | 2 | 392 | 17 | 13287 | 73 | flat BUS_W=4096 (pad) | OUT_KIND=2 flat-gather (OC=384) |
| 860 | 384 | 14x14 | 196 | 2 | 2 | 392 | 20 | 13305 | 75 | flat BUS_W=4096 (pad) | OUT_KIND=2 flat-gather |
| 866 | 384 | 14x14 | 196 | 2 | 2 | 392 | 23 | 13323 | 77 | flat BUS_W=4096 (pad) | OUT_KIND=2 flat-gather |
| 872 | 384 | 14x14 | 196 | 2 | 2 | 392 | 26 | 13341 | 79 | flat BUS_W=4096 (pad) | OUT_KIND=2 flat-gather |
| 878 | 576 | 14x14 | 196 | 3 | 3 | 588 | 29 | 13359 | 81 | tiled 18 tiles/pos | OUT_KIND=1 tiled-256 (OC=576) |
| 884 | 576 | 14x14 | 196 | 3 | 3 | 588 | 32 | 13386 | 84 | tiled 18 tiles/pos | OUT_KIND=1 tiled-256 |

Engine config per dispatch: `k_total = 9`, `oc_passes = ceil(C/256)`,
per-pixel act words = `ceil(C*8/2048)` = oc_passes (chunk p = channels
256p..256p+255, lane L of pass p = channel 256p+L; dead lanes have zero
weights/bias/scale and the bridge never emits their bytes).

**Dispatch schedule (46):** the EXT inserts each DW conv before its project
consumer: `824@4 836@9 842@12 854@17 860@20 866@23 872@26 878@29 884@32`,
shifting everything after (+1 cumulative); the P1 trio moves `28/31/34 ->
37/40/43`. Final dispatch (912) is slot 45, `LAST_DISPATCH=6'd45`.

## 3. Why three I/O geometry classes (all e2e-proven precedents)

Unlike P1 (where all 3 convs sat on 256b NATIVE_TILED buses), the 9 sit on
three different bus contracts in the P1 top:

1. **824/836/842 — flat narrow** (`wire [1151:0]` / `[1535:0]`): producer
   relu emits ONE flat pixel per beat. Input loader = the existing
   `stream_to_act_bram_bridge` `g_w_lt` branch (1 zero-extended 2048b word
   per pixel — same branch every project-conv loader uses). Output bridge =
   `OUT_KIND=0` low-slice, 1 beat/pos (identical class to conv_820/826/...).
2. **854/860/866/872 — flat 3072b**: loader = `BUS_W=4096` with
   `{1024'b0, data}` pad (2 words/beat `g_w_gt` slicing — byte-identical to
   the existing `u_ldr_node_conv_856/862/868/874` pattern). Bridge =
   `OUT_KIND=2` flat-gather OC=384 (identical to the conv_852/858/864/870
   bridges).
3. **878/884 — NATIVE_TILED 256b**: loader = P1's
   `tiled_stream_to_act_bram_bridge` (18 tiles/pos -> 3 words/pos; the
   TILES_PER_POS=18 last-word gap is 2 tiles, safely backpressured by
   `in_ready` — the ">= 6 beats apart" P1 note was a no-stall observation,
   not a correctness requirement). Bridge = `OUT_KIND=1` tiled-256 OC=576
   (identical to the conv_876/882/888 bridges). The NATIVE_TILED zones use
   RAW valid handshakes; like P1, the consumer relu's `valid_in` gains
   `& spatial_run` when it becomes bridge-fed.

Residual adds (198/.../828/900): NOT touched. The DW zones are strictly
`expand-bridge -> relu -> [DW] -> relu -> project-loader` chains; each
output bridge re-drives the old `node_conv_X_valid_out/data_out` nets with
`ready_out = (consumer_relu_ready_in & spatial_run)`, so everything
downstream (incl. all add joins and skip FIFOs) is wired byte-identically.

## 4. Act-region allocation + hazard proof

New regions live in `[9368, 24264)` (in-region then scratch, sequential):

```
824: in [ 9368,+3136) out [12504,+3136)    872: in [21128,+392) out [21520,+392)
836: in [15640,+784 ) out [16424,+784 )    878: in [21912,+588) out [22500,+588)
842: in [17208,+784 ) out [17992,+784 )    884: in [23088,+588) out [23676,+588)
854: in [18776,+392 ) out [19168,+392 )    896/902/908 (P1): [8192..9368) unchanged
860: in [19560,+392 ) out [19952,+392 )    ACT_DEPTH 25600, max used 24264
866: in [20344,+392 ) out [20736,+392 )
```

All 24 DW regions are pairwise **strictly disjoint** and disjoint from every
ping-pong region (all < 7232). They overlay only the frame-start `ldr0`
([0,+12544)) / `ldr816` ([12544,+12544)) regions, whose consumers (d0/d1)
retire long before the first DW fill opens (d3) — the SAME lifetime argument
that placed P1's regions at 8192+ (C5 `od < fill_start` rule, e2e-proven).

`scripts/check_mbv2_act_region_hazards_ext.py` (baseline = `.preext` = the P1
files): **PASS** — all 18 EXT-touched pairs strictly disjoint (PART A);
all inherited pairs byte-identical to the e2e-proven P1 baseline (PART B);
C1/C5 hold. Side effect: rate-bounded pairs drop 9 -> 3 (six previously
rate-argued fills now happen during the DW dispatch window, after their
reader retired — strictly safer than baseline).

## 5. Maps (append-only, lane-major proof)

`scripts/extend_mbv2_engine_maps_dw_ext.py` (P1 encoder functions reused):
banks 13260 -> **13413** (+153 = sum oc_passes*9), bias/scale 70 -> **87**
(+17 = sum oc_passes). Scale slots are the convs' exact const-shift `mult'`
values from their per-conv `node_conv_*_scale.mem` (post-`a67b39d`,
`[30:0] = mult << (23-shift)`, asserted `< 2^31`). Independent lane-major
second-pass verification PASS; sidecar JSONs extended. Bank `DEPTH`
parameter on all 8 instances: 13260 -> 13413.

## 6. Dispatch-count anchors audited (the P1 SLOT-truncation trap class)

37 -> 46 dispatches:
* scheduler: all 22 ROMs rebuilt 37 -> 46 entries; `depthwise_rom` 3 -> 12
  ones; `LAST_DISPATCH 6'd36 -> 6'd45` (`dispatch_idx` is [5:0], max 63 — OK);
  `LAST_STEP` unchanged (13, the 0x3C write existed since P1).
* every `engine_output_bridge`: `NUM_DISPATCHES(37) -> (46)` on all 37
  pre-existing + 46 on the 9 new (46 total instances). `DC_W` derives from
  NUM_DISPATCHES (clog2(46)+1 = 7 bits) — no truncation.
* `all_loaded`/`all_drain` are `wire [63:0]` — rows 0..45 rebuilt (verified
  row-by-row against the EXT schedule), 46..63 stay `1'b1`.
* SLOT renumber: anchored per instance name, asserted old == P1 index.
* act write arbiter: 9 new grants at lowest priority + en/addr/data mux terms
  (spatial chain is serial — at most one loader active at a time, as before).

## 7. Gate results

| gate | result |
|------|--------|
| (a) Verilator lint (`--lint-only`, harness waiver set) | **0 errors** (8 pre-existing rtl_library TIMESCALEMOD warnings, untouched files) — `lint.log` |
| (b) engine-ISO WLAT=2, per conv vs per-module golden | **9/9 PASS mismatch=0** — `iso_<conv>.log`: 824 = 65,858 cyc; 836/842 = 16,466; 854/860/866/872 = 7,842; 878/884 = 11,566 (sum +153,290 engine cycles) |
| (c) 8/8 e2e byte-exact + cycle count | **8/8 PASS, mismatch=0, e2e_cycles = 6,088,099 IDENTICAL on all 8 vectors** (was 7,415,501; −1,327,402 = −17.9%) — `e2e_full.log` |
| (d) act-region hazard proof | **PASS** — `hazard_ext.log` |

## 8. Cycle accounting

* Engine-side cost added (ISO-measured): **+153,290** cycles serial
  (= px x oc_passes x (9 taps + ~12 FSM/requant overhead) per conv).
* Frame MEASURED: **7,415,501 -> 6,088,099 (−1,327,402, −17.9%)**, byte-exact
  + cycle-identical on all 8 vectors. Session total (P1 + EXT):
  7,592,966 -> 6,088,099 (**−19.8%**). At 200 MHz: 30.4 fps eq.
* The measured win is below the verdict's −1.62-1.65M estimate: the verdict
  counted the spatial DW convs' standalone busy time, but part of that time
  was OVERLAPPED with engine drains/fills in the elastic chain; engine
  dispatching serializes fill -> run -> drain and adds the +153K engine
  cycles. Net −1.33M is the true serial saving.
* The frame remains SPATIAL-bound (stem conv_810 + DW 812 + the stride-2
  quartet dominate); engine-serial floor grows only ~2.5% — the engine is
  NOT the new bottleneck.

## 9. Projected LUT delta

The 17 spatial DW convs measured ~918K LUT (74.7% of design) pre-P1 at
sum(C)=7,136 lanes -> ~129 LUT/channel (MP-16 datapath + window mux
dominate, ~C-proportional). The 9 moved convs carry sum(C)=3,216 channels ->
**~ −350-415K LUT** removed, minus ~15-25K added back (9 loaders, 9 bridges
— the widest is the 824 OUT_KIND=0 slice, trivial; no retile_gather is
deleted here, unlike P1). Net projection: **~ −330-400K LUT**, concentrated
in the same congestion-hot DW zone the pblock pins. To be confirmed at the
next MBV2 synth.

## 10. Reproduction / promotion notes

Order (all worktree-relative, idempotent/declarative):
1. `python scripts/extend_mbv2_engine_maps_dw_ext.py` (after
   `extend_mbv2_engine_maps_dw.py` — asserted; **joins the regen checklist**:
   any `generate_golden` regen must re-run BOTH before the engine maps are
   consumed, per [[feedback_regen_must_rebuild_engine_maps]]).
2. `python scripts/apply_mbv2_dw_engine_ext.py [--convs 824,...]`
   (default all 9; re-runnable with any subset — restores the P1 baseline
   from `.preext` first, so it is bisectable).
3. Gates: lint -> `scripts/check_mbv2_act_region_hazards_ext.py` ->
   `bash scripts/run_mbv2_e2e_parallel.sh`.
   Engine-ISO per conv: `scripts/gen_dw_engine_iso_cfg.py <conv>` + the
   verilator build in §7 logs (tb/engine_iso_wrap_mbv2.v harness, WLAT=2).
4. The 9 spatial wrapper files (node_conv_824.v etc.) stay on disk, unused
   (same policy as P1's trio) — the e2e build compiles them as dead modules.

Remaining DW-on-engine coverage: only 812 (stem-zone, region-capacity-bound)
and the stride-2 quartet (needs a strided act-read mode or a decimating
loader — a NEW windowing feature, out of scope here).
