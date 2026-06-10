# FC-ON-ENGINE — moving node_linear (the FC/Gemm classifier) onto the shared engine

**Date:** 2026-06-10 · **Base:** DW-ENGINE EXT (commit `2937dbd`) · **Status:** gates in `output/mobilenet-v2/reports/fc_iso/`

## 1. Goal

node_linear (M=1000 logits x K=1280 features) is a SERIAL MAC wrapper
(`M*(K+2) ≈ 1.282M cycles`) that runs as an UNOVERLAPPED tail at frame end —
~21% of the 6,088,099-cycle DW-EXT frame. As ONE dense engine dispatch
(IC=1280 ≤ MAX_IC 2048, OC=1000 → 4 oc_passes = 1024 lanes, 24 dead) it costs
**5,164 engine cycles** (ISO-measured) — a ~−1.277M cycle win. Dispatch 46 of
47, APPENDED after conv_912 (no SLOT renumbering — unlike the EXT inserts).

## 2. REQUANT IDENTITY PROOF (the make-or-break)

The 8/8 e2e gate compares against the INTEGER FC golden
`node_linear.goldout` (commit 13d23d8). The engine path must be
byte-identical to what `node_linear.v` computes today.

**node_linear.v** (verbatim, `ST_FIN`):

```
biased = acc + bias[m]                      // acc 27b signed, bias 32b, BIASED_W=33
scaled = biased * 4071                      // SCALE_MULT_CONST 16b signed, SCALED_W=49
v      = (scaled + 2^19) >>> 20             // UNCONDITIONAL +half (SCALE_ROUND_HALF; the
                                            // _M1 sign-aware constant is declared but UNUSED)
out    = v > 127 ? 127 : v < -128 ? -128 : v[7:0]    // signed clamp — NO relu
```

**engine requant_pipeline.v** (FIT-FIX constant-shift form, per-OC slot):

```
biased = acc + bias[lane]                   // acc 32b, bias 32b, BIASED_W=33
scaled = biased * mult'                     // mult' = scale slot [30:0], SCALED_W=65
v      = (scaled + 2^22) >>> 23             // ROUND_CONST = 2^(FIXED_SHIFT-1), FIXED_SHIFT=23
out    = v > 127 ? 127 : v < -128 ? -128 : v[7:0]    // signed clamp — NO relu anywhere
                                            // in the engine output path (verified: the
                                            // clamp is the LAST operation before data_out;
                                            // conv dispatches get their relus EXTERNALLY)
```

**Algebraic identity** with `mult' = 4071 << (23-20) = 32568` (fits [30:0]):

```
(B*32568 + 2^22) >>> 23
  = ((B*4071) << 3  +  (2^19) << 3) >>> 23
  = ((B*4071 + 2^19) << 3) >>> 23
  = (B*4071 + 2^19) >>> 20                 -- EXACT for every integer B (incl. negatives):
                                              the <<3 leaves the low 3 bits zero, and an
                                              arithmetic >>>23 of x<<3 == arithmetic >>>20 of x.
```

This is the same `floor((b*m*2^(FS-s) + 2^(FS-1))/2^FS) == floor((b*m + 2^(s-1))/2^s)`
lemma the FIT-FIX requant rests on — here with the per-TENSOR (4071, 20) pair.

**Width/overflow audit** (identity of the ACC too):
* True dot product bound: |acc| ≤ 1280·127·128 = 20,807,680 < 2^25; measured
  max over the 8 golden vectors = **43,783** (the GAP inputs are small).
  node_linear's 27-bit acc and the engine's 32-bit lanes therefore both hold
  the EXACT integer sum — no wrap in either, any accumulation order.
* The MAC walks the same products: engine dense 1x1 walk broadcasts act byte
  `k&255` of word `k>>8` (= node_mean beat layout, ch 256t..256t+255 per beat
  t) against per-lane weight `W[256p+L][k]` — exactly `in_buf2d[k>>8][(k&255)*8+:8]
  * weights[m*K+k]`.
* biased: 33b both sides; scaled: 49b (node_linear) vs 65b (engine) — product
  of a 33b by 16b value fits both. Same clamp literals.

**Empirical proof** (`scripts/extend_mbv2_engine_maps_fc.py --verify-requant`):
for ALL 8 golden vectors x 1000 logits, `(acc+bias)` pushed through BOTH
formulas == each other pre-clamp AND == `node_linear.goldout` byte-exact
(8,000/8,000). Also re-proven in-RTL by gate (b) below (the REAL
requant_pipeline against the same golden).

**Relu check:** logits are signed and must NOT be relu'd. The engine output
path is `mac acc -> bias-add -> mult -> round/shift -> clamp[-128,127] ->
data_out` — no relu exists in `requant_pipeline.v` / `bram_to_stream_bridge.v`
/ `engine_output_bridge`; relus for conv dispatches are EXTERNAL spatial
modules. The FC bridge feeds `output_serializer` directly. VERIFIED.

**VERDICT: identity holds exactly; the engine path is configured (not
approximated) to match — GO.**

## 3. Dispatch configuration

| param | value | note |
|-------|-------|------|
| geometry | 1x1 conv, 1x1 image | KH=KW=1, S=1, P=0, IH=IW=OH=OW=1, px=1 |
| channel_in / k_total | 1280 | 5 act words/pixel (ic_chunks_total=5) |
| channel_out | 1000 | oc_passes = ceil(1000/256) = 4; lanes 1000..1023 dead (zero w/b/s) |
| weight base | 13413 | banks 13413 -> **18533** (+5120 = 4 passes x 1280 taps) |
| bias/scale base | 87 | bias.mem / scale.mem 87 -> **91** (+4) |
| scale slots | 32568 | = mult' (identity proof §2); dead lanes 0 |
| act_in | [25088, +5) | node_mean's 5-beat GAP vector |
| act_out | [25093, +4) | engine scratch, never read; max used 25097 ≤ ACT_DEPTH 25600 |
| depthwise_rom | 0 | DENSE dispatch (the DW rows stay exactly 12) |

## 4. I/O integration (both classes e2e-proven precedents)

* **Input loader** `u_ldr_node_linear` = `stream_to_act_bram_bridge`
  `g_w_eq` (BUS_W=2048 — the exact-width branch every 2048b loader uses):
  node_mean's 5 valid/ready output beats (beat t = channels 256t..255+256t)
  land at act words 25088..25092. node_mean's `out_ready_in` retargets from
  `node_linear_ready_in & spatial_run` to `ldr_fc_in_ready & spatial_run`.
* **Output bridge** `u_engine_out_node_linear` = `engine_output_bridge`
  `OUT_KIND=2` flat-gather (the conv_852/858/... class) with OC=1000,
  POSITIONS=1, DATA_W=8000: gathers the 4 oc_pass beats (lane L of pass p =
  logit 256p+L) and emits ONE 8000b beat = `gather[7999:0]` = logits 0..999
  — dead lanes (beat 3 bytes 232..255) are never emitted. It re-drives the
  old `node_linear_valid_out/data_out` nets; `output_serializer` (32x256b
  m_axis re-slice) is untouched. Handshake follows the bridge convention:
  `ready_out = (ser_ready_out & spatial_run)`, serializer `valid_in` gains
  `& spatial_run` (it was RAW when node_linear-fed).
* **Scheduling:** the FC is the LAST dispatch. Its loader fills during
  dispatch 46's S_WAIT_LOAD window: dispatch 45 (conv_912) fully drains in
  its S_WAIT_DRAIN (n4_35 -> br_mean retile -> node_mean consumes all 245
  beats), node_mean runs SCALE/ROUND/PACK (~240 cyc) and emits its 5 beats
  into the loader; `all_loaded[46]=ldr_fc_loaded` releases engine_start.
  After engine_done + the bridge's single-beat drain (`all_drain[46]`),
  S_NEXT_DISP -> S_DONE.

## 5. Width/count anchors re-audited (47 dispatches)

* **Banks > 2^14 — the one NEW anchor class:** depth 13413 -> 18533 exceeds
  14 bits. `weight_bank_rd_addr` slice widened `[13:0]` -> `[14:0]`, all 8
  `uram_weight_bank` instances `DEPTH(18533) / ADDR_W(15)` (XPM
  ADDR_WIDTH_A follows the parameter; behavioral mem indexes rd_addr
  directly). The ISO harness's own banks were already 17-bit/131072-deep.
* `NUM_DISPATCHES(46) -> (47)` on all 46 pre-existing bridges + 47 on the
  new one (47 instances total). `DC_W = clog2(47)+1 = 7` — unchanged vs 46,
  no truncation (dispatch_count max 47 < 128).
* `LAST_DISPATCH 6'd45 -> 6'd46` (`dispatch_idx` [5:0], max 63 — OK).
* scheduler: all 22 ROMs gain row `6'd46` APPEND-ONLY (rows 0..45 verified
  untouched by the hazard prover PART B); `depthwise_rom` unchanged (12 ones).
* `all_loaded[46]` / `all_drain[46]`: rebound from `1'b1` to the FC loader /
  bridge; rows 0..45 untouched. Rows 47..63 stay `1'b1`.
* No SLOT renumber: the FC is appended, slots 0..45 unchanged (asserted:
  SLOT set == 0..46).
* cfg widths: cfg_ic=1280 (12b OK), cfg_oc=1000 -> oc_pass_total=4 (3b
  oc_pass_idx OK), k_total=1280 (16b k_cnt OK), weight addr ≤ 18532 < 2^22.

## 6. Act-region hazard proof

FC regions [25088,+5) and [25093,+4) sit **ABOVE the GLOBAL act-mem maximum
ever used** (25088 = top of the frame-start d0-write/d1 in-place region
[12544,+12544)) and below ACT_DEPTH 25600 — STRICTLY disjoint from every
dispatch read/write span and every loader fill span, with **NO lifetime
argument at all** (stronger than the P1/EXT DW regions, which overlay
[12544,25088) under the d0/d1-retire rule). An earlier draft placed the FC at
24264 (the DW-region top); the prover caught that 24264 is INSIDE the d0/d1
region — moved to 25088 for the unconditional proof.

`scripts/check_mbv2_act_region_hazards_fc.py` (baseline = `.prefc` = the
DW-EXT files): **PASS** — PART A: all FC-touched pairs strictly disjoint;
PART B: all 46 inherited dispatch rows (regions, kh, wpp, loader bindings,
verdicts) byte-identical to the e2e-proven DW-EXT baseline; C1/C5 hold.

## 7. Gate results

| gate | result |
|------|--------|
| (a) Verilator lint (`--lint-only`, e2e waiver set) | **0 errors** (same 8 pre-existing warnings as the EXT baseline: 1 DEFOVERRIDE + 7 rtl_library TIMESCALEMOD, untouched files) — `lint.log` |
| (b) engine-ISO WLAT=2 vs the INTEGER golden `node_linear.goldout` | **8/8 vectors PASS mismatch=0** (task asked ≥2; ran all 8) — `iso_linear_vec*.log`; engine cost = **5,164 cycles** per frame (cycle-identical across vectors) |
| (c) 8/8 e2e byte-exact + cycle count | **8/8 PASS, mismatch=0, e2e_cycles = 4,811,270 IDENTICAL on all 8 vectors** (was 6,088,099; **−1,276,829 = −21.0%**) — `e2e_full.log` |
| (d) act-region hazard proof | **PASS** — `hazard_fc.log` |

## 8. Cycle accounting

* Old serial FC tail: `M*(K+2)+~7 ≈ 1,282,007` cycles, fully UNOVERLAPPED
  (frame-end, nothing else runs).
* New: ~80 (14 AXI config writes) + load-wait (absorbed in node_mean's
  unchanged pack tail) + **5,164** engine + ~25 gather/drain/handshake.
* Frame MEASURED: **6,088,099 -> 4,811,270 (−1,276,829, −21.0%)**, byte-exact
  + cycle-identical on all 8 vectors. At 200 MHz: 41.6 fps equivalent.
  Session total (P1 + EXT + FC): 7,592,966 -> 4,811,270 (**−36.6%**).
* Engine-serial floor grows by only +5.2K (~0.2%); the frame stays
  SPATIAL-bound (stem conv_810 + DW 812 + the stride-2 quartet).

## 9. LUT note

node_linear's datapath is small (serial MAC + one DSP); the win here is
CYCLES, not area. Its 5 banked 262144x8 weight ROMs (~58 RAMB36 each via
`rom_style=block`) are DELETED from the live netlist (the module file stays
on disk, uninstantiated) and replaced by +5120 URAM bank words (5120x288b ≈
18 URAM-words-worth across the 8 existing banks — they grow 13413->18533,
~2.26Mb total). Net BRAM delta ≈ −290 RAMB36-equivalent bits moved into the
URAM banks' existing slack; to be confirmed at the next MBV2 synth.

## 10. Reproduction / promotion notes

Order (all worktree-relative, idempotent/declarative):
1. `python scripts/extend_mbv2_engine_maps_fc.py [--verify-requant]`
   (after `extend_mbv2_engine_maps_dw.py` + `extend_mbv2_engine_maps_dw_ext.py`
   — asserted via line counts; **joins the regen checklist**: any
   `generate_golden` regen must re-run ALL THREE before the engine maps are
   consumed, per [[feedback_regen_must_rebuild_engine_maps]]).
2. `python scripts/apply_mbv2_fc_engine.py` (restores the DW-EXT baseline
   from `.prefc` first — safely re-runnable/bisectable).
3. Gates: lint -> `scripts/check_mbv2_act_region_hazards_fc.py` ->
   ISO (`scripts/gen_dw_engine_iso_cfg.py linear <vec>` + the verilator
   build in §7 logs, tb/engine_iso_wrap_mbv2.v harness, WLAT=2) ->
   `bash scripts/run_mbv2_e2e_parallel.sh`.
4. `node_linear.v` stays on disk, uninstantiated (same policy as the DW
   convs); its $readmemh paths are absolute main-repo paths but never
   execute (module not instantiated).

Remaining engine-coverage candidates after FC: stem-zone 812 and the
stride-2 DW quartet (need a strided act-read mode — out of scope), and
node_mean (GAP) itself (different op class, ~12K cycles — low value).
