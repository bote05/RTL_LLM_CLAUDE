# DW-ENGINE P1 — MobileNetV2 wide depthwise convs on the shared engine

2026-06-10. Branch state: a67b39d (mbv2 DW-CONSTSHIFT) + this change.
ALL FOUR GATES GREEN. Built and verified in the agent worktree.

## What moved

The 3 WIDE depthwise convs `node_conv_896 / 902 / 908` (C=960, 3x3, stride 1,
pad 1, 7x7 feature map — the NATIVE_TILED retile wrappers that, with their
retile_gather bridges, are the pblock-pinned congestion drivers, ~376K LUT =
30.6% of the design) moved from the spatial fabric onto the shared engine as
DEPTHWISE dispatches **28 / 31 / 34** (34 -> 37 dispatches). The 3
`retile_gather` bridges `br_ldr28/30/32` are DELETED outright (their
pixel-alignment job is absorbed by a new ~120-line word-aligning loader).

Per conv the chain `conv_894 -> n4_29 -> [896] -> n4_30 -> [898]` becomes:

```
bridge27(conv_894) -> n4_29(relu, unchanged) -> tiled loader  -> act[8192..8387]
engine dispatch 28 (DEPTHWISE)  reads act[8192..8387], writes scratch[8780..]
   + its FIFO stream -> bridge SLOT 28 (OUT_KIND=1, OC=960, POS=49)
   -> node_conv_896_valid_out/data_out (the SAME 30x256b tile stream the
      spatial conv used to drive) -> n4_30(relu, unchanged)
n4_30 -> tiled loader (CONVERTED ldr28; br_ldr28 retile_gather DELETED)
   -> act[0..195] for dispatch 29 (conv_898)
```

## The engine DEPTHWISE mode (the ~50-line per-lane lever)

All shared with ResNet; **provably inert when disabled** (`ENABLE_DEPTHWISE=0`
default ties `dw_mode` to a hard 0; ResNet's scheduler never writes reg 0x3C,
which resets to 0 — every dense expression reduces to the original form, bit-
and cycle-identical):

* `output/rtl/engine/config_register_block.v` — new reg **0x3C DEPTHWISE**
  (bit 0) -> `cfg_depthwise` output.
* `output/rtl/engine/address_generator.v` — `cfg_depthwise` input; 3 muxes:
  `k_total = KH*KW` (9 taps, no ic reduction), `loop_ic = 1` (ic counter
  parks at 0, kw/kh walk gives tap t = kh*3+kw at cycle k), and the act-word
  channel-chunk index = `oc_pass_idx` (lane L of pass p = channel 256p+L).
  Padding falls out for free: out-of-bounds taps drop `act_in_rd_en`, which
  already gates `mac_valid_in` (skip == accumulate 0 == the spatial conv's
  zero-pad).
* `output/rtl/engine/mac_array.v` — new `dw_mode` + `act_word[2047:0]` ports;
  per-lane act select `dw_mode ? act_word[lane*8 +: 8] : act_byte` feeding the
  same DSP multiply. `act_word` is the skeleton's `act_in_rd_data_d` (the
  2-cycle-URAM-aligned held word — same N+2 alignment as the dense byte mux).
* `output/rtl/shared_engine_skeleton.v` — `ENABLE_DEPTHWISE` param, the
  `dw_mode` gate, wiring; iverilog stubs updated to match.

Requant is UNCHANGED: per-OC bias word + per-OC constant-shift scale word
(slot[30:0] = mult' = mult << (23-shift), FIXED_SHIFT=23, unconditional +2^22
round, clamp) — arithmetically identical to the spatial DW conv's
[DW-CONSTSHIFT] datapath (engine biased is 33b vs spatial 34b; DW acc <= 9 *
127 * 127 + bias fits both — no overflow), hence byte-exactness by
construction once the maps carry the SAME per-channel values.

## Lane mapping / memory layout

* **Channels -> lanes:** oc_pass p in 0..3 covers channels 256p..256p+255;
  lane L computes channel 256p+L; pass 3 lanes 192..255 are dead (zero
  weights/bias/scale -> deterministic 0; the OUT_KIND=1 bridge never emits
  bytes 192..255 of the last beat, so they never reach the chain).
* **Weights** (appended to `uram_weights_bank0..7.mem`, now 13260 words):
  word (base + p*9 + t), t = kh*3+kw; lane L byte = `weights.hex[(256p+L)*9+t]`
  — matches the DW address walk exactly. Bases 13152 / 13188 / 13224.
* **Bias/scale** (`bias.mem`/`scale.mem`, 58 -> 70 words): word (base + p),
  slot L = per-channel value from the spatial conv's own hex/mem files.
  Bases 58 / 62 / 66. All existing words byte-identical (append-only, with
  independent lane-major re-verification — `extend_mbv2_engine_maps_dw.py`).
* **Act regions:** DW inputs 8192/8388/8584 (+196 each), engine act_out
  scratch 8780/8976/9172 (+196; never read — the FIFO->bridge stream is the
  consumed copy, same redundant-write pattern as every other dispatch). All
  six regions live inside the stem-only region (consumed at dispatch 0, dead
  by dispatch 27) and are mutually disjoint.

## New RTL building block

`tiled_stream_to_act_bram_bridge` (defined in the top's module section):
256b-tile stream -> act BRAM with PER-POSITION word alignment (8 tiles/word,
partial 4th word zero-flushed per pixel) — byte-identical layout to
retile_gather(OUT_BEATS=4)+2048b-loader, which is exactly the
ceil(C/256)-chunks-per-pixel layout the engine reads (dense AND depthwise).
Six instances: 3 NEW DW input loaders (from n4_29/31/33) + 3 CONVERSIONS of
ldr28/30/32 (from n4_30/32/34 — this is what deletes br_ldr28/30/32).
Word submissions are >= 6 producer beats apart, so single-cycle arbiter
denial never drops a beat (`in_ready` holds the elastic relu producer only on
a word-completing tile while the previous word still awaits its grant).

## Dispatch configs (scheduler rows 28/31/34)

ic=oc=960, k=3x3, s=1, p=1, ih=iw=oh=ow=7, depthwise=1 (new ROM + write step
13 -> reg 0x3C; LAST_STEP 12->13, LAST_DISPATCH 33->36), weight bases
13152/13188/13224, bias/scale bases 58/62/66, act_in 8192/8388/8584, act_out
8780/8976/9172, scale_mult/shift 0 (vestigial — requant is per-OC).
Renumber: old 28..33 -> 29/30/32/33/35/36; engine_output_bridge SLOTs moved
accordingly, `NUM_DISPATCHES` 34->37 on all 37 bridges (DC_W auto-widens from
the param — no ResNet-K5-style SLOT-truncation trap), `all_loaded`/`all_drain`
remapped ([63:0] vectors already wide enough).

## Hazard proof (gate d) — scripts/check_mbv2_act_region_hazards.py

Two-part structure (the MBV2 OVERLAP design already shipped with rate-bounded
read-vs-concurrent-fill overlaps — e.g. stride-2/channel-expanding spatial
segments — that are NOT address-monotonic-provable and are covered by the
baseline's byte-exact e2e):

* **PART A (strict):** every pair the DW change touches (DW reads/writes/
  loaders + fills concurrent with dispatches 27/28/30/31/33/34) is STRICTLY
  DISJOINT. PASS.
* **PART B (equivalence):** every inherited pair is identical to the pre-DW
  baseline (parsed from the .predw backups) under the renumber map, or became
  strictly disjoint. 9 inherited rate-bounded pairs, all baseline-identical.
  PASS.
* C1 (loader region == engine read region, 36 loaders + the loader-less
  conv_876 chained read), C5 (bounds + DW-region lifetime vs the stem loader's
  dead data) PASS.

## Gate results

| gate | result |
|---|---|
| (a) Verilator lint | 0 errors (only pre-existing benign TIMESCALEMOD warnings) |
| (b) ENGINE-ISO, WLAT=2 | conv_896 vec0: 47040 bytes mismatch=0; conv_902 vec0: mismatch=0; conv_908 vec0 AND vec5: mismatch=0 (3824 engine cycles/dispatch) |
| (c) full e2e | **RESULT: PASS (8/8 byte-exact)**, mismatch 0 on all vectors |
| (c) cycles | **e2e_cycles = 7,415,501** (identical all 8 vectors) vs baseline 7,592,966 -> **-177,465 (-2.34%)**, inside the predicted 7.40-7.50M band |
| (d) hazard prover | PASS (PART A strict + PART B baseline-equivalent) |

## LUT-delta expectation (to confirm at synth — no Vivado run this session)

DELETED: 3x spatial wide-DW conv (C=960 line_buf_window TILE_STORAGE +
240-pass MP=16 datapath + per-conv ROM/control) + 3x retile_gather (2x7680b
ping-pong regs + SYNTH_FIXED_MUX trees) ~= -376K LUT (30.6%) per the
feasibility analysis. ADDED: per-lane act mux (256 x 8b 2:1), 6 tiled
loaders (~2048b datapath each), 3 OUT_KIND=1 bridges, +108 bank words
(URAM depth, BRAM-free) ~= +15-25K LUT. Net expectation ~= **-350-360K LUT**
plus the deletion of the DW convs' URAM/BRAM line buffers; these were the
pblock congestion drivers, so the routability win is the headline.

## Reproduction chain (worktree-relative)

1. `python scripts/extend_mbv2_engine_maps_dw.py`   (banks/bias/scale append + proof)
2. `python scripts/apply_mbv2_dw_engine_p1.py`      (top + scheduler surgery; asserts the engine-core [DW-ENGINE P1] edits + extended maps; idempotent; .predw backups)
3. gates: verilator lint -> `scripts/gen_dw_engine_iso_cfg.py {896,902,908}` + engine-ISO build -> `bash scripts/run_mbv2_e2e_parallel.sh` -> `python scripts/check_mbv2_act_region_hazards.py`

Engine-core edits (in the git diff, asserted by the apply script):
`output/rtl/engine/{config_register_block,address_generator,mac_array}.v`,
`output/rtl/shared_engine_skeleton.v`, `tb/engine_iso_wrap_mbv2.v`,
`tb/engine_iso_wrap_mbv2_tb.cpp`.

## Main-tree promotion caveats

* The shared engine files are used by ResNet too. The change is provably
  inert there (param default 0 + reg 0x3C never written + resets to 0), but a
  ResNet e2e re-gate on promotion is cheap insurance.
* `bias.mem` / `uram_weights_bank*.mem` are untracked artifacts — promotion
  must re-run `extend_mbv2_engine_maps_dw.py` against the main tree's weights
  dir (it is idempotent and self-verifying), NOT copy from the worktree.
* Per feedback_regen_must_rebuild_engine_maps: any future
  `generate_golden`-class regen must ALSO re-run `extend_mbv2_engine_maps_dw.py`
  (the DW convs are not in the heavy-pointwise list the standard map builders
  walk) — or extend `mbv2-heavy-pointwise.txt`/build scripts to include them
  natively in PHASE 2.
* The spatial `node_conv_896/902/908.v` files remain on disk (uninstantiated;
  auto-pruned by synth, same as ResNet K5's precedent).
* `scripts/run_mbv2_synth.ts` collectSources includes the whole rtl dir — the
  `.predw` backups do not end in `.v`, so they are not picked up.
