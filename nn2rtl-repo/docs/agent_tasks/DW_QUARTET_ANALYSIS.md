# DW-QUARTET — moving the 4 STRIDE-2 depthwise convs onto the engine

**Date:** 2026-06-10 · **Base:** ENG-PIPE (commit `5fe7327`; 47 dispatches, frame 1,957,391, 8/8)
**Status:** gates in `output/mobilenet-v2/reports/dw_quartet/`

## 1. Goal + the "stride-2 is config-only" finding

P1/EXT moved the 12 stride-1 depthwise convs onto the shared engine but
declared the stride-2 quartet (818/830/848/890) out of scope ("needs new
windowing"). That verdict was WRONG at the RTL level:
`output/rtl/engine/address_generator.v` (lines ~196-237) ALREADY computes

```
base_r = pixel_h * cfg_stride_h          base_c = pixel_w * cfg_stride_w
in_r   = base_r + kh - cfg_pad_h         (bounds-checked vs cfg_ih/cfg_iw)
act_in = act_in_base + (in_r*cfg_iw + in_c)*ic_chunks + chunk
chunk  = oc_pass_idx                      (DW mode)
```

and the FSM iterates pixels over `cfg_oh/cfg_ow`. The scheduler ROMs already
carry independent stride/pad/ih/oh fields per dispatch. So stride-2 DW on the
engine = CONFIG (scheduler rows with s=2, ih=2·oh) + PACKING (weight/bias/
scale map append) + INTEGRATION (loader sized for the FULL IH×IW input image
+ OUT-sized bridge) — zero new windowing RTL. **No engine-core file is
touched by this change** (ResNet inertness is structural; re-gated anyway).

## 2. The quartet — verified shapes and dispatch configs

All verified from wrapper localparams (asserted by the applier preflight) —
every one is 3×3, stride 2, pad 1, IH=2·OH, MP=16 spatial:

| conv | C | IH→OH | in wds | out beats | passes | loader kind | bridge kind | wgt base (kp8) | b/s |
|------|-----|---------|-------|------|---|------------------------|--------------------|-------|----|
| 818 | 96  | 112→56 | 12544 | 3136 | 1 | flat BUS_W=768  | OUT_KIND=0 low-slice | 18536 | 91 |
| 830 | 144 | 56→28  | 3136  | 784  | 1 | flat BUS_W=1152 | OUT_KIND=0 low-slice | 18545 | 92 |
| 848 | 192 | 28→14  | 784   | 196  | 1 | flat BUS_W=1536 | OUT_KIND=0 low-slice | 18554 | 93 |
| 890 | 576 | 14→7   | 588   | 147  | 3 | tiled 18 t/pos  | OUT_KIND=1 tiled-256 | 18563 | 94 |

The integration classes are EXACTLY the EXT precedents (flat `g_w_lt` loader
/ tiled 18-tile loader; OUT_KIND 0/1 bridges) — the ONLY structural novelty
vs EXT is `loader TOTAL (= IH·IW·chunks) != bridge EXPECTED_BEATS (=
OH·OW·passes)` (4× ratio), which are independent parameters anyway.

**Dispatch schedule:** each conv inserted before its project consumer.
STAGE 1 (830/848/890): 47→50, inserts `830@7 848@16 890@37`, FC→49.
STAGE 2 (+818): 50→51, insert `818@2`, FC→50. `LAST_DISPATCH` 46→49→50;
`dispatch_idx` is [5:0] (max 63) — no truncation; bridge `DC_W` derives from
`NUM_DISPATCHES` (clog2(51)+1 = 7b).

## 3. Maps (append-only in BOTH bank domains, lane-major + re-expansion proofs)

`scripts/extend_mbv2_engine_maps_dw_quartet.py` (P1/EXT encoder functions):

* OLD-domain banks 18533 → **18587** (+54 = Σ oc_passes·9); kp8 banks 2317 →
  **2324** wide lines. KEY ALIGNMENT FACT: appended words start at RELOCATED
  address 18536 = 8·2317 exactly (the +3 FC-pad makes the old image end on a
  line boundary), so the kp8 append is 7 fresh lines (2 zero tail words) with
  NO rewrite of existing lines; construction text-identical to
  `repack_mbv2_kpar8_banks.py` (tap-major `{w[8g+7],…,w[8g]}`).
* Scheduler weight bases use the RELOCATED values (818@18536 830@18545
  848@18554 890@18563) — same domain as the FC row 13416.
* bias/scale 91 → **97** words; scale slots are the convs' const-shift
  `mult'` values from `node_conv_*_scale.mem` (post-`a67b39d`, [30:0] =
  mult<<(23−shift)), asserted <2³¹.
* Proofs: independent lane-major second pass (weights+bias+scale) PASS; kp8
  re-expansion (every tap slice == appended old word) PASS ×8 banks.
* ALL FOUR convs are appended in one run regardless of stage (unwired map
  words are never read) → stage gating is purely RTL-side and bisectable.
* NOTE for future regens: `repack_mbv2_kpar8_banks.py` has OLD_DEPTH=18533
  hard-coded and P0 asserts 47 scheduler rows — a full regen must run the
  quartet maps script AFTER the kp8 repack (it appends to both domains), or
  the repack constants need bumping. **Joins the regen checklist** per
  [[feedback_regen_must_rebuild_engine_maps]].

## 4. Act-region plan + hazard proof (NO act-mem growth; ACT_DEPTH 25600)

The e2e-proven concurrency model (P1→EXT→FC provers): while dispatch d
runs/drains, the ONLY loader filling act-mem is dispatch d+1's; loaders latch
`loaded` after their full region and never write again. So every region needs
checking ONLY against its adjacent dispatch windows — address reuse across
non-adjacent dispatches is structurally non-concurrent.

```
818: in [12544,+12544)  = d(816)'s in-place region (see lag proof below)
     out [0,+3136)      scratch; next fill of [0,..) opens 2 dispatches later
830: in [9368,+3136)    = 824's retired in-window     out [12504,+784)  = 824's retired scratch
848: in [15640,+784)    = 836's retired in-window     out [16424,+196)  = 836's retired scratch
890: in [21912,+588)    = 878's retired in-window     out [22500,+147)  = 878's retired scratch
```

`scripts/check_mbv2_act_region_hazards_quartet.py` (baseline = `.prequartet`,
applied subset auto-detected): **PASS both stages** — every touched adjacent
pair STRICTLY DISJOINT except the 818 overlay, which resolves to two lag
classes:

* **rxf = lag-safe-1x1** (existing, e2e-proven rule): ldr_dw818 fills
  [12544,+12544) with data derived from d(816)'s own bridge beats, so fill
  word i arrives only after 816's monotonic 1×1 walk already read word i.
* **wxf = lag-safe-inplace-fill** (NEW class, same lag argument applied to
  the write port): 816 writes act word i the SAME cycle as the FIFO push of
  beat i (`engine_act_wr_commit = act_out_wr_en & eofifo_in_ready` in the
  top), and the loader's write of word i flows bridge→n4_3→arbiter ≥3 cycles
  later → engine-write(i) < loader-write(i) ALWAYS; final content = the
  relu'd copy 818 then reads. The prover STRUCTURALLY traces the loader's
  in_valid back through the relu stage to `node_conv_816_valid_out` before
  granting the verdict.

Region capacity note: free act-mem is only ~2.3K words ([7232,8192) +
[24264,25088) + [25097,25600)) vs the 5.6K (stage 1) + 15.7K (818) needed —
retired-window reuse was REQUIRED, and the adjacent-pair model makes it
free of any new rate argument except the 818 overlay above.

## 5. Why three integration zones are byte-identical wiring classes

* 818/830/848 (flat 768/1152/1536b): producer relus (n4_3/n4_7/n4_13) were
  `out_ready_in(node_conv_X_ready_in & spatial_run)` → retargeted to the
  loader's in_ready; consumer relus (n4_4/n4_8/n4_14) were ALREADY
  `valid_in(... & spatial_run)` — bridge-fed without change.
* 890 (NATIVE_TILED 256b): producer n4_27 had the RAW handshake
  (`out_ready_in(node_conv_890_ready_in)`, no gate) → retargeted with
  `& spatial_run` added; consumer n4_28's raw `valid_in` gains
  `& spatial_run` (the same NATIVE_TILED-consumer pattern as P1/EXT).
* Residual adds: untouched (the quartet zones are strictly
  expand-bridge → relu → [DW] → relu → project-loader chains).
* The 4 spatial wrapper files stay on disk as dead modules (P1/EXT policy).

## 6. Dispatch-count anchors audited (the SLOT-truncation trap class)

47 → 50 → 51 dispatches:
* scheduler: all 22 ROMs rebuilt (regex-span replace — FC rows carry trailing
  comments); `depthwise_rom` 12 → 15 → 16 ones; stride-2 rows = EXACTLY the
  quartet rows (asserted); `LAST_DISPATCH` bumped.
* every `engine_output_bridge`: `NUM_DISPATCHES(47)→(50/51)` on all 47
  pre-existing + the new ones; SLOT renumber anchored per instance name and
  asserted against the FC slot table; SLOT set verified == 0..ndisp-1.
* `all_loaded`/`all_drain`: rows 0..63 rebuilt row-by-row against the new
  schedule (verified), ties above ndisp-1 stay `1'b1`.
* act write arbiter: new grants at LOWEST priority (after ldr_fc) +
  en/addr/data mux terms; serial spatial chain ⇒ at most one loader active.
* kp8 bank `DEPTH(2317)→(2324)` on all 8 instances.

## 7. Gate results

### STAGE 1 (830/848/890 → 50 dispatches)

| gate | result |
|------|--------|
| (a) Verilator lint (e2e flist + waiver set) | **0 errors** (8 pre-existing rtl_library TIMESCALEMOD, untouched files) — `lint_stage1.log` |
| (b) engine-ISO WLAT=2 (KPAR8+ENG_PIPE deployment config), vec0+vec1 | **6/6 PASS mismatch=0**: 830 = 9,420 cyc; 848 = 2,364 cyc; 890 = 1,776 cyc (vec0==vec1 cycle-identical) — `iso_<conv>_v<vec>.log` |
| (c) act-region hazard proof | **PASS** — all touched pairs strictly disjoint — `hazard_stage1.log` |
| (d) MBV2 8/8 e2e byte-exact + cycles | **8/8 PASS, mismatch=0, e2e_cycles = 1,725,683 IDENTICAL on all 8 vectors** (was 1,957,391; **−231,708 = −11.8%**) |
| (e) ResNet inertness vec0 | **PASS 0/100352 @ EXACTLY 5,664,715** — `resnet_inertness_stage1.log` (structural: no shared engine file touched) |

### STAGE 2 (+818 → 51 dispatches)

| gate | result |
|------|--------|
| (a) lint | TBD |
| (b) ISO 818 vec0+vec1 | TBD |
| (c) hazard proof (818 overlay lag classes confirmed) | TBD |
| (d) MBV2 8/8 e2e | TBD |
| (e) ResNet inertness | TBD |

## 8. Cycle accounting

TBD after gates (raw spatial busy on the old frame: 818=413,952,
830=155,232, 848=51,744, 890=38,808; engine-side serial adds measured by
ISO; net frame deltas measured by the 8/8 gate).

## 9. Reproduction / promotion notes

Order (all worktree-relative, idempotent/declarative):
1. `python scripts/extend_mbv2_engine_maps_dw_quartet.py` (after the FC/KPAR8
   maps chain; asserted by the applier preflight).
2. `python scripts/apply_mbv2_dw_engine_quartet.py [--convs 830,848,890]`
   (default all 4; restores the FC baseline from `.prequartet` first —
   re-runnable with any subset, bisectable).
3. Gates: lint → `scripts/check_mbv2_act_region_hazards_quartet.py` →
   `bash scripts/run_mbv2_engine_iso_quartet.sh <convs>` →
   `bash scripts/run_mbv2_e2e_parallel.sh` → ResNet
   `NN2RTL_VALUE_THREADS=1 NN2RTL_VALUE_XINIT=0 npx tsx scripts/run_nn2rtl_top_value.ts 0`.

Remaining DW-on-engine coverage after the quartet: ONLY 812 (C=32, 112×112,
stem-zone; its 2×12544-word regions cannot coexist with the stem regions —
unchanged verdict from EXT).
