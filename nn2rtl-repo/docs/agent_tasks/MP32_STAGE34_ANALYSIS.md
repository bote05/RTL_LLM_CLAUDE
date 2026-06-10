# MP32 STAGE-3/4 — ResNet stage-3/4 spatial 1x1 convs MP 16→32

**Base:** a1f088a (KPAR4-RN; frame 7,229,214 cyc, vec0+vec1 byte-exact)
**Applier:** `scripts/apply_resnet_mp32_stage34.py` (anchor-asserted, idempotent,
backups in `backups/mp32_stage34/`, `--revert`, `--convs` bisectable)
**Date:** 2026-06-10

## 1. Why

Post-KPAR4 attribution: the engine region's inter-dispatch windows are paced by
the stage-3/4 spatial 1x1 convs at MP=16. The six conv_248-class 256→1024 convs
cost exactly 501,760 cycles each (196 px × 64 OC-passes × 40 cyc/pass: 32
K-groups + 6 stages + TAIL_PIPE2's +2) and the engine at K_PAR=4 waits on them
~62% of the region. MP=32 halves the OC-pass count → ~250,880 each.

## 2. Scope (verified before patching)

11 live spatial `conv_datapath_mp_k` wrappers, all `MP=16, MP_K=8`, all 1x1,
all instantiated in `nn2rtl_top.v` (conv_264 is engine-dispatched → correctly
NOT in scope):

| conv | shape | join (lhs) | input skid |
|------|-------|-----------|------------|
| 248 | 256→1024 14² | add_7 via `u_skip_node_add_7` (already buffered) | 2048 |
| 252 | 1024→256 14² | — (mid-block) | 2 → **64** |
| 256 | 256→1024 14² | **add_8 direct tie → lhs skid** | 2048 |
| 258 | 1024→256 14² | — | 2 → **64** |
| 262 | 256→1024 14² | **add_9 direct tie → lhs skid** | 2048 |
| 268 | 256→1024 14² | **add_10 direct tie → lhs skid** | 2048 |
| 270 | 1024→256 14² | — | 2 → **64** |
| 274 | 256→1024 14² | **add_11 direct tie → lhs skid** | 2048 |
| 276 | 1024→256 14² | — | 2 → **64** |
| 280 | 256→1024 14² | **add_12 direct tie → lhs skid** | 2048 |
| 288 | 1024→2048 14²→7² s2, **INT3** | add_13 via `u_skip_node_add_13` (already buffered) | 4096 |

## 3. The deadlock class, engineered out BEFORE the first run

History: MP-increase deadlocked twice (B22; `project_mp_increase_deadlock`) —
root cause was beat-desync at the synchronized residual-add joins, and the
TAIL_PIPE2 forensics (§1b of `TAIL_PIPE2_ANALYSIS.md`) later proved the latent
defect is the **narrow-relu one-cycle last-beat offer** (B20 class): every relu
pixel's final beat is presented for exactly one cycle; if the (often combined
fork) ready is low that cycle the beat is silently dropped and the downstream
lockstep join wedges permanently. ANY cadence change re-rolls those dice.

Three mitigations shipped together with the MP change:

1. **LHS skid on add_8..12** (template: the prepped-never-built
   `scripts/apply_conv202_lhs_skid.py`): conv_{256,262,268,274,280} drained
   DIRECTLY into their join with the circular tie
   `.ready_out(skip_valid & spatial_run & add_ready_in)`. Each now drains into
   a DEPTH=512 `skip_fifo` (`u_skip_node_add_N_main`, URAM FWFT — same class
   as the RHS skips); the join consumes the buffered arm and the RHS pop gate
   swaps `conv_valid_out` → `main_valid`. FIFO preserves value+order →
   byte-exact. add_7/add_13 need nothing: their accelerated convs (248/288)
   already ARE the buffered (skip-FIFO) arm.
2. **Fork-receiver bumps 2→64** on `u_skid_node_conv_{252,258,270,276}` — the
   post-add relus 24/27/33/36 fork into these DEPTH=2 receivers; a full
   receiver on a last-beat-offer cycle is exactly the B20 trigger (TP2
   precedent: skid218 128→1024). LUTRAM, value-preserving.
3. **B20 drop detectors** on the 14 stage-3/4 narrow relus
   (23,24,26,27,29,30,32,33,35,36,38,39,41,42): a counter flags the exact drop
   signature `valid_out_d & ~accepted_d & ~valid_out` + `final $display`
   (`[b20-drop] relu_N drops=…`). $display-only sink → synth-pruned, same
   convention as the `[fifo-peak]` audit. Gate requires all 0.

## 4. Weight repack (the MP-dependent artifact)

`conv_datapath_mp_k`'s ROM word is `MP*MP_K*WGT_BITS` bits, lane-major
(`bits[(lane*MP_K+kpos)*WGT_BITS +: WGT_BITS]`), so the wide hex packing is a
function of MP. The applier repacks from the flat `node_conv_<id>_weights.hex`
via `repack_weights_wide.write_wide_weights(..., mp=32, mp_k=8, wgt_bits=4|3)`
into **new filenames** `node_conv_<id>_weights_mp32_k8.hex` (the canonical
`_mp_k_8.hex` files encode MP only implicitly — same name at a different MP is
the silent-garbage footgun; a new name makes staleness structurally
impossible). `WEIGHTS_PATH` is rewritten to the repo-root-derived absolute
path, so applying the script in any checkout points at that checkout's weights.

- **conv_288 footgun honored: INT3 → repacked at WGT_BITS=3** (3-bit stride +
  mask; 8192 entries × 768 bits).
- Every repack is inverse-verified: full unpack of the wide file compared
  against the flat source (262,144 weights × 10 INT4 convs + 2,097,152 × 1
  INT3 conv, all equal).
- `scale.mem` / `bias.hex` index by ABSOLUTE output channel
  (`oc_group*MP+lane`) → MP-independent → untouched.

## 5. Atomic-arch rule (latency formula + headers + goldens)

- `scripts/onnx_frontend.py`: added `_MP32_STAGE34_OVERRIDE` (the 11 ids → 32)
  consulted in `_conv_mac_parallelism`, and `_conv_mp_k` returns 8 for them —
  same convention as the MBV2 `_A2_MP_OVERRIDE`, keeping
  `compute_conv2d_latency_cycles` consistent with the live RTL for any future
  regen (pointwise dense mp_k branch: `1 + OC_PASSES*(K_GROUPS + 6)`).
- Wrapper headers: the `MP=16` comments in the 11 `.v` files updated to 32.
- **Goldens untouched** (asserted by the applier): goldens are VALUE streams;
  MP moves only timing. Nothing in this change regenerates them; the e2e gate
  compares against the existing contract goldens.

## 6. Gates (all green)

- **Lint:** Verilator `--lint-only` over the full e2e source set: **0 errors /
  0 warnings** (same waiver set as the value runner).
- **e2e vec0:** `NN2RTL_VALUE_THREADS=1 NN2RTL_VALUE_XINIT=0
  npx tsx scripts/run_nn2rtl_top_value.ts 0` →
  **result=PASS beats=3136/3136 mismatch_bytes=0**.
- **e2e vec1:** (`NN2RTL_VALUE_RUNONLY=1 … 1`) → **result=PASS
  mismatch_bytes=0**.
- **e2e_cycles: 5,664,715** (vec0) vs 7,229,214 baseline = **−1,564,499
  (−21.6%)** — slightly BETTER than the predicted 0.9–1.5M window.
  vec1: 5,664,715 (cycle-identical). (An intermediate bisect build during
  bring-up measured 6,084,409; superseded by this final full-application run —
  console preserved as `mp32_vec0_console.log`.)
- **[b20-drop]: 0 drops on all 14 instrumented relus, both vectors.**

### FIFO peaks (final build, vec0 — vec1 identical; the headroom evidence)

| fifo | depth | peak | margin |
|------|-------|------|--------|
| u_skip_node_add_8  | 8192 | 6271 | 1921 (full skip tensor parks; MP32 cannot raise it past 6272) |
| u_skip_node_add_9  | 8192 | 6271 | 1921 |
| u_skip_node_add_10 | 8192 | 6271 | 1921 |
| u_skip_node_add_11 | 8192 | 6271 | 1921 |
| u_skip_node_add_12 | 8192 | 6271 | 1921 |
| u_skip_node_add_13 (conv_288's output join skip) | 4096 | 3135 | full 7²-stage skip tensor parks; 961 margin |
| u_skid_node_conv_288 (conv_288's input skid) | 4096 | 2688 | 1408 margin (intermediate bisect build hit 4096-full; final MP32 conv_288 drains it) |
| u_skip_node_add_N_main (new lhs skids ×5) | 512 | 1 | join-paced; phase-slip absorber barely used = healthy |
| u_skid_node_conv_{252,258,270,276} | 64 | 1 | the 2→64 bump is pure OFFER-cycle margin — detectors prove 0 drops |
| u_skid_node_conv_280 | 2048 | 856 | mid-chain feeder, ample |
| u_skip_node_add_1/2 (the TP2 thin-margin watch) | 512 | 479 | unchanged vs TP2 run-4 (stage-1 cadence not perturbed) |
| u_skip_node_add_7 | 2 | 2 | baseline 2-deep lossless skid, untouched; full = normal backpressure (producer is the backpressured conv_248 streamer, not a B20 one-cycle-offer relu) |

(Exact per-instance numbers in
`output/reports_integrated/verilator_nn2rtl_top_value/run.log` and
`mp32_vec0_console.log`.)

## 7. Area delta (for the synth decision)

- **DSP:** +16 lanes × MP_K=8 = +128 multipliers per conv × 11 = **+1408
  4b/3b×8b multipliers** (use_dsp=yes → ~1 DSP each): 12,288-DSP U250 goes
  ~60-67% → **~72-78%**. If DSP-tight, these narrow products are LUT-mappable.
- **BRAM (weight ROMs):** total bits UNCHANGED (same weights); geometry goes
  2× wider / half depth (e.g. conv_248: 2048×512b → 1024×1024b; both pack to
  ~29 RAMB36 bit-bound) → **~neutral (±5% packing noise)**.
- **URAM:** 5 new DEPTH=512 lhs skids × (256b/72) = **+20 URAM288 (+1.6% of
  1280)**.
- **LUT/FF:** fork bumps 4×64-deep LUTRAM ≈ +1.4K LUT; doubled lane regs
  (acc/biased/scaled/tail regs) ≈ +3.6K FF & +1.8K LUT per conv → **≈ +40K FF
  / +21K LUT** (FF was 38%, LUT the watch item). Detector counters are
  display-only → pruned.

## 8. Promotion notes

1. Promote the worktree commit (RTL + applier + frontend override + this doc).
2. In the main repo run `python scripts/apply_resnet_mp32_stage34.py` — it is
   the atomic unit: patches RTL (if promoting the script rather than the .v
   files), repacks `*_weights_mp32_k8.hex` from the main repo's flat weights
   (inverse-verified), and points WEIGHTS_PATH at the main repo's absolute
   path. **The worktree .v files carry worktree-absolute WEIGHTS_PATHs** — in
   the main repo either re-run the applier after `--revert`-free promotion or
   sed the path prefix; the applier run is the recommended route.
3. The old `_mp_k_8.hex` files stay valid for MP=16 checkouts; nothing shared
   was clobbered.
4. Re-gate: lint + vec0 + vec1 byte-exact (+ `[b20-drop]` all 0) before any
   Vivado run, per the hard rule.
5. Known watch items: `u_skip_node_add_1/2` margins (~33 beats, TP2 finding)
   are untouched by this change but remain thin for ANY future cadence
   perturbation — the durable fix is still the elastic-hold narrow-relu
   retrofit (queued).
