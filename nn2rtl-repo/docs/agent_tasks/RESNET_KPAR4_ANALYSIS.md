# KPAR4-RN — RESNET ENGINE K-PARALLEL P=4 (4 taps/cycle/lane, incl. dense 3x3)

Date: 2026-06-10 · Base: d76ed8f (MBV2 KPAR4 shipped; ResNet frame 9,622,057)
Scripts: `scripts/apply_resnet_kpar4.py` (RTL, anchor-asserted/idempotent,
`.prekp4r` backups), `scripts/repack_resnet_kpar4_banks.py` (bank repack +
3x3 transposition + proofs P0-P4), `scripts/gen_resnet_engine_iso_cfg.py` +
`scripts/run_resnet_engine_iso_kpar.sh` (ISO A/B gate),
`tb/engine_iso_wrap_resnet.v` (ResNet-geometry ISO wrapper).

## HEADLINE

| metric | before (d76ed8f) | after KPAR4-RN | delta |
|---|---|---|---|
| e2e frame cycles (vec0, vec1 identical) | 9,622,057 | **7,229,214** | **−2,392,843 (−24.9%)** |
| e2e byte-exactness vec0 AND vec1 | PASS 0/100352 | **PASS 0/100352** | — |
| BRAM weight-bank bits/bank | 67072×96 = 6,438,912 | 16768×384 = 6,438,912 | **0 (67072%4==0, zero pad)** |
| MBV2 8/8 gate (shared-RTL re-gate) | 2,620,778 | **2,620,778 (8/8 PASS)** | cycle-EXACT |

SHIPPED: FULL fast P=4 on ALL 17 ResNet dispatches — 8 dense 1x1 **AND the
9 dense 3x3** (conv_246 + the seven 14x14 3x3s + the K5 trio 284/292/298),
via a pos-major bank transposition that MBV2 did not need. No serial
fallback remains in the ResNet dispatch set.

## THE 3x3 EXTENSION (what is new vs MBV2's KPAR4)

MBV2 gated fast eligibility to dense **1x1** because of a layout/walk
mismatch, NOT a datapath limitation:

* The AG walks K with **ic innermost** (kh, kw, ic), but the legacy weight
  layout is **ic-major**: word = base + pass·KT + ic·KH·KW + (kh·KW+kw).
  For 1x1 (KH·KW=1) walk order == address order → 4 consecutive walk steps
  are 4 consecutive words (one repacked line). For 3x3 they have address
  STRIDE 9 — unfetchable in one line.
* Fix: **transpose each 3x3 region to pos-major** in the repacked banks
  (word at (kh·KW+kw)·IC + ic) and give the AG's fast walk the matching
  address formula. For 1x1 the two formulas coincide (pos=0), so MBV2's
  untransposed banks remain correct under the same RTL.

Why a 4-group is always safe in a dense KxK walk (the load-bearing facts):
* IC%4==0 ⇒ k%4 == ic%4 ⇒ a 4-aligned group never crosses a (kh,kw)
  boundary: all 4 taps share ONE (kh,kw) position.
* Same position ⇒ same act pixel word, same ic chunk (ic0%4==0, 256%4==0 ⇒
  bytes ic0..ic0+3 in one 2048b word), and ONE in_bounds decision — a
  padded position drops act_in_rd_en for the whole group, contributing
  exactly 0 just as 4 serial skipped steps did.
* K_TOTAL = IC·KH·KW %4==0 ⇒ no partial groups, mask always 4'b1111.

Eligibility (all per-layer constants): `!depthwise && IC%4==0 && IC>=4 &&
weight_base%4==0` (the KH==KW==1 term DELETED). Verified for all 17 ResNet
dispatches: bases {0, 2304, 4352, 6656, 8960, 9984, 12288, 14592, 16896,
18944, 28160, 32256, 36352, 45568, 49664, 53760, 62976} all %4==0
(asserted as proof P0 on every repack run), IC ∈ {256,512,1024,2048}.
P0 also proves the 17 regions tile [0, 67072) EXACTLY — nothing else lives
in the banks, so the transposition cannot alias another consumer. The
eligibility assert is load-bearing: with transposed banks the SERIAL path
would fetch wrong words for 3x3 regions, so every dispatch MUST be fast
(all 17 are; the serial walk survives only for foreign tops e.g. MBV2 DW/FC).

## RTL CHANGES

### Shared (output/rtl/engine/address_generator.v — K_PAR>1 branch ONLY)
`scripts/apply_resnet_kpar4.py` hunks (the K_PAR==1 verbatim-legacy branch
is untouched; ResNet's and MBV2's K_PAR=1 instances are textually identical
to before):
1. `kpar_fast` drops `(cfg_kh==1 && cfg_kw==1)`.
2. New fast weight address: `weight_addr_next_fast = base + pass_offset +
   (kh_offset+kw)·IC + ic` (pos-major); `weight_rd_addr <= kpar_fast ?
   fast : legacy` (split-anchored to the g_walk_kpar branch only).
3. New fast ic-wrap advance: at `ic_cnt == IC-4` inside a (kh,kw) position,
   reset ic, advance kw/kh, `k_cnt += 4`. **Unreachable for 1x1** (there
   `k_at_last` fires the same cycle and has if-priority) ⇒ MBV2's fast
   walks are cycle-identical — re-gated, see gate (d).

mac_array.v and shared_engine_skeleton.v needed **zero changes**: the
KPAR4 RTL is geometry-generic over WGT_W (8→3) and URAM_DATA_W (8192→3072)
— the tap slices are `(j*MAC_COUNT + lane)*WGT_W`, the subword select is
`wsub_d2*(MAC_COUNT*WGT_W)`, and the act-side byte selects are ACT_BUS_W-
relative (2048b on both nets). Lint 0 at both configs
(`output/reports_integrated/kpar4rn/lint_{legacy,kpar4}.log`).

### ResNet top (output/rtl/nn2rtl_top.v)
* `ENGINE_K_PAR=4`; `ENGINE_WBUS_W = 4*8*ENGINE_LANE_B = 3072`.
* Banks: DEPTH 67072→16768, ADDR_W 17→15, WORD_W 96→384 (`_kp4.mem`).
* `weight_bank_rd_addr = engine_weight_rd_addr[14:0]` (engine exports the
  GROUP address old>>2; fast groups are 4-aligned ⇒ subsel 0).
* Bus: tap-major generate — tap j word = concat over banks of
  `bank[j*ENGINE_BANK_W +: ENGINE_LANE_B]` (lane order per tap identical).
* `.K_PAR(ENGINE_K_PAR)` on `u_shared_engine`.

### Weight banks (scripts/repack_resnet_kpar4_banks.py)
Two-step: (1) pos-major TRANSPOSE of the 9 3x3 regions (per oc-pass block:
`T[blk + pos*IC + ic] = OLD[blk + ic*KHKW + pos]`; 1x1 regions identity);
(2) 4-taps-per-line pack `new[g] = {T[4g+3],…,T[4g]}`. Dispatch geometry is
parsed from the DEPLOYED scheduler ROMs (no drift possible). Proofs on
every run: P0 (region tiling + all-17 fast-eligibility), P1 (permutation
bijectivity + full re-expansion), **P2 (4096-sample WALK-EQUIVALENCE: the
engine's fetched 3-bit lane weight at the transposed/fast address ==
original bank word at the LEGACY address)**, P3 (aligned-group tap-slice
identity), P4 (1x1 regions byte-identical). `_kp4.mem` files are
GITIGNORED — regenerate on promotion (proofs rerun; abort = no writes).

### Scheduler / maps / goldens
UNCHANGED. The scheduler still programs OLD-domain word bases (the engine
shifts internally; the AG's pos-major offset is internal); bias/scale maps
and act layouts untouched.

## GATES

(a) **Lint 0** — shared_engine @ ResNet legacy defaults AND @
    K_PAR=4/WGT_W=3/URAM_DATA_W=3072 (logs in
    `output/reports_integrated/kpar4rn/`).
(b) **Engine-ISO, WLAT=2 — A/B equivalence PASS**
    (`run_resnet_engine_iso_kpar.sh`): conv_246 (3x3 IC=256, the transposed
    walk), conv_250 (1x1 IC=512/OC=1024, chunk rotation), conv_284 (3x3
    IC=512 stride2, 2-chunk), vec0+vec1: KPAR4-build output bytes ==
    LEGACY-build output bytes (cmp on raw dumps), with the expected ~3.9x
    fast-walk cycle reduction (246: 453,938→115,250; 250: 409,642→108,586).
    NOTE: the per-case CONTRACT-golden comparison is INFORMATIONAL ONLY and
    mismatches IDENTICALLY in both builds — the intermediate contract
    goldens are STALE (2026-05-30, pre the 06-07 FIT-FIX scale.mem requant
    change; only the final relu_48 golden was refreshed 06-09). The
    authoritative byte-exact reference is the e2e gate (c). The stale-
    golden artifact was diagnosed, not papered over: identical mismatch
    counts/maps in both builds + identical [dbg]/[ACC8] dumps prove the
    KPAR4 datapath equals legacy bit-for-bit on real data.
(c) **e2e vec0 AND vec1: PASS 0/100352**, `e2e_cycles=7,229,214`
    (was 9,622,057 → **−24.9%**), `NN2RTL_VALUE_THREADS=1`,
    `NN2RTL_VALUE_XINIT=0`.
(d) **MBV2 inertness re-gate** (shared AG K_PAR>1 branch was touched):
    `run_mbv2_e2e_parallel.sh` → **8/8 PASS, mismatch 0,
    e2e_cycles == 2,620,778 EXACTLY** on all 8 vectors (the required
    cycle-exact bar: the eligibility change keeps every MBV2 dense
    dispatch's gate value, the pos-major formula degenerates to the legacy
    one at KH=KW=1, and the new ic-wrap branch is unreachable for 1x1).

## CYCLE ACCOUNTING

Engine MAC-walk total (pixels × oc_passes × K_TOTAL) ≈ 6.07M of the 9.62M
frame; P=4 removes ~3/4 of walk cycles ≈ −4.55M *of engine time*. The
frame shrinks −2.39M (not −4.55M) because post-OVERLAP the frame is
max(spatial, engine) per region: the engine (93.3% duty before) is no
longer the binding constraint in long stretches — the SPATIAL chain now
is. Follow-up levers are therefore spatial-side (per the MBV2 playbook:
wide-3 spatial parallelism / more dispatches to the now-cheap engine).

## AREA NOTES
* BRAM (banks): exactly neutral — same bits, width×4 / depth÷4
  (67072%4==0 ⇒ zero pad words). 384b×16768 lines per bank.
* DSP: mac lanes 256→1024 product DSPs (+768) — U250 DSP was 60-67% used;
  ResNet engine WGT_W=3 products are 8x3 → may pack 2/DSP; budget fine.
* LUT adds: 768b 4:1 tap0 subword mux, 3 extra 256:1 act byte selects,
  per-lane 4:1 add tree + mask muxes — small vs the ~250-380K relief
  already banked (K5+K1).

## PROMOTION CHECKLIST
1. Run `scripts/apply_resnet_kpar4.py` on the target tree (idempotent,
   anchors assert; needs the MBV2 KPAR4 base d76ed8f-class shared RTL).
2. Run `scripts/repack_resnet_kpar4_banks.py` (gitignored `_kp4.mem` must
   be regenerated in the target checkout; proofs P0-P4 rerun every time).
3. Re-gate ResNet e2e (vec0+vec1, expect PASS 0/100352 @ 7,229,214) and
   MBV2 8/8 (expect PASS @ 2,620,778 EXACTLY).
4. REGEN NOTE (per feedback_regen_must_rebuild_engine_maps): any future
   `generate_golden`/bank rebuild must re-run BOTH repack scripts
   (`repack_mbv2_kpar4_banks.py` AND `repack_resnet_kpar4_banks.py`) after
   `dedup_engine_banks_k5.py` — the _kp4 banks are derived artifacts.
   If the dedup layout ever changes, P0's tiling assert catches a stale
   dispatch table immediately.
5. Vivado: bank ROMs are now 384b×16768 inferred block-RAM (same
   `ram_style="block", cascade_height=8` attributes); confirm BRAM tile
   count in the next ResNet synth (bit-count is unchanged; wider/shallower
   shape usually packs the same or better at depth 16K).
6. Stale-contract-golden debt (pre-existing, NOT introduced here): the
   intermediate contract goldens under output/goldens/contracts are
   2026-05-30 vintage and no longer match the deployed scale.mem. Any
   future per-module ISO work should refresh them or use A/B gating as
   done here.
