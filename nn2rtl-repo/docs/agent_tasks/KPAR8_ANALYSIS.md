# KPAR8 — MBV2 ENGINE K-PARALLEL P=8 (8 taps/cycle/lane) + FC-PAD rider

Date: 2026-06-10 · Base: 44294e9 (int4-imagenet-gptq tip; MBV2 KPAR4 frame
2,620,778, ResNet KPAR4+MP32 frame 5,664,715)
Scripts: `scripts/apply_mbv2_kpar8.py` (RTL+scheduler+cfg-gen, anchor-asserted/
idempotent, `.prekp8` backups), `scripts/repack_mbv2_kpar8_banks.py` (bank
repack + FC relocation + layout proofs P0..P4),
`scripts/run_mbv2_engine_iso_kpar8.sh` (ISO gate).

## HEADLINE

| metric | before (KPAR4) | after KPAR8+FC-PAD | delta |
|---|---|---|---|
| e2e frame cycles (all 8 vecs identical) | 2,620,778 | **2,264,013** | **−356,765 (−13.6%)** |
| e2e byte-exactness | 8/8 PASS | **8/8 PASS, mismatch 0** | — |
| ResNet inertness (K_PAR=4) | 5,664,715 / 0 mism | **5,664,715 / 0 mism (EXACT)** | 0 |
| URAM weight-bank bits/bank | 4634×1152 = 5,338,368 | 2317×2304 = 5,338,368 | **0** (identical) |

Attribution predicted −357,504 (P8 dense) − ~4,482 (FC rider) ≈ 2,258,792;
measured 2,264,013 = 99.77% of the predicted saving (the small shortfall is
overlap shadowing: parts of some now-faster engine runs hide behind spatial
work that the static attribution counted as exposed).

## DESIGN

### Param space {1, 4, 8} — branch insertion, not branch mutation
The K_PAR=4 implementation (commit lineage c5c68e5/a1f088a) is LOAD-BEARING
for ResNet (its top sets K_PAR=4). KPAR8 therefore INSERTS a new
`else if (K_PAR == 8)` generate branch between the legacy branch and the
existing K_PAR>1 branch in all three shared files:

    if (K_PAR == 1)      : g_p1 / g_walk_legacy / g_waddr_legacy / g_ktap_legacy   (VERBATIM)
    else if (K_PAR == 8) : g_p8 / g_walk_kpar8  / g_waddr_kpar8  / g_ktap_kpar8    (NEW)
    else                 : g_p4 / g_walk_kpar   / g_waddr_kpar   / g_ktap_kpar     (VERBATIM)

so the K_PAR==1 and K_PAR==4 elaborations keep their original text AND scope
names. The only shared-file lines REMOVED are the fixed-width declarations of
`act_bytes_ext` / `tap_mask` / `k_tap_mask` (+ their comments), replaced by
max(K_PAR,4)-based width expressions `(((K_PAR > 4) ? K_PAR : 4))` that
evaluate to the ORIGINAL widths (24b / 4b) at K_PAR<=4 and widen (56b / 8b)
only at K_PAR=8 (diff-audited: `git diff | grep '^-'` shows exactly those 7
declaration lines across mac_array/skeleton + 2 in address_generator).

### Datapath (shared files)
* `output/rtl/engine/mac_array.v` — `g_p8`: per lane, 8 DSP product regs
  (`mul_q1_0..7`, `use_dsp`) + a **combinational** 8:1 adder tree into the
  same 32b accumulator. INT8xINT8 -> 32b accumulation is exact (no
  rounding), so group order cannot change the sum. Pipeline SHAPE (stage-1
  product regs, stage-2 gated accumulate) is unchanged ⇒ the skeleton's
  `ag_mac_done_d5` requant capture is UNCHANGED (TREE_STAGES=0). Masked
  taps multiply a ZEROED act byte (contribution exactly 0).
* `output/rtl/engine/address_generator.v` — `g_walk_kpar8` is derived from
  the verbatim K_PAR==4 walk text by 10 asserted sub-edits (the apply
  script captures the kp4 branch and transforms a COPY): eligibility
  `IC%8==0 && IC>=8 && weight_base%8==0` (`cfg_ic[2:0]`,
  `cfg_weight_uram_base[2:0]`), `k_at_last` at `K_TOTAL-8`, 8-bit
  `fast_mask` (partial-group capable; always 8'hFF on MBV2), walk step 8.
  Depthwise (cfg_depthwise=1) keeps the SERIAL walk (step 1, mask
  8'b0000_0001) — proven cycle-IDENTICAL (ISO: DW 896 = 3,824 cycles in
  KPAR4 and KPAR8 builds).
* `output/rtl/shared_engine_skeleton.v` — `g_waddr_kpar8`: weight_rd_addr
  export = GROUP address (`old>>3`). `g_ktap_kpar8`: tap0 subword-selected
  by the OLD address's `[2:0]` **piped 2 cycles** (the WLAT=2 URAM
  alignment); taps 1..7 map straight from the 16384b line; 7 extra act-byte
  selects `ag_act_in_ic_byte_idx_d2 + 1..7` (8-bit-wrap intermediates) off
  the HELD act word.

### Why one act read/cycle still suffices (dense)
A fast group consumes 8 consecutive ic of ONE pixel (1x1, pad 0). Groups
are 8-aligned and the act word is 256 ic bytes with 256%8==0 ⇒ all 8 bytes
always sit in the SAME 2048b word. IC>256 (898: IC=960 = 4 chunks; FC:
IC=1280 = 5 chunks) rotates chunks 8× faster — still 1 read/cycle.
(Depthwise P>1 would need 8 *window positions*/lane/cycle ⇒ left serial.)

### Weight-bus geometry — DECISION: wide 16384b bus (not 2-read prefetch)
`uram_weight_bank` WORD_W 1152→2304 (= 32×72b), DEPTH 4634→2317, ADDR_W
13→12; XPM `READ_DATA_WIDTH_A`/`MEMORY_SIZE` follow WORD_W. Verified
scaling: bits/bank IDENTICAL to KPAR4 (5,338,368); URAM count UNCHANGED at
32/bank = 256 total — KPAR4 packed them 16-wide × 2-deep-cascade (4634 >
4096), KPAR8 packs 32-wide × 1 (2317 ≤ 4096), so P8 actually REMOVES the
depth cascade. The cost is wiring: the live bank read output doubles
(8×1152 = 9,216 → 8×2304 = 18,432 bits) and the engine weight bus becomes
16384b (8 tap-major 2048b words), every bit of which terminates in a DSP
input mux. This is structurally reasonable (same memory count, pure
width), so the alternative — 2 reads/cycle of 1152b + a 1-deep weight
prefetch — was REJECTED: it would change the weight-read timing shape
(the WLAT=2 alignment and the `~mac_done` read-gating are both
load-bearing, see the 2026-05-24/05-28 bug archaeology), for zero URAM
saving. depth pad: 18533+3 = 18536 = 8×2317 EXACTLY (0 tail pad; the 3 FC
pad words below are the only filler).

### [FC-PAD] rider — FC base 13413 → 13416
13413 % 8 == 5 ⇒ serial at P8. Padded to 13416 (%8==0):
* `nn2rtl_scheduler.v` row 46 `weight_base_word_rom` 13413 → 13416 (the
  ONLY scheduler change; bias/scale base 87 unchanged — independent ROMs).
* bank images: repack inserts 3 ZERO words at 13413..13415 and relocates
  the FC region (old 13413..18532 → new 13416..18535). Dense + DW regions
  (< 13413) are untouched; every other dispatch base is unchanged.
* `gen_dw_engine_iso_cfg.py` linear row now PARSES the scheduler row 46
  (no more hardcoded 13413 — cannot drift).
* eligibility now passes for dispatch 46: IC=1280%8==0, base 13416%8==0,
  K_TOTAL=1280 ⇒ 160 groups × 4 oc_passes ⇒ ISO **684 cycles vs 5,164
  serial** (−4,480/frame ≈ the predicted −4,482).

### What is fast vs serial under K_PAR=8
* FAST (8 taps/cycle): all 34 dense pointwise dispatches (P0 proof: every
  `weight_base_word_rom` dense base %8==0 — bases 0,32,48,144,168,312,336,
  480,512,704,736,928,960,1152,1280,1664,1792,2176,2304,2688,2816,3200,
  3488,4064,4352,4928,5216,5792,6432,7392,8032,8992,9632,11552; every IC ∈
  {16..1280} %8==0) **+ the FC dispatch 46 (post-pad)**.
* SERIAL (1 tap/cycle, bit- and cycle-identical): the 12 depthwise
  dispatches (K_TOTAL=9, per-lane act; any base alignment via the 3-bit
  subword select).

### Scheduler / maps / goldens
Bias/scale maps and ALL goldens untouched (byte-exact by construction).
Scheduler: ONLY row 46's weight base (FC-PAD).

## GATES

(a) **Lint 0**: `verilator --lint-only` — 0 errors AND 0 warnings for the
    iso wrapper at K_PAR=1 (no define), K_PAR=4 (-DKPAR4), K_PAR=8
    (-DKPAR8), and `shared_engine` at ResNet-class defaults
    (`output/mobilenet-v2/reports/kpar8/lint_*.log`).
(b) **Engine-ISO, WLAT=2, mismatch=0** (`run_mbv2_engine_iso_kpar8.sh`,
    logs `output/mobilenet-v2/reports/kpar8/iso_*.log`), KPAR8 build ×
    vec0+vec1:
    * node_conv_816 — dense fast, IC=16 (2 groups/pass-pixel), 1,204,224
      bytes, mismatch=0.
    * node_conv_898 — dense fast, IC=960, 4-chunk act rotation, mismatch=0,
      **6,470 cycles** (KPAR4 measured ~3.85× over serial; P8 ≈ 2× over P4).
    * node_linear — FC post-pad FAST, subsel=0, **684 cycles** (serial was
      5,164), mismatch=0 vs the same integer FC golden the e2e compares.
    * node_conv_896 — depthwise SERIAL fallback, **3,824 cycles =
      cycle-IDENTICAL** to the KPAR4 build (kp4 reference build re-run in
      the same gate) and to the documented KPAR4/legacy value.
(c) **MBV2 e2e 8/8 PASS**, mismatch 0, `e2e_cycles=2,264,013` on ALL 8
    vectors (`scripts/run_mbv2_e2e_parallel.sh`; baseline re-measured in
    this worktree FIRST: 8/8 PASS @2,620,778 — environment validated
    before any change). Logs: `output/mobilenet-v2/reports/e2e_par/`.
(d) **ResNet inertness e2e PASS** (vec0, `NN2RTL_VALUE_THREADS=1
    NN2RTL_VALUE_XINIT=0 npx tsx scripts/run_nn2rtl_top_value.ts 0`):
    result=PASS, mismatch_bytes=0/100352, `e2e_cycles=5,664,715` —
    EXACTLY the pre-change frame, so the ResNet top's K_PAR=4 elaboration
    is bit- AND cycle-identical (its `g_p4`/`g_walk_kpar`/`g_waddr_kpar`/
    `g_ktap_kpar` branch bodies are textually verbatim; the only shared-
    file text it re-elaborates differently is the max(K_PAR,4) width
    expressions, which evaluate to the original 24b/4b at K_PAR=4).

## AREA / FMAX-RISK NOTES
* URAM: bit- and count-NEUTRAL vs KPAR4 (256 URAM288; cascade removed).
* DSP: mac lanes 1024 → 2048 product DSPs (+1024; U250 has 12,288 — fine).
* **Fmax risk #1 — stage-2 accumulate**: now a 9-operand combinational sum
  (acc + 8 sign-extended 16b products) into the 32b acc register. Vivado
  ternary-adder mapping ⇒ ~3 carry-propagate levels vs ~2 at P4: the
  deepest combinational path in the engine grew by roughly one 32b adder
  level. CYCLES-FIRST per the task; if a later synth wave flags this path,
  the mitigation is registering the 8:1 tree (TREE_STAGES=1), which moves
  the requant capture d5→d6 — a cycle-shape change that needs its own
  byte-exact re-gate (one extra cycle per (pixel × oc_pass) drain).
* Fmax risk #2 (minor): tap0 subword mux is now 8:1 × 2048b (was 4:1) on
  the URAM output before the DSP input regs (~1 extra LUT level), and the
  held-act-word byte select count grows 3 → 7 instances of 256:1×8b.
* Wiring: live bank read width 9,216 → 18,432 bits + 16384b weight bus —
  watch placement congestion around the engine column in the next MBV2
  synth wave (same URAMs, double the output net count).

## PROMOTION CHECKLIST
1. Run `scripts/apply_mbv2_kpar8.py` on a 44294e9-class tree (anchor-
   asserted + idempotent; `.prekp8` backups). It requires the KPAR4 state
   to be present (its anchors are the KPAR4 lines).
2. Run `scripts/repack_mbv2_kpar8_banks.py` — the `_kp8.mem` banks are
   GITIGNORED and must be regenerated in the target checkout (P0..P4
   proofs run on every invocation; abort = no partial writes). P0 reads
   the scheduler, so run AFTER the apply script (it warns if row 46 is
   still 13413).
3. Re-gate ResNet e2e (K_PAR=4 — must stay byte-exact AND cycle-exact at
   5,664,715) + MBV2 8/8.
4. The `_kp4.mem` banks are now UNUSED by MBV2 (still used by nothing else;
   ResNet has its own `_kp4r` 384b banks under output/weights/). Keep until
   the next cleanup wave in case of K_PAR=4 fallback experiments — but note
   post-FC-PAD the kp4 banks' FC region no longer matches the scheduler.
5. Vivado: XPM URAM word width 2304 (32×72b) per bank — same URAM count;
   confirm in the next MBV2 synth wave (plus Fmax notes above).
