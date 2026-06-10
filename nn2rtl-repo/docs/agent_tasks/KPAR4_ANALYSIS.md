# KPAR4 — MBV2 ENGINE K-PARALLEL P=4 (4 taps/cycle/lane)

Date: 2026-06-10 · Base: bff75bd (MBV2 engine stack: 47 dispatches, frame 4,811,270)
Scripts: `scripts/apply_mbv2_kpar4.py` (RTL, anchor-asserted/idempotent, `.prekp4` backups),
`scripts/repack_mbv2_kpar4_banks.py` (bank repack + layout proof),
`scripts/gen_mbv2_dense_engine_iso_cfg.py` + `scripts/run_mbv2_engine_iso_kpar.sh` (ISO gate).

## HEADLINE

| metric | before (bff75bd) | after KPAR4 | delta |
|---|---|---|---|
| e2e frame cycles (all 8 vecs identical) | 4,811,270 | **2,620,778** | **−2,190,492 (−45.5%)** |
| e2e byte-exactness | 8/8 PASS | **8/8 PASS, mismatch 0** | — |
| URAM weight-bank bits/bank | 18533×288 = 5,337,504 | 4634×1152 = 5,338,368 | +0.016% (3 pad words) |

The −2.19M beats the verdict's −1.4–2.1M estimate (project_mbv2_cycle_verdict_20260610).
SHIPPED: FULL dense P=4 (all 34 pointwise dispatches fast) + serial fallback for the
12 depthwise dispatches and the FC dispatch (see "What is fast vs serial").

## DESIGN

### Datapath (shared files, param-gated `K_PAR`, default 1)
* `output/rtl/engine/mac_array.v` — new param `K_PAR`; `generate if (K_PAR==1)`
  holds the ORIGINAL lane datapath VERBATIM. `K_PAR==4` branch: per lane,
  4 DSP product regs (`mul_q1_0..3`, `use_dsp`) + a **combinational** 4:1 adder
  tree into the same 32b accumulator. INT8xINT8 products into a 32b acc are
  exact (no rounding), so group order cannot change the sum. Pipeline SHAPE
  (stage-1 product regs, stage-2 gated accumulate) is unchanged ⇒ the
  skeleton's `ag_mac_done_d5` requant capture is UNCHANGED (TREE_STAGES=0 —
  the "derived drain depth d{5+TREE_STAGES}" from the plan collapses to d5).
  New fixed-width ports: `act_bytes_ext[23:0]` (taps 1..3 act bytes),
  `tap_mask[3:0]` (per-tap valid). A masked tap's act byte is ZEROED before
  the multiply ⇒ contribution exactly 0.
* `output/rtl/engine/address_generator.v` — new param `K_PAR` + output
  `k_tap_mask[3:0]`. The original walk is captured VERBATIM into the
  `K_PAR==1` generate branch (the apply script derives the `K_PAR>1` branch
  from that same captured text via asserted sub-edits). Fast eligibility
  (per-layer constants): `!depthwise && KH==KW==1 && IC%4==0 && IC>=4 &&
  weight_base%4==0`. Fast walk: `ic_cnt/k_cnt += 4`, `k_at_last` at
  `K_TOTAL-4`, mask = partial-group-capable `fast_mask` (always 4'b1111 on
  MBV2 since every pointwise IC % 4 == 0). Non-eligible layers run the
  SERIAL walk (step 1, mask 4'b0001) — proven cycle-IDENTICAL (ISO: DW 3,824,
  FC 5,164 cycles in both legacy and KPAR4 builds).
* `output/rtl/shared_engine_skeleton.v` — new param `K_PAR` (forwarded to
  both submodules). `K_PAR==4`: `weight_rd_addr` export becomes the GROUP
  address (`old>>2`); tap0 of the mac weight bus is subword-selected by the
  OLD address's `[1:0]` **piped 2 cycles** (mirrors `ag_weight_rd_en_d/_d2`,
  i.e. the WLAT=2 URAM alignment); taps 1..3 map straight from the wide
  line. Dense act bytes for taps 1..3 = consecutive ic bytes of the HELD
  act word (`ag_act_in_ic_byte_idx_d2 + 1..3`, 8-bit-wrap intermediates so a
  serial-mode idx=255 stays an in-range masked-dead select). The mask is
  piped d1/d2 alongside.

### Why one act read/cycle suffices (dense)
The act word is 2048b = 256 ic bytes/chunk. A fast group consumes 4
consecutive ic of ONE pixel (1x1 ⇒ single (kh,kw)=(0,0) position, pad 0 ⇒
always in bounds); groups are 4-aligned and 256%4==0 ⇒ all 4 bytes always
sit in the SAME word. IC>256 (e.g. FC-class) just rotates chunks 4× faster —
still 1 read/cycle. (Depthwise P>1 would need 4 *window positions* per lane
per cycle = 4 act reads ⇒ left serial, as sanctioned by the task.)

### Weight banks (MBV2-only files)
`uram_weight_bank` (defined inside `nn2rtl_top_engine.v`) gets `WORD_W`
(288→1152), `DEPTH` 18533→4634 (=ceil/4, 3 zero pad words), `ADDR_W` 15→13;
XPM `READ_DATA_WIDTH_A`/`MEMORY_SIZE` follow `WORD_W`. URAM bit-count is
neutral (width×4, depth÷4). Bus: `engine_weight_rd_data` 2048→8192b; tap j
word = concat over banks of `bank[j*288 +: 256]` (lane order per tap is
IDENTICAL to the old bus). Repack layout + PROOF (P1 full re-expansion, P2
4096-sample byte cross-check, P3 aligned-group tap-slice identity) live in
`scripts/repack_mbv2_kpar4_banks.py`; output `uram_weights_bank{0..7}_kp4.mem`
(gitignored, REGENERATE on promotion).

### What is fast vs serial under K_PAR=4
* FAST (4 taps/cycle): all 34 dense pointwise dispatches — every base in
  `weight_base_word_rom` is %4==0 and every IC ∈ {16..1280} is %4==0.
* SERIAL (1 tap/cycle, bit- and cycle-identical to before): 12 depthwise
  dispatches (K_TOTAL=9, per-lane act) and the FC dispatch 46 (node_linear,
  base 13413 % 4 == 1 → subword select; costs only ~3.8K cycles/frame, so
  realigning the FC region was deliberately NOT done — zero scheduler/map
  churn). Follow-up (optional): pad FC base to 13416 to make it fast.

### Scheduler / maps / goldens
UNCHANGED. The scheduler still programs OLD-domain word bases (the engine
shifts internally), bias/scale maps untouched, goldens untouched (the
change is byte-exact by construction).

## GATES (all GREEN)

(a) **Lint 0**: `verilator --lint-only` clean for shared_engine @K_PAR=1
    (ResNet defaults), @K_PAR=4/WGT_W=8/URAM_DATA_W=8192, and the ISO
    wrapper in both builds (`output/mobilenet-v2/reports/kpar4/lint_*.log`;
    `-Wno-PINMISSING` is pre-existing: the iso wrapper never connected the
    backpressure-era `out_ready`).
(b) **Engine-ISO, WLAT=2, mismatch=0** (`run_mbv2_engine_iso_kpar.sh`, logs
    in `output/mobilenet-v2/reports/kpar4/iso_*.log`):
    KPAR4 build × vec0+vec1: node_conv_816 (dense fast, IC=16, 1,204,224
    bytes), node_conv_898 (dense fast, IC=960, 4-chunk rotation),
    node_conv_896 (depthwise serial fallback), node_linear (FC serial
    fallback exercising subsel=1,2,3). LEGACY build (K_PAR=1 + original
    banks) regression: 816/896/linear vec0 mismatch=0.
    Cycle evidence: 898 ≈3.85× faster; DW/FC cycle-IDENTICAL across builds.
(c) **e2e 8/8 PASS**, mismatch 0, `e2e_cycles=2,620,778` on ALL 8 vectors
    (baseline re-measured in this worktree first: 8/8 PASS @4,811,270).
(d) **Hazard checker PASS**: `check_mbv2_act_region_hazards_fc.py` — the
    proof is structural (region disjointness + the "only dispatch d+1's
    loader fills while d runs" ordering invariant, which is enforced by
    scheduler handshakes, not durations), so faster dispatch runs cannot
    invalidate it; re-run confirms PART A/B + C1..C5 hold on the patched top.
(e) **ResNet INERTNESS**:
    1. *Textual*: every shared-file change is a parameter default (K_PAR=1)
       plus `generate if (K_PAR==1)` branches whose bodies are the
       pre-change text VERBATIM (the apply script literally re-emits the
       captured original block for the AG); new mac/AG ports are tied to
       constants and unused in the legacy branches; `K_PAR*...` widths
       evaluate to the original widths at K_PAR=1. No other instantiation
       in the repo passes K_PAR (grep-verified; ResNet tops instantiate
       `shared_engine #(...)` without it).
    2. *Mechanical*: Verilator-generated C++ for `shared_engine` at default
       params, pre-patch (.prekp4) vs post-patch, differs ONLY in (i)
       generate-scope names (`g_p1.`, `g_walk_legacy.` etc.) and (ii) the
       scope-hash-derived `VL_SCOPED_RAND_RESET` seeds for uninitialized
       regs — 0 non-RAND_RESET diff lines. Gates run `--x-initial 0` and
       hardware is power-on-0, so the seeds are simulation-irrelevant.
    User re-gates ResNet e2e on promotion as planned.

## AREA NOTES
* URAM: neutral (same bits/bank ±3 pad words).
* DSP: mac lanes 256→1024 product DSPs (+768) — well within U250 budget.
* LUT adds (K_PAR=4 elaboration only): 2048b 4:1 tap0 subword mux, 3 extra
  256:1 byte selects on the held act word, per-lane 4-input add + 4 mask
  muxes. The serial→P4 conversion adds no new big memories or pipes.

## PROMOTION CHECKLIST
1. Apply commits (or run `scripts/apply_mbv2_kpar4.py` on a clean bff75bd-
   class tree — it is anchor-asserted + idempotent, `.prekp4` backups).
2. Run `scripts/repack_mbv2_kpar4_banks.py` — the `_kp4.mem` banks are
   GITIGNORED and must be regenerated in the target checkout (proofs run
   on every invocation; abort = no partial writes).
3. Re-gate ResNet (e2e byte-exact AND cycle-exact expected; see (e)).
4. NOTE: worktree commit e1293fd ("absolute→repo-root-relative $readmemh")
   is an ENVIRONMENT isolation fix for the agent worktree (the main
   checkout's spatial-module weight paths are absolute into the main tree
   and were being mutated by a concurrent agent). Keep or drop per the main
   checkout's path convention — it is orthogonal to KPAR4.
5. Vivado: XPM URAM word width 1152 (16×72b) per bank — same URAM count;
   confirm in the next MBV2 synth wave.
6. Optional follow-ups: pad FC base 13413→13416 (+3 zero words + scheduler
   row 46 + gen_dw_engine_iso_cfg linear row) to make FC fast (~3.8K cyc);
   DW-on-engine P>1 would need a 4-position act window — not worth it.
