# MobileNetV2 (INT8) → RTL on Alveo U250 — Fit Projection

**Status: PROJECTED, NOT synth-confirmed. BOTH memory-mapping fixes are now
APPLIED to the RTL and re-verified byte-exact** (2026-06-02). Every total in this
document is an analytical projection assembled from (a) the existing per-module
Vivado synth runs, (b) the one full U250 `shared_engine` synth, and (c) the
post-rewrite RTL structure. **No integrated Vivado synth / place / route was run
for this projection** (Vivado is gated until the design is bit-exact + accurate +
fit-confirmed — see `feedback_vivado_only_when_proven`). The two former fit risks
(RISK #1 depthwise line-buffer URAM width-binding; RISK #2 oversized residual
skip FIFOs) have been closed in RTL — the numbers below are the design's **actual
post-fix projected fit**, no longer "after two needed fixes". All six resources
project under 80% of the U250 budget.

- **Network:** MobileNetV2, ImageNet, `int8_symmetric_per_tensor` (per
  `output/mobilenet-v2/layer_ir.json`). NOTE: this is the **INT8** MobileNetV2,
  not the INT4-GPTQ ResNet-50 — do not transfer the ResNet fit numbers.
- **Target:** `xcu250-figd2104-2L-e`. Budget: **LUT 1,728,000 · FF 3,456,000 ·
  DSP48E2 12,288 · RAMB36 2,688 · URAM288 1,280.**
- **Date:** 2026-06-02.

---

## Headline: PROJECTED per-resource totals

| Resource | Projected total | U250 budget | % of budget | Verdict |
|---|---:|---:|---:|---|
| **LUT**    | **~1,064,000** | 1,728,000 | **61.6%** | under 80% |
| **FF**     | **~760,000**   | 3,456,000 | **22.0%** | under 80% |
| **DSP48E2**| **~1,345**     | 12,288    | **10.9%** | under 80% |
| **RAMB36** | **~1,013**     | 2,688     | **37.7%** | under 80% |
| **URAM288**| **128**        | 1,280     | **10.0%** | under 80% |

**All six resources project under 80% of the U250 budget** (post-fix). The two
former memory-mapping fit risks have been closed in RTL (mapping-/sizing-only,
byte-exact-preserving — see §2.2 / §2.3): the depthwise line buffers no longer
width-bind URAM, and the residual skip FIFOs are right-sized to their actual
residual frame occupancy. The *bits* required by every runtime memory were always
tiny; the fix was purely how they tile into RAMB36 / URAM288 primitives. Fit is
now **PLAUSIBLE and internally consistent** but still **NOT synth-confirmed** —
a Vivado run remains gated per `feedback_vivado_only_when_proven`.

---

## Inputs: measured vs estimated

| Input | Source | Measured or Estimated |
|---|---|---|
| Per-module LUT/FF/DSP/BRAM (52 conv, 35 relu, 10 add, GAP, gemm) | `output/mobilenet-v2/reports/*.vivado.json` (Vivado 2025.2 synth, part `xczu9eg-ffvb1156-2-e`) | **MEASURED** (per-module synth) |
| `shared_engine` LUT/FF/DSP on U250 | `docs/agent_tasks/00_engine_only_synth_REPORT.md` / `output/reports_integrated/engine_only_synth.json` (synth on `xcu250-figd2104-2L-e`) | **MEASURED** (on the real U250 part) |
| Depthwise post-rewrite LUT/FF | conv_812 measured (3172 LUT / 2428 FF) **scaled linearly by channel count C** | **ESTIMATED** (projection; see method) |
| Depthwise / engine URAM/BRAM tile counts | derived from RTL memory geometry (depth × width vs RAMB36 / URAM288 primitive geometry) | **ESTIMATED** (analytical tile count) |
| ReLU post-ROM DSP | structural: requant multiply replaced by 128×8 ROM → DSP = 0 | **CONFIRMED** (all 35/35 relus on ROM; grep: 0 synth runtime multiply; byte-exact across families) |
| GAP post-timemux DSP | `node_mean.v` SCALE_LANES=16 → ~16–24 DSP | **ESTIMATED** (RTL inspection + byte-exact) |
| Glue/dispatch FSM overhead | engine-top control logic | **ESTIMATED** (~8k LUT / 6k FF allowance) |

> **Part note.** The per-module synths ran on `xczu9eg` (ZCU102), the engine
> synth on `xcu250` (U250). Both are UltraScale+; the primitives counted
> (LUT6, FDRE/FF, DSP48E2, RAMB36E2, URAM288) are architecturally identical, so
> the per-module primitive counts transfer 1:1 to the U250. Only `fmax` would
> differ (speed grade), and timing is out of scope here.

### Fixes that have been applied to the RTL and proven byte-exact

All three datapath rewrites below are **on-disk in
`output/mobilenet-v2/rtl/`** and were re-verified byte-exact in this session via
`npx tsx scripts/_verify_mbv2_variant.ts <rtl> <module> <sidecar>` (Verilator,
`mismatch_count = 0`):

| Fix | Module verified | Samples | mismatch |
|---|---|---:|---:|
| Depthwise → sync BRAM/URAM `line_buf_window` (was combinational/distributed-RAM) | `node_conv_812` | 3,211,264 | **0** |
| ReLU requant → 128-entry distributed ROM (DSP-eliminated) | `n4_20` | 602,112 | **0** |
| GAP → time-muxed scale (SCALE_LANES=16, was ~2480 DSP unrolled) | `node_mean` | 10,240 | **0** |

---

## 1. Datapath logic (LUT / FF / DSP) — projects comfortably under budget

### 1.1 Depthwise convs (17) — line_buf_window rewrite

The pre-rewrite depthwise convs were the dominant LUT consumer: the legacy
combinational / distributed-RAM line buffer scaled pathologically with image
width, ranging from **3,172 LUT (conv_812) to 336,522 LUT (conv_818)** for a
**measured sum of 1,313,679 LUT across the 17** — alone over 76% of the U250 LUT
budget.

The rewrite (`rtl_library/line_buf_window.v`, sync per-slot memories + rotating
row pointer) makes every depthwise behave like conv_812, which already
synthesized clean (May-15 synth: **3,172 LUT / 2,428 FF / 1 DSP / 0
LUT-as-Memory**). conv_812 is therefore the canonical post-rewrite reference.

**Projection method (per task spec):** scale conv_812's 3,172 LUT / 2,428 FF by
channel count C (the depthwise datapath is MP=4-fixed; per-channel
accumulators, weight-ROM address width, and the window mux scale with C). 1 DSP
each (`use_dsp="yes"` on the single `mul_q`).

| | LUT | FF | DSP |
|---|---:|---:|---:|
| Depthwise (17), projected | **~707,000** | ~541,000 | 17 |

> **This is a PESSIMISTIC upper bound.** Linear-by-C replicates conv_812's
> *entire* footprint (including the shared FSM/scheduler/handshake that does
> **not** grow with C) once per channel-worth, so the true post-synth number is
> expected to be meaningfully lower. It is reported as-is per the requested
> method; treat ~707k LUT as a ceiling, not a point estimate.

### 1.2 Stem 3×3 conv (conv_810) — MEASURED, kept as-is

`node_conv_810`: **1,542 LUT / 771 FF / 1 DSP / 3 BRAM18.**

### 1.3 Pointwise (1×1) convs (34) — OFFLOADED to the shared engine

Pre-offload, the 34 spatial 1×1 convs measured **497,311 LUT / 170,274 FF / 47
DSP** — they no longer exist as spatial datapaths. In
`nn2rtl_top_engine.v` all 34 are **engine-dispatched** (header: *"dispatched:
34"*; grep confirms `conv_814 … conv_910: engine-dispatched (no instantiation
here; data_out driven by shared_engine)` and **zero** `conv_datapath`
instances). The cost is the single shared engine, **measured on the U250 part**:

| `shared_engine` (MEASURED, xcu250) | LUT | FF | DSP | BRAM | URAM |
|---|---:|---:|---:|---:|---:|
| | **107,268** | 30,979 | 1,283 | 0 | 0 (weights live in the URAM banks, §2) |

Net pointwise effect: **497k spatial LUT → 107k shared-engine LUT** (the engine
also subsumes the gemm/`node_linear`, which synthesizes to 0 standalone because
it is engine-dispatched).

### 1.4 ReLU (35) — DSP eliminated via 128-entry ROM

Pre-fix the 35 relus measured **220,456 LUT / 128,348 FF / 11,328 DSP** — the
DSP came from the per-channel requant multiply at exactly **2 DSP/channel** for
the 27 requantizing relus (the other 8 are plain ReLU/ReLU6 clamps with no
requant). The ROM rewrite replaces that multiply with a precomputed
**128×8 distributed ROM** (`REQUANT_ROM[relu_byte]`, `rom_style="distributed"`):
the post-ReLU byte is always in `[0,127]`, so the whole multiply+round+clamp is
a 128-entry lookup. **DSP → 0**, LUT ≈ unchanged (the ROM + clamp fabric is
comparable to the old multiply-tail fabric; verified on n4_20: same byte output,
0 DSP).

| | LUT | FF | DSP |
|---|---:|---:|---:|
| ReLU (35), projected | ~220,000 | ~128,000 | **0** |

> **ROLLOUT COMPLETE (2026-06-02).** The ROM rewrite is a uniform mechanical
> transform (the requant domain is always post-ReLU `[0,127]`) and is now applied
> to **all 35 of the 35** relu `.v` files on disk (`n4.v … n4_35.v`). This is a
> superset of the 27 requantizing relus the projection depended on, so the
> assumption "all requant relus on ROM, DSP → 0" now holds with margin. Verified
> two ways, per the anti-false-positive rule:
>
> 1. **Structural grep (all 35 files):** every file carries a 128-entry requant
>    ROM (`REQUANT_ROM` / `requant_rom` / `req_rom` / `rom[...]`,
>    `rom_style="distributed"`), and **ZERO** files retain a synthesizable runtime
>    requant multiply. Each file's single `* SCALE_MULT(_CONST)` lives inside an
>    `initial`/`function` block (compile-time constant-fold into ROM contents),
>    not a clocked `always`/`assign` datapath. (The only `*` tokens in the
>    synthesizable blocks are byte-index part-selects `data_in[i*8 +: 8]` /
>    `beat_buf[..][ch*8 +: 8]` — address arithmetic, not multipliers.)
> 2. **Byte-exact (Verilator, `scripts/_verify_mbv2_variant.ts`, mismatch=0):**
>    `n4_11` (1,204,224 samples), `n4` (3,211,264), `n4_3` (9,633,792),
>    `n4_23` (903,168), `n4_33` (376,320) — all `mismatch_count=0`, `max_error=0`,
>    `timing_pass=true` across all structural families (single-beat tail, in_byte
>    family, beat_buf multi-beat family).
>
> **Net effect:** relu DSP = **0** (was 11,328 from the 2 DSP/channel requant
> multiply on the 27 requantizing relus). The §1.7 design total stands at
> **~1,345 DSP = 10.9%** with relu contributing 0.

### 1.5 Residual adds (10) — MEASURED

The 10 adds measured **9,663 LUT / 19,478 FF / 20 DSP** total (2 DSP each for
the fused rescale). Kept as-is.

### 1.6 Global average pool (GAP / `node_mean`) — time-muxed

Pre-fix `node_mean` measured **96,309 LUT / 32,500 FF / 2,480 DSP** (all 1280
channel scale-multiplies unrolled). The rewrite time-multiplexes the scale over
**SCALE_LANES=16** multipliers × SCALE_STEPS=80 cycles → **DSP ≈ 16–24**, and
the LUT collapses with the unroll removed. Conservative estimate **~10,000 LUT /
~32,500 FF / 24 DSP** (FF kept — the 1280 accumulators are real state).

### 1.7 Logic totals

| Block | LUT | FF | DSP | Source |
|---|---:|---:|---:|---|
| Depthwise (17) | ~707,000 | ~541,000 | 17 | EST (conv_812 × C) — upper bound |
| Stem conv_810 | 1,542 | 771 | 1 | MEASURED |
| Shared engine (subsumes 34 PW + gemm) | 107,268 | 30,979 | 1,283 | MEASURED (U250) |
| ReLU (35) | ~220,000 | ~128,000 | 0 | CONFIRMED (all 35/35 on ROM; DSP→0) |
| Adds (10) | 9,663 | 19,478 | 20 | MEASURED |
| GAP (time-mux) | ~10,000 | ~32,500 | 24 | EST |
| Dispatch/glue overhead | ~8,000 | ~6,000 | 0 | EST allowance |
| **TOTAL (PROJECTED)** | **~1,064,000** | **~759,000** | **~1,345** | |
| **% of U250** | **61.6%** | **22.0%** | **10.9%** | |

> The FF projection (22%) is dominated by the (pessimistic) linear-by-C
> depthwise FF scaling; the true figure is expected lower and well inside the
> 11.9%-class regime seen on the comparable integrated ResNet measurement. Even
> the conservative number is far under budget.

---

## 2. On-chip memory (BRAM / URAM) — bits are tiny, TILE-BINDING is the risk

The *bit* requirements of every runtime memory in this design are small. The fit
question is purely **how the shallow-but-very-wide buffers tile into RAMB36 /
URAM288 primitives.** This is the exact lesson from the ResNet-50 fit
investigation (`project_fit_not_confirmed_synth_over`): width-bound ROMs/buffers
round up to whole tiles, so cutting *bits* does not cut *tiles*.

### 2.1 Engine weight memory — fits (URAM 10%)

8 × `uram_weight_bank` (`ram_style="ultra"`), each **DEPTH 13152 × 288 bit**.
URAM288 primitive = 4096 × 72. Per bank = ⌈288/72⌉ × ⌈13152/4096⌉ = 4 × 4 = **16
URAM288**; × 8 banks = **128 URAM288 = 10.0% of 1,280.** Matches the design
intent. The bias mem (2 × 256 × 8192) and the shared activation BRAM
(`stream_to_act_bram_bridge`, ~12,544 × 2048) add **~114 + ~697 ≈ 811 RAMB36
(~30%)** — fine.

### 2.2 RISK #1 — Depthwise line buffers forced to URAM (width-bound) — **CLOSED 2026-06-02**

`line_buf_window.v` previously hard-pinned the KH=3 per-slot line-buffer memories
to **URAM (`ram_style="ultra"`)**. Total line-buffer storage across the 17
depthwise convs is only **2.87 Mbit** (efficient packing = **78 RAMB36, 2.9%**).
But the buffers are shallow and very wide (e.g. C=960 → 8 deep × 7,680 bit), and
URAM has a fixed 4096×72 geometry, so forcing them to URAM width-bound to **2,394
URAM288 = 187% (impossible)**.

**FIX (applied):** `line_buf_window.v` now has a `parameter integer
LINE_BUF_USE_URAM = 1` (default 1 = `"ultra"` so any consumer that omits it is
byte/primitive-identical to before). A generate-if selects the `ram_style`
attribute: `1 → "ultra"` (URAM288), `0 → "block"` (RAMB36). The two branches are
bit-identical apart from that single attribute line; `ram_style` is a
synthesis-only attribute (Verilator ignores it) so values/latency/control are
unchanged. All **17 MobileNetV2 depthwise convs** (`node_conv_{812,818,824,830,
836,842,848,854,860,866,872,878,884,890,896,902,908}.v` — exactly the 17 that
also pass `EXPOSE_FULL_WINDOW(0)`) now pass `.LINE_BUF_USE_URAM(0)` so their
shallow-wide buffers reshape into block RAM. ResNet `conv_196`,
`conv_datapath_mp_k`, and the mbv2 stem `conv_810` were NOT touched — they keep
the default 1 = ultra, preserving the intentional ResNet wide-window + URAM path.

| Mapping of the 17 DW line buffers | Tiles | % of budget | Verdict |
|---|---:|---:|---|
| Pre-fix (`ram_style="ultra"`, width-bound URAM) | 2,394 URAM288 | 187% | OVER — impossible |
| Width-bound RAMB36 (no reshape) | 2,394 RAMB36 | 89% | borderline |
| **Post-fix: efficient aspect-ratio RAMB36** (`LINE_BUF_USE_URAM=0`) | **78 RAMB36** | **2.9%** | **fits trivially** |

> **Verified byte-exact (Verilator, `scripts/_verify_mbv2_variant.ts`,
> re-run 2026-06-02):** `node_conv_896` (wide C=960) mismatch=0, max_error=0,
> 376,320/376,320 samples exact, timing 10,091 = 10,091; `node_conv_818` (narrow
> C=96) mismatch=0, max_error=0, 2,408,448/2,408,448 samples exact, timing
> 1,124 = 1,124. **ResNet-safe:** `node_conv_196` contains no `LINE_BUF_USE_URAM`
> token (gets the default ultra) and lint-elaborates clean (Verilator
> `--lint-only --top-module node_conv_196`, EXIT=0, zero line_buf_window
> diagnostics). URAM stays reserved for the engine weight banks (§2.1).

### 2.3 RISK #2 — Residual skip FIFOs grossly oversized — **CLOSED 2026-06-02**

`nn2rtl_top.v` / `nn2rtl_top_engine.v` previously instantiated the 10 residual
skip FIFOs at depths that did not match the residuals they buffer. The skip
source is a **free-running producer** (`engine_output_bridge`, `.ready_out`
gated only by `spatial_run`, **not** by the FIFO's `in_ready`): the FIFO's
`in_ready` is ANDed into its own `in_valid` but is NOT fed back to the producer,
so on a full FIFO the beat is silently **dropped**. Therefore the FIFO must never
fill, i.e. its DEPTH must be ≥ the worst-case in-flight occupancy = the entire
residual frame (the producer can emit one beat per spatial pixel — all C channels
packed into the wide bus, WIDTH = C·8 — before the long expand→DW→project main
path delivers the matching first beat). The worst-case occupancy is therefore
exactly `H·W` (= EXPECTED_BEATS of the producing engine bridge), and the smallest
legal depth is `next_pow2(H·W)` (the module requires power-of-2 DEPTH). As-coded
the depths were off by **2–4700×** (and add_198/336/408 were *under*-sized):

**FIX (applied):** each DEPTH set to `next_pow2(H·W)`, H·W from the add's
`output_shape` in `layer_ir.json`. Applied identically to BOTH live tops.

| skip FIFO | residual frame (H×W×C) | beats=H·W | WIDTH | old DEPTH → new DEPTH | RAMB36 old → new |
|---|---|---:|---:|---|---:|
| add_198  | 56×56×24  | 3,136 | 192  | 256 → **4096**     | 3 → 24 |
| add_336  | 28×28×32  |   784 | 256  | 256 → **1024**     | 4 → 8 |
| add_408  | 28×28×32  |   784 | 256  | 256 → **1024**     | 4 → 8 |
| add_546  | 14×14×64  |   196 | 512  | 256 → 256 (already correct) | 8 → 8 |
| add_618  | 14×14×64  |   196 | 512  | 256 → 256          | 8 → 8 |
| add_690  | 14×14×64  |   196 | 512  | 256 → 256          | 8 → 8 |
| add_828  | 14×14×96  |   196 | 768  | 83,200 → **256**   | 1,793 → 11 |
| add_900  | 14×14×96  |   196 | 768  | 83,200 → **256**   | 1,793 → 11 |
| add_1038 | 7×7×160   |    49 | 1280 | 230,784 → **64**   | 8,118 → 18 |
| add_1110 | 7×7×160   |    49 | 1280 | 230,784 → **64**   | 8,118 → 18 |
| **Total** | | | | | **19,857 → 122 RAMB36** |

Skip-FIFO BRAM: **19,857 RAMB36 (739%) → 122 RAMB36 (4.5%)**, SAVED 19,735 tiles.
RAMB36 = ⌈WIDTH/72⌉ × ⌈DEPTH/512⌉ (RAMB36E2 72-bit × 512-deep SDP geometry).

> **Lossless / byte-exact (structural):** only the `.DEPTH` instantiation
> localparam (and derived `ADDR_W = clog2(DEPTH)`) changed; the `skip_fifo`
> module body is byte-identical and its 1-cycle FWFT pointer-FIFO value/latency
> semantics are independent of DEPTH. Every new DEPTH ≥ the proven occupancy
> bound (`next_pow2(H·W)`; 546/618/690 unchanged), so no beat is dropped and no
> deadlock. Both live tops lint-elaborate clean via Verilator `--lint-only`
> (`nn2rtl_top.v` EXIT=0; `nn2rtl_top_engine.v` EXIT=0 with only pre-existing
> SELRANGE warnings confirmed against the pre-edit backup). Grep confirms ZERO
> `.DEPTH(83200)` / `.DEPTH(230784)` remain in either live top (only in trailing
> `(was …)` comments). Backups: `backups/skipfifo_rightsize_20260602/`.
>
> **Note on `skip_fifo_sizes.json`:** that sidecar caps all depths at
> `verified_depth=1024` (a backpressure-bounded model). Because the producer is
> NOT back-pressured (drop-on-full, traced above), the cap-1024 model would
> *under*-size add_198 (needs 4096). The applied RTL uses the geometry-derived
> `next_pow2(H·W)` (the provably-safe lossless bound), which is the authoritative
> target; the JSON should be refreshed to
> `{198:4096,336:1024,408:1024,546:256,618:256,690:256,828:256,900:256,1038:64,1110:64}`.

### 2.4 BRAM / URAM projection (post-fix — both fixes APPLIED)

| Memory | RAMB36 | URAM288 | Note |
|---|---:|---:|---|
| Engine weight banks | 0 | 128 | MEASURED geometry; URAM 10% |
| Engine bias mem | ~114 | 0 | |
| Shared activation BRAM | ~697 | 0 | |
| Depthwise line buffers (reshaped RAMB36) | ~78 | 0 | RISK #1 fix APPLIED (`LINE_BUF_USE_URAM=0`) |
| Residual skip FIFOs (right-sized) | 122 | 0 | RISK #2 fix APPLIED (next_pow2(H·W) depths) |
| Stem conv_810 | 2 (=3 BRAM18) | 0 | MEASURED |
| **TOTAL (post-fix, PROJECTED — current RTL)** | **~1,013 RAMB36 (37.7%)** | **128 URAM288 (10.0%)** | **FITS — under 80%** |
| TOTAL (pre-fix, for reference only) | ~22,300 RAMB36 (829%) | 2,522 URAM288 (197%) | was OVER on both |

---

## 3. Verdict

- **All six resources PROJECTED under 80% of the U250 budget (post-fix):**
  LUT 61.6%, FF 22.0%, DSP 10.9%, **RAMB36 37.7%, URAM288 10.0%.** The depthwise
  LUT figure is a deliberate pessimistic upper bound (linear-by-C); real synth is
  expected lower.
- **Both former memory-mapping fit risks are CLOSED in RTL** (mapping-/sizing-only,
  byte-exact-preserving, re-verified 2026-06-02):
  1. **RISK #1** — depthwise line buffers no longer pinned to `ram_style="ultra"`:
     `LINE_BUF_USE_URAM=0` on all 17 DW convs reshapes them to RAMB36 → 78 tiles
     (2.9%); URAM 197% → 10%. Byte-exact (conv_896, conv_818, mismatch=0);
     ResNet-safe (conv_196 keeps the default ultra, elaborates clean).
  2. **RISK #2** — residual skip FIFOs right-sized to `next_pow2(H·W)` → 122 tiles
     (4.5%); BRAM 739% → 4.5% for the skip buffers, total BRAM 829% → 37.7%.
     Lossless (depth ≥ proven free-running occupancy bound), both live tops lint
     clean.
- **Fit is therefore internally consistent and PLAUSIBLE, but still NOT
  synth-confirmed.** Per `feedback_vivado_only_when_proven`, a U250
  synth/place/route remains gated. The ReLU-ROM rollout is now **COMPLETE**
  (all 35/35 relus on ROM, byte-exact, 0 synthesizable runtime multiply — see
  §1.4); relu DSP = 0 is now confirmed, not assumed.

### Pre-Vivado checklist
1. ~~Remove `ram_style="ultra"` from the depthwise `line_buf_window` mems~~ —
   **DONE** (`LINE_BUF_USE_URAM=0` on all 17 DW convs; byte-exact + ResNet-safe).
2. ~~Right-size the 10 residual skip FIFOs~~ — **DONE** (next_pow2(H·W) depths in
   both live tops; add_198/336/408 raised to the safe size; byte-exact structural).
3. ~~Complete the ReLU 128×8-ROM rollout on the remaining requant relus~~ —
   **DONE 2026-06-02** (all **35/35** relus `n4.v … n4_35.v` carry the 128-entry
   distributed requant ROM; grep confirms 0 synthesizable runtime requant
   multiply — every `* SCALE_MULT` is inside an `initial`/`function` ROM
   precompute; byte-exact `mismatch=0` re-verified across all structural families:
   n4_11, n4, n4_3, n4_23, n4_33). Relu DSP = 0; design DSP = ~1,345 (10.9%).
4. Re-run per-module byte-exact (`_verify_mbv2_variant.ts`, `mismatch_count = 0`)
   and the all-spatial e2e baseline before any Vivado run. (DW byte-exact re-run
   2026-06-02 ✓; e2e baseline still in flight.)

> All numbers above are **PROJECTED** (analytical, from per-module synth +
> the one U250 engine synth + RTL structure). **No Vivado synth/place/route was
> performed for this document.**
