# MBV2 Depthwise CONSTANT-SHIFT Requant Conversion — Analysis & Audit

Date: 2026-06-10 · Applier: `scripts/apply_mbv2_dw_constshift.py` · Marker: `[DW-CONSTSHIFT]`
Scope: the 17 inlined MobileNetV2 depthwise wrappers
`output/mobilenet-v2/rtl/node_conv_{812,818,824,830,836,842,848,854,860,866,872,878,884,890,896,902,908}.v`
and their per-channel scale ROMs
`output/mobilenet-v2/weights/node_conv_<id>_scale.mem`.

This is the FIT-FIX constant-shift form proven on the ResNet engine
(`output/rtl/engine/requant_pipeline.v` `[FIT-FIX 2026-06-07]` +
`scripts/build_scale_memory_map.py`), ported to the DW spatial wrappers.

---

## 1. Format decode (BEFORE)

Confirmed by reading the live RTL (post `[PER-OC 2026-06-08]` + `[K1-MBV2]` state):

* `scale_rom` — `reg [31:0] scale_rom [0:C-1]`, `$readmemh(...node_conv_<id>_scale.mem)`,
  one 32-bit hex word per output channel:
  * bits `[15:0]`  = `mult`  (`compute_scale_approx`, 15-bit cap: `1 ≤ mult ≤ 32767`)
  * bits `[21:16]` = `shift` (range `[0, 23]`, the `compute_scale_approx` search range)
  * bits `[31:22]` = 0 (verified on every slot)
* `ST_SCALE` (Block A, sync-only):
  `scaled[lane] <= $signed(biased[lane]) * $signed(scale_rom[sc_oc][15:0]);`
  with `SCALED_W = BIASED_W + 16 = 34 + 16 = 50`.
* `ST_OUTPUT` (Block A): per-lane VARIABLE decode + barrel shift —
  ```verilog
  out_shift = scale_rom[out_oc][21:16];
  out_round = (out_shift == 6'd0) ? 0 : 1 <<< (out_shift - 1);
  v_tmp     = (scaled[lane_i] + out_round) >>> out_shift;   // 50-bit VARIABLE shifter
  ```
  ≈ 272 lanes total (17 convs × MP=16), each burning a ~50-bit variable arithmetic
  barrel shifter + a variable round-constant decoder — pure LUT fabric inside the
  congestion-hot DW convs.

## 2. Format decode (AFTER)

* mem slot: bits `[30:0]` = **`mult' = mult << (23 - shift)`** (pre-widened multiplier,
  always `< 2^31`; measured max is `2^23`, see audit), bit `[31]` = 0.
  Each mem now carries a leading `// [DW-CONSTSHIFT ...]` comment header (legal for
  `$readmemh` in Verilator/iverilog/Vivado) which doubles as the format/idempotency marker.
* RTL (`[DW-CONSTSHIFT 2026-06-10]` markers):
  ```verilog
  localparam integer MULTP_W        = 32;                 // {1'b0, slot[30:0]} signed operand
  localparam integer SCALED_W       = BIASED_W + MULTP_W; // 66 (was 50)
  localparam integer DW_FIXED_SHIFT = 23;
  localparam signed [SCALED_W-1:0] DW_ROUND_CONST = 1 <<< (DW_FIXED_SHIFT - 1); // 2^22

  // ST_SCALE:
  scaled[lane_i] <= $signed(biased[lane_i]) * $signed({1'b0, scale_rom[sc_oc][30:0]});
  // ST_OUTPUT:
  v_tmp = (scaled[lane_i] + DW_ROUND_CONST) >>> DW_FIXED_SHIFT;   // compile-time shift
  ```
  `out_shift` / `out_round` temporaries deleted. FSM, states, beat timing untouched →
  **zero cycle change** (e2e_cycles must stay exactly 7,592,966).

## 3. Byte-exact identity proof

For `s ∈ [0,23]`, `mult ∈ [1,32767]`, `FS = 23`, any signed `x` (= `biased`):

```
floor((x·mult + 2^(s-1)) / 2^s)  ==  floor((x·(mult << (FS-s)) + 2^(FS-1)) / 2^FS)
```

Proof: `x·mult·2^(FS-s) + 2^(FS-1) = 2^(FS-s) · (x·mult + 2^(s-1))` **exactly**
(because `2^(FS-1) = 2^(FS-s)·2^(s-1)` for `s ≥ 1`), and `floor(2^k·A / 2^FS) =
floor(A / 2^s)` with `k = FS-s`; arithmetic `>>>` on a signed value is exactly
floor-division by a power of two (both Verilog and Python `>>`).
The `s = 0` case (old RTL: `round = 0`, plain `x·mult`) is also exact:
`floor((x·mult·2^23 + 2^22)/2^23) = x·mult + floor(2^22/2^23) = x·mult`.

Width safety: `|biased| < 2^33` (BIASED_W=34) and `mult' < 2^31` ⇒ `|scaled| < 2^64`,
held in `SCALED_W = 66` signed — no truncation. `mult'` is zero-extended to a 32-bit
signed operand so the multiply is never sign-misread (old code had the same property
with `mult ≤ 32767` in a 16-bit signed operand).

**Empirical confirmation** (run during conversion): all 7,136 slots × 213 sampled
`x` values (extremes ±(2^33−1), ties, random) = **1,519,968 (slot,x) pairs,
OLD formula == NEW formula, 0 mismatches** (pre- and post-clamp).

## 4. Per-mem slot audit (decoded from the live OLD mems, pre-conversion)

| conv | C (slots) | shift min..max | #shift=0 | max mult' | mult' bits |
|-----:|----------:|:--------------:|---------:|----------:|-----------:|
| 812  | 32   | 0..23  | 2 | 8,388,608 | 24 |
| 818  | 96   | 17..23 | 0 | 33,630    | 16 |
| 824  | 144  | 0..23  | 1 | 8,388,608 | 24 |
| 830  | 144  | 17..23 | 0 | 53,720    | 16 |
| 836  | 192  | 13..23 | 0 | 346,496   | 19 |
| 842  | 192  | 12..23 | 0 | 92,280    | 17 |
| 848  | 192  | 16..23 | 0 | 38,074    | 16 |
| 854  | 384  | 0..23  | 1 | 8,388,608 | 24 |
| 860  | 384  | 0..23  | 1 | 8,388,608 | 24 |
| 866  | 384  | 15..23 | 0 | 161,936   | 18 |
| 872  | 384  | 14..23 | 0 | 352,288   | 19 |
| 878  | 576  | 13..23 | 0 | 216,448   | 18 |
| 884  | 576  | 0..23  | 1 | 8,388,608 | 24 |
| 890  | 576  | 11..23 | 0 | 136,784   | 18 |
| 896  | 960  | 0..23  | 3 | 8,388,608 | 24 |
| 902  | 960  | 12..23 | 0 | 271,792   | 19 |
| 908  | 960  | 12..23 | 0 | 60,334    | 16 |
| **Σ** | **7,136** | 0..23 | **10** | **8,388,608 = 2^23** | **24** |

Findings vs the prior analysis the task quoted:

* **Total slots = 7,136, not 4,560** (4,560 may have been a count over a subset;
  7,136 = Σ C over the 17 DW convs, verified against both the mems and
  `layer_ir.json` `scale_factor_per_oc` lengths).
* `shift ≤ 23` holds on **all** 7,136 slots — CONFIRMED.
* `6/17 mems contain shift=0` — CONFIRMED (812×2, 824, 854, 860, 884, 896×3 = 10 slots).
  Every one is `(mult=1, shift=0)` = **effective factor exactly 1.0**, so
  `mult' = 1<<23 = 2^23` — this is the global maximum; `mult' < 2^31` holds with
  **7 bits of headroom** (max is 24-bit).
* Effective-factor identity `mult'·2^shift == mult·2^23` verified EXACTLY
  (integer arithmetic) on all 7,136 slots.
* Provenance: every on-disk `(mult, shift)` == `compute_scale_approx(`
  `layer_ir.scale_factor_per_oc[ch])` — 7,136/7,136 match, so the mems were in
  sync with the authoritative quantization source before converting.

## 5. The atomicity hazard + regen-pipeline note (READ BEFORE ANY REGEN)

This repo's worst bug class (ResNet "2953", memory
`project_resnet_2953_stale_scalemem`): **RTL slot format and .mem format must flip
together**. Mitigations shipped:

1. `scripts/apply_mbv2_dw_constshift.py` is ONE atomic script: validates every RTL
   anchor (count==1) and every mem slot of all 17 convs FIRST, writes RTL+mem
   together only after everything passes, hard-aborts on any **mixed state**
   (RTL new + mem old or vice versa), idempotent via the `[DW-CONSTSHIFT]` marker,
   `--dry-run`, full backups (`backups/mbv2_dw_constshift_<ts>/`).
2. **Authoritative generator extended** — the DW mems are produced by
   `scripts/build_spatial_scale_mems.py` (run with
   `NN2RTL_GOLDEN_BASE=output/mobilenet-v2`; it was the script that deployed the
   per-OC DW quant). It now **reads the consuming RTL** (`BASE/rtl/<module_id>.v`)
   and, when the `[DW-CONSTSHIFT]` marker is present, emits the NEW `mult'` format
   (with header comment + `shift≤23` / `mult'<2^31` asserts); otherwise it emits the
   legacy `{shift,mult}` layout (ResNet spatial convs, `conv_datapath_mp_k`
   SCALE_PATH consumers are untouched). A future
   `generate_golden → build_spatial_scale_mems` regen therefore **cannot silently
   revert** the DW mems to the old format under constant-shift RTL.
   Verified: post-conversion regen reproduces all 17 mems payload-identical.
3. The mems self-describe: a converted mem starts with `// [DW-CONSTSHIFT ...]`;
   an old-format mem decoded by the applier must have bits `[31:22] = 0`, which a
   `mult' ≥ 2^22` new-format word violates → re-running the applier on an
   already-converted mem cannot mis-decode it as old format (and the marker check
   fires first anyway).

NOT affected: the engine's wide `scale.mem` (already constant-shift via
`build_scale_memory_map.py`), the stem `node_conv_810` (still per-tensor
localparam requant), pointwise engine convs, goldens (`golden_impl` computes from
`layer_ir`, not from the mems — and the conversion is bit-exact anyway).

## 6. Gate results (2026-06-10)

* **Lint**: `verilator_bin --lint-only` on all 17 modules (+ `line_buf_window.v`,
  `coord_scheduler.v`): **0 errors**; warning fingerprints **byte-identical to the
  pre-conversion baseline** (all pre-existing `rtl_library` WIDTHEXPAND/WIDTHTRUNC/
  PINMISSING/WIDTHCONCAT — none introduced by this change).
* **e2e**: `bash scripts/run_mbv2_e2e_parallel.sh` — see final report
  (requirement: `RESULT: PASS (8/8 byte-exact)` and `e2e_cycles == 7,592,966`
  on all 8 vectors).

## 7. Expected PPA delta (Vivado, not yet run — synth gated per standing rule)

* **LUT**: each lane loses a ~50-bit variable arithmetic barrel shifter
  (~6 levels × ~50b ≈ 250–330 LUT) + variable round decode (~50 LUT).
  ≈ 272 lanes ⇒ **−45…−60K LUT**, concentrated inside the congestion-hot DW convs.
* **DSP**: ST_SCALE multiply widens 34×16 → 34×32 (`use_dsp = "yes"`):
  ~2 → ~4 DSP48E2 per lane ⇒ **roughly +500–550 DSP** total (U250 DSP is ~90% idle).
* **FF**: `scaled[]` widens 50→66 bits: +16 FF × 16 lanes × 17 convs ≈ **+4.4K FF**
  (~0.6% of the design's FF) — effectively flat. `out_shift`/`out_round` were
  blocking-assigned temporaries (combinational), so no FF returned there.
* **Cycles**: **zero** — no FSM/state/latency change (gate-verified).
