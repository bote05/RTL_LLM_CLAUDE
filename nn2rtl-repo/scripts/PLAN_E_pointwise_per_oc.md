# PLAN E — MobileNetV2 ENGINE (1×1 pointwise) requant PER-OUTPUT-CHANNEL

Status: PLAN ONLY (read-only investigation; no .v/.mem/.json/.py mutated).
Date: 2026-06-09.

---

## (1) CRUX VERDICT: the engine RTL is ALREADY per-output-channel. E is the EASY path.

The engine's requant stage is **per-OC by construction** — 256 parallel lanes, each lane =
one output channel, each lane reads its OWN multiplier from the scale ROM. NO engine RTL
change is required. The reason mbv2 pointwise is per-tensor TODAY is purely a *data* default:
`build_scale_memory_map.py` BROADCASTS the single per-tensor scale across all 256 lanes when
the LayerIR layer has no `scale_factor_per_oc`, and the frontend never sets `weight_scale_per_oc`
on pointwise (groups==1) convs.

File:line evidence:

- `output/rtl/engine/requant_pipeline.v:182` — `for (lane = 0; lane < 256; lane = lane + 1) begin : g_lane`
  (256 lanes = output-channel-parallel).
- `output/rtl/engine/requant_pipeline.v:197-198` — each lane's multiplier is per-lane:
  `wire signed [SCALE_W-1:0] mult_lane = $signed({1'b0, scale_q1[lane*32 +: 31]});`
  i.e. lane `i` reads slot `scale_in[i*32 +: 32]` → fully per-OC.
- `output/rtl/engine/requant_pipeline.v:61-71` — port doc: "PER-OUTPUT-CHANNEL scale … One 32-bit
  slot per lane (256 lanes = 8192 bits), aligned with bias_in … Replaces the former shared
  scale_mult[31:0]/scale_shift[5:0] (per-tensor)."
- `output/rtl/shared_engine_skeleton.v:505-507` — scale ROM addressed identically to bias:
  `assign scale_rd_addr = ag_bias_rd_addr; assign scale_rd_en = ag_bias_rd_en;
   wire [MAC_COUNT*32-1:0] requant_scale_in = scale_rd_data;` → 740 feeds `.scale_in(requant_scale_in)`.
- `output/rtl/shared_engine_skeleton.v:929-930` — the old per-tensor config-register scale is DEAD:
  `assign cfg_scale_mult = 32'd0; assign cfg_scale_shift = 6'd0;` (vestigial, not wired into requant).
- `scripts/build_scale_memory_map.py:83-96` — the BROADCAST branch: when `scale_factor_per_oc`
  is None (current mbv2 pointwise), `per_oc = [float(sf)] * oc` → all 256 lanes get the SAME
  per-tensor scale. When `scale_factor_per_oc` IS present (lines 99-109) it emits TRUE per-OC slots.
- `scripts/onnx_frontend.py:880-900` — frontend: depthwise (`_dw`) → per-OC branch (sets
  `weight_scale_per_oc` + `gptq_qweight`); pointwise (groups==1, `_dw` False) → `else` at 899-900
  `spec.weight_scale = _safe_weight_scale(...)` = per-tensor. Comment at line 890-891 confirms the
  intent: "Pointwise/engine convs stay per-tensor (the engine already requants per-OC)."

The Verilator-format engine isolation harness already wires the scale ROM as a per-OC 8192-bit
word: `tb/engine_iso_wrap_mbv2.v:8,60-62,83-84` ("scale_mem … 8192-bit = 256 INT32 per oc_pass").

CONCLUSION: E reduces to (a) frontend sets `weight_scale_per_oc` for the 34 pointwise convs,
(b) regen so `build_scale_memory_map.py` emits true per-OC slots, golden re-quantizes per-OC,
and engine weight banks / bias use the per-OC int weights. No RTL edit.

---

## (2) ENGINE-DISPATCHED POINTWISE NODE-IDS (34 convs)

`output/mobilenet-v2/mbv2-heavy-pointwise.txt` (34 entries, ALL groups==1, ALL 1×1):
```
814 816 820 822 826 828 832 834 838 840 844 846 850 852 856 858 862 864
868 870 874 876 880 882 886 888 892 894 898 900 904 906 910 912   (prefix node_conv_)
```
Verified from LayerIR: every entry has groups==1, weight_shape `[OC,IC,1,1]`, currently carries
`scale_factor` (per-tensor) and NO `scale_factor_per_oc`. Max OC = 1280 (node_conv_912) → 5
oc_passes; the scale ROM + `oc_passes` loop (`build_scale_memory_map.py:111-113`) already handle
multi-pass per-OC.

These are DISTINCT from the 17 INLINED DEPTHWISE convs (812,818,824,830,836,842,848,854,860,866,
872,878,884,890,896,902,908) which already got per-OC in change #1 — those use a SEPARATE per-conv
scale ROM (`node_conv_<id>_scale.mem`, built by `scripts/build_spatial_scale_mems.py`, RTL patched
by `scripts/apply_mbv2_depthwise_per_oc_scale.py`) with the OLD format `{shift[21:16], mult[15:0]}`
and a per-OC VARIABLE shift. **Do NOT touch the depthwise path for E.**

---

## (3) PER-OC SCALE MEMORY FORMAT (what the ENGINE expects)

Engine path = `scripts/build_scale_memory_map.py` → `output/mobilenet-v2/weights/scale.mem`.
Per-lane 32-bit slot is the **constant-shift (FIT-FIX 2026-06-07) format**, NOT the depthwise
`{shift,mult}` format:
- `bits[30:0]` = `mult' = mult << (FIXED_SHIFT - shift)`  (per-OC shift folded into the multiplier).
- `bits[31]`   = 0 (always positive, < 2^31).
- `FIXED_SHIFT = 23` (MUST match `output/rtl/engine/requant_pipeline.v:100` localparam FIXED_SHIFT).
- `(mult, shift) = compute_scale_approx(scale_factor_per_oc[ch])` (`scripts/golden_impl.py:1026`,
  shift∈[0,23], mult∈[1,32767]).
- Word layout: 256 slots, slot 255 first, big-endian per slot, 8192 bits = 2048 hex/line
  (`build_scale_memory_map.py:42-49`). The RTL recovers slot via `scale_in[lane*32 +: 32]`.

The engine RTL applies ONE compile-time `>>> FIXED_SHIFT` (requant_pipeline.v:244), so the
`[21:16]=shift` field is IGNORED on the engine path — only the pre-widened `mult'` (low 31 bits)
matters. The byte-exact identity is documented at `build_scale_memory_map.py:37-38` and
`requant_pipeline.v:99`:
`floor((biased·mult·2^(FS-shift) + 2^(FS-1)) / 2^FS) == floor((biased·mult + 2^(shift-1)) / 2^shift)`.

---

## (4) EXACT ORDERED EDIT LIST

### EDIT 1 — `scripts/onnx_frontend.py` (≈line 899-900): add a per-OC branch for pointwise/engine convs.

Currently the `else` of the depthwise test sets a per-tensor `weight_scale`. Replace the single
`else:` body so that 1×1 groups==1 (engine-dispatched) convs ALSO get per-OC. Concretely, change
the `else` (line 899-900) to a gated per-OC branch mirroring the depthwise branch, falling back to
per-tensor when disabled:

```python
            else:
                _pw = (int(spec.groups) == 1 and
                       int(spec.weight.shape[2]) == 1 and int(spec.weight.shape[3]) == 1)
                if _pw and os.environ.get("NN2RTL_PW_PER_CHANNEL", "1") != "0":
                    # [ACCURACY E] per-OUTPUT-CHANNEL INT8 for ENGINE 1x1 pointwise convs.
                    # The engine requant_pipeline already reads a per-OC scale ROM
                    # (output/rtl/engine/requant_pipeline.v lane loop); build_scale_memory_map.py
                    # emits true per-OC slots when scale_factor_per_oc is present. Same plain
                    # max/qmax per-OC as depthwise (no GPTQ; INT8). Downstream per-OC machinery
                    # (_spec_int_weight_and_scale / _spec_bias_int / _composite_conv_scale_per_oc /
                    # layer_ir export) applies unchanged. Set NN2RTL_PW_PER_CHANNEL=0 to revert.
                    _W = spec.weight
                    _per_oc = np.maximum(
                        np.abs(_W).reshape(_W.shape[0], -1).max(axis=1) / _WQMAX, 1e-12)
                    spec.weight_scale_per_oc = _per_oc.astype(np.float64)
                    spec.gptq_qweight = np.clip(
                        np.round(_W / _per_oc.reshape((-1,) + (1,) * (_W.ndim - 1))),
                        _WQMIN, _WQMAX).astype(np.int8)
                else:
                    spec.weight_scale = _safe_weight_scale(float(np.abs(spec.weight).max()))
```

WHY: `weight_scale_per_oc` is the single switch that turns ON the existing per-OC pipeline. With it
set, `_spec_int_weight_and_scale` (onnx_frontend.py:992-995) returns `gptq_qweight`, `_spec_bias_int`
(998-1004) quantizes bias per-OC, `_composite_conv_scale_per_oc` (1007-1013) produces the per-OC
composite scale, and the LayerIR export (1795-1801) writes `scale_factor_per_oc`. Engine weight
banks pick up the per-OC int weights via `_write_conv_hex_artifacts` (1275) → `build_weight_memory_map.py`.

NOTE on scope: the gate restricts the per-OC branch to true 1×1 groups==1 convs. The stem conv
(node_conv_810, 3×3 groups==1) and the Gemm (node_linear) are NOT 1×1, so they keep per-tensor —
matching the heavy-pointwise engine set exactly. (The stem is a spatial datapath, not engine; the
Gemm is op_type "gemm", a different branch at onnx_frontend.py:902-903, untouched.)

WHY NOT GPTQ: WEIGHT_BITS for mbv2 is 8 (INT8) → `USE_GPTQ` is False (onnx_frontend.py:986), and
INT8 per-OC needs no Hessian compensation. Plain max/qmax is correct and matches the depthwise #1
approach byte-for-byte.

### EDIT 2 — NO change to `scripts/build_scale_memory_map.py`.

It is already correct: with `scale_factor_per_oc` present on the pointwise layers it takes the
true-per-OC branch (lines 99-109) instead of the broadcast fallback (83-96). Re-running it after
the golden regen is sufficient.

### EDIT 3 — NO change to engine RTL (`requant_pipeline.v` / `shared_engine_skeleton.v`).

Per the verdict, the lane loop is already per-OC. The scale ROM addressing/width are already
per-oc_pass × 256 × 32b.

### EDIT 4 (optional, measurement only) — accuracy harness.

`scripts/measure_deployed_mbv2_acc.py` is already the per-OC-aware deployed measurement (it reads
the live weights/scales). It needs the SAME env (`NN2RTL_PW_PER_CHANNEL=1`) only if it re-runs the
frontend; if it reads the regenerated artifacts, no change. Confirm before measuring (it is
currently `M` modified in git — read it before relying on it).

---

## (5) EXACT REGEN COMMAND SEQUENCE

Run from repo root. `PY=python`. This mirrors the hardened regen rule
([[feedback-regen-must-rebuild-engine-maps]]) but for the mbv2 ENGINE path (no spatial repack of the
4 ResNet INT3 convs — those are ResNet). The depthwise per-OC RTL (#1) is already in place and is
NOT regenerated by these steps (its `node_conv_<id>_scale.mem` come from build_spatial_scale_mems).

```bash
# 0. BACK UP the current byte-exact baseline FIRST (per standing directive).
mkdir -p backups/mbv2_pw_per_oc_pre
cp output/mobilenet-v2/layer_ir.json                 backups/mbv2_pw_per_oc_pre/
cp output/mobilenet-v2/weights/scale.mem             backups/mbv2_pw_per_oc_pre/
cp output/mobilenet-v2/weights/bias.mem              backups/mbv2_pw_per_oc_pre/
cp output/mobilenet-v2/weights/uram_weights_bank*.mem backups/mbv2_pw_per_oc_pre/
cp output/mobilenet-v2/goldens/node_linear.goldout   backups/mbv2_pw_per_oc_pre/   # integer FC golden

# 1. Regenerate goldens + LayerIR with BOTH per-OC switches ON (depthwise #1 stays on).
NN2RTL_DW_PER_CHANNEL=1 NN2RTL_PW_PER_CHANNEL=1 \
  $PY scripts/generate_golden.py checkpoints/mobilenet_v2.onnx --network mobilenet-v2
#   (calibration must match the deployed/measured calibration — see [[project-deploy-vs-measure-calibration]];
#    use the same NN2RTL_IMAGENET_CALIB the depthwise #1 regen used, if any, to stay accuracy-faithful.)

# 2. Rebuild engine weight banks from the per-OC int-weight hex (URAM banks).
$PY scripts/build_weight_memory_map.py --network mobilenet-v2

# 3. Rebuild the per-OC ENGINE scale ROM (now emits TRUE per-OC slots for the 34 pointwise).
$PY scripts/build_scale_memory_map.py --network mobilenet-v2 \
    --heavy-list output/mobilenet-v2/mbv2-heavy-pointwise.txt

# 4. Rebuild the engine BIAS map (per-OC bias int changed → MANDATORY, the #16% root cause).
$PY scripts/build_bias_memory_map.py --network mobilenet-v2 \
    --heavy-list output/mobilenet-v2/mbv2-heavy-pointwise.txt

# 5. Rebuild the DEPTHWISE per-conv scale ROMs (unchanged content, but keep the pipeline whole).
$PY scripts/build_spatial_scale_mems.py            # writes node_conv_<dw>_scale.mem (mbv2 default dir)

# 6. e2e byte-exact gate (8 vectors) — must be 8/8 mismatch=0 vs the FRESH per-OC golden.
SKIP_BUILD=0 bash scripts/run_mbv2_e2e_parallel.sh

# 7. Accuracy gate (deployed top-1, ≥256 imgs, all trust gates).
NN2RTL_PW_PER_CHANNEL=1 NN2RTL_DW_PER_CHANNEL=1 $PY scripts/measure_deployed_mbv2_acc.py   # confirm args/usage first
```

VERIFY the heavy-list / network arg defaults before running: `build_scale_memory_map.py` and
`build_bias_memory_map.py` both DEFAULT `--network resnet-50` and a ResNet heavy list
(`build_scale_memory_map.py:30,55-56`; `build_bias_memory_map.py:124,126`). For mbv2 you MUST pass
`--network mobilenet-v2 --heavy-list output/mobilenet-v2/mbv2-heavy-pointwise.txt` or it will build
the WRONG network's maps. (Check whatever mbv2 regen the depthwise #1 used for the exact invocation
it relied on; replicate it.)

CRITICAL: build_scale_memory_map asserts `shift <= FIXED_SHIFT(23)` (line 102) and
`0 <= mult' < 2^31` (line 105) per channel — see risk (5) below; these are expected to pass.

---

## (4-bis) BYTE-EXACTNESS ARGUMENT (RTL matches the regenerated per-OC golden)

The RTL does not change, so byte-exactness is purely "does the regenerated golden equal what the
unchanged per-OC RTL computes from the regenerated mems". Each artifact is produced from the SAME
`weight_scale_per_oc` and the SAME `compute_scale_approx`, so they are mutually consistent:

1. **Weights**: engine URAM banks are built from `_write_conv_hex_artifacts` →
   `_spec_int_weight_and_scale` → `gptq_qweight` (per-OC rounded int weights). The golden
   `Int8Conv2d` (golden_impl.py:272-292) consumes the SAME `w_int8` (via `_build_int8_module`,
   onnx_frontend.py:1102). Same integer weights in RTL and golden.
2. **Bias**: `_spec_bias_int` (onnx_frontend.py:998-1004) computes `bias_int[oc] =
   round(b[oc] / (in_scale·ws[oc]))` per-OC; the SAME ints go to `bias.mem` (build_bias_memory_map)
   and to the golden module. Same integer bias.
3. **Scale**: golden requant = `requantize_tensor_with_scale_per_oc(y, scale_factor_per_oc)`
   (golden_impl.py:291), where `scale_factor_per_oc[ch] = in_scale·ws[ch]/out_scale`
   (`_composite_conv_scale_per_oc`, onnx_frontend.py:1007-1013). The engine applies
   `compute_scale_approx(scale_factor_per_oc[ch])` folded as `mult' = mult<<(23-shift)` then
   `>>>23`. The constant-shift identity (build_scale_memory_map.py:37-38) makes that byte-identical
   to `(biased·mult + 2^(shift-1))>>>shift`. The golden's `requantize_*_per_oc` MUST use the same
   `compute_scale_approx`-derived integer arithmetic (it does — same module that the depthwise #1
   path already proved byte-exact e2e 8/8).
4. **Rounding**: requant_pipeline uses unconditional `+ROUND_CONST` = round-half-up toward +inf
   (requant_pipeline.v:235-243), matching `golden_impl.requantize_tensor_with_scale_per_oc`
   (documented at requant_pipeline.v:237-242). Same as the proven depthwise per-OC path.
5. **Precedent**: change #1 (depthwise per-OC) used the IDENTICAL per-OC machinery and is e2e 8/8
   byte-exact AND +4.0% top-1 (onnx_frontend.py:884-885). E exercises the SAME code paths but on
   the engine's already-per-OC datapath, so the same byte-exactness holds, conditioned on the regen
   sequence in (5) being run completely (omitting build_bias_memory_map or build_scale_memory_map =
   stale per-tensor mems vs per-OC golden = guaranteed mismatch — this is the canonical regen trap).

---

## (5) RISKS / BLOCKERS + IS IT WORTH IT

### Risk R1 — FIXED_SHIFT / 31-bit slot overflow. LOW / essentially none.
`multp = mult << (23 - shift)` must be < 2^31 (build_scale_memory_map.py:105). `multp ≈ round(scale·2^23)`,
monotonic in scale. Per-OC composite scale ≤ per-tensor composite (the per-tensor scale uses
max-abs over ALL channels, so every per-OC scale ≤ it). MEASURED: the current per-tensor pointwise
scales give worst `multp/2^31 = 9.3e-5` (node_conv_814: sf=0.00372, multp=31204). Even the largest
plausible per-OC scale is far below 2^31. The small-scale end is also safe: tiny scales (the
deployed depthwise per-OC range goes down to ~7.8e-14) yield `compute_scale_approx → (mult=1,
shift=23)` → `multp = 1`. So both the `shift>23` assert (line 102) and the overflow assert (line
105) pass with huge margin. If an unexpected channel ever trips line 105, the message tells you to
lower FIXED_SHIFT or widen the slot — but it will not happen here.

### Risk R2 — regen incompleteness (the real trap). MEDIUM, fully mitigated by (5).
Skipping `build_bias_memory_map` or `build_scale_memory_map` (or running them with the default
ResNet `--network`/heavy-list) leaves stale per-tensor mems vs a per-OC golden → e2e mismatch
masked or wrong-network artifacts. Mitigation: run the FULL sequence in (5) with explicit
`--network mobilenet-v2 --heavy-list …`. This is exactly the [[feedback-regen-must-rebuild-engine-maps]]
rule.

### Risk R3 — calibration faithfulness. MEDIUM (accuracy, not byte-exactness).
Deployed weights ≠ accuracy-measurement weights unless calibration is pinned
([[project-deploy-vs-measure-calibration]]). For E to *measurably* recover accuracy, generate_golden
must use the same calibration the depthwise #1 deployment used. Byte-exactness (e2e gate) is
unaffected by calibration; only the top-1 number is.

### Risk R4 — node_linear (Gemm) and stem stay per-tensor. By design, not a blocker.
The Gemm path (onnx_frontend.py:902-903) and the 3×3 stem are untouched; only the 34 1×1 engine
convs change. This matches the heavy-pointwise engine set exactly.

### Is per-OC pointwise WORTH it? Modest but positive; LOW effort.
Pointwise (1×1) convs are far more robust to per-tensor quantization than depthwise: depthwise has
1 input channel per filter so per-filter scale spread is large (hence #1's big +4.0%), whereas
pointwise sums over many input channels and its per-OC weight ranges are much tighter. Expected
deployed top-1 delta from E alone: roughly **+0.3% to +0.8%** (a fraction of the depthwise win),
narrowing the remaining ~1.46% gap to float but not closing it. The effort is tiny (one frontend
branch + a standard regen, ZERO RTL change, ZERO new byte-exactness risk beyond running the regen
fully), so the cost/benefit is favorable — it is the natural next lever after #1. If the measured
delta is below noise on the eval set, the change is still safe to keep (byte-exact) and harmless.

---

## SUMMARY

- Engine RTL is ALREADY per-OC (requant_pipeline.v:182,197-198; shared_engine_skeleton.v:505-507).
  NO RTL change.
- One frontend edit (onnx_frontend.py ~line 899: add a 1×1 groups==1 per-OC branch gated on
  NN2RTL_PW_PER_CHANNEL) turns on the existing per-OC pipeline for all 34 engine pointwise convs.
- Regen: generate_golden (both per-OC switches) → build_weight_memory_map → build_scale_memory_map
  → build_bias_memory_map → build_spatial_scale_mems, ALL `--network mobilenet-v2` (+ mbv2 heavy
  list for the scale/bias maps) → e2e 8/8 gate → accuracy gate.
- Byte-exact by construction (same per-OC ints/scales feed RTL mems and golden; same proven path as
  depthwise #1). FIXED_SHIFT(23)/31-bit-slot overflow is not a concern (worst multp = 9.3e-5 of 2^31).
- Worth it: modest +0.3–0.8% expected (pointwise robust to per-tensor), but cheap and zero-RTL-risk.
