# PLAN B — Harden the MobileNetV2 golden requant to exact integer fixed-point

Status: DESIGN ONLY (read-only investigation; a build is running — no files were
modified except this plan). All facts below are from the on-disk MBV2 artifacts
(`output/mobilenet-v2/`) and were verified empirically with `python`.

## TL;DR verdict

| Op | Harden to integer? | Why |
|----|-----|-----|
| **node_mean (GlobalAveragePool)** | **YES — SAFE, do it** | `compute_scale_approx(scale_factor)` returns **exactly (7619, 18) == the RTL constants**, and `requantize_tensor_with_scale(acc, scale_factor)` is **byte-identical to the RTL over the ENTIRE accumulator domain (0 diffs)**. Differs from the current float golden at only 6 acc values (the ties) — exactly the robustness fix wanted. |
| **Int8ReLU** | **NO — SKIP / DO NOT harden generically** | The relu RTL (n4*.v) constants are **per-module, agent-chosen, NOT reproducible by `compute_scale_approx`** (mult widths 15–26 bits, shift varies). For `n4_4` the generic helper returns **119** at input 28 while the **actual RTL returns 118** → would BREAK the 8/8 gate. |
| **Int8Add** | **NO — SKIP** | The add RTL instances use **inconsistent rounding** (some unconditional `+HALF`, some sign-dependent `FUSED_HALF_M1` on negatives) and per-module mult/shift. The current FLOAT golden **already disagrees with RTL** at 31+ input pairs (198: 2, 1038: 27, 1110: 2) — masked only because those pairs don't occur in the 8 vectors. No single integer formula matches all instances. |

Net: **harden ONLY node_mean.** Hardening relu or add with the existing
`compute_scale_approx`/`requantize_*` helpers would change bytes that the RTL
does NOT produce → 8/8 gate breaks.

---

## How the MBV2 goldens are produced (so we edit the right thing)

- MBV2 goldens come from the ONNX path: `scripts/onnx_frontend.py`
  `run_int8_network()` (L1203) runs the Int8 nn.Modules forward, and
  `_build_goldout_for_spec()` (L1373) packs each module's output into
  `.goldout`. The per-module `.goldin/.goldout` are the e2e gate references.
- `_build_int8_module()` (onnx_frontend L1108) instantiates the three target
  modules WITH real scales:
  - relu  → `Int8ReLU(input_scale=spec.input_scale, output_scale=spec.output_scale)` (L1159)
  - add   → `Int8Add(lhs_scale, rhs_scale, output_scale)` (L1171)
  - gap   → `Int8GlobalAveragePool(input_scale, output_scale)` (L1177)
- Those three classes live in `scripts/golden_impl.py` (Int8ReLU L333,
  Int8Add L369, Int8GlobalAveragePool L398). **Editing these classes IS the
  golden-hardening change.** (Note: the PyTorch-checkpoint path at golden_impl
  L562-569 builds `Int8ReLU()` with NO scales — that path is ResNet/legacy and
  not the MBV2 gate; ignore it.)
- The relu/gap golden's effective scale == the LayerIR `scale_factor` field
  (relu: `input_scale/output_scale`; gap: `input_scale/output_scale/(H*W)`),
  confirmed against `output/mobilenet-v2/layer_ir.json`.

Integer helpers (golden_impl.py): `compute_scale_approx` (L1026, the SDK-mirror
search: shift∈[0,23], 1≤mult<32768, first-strict-improve tie-break),
`requantize_fixed_point_int` (L1055, scalar), `requantize_tensor_with_scale`
(L1101, tensor: `(round(acc)*mult + (1<<(shift-1))) >> shift`, floor-div on
negatives == Verilog `>>>`), `requantize_tensor_with_scale_per_oc` (L1148).
These are the SAME integer formula `compute_scale_approx`-derived RTL uses for
conv/gemm — but the relu/add modules were NOT generated with this helper.

---

## OP 1 — node_mean (GlobalAveragePool): HARDEN (safe, proven)

### Exact RTL integer formula
`output/mobilenet-v2/rtl/node_mean.v`:
- L30-31: `SCALE_MULT = 7619`, `SCALE_SHIFT = 18`.
- Accumulate: L82-86 `acc[ch] = Σ over 49 cells of $signed(int8)` (exact int sum).
- Scale: L93-96 `scaled = acc * SCALE_MULT_CONST`.
- Round+shift: L100-106
  `v_tmp = (scaled + (scaled[sign] ? SCALE_ROUND_HALF_M1 : SCALE_ROUND_HALF)) >>> SCALE_SHIFT`
  where `SCALE_ROUND_HALF = 1<<(SHIFT-1) = 131072`, `SCALE_ROUND_HALF_M1 = 131071`.
- Clamp: L180-182 `clamp(v_tmp, -128, 127)`.

The sign-dependent bias (`HALF` for ≥0, `HALF-1` for <0) is the canonical
round-half-toward-+inf, and it is provably IDENTICAL to the unconditional
`+HALF` + arithmetic floor-shift that `requantize_tensor_with_scale` uses (the
`-1` on negatives exactly compensates the floor toward -inf). Verified: **RTL
sign-dep == unconditional +HALF, 0 diffs over the entire acc domain.**

### Exact golden edit (golden_impl.py Int8GlobalAveragePool.forward, ~L415-438)
Replace the float requant tail (current L428-435):
```python
acc = x.to(torch.float32).sum(dim=(2, 3))                 # KEEP — exact int sum
composite = self.input_scale / float(self.output_scale) / float(h * w)
rescaled = round_half_up_toward_pos_inf(acc * composite)  # <-- REMOVE (float)
clamped = torch.clamp(rescaled, -128, 127)                # <-- REMOVE
```
with the integer path (reuse the existing helper):
```python
acc = x.to(torch.float64).sum(dim=(2, 3))                 # exact int sum, int64-safe
composite = self.input_scale / float(self.output_scale) / float(h * w)
clamped = requantize_tensor_with_scale(acc, composite)    # integer mult/add/floor-shift + clamp
```
`requantize_tensor_with_scale` (L1101) does
`compute_scale_approx(composite) -> (mult,shift); round(acc) -> int64;
(acc*mult + (1<<(shift-1))) // 2^shift; clamp(-128,127)`. With
composite = 0.029064472933370725 it picks **(7619, 18) == RTL exactly**.

Keep the `.unsqueeze(-1).unsqueeze(-1)` shape restore (L438). Import is already
in the module (`requantize_tensor_with_scale` is defined in the same file).

### Byte-exactness PROOF (all inputs, not just ties)
1. The golden acc is the exact integer Σ of int8 over 49 cells; the RTL acc is
   the exact integer Σ of `$signed(int8)` over 49 cells (node_mean.v L82-86).
   Identical accumulators. Acc range = [-128·49, 127·49] = [-6272, 6223].
2. `compute_scale_approx(composite) = (7619, 18)` == RTL `SCALE_MULT/SHIFT`
   (verified). So the multiply `acc*7619` is identical.
3. golden does `(acc*7619 + 131072) >> 18` (floor); RTL does
   `(acc*7619 + (acc<0 ? 131071 : 131072)) >>> 18`. For acc·7619 = q·2^18 + r,
   0≤r<2^18: golden = q + (r≥131072); RTL≥0 = same; RTL<0 = floor((q·2^18 + r +
   131071)/2^18) = q + (r≥131073) but the arithmetic shift floors, and for
   negative products the `-1` exactly cancels — empirically **0 diffs over every
   acc in [-6272, 6223]** (brute-forced all 12,496 values).
4. Both clamp to [-128, 127] identically.
∴ The new integer golden == RTL for **every possible input**, not just ties.

### What changes vs the current float golden
At 6 acc values (±2116, ±3217, ±4318) the current float golden rounds the .5
tie the "true math" way while the RTL/integer rounds via fixed-point. The new
golden matches RTL at those 6; the 8 e2e vectors do not hit them (gate stays
8/8), and B makes node_mean strictly more RTL-faithful.

---

## OP 2 — Int8ReLU: DO NOT harden (would break the gate)

### Exact RTL integer formula (per-module, e.g. n4.v / n4_2.v / n4_4.v)
relu modules requant a post-ReLU byte (input domain **0..127 only**, 7-bit ROM
address — negatives clamped to 0 first):
```
rom_scaled = $signed(k) * SCALE_MULT;
rom_vtmp   = (rom_scaled + (rom_scaled[sign] ? SCALE_ROUND_HALF_M1 : SCALE_ROUND_HALF)) >>> SCALE_SHIFT;
rom[k]     = clamp(rom_vtmp, -128, 127);    // upper clamp = the ReLU6 ceiling
```
(n4.v L56-66; identical structure in every n4*.v with per-module constants.)
Because the input is always ≥0, `rom_scaled ≥ 0`, so the sign-dependent branch
NEVER takes M1 — relu rounding is effectively unconditional `+HALF`.

### Why `compute_scale_approx` does NOT replicate it (make-or-break)
The relu `SCALE_MULT/SCALE_SHIFT` were chosen PER MODULE by the Foundry agent,
NOT by `compute_scale_approx`. They use mult fields of **15–26 bits** (n4:
28942851/2^16; n4_2: 13151/2^10; n4_4: 8667/2^11; n4_6: 23483/2^12), which
`compute_scale_approx` (capped mult<32768) cannot reproduce. Auditing all 35
relus, `compute_scale_approx`'s (mult,shift) == the RTL constants for only ~2.

The killer case — **n4_4 (scale_factor = 4.232076327006022), input k=28**:
- exact = 28·4.232076 = **118.498**  → true round 118
- RTL n4_4 (mult=8667, shift=11): `(28·8667 + 1024) >> 11 = 243700 >> 11 = ` **118**  (== float golden)
- `compute_scale_approx(4.232076)` returns **(17335, 12)** ≠ RTL → `(28·17335 + 2048) >> 12 = ` **119**
- `requantize_tensor_with_scale(28, 4.232076)` returns **119** (verified live).

So a generic `requantize_tensor_with_scale`-based relu golden would emit **119**
where the RTL emits **118** → **8/8 gate breaks** (if any feature map hits 28 on
that channel). The current float golden happens to give 118 (correct), so the
**current float relu golden is SAFER than a csa-integer harden.**

### Only safe way to harden relu (NOT recommended for B)
Mirror EACH module's ACTUAL constants read from its `n4*.v` (parse `SCALE_MULT`,
`SCALE_SHIFT`; rounding is unconditional `+HALF` since domain≥0; clamp
[-128,127]), feeding per-module (mult,shift) into the golden — analogous to
`apply_relu_rescale.py`'s `relu_rescale_params.json` recovery but for the n4*
constants. This is a per-module table threaded into Int8ReLU, a much larger
change than "swap a helper," and offers no benefit over the already-correct
float golden (the relu input domain is just 128 values; any close approx that
the agent verified is byte-exact, and the float golden matches it). **Flag and
skip.**

---

## OP 3 — Int8Add: DO NOT harden (inconsistent RTL + already-divergent)

### Exact RTL integer formula (per-module, two styles)
`output/mobilenet-v2/rtl/node_add_*.v`:
```
lhs_term = lhs_int8 * LHS_FUSED_MULT;   rhs_term = rhs_int8 * RHS_FUSED_MULT;
sum_pre  = lhs_term + rhs_term;
sum      = sum_pre + ROUND;             // ROUND differs per module (see below)
out      = clamp(sum >>> FUSED_SHIFT, -128, 127);
```
The fused mults approximate `lhs_scale/out_scale·2^SHIFT` and
`rhs_scale/out_scale·2^SHIFT`; SHIFT chosen per-module by `apply_add_rescale.py`
`best_shift()` (smallest shift byte-exact over all 65536 int8 pairs vs the
THEN-current golden). node_add_198 example header L4-7.

### The blocker: ROUND is inconsistent across instances
- **Unconditional `+FUSED_HALF`** (round-half-toward-+inf): node_add_198 (L151),
  336, 408, 546, 828, 900.
- **Sign-dependent `sum_pre[sign] ? FUSED_HALF_M1 : FUSED_HALF`**: node_add_618,
  690, 1038, 1110. This rounds negative .5 ties toward -inf — **NOT** the
  float golden's `floor(x+0.5)` (toward +inf).

So no single integer Int8Add can match all instances; you'd need per-module
mult/shift/round-mode (and 546/690/618 don't even expose `*_FUSED_MULT` — they
use `LHS_M/RHS_M` via a 3-stage pre-quant pipeline).

### The current float golden ALREADY disagrees with RTL (verified, 65536 pairs)
| module | shift | round | diffs (RTL vs current float golden) |
|---|---|---|---|
| node_add_198 | 20 | uncond | **2** (e.g. lhs=-77,rhs=93: exact 8.500003 → golden 9, RTL 8) |
| node_add_336/408/828/900 | 21/22/19/22 | uncond | 0 |
| node_add_1038 | 15 | sign-dep | **27** |
| node_add_1110 | 23 | sign-dep | **2** |
| node_add_546/618/690 | 20/22/22 | (LHS_M form) | not audited (parse) — sign-dep on 618/690 |

These diffs are masked only because the 8 e2e vectors don't hit those exact
(lhs,rhs) pairs. Hardening to ANY uniform integer formula would (a) still
mismatch the sign-dependent modules, and (b) for node_add_198 the integer mirror
gives 8 where the current float golden gives 9 — changing bytes the gate didn't
need changed and risking new divergence. **Flag and skip.** (If the user ever
wants Add hardened, the correct route is `apply_add_rescale.py`'s exact-mirror
per module incl. the sign-dep round-mode — a separate, RTL-constant-driven task,
not a golden_impl helper swap.)

---

## Regen + verify plan (after applying B = node_mean only, with E also on)

1. Back up the current E-only goldens before regen:
   `cp -r output/mobilenet-v2/goldens output/mobilenet-v2/goldens.E_backup`
   (or use the repo's backups/ convention).
2. Apply the node_mean edit in `scripts/golden_impl.py` (OP 1 above) only.
3. Regenerate with E+B on (same calibration/env the current 8/8 used; do NOT
   change `NN2RTL_GPTQ_CALIB`, `NN2RTL_GOLDEN_VECTORS`, seed, or input-scale
   flags — those fix the 8 vectors).
4. CHEAP safety check (the make-or-break, no Vivado): diff the regenerated
   relu/add/mean `.goldout` against the E-only backup:
   - **relu + add goldens MUST be byte-identical** to the backup (B touched
     neither). If they differ, something else regressed — STOP.
   - **node_mean.goldout**: compare to backup. For the CURRENT 8 vectors it
     should be byte-identical too (the 6 differing acc values are ties that the
     8 vectors don't hit). If node_mean.goldout differs even slightly, inspect:
     a diff means a vector hit one of the 6 tie accs — that is the intended
     robustness improvement, and the new value is the RTL-correct one, so it is
     still SAFE (RTL will match the new golden). Confirm the differing bytes
     equal `rtl_mean(acc)` for the implicated acc.
5. Run the per-module verify for `node_mean` (and a relu + an add as
   regression canaries) — expect mismatch=0 against RTL.
6. Run the full e2e gate — expect **8/8**.

Because B's node_mean golden is proven byte-identical to the RTL over the entire
accumulator domain, step 5/6 cannot regress node_mean; the only observable
change vs E-only is at tie accs, where the new golden is the one the RTL
actually produces.

---

## File:line index (evidence)

- Golden classes: `scripts/golden_impl.py` — Int8ReLU L333-366, Int8Add
  L369-395, Int8GlobalAveragePool L398-438.
- Integer helpers: `scripts/golden_impl.py` — compute_scale_approx L1026-1052,
  requantize_fixed_point_int L1055-1076, round_half_up_toward_pos_inf L1079-1098,
  requantize_tensor_with_scale L1101-1145, _per_oc L1148-1179.
- ONNX golden build: `scripts/onnx_frontend.py` — _build_int8_module L1108-1200
  (relu L1159, add L1171, gap L1177), run_int8_network L1203-1232,
  _build_goldout_for_spec L1373-1385, scale_factor dispatch L943-959.
- node_mean RTL: `output/mobilenet-v2/rtl/node_mean.v` — consts L30-31, accum
  L82-86, scale L93-96, round L100-106, clamp L180-182.
- relu RTL: `output/mobilenet-v2/rtl/n4.v` L36-66 (n4_4 consts: mult 8667 /
  shift 11); per-module constants vary across n4*.v.
- add RTL: `output/mobilenet-v2/rtl/node_add_198.v` consts L64-67, round L151,
  shift/sat L99-102; sign-dep variants node_add_690/1038/1110.
- add constant recovery: `scripts/apply_add_rescale.py` (best_shift over 65536).
- LayerIR scales: `output/mobilenet-v2/layer_ir.json` (relu n4 sf=441.63,
  n4_4 sf=4.232076; gap node_mean sf=0.029064472933370725 = mult7619/2^18;
  add node_add_198 ls=0.49070 rs=0.44968 os=0.47486).
