#!/usr/bin/env python3
"""Recompute residual-add fused requant constants from the CURRENT layer_ir
activation scales and (optionally) patch the add wrappers' localparams.

Root cause (2026-05-28): the INT4-GPTQ + ImageNet recalibration changed the
activation scales and regenerated the conv scales + goldens, but the residual
`add` wrappers kept their fused constants (LHS_FUSED_MULT / RHS_FUSED_MULT /
FUSED_ROUND_BIAS) from the PRIOR calibration -> every residual junction mis-
scales -> e2e value mismatch. The convs are per-OC and isolation-byte-exact;
the adds are the only op-type that was never refreshed.

Golden arithmetic (scripts/golden_impl.py Int8Add):
    out = clamp( round_half_up_toward_pos_inf(
              (lhs*lhs_scale + rhs*rhs_scale) / out_scale ), -128, 127)
RTL fixed-point (output/rtl/node_add*.v):
    out = sat( (lhs*LHS_MULT + rhs*RHS_MULT + HALF) >>> SHIFT )
We pick the SMALLEST shift (>= the wrapper's current one) for which the fixed-
point matches the golden float for ALL 65536 int8 (lhs,rhs) pairs, then emit
LHS_MULT=round(lhs_scale/out_scale*2^shift), RHS_MULT=round(rhs_scale/out_scale
*2^shift), HALF=1<<(shift-1).

Usage:
  python scripts/apply_add_rescale.py            # validate + report only
  python scripts/apply_add_rescale.py --apply    # also patch the RTL
"""
from __future__ import annotations
import json, re, sys, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
APPLY = "--apply" in sys.argv
MAX_SHIFT = 30  # keep mults within the 34-bit signed datapath


def round_half_up(x: np.ndarray) -> np.ndarray:
    return np.floor(x + 0.5)


def golden_add(lhs, rhs, ls, rs, os_):
    summed = lhs.astype(np.float64) * ls + rhs.astype(np.float64) * rs
    return np.clip(round_half_up(summed / os_), -128, 127).astype(np.int64)


def fixed_add(lhs, rhs, lm, rm, shift):
    half = (1 << (shift - 1)) if shift > 0 else 0
    v = lhs.astype(np.int64) * lm + rhs.astype(np.int64) * rm + half
    return np.clip(v >> shift, -128, 127).astype(np.int64)


def best_shift(ls, rs, os_, min_shift, max_mult=(1 << 33)):
    grid = np.arange(-128, 128, dtype=np.int64)
    L, R = np.meshgrid(grid, grid, indexing="ij")
    gold = golden_add(L, R, ls, rs, os_)
    for shift in range(min_shift, MAX_SHIFT + 1):
        lm = round(ls / os_ * (1 << shift))
        rm = round(rs / os_ * (1 << shift))
        # Respect the MULT field width: the constant is stored signed in MULT_W
        # bits, so it must be < 2^(MULT_W-1). Style A is 34-bit, Style B is 24-bit.
        if lm >= max_mult or rm >= max_mult:
            continue
        if np.array_equal(fixed_add(L, R, lm, rm, shift), gold):
            return shift, lm, rm, (1 << (shift - 1))
    return None  # no shift up to MAX_SHIFT works


def patch_file(path: Path, shift, lm, rm, half) -> bool:
    txt = path.read_text()
    orig = txt
    txt = re.sub(r"(FUSED_SHIFT\s*=\s*)\d+", rf"\g<1>{shift}", txt)
    txt = re.sub(r"(FUSED_ROUND_BIAS\s*=\s*)\d+'sd\d+", rf"\g<1>34'sd{half}", txt)
    # Two RTL naming conventions exist: Style A node_add_*.v use LHS_FUSED_MULT
    # (34-bit field); Style B (e.g. node_add_9/15) use FUSED_LHS_MULT with a
    # narrower MULT_W field. Patch whichever is present, preserving its width.
    if "LHS_FUSED_MULT" in txt:
        txt = re.sub(r"(LHS_FUSED_MULT\s*=\s*)\d+'sd\d+", rf"\g<1>34'sd{lm}", txt)
        txt = re.sub(r"(RHS_FUSED_MULT\s*=\s*)\d+'sd\d+", rf"\g<1>34'sd{rm}", txt)
    else:
        wm = re.search(r"FUSED_LHS_MULT\s*=\s*(\d+)'sd\d+", txt)
        w = wm.group(1) if wm else "24"
        txt = re.sub(r"(FUSED_LHS_MULT\s*=\s*)\d+'sd\d+", rf"\g<1>{w}'sd{lm}", txt)
        txt = re.sub(r"(FUSED_RHS_MULT\s*=\s*)\d+'sd\d+", rf"\g<1>{w}'sd{rm}", txt)
    if txt != orig:
        path.write_text(txt)
        return True
    return False


def main() -> int:
    layers = {l["module_id"]: l for l in json.load(open(ROOT / "output/layer_ir.json"))["layers"]}
    adds = [m for m in layers if layers[m]["op_type"] == "add"]
    print(f"{'add':<13}{'cur_sh':>7}{'new_sh':>7}{'LHS_MULT':>11}{'RHS_MULT':>11}{'HALF':>11}  status")
    n_ok = n_patched = n_fail = 0
    for mid in sorted(adds, key=lambda x: (len(x), x)):
        rtl = ROOT / f"output/rtl/{mid}.v"
        rtl_txt = rtl.read_text()
        cur_sh = int(re.search(r"FUSED_SHIFT\s*=\s*(\d+)", rtl_txt).group(1))
        # MULT field width caps the constant (signed): Style A = 34-bit (LHS_FUSED_MULT),
        # Style B = MULT_W-bit (FUSED_LHS_MULT, typically 24). Constant must be < 2^(W-1).
        if "LHS_FUSED_MULT" in rtl_txt:
            mult_w = 34
        else:
            mw = re.search(r"FUSED_LHS_MULT\s*=\s*(\d+)'sd", rtl_txt)
            mult_w = int(mw.group(1)) if mw else 24
        l = layers[mid]
        ls, rs, os_ = l["lhs_scale_factor"], l["rhs_scale_factor"], l["scale_factor"]
        res = best_shift(ls, rs, os_, min_shift=max(8, cur_sh - 4), max_mult=(1 << (mult_w - 1)))
        if res is None:
            print(f"{mid:<13}{cur_sh:>7}{'--':>7}{'(no exact shift<=MAX)':>33}")
            n_fail += 1
            continue
        shift, lm, rm, half = res
        status = "byte-exact"
        if APPLY:
            changed = patch_file(rtl, shift, lm, rm, half)
            status += " PATCHED" if changed else " (already)"
            n_patched += changed
        n_ok += 1
        print(f"{mid:<13}{cur_sh:>7}{shift:>7}{lm:>11}{rm:>11}{half:>11}  {status}")
    print(f"\n{n_ok}/{len(adds)} adds byte-exact-validated; "
          f"{n_patched} patched; {n_fail} failed."
          + ("" if APPLY else "  (dry-run; pass --apply to patch)"))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
