#!/usr/bin/env python3
"""Curiosity sweep: top-1 accuracy of UNIFORM weight quantization at INT2/3/4/5/6/8,
using the exact GPTQ per-channel + A8 pipeline that gave 79.47% at INT4. Shows the
accuracy-vs-bits curve so the mixed-precision tradeoff is grounded in real numbers.
Storage (bits/weight) is the BRAM-fit driver, printed alongside.

Usage: gptq_bitwidth_sweep.py [n_calib=256] [n_val=1500]
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import torch
import gptq_int4 as g
import imagenet_util as iu
from torchvision.models import resnet50, ResNet50_Weights

# symmetric signed ranges: INTn -> [-(2^(n-1)), 2^(n-1)-1]
RANGES = {2: (1, -2), 3: (3, -4), 4: (7, -8), 5: (15, -16), 6: (31, -32), 8: (127, -128)}


def main() -> int:
    n_calib = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    n_val   = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    xcal, _    = iu.load_batch(n_calib)
    xval, yval = iu.load_batch(n_val, skip=n_calib)
    print(f"Bitwidth sweep (uniform weight INTn, GPTQ per-channel + A8): n_val={n_val} dev={dev}")
    print(f"  float : {g.eval_top1(base, xval, yval, dev)*100:.2f}%\n")
    print(f"  {'bits':>4} {'levels':>7} {'bits/wt':>8}  {'top-1 (+A8)':>12}")
    for bits in (8, 6, 5, 4, 3, 2):
        g.QMAX, g.QMIN = RANGES[bits]
        mg = g.gptq_quant(base, xcal, dev, scale_mode="channel")
        a8 = g.eval_top1(mg, xval, yval, dev, act_int8=True) * 100
        del mg
        print(f"  {bits:>4} {2**bits:>7} {bits:>8}  {a8:>11.2f}%")
    print("\n(All weights at the same width. Mixed precision = per-layer mix of these,")
    print(" giving more bits to sensitive layers. Storage/BRAM scales ~linearly with avg bits.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
