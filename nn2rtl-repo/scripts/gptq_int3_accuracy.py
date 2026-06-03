#!/usr/bin/env python3
"""Measure INT3 vs INT4 weight accuracy using the EXACT same GPTQ + per-channel +
A8 pipeline that produced the 79.47% INT4 number (scripts/gptq_int4.py). Only the
quantization bit-width changes (QMAX/QMIN module globals), so this is apples-to-
apples. Runs on GPU.

Usage: gptq_int3_accuracy.py [n_calib=256] [n_val=1500]
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import gptq_int4 as g          # the proven INT4 pipeline
import imagenet_util as iu
from torchvision.models import resnet50, ResNet50_Weights


def run_at_bits(label, qmax, qmin, base, xcal, xval, yval, dev):
    """Run the per-CHANNEL GPTQ + A8 eval at the given integer range."""
    g.QMAX, g.QMIN = qmax, qmin          # override module globals (call-time lookup)
    mg = g.gptq_quant(base, xcal, dev, scale_mode="channel")
    wonly = g.eval_top1(mg, xval, yval, dev) * 100
    a8    = g.eval_top1(mg, xval, yval, dev, act_int8=True) * 100
    del mg
    print(f"  {label:42s}: w-only {wonly:.2f}%   +A8 {a8:.2f}%")
    return a8


def main() -> int:
    n_calib = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    n_val   = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    xcal, _    = iu.load_batch(n_calib)
    xval, yval = iu.load_batch(n_val, skip=n_calib)
    print(f"INT3-vs-INT4 accuracy: n_calib={n_calib} n_val={n_val} dev={dev}")
    print(f"  float                                     : {g.eval_top1(base, xval, yval, dev)*100:.2f}%")
    # sanity: reproduce INT4 (expect ~79.5%)
    run_at_bits("INT4 GPTQ per-CHANNEL + A8 (reproduce)", 7, -8, base, xcal, xval, yval, dev)
    # the real question: INT3 everywhere
    run_at_bits("INT3 GPTQ per-CHANNEL + A8 (full)",      3, -4, base, xcal, xval, yval, dev)
    print("\n(INT3 = QMAX/QMIN 3/-4 = 3-bit symmetric. Same GPTQ+per-channel+A8 as the 79.47% run.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
