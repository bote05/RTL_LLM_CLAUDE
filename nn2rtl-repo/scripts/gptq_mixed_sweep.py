#!/usr/bin/env python3
"""Mixed-precision sweep: demote the LARGEST conv layers (by weight count) to INT3,
keep the rest at INT4, sweeping how many are demoted. Reports top-1 accuracy AND the
cumulative weight storage (Mbit) at each step — so we can read off the best accuracy
that still fits BRAM. Uses the same GPTQ per-channel + A8 pipeline as the 79.47% run.

The fit target: weights must drop enough that BRAM < 2688 after the other measured
levers. From OOC: all-INT4 weights = 3130 tiles (after cascade+line_buf->URAM, over
by 442); each demoted layer's tiles shrink ~0.72x. We report weight-Mbit and an
approximate tile estimate so the fit crossover is visible.

Usage: gptq_mixed_sweep.py [n_calib=256] [n_val=1500]
"""
from __future__ import annotations
import sys, copy
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import torch
import torch.nn as nn
import gptq_int4 as g
import imagenet_util as iu
from torchvision.models import resnet50, ResNet50_Weights

R3 = (3, -4)   # INT3
R4 = (7, -8)   # INT4


def gptq_quant_mixed(base, xs_cal, dev, int3_layer_ids: set):
    """Same GPTQ per-channel pipeline as g.gptq_quant, but layer i uses INT3 if
    i in int3_layer_ids else INT4. Re-implemented minimally by monkeypatching the
    per-layer (QMAX,QMIN) inside g via a wrapper around its quant step is hard;
    instead we run g.gptq_quant TWICE (all-INT4 and all-INT3) and splice weights:
    take INT3-quantized weights for the demoted layers, INT4 for the rest. This is
    exact because GPTQ here is non-sequential (per-layer independent Hessian)."""
    g.QMAX, g.QMIN = R4
    m4 = g.gptq_quant(base, xs_cal, dev, scale_mode="channel")
    g.QMAX, g.QMIN = R3
    m3 = g.gptq_quant(base, xs_cal, dev, scale_mode="channel")
    cs4, cs3 = g.convs(m4), g.convs(m3)
    with torch.no_grad():
        for i, c in enumerate(cs4):
            if i in int3_layer_ids:
                c.weight.copy_(cs3[i].weight.data)
    del m3
    return m4


def main() -> int:
    n_calib = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    n_val   = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    xcal, _    = iu.load_batch(n_calib)
    xval, yval = iu.load_batch(n_val, skip=n_calib)

    cs = g.convs(base)
    sizes = [(i, c.weight.numel()) for i, c in enumerate(cs)]
    order = [i for i, _ in sorted(sizes, key=lambda t: -t[1])]   # biggest first
    total_w = sum(n for _, n in sizes)
    print(f"Mixed-precision sweep: {len(cs)} convs, {total_w/1e6:.1f}M weights, n_val={n_val} dev={dev}")
    print(f"  float: {g.eval_top1(base, xval, yval, dev)*100:.2f}%")
    print(f"  all-INT4 weight bits = {total_w*4/1e6:.1f} Mbit ; all-INT3 = {total_w*3/1e6:.1f} Mbit\n")
    print(f"  {'#INT3 layers':>12} {'INT3 wt%':>9} {'avg bits':>9} {'wt Mbit':>8}  {'top-1':>8}")

    # sweep: 0, then progressively demote biggest-first
    for k in (0, 4, 8, 12, 16, 20, 26, 35, len(cs)):
        ids = set(order[:k])
        int3_w = sum(cs[i].weight.numel() for i in ids)
        avg_bits = (int3_w*3 + (total_w-int3_w)*4) / total_w
        wt_mbit = (int3_w*3 + (total_w-int3_w)*4)/1e6
        m = gptq_quant_mixed(base, xcal, dev, ids)
        acc = g.eval_top1(m, xval, yval, dev, act_int8=True)*100
        del m
        print(f"  {k:>12} {100*int3_w/total_w:>8.0f}% {avg_bits:>9.2f} {wt_mbit:>8.1f}  {acc:>7.2f}%")
    print("\n(INT3 on the k largest layers, INT4 on the rest. Lower wt-Mbit -> fewer BRAM tiles.")
    print(" all-INT4=108Mbit~3130 tiles(over by 442); need ~<94Mbit-equiv to fit. Read the accuracy at that point.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
