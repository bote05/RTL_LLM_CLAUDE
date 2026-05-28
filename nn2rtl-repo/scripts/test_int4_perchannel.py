#!/usr/bin/env python3
"""Decisive weight-quantization experiment (fast, float-activation, on GPU):
quantize torchvision ResNet-50 V2 weights to INT4/INT8, per-tensor vs
per-output-channel, dequantize, and measure top-1 on N val images.

Isolates the WEIGHT-quantization effect (activations kept float) to decide
whether per-tensor INT4 (Scheme A as specified) is the problem and whether
per-channel INT4 recovers accuracy -> informs the precision decision.
  py scripts/test_int4_perchannel.py [N=512]
"""
from __future__ import annotations
import sys
from pathlib import Path
import copy
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import imagenet_util as iu  # noqa: E402


def quant_dequant(w: torch.Tensor, bits: int, per_channel: bool) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    if per_channel:                      # per output channel = dim 0
        dims = tuple(range(1, w.dim()))
        s = w.abs().amax(dim=dims, keepdim=True) / qmax
    else:
        s = w.abs().max() / qmax
    s = s.clamp_min(1e-12)
    q = torch.clamp(torch.round(w / s), -qmax - 1, qmax)
    return q * s


def quantize_model(base, bits, per_channel, dev):
    m = copy.deepcopy(base).to(dev).eval()
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
                mod.weight.copy_(quant_dequant(mod.weight.data, bits, per_channel))
    return m


def acc(m, xs, ys, dev):
    top1 = 0
    bs = 128
    with torch.no_grad():
        for s in range(0, len(ys), bs):
            xb = torch.from_numpy(xs[s:s+bs]).to(dev)
            p = m(xb).argmax(1).cpu().numpy()
            top1 += int((p == ys[s:s+bs]).sum())
    return top1 / len(ys)


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from torchvision.models import resnet50, ResNet50_Weights
    base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    xs, ys = iu.load_batch(n)
    print(f"N={n} val images, dev={dev}")
    configs = [
        ("float (baseline)", None, None),
        ("INT8 per-tensor", 8, False),
        ("INT8 per-channel", 8, True),
        ("INT4 per-tensor", 4, False),
        ("INT4 per-channel", 4, True),
    ]
    for name, bits, pc in configs:
        m = base.to(dev).eval() if bits is None else quantize_model(base, bits, pc, dev)
        a = acc(m, xs, ys, dev)
        print(f"  {name:22s}: top1={a*100:.2f}%")
        if bits is not None:
            del m
    print("\n=> if INT4 per-channel recovers but per-tensor collapses, INT4 needs "
          "per-channel weight scales (requant rework) — not Scheme A as specified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
