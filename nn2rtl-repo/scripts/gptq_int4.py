#!/usr/bin/env python3
"""Advanced INT4 weight-quantization FEASIBILITY study (Phase 1b).

Naive INT4 PTQ collapses ResNet-50 (per-tensor 0%, per-channel 32%). This tests
RTL-COMPATIBLE advanced methods that keep a single PER-OUTPUT-CHANNEL scale
(applied once post-accumulation, like the existing requant — no mid-accumulation
dequant, so the RTL rework stays bounded):

  - per-channel min-max            (baseline, ~32%)
  - per-channel + bias correction  (correct mean output error from calibration)
  - GPTQ per-output-channel        (Hessian error-compensated; the main hope)

All quantize every Conv2d in torchvision ResNet-50 V2 (the reference backbone;
the fc head is NOT in the RTL design so it stays float). Reports top-1 on val.
Optionally adds INT8 per-tensor activation fake-quant for the realistic W4A8
number (the deployable Scheme A').

  py scripts/gptq_int4.py [n_calib=256] [n_val=512]
"""
from __future__ import annotations
import sys
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import imagenet_util as iu  # noqa: E402

QMAX, QMIN = 7, -8  # INT4


def per_channel_scale(W2d: torch.Tensor) -> torch.Tensor:
    return (W2d.abs().amax(dim=1, keepdim=True) / QMAX).clamp_min(1e-12)


def quant_w(W2d, s):
    return torch.clamp(torch.round(W2d / s), QMIN, QMAX)


def convs(model):
    return [m for m in model.modules() if isinstance(m, nn.Conv2d)]


def eval_top1(model, xs, ys, dev, act_int8=False):
    model = model.to(dev).eval()
    handles = []
    if act_int8:
        # per-tensor INT8 fake-quant on every conv input (matches Scheme A act path)
        def mk(_m):
            def hook(mod, inp):
                x = inp[0]
                s = (x.abs().max() / 127.0).clamp_min(1e-12)
                return (torch.clamp(torch.round(x / s), -128, 127) * s,)
            return hook
        for m in convs(model):
            handles.append(m.register_forward_pre_hook(mk(m)))
    top1 = 0
    bs = 128
    with torch.no_grad():
        for s in range(0, len(ys), bs):
            xb = torch.from_numpy(xs[s:s+bs]).to(dev)
            p = model(xb).argmax(1).cpu().numpy()
            top1 += int((p == ys[s:s+bs]).sum())
    for h in handles:
        h.remove()
    return top1 / len(ys)


def quant_perchannel(base, dev):
    m = copy.deepcopy(base).to(dev).eval()
    with torch.no_grad():
        for c in convs(m):
            W = c.weight.data
            W2d = W.reshape(W.shape[0], -1)
            s = per_channel_scale(W2d)
            c.weight.copy_((quant_w(W2d, s) * s).reshape(W.shape))
    return m


def quant_perchannel_biascorr(base, xs_cal, dev):
    """Per-channel INT4 + bias correction: add delta_bias = mean over calib of
    (float_out - quant_out) per output channel, folded into the conv bias.
    Accumulates the per-channel output-error sum IN the hook (memory-light)."""
    base = base.to(dev).eval()
    m = copy.deepcopy(base).eval()
    cs, bcs = convs(m), convs(base)
    with torch.no_grad():
        for c, bc in zip(cs, bcs):
            W = bc.weight.data
            W2d = W.reshape(W.shape[0], -1)
            s = per_channel_scale(W2d)
            c.weight.copy_((quant_w(W2d, s) * s).reshape(W.shape))
    dsum = [None] * len(cs)
    dcnt = [0] * len(cs)

    def mk(i):
        def hook(mod, inp):
            x = inp[0]
            of = Fnn.conv2d(x, bcs[i].weight, None, mod.stride, mod.padding, mod.dilation, mod.groups)
            oq = Fnn.conv2d(x, cs[i].weight, None, mod.stride, mod.padding, mod.dilation, mod.groups)
            d = (of - oq).sum(dim=(0, 2, 3))
            dsum[i] = d if dsum[i] is None else dsum[i] + d
            dcnt[i] += of.shape[0] * of.shape[2] * of.shape[3]
        return hook
    handles = [bcs[i].register_forward_pre_hook(mk(i)) for i in range(len(bcs))]
    with torch.no_grad():
        bs = 64
        for s0 in range(0, xs_cal.shape[0], bs):
            _ = base(torch.from_numpy(xs_cal[s0:s0+bs]).to(dev))
    for h in handles:
        h.remove()
    with torch.no_grad():
        for i, c in enumerate(cs):
            delta = dsum[i] / max(1, dcnt[i])
            if c.bias is not None:
                c.bias.add_(delta)
            else:
                c.bias = nn.Parameter(delta)
    return m


def _gptq_scale(W2d, mode):
    if mode == "tensor":
        sc = (W2d.abs().max() / QMAX).clamp_min(1e-12)
        return sc.expand(W2d.shape[0], 1).contiguous()  # [OC,1] all equal
    return per_channel_scale(W2d)                        # [OC,1] per output channel


def gptq_quant(base, xs_cal, dev, blocksize=128, damp=0.01, scale_mode="channel"):
    """GPTQ INT4 on every Conv2d. scale_mode: 'channel' (per-output-channel,
    needs per-channel requant in RTL) or 'tensor' (single scale/layer, keeps the
    existing per-tensor requant = Scheme A intact). Non-sequential variant:
    Hessian H=X^T X per layer from the FLOAT model's calibration activations."""
    m = copy.deepcopy(base).to(dev).eval()
    cs = convs(m)
    # 1) accumulate per-layer Hessian via hooks (one float calib pass)
    H = [None] * len(cs)
    cnt = [0] * len(cs)
    handles = []
    def mk(i):
        def hook(mod, inp):
            x = inp[0]
            if isinstance(mod, nn.Conv2d):
                xu = Fnn.unfold(x, mod.kernel_size, mod.dilation, mod.padding, mod.stride)
                # xu: [N, K, L] -> [N*L, K]
                xu = xu.transpose(1, 2).reshape(-1, xu.shape[1]).float()
            else:
                xu = x.reshape(-1, x.shape[-1]).float()
            h = (xu.t() @ xu).cpu()   # keep Hessians on CPU to avoid GPU OOM
            if H[i] is None:
                H[i] = h
            else:
                H[i] += h
            cnt[i] += xu.shape[0]
        return hook
    for i, c in enumerate(cs):
        handles.append(c.register_forward_pre_hook(mk(i)))
    with torch.no_grad():
        bs = 64
        for s in range(0, xs_cal.shape[0], bs):
            _ = m(torch.from_numpy(xs_cal[s:s+bs]).to(dev))
    for h in handles:
        h.remove()

    # 2) GPTQ per layer
    with torch.no_grad():
        for i, c in enumerate(cs):
            W = c.weight.data.clone()
            OC = W.shape[0]
            W2d = W.reshape(OC, -1).float()         # [OC, K]
            K = W2d.shape[1]
            Hm = (H[i] / max(1, cnt[i])).to(dev)    # move this layer's Hessian to GPU
            H[i] = None                             # free CPU copy
            d = damp * torch.diag(Hm).mean()
            diagidx = torch.arange(K, device=dev)
            Hm[diagidx, diagidx] += d
            # dead columns
            dead = torch.diag(Hm) == 0
            Hm[dead, dead] = 1
            W2d[:, dead] = 0
            # static scale (from original W): per-channel or per-tensor
            s = _gptq_scale(W2d, scale_mode)        # [OC,1]
            # Hinv (upper Cholesky factor of inverse)
            L = torch.linalg.cholesky(Hm)
            Hinv = torch.cholesky_inverse(L)
            Hinv = torch.linalg.cholesky(Hinv, upper=True)
            Q = torch.zeros_like(W2d)
            for b0 in range(0, K, blocksize):
                b1 = min(b0 + blocksize, K)
                Wb = W2d[:, b0:b1].clone()
                Qb = torch.zeros_like(Wb)
                Eb = torch.zeros_like(Wb)
                Hb = Hinv[b0:b1, b0:b1]
                for j in range(b1 - b0):
                    w = Wb[:, j]
                    dinv = Hb[j, j]
                    q = torch.clamp(torch.round(w / s.squeeze(1)), QMIN, QMAX) * s.squeeze(1)
                    Qb[:, j] = q
                    err = (w - q) / dinv
                    Wb[:, j:] -= err.unsqueeze(1) * Hb[j, j:].unsqueeze(0)
                    Eb[:, j] = err
                Q[:, b0:b1] = Qb
                W2d[:, b1:] -= Eb @ Hinv[b0:b1, b1:]
            c.weight.copy_(Q.reshape(W.shape).to(c.weight.dtype))
    return m


def main() -> int:
    n_calib = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    n_val = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from torchvision.models import resnet50, ResNet50_Weights
    base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    xcal, _ = iu.load_batch(n_calib)
    xval, yval = iu.load_batch(n_val, skip=n_calib)
    print(f"GPTQ INT4 feasibility: n_calib={n_calib} n_val={n_val} dev={dev}")

    mode = sys.argv[3] if len(sys.argv) > 3 else "all"
    print(f"  float                         : {eval_top1(base, xval, yval, dev)*100:.2f}%")
    if mode in ("all", "tensor"):
        mt = gptq_quant(base, xcal, dev, scale_mode="tensor")
        print(f"  INT4 GPTQ per-TENSOR (w-only) : {eval_top1(mt, xval, yval, dev)*100:.2f}%")
        print(f"  INT4 GPTQ per-TENSOR + A8     : {eval_top1(mt, xval, yval, dev, act_int8=True)*100:.2f}%   <-- Scheme A intact if good")
        del mt
    if mode in ("all", "channel"):
        if mode == "all":
            print(f"  INT4 per-channel (naive)      : {eval_top1(quant_perchannel(base, dev), xval, yval, dev)*100:.2f}%")
            print(f"  INT4 per-channel + bias-corr  : {eval_top1(quant_perchannel_biascorr(base, xcal, dev), xval, yval, dev)*100:.2f}%")
        mg = gptq_quant(base, xcal, dev, scale_mode="channel")
        print(f"  INT4 GPTQ per-CHANNEL (w-only): {eval_top1(mg, xval, yval, dev)*100:.2f}%")
        print(f"  INT4 GPTQ per-CHANNEL + A8    : {eval_top1(mg, xval, yval, dev, act_int8=True)*100:.2f}%   <-- per-channel requant rework")
    print("\n=> if per-TENSOR GPTQ is usable (~74%+), keep per-tensor requant (Scheme A).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
