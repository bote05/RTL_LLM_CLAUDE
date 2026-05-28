#!/usr/bin/env python3
"""Reusable GPTQ inner routine (validated in scripts/gptq_int4.py — that flow
reached W4A8 77.5% top-1 on ResNet-50 V2). Returns the INTEGER quantized
weights so the golden generator can emit them as INT4 hex + a per-output-channel
scale, and the RTL can compute acc = int4_w * int8_act then requant per-OC.

per-output-channel static scale; blocked column-wise error compensation using
the layer input Hessian H = XᵀX (im2col of conv inputs over calibration).
"""
from __future__ import annotations
import torch


def per_oc_scale(W2d: torch.Tensor, qmax: int) -> torch.Tensor:
    """[OC,1] symmetric per-output-channel scale = max|row| / qmax."""
    return (W2d.abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-12)


def gptq_int_weights(W2d: torch.Tensor, H: torch.Tensor, s: torch.Tensor,
                     qmin: int, qmax: int, blocksize: int = 128,
                     damp: float = 0.01) -> torch.Tensor:
    """GPTQ-quantize W2d [OC,K] given input Hessian H [K,K] and per-OC scale
    s [OC,1]. Returns INTEGER weights [OC,K] in [qmin,qmax] (float-typed)."""
    dev = W2d.device
    W = W2d.clone().float()
    OC, K = W.shape
    Hm = H.to(dev).float().clone()
    d = damp * torch.diag(Hm).mean()
    idx = torch.arange(K, device=dev)
    Hm[idx, idx] += d
    dead = torch.diag(Hm) == 0
    Hm[dead, dead] = 1
    W[:, dead] = 0
    s1 = s.squeeze(1)  # [OC]

    L = torch.linalg.cholesky(Hm)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)  # upper-tri factor

    Qint = torch.zeros_like(W)
    for b0 in range(0, K, blocksize):
        b1 = min(b0 + blocksize, K)
        Wb = W[:, b0:b1].clone()
        Qb = torch.zeros_like(Wb)
        Eb = torch.zeros_like(Wb)
        Hb = Hinv[b0:b1, b0:b1]
        for j in range(b1 - b0):
            w = Wb[:, j]
            dinv = Hb[j, j]
            qint = torch.clamp(torch.round(w / s1), qmin, qmax)
            q = qint * s1
            Qb[:, j] = qint
            err = (w - q) / dinv
            Wb[:, j:] -= err.unsqueeze(1) * Hb[j, j:].unsqueeze(0)
            Eb[:, j] = err
        Qint[:, b0:b1] = Qb
        W[:, b1:] -= Eb @ Hinv[b0:b1, b1:]
    return Qint
