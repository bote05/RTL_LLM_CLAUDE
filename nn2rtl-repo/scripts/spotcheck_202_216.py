#!/usr/bin/env python3
"""Spot-check conv_202 and conv_216 (both 1x1 pad0 stride1, IC=64 OC=256, 56x56) for
golden-vs-deployed-weight consistency. Recompute output from goldin + PLAIN weights
(multiset==mp_k file the RTL reads) + per-OC requant, compare to goldout."""
from __future__ import annotations
import struct, sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from golden_impl import compute_scale_approx  # noqa: E402

L = {l["module_id"]: l for l in json.load(open(ROOT / "output/layer_ir.json"))["layers"]}
IC, OC, IH, IW = 64, 256, 56, 56


def load_plain_i8(p):
    return np.array([int(x, 16) for x in Path(p).read_text().split()], dtype=np.int32).astype(np.int8).astype(np.int32)


def i32(x):
    v = int(x, 16)
    return v - (1 << 32) if v >= (1 << 31) else v


def frame0(mid, tag):
    raw = (ROOT / f"output/goldens/{mid}.{tag}").read_bytes()
    _, nv, _, spv, bps = struct.unpack("<4sIIII", raw[:20])
    return np.frombuffer(raw[20:20 + spv * bps], dtype=np.int8).astype(np.int32)


def run(num):
    MID = f"node_conv_{num}"
    inp = frame0(MID, "goldin").reshape(IH * IW, IC)            # [3136,64] HWC
    gold = frame0(MID, "goldout").reshape(IH * IW, OC)          # [3136,256] HWC
    w = load_plain_i8(ROOT / f"output/weights/{MID}_weights.hex")
    assert w.size == OC * IC, f"weights size {w.size} != {OC*IC}"
    w = w.reshape(OC, IC)                                       # [oc,ic] (1x1)
    bp = ROOT / f"output/weights/{MID}_bias.hex"
    bias = np.array([i32(x) for x in bp.read_text().split()], dtype=np.int64) if bp.exists() else np.zeros(OC, np.int64)
    spo = L[MID].get("scale_factor_per_oc")

    # 1x1 pad0 stride1: out[p,oc] = sum_ic inp[p,ic]*w[oc,ic] + bias[oc]
    acc = inp.astype(np.int64) @ w.T.astype(np.int64) + bias[None, :]   # [3136,256]

    def requant(a, scale):
        mult, shift = compute_scale_approx(float(scale))
        rnd = 0 if shift == 0 else (1 << (shift - 1))   # RTL: shift==0 -> no round bias
        return np.clip((a * mult + rnd) >> shift, -128, 127)

    my = np.zeros((IH * IW, OC), np.int32)
    if spo:
        for oc in range(OC):
            my[:, oc] = requant(acc[:, oc], spo[oc])
    else:
        my = requant(acc, L[MID]["scale_factor"])

    d = np.abs(my - gold)
    mismatch = int((d != 0).sum())
    total = int(my.size)
    maxerr = int(d.max())
    ms = bool(np.array_equal(np.sort(my.ravel()), np.sort(gold.ravel())))
    print(f"=== conv_{num} (1x1 pad0 stride1 IC=64 OC=256) ===")
    print(f"  weights int4={w.min()>=-8 and w.max()<=7} range[{w.min()},{w.max()}] per_oc_scale={'YES' if spo else 'NO'}")
    print(f"  mismatch={mismatch}/{total} ({mismatch/total*100:.4f}%) max|err|={maxerr}")
    print(f"  multiset(my)==multiset(gold): {ms}")
    return dict(num=num, mismatch=mismatch, total=total, maxerr=maxerr, ms=ms,
                int4=bool(w.min() >= -8 and w.max() <= 7), consistent=(mismatch == 0))


if __name__ == "__main__":
    r202 = run("202")
    print()
    r216 = run("216")
