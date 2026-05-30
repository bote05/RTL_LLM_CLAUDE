#!/usr/bin/env python3
"""Triangulate conv_200 (first 3x3 windowed conv) — recompute its output in Python
from its OWN goldin (contract input = relu_1 output), with 3x3 pad-1 windowing and
PER-OC requant, then compare to:
  (a) conv_200 goldout (contract)  -> if my==goldout, golden is self-consistent + my
                                       recompute convention is correct.
  (b) conv_200 in-chain cap (probe) MULTISET -> the RTL's actual output values.

Resolves the puzzle: conv_200 in-chain multiset is 94% DIFFER vs golden, yet relu_48
(final) is only ~1% DIFFER. Either the golden is wrong (my!=goldout) or the cap is a
tap artifact (cap multiset == goldout multiset) or the RTL line_buf is really wrong.
"""
from __future__ import annotations
import struct, sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from golden_impl import compute_scale_approx  # noqa: E402

L = {l["module_id"]: l for l in json.load(open(ROOT / "output/layer_ir.json"))["layers"]}
DUMP = ROOT / "output/reports_integrated/verilator_nn2rtl_top_probe"
MID = "node_conv_200"
IC, OC, IH, IW = 64, 64, 56, 56


def load_hex_i8(p):
    return np.array([int(x, 16) for x in Path(p).read_text().split()], dtype=np.int32).astype(np.int8).astype(np.int32)


def i32(x):
    v = int(x, 16)
    return v - (1 << 32) if v >= (1 << 31) else v


def frame0(tag):
    raw = (ROOT / f"output/goldens/{MID}.{tag}").read_bytes()
    _, nv, _, spv, bps = struct.unpack("<4sIIII", raw[:20])
    return np.frombuffer(raw[20:20 + spv * bps], dtype=np.int8).astype(np.int32)  # frame 0


def main():
    inp = frame0("goldin").reshape(IH, IW, IC)              # [56,56,64] HWC (relu_1 output, >=0)
    gold = frame0("goldout").reshape(IH * IW, OC)            # [3136,64] HWC
    w = load_hex_i8(ROOT / "output/weights/node_conv_200_weights.hex")
    assert w.size == OC * IC * 9, f"weights size {w.size} != {OC*IC*9}"
    w = w.reshape(OC, IC, 3, 3)                              # [oc,ic,kh,kw]
    bp = ROOT / "output/weights/node_conv_200_bias.hex"
    bias = np.array([i32(x) for x in bp.read_text().split()], dtype=np.int64) if bp.exists() else np.zeros(OC, np.int64)
    spo = L[MID].get("scale_factor_per_oc")
    print(f"inp range[{inp.min()},{inp.max()}] (relu'd>=0? {inp.min()>=0}); weights range[{w.min()},{w.max()}] "
          f"(int4? {w.min()>=-8 and w.max()<=7}); bias[:3]={bias[:3].tolist()}; per_oc_scale={'YES' if spo else 'NO (tensor)'}")

    pad = np.zeros((IH + 2, IW + 2, IC), np.int64); pad[1:-1, 1:-1, :] = inp
    acc = np.zeros((IH, IW, OC), np.int64)
    for kh in range(3):
        for kw in range(3):
            acc += pad[kh:kh + IH, kw:kw + IW, :].reshape(-1, IC) @ w[:, :, kh, kw].T.astype(np.int64) \
                .reshape(IC, OC) if False else \
                (pad[kh:kh + IH, kw:kw + IW, :] @ w[:, :, kh, kw].T.astype(np.int64))
    acc = acc.reshape(IH * IW, OC) + bias[None, :]

    my = np.zeros((IH * IW, OC), np.int32)
    if spo:
        for oc in range(OC):
            mult, shift = compute_scale_approx(float(spo[oc]))
            my[:, oc] = np.clip((acc[:, oc] * mult + (1 << (shift - 1))) >> shift, -128, 127)
    else:
        mult, shift = compute_scale_approx(float(L[MID]["scale_factor"]))
        my = np.clip((acc * mult + (1 << (shift - 1))) >> shift, -128, 127)

    d = np.abs(my - gold)
    print(f"\nmy vs GOLDOUT (per-position, HWC): mismatch={int((d!=0).sum())}/{my.size} "
          f"({(d!=0).mean()*100:.2f}%) max|err|={int(d.max())}")
    print(f"  my:   range[{my.min()},{my.max()}] mean={my.mean():.3f} #zero={int((my==0).sum())}")
    print(f"  gold: range[{gold.min()},{gold.max()}] mean={gold.mean():.3f} #zero={int((gold==0).sum())}")
    ms_my = np.array_equal(np.sort(my.ravel()), np.sort(gold.ravel()))
    print(f"  multiset(my)==multiset(gold): {ms_my}")

    capf = DUMP / f"probe_{MID}.bin"
    if capf.exists():
        cap = np.frombuffer(capf.read_bytes(), dtype=np.int8).astype(np.int32)
        print(f"\nin-chain CAP: n={cap.size} range[{cap.min()},{cap.max()}] mean={cap.mean():.3f} #zero={int((cap==0).sum())}")
        print(f"  multiset(cap)==multiset(gold): {np.array_equal(np.sort(cap), np.sort(gold.ravel()))}")
        print(f"  multiset(cap)==multiset(my):   {np.array_equal(np.sort(cap), np.sort(my.ravel()))}")
    print("\nVERDICT:")
    print("  my==goldout            -> golden correct; in-chain cap (94% off) = REAL RTL line_buf bug.")
    print("  my!=goldout (conv math)-> my convention wrong OR golden uses zero-point/diff requant.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
