#!/usr/bin/env python3
"""Triangulate conv_248: recompute its output in Python from conv_246's
BYTE-EXACT captured output (the in-chain probe), using the SAME restored INT8
weights the RTL uses, and compare to (a) conv_248's CAPTURED output and (b) its
contract goldout. Decides:
  my==captured, my!=goldout  -> conv_248 RTL is CORRECT; goldout is STALE (like
                                the proven-stale goldins) => not a real bug.
  my!=captured               -> conv_248 RTL has a real compute error.
Uses data already on disk (no probe re-run). conv_246/248 are 1x1-spatial-equiv
196px streams; conv_248: 1x1 IC=256 OC=1024.
"""
from __future__ import annotations
import struct, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from golden_impl import compute_scale_approx  # noqa: E402

DUMP = ROOT / "output/reports_integrated/verilator_nn2rtl_top_probe"
import json
IR = json.load(open(ROOT / "backups/phase1_20260528T175041/layer_ir.json"))
L = {l["module_id"]: l for l in IR["layers"]}


def load_hex_i8(p):
    return np.array([int(x, 16) for x in Path(p).read_text().split()], dtype=np.int32).astype(np.int8).astype(np.int32) \
        if Path(p).exists() else None


def load_i8(p, n):
    b = Path(p).read_bytes()
    return np.frombuffer(b[:n], dtype=np.int8).astype(np.int32)


def requant(acc, scale):
    mult, shift = compute_scale_approx(float(scale))
    rb = 1 << (shift - 1)
    out = (acc * mult + rb) >> shift
    return np.clip(out, -128, 127)


def main():
    NPIX = 196
    # conv_246 captured output (byte-exact) = [196,256] int8
    c246 = load_i8(DUMP / "probe_node_conv_246.bin", NPIX * 256).reshape(NPIX, 256)
    # relu_23 (unbounded relu, scale 1): conv_248 input = max(0, c246)
    inp = np.maximum(0, c246)                      # [196,256]
    # conv_248 weights [OC=1024, IC=256] (1x1), bias int32 [1024]
    w = load_hex_i8(ROOT / "output/weights/node_conv_248_weights.hex")
    if w is None:
        print("no conv_248 weights"); return 1
    w = w.reshape(1024, 256)
    bpath = ROOT / "output/weights/node_conv_248_bias.hex"
    def _i32(x):
        v = int(x, 16)
        return v - (1 << 32) if v >= (1 << 31) else v   # big-endian signed int32 hex
    bias = np.array([_i32(x) for x in bpath.read_text().split()],
                    dtype=np.int64) if bpath.exists() else np.zeros(1024, dtype=np.int64)
    scale = L["node_conv_248"]["scale_factor"]
    print(f"conv_248 scale={scale} -> (mult,shift)={compute_scale_approx(scale)}; bias[:3]={bias[:3].tolist()}")

    # 1x1 conv: acc[px,oc] = sum_ic inp[px,ic]*w[oc,ic] + bias[oc]
    acc = inp.astype(np.int64) @ w.T.astype(np.int64) + bias[None, :]   # [196,1024]
    myout = requant(acc, scale).astype(np.int8)                          # [196,1024]

    cap = load_i8(DUMP / "probe_node_conv_248.bin", NPIX * 1024).reshape(NPIX, 1024).astype(np.int8)
    # goldout (contract)
    g = list(ROOT.glob("output/goldens/contracts/node_conv_248_*/node_conv_248.goldout"))
    gold = None
    if g:
        raw = g[0].read_bytes(); _, _, nv, spv, bps = struct.unpack("<4sIIII", raw[:20])
        gold = np.frombuffer(raw[20:20 + spv * bps], dtype=np.int8).reshape(spv, bps)[:, :1024] if bps >= 1024 \
            else np.frombuffer(raw[20:20 + spv * bps], dtype=np.int8).reshape(spv, -1)

    def cmp(a, b, name):
        n = min(a.size, b.size); a = a.reshape(-1)[:n].astype(np.int32); b = b.reshape(-1)[:n].astype(np.int32)
        d = np.abs(a - b); print(f"  my vs {name:10s}: mismatch={int((d!=0).sum())}/{n} ({(d!=0).mean()*100:.1f}%) max|err|={int(d.max())}")

    print("recompute conv_248 from conv_246-captured (byte-exact) input:")
    cmp(myout, cap, "CAPTURED")
    if gold is not None:
        cmp(myout, gold, "goldout")
        cmp(cap, gold, "(cap vs gold)")
    print("\n=> my==CAPTURED & my!=goldout: conv_248 RTL CORRECT, goldout STALE.\n"
          "   my!=CAPTURED: real conv_248 compute bug.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
