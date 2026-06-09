#!/usr/bin/env python3
"""[RESNET 2953 ROOT-CAUSE] Per-output-channel ratio RTL/golden for relu_23
(conv_246, the FIRST engine dispatch, where corruption begins). If the engine
requant applied the wrong per-OC shift (over-widened scale.mem multiplier), the
RTL output is golden * 2^k with a CONSISTENT integer k per channel. This script
measures, per channel: median ratio, whether it is a clean power-of-2, and the
implied shift error -- pinning the bug to the scale memory.

relu_23 = 14x14x256 tiled-streaming: beat = pixel*8 + tile, channel = tile*32 + byte.

Usage: python scripts/analyze_relu23_perchannel_ratio.py
"""
from __future__ import annotations
import struct
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TAP = ROOT / "output" / "taps" / "node_relu_23.bin"
CONTRACTS = ROOT / "output" / "goldens" / "contracts"


def load_golden(node):
    for e in sorted(CONTRACTS.iterdir()):
        if e.name.startswith(node + "_"):
            raw = (e / f"{node}.goldout").read_bytes()
            magic, ver, nv, spv, bps = struct.unpack_from("<4sIIII", raw, 0)
            wps = (bps + 3) // 4
            w = np.frombuffer(raw[20:20 + spv * wps * 4], dtype="<u4").reshape(spv, wps)
            return w.view(np.uint8).reshape(spv, 32)
    raise SystemExit("no golden")


def main():
    g = load_golden("node_relu_23")
    rtl = np.frombuffer(TAP.read_bytes(), dtype=np.uint8)
    n = min(rtl.size // 32, g.shape[0]); rtl = rtl[:n*32].reshape(n,32); g = g[:n]
    rs = rtl.astype(np.float64); gs = g.astype(np.float64)
    # relu output is unsigned [0,127] (post-relu, no negatives) -> treat as-is 0..255 but
    # values are <=127 so no sign issue. tiles=8, pixels=n/8.
    tiles = 8; pixels = n // tiles
    rs = rs.reshape(pixels, tiles, 32); gs = gs.reshape(pixels, tiles, 32)
    # channel c = tile*32 + byte
    print(f"relu_23: pixels={pixels} tiles={tiles} channels=256")
    print(f"{'ch':>4} {'gold_nz':>7} {'med_ratio':>9} {'pow2?':>6} {'impl_k':>6} {'note'}")
    pow2_chs = 0; clean_chs = 0; sample = []
    k_hist = {}
    for c in range(256):
        t, b = divmod(c, 32)
        rv = rs[:, t, b]; gv = gs[:, t, b]
        mask = gv > 0
        nz = int(mask.sum())
        if nz < 3:
            continue
        ratio = rv[mask] / gv[mask]
        med = float(np.median(ratio))
        # implied shift error k = round(log2(med))
        k = round(np.log2(med)) if med > 0 else 0
        # clean power-of-2 if rv ~= gv * 2^k for most positions (within rounding +-1)
        pred = gv * (2.0 ** k)
        clean = np.mean(np.abs(rv - pred) <= 1.0)
        is_pow2 = abs(med - 2.0**k) < 0.25 * (2.0**k)
        if is_pow2: pow2_chs += 1
        if clean > 0.8: clean_chs += 1
        k_hist[k] = k_hist.get(k, 0) + 1
        if len(sample) < 25:
            sample.append((c, nz, med, is_pow2, k, f"clean@2^{k}={clean:.0%}"))
    for row in sample:
        print(f"{row[0]:>4} {row[1]:>7} {row[2]:>9.3f} {str(row[3]):>6} {row[4]:>6} {row[5]}")
    print(f"\nchannels with power-of-2 median ratio: {pow2_chs}/256")
    print(f"channels where RTL == golden*2^k within +-1 for >80% positions: {clean_chs}/256")
    print(f"implied shift-error k histogram (k = log2(RTL/golden)): {dict(sorted(k_hist.items()))}")
    # overall: if most channels are clean*2^k with k>0, it's a per-OC scale/shift inflation
    if clean_chs > 200:
        print("\n=> CONFIRMED: relu_23 = golden * 2^k per channel (k>0) -> ENGINE REQUANT SHIFT/WIDENING ERROR (scale.mem).")
    else:
        print("\n=> NOT a clean per-channel power-of-2 scale -> not a pure shift error; inspect weights/acc.")


if __name__ == "__main__":
    main()
