#!/usr/bin/env python3
"""ORDER-INVARIANT in-chain localization via SATURATION COUNT.

The probe streaming taps can't be value-compared (chain emission order != contract
golden order). BUT the COUNT of saturated bytes (==127 or ==-128) in a layer's
output is order-invariant. The golden for a correct layer has some baseline
saturation count; if the in-chain capture has SUBSTANTIALLY MORE saturated bytes
than its golden, that layer (or an upstream one) injected spurious saturation.
Walking in chain order, the FIRST layer whose excess-saturation jumps localizes
the in-chain saturation bug (which feeds the engine garbage -> relu_48 +127 error).

Compares each probe_<id>.bin saturation fraction to the layer's LOGICAL golden
(output/goldens/<id>.goldout) saturation fraction.

Usage: python scripts/localize_saturation.py [probe_dir]
"""
from __future__ import annotations
import struct, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROBE = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    ROOT / "output/reports_integrated/verilator_nn2rtl_top_probe"
GOLD = ROOT / "output/goldens"

CHAIN = ["node_conv_196", "node_max_pool2d", "node_add"] + \
        [f"node_add_{i}" for i in range(1, 16)] + ["node_relu_48"]


def sat_frac(arr: np.ndarray) -> tuple[int, int, float]:
    n127 = int((arr == 127).sum())
    nm128 = int((arr == -128).sum())
    return n127, nm128, (n127 + nm128) / arr.size * 100 if arr.size else 0.0


def golden_arr(mid: str):
    p = GOLD / f"{mid}.goldout"
    if not p.exists():
        return None
    raw = p.read_bytes()
    _, _, _, spv, bps = struct.unpack("<4sIIII", raw[:20])
    return np.frombuffer(raw[20:20 + spv * bps], dtype=np.int8)


def main() -> int:
    print(f"probe = {PROBE}")
    print(f"{'checkpoint':<18}{'cap %sat':>9}{'gold %sat':>10}{'cap 127':>9}{'gold 127':>9}  flag")
    print("-" * 74)
    first = None
    for mid in CHAIN:
        pb = PROBE / f"probe_{mid}.bin"
        g = golden_arr(mid)
        if not pb.exists() or g is None:
            print(f"{mid:<18}{'--':>9}{'--':>10}  (missing probe/golden)")
            continue
        cap = np.frombuffer(pb.read_bytes(), dtype=np.int8)
        c127, cm128, cpct = sat_frac(cap)
        g127, gm128, gpct = sat_frac(g)
        # Excess saturation: cap has materially more saturated bytes than golden.
        excess = cpct - gpct
        flag = ""
        if excess > 1.0 and first is None:  # >1% spurious saturation
            first = mid
            flag = f"<== SATURATION JUMP (+{excess:.1f}%)"
        elif excess > 1.0:
            flag = f"(+{excess:.1f}%)"
        print(f"{mid:<18}{cpct:>8.2f}%{gpct:>9.2f}%{c127:>9}{g127:>9}  {flag}")
    print("-" * 74)
    print(f"FIRST SATURATION JUMP: {first or 'NONE (no layer injects >1% spurious saturation)'}")
    print("Note: order-invariant (counts only). Localizes WHERE spurious saturation first appears.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
