#!/usr/bin/env python3
"""[RESNET LOCALIZER] Structural analysis of ANY tapped node vs its contract
golden. Same evidence dump as analyze_relu48_mismatch.py but parametrized.

Usage: python scripts/analyze_node_mismatch.py node_relu_23
"""
from __future__ import annotations
import sys, struct
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NODE = sys.argv[1] if len(sys.argv) > 1 else "node_relu_23"
TAP = ROOT / "output" / "taps" / f"{NODE}.bin"
CONTRACTS = ROOT / "output" / "goldens" / "contracts"


def load_golden(node):
    for e in sorted(CONTRACTS.iterdir()):
        if e.name.startswith(node + "_"):
            raw = (e / f"{node}.goldout").read_bytes()
            magic, ver, nv, spv, bps = struct.unpack_from("<4sIIII", raw, 0)
            assert magic == b"NN2V" and bps == 32
            wps = (bps + 3) // 4
            w = np.frombuffer(raw[20:20 + spv * wps * 4], dtype="<u4").reshape(spv, wps)
            return w.view(np.uint8).reshape(spv, 32)
    raise SystemExit(f"no golden for {node}")


def main():
    g = load_golden(NODE)
    rtl = np.frombuffer(TAP.read_bytes(), dtype=np.uint8)
    n = min(rtl.size // 32, g.shape[0])
    rtl = rtl[:n * 32].reshape(n, 32); g = g[:n]
    rs = rtl.astype(np.int16); rs[rs > 127] -= 256
    gs = g.astype(np.int16); gs[gs > 127] -= 256
    diff = rs - gs
    bad = diff != 0
    nbad = int(bad.sum())
    print(f"[{NODE}] beats={n} mismatch={nbad}/{rtl.size} ({100*nbad/rtl.size:.2f}%)")
    if not nbad:
        print("BYTE-EXACT"); return
    print(f"  sign: +{int((diff>0).sum())} / -{int((diff<0).sum())}   max|err|={int(np.abs(diff).max())} mean|err|(bad)={np.abs(diff[bad]).mean():.1f}")
    print(f"  RTL==0&bad: {int(((rs==0)&bad).sum())} ({100*int(((rs==0)&bad).sum())/nbad:.0f}%)   golden==0&bad: {int(((gs==0)&bad).sum())} ({100*int(((gs==0)&bad).sum())/nbad:.0f}%)   both!=0&bad: {int(((rs!=0)&(gs!=0)&bad).sum())}")
    mags = np.abs(diff[bad])
    print("  |err| hist:", {f"{lo}-{hi}": int(((mags>=lo)&(mags<=hi)).sum()) for lo,hi in [(1,1),(2,2),(3,4),(5,8),(9,16),(17,32),(33,64),(65,127)] if ((mags>=lo)&(mags<=hi)).sum()})
    # per-channel (within tile, 32 bytes -> but node has C channels = (n/pixels)*32)
    bp = bad.sum(axis=0)
    print("  per-byte-lane (0..31) bad counts:", {i:int(bp[i]) for i in range(32)})
    # per-beat
    beat_bad = bad.sum(axis=1)
    print(f"  beats with >=1 bad: {int((beat_bad>0).sum())}/{n}  first_bad_beat={int(np.where(beat_bad>0)[0][0])}")
    print("  bad-bytes-per-beat hist:", {int(v):int(c) for v,c in zip(*np.unique(beat_bad,return_counts=True))})
    # is RTL a shifted/rolled version of golden? check if rtl[beat] == golden[beat+k] for small k
    print("\n  --- misalignment probe: does RTL match a SHIFTED golden? ---")
    for shift in [-2,-1,1,2]:
        if abs(shift) >= n: continue
        a = rs[max(0,shift):n+min(0,shift)]
        b = gs[max(0,-shift):n+min(0,-shift)]
        m = int((a != b).sum())
        print(f"   beat-shift {shift:+d}: mismatch {m}/{a.size} ({100*m/a.size:.1f}%)")
    # within-beat byte rotation probe (tile/channel order swap)
    print("  --- byte-rotation probe (channel reorder within 32B tile) ---")
    for rot in [1, 2, 4, 8, 16]:
        rolled = np.roll(gs, rot, axis=1)
        m = int((rs != rolled).sum())
        print(f"   byte-roll {rot:>2}: mismatch {m}/{rs.size} ({100*m/rs.size:.1f}%)")
    # first few beats raw
    print("\n  first bad beat raw (RTL vs golden, int8):")
    fb = int(np.where(beat_bad>0)[0][0])
    print("   RTL :", rs[fb].tolist())
    print("   GOLD:", gs[fb].tolist())


if __name__ == "__main__":
    main()
