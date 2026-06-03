#!/usr/bin/env python3
"""Characterize the INT3 e2e mismatch: compare the dumped RTL relu_48 m_axis
(relu48_dump.bin) to the contract goldout, to reveal the BUG CLASS:
  - diffs all +-1  -> rounding/requant-bias
  - diffs large + RTL saturated (127/-128/0) -> overflow/saturation
  - RTL ~= golden * k -> scale-factor error
  - specific channels wrong -> per-OC scale/bias
Usage: python scripts/analyze_relu48_mismatch.py
"""
from __future__ import annotations
import sys, glob, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent


def read_bytes_signed(path: Path) -> list[int]:
    data = path.read_bytes()
    return [b - 256 if b > 127 else b for b in data]


def load_goldout() -> list[int]:
    # contract goldout: try raw bytes (the harness compares m_axis bytes to it)
    g = sorted(glob.glob(str(ROOT / "output/goldens/contracts/node_relu_48_*/node_relu_48.goldout")))
    if not g:
        raise SystemExit("no relu_48 goldout found")
    p = Path(g[0])
    raw = p.read_bytes()
    # goldout may have an ascii header line; detect: if it's hex-text vs binary
    txt = None
    try:
        txt = raw.decode("ascii")
    except Exception:
        pass
    if txt is not None and all(c in "0123456789abcdefABCDEF \n\r" for c in txt[:200]):
        # hex text, one byte per token
        vals = []
        for tok in txt.split():
            v = int(tok, 16)
            vals.append(v - 256 if v > 127 else v)
        return vals
    return [b - 256 if b > 127 else b for b in raw]


def main() -> int:
    dump = ROOT / "relu48_dump.bin"
    if not dump.exists():
        raise SystemExit("relu48_dump.bin not found (run the dump first)")
    rtl = read_bytes_signed(dump)
    gold = load_goldout()
    n = min(len(rtl), len(gold))
    print(f"rtl_bytes={len(rtl)} gold_bytes={len(gold)} compare_n={n}")
    mism = [(i, rtl[i], gold[i]) for i in range(n) if rtl[i] != gold[i]]
    print(f"mismatches: {len(mism)}/{n} ({100*len(mism)/n:.1f}%)  first@{mism[0][0] if mism else -1}")
    if not mism:
        print("NO MISMATCH in compared range")
        return 0
    # diff histogram
    diffs = collections.Counter(r - g for _, r, g in mism)
    print("top diff(rtl-gold) values:", diffs.most_common(10))
    # saturation: how many RTL values are at clamp edges or 0
    sat = collections.Counter()
    for _, r, g in mism:
        if r == 127: sat["rtl=+127"] += 1
        elif r == -128: sat["rtl=-128"] += 1
        elif r == 0: sat["rtl=0"] += 1
        if g == 0: sat["gold=0"] += 1
    print("saturation/zero among mismatches:", dict(sat))
    # ratio check (scale error): rtl/gold where gold!=0
    ratios = [r / g for _, r, g in mism if g != 0]
    if ratios:
        ratios.sort()
        print(f"rtl/gold ratio: median={ratios[len(ratios)//2]:.3f} min={ratios[0]:.3f} max={ratios[-1]:.3f}")
    # positional pattern: relu_48 is 256-ch tiled; show which byte-lanes (mod 32 / mod 256) dominate
    modch = collections.Counter(i % 256 for i, _, _ in mism)
    print("mismatch positions mod 256 (top channels):", modch.most_common(8))
    # magnitude
    big = [(i, r, g) for i, r, g in mism if abs(r - g) > 8]
    print(f"large diffs (>8): {len(big)}/{len(mism)}  examples:", [(i, r, g) for i, r, g in mism[:8]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
