#!/usr/bin/env python3
"""Compare the early-dump output pixels (output/engine_sweep/early_pixels.txt)
to node_conv_246.goldout. Each line is "pixel_index <512-hex>" (2048-bit, big-
endian; channel 0 = LSByte, same convention as compare_engine_output.py).

For each captured pixel, byte-compares its first 256 channels to goldout[pixel].
PASS iff every captured pixel matches exactly.
"""
from __future__ import annotations
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EARLY = ROOT / "output/engine_sweep/early_pixels.txt"
GOLD = ROOT / "output/goldens/node_conv_246.goldout"
OC_BYTES = 256


def read_gold():
    raw = GOLD.read_bytes()
    magic, ver, nv, spv, bps = struct.unpack("<4sIIII", raw[:20])
    assert magic == b"NN2V" and bps == OC_BYTES, (magic, bps)
    body = raw[20:20 + spv * bps]
    return [body[p * bps:(p + 1) * bps] for p in range(spv)]


def main() -> int:
    gold = read_gold()
    lines = [l.strip() for l in EARLY.read_text().splitlines() if l.strip()]
    if not lines:
        print("FAIL: no early pixels captured"); return 1

    total = mism = max_err = 0
    bad_pixels = []
    for ln in lines:
        pix_s, hexs = ln.split()
        pix = int(pix_s)
        word = int(hexs, 16).to_bytes(OC_BYTES, "big")[::-1]  # channel 0 first
        g = gold[pix]
        diffs = []
        for c in range(OC_BYTES):
            o, e = word[c], g[c]
            os_, es_ = (o - 256 if o >= 128 else o), (e - 256 if e >= 128 else e)
            total += 1
            if o != e:
                mism += 1
                max_err = max(max_err, abs(os_ - es_))
                if len(diffs) < 4:
                    diffs.append((c, es_, os_))
        if diffs:
            bad_pixels.append((pix, diffs))

    n_pix = len(lines)
    print(f"early pixels captured: {n_pix}  (pixel idxs: {[int(l.split()[0]) for l in lines]})")
    if mism == 0:
        print(f"PASS: all {n_pix} gate-computed pixels byte-exact "
              f"({total} bytes, max_error=0, mismatch=0)")
        return 0
    print(f"FAIL: {mism}/{total} bytes differ across {len(bad_pixels)} pixels, "
          f"max|err|={max_err}")
    for pix, diffs in bad_pixels[:8]:
        print(f"  pixel {pix}: (ch,gold,got)={diffs}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
