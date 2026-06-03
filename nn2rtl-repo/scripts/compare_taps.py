#!/usr/bin/env python3
"""Compare passive [R23]/[R47] tap dumps (engine-bisect) to fresh 32-byte-tiled goldens.
[R23] = node_relu_23 (conv_246 FIRST engine conv output, golden tiled32_node_relu_23).
[R47] = node_relu_47 (conv_300 LAST engine conv input,  golden tiled32_node_relu_47).
%h is MSB-first -> reverse to byte0..byte31 channel order.

Usage: python scripts/compare_taps.py taps_dump.log
"""
from __future__ import annotations
import sys, struct, re
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "taps_dump.log"

TAPS = {"R23": "node_relu_23", "R47": "node_relu_47"}


def tiled_vec0(mid):
    raw = (ROOT / "output/goldens" / f"tiled32_{mid}.goldout").read_bytes()
    _, ver, nv, ns, bps = struct.unpack_from("<4sIIII", raw, 0)
    return np.frombuffer(raw[20:20 + ns * bps], dtype=np.uint8).reshape(ns, 32), ns


def hexrev(h):
    return np.frombuffer(bytes.fromhex(h.strip().rjust(64, "0")[-64:]), dtype=np.uint8)[::-1].copy()


def main():
    txt = LOG.read_text(errors="ignore")
    for tag, mid in TAPS.items():
        g, ns = tiled_vec0(mid)
        beats = []
        for m in re.finditer(rf"\[{tag}\]\s+(\d+)\s+([0-9a-fA-F]+)", txt):
            beats.append(hexrev(m.group(2)))
        if not beats:
            print(f"[{tag}] {mid}: NO dump lines found")
            continue
        n = min(len(beats), ns)
        d = np.array(beats[:n], dtype=np.int16); gg = g[:n].astype(np.int16)
        d[d > 127] -= 256; gg[gg > 127] -= 256
        diff = d - gg
        mm = int((diff != 0).sum())
        fb = int(np.where((diff != 0).any(axis=1))[0][0]) if mm else -1
        print(f"[{tag}] {mid}: parsed {len(beats)} beats (golden {ns}); "
              f"mismatch={mm}/{d.size} ({100*mm/d.size:.2f}%) +b={int((diff>0).sum())} -b={int((diff<0).sum())} "
              f"max|d|={int(np.abs(diff).max())} first_bad_beat={fb}")
    print("=> R23 wrong => engine corrupts from the FIRST engine conv (conv_246). "
          "R23 clean + R47 wrong => corruption accumulates in middle engine convs.")


if __name__ == "__main__":
    main()
