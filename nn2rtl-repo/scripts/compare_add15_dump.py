#!/usr/bin/env python3
"""Parse the [ADD15] operand dump (engine conv_300 + skip relu_45 per add_15 accept
beat) and compare each operand to its FRESH 32-byte-tiled golden, to localize
whether the engine operand or the skip operand carries the positive bias.

[ADD15] <n> <ENG_hex_256b> <SKIP_hex_256b>  (ENG = node_conv_300_data_out,
SKIP = node_add_15_skip_data; %h is MSB-first so byte31..byte0 -> reverse to
byte0..byte31 = channel order matching the tiled golden).

Usage: python scripts/compare_add15_dump.py add15_dump.log
"""
from __future__ import annotations
import sys, struct, re
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "add15_dump.log"


def tiled_vec0(mid):
    raw = (ROOT / "output/goldens" / f"tiled32_{mid}.goldout").read_bytes()
    m, ver, nv, ns, bps = struct.unpack_from("<4sIIII", raw, 0)
    assert bps == 32, bps
    return np.frombuffer(raw[20:20 + ns * bps], dtype=np.uint8).reshape(ns, 32), ns


def hex256_to_bytes(h):
    h = h.strip().rjust(64, "0")[-64:]
    b = bytes.fromhex(h)              # MSB-first: byte31..byte0
    return np.frombuffer(b, dtype=np.uint8)[::-1].copy()   # -> byte0..byte31 (ch order)


def main():
    eng_g, ns = tiled_vec0("node_conv_300")
    skip_g, _ = tiled_vec0("node_relu_45")
    eng_d, skip_d = [], []
    pat = re.compile(r"\[ADD15\]\s+(\d+)\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)")
    for line in LOG.read_text(errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            eng_d.append(hex256_to_bytes(m.group(2)))
            skip_d.append(hex256_to_bytes(m.group(3)))
    print(f"parsed {len(eng_d)} ADD15 beats; golden vec0 = {ns} beats")
    if not eng_d:
        print("NO [ADD15] lines found (build/run may have failed or define missing)")
        return
    n = min(len(eng_d), ns)
    eng_d = np.array(eng_d[:n], dtype=np.int16); skip_d = np.array(skip_d[:n], dtype=np.int16)
    eng_g = eng_g[:n].astype(np.int16); skip_g = skip_g[:n].astype(np.int16)
    # treat as signed int8
    for a in (eng_d, skip_d, eng_g, skip_g):
        a[a > 127] -= 256
    for name, d, g in [("ENGINE(conv_300)", eng_d, eng_g), ("SKIP(relu_45)", skip_d, skip_g)]:
        diff = d - g
        mm = int((diff != 0).sum())
        pos = int((diff > 0).sum()); neg = int((diff < 0).sum())
        print(f"{name}: mismatch={mm}/{d.size} ({100*mm/d.size:.2f}%)  +bias={pos} -bias={neg}  "
              f"max|d|={int(np.abs(diff).max())}  first_mismatch_beat={int(np.where((diff!=0).any(axis=1))[0][0]) if mm else -1}")
    print("=> The operand with the large mismatch is where the bug enters "
          "(engine datapath if ENGINE, residual/skip path if SKIP).")


if __name__ == "__main__":
    main()
