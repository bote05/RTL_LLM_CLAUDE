#!/usr/bin/env python3
"""Compare the engine's RAW in-chain output for dispatch 0 (conv_246) to
node_conv_246.goldout. This tests the engine's FULL in-chain computation
(activation read + weight read + MAC + per-OC requant), address-keyed so it is
immune to the streaming-order issue that breaks the probe's streaming taps.

probe dump engreads_engout.bin: repeated records of
  uint32 addr (LE) + 256 bytes (= 256 int8 output channels, channel 0 = LSByte).
addr = act_out_base + output_pixel  ->  pixel = addr - min(addr).

conv_246.goldout (NN2V): vec0 = [196 pixels (14x14), 256 channels] int8.

PASS iff every captured engine output pixel matches goldout[pixel] byte-exact.

Usage: python scripts/compare_engout_d0.py [engout.bin] [conv_246.goldout]
"""
from __future__ import annotations
import struct, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ENGOUT = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    ROOT / "output/reports_integrated/verilator_nn2rtl_top_probe/engreads_engout.bin"
GOLD = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "output/goldens/node_conv_246.goldout"
OC = 256


def load_engout(p: Path):
    raw = p.read_bytes()
    rec = 4 + 256
    n = len(raw) // rec
    out = {}
    for i in range(n):
        off = i * rec
        addr = struct.unpack_from("<I", raw, off)[0]
        data = np.frombuffer(raw[off + 4:off + 4 + 256], dtype=np.int8)
        out[addr] = data
    return out


def load_goldout(p: Path):
    raw = p.read_bytes()
    magic, ver, nv, spv, bps = struct.unpack("<4sIIII", raw[:20])
    assert magic == b"NN2V" and bps == OC, (magic, bps)
    body = np.frombuffer(raw[20:20 + spv * bps], dtype=np.int8).reshape(spv, bps)
    return body  # [pixels, 256]


def main() -> int:
    eng = load_engout(ENGOUT)
    gold = load_goldout(GOLD)
    if not eng:
        print(f"no engout records in {ENGOUT}"); return 2
    base = min(eng)
    print(f"engout: {len(eng)} pixels captured (addr base={base}); goldout pixels={gold.shape[0]}")
    tot = bad_px = bad_bytes = 0
    maxerr = 0
    examples = []
    for addr in sorted(eng):
        px = addr - base
        if px >= gold.shape[0]:
            continue
        got = eng[addr].astype(np.int16)
        exp = gold[px].astype(np.int16)
        d = np.abs(got - exp)
        nz = int((d != 0).sum())
        tot += 1
        if nz:
            bad_px += 1
            bad_bytes += nz
            maxerr = max(maxerr, int(d.max()))
            if len(examples) < 6:
                ch = int(np.argwhere(d != 0)[0][0])
                examples.append((px, ch, int(exp[ch]), int(got[ch])))
    print(f"pixels compared: {tot}; mismatching pixels: {bad_px}; mismatching bytes: {bad_bytes}; max|err|={maxerr}")
    for px, ch, e, g in examples:
        print(f"   px={px} ch={ch} gold={e} got={g}")
    print("\n=> engine IN-CHAIN conv_246 " +
          ("BYTE-EXACT — engine compute is correct in-chain; bug is downstream/other dispatches/spatial."
           if bad_bytes == 0 else
           "DIVERGES — engine mis-computes in-chain (despite isolation pass)."))
    return 0 if bad_bytes == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
