#!/usr/bin/env python3
"""Compare the engine's actual in-chain activation reads (dispatch 0 = conv_246)
to conv_246's goldin. Answers: does the engine READ the correct input activations
in-chain (=> bug is in the engine compute/state), or wrong ones (=> loader /
BRAM contention / timing)?

engreads_{same,delayed}.bin: per unique addr -> uint32 addr + 64 uint32 words.
conv_246 act_in_base = 8192; addr-8192 = input pixel index (0..783).
node_conv_246.goldin vec0: 784 samples x 256 bytes (256 channels each).
"""
from __future__ import annotations
import struct, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DUMP = ROOT / "output/reports_integrated/verilator_nn2rtl_top_probe"
ACT_IN_BASE = 8192
N_PIX = 784


def load_goldin_vec0():
    p = ROOT / "output/goldens/node_conv_246.goldin"
    raw = p.read_bytes()
    mg, v, nv, spv, bps = struct.unpack("<4sIIII", raw[:20])
    assert bps == 256 and spv == N_PIX, (bps, spv)
    body = np.frombuffer(raw[20:20 + spv * bps], dtype=np.uint8).reshape(spv, bps)
    return body  # [784, 256]


def load_engreads(name):
    p = DUMP / f"engreads_{name}.bin"
    if not p.exists():
        return None
    raw = p.read_bytes()
    rec = 4 + 256  # addr + 256 bytes
    n = len(raw) // rec
    out = {}
    for i in range(n):
        off = i * rec
        addr = struct.unpack_from("<I", raw, off)[0]
        data = np.frombuffer(raw[off + 4:off + 4 + 256], dtype=np.uint8)
        out[addr] = data
    return out


def compare(name, gold):
    er = load_engreads(name)
    if er is None:
        print(f"[{name}] (no file)"); return
    in_range = {a: d for a, d in er.items() if ACT_IN_BASE <= a < ACT_IN_BASE + N_PIX}
    print(f"\n=== engreads_{name}: {len(er)} unique addrs, {len(in_range)} in act_in range [{ACT_IN_BASE},{ACT_IN_BASE+N_PIX}) ===")
    if not in_range:
        print("   addr range seen:", min(er), "..", max(er)); return
    match = mismatch = 0; first = []
    for addr, data in sorted(in_range.items()):
        pix = addr - ACT_IN_BASE
        exp = gold[pix]
        if np.array_equal(data, exp):
            match += 1
        else:
            mismatch += 1
            if len(first) < 6:
                nd = int((data != exp).sum())
                first.append((pix, nd, data[:8].tolist(), exp[:8].tolist()))
    print(f"   pixels read: {len(in_range)}  EXACT match: {match}  MISMATCH: {mismatch}")
    for pix, nd, dgot, dexp in first:
        print(f"   pix={pix}: {nd}/256 bytes differ  got[0:8]={dgot}  gold[0:8]={dexp}")


def main():
    gold = load_goldin_vec0()
    print(f"goldin: {gold.shape} (784 pixels x 256 channels)")
    for nm in ("same", "delayed"):
        compare(nm, gold)
    print("\nIf one latency shows ~all-exact -> engine reads CORRECT activations "
          "-> compute/state bug. If both mismatch -> wrong activations (loader/contention/timing).")


if __name__ == "__main__":
    main()
