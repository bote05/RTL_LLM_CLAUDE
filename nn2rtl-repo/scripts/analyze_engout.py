#!/usr/bin/env python3
"""Compare the engine's RAW output (engine_act_out_wr_data, pre-bridge) for
dispatch 0 (conv_246) to node_conv_246.goldout. Decides:
  match    => engine COMPUTE is correct -> the in-chain bug is in the output
              path (engine_output_bridge / FIFO routing / tile slicing).
  mismatch => engine COMPUTE itself is wrong under Verilator.

engout.bin: per addr -> uint32 addr + 64 uint32 (2048-bit). act_out_base=4096;
addr-4096 = output pixel index (0..195). goldout: 196 px x 256 bytes (2048-bit).
"""
from __future__ import annotations
import struct
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "output/reports_integrated/verilator_nn2rtl_top_probe"
ACT_OUT_BASE = 4096
N_PIX = 196


def main() -> None:
    raw = (ROOT / "output/goldens/node_conv_246.goldout").read_bytes()
    mg, v, nv, spv, bps = struct.unpack("<4sIIII", raw[:20])
    assert bps == 256 and spv == N_PIX, (bps, spv)
    gold = np.frombuffer(raw[20:20 + spv * bps], dtype=np.uint8).reshape(spv, bps)

    p = D / "engreads_engout.bin"
    if not p.exists():
        print("(no engreads_engout.bin)"); return
    b = p.read_bytes(); rec = 4 + 256
    reads = {}
    for i in range(len(b) // rec):
        off = i * rec
        addr = struct.unpack_from("<I", b, off)[0]
        reads[addr] = np.frombuffer(b[off + 4:off + 4 + 256], dtype=np.uint8)
    inr = {a: d for a, d in reads.items() if ACT_OUT_BASE <= a < ACT_OUT_BASE + N_PIX}
    print(f"engine raw-output writes captured: {len(reads)}; in act_out range: {len(inr)}")
    match = mismatch = 0; first = []
    for addr, data in sorted(inr.items()):
        pix = addr - ACT_OUT_BASE
        if np.array_equal(data, gold[pix]):
            match += 1
        else:
            mismatch += 1
            if len(first) < 6:
                nd = int((data != gold[pix]).sum())
                # show as signed int8 for a few channels
                dg = data.view(np.int8); gg = gold[pix].view(np.int8)
                diffs = [(c, int(gg[c]), int(dg[c])) for c in range(256) if dg[c] != gg[c]][:5]
                first.append((pix, nd, diffs))
    print(f"  pixels: {len(inr)}  EXACT match: {match}  MISMATCH: {mismatch}")
    for pix, nd, diffs in first:
        print(f"   pix={pix}: {nd}/256 ch differ; sample (ch,gold,got): {diffs}")
    print("\n=> match => engine COMPUTE correct, bug is in output bridge/FIFO; "
          "mismatch => engine compute wrong under Verilator")


if __name__ == "__main__":
    main()
