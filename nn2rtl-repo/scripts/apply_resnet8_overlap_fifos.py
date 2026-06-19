#!/usr/bin/env python3
"""Swap the three full-frame BARRIER FIFOs (frame_gate_fifo) in the ResNet-8 top
for ELASTIC pass-through FIFOs (skip_fifo) so adjacent layers stream-OVERLAP.

WHY
---
nn2rtl_top.v gates the inputs of node_conv2d_2 / node_conv2d_3 / node_conv2d_6
behind frame_gate_fifo, which holds back the ENTIRE producer frame (FRAME beats)
before releasing the first output beat. That fully SERIALIZES the producer and
consumer across each boundary: with the convs re-parallelized
(apply_resnet8_kpar_convs.py + apply_resnet8_kpar_1x1.py) the e2e was gated at
~2x the slowest single conv because of these barriers.

A conv with a line-buffer window scheduler produces outputs in raster order and
consumes inputs in raster order, so it can START consuming as soon as the first
rows arrive -- the full-frame barrier is unnecessary. skip_fifo is the SAME
2-pointer elastic FIFO interface (in_valid/in_ready/out_valid/out_ready,
out_valid=~empty so it releases immediately) at the SAME depth, so the swap is a
drop-in. Result: the layers overlap and the frame is gated by the slowest SINGLE
conv instead of the SUM.

BYTE-EXACTNESS: the data ordering through the FIFO is unchanged (same FIFO, same
depth, raster in -> raster out); only the release SCHEDULE changes. The design is
handshake-elastic so a different release schedule cannot change values. Verified
e2e 8/8 PASS mismatch_bytes=0.

Idempotent; writes a .preoverlap backup of nn2rtl_top.v.

VERIFY: NN2RTL_VALUE_THREADS=1 NN2RTL_VALUE_XINIT=0 \
        npx tsx scripts/run_resnet8_top_value.ts 0   -> result=PASS mismatch_bytes=0
"""
from __future__ import annotations
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOP = REPO / "output" / "resnet8" / "rtl" / "nn2rtl_top.v"
BACKUP = TOP.with_suffix(".v.preoverlap")

# (frame_gate instantiation header -> elastic skip_fifo header), same WIDTH/DEPTH.
SWAPS = [
    ("frame_gate_fifo #(.WIDTH(128), .DEPTH(2048), .FRAME(1024)) u_infifo_node_conv2d_2 (",
     "skip_fifo #(.WIDTH(128), .DEPTH(2048)) u_infifo_node_conv2d_2 ("),
    ("frame_gate_fifo #(.WIDTH(128), .DEPTH(2048), .FRAME(1024)) u_infifo_node_conv2d_3 (",
     "skip_fifo #(.WIDTH(128), .DEPTH(2048)) u_infifo_node_conv2d_3 ("),
    ("frame_gate_fifo #(.WIDTH(256), .DEPTH(512), .FRAME(256)) u_infifo_node_conv2d_6 (",
     "skip_fifo #(.WIDTH(256), .DEPTH(512)) u_infifo_node_conv2d_6 ("),
]


def main():
    txt = TOP.read_text()
    changed = 0
    for a, b in SWAPS:
        if b in txt:
            print(f"  already elastic: {b.split('(')[1].strip()}")
            continue
        if a not in txt:
            raise SystemExit(f"frame_gate header not found (and not already swapped):\n  {a}")
        if not BACKUP.exists():
            BACKUP.write_bytes(TOP.read_bytes())
            print(f"  backup -> {BACKUP.name}")
        txt = txt.replace(a, b)
        changed += 1
        print(f"  swapped frame_gate -> skip_fifo: {b.split(')')[1].strip()}")
    TOP.write_text(txt)
    print(f"Done. ({changed} barrier FIFOs made elastic)")


if __name__ == "__main__":
    main()
