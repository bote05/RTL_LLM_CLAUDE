#!/usr/bin/env python3
"""Insert a MAIN-operand FIFO between node_conv2d_5 and node_add_56, mirroring the
existing add_87 main-operand FIFO (u_main_node_add_87).

WHY
---
After serializing node_conv2d_5 to the MP=4 FSM (apply_resnet8_serialize_convs.py),
the stage-2 residual add_56 DEADLOCKS (probe: conv_5 produced all 256 outputs but
add_56 consumed only ~35, then froze permanently).

Root cause: the serial conv FSM (like its siblings conv_4/_7) asserts valid_out
for exactly ONE cycle per output pixel and does NOT honour downstream ready --
there is no output-side backpressure. node_add_56 only fires when BOTH its main
operand (conv_5) AND its skip operand (conv_3 via skip_node_add_56) are valid the
SAME cycle. With the parallel conv_5 the two operand streams were latency-matched
(the LATENCY_PAD shift register aligned conv_5's output with the skip FIFO), so
the add always fired. The serial conv_5 produces on a completely different,
slower cadence; its 1-cycle valid_out pulses almost never coincide with a
skip-valid cycle, so conv_5's outputs are DROPPED and add_56 starves.

The design already solves this exact problem for add_87 with a two-FIFO join:
node_conv2d_8 -> u_main_node_add_87 (buffers the free-running main operand) and
node_conv2d_6 -> u_skip_node_add_87, both popped only when BOTH are present
(skew-immune). conv_5 -> add_56 was left WITHOUT a main FIFO (it fed add_56
directly), which only worked while conv_5 was the latency-matched parallel pipe.

FIX (idempotent; writes nn2rtl_top.v.preadd56fix backup)
--------------------------------------------------------
Insert u_main_node_add_56 (skip_fifo WIDTH=256 DEPTH=512 -- a full stage-2 frame
is 256 pixels, 512 = 2x headroom so the FIFO never fills during conv_5's 256
outputs => its in_ready stays high => no dropped outputs) buffering conv_5's
output, and rewire node_add_56 to take its main operand from that FIFO:

  main FIFO out_ready = node_add_56_skip_valid & node_add_56_ready_in
  add_56  valid_in   = node_add_56_main_valid & node_add_56_skip_valid
  add_56  data_in    = {node_add_56_main_data, node_add_56_skip_data}

This is BYTE-EXACT: pure buffering + a join, no datapath change. Same values,
same order; only the timing changes (the add now waits for both operands).

VERIFY: NN2RTL_VALUE_THREADS=1 NN2RTL_VALUE_XINIT=0 \
        npx tsx scripts/run_resnet8_top_value.ts 0   -> result=PASS mismatch_bytes=0
"""
from __future__ import annotations
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOP = REPO / "output" / "resnet8" / "rtl" / "nn2rtl_top.v"
BACKUP = TOP.with_suffix(TOP.suffix + ".preadd56fix")

OLD_ADD56 = """    // residual add: lhs (LOW half) from skip FIFO, rhs (HIGH half) from main path
    node_add_56 u_node_add_56 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(node_conv2d_5_valid_out & node_add_56_skip_valid & spatial_run),
        .ready_in(node_add_56_ready_in),
        .data_in({node_conv2d_5_data_out[255:0], node_add_56_skip_data}),
        .valid_out(node_add_56_valid_out),
        .data_out(node_add_56_data_out)
    );"""

NEW_ADD56 = """    // [ADD56 MAIN-OPERAND FIFO] inserted by apply_resnet8_add56_main_fifo.py
    // Buffers the serialized conv_5 (main) operand so node_add_56 can pair it with
    // the skip operand (projection conv_3) at its own pace. Without this, conv_5's
    // 1-cycle valid_out pulses (no output backpressure) miss the skip-valid window
    // and are dropped -> add_56 starves. Two-FIFO join, immune to operand skew.
    // DEPTH=512 >= the 256-pixel stage-2 frame so the FIFO never fills during
    // conv_5's 256 outputs (its in_ready stays high -> no dropped outputs).
    wire node_add_56_main_valid;
    wire [255:0] node_add_56_main_data;
    wire node_add_56_main_in_ready;
    skip_fifo #(.WIDTH(256), .DEPTH(512)) u_main_node_add_56 (
        .clk(clk), .rst_n(rst_n),
        .in_valid(node_conv2d_5_valid_out & spatial_run & node_add_56_main_in_ready),
        .in_data(node_conv2d_5_data_out[255:0]),
        .in_ready(node_add_56_main_in_ready),
        .out_valid(node_add_56_main_valid),
        .out_data(node_add_56_main_data),
        .out_ready(node_add_56_skip_valid & node_add_56_ready_in)
    );

    // residual add: lhs (LOW half) from skip FIFO, rhs (HIGH half) from main FIFO
    node_add_56 u_node_add_56 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(node_add_56_main_valid & node_add_56_skip_valid),
        .ready_in(node_add_56_ready_in),
        .data_in({node_add_56_main_data, node_add_56_skip_data}),
        .valid_out(node_add_56_valid_out),
        .data_out(node_add_56_data_out)
    );"""


# The skip FIFO (u_skip_node_add_56) popped on the OLD direct-feed condition
# (node_conv2d_5_valid_out). Now that conv_5 feeds the MAIN FIFO, the skip FIFO
# must pop when the add actually fires = main_valid & ready (mirroring add_87's
# u_skip_node_add_87). Leaving the old condition desyncs the skip operand order
# -> wrong residual pairing -> wrong logits. Fix the skip FIFO out_ready too.
OLD_SKIP_OUTREADY = "        .out_ready(node_conv2d_5_valid_out & node_add_56_ready_in)\n    );"
NEW_SKIP_OUTREADY = "        .out_ready(node_add_56_main_valid & node_add_56_ready_in)\n    );"


def main():
    txt = TOP.read_text()
    already_main = "u_main_node_add_56" in txt
    already_skip = OLD_SKIP_OUTREADY not in txt
    if already_main and already_skip:
        print("nn2rtl_top.v: add_56 main FIFO + skip out_ready already patched; skip")
        return
    if not already_main and OLD_ADD56 not in txt:
        raise SystemExit("nn2rtl_top.v: add_56 instantiation anchor not found "
                         "(already partially patched or layout changed)")
    if not BACKUP.exists():
        BACKUP.write_bytes(txt.encode())
        print(f"  backup -> {BACKUP.name}")
    if not already_main:
        txt = txt.replace(OLD_ADD56, NEW_ADD56, 1)
        print("nn2rtl_top.v: inserted u_main_node_add_56 (WIDTH=256 DEPTH=512) + "
              "rewired add_56 to the two-FIFO join")
    if OLD_SKIP_OUTREADY in txt:
        txt = txt.replace(OLD_SKIP_OUTREADY, NEW_SKIP_OUTREADY, 1)
        print("nn2rtl_top.v: fixed u_skip_node_add_56 out_ready -> "
              "(node_add_56_main_valid & node_add_56_ready_in)")
    TOP.write_text(txt)


if __name__ == "__main__":
    main()
