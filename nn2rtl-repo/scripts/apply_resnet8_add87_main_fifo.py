#!/usr/bin/env python3
"""
apply_resnet8_add87_main_fifo.py

Fix the ResNet-8 engine-less top deadlock at the THIRD residual add (node_add_87).

ROOT CAUSE (localized via NN2RTL_DBG_TAPS beat-counter instrumentation; see the
deadlock-localization run in output/resnet8/reports)
-----------------------------------------------------------------------------
node_add_87 is the residual add for stage 3 (512-wide). Its two operands are:
  - MAIN : node_conv2d_8 output (relu_4 -> conv_7 -> relu_5 -> conv_8), a
           free-running stream of exactly 64 beats (the 8x8 feature map).
  - SKIP : node_conv2d_6 output, the 1x1 STRIDE-2 *projection* conv on the skip
           branch. conv_6 is fronted by a frame_gate_fifo (it cannot release
           until its full 256-beat input frame is buffered) so it produces its
           64 output beats VERY LATE.

The wrapper buffers ONLY the SKIP operand (u_skip_node_add_87). The MAIN operand
(conv_8) flows LIVE into node_add_87 with no buffering. node_add_87 fires only
when conv_8_valid_out AND skip_87_valid co-occur, and skip_87 pops only on the
same co-occurrence. But conv_6 (skip) arrives ~1.35M cycles AFTER conv_8 (main):
by the time skip_87 finally goes valid, conv_8 has already emitted ~42 of its 64
beats into the void (consumed by nobody, lost). Only the LAST ~22 conv_8 beats
overlap with skip_87 being valid, so node_add_87 fires 22 times and then STARVES
(conv_8 is done, will not re-emit). node_relu_6 -> node_mean never receive a full
64-beat frame, node_mean never produces output, and the top deadlocks
(m_axis_tvalid never asserts).

  MEASURED at deadlock: a87=22  s87pop=22  c8av=64  (conv_8 presented 64 main
  beats, only 22 paired with the skip; the other 42 were dropped).

This is the residual-add OPERAND-SKEW deadlock: the design assumes the buffered
(skip) operand arrives no later than the unbuffered (main) operand. For add_25
and add_56 the skip operand DOES arrive first (so the buffered-early / live-late
assumption holds and they work). For add_87 the skip operand (slow projection
conv_6) arrives LAST -> the assumption is violated -> the live main operand is
dropped.

FIX (this script)
-----------------
Buffer the MAIN operand of node_add_87 in its own elastic skip_fifo and turn the
add into a TRUE two-FIFO synchronized join: each FIFO pops only when BOTH have a
beat, so the add always consumes a matched pair regardless of which operand
arrives first. Immune to arrival skew in either direction.

  conv_8 --push--> u_main_node_add_87 (DEPTH 128 >= 64-beat frame) --+
                                                                     +--> node_add_87
  conv_6 --frame_gate--> u_skip_node_add_87 -----------------------+
                          (both pop iff both out_valid & add.ready_in)

Only node_add_87 is touched: add_25/add_56 work today (their skip operand arrives
first; arrival order is structural/data-independent, so they are safe on all 8
golden vectors). Idempotent: re-running detects u_main_node_add_87 and is a no-op.
Writes a .preadd87fix backup once.
"""
import argparse
import os
import re
import sys

DEFAULT_TOP = r"D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/resnet8/rtl/nn2rtl_top.v"
MARKER = "u_main_node_add_87"

# The main-operand FIFO + the rewired add + rewired skip pop, inserted as a block
# that REPLACES the original add_87 instance. DEPTH 128 is the next pow2 >= the
# 64-beat conv_8 output frame (so it never overflows before draining begins).
MAIN_FIFO_AND_ADD = """\
    // [ADD87 MAIN-OPERAND FIFO] inserted by apply_resnet8_add87_main_fifo.py
    // Buffers the free-running conv_8 (main) operand so node_add_87 can pair it
    // with the LATE-arriving skip operand (projection conv_6). Two-FIFO join:
    // both FIFOs pop only when BOTH present, immune to operand arrival skew.
    wire node_add_87_main_valid;
    wire [511:0] node_add_87_main_data;
    wire node_add_87_main_in_ready;
    skip_fifo #(.WIDTH(512), .DEPTH(128)) u_main_node_add_87 (
        .clk(clk), .rst_n(rst_n),
        .in_valid(node_conv2d_8_valid_out & spatial_run & node_add_87_main_in_ready),
        .in_data(node_conv2d_8_data_out[511:0]),
        .in_ready(node_add_87_main_in_ready),
        .out_valid(node_add_87_main_valid),
        .out_data(node_add_87_main_data),
        .out_ready(node_add_87_skip_valid & node_add_87_ready_in)
    );

    // residual add: lhs (LOW half) from skip FIFO, rhs (HIGH half) from main FIFO
    node_add_87 u_node_add_87 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(node_add_87_main_valid & node_add_87_skip_valid),
        .ready_in(node_add_87_ready_in),
        .data_in({node_add_87_main_data, node_add_87_skip_data}),
        .valid_out(node_add_87_valid_out),
        .data_out(node_add_87_data_out)
    );
"""

ORIG_ADD87_RE = re.compile(
    r"    // residual add: lhs \(LOW half\) from skip FIFO, rhs \(HIGH half\) from main path\n"
    r"    node_add_87 u_node_add_87 \((?:.|\n)*?\n    \);\n",
)

# skip_87 out_ready must now pop in lockstep with the main FIFO (was conv_8 live).
SKIP87_OLD = ".out_ready(node_conv2d_8_valid_out & node_add_87_ready_in)"
SKIP87_NEW = ".out_ready(node_add_87_main_valid & node_add_87_ready_in)"


def main():
    ap = argparse.ArgumentParser(description="Fix ResNet-8 node_add_87 operand-skew deadlock.")
    ap.add_argument("--top", default=os.environ.get("NN2RTL_TOP") or DEFAULT_TOP)
    args = ap.parse_args()
    with open(args.top, "r", encoding="utf-8") as f:
        text = f.read()

    if MARKER in text:
        print(f"[add87-fix] {MARKER} already present in {args.top} -- no-op (idempotent).")
        return 0

    # backup once
    bak = args.top + ".preadd87fix"
    if not os.path.exists(bak):
        with open(bak, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[add87-fix] wrote backup {bak}")

    # 1) replace the add_87 instance block with the main-FIFO + rewired add
    new_text, n = ORIG_ADD87_RE.subn(MAIN_FIFO_AND_ADD, text, count=1)
    if n != 1:
        raise RuntimeError("could not locate the original node_add_87 instance block")
    text = new_text

    # 2) rewire the skip_87 FIFO out_ready to lockstep with the main FIFO
    if SKIP87_OLD not in text:
        raise RuntimeError("could not locate skip_87 out_ready to rewire")
    text = text.replace(SKIP87_OLD, SKIP87_NEW, 1)

    with open(args.top, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[add87-fix] node_add_87 main-operand FIFO + two-FIFO join applied to {args.top}")
    print("[add87-fix]   - added u_main_node_add_87 (skip_fifo WIDTH=512 DEPTH=128)")
    print("[add87-fix]   - add_87.valid_in = main_valid & skip_valid (synchronized join)")
    print("[add87-fix]   - skip_87.out_ready = main_valid & ready_in (lockstep pop)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
