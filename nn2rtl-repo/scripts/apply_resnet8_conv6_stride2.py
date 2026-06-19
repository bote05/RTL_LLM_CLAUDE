#!/usr/bin/env python3
"""Give node_conv2d_6 the STRIDE-2 input decimation it is missing.

BUG
---
node_conv2d_6 is the 1x1 stride-2 projection of stage 3 (IC=32 OC=64, IH=IW=16,
OH=OW=8, SH=SW=2). Its sibling node_conv2d_3 (the stage-2 projection) decimates
the input stream: it tracks in_row/in_col, accepts EVERY input pixel, but only
runs the MAC + emits an output on stride-aligned pixels ((in_row[0]==0) &&
(in_col[0]==0)). node_conv2d_6 was generated WITHOUT that logic -- its ST_STREAM
enters ST_RUNNING (and emits valid_out) on EVERY valid_in, so it produces 256
outputs for the 256 input pixels instead of 64. The 64-wide stride-2-aligned
subset is correct, but the extra (non-aligned) outputs are pushed into the
node_add_87 skip FIFO, desyncing the residual add's lhs operand by one+ beats
(add_87 lhs beat1 != conv_6.goldout[1]; lhs beat2 == conv_6.goldout[1]). This
corrupts node_add_87 -> node_relu_6 -> node_mean -> node_linear and is the e2e
first-divergence at the network output (the conv requant fixes upstream were
correct but invisible because this swamps them).

FIX
---
Mirror node_conv2d_3's stride-2 handling exactly (parametric in IH/IW/OH_OW):
  * add in_row / in_col / out_count registers (+ OH_OW localparam),
  * reset them,
  * ST_STREAM: increment in_row/in_col every valid_in; enter ST_RUNNING ONLY on
    a stride-aligned pixel,
  * ST_OUTPUT: count emitted outputs and reset in_row/in_col/out_count on the
    last output of the frame (inter-frame alignment, identical to conv_3).
The MAC / BIAS / SCALE / per-tensor requant are untouched.

Idempotent; writes a .prestride2 backup.
"""
from __future__ import annotations
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PATH = REPO / "output" / "resnet8" / "rtl" / "node_conv2d_6.v"
BACKUP = PATH.with_suffix(".v.prestride2")


def main():
    txt = PATH.read_text()
    if "in_row" in txt and "out_count" in txt:
        print("node_conv2d_6: already has stride-2 handling (in_row/out_count); skip")
        return
    if not BACKUP.exists():
        BACKUP.write_bytes(PATH.read_bytes())
        print(f"  backup -> {BACKUP.name}")

    # 1) Add OH_OW localparam next to OW.
    old = "    localparam OH        = 8;\n    localparam OW        = 8;\n"
    new = ("    localparam OH        = 8;\n    localparam OW        = 8;\n"
           "    localparam OH_OW     = OH * OW;\n")
    assert old in txt, "OH/OW localparam block not found"
    txt = txt.replace(old, new, 1)

    # 2) Add in_row / in_col / out_count registers next to oc_group/state.
    old = ("    reg [$clog2(OC_PASSES+1)-1:0] oc_group;\n"
           "    reg [2:0] state;\n")
    new = ("    reg [$clog2(OC_PASSES+1)-1:0] oc_group;\n"
           "    reg [2:0] state;\n\n"
           "    // stride-2 input decimation (mirrors node_conv2d_3)\n"
           "    reg [$clog2(IH)-1:0] in_row;\n"
           "    reg [$clog2(IW)-1:0] in_col;\n"
           "    reg [$clog2(OH_OW+1)-1:0] out_count;\n")
    assert old in txt, "oc_group/state decl block not found"
    txt = txt.replace(old, new, 1)

    # 3) Reset in_row/in_col/out_count (append to the reset block, right after
    #    oc_group reset).
    old = ("            oc_group         <= 0;\n"
           "            data_out         <= {(OC*8){1'b0}};\n")
    new = ("            oc_group         <= 0;\n"
           "            in_row           <= 0;\n"
           "            in_col           <= 0;\n"
           "            out_count        <= 0;\n"
           "            data_out         <= {(OC*8){1'b0}};\n")
    assert old in txt, "reset block (oc_group<=0; data_out) not found"
    txt = txt.replace(old, new, 1)

    # 4) ST_STREAM: increment in_row/in_col every valid_in and only enter
    #    ST_RUNNING on a stride-aligned pixel.
    old = (
        "            ST_STREAM: begin\n"
        "                valid_out    <= 1'b0;\n"
        "                mac_valid_q1 <= 1'b0;\n"
        "                mac_valid_q2 <= 1'b0;\n"
        "                if (valid_in) begin\n"
        "                    for (i = 0; i < IC; i = i + 1)\n"
        "                        in_latch[i] <= $signed(data_in[i*8 +: 8]);\n"
        "                    ready_in         <= 1'b0;  // [INVARIANT:READY_IN_GATING]\n"
        "                    k_counter        <= 0;\n"
        "                    lane_counter     <= 0;\n"
        "                    oc_group         <= 0;\n"
        "                    mac_done_issuing <= 1'b0;\n"
        "                    mac_lane_q1      <= 0;\n"
        "                    mac_k_q1         <= 0;\n"
        "                    mac_global_oc_q1 <= 0;\n"
        "                    mac_lane_q2      <= 0;\n"
        "                    mac_global_oc_q2 <= 0;\n"
        "                    mul_q            <= 0;\n"
        "                    for (lane = 0; lane < MP; lane = lane + 1)\n"
        "                        acc[lane] <= 0;\n"
        "                    state <= ST_RUNNING;\n"
        "                end\n"
        "            end\n"
    )
    new = (
        "            ST_STREAM: begin\n"
        "                valid_out    <= 1'b0;\n"
        "                mac_valid_q1 <= 1'b0;\n"
        "                mac_valid_q2 <= 1'b0;\n"
        "                if (valid_in) begin\n"
        "                    // advance the input raster position every accepted pixel\n"
        "                    if (in_col == IW - 1) begin\n"
        "                        in_col <= 0;\n"
        "                        if (in_row == IH - 1) in_row <= 0;\n"
        "                        else                  in_row <= in_row + 1;\n"
        "                    end else begin\n"
        "                        in_col <= in_col + 1;\n"
        "                    end\n"
        "                    // run the MAC only on stride-2-aligned pixels\n"
        "                    if ((in_row[0] == 1'b0) && (in_col[0] == 1'b0)) begin\n"
        "                        for (i = 0; i < IC; i = i + 1)\n"
        "                            in_latch[i] <= $signed(data_in[i*8 +: 8]);\n"
        "                        ready_in         <= 1'b0;  // [INVARIANT:READY_IN_GATING]\n"
        "                        k_counter        <= 0;\n"
        "                        lane_counter     <= 0;\n"
        "                        oc_group         <= 0;\n"
        "                        mac_done_issuing <= 1'b0;\n"
        "                        mac_lane_q1      <= 0;\n"
        "                        mac_k_q1         <= 0;\n"
        "                        mac_global_oc_q1 <= 0;\n"
        "                        mac_lane_q2      <= 0;\n"
        "                        mac_global_oc_q2 <= 0;\n"
        "                        mul_q            <= 0;\n"
        "                        for (lane = 0; lane < MP; lane = lane + 1)\n"
        "                            acc[lane] <= 0;\n"
        "                        state <= ST_RUNNING;\n"
        "                    end\n"
        "                end\n"
        "            end\n"
    )
    assert old in txt, "ST_STREAM block not matched"
    txt = txt.replace(old, new, 1)

    # 5) ST_OUTPUT final branch: track out_count + reset in_row/in_col/out_count
    #    on the last output of the frame.
    old = (
        "                end else begin\n"
        "                    valid_out    <= 1'b1;  // [INVARIANT:VALID_OUT_LATENCY]\n"
        "                    ready_in     <= 1'b1;  // [INVARIANT:READY_IN_GATING]\n"
        "                    oc_group     <= 0;\n"
        "                    k_counter    <= 0;\n"
        "                    lane_counter <= 0;\n"
        "                    state        <= ST_STREAM;\n"
        "                end\n"
    )
    new = (
        "                end else begin\n"
        "                    valid_out    <= 1'b1;  // [INVARIANT:VALID_OUT_LATENCY]\n"
        "                    ready_in     <= 1'b1;  // [INVARIANT:READY_IN_GATING]\n"
        "                    oc_group     <= 0;\n"
        "                    k_counter    <= 0;\n"
        "                    lane_counter <= 0;\n"
        "                    state        <= ST_STREAM;\n"
        "                    // inter-frame alignment: reset the raster on last output\n"
        "                    if (out_count == OH_OW - 1) begin\n"
        "                        out_count <= 0;\n"
        "                        in_row    <= 0;\n"
        "                        in_col    <= 0;\n"
        "                    end else begin\n"
        "                        out_count <= out_count + 1;\n"
        "                    end\n"
        "                end\n"
    )
    assert old in txt, "ST_OUTPUT final-emit branch not matched"
    txt = txt.replace(old, new, 1)

    PATH.write_text(txt)
    print("node_conv2d_6: added stride-2 input decimation (mirrors node_conv2d_3)")


if __name__ == "__main__":
    main()
