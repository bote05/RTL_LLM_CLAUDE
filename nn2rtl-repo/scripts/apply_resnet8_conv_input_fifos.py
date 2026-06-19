#!/usr/bin/env python3
"""
apply_resnet8_conv_input_fifos.py

Insert an elastic input FIFO in front of every scheduler-paced conv in the
ResNet-8 engine-less top so the free-running spatial chain stops DROPPING beats.

ROOT CAUSE (localized via output/resnet8/reports debug build + iverilog isos)
-----------------------------------------------------------------------------
Every ResNet-8 conv accepts input under a coord_scheduler that only raises its
internal ready_in on *real-input* cycles (it goes LOW during the 3x3 zero-pad
border cycles and the registered output_fires cycle). Most conv modules expose
that true ready_in on their .ready_in port. BUT the wrapper wires each module's
.valid_in = <producer>_valid_out & spatial_run and NEVER feeds the consumer's
ready_in back to the producer; and the producers (relu = 1-cycle passthrough,
the systolic convs) are FREE-RUNNING — they emit one beat per cycle and cannot
hold. So whenever a conv's ready_in is low while its producer presents a beat,
that beat is consumed by nobody and LOST. Proven in isolation:
    node_conv2d_1 fed gated-on-ready_in  -> 1024/1024 out   (per-module contract)
    node_conv2d_1 fed 1024 back-to-back  ->  488/1024 out   (free-running chain)
The dropped beats desync the line buffer, the frame never completes, and the
whole top deadlocks (all 1024 inputs consumed, zero outputs).

FIX (proven: free-running producer -> skip_fifo -> conv on conv.ready_in ->
1024/1024 out)
--------------------------------------------------------------------------
Drop a `skip_fifo` (the wrapper's standard FWFT elastic buffer, already defined
in nn2rtl_top.v) between each conv's producer and the conv:
    producer (free-running, 1/cyc) --push--> FIFO --pull on conv.ready_in--> conv
The producer pushes every cycle (gated only by FIFO !full); the conv pulls only
when its scheduler can accept. The FIFO's average drain rate == the conv's
output rate == 1/cyc, so a frame-deep FIFO never overflows; it only absorbs the
pad-cycle gaps. Conv input frame sizes are <= 1024, so DEPTH = next-pow2 of the
frame size (>= the worst-case producer/consumer skew) is safe.

The stem (node_conv2d) is left alone: it is fed by s_axis, which IS elastic (an
AXI master holds the beat until tready = node_conv2d_ready_in), and it already
exposes its real ready_in. node_conv2d_2 keeps ready_in=1'b1 (it buffers the
whole frame internally and genuinely accepts 1/cyc) but still gets a FIFO so it
receives a CONTIGUOUS gap-free stream regardless of upstream pacing (verified:
contiguous 1024 -> 1024 out).

This patches ONLY the per-conv .valid_in/.data_in of the listed convs and adds
the FIFO instances; it does not touch the add wiring, skip FIFOs, the stem, or
any module body. Idempotent (re-running detects the inserted FIFO and skips).
"""
import argparse
import os
import re
import sys

DEFAULT_TOP = r"D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/resnet8/rtl/nn2rtl_top.v"

# conv_inst -> (input_width_bits, fifo_depth, kind, frame).
#   kind "skip"  : streaming coord_scheduler conv -> plain elastic skip_fifo,
#                  drains on the conv's real ready_in. Tolerates gappy input.
#   kind "frame" : whole-frame cycle_count conv (node_conv2d_2/3/6) whose
#                  TIME-BASED output trigger mis-fires on gappy input. Use a
#                  frame_gate_fifo: buffer the full frame, then release it as a
#                  CONTIGUOUS gap-free burst (verified 1024->1024).
# Stem (node_conv2d) excluded — its s_axis feed is already elastic.
# Depth = next-pow2 >= input frame size (>= worst-case producer/consumer skew).
# `frame` = the conv's input frame size IH*IW.
CONV_FIFOS = [
    ("node_conv2d_1", 128, 1024, "skip",  1024),
    ("node_conv2d_2", 128, 2048, "frame", 1024),
    ("node_conv2d_3", 128, 2048, "frame", 1024),
    ("node_conv2d_4", 128, 1024, "skip",  1024),
    ("node_conv2d_5", 256, 256,  "skip",  256),
    ("node_conv2d_6", 256, 512,  "frame", 256),
    ("node_conv2d_7", 256, 256,  "skip",  256),
    ("node_conv2d_8", 512, 64,   "skip",  64),
]
MARKER = "// [CONV INPUT FIFO] inserted by apply_resnet8_conv_input_fifos.py"

# frame_gate_fifo module body, appended to the top (after the skip_fifo body).
FRAME_GATE_FIFO = r"""
// [FRAME-GATE FIFO] appended by apply_resnet8_conv_input_fifos.py.
// Buffers FRAME beats, then releases them as a contiguous gap-free burst.
// Fronts the cycle_count(whole-frame) convs whose time-based output trigger
// mis-fires on gappy input.
module frame_gate_fifo #(
    parameter integer WIDTH = 128,
    parameter integer DEPTH = 2048,
    parameter integer FRAME = 1024
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             in_valid,
    input  wire [WIDTH-1:0] in_data,
    output wire             in_ready,
    output wire             out_valid,
    output wire [WIDTH-1:0] out_data,
    input  wire             out_ready
);
    function integer clog2; input integer value; integer v; begin
        v = value - 1; for (clog2 = 0; v > 0; clog2 = clog2 + 1) v = v >> 1; end
    endfunction
    localparam integer AW = clog2(DEPTH);
    localparam integer FW = clog2(FRAME + 1);

    reg [WIDTH-1:0] mem [0:DEPTH-1];
    reg [AW:0] wr_ptr, rd_ptr;
    wire [AW-1:0] wr_idx = wr_ptr[AW-1:0];
    wire [AW-1:0] rd_idx = rd_ptr[AW-1:0];
    wire empty = (wr_ptr == rd_ptr);
    wire full  = (wr_ptr[AW] != rd_ptr[AW]) && (wr_ptr[AW-1:0] == rd_ptr[AW-1:0]);

    reg [FW-1:0] fill_q;
    reg          releasing_q;
    reg [FW-1:0] drain_q;

    wire push = in_valid && ~full;
    wire pop  = out_valid && out_ready;

    assign in_ready  = ~full;
    assign out_valid = releasing_q && ~empty;
    assign out_data  = mem[rd_idx];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= 0; rd_ptr <= 0;
            fill_q <= 0; releasing_q <= 1'b0; drain_q <= 0;
        end else begin
            if (push) begin mem[wr_idx] <= in_data; wr_ptr <= wr_ptr + 1'b1; end
            if (pop) rd_ptr <= rd_ptr + 1'b1;
            if (!releasing_q) begin
                if (push) begin
                    if (fill_q == FRAME - 1) begin
                        releasing_q <= 1'b1; drain_q <= 0; fill_q <= 0;
                    end else fill_q <= fill_q + 1'b1;
                end
            end else begin
                if (pop) begin
                    if (drain_q == FRAME - 1) begin releasing_q <= 1'b0; drain_q <= 0; end
                    else drain_q <= drain_q + 1'b1;
                end
                if (push) fill_q <= fill_q + 1'b1;
            end
        end
    end
endmodule
"""


def find_inst(text, inst):
    m = re.search(r"(    " + re.escape(inst.replace("node_", "node_", 1)) +
                  r"\s+u_" + re.escape(inst) + r"\s*\((?:.|\n)*?\n    \);)", text)
    if not m:
        # the module type == inst name for convs (node_conv2d_1 u_node_conv2d_1)
        m = re.search(r"(" + re.escape(inst) + r"\s+u_" + re.escape(inst) +
                      r"\s*\((?:.|\n)*?\n\s*\);)", text)
    if not m:
        raise RuntimeError(f"conv instance u_{inst} not found")
    return m.start(1), m.end(1), m.group(1)


def patch_conv(text, inst, width, depth, kind, frame):
    s, e, block = find_inst(text, inst)
    if f"u_infifo_{inst}" in text:
        return text, False  # already inserted

    # capture the producer valid/data from the current .valid_in / .data_in
    vi = re.search(r"\.valid_in\(\s*([A-Za-z0-9_]+)_valid_out\s*&\s*spatial_run\s*\)", block)
    di = re.search(r"\.data_in\(\s*([A-Za-z0-9_]+)_data_out\s*\)", block)
    if not vi or not di:
        raise RuntimeError(f"could not parse producer from u_{inst}")
    prod_v = vi.group(1)
    prod_d = di.group(1)

    fifo_ov = f"{inst}_infifo_out_valid"
    fifo_od = f"{inst}_infifo_out_data"
    fifo_ir = f"{inst}_infifo_in_ready"

    if kind == "frame":
        decl = f"    frame_gate_fifo #(.WIDTH({width}), .DEPTH({depth}), .FRAME({frame})) u_infifo_{inst} ("
    else:
        decl = f"    skip_fifo #(.WIDTH({width}), .DEPTH({depth})) u_infifo_{inst} ("

    fifo = (
        f"    {MARKER} ({kind})\n"
        f"    wire {fifo_ov};\n"
        f"    wire [{width-1}:0] {fifo_od};\n"
        f"    wire {fifo_ir};\n"
        f"{decl}\n"
        f"        .clk(clk), .rst_n(rst_n),\n"
        f"        .in_valid({prod_v}_valid_out & spatial_run & {fifo_ir}),\n"
        f"        .in_data({prod_d}_data_out[{width-1}:0]),\n"
        f"        .in_ready({fifo_ir}),\n"
        f"        .out_valid({fifo_ov}),\n"
        f"        .out_data({fifo_od}),\n"
        f"        .out_ready({inst}_ready_in)\n"
        f"    );\n"
    )

    # rewrite the conv instance to pull from the FIFO
    new_block = block
    new_block = re.sub(r"\.valid_in\([^)]*\)", f".valid_in({fifo_ov})", new_block, count=1)
    new_block = re.sub(r"\.data_in\([^)]*\)", f".data_in({fifo_od})", new_block, count=1)

    return text[:s] + fifo + "\n" + new_block + text[e:], True


def main():
    ap = argparse.ArgumentParser(description="Insert elastic input FIFOs before ResNet-8 convs.")
    ap.add_argument("--top", default=os.environ.get("NN2RTL_TOP") or DEFAULT_TOP)
    args = ap.parse_args()
    with open(args.top, "r", encoding="utf-8") as f:
        text = f.read()

    report = []
    for inst, width, depth, kind, frame in CONV_FIFOS:
        text, changed = patch_conv(text, inst, width, depth, kind, frame)
        report.append((inst, width, depth, kind, changed))

    # append the frame_gate_fifo module body once (before the final `endif`/EOF).
    if "module frame_gate_fifo" not in text:
        # insert just before the closing `endif` guard if present, else at EOF
        marker = "`endif"
        idx = text.rfind(marker)
        if idx != -1:
            text = text[:idx] + FRAME_GATE_FIFO + "\n" + text[idx:]
        else:
            text = text + "\n" + FRAME_GATE_FIFO + "\n"

    with open(args.top, "w", encoding="utf-8") as f:
        f.write(text)

    print("ResNet-8 conv input FIFOs in", args.top)
    for inst, width, depth, kind, changed in report:
        print(f"  u_infifo_{inst:16s} WIDTH={width:4d} DEPTH={depth:5d} kind={kind:5s} "
              f"[{'inserted' if changed else 'already-present'}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
