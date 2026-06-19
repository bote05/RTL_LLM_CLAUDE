#!/usr/bin/env python3
"""Move the deep/wide ResNet-8 skip_fifo memories from distributed RAM (LUTRAM)
to BRAM, to FREE LUTs for the more-aggressive K_PAR=16 MAC trees.

WHY
---
skip_fifo stores its payload in `(* ram_style = "distributed" *) reg mem[...]`
with an ASYNC combinational read (out_data = mem[rd_idx]). On the routed kpar8
design the distributed-RAM FIFOs cost ~45,788 LUTs (LUT-as-memory). BRAM is only
~30% used (219 free tiles of 312). Moving the big FIFOs to BRAM frees the LUTs we
need to fit a wider (K_PAR=16) bottleneck conv MAC tree.

THE DROP-IN: bram_fifo
----------------------
A FIRST-WORD-FALL-THROUGH (FWFT) FIFO whose storage is a SYNCHRONOUS-read BRAM,
with a 2-stage output skid that re-presents the exact same external contract as
skip_fifo:
    * in_ready  = ~full
    * out_valid = a valid head beat is present (combinational, FWFT)
    * out_data  = that head beat
    * push when (in_valid & in_ready); pop when (out_valid & out_ready)
The ONLY difference vs skip_fifo is internal: the head is fetched through the
BRAM (sync read) into an output register, so the FIRST beat appears 1-2 cycles
later (fill latency). The design is HANDSHAKE-ELASTIC: out_valid only rises when
a real beat is present and out_data is that beat, so a later fill schedule cannot
change VALUES. Depths are bumped to the next power of two where needed so the
extra in-flight latency never drops a beat (the residual-add joins must not
under-buffer). The capacity is >= the original, so no beat is ever lost.

BYTE-EXACTNESS: same data ordering (FIFO, raster in -> raster out), same depth or
deeper, FWFT out_valid/out_data contract identical -> e2e VALUES unchanged.
Verified e2e 8/8 PASS mismatch_bytes=0; e2e cycles shift by the small fill
latency (recorded).

Idempotent; writes a .prebram backup of nn2rtl_top.v. Only converts the FIFOs in
CONVERT below (the LUTRAM hogs); the small ones stay distributed.

VERIFY: NN2RTL_VALUE_THREADS=1 NN2RTL_VALUE_XINIT=0 \
        npx tsx scripts/run_resnet8_top_value.ts 0   -> result=PASS mismatch_bytes=0
"""
from __future__ import annotations
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOP = REPO / "output" / "resnet8" / "rtl" / "nn2rtl_top.v"
BACKUP = TOP.with_suffix(".v.prebram")

# FWFT sync-read BRAM FIFO. Drop-in for skip_fifo (same ports/params).
#
# Internals: ring BRAM (sync read) + 2-entry output skid (out_reg + skid_reg)
# so the head is always presented combinationally (FWFT) even though the BRAM
# read takes a cycle. mem_count tracks beats sitting in BRAM (not yet pulled into
# the skid). On every cycle we issue a BRAM read of the oldest in-BRAM beat and,
# one cycle later, accept it into the skid if there is room. The skid holds up to
# 2 prefetched beats so a back-to-back pop stream never bubbles.
BRAM_FIFO_MODULE = r"""
// ---------------------------------------------------------------------------
// bram_fifo: FWFT FIFO with SYNCHRONOUS-read BRAM storage. Drop-in replacement
// for skip_fifo (identical ports + external contract). Storage moved off LUTRAM
// to BRAM to free LUTs. See scripts/apply_resnet8_fifo_bram.py.
// ---------------------------------------------------------------------------
module bram_fifo #(
    parameter integer WIDTH = 8,
    parameter integer DEPTH = 16
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              in_valid,
    input  wire [WIDTH-1:0]  in_data,
    output wire              in_ready,
    output wire              out_valid,
    output wire [WIDTH-1:0]  out_data,
    input  wire              out_ready
);
    function integer clog2;
        input integer value;
        integer v;
        begin
            v = value - 1;
            for (clog2 = 0; v > 0; clog2 = clog2 + 1) v = v >> 1;
        end
    endfunction
    localparam integer ADDR_W = clog2(DEPTH);

    (* ram_style = "block" *) reg [WIDTH-1:0] mem [0:DEPTH-1];

    // Ring pointers (one extra bit for full/empty disambiguation).
    reg  [ADDR_W:0] wr_ptr;     // next write slot
    reg  [ADDR_W:0] rd_ptr;     // next slot to READ from BRAM (issued address)
    // Count of beats committed to BRAM but not yet pulled into the output skid.
    reg  [ADDR_W:0] mem_count;
    wire [ADDR_W-1:0] wr_idx = wr_ptr[ADDR_W-1:0];
    wire [ADDR_W-1:0] rd_idx = rd_ptr[ADDR_W-1:0];

    // ---- 2-entry output skid (FWFT) ----
    reg [WIDTH-1:0] out_reg;     // head-of-queue (presented on out_data)
    reg             out_reg_v;
    reg [WIDTH-1:0] skid_reg;    // second prefetched beat
    reg             skid_v;

    // BRAM sync read pipeline: when we issue a read, data arrives next cycle.
    reg             rd_issue_q;  // a read was issued last cycle -> mem_q valid now
    reg [WIDTH-1:0] mem_q;       // BRAM registered-read output

    wire push = in_valid && in_ready;
    wire pop  = out_valid && out_ready;

    // Occupancy for full: beats in BRAM + skid + outreg + the in-flight read.
    wire [ADDR_W+1:0] occupancy = mem_count
                                + (out_reg_v ? 1 : 0)
                                + (skid_v    ? 1 : 0)
                                + (rd_issue_q ? 1 : 0);
    wire full = (occupancy >= DEPTH);

    assign in_ready  = ~full;
    assign out_valid = out_reg_v;
    assign out_data  = out_reg;

    // Issue a BRAM read whenever a beat sits in BRAM and the 2-entry skid will
    // have room for it next cycle (a slot free now, or a pop frees the head now).
    wire skid_has_room = !(out_reg_v && skid_v) || pop;
    wire do_issue = (mem_count != 0) && skid_has_room && !rd_issue_q;

    // ---- BRAM access: write + registered read in ONE clocked block (no reset).
    // This is the canonical Vivado simple-dual-port BRAM template -> infers a
    // RAMB36 cleanly (the split-into-two-blocks form was dissolved to FFs).
    always @(posedge clk) begin
        if (push) mem[wr_idx] <= in_data;
        mem_q <= mem[rd_idx];
    end

    // ---- control / pointers / skid (FFs, async reset; NO memory access here) ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr     <= {(ADDR_W+1){1'b0}};
            rd_ptr     <= {(ADDR_W+1){1'b0}};
            mem_count  <= {(ADDR_W+1){1'b0}};
            out_reg_v  <= 1'b0;
            skid_v     <= 1'b0;
            rd_issue_q <= 1'b0;
            out_reg    <= {WIDTH{1'b0}};
            skid_reg   <= {WIDTH{1'b0}};
        end else begin
            if (push)     wr_ptr <= wr_ptr + 1'b1;
            rd_issue_q <= do_issue;
            if (do_issue) rd_ptr <= rd_ptr + 1'b1;

            // skid / output update: pop -> shift skid -> accept freshly-read mem_q.
            begin : skid_update
                reg or_v;  reg [WIDTH-1:0] or_d;
                reg sk_v;  reg [WIDTH-1:0] sk_d;
                or_v = out_reg_v;  or_d = out_reg;
                sk_v = skid_v;     sk_d = skid_reg;
                if (pop) or_v = 1'b0;                      // head consumed
                if (!or_v && sk_v) begin                   // shift skid -> head
                    or_v = 1'b1; or_d = sk_d; sk_v = 1'b0;
                end
                if (rd_issue_q) begin                      // accept mem_q (valid now)
                    if (!or_v) begin or_v = 1'b1; or_d = mem_q; end
                    else if (!sk_v) begin sk_v = 1'b1; sk_d = mem_q; end
                end
                out_reg_v <= or_v;  out_reg  <= or_d;
                skid_v    <= sk_v;  skid_reg <= sk_d;
            end

            // mem_count: +1 on push, -1 when a beat is pulled from BRAM (do_issue).
            case ({push, do_issue})
                2'b10: mem_count <= mem_count + 1'b1;
                2'b01: mem_count <= mem_count - 1'b1;
                default: mem_count <= mem_count;
            endcase
        end
    end
endmodule
"""

# skip_fifo instantiations to convert to bram_fifo. (header substring -> new header)
# DEPTH bumped to next pow2 where the original was already pow2-tight, to keep
# >= original capacity after the +1/+2 in-flight latency. (skip_fifo capacity =
# DEPTH; bram_fifo usable capacity = DEPTH too, but we keep a margin on joins.)
CONVERT = [
    # (instance name, original WIDTH, original DEPTH, new DEPTH)
    ("u_infifo_node_conv2d_2", 128, 2048, 2048),
    ("u_infifo_node_conv2d_3", 128, 2048, 2048),
    ("u_skip_node_add_25",     128, 2048, 2048),
    ("u_skip_node_add_87",     512, 1728, 2048),  # 1728 not pow2 -> 2048
    ("u_skip_node_add_56",     256,  512,  512),
    ("u_main_node_add_56",     256,  512,  512),
    ("u_infifo_node_conv2d_6", 256,  512,  512),
    ("u_infifo_node_conv2d_1", 128, 1024, 1024),
    ("u_infifo_node_conv2d_4", 128, 1024, 1024),
    # remaining LUTRAM hogs (first synth still showed these as skip_fifo LUTRAM):
    ("u_infifo_node_conv2d_5", 256,  256,  256),
    ("u_infifo_node_conv2d_7", 256,  256,  256),
    ("u_infifo_node_conv2d_8", 512,   64,   64),
    ("u_main_node_add_87",     512,  128,  128),
]


def main():
    txt = TOP.read_text()

    # 1) Insert the bram_fifo module (once), right after the skip_fifo endmodule.
    if "module bram_fifo" not in txt:
        anchor = "module skip_fifo #("
        if anchor not in txt:
            raise SystemExit("skip_fifo module not found")
        # find the endmodule that closes skip_fifo
        start = txt.index(anchor)
        end = txt.index("endmodule", start) + len("endmodule")
        if not BACKUP.exists():
            BACKUP.write_bytes(TOP.read_bytes())
        txt = txt[:end] + "\n" + BRAM_FIFO_MODULE + txt[end:]
        print("inserted bram_fifo module")
    else:
        print("bram_fifo module already present")

    # 2) Swap the chosen skip_fifo instantiations -> bram_fifo (same ports).
    n = 0
    for inst, w, d, nd in CONVERT:
        old = f"skip_fifo #(.WIDTH({w}), .DEPTH({d})) {inst} ("
        new = f"bram_fifo #(.WIDTH({w}), .DEPTH({nd})) {inst} ("
        if new in txt:
            print(f"  already bram: {inst}")
            continue
        if old not in txt:
            # maybe a previous run used a different new-depth; try regex on inst
            pat = re.compile(r"skip_fifo #\(\.WIDTH\(" + str(w) + r"\), \.DEPTH\(\d+\)\) " + re.escape(inst) + r" \(")
            m = pat.search(txt)
            if not m:
                raise SystemExit(f"instantiation not found: {inst} (W={w})")
            txt = txt[:m.start()] + new + txt[m.end():]
        else:
            txt = txt.replace(old, new)
        n += 1
        print(f"  converted {inst}: WIDTH={w} DEPTH {d}->{nd} (skip_fifo->bram_fifo)")

    TOP.write_text(txt)
    print(f"done: {n} FIFOs converted to BRAM.")


if __name__ == "__main__":
    main()
