#!/usr/bin/env python3
"""conv_224 is a 1x1 STRIDE-2 conv. The split-arch lib (coord_scheduler +
line_buf_window) handles 3x3-s2 and 1x1-s1 but NOT 1x1-s2 (verified: max_error=15).
Work around it WITHOUT touching the lib: decimate the 56x56 input to the even/even
28x28 grid in the wrapper, then feed a PROVEN stride-1 28x28 1x1 inner conv. The
kept input pixels arrive in input raster order, which is exactly 28x28 raster order,
so the inner conv sees a clean contiguous frame. Backpressured output streamer.

MP=16 MP_K=8 (128 DSP), weights repacked as node_conv_224_weights_mp_k_8.hex.
Verify byte-exact with --equiv (ready_out tied high), then emit the real port form.

USAGE:
  python scripts/apply_conv224_decimator.py            # emit real (ready_out port)
  python scripts/apply_conv224_decimator.py --equiv    # emit tie-high for equiv
"""
from __future__ import annotations
import argparse
from pathlib import Path

RTL = Path("output/rtl")
WEIGHTS = Path("output/weights")

IC, OC = 256, 512
IH_FULL, IW_FULL = 56, 56
OH, OW = 28, 28
SCALE_MULT, SCALE_SHIFT = 8439, 19   # conv_224 effective requant (verified)
MP, MP_K = 16, 8


def gen(tie_ready_high: bool) -> str:
    wpath = (Path.cwd() / WEIGHTS / f"node_conv_224_weights_mp_k_{MP_K}.hex").as_posix()
    bpath = (Path.cwd() / WEIGHTS / "node_conv_224_bias.hex").as_posix()
    if tie_ready_high:
        ready_port = ""
        ready_decl = "    wire ready_out = 1'b1;  // equiv: consumer always ready\n"
    else:
        ready_port = "    input  wire                       ready_out,\n"
        ready_decl = ""
    return f"""`timescale 1ns / 1ps
// node_conv_224 - 1x1 conv2d STRIDE-2 via INPUT DECIMATION + stride-1 split-arch.
// coord_scheduler/line_buf_window don't support 1x1-s2 directly, so the wrapper
// keeps only even-row/even-col input pixels (the s2 sampling grid) and feeds a
// proven stride-1 28x28 1x1 inner conv (coord_scheduler+line_buf_window+
// conv_datapath_mp_k). Backpressured output streamer. Auto-gen apply_conv224_decimator.py.
//   IC={IC} OC={OC}  56x56 -> 28x28 (s2)  MP={MP} MP_K={MP_K}
module node_conv_224 (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       valid_in,
    output wire                       ready_in,
    input  wire [255:0]               data_in,
    output wire                       valid_out,
{ready_port}    output wire [255:0]               data_out
);
    localparam integer IC={IC}, OC={OC};
    localparam integer IH_FULL={IH_FULL}, IW_FULL={IW_FULL};
    localparam integer OH={OH}, OW={OW};
    localparam integer KH=1, KW=1, SH=1, SW=1, PH=0, PW=0;   // INNER conv: stride-1
    localparam integer K_TOTAL=IC*KH*KW;
    localparam integer MP={MP}, MP_K={MP_K};
    localparam integer SCALE_MULT={SCALE_MULT}, SCALE_SHIFT={SCALE_SHIFT};
    localparam integer CHANNEL_TILE=32, TILE_BITS=256;
    localparam integer IN_BEATS=IC/CHANNEL_TILE;     // 8
    localparam integer OUT_BEATS=OC/CHANNEL_TILE;    // 16
    localparam integer IN_PIXEL_BITS=IC*8;
    localparam integer OUT_PIXEL_BITS=OC*8;
    localparam integer INB_W=$clog2(IN_BEATS);
    localparam integer OUTB_W=$clog2(OUT_BEATS);
    localparam integer ROW_W=$clog2(IH_FULL);
    localparam integer COL_W=$clog2(IW_FULL);
{ready_decl}
    wire sched_needs_real_input, sched_ready_in, sched_output_fires, sched_advance;
    wire [$clog2(OH+PH+1)-1:0] sched_in_row;
    wire [$clog2(OW+PW+1)-1:0] sched_in_col;
    wire [$clog2(OH*OW+1)-1:0] sched_outputs_emitted;
    wire sched_out_frame_done;
    wire [KH*KW*IC*8-1:0] window_flat;
    wire mac_busy, lib_valid_out_w;
    wire [OUT_PIXEL_BITS-1:0] lib_data_out_w;

    reg start_pulse;
    reg [1:0] frame_state;
    localparam ST_ARM=2'd0, ST_RUN=2'd1, ST_WAIT=2'd2;

    // ---- input beat aggregation + even/even decimation over 56x56 ----
    reg [INB_W-1:0] in_beat_idx;
    reg [IN_PIXEL_BITS-TILE_BITS-1:0] in_lo;
    reg [ROW_W-1:0] irow;
    reg [COL_W-1:0] icol;
    wire is_last_in_beat = (in_beat_idx == IN_BEATS-1);
    wire keep = (irow[0]==1'b0) && (icol[0]==1'b0);
    // beats 0..IN_BEATS-2 accepted freely; on last beat a KEPT pixel waits for
    // the inner scheduler, a DROPPED pixel is accepted+discarded immediately.
    assign ready_in = is_last_in_beat ? (keep ? sched_ready_in : 1'b1) : 1'b1;
    wire beat_fire = valid_in && ready_in;
    wire last_beat_fire = beat_fire && is_last_in_beat;
    wire lib_valid_in_w = last_beat_fire && keep;
    wire [IN_PIXEL_BITS-1:0] lib_data_in_w = {{data_in, in_lo}};

    // ---- backpressured output streamer ----
    reg [OUT_PIXEL_BITS-1:0] out_pix;
    reg [OUTB_W:0]           out_idx;
    reg                      out_busy;
    assign valid_out = out_busy;
    assign data_out  = out_pix[out_idx*TILE_BITS +: TILE_BITS];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            frame_state<=ST_ARM; start_pulse<=1'b0;
            in_beat_idx<=0; in_lo<=0; irow<=0; icol<=0;
            out_pix<=0; out_idx<=0; out_busy<=1'b0;
        end else begin
            start_pulse <= 1'b0;
            case (frame_state)
                ST_ARM:  begin start_pulse<=1'b1; frame_state<=ST_RUN; end
                ST_RUN:  begin if (sched_out_frame_done) frame_state<=ST_WAIT; end
                ST_WAIT: begin if (!mac_busy) frame_state<=ST_ARM; end
                default: frame_state<=ST_ARM;
            endcase
            if (beat_fire) begin
                if (!is_last_in_beat) begin
                    in_lo[in_beat_idx*TILE_BITS +: TILE_BITS] <= data_in;
                    in_beat_idx <= in_beat_idx + 1'b1;
                end else begin
                    in_beat_idx <= 0;
                    // advance input pixel coord (raster over the full 56x56)
                    if (icol == IW_FULL-1) begin
                        icol <= 0;
                        irow <= (irow == IH_FULL-1) ? {{ROW_W{{1'b0}}}} : irow + 1'b1;
                    end else begin
                        icol <= icol + 1'b1;
                    end
                end
            end
            if (lib_valid_out_w && !out_busy) begin
                out_pix  <= lib_data_out_w;
                out_idx  <= 0;
                out_busy <= 1'b1;
            end else if (out_busy && ready_out) begin
                if (out_idx == OUT_BEATS-1) out_busy <= 1'b0;
                else                        out_idx  <= out_idx + 1'b1;
            end
        end
    end

    wire stall_in = mac_busy || out_busy;
    // INNER conv: stride-1 28x28 1x1 over the kept pixels.
    coord_scheduler #(.IH(OH),.IW(OW),.OH(OH),.OW(OW),.KH(KH),.KW(KW),.SH(SH),.SW(SW),.PH(PH),.PW(PW)) scheduler (
        .clk(clk),.rst_n(rst_n),.start(start_pulse),.stall_in(stall_in),
        .valid_in(lib_valid_in_w),.ready_in(sched_ready_in),
        .needs_real_input(sched_needs_real_input),
        .in_row(sched_in_row),.in_col(sched_in_col),
        .output_fires(sched_output_fires),.advance(sched_advance),
        .in_frame_done(),.out_frame_done(sched_out_frame_done),
        .outputs_emitted(sched_outputs_emitted));
    line_buf_window #(.IC(IC),.IW(OW),.IH(OH),.KH(KH),.KW(KW),.PW(PW),.PH(PH)) lbw (
        .clk(clk),.rst_n(rst_n),.frame_start(start_pulse),
        .sched_in_row(sched_in_row),.sched_in_col(sched_in_col),
        .sched_needs_real_input(sched_needs_real_input),
        .sched_advance(sched_advance),.sched_output_fires(sched_output_fires),
        .valid_in(lib_valid_in_w),.data_in(lib_data_in_w),.window_flat(window_flat));
    conv_datapath_mp_k #(.IC(IC),.OC(OC),.KH(KH),.KW(KW),.K_TOTAL(K_TOTAL),.MP(MP),
        .MP_K(MP_K),.SCALE_MULT(SCALE_MULT),.SCALE_SHIFT(SCALE_SHIFT),
        .WEIGHTS_PATH("{wpath}"),
        .BIAS_PATH("{bpath}")) dp (
        .clk(clk),.rst_n(rst_n),.window_flat(window_flat),
        .start_mac(sched_output_fires),
        .valid_out(lib_valid_out_w),.data_out(lib_data_out_w),.mac_busy(mac_busy));
endmodule
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--equiv", action="store_true", help="emit ready_out-tied-high variant")
    args = ap.parse_args()
    (RTL / "node_conv_224.v").write_text(gen(tie_ready_high=args.equiv))
    print(f"[{'equiv' if args.equiv else 'ok'}] node_conv_224.v decimator "
          f"(56x56->28x28 s2, MP={MP} MP_K={MP_K})")


if __name__ == "__main__":
    main()
