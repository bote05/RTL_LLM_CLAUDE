`timescale 1ns/1ps
// Sim wrapper: drives both FIFOs from C++ and exposes their ports.
module fifo_equiv_top #(parameter WIDTH=16, parameter DEPTH=64) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             feed_valid_s,
    input  wire             feed_valid_b,
    input  wire [WIDTH-1:0] feed_data,
    output wire             s_in_ready,
    output wire             b_in_ready,
    input  wire             s_out_ready,
    input  wire             b_out_ready,
    output wire             s_out_valid,
    output wire             b_out_valid,
    output wire [WIDTH-1:0] s_out_data,
    output wire [WIDTH-1:0] b_out_data
);
    skip_fifo #(.WIDTH(WIDTH), .DEPTH(DEPTH)) u_skip (
        .clk(clk), .rst_n(rst_n),
        .in_valid(feed_valid_s), .in_data(feed_data), .in_ready(s_in_ready),
        .out_valid(s_out_valid), .out_data(s_out_data), .out_ready(s_out_ready)
    );
    bram_fifo #(.WIDTH(WIDTH), .DEPTH(DEPTH)) u_bram (
        .clk(clk), .rst_n(rst_n),
        .in_valid(feed_valid_b), .in_data(feed_data), .in_ready(b_in_ready),
        .out_valid(b_out_valid), .out_data(b_out_data), .out_ready(b_out_ready)
    );
endmodule
