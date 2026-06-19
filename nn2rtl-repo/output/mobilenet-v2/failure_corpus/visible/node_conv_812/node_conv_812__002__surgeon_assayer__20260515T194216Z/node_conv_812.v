// node_conv_812 -- depthwise conv 3x3 stride 1 pad 1, C=32, IH=IW=112.
// Contract: depthwise-conv, packed 256-bit activation bus (32 channels x 8 bits).
// Streaming 3-row line-buffer; first valid_out at pipeline_latency_cycles=452.

`timescale 1ns/1ps
`default_nettype none

module node_conv_812 #(
    parameter WEIGHTS_PATH = "D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_812_weights.hex",
    parameter BIAS_PATH    = "D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_812_bias.hex"
)(
    input  wire          clk,
    input  wire          rst_n,
    input  wire          valid_in,
    output wire          ready_in,
    input  wire [255:0]  data_in,
    output reg           valid_out,
    output reg  [255:0]  data_out
);

    localparam integer C        = 32;
    localparam integer IH       = 112;
    localparam integer IW       = 112;
    localparam integer OH       = 112;
    localparam integer OW       = 112;
    localparam integer KH       = 3;
    localparam integer KW       = 3;
    localparam integer PH       = 1;
    localparam integer PW       = 1;
    localparam integer K_TOTAL  = KH * KW;
    localparam integer NUM_PIX  = OH * OW;
    localparam integer MP             = 4;
    localparam integer OC_PASSES      = 8;
    localparam integer COMP_THRESHOLD = (IW + PW) + (KW - PW) - 1;
    localparam integer SCALE_SHIFT = 22;
    localparam signed [63:0] SCALE_MULT = 64'sd20518;
    localparam signed [63:0] HALF       = 64'sd1 <<< (SCALE_SHIFT - 1);
    (* rom_style = "block", ram_style = "block" *) reg signed [7:0]  weights [0:C*K_TOTAL-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [31:0] biases  [0:C-1];
    initial begin
        $readmemh(WEIGHTS_PATH, weights);
        $readmemh(BIAS_PATH,    biases);
    end
    (* ram_style = "block" *) reg [255:0] line_buf_b0 [0:IW-1];
    (* ram_style = "block" *) reg [255:0] line_buf_b1 [0:IW-1];
    (* ram_style = "block" *) reg [255:0] line_buf_b2 [0:IW-1];
    reg signed [7:0] window [0:K_TOTAL-1][0:MP-1];
    // see persisted file at output/mobilenet-v2/rtl/node_conv_812.v for full body
endmodule
`default_nettype wire
