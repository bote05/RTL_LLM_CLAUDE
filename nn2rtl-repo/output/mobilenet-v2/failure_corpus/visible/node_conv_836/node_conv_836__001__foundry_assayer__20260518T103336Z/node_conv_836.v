`timescale 1ns / 1ps
`default_nettype none

module node_conv_836 #(
    parameter WEIGHTS_PATH = "D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_836_weights.hex",
    parameter BIAS_PATH    = "D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_836_bias.hex"
) (
    input  wire           clk,
    input  wire           rst_n,
    input  wire           valid_in,
    output wire           ready_in,
    input  wire [1535:0]  data_in,
    output reg            valid_out,
    output reg  [1535:0]  data_out
);
    localparam integer C=192,IH=28,IW=28,OH=28,OW=28,KH=3,KW=3,SH=1,SW=1,PH=1,PW=1,MP=4,K_TOTAL=9,OC_PASSES=48,FIRST_OUT_CYC=2048,COMPUTE_START=31,N_PIX=784,BEAT_W=1536,SCALE_SHIFT=19;
    localparam signed [63:0] SCALE_MULT_64 = 64'sd21656;
    (* ram_style="block" *) reg [BEAT_W-1:0] line_buf_b0 [0:511];
    (* ram_style="block" *) reg [BEAT_W-1:0] line_buf_b1 [0:511];
    (* ram_style="block" *) reg [BEAT_W-1:0] out_buf_b0  [0:511];
    (* ram_style="block" *) reg [BEAT_W-1:0] out_buf_b1  [0:511];
    (* rom_style="block", ram_style="block" *) reg signed [7:0]  weights [0:C*K_TOTAL-1];
    (* rom_style="block", ram_style="block" *) reg signed [31:0] biases  [0:C-1];
    initial begin $readmemh(WEIGHTS_PATH, weights); $readmemh(BIAS_PATH, biases); end
    /* full RTL persisted to output/mobilenet-v2/rtl/node_conv_836.v */
endmodule
`default_nettype wire
