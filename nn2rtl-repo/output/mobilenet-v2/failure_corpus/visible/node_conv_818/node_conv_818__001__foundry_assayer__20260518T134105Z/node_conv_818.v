`timescale 1ns / 1ps
`default_nettype none

module node_conv_818 #(
    parameter WEIGHTS_PATH = "D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_818_weights.hex",
    parameter BIAS_PATH    = "D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_818_bias.hex"
) (
    input  wire           clk,
    input  wire           rst_n,
    input  wire           valid_in,
    output wire           ready_in,
    input  wire [767:0]   data_in,
    output reg            valid_out,
    output reg  [767:0]   data_out
);
    localparam integer C=96, IH=112, IW=112, OH=56, OW=56, KH=3, KW=3, SH=2, SW=2, PH=1, PW=1, MP=4, K_TOTAL=9, OC_PASSES=24, FIRST_OUT_CYC=1124, COMPUTE_START=115, N_IN_PIX=12544, N_OUT_PIX=3136, BEAT_W=768, SCALE_SHIFT=22;
    localparam signed [63:0] SCALE_MULT_64 = 64'sd16815;
    // Full module body persisted to output/mobilenet-v2/rtl/node_conv_818.v via Write tool (write_verilog MCP tool was not available in this dispatch).
endmodule
`default_nettype wire
