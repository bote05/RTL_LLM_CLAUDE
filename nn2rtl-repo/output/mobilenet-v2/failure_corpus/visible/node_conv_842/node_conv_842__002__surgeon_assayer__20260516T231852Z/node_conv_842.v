`timescale 1ns / 1ps
`default_nettype none

module node_conv_842 #(
    parameter WEIGHTS_PATH = "D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_842_weights.hex",
    parameter BIAS_PATH    = "D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_842_bias.hex"
) (
    input  wire           clk,
    input  wire           rst_n,
    input  wire           valid_in,
    output wire           ready_in,
    input  wire [1535:0]  data_in,
    output reg            valid_out,
    output reg  [1535:0]  data_out
);
    // See file at output/mobilenet-v2/rtl/node_conv_842.v for full body
endmodule

`default_nettype wire
