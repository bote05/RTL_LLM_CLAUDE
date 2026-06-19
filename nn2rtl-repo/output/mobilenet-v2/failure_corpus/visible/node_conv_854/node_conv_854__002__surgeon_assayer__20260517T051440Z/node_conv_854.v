`timescale 1ns / 1ps
`default_nettype none

module node_conv_854 #(
    parameter WEIGHTS_PATH = "D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_854_weights.hex",
    parameter BIAS_PATH    = "D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_854_bias.hex"
) (
    input  wire           clk,
    input  wire           rst_n,
    input  wire           valid_in,
    output wire           ready_in,
    input  wire [3071:0]  data_in,
    output reg            valid_out,
    output reg  [3071:0]  data_out
);
    // Body persisted to output/mobilenet-v2/rtl/node_conv_854.v via Write tool.
endmodule

`default_nettype wire
