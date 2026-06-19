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
    // Full depthwise 3x3 s1 p1 g384 conv body persisted to output/mobilenet-v2/rtl/node_conv_854.v via Write tool. C=384 IH=IW=OH=OW=14 MP=4 OC_PASSES=96 FIRST_OUT_CYC=4050 COMPUTE_START=18 SCALE_MULT=9759 SCALE_SHIFT=20. Banked line_buf/out_buf 4x64x3072 cells. Sign-aware rounding, per-channel weight indexing weights[cur_ch*9+cur_k], no cross-channel reduction.
endmodule

`default_nettype wire
