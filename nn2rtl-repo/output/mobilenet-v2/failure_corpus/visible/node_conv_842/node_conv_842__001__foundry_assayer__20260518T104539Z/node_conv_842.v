`timescale 1ns / 1ps
`default_nettype none

// ---------------------------------------------------------------------------
// node_conv_842 -- MobileNet-v2 depthwise 3x3 stride-1 padding-1 conv
//   C=192, IH=IW=28, OH=OW=28, KH=KW=3, PH=PW=1, stride 1
//   MP=4, OC_PASSES=48, pass=42 cyc, FIRST_OUT_CYC=2048, COMPUTE_START=31
//   Contract: depthwise-conv (channel_tile=192=C, 1 beat per pixel)
//   Bus 1536b in/out. SCALE_MULT/2^SCALE_SHIFT = 23074/2^21 ~= 0.011000872
//   line_buf/out_buf banked 2 x depth-512 (under Vivado ~900k-bit cap).
// ---------------------------------------------------------------------------

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
    // (full source persisted to output/mobilenet-v2/rtl/node_conv_842.v)
endmodule

`default_nettype wire
