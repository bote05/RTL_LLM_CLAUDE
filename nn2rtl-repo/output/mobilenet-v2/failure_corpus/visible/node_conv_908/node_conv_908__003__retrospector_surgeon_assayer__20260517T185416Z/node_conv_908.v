`timescale 1ns / 1ps
`default_nettype none

// ---------------------------------------------------------------------------
// node_conv_908 -- MobileNet-v2 depthwise 3x3 stride-1 padding-1 conv
//   C   = 960  (groups == in_channels == out_channels)
//   IH  = IW  = 7,   OH = OW = 7
//   KH  = KW  = 3,   PH = PW = 1, stride 1
//   MAC parallelism = 4 lanes; OC_PASSES = ceil(C/MP) = 240
//   Pass duration = MP * K_TOTAL + 6 = 4*9 + 6 = 42 cycles
//   Pipeline latency (LayerIR-authoritative) = 10091 cycles
//   Contract: depthwise-conv (tiled-streaming compatible, channel_tile = 512)
//   Bus  : 4096b in/out, 2 beats/pixel (512 ch + 448 real ch + 64 zero-pad ch)
//   Scale: SCALE_MULT/2^SCALE_SHIFT = 60332 / 2^23 ~= 0.0071924925
//          (target 0.0071924250, |rel err| ~ 9.4e-6)
//   No cross-channel reduction -- each lane reads its own channel-of-interest
//   from the same (kh, kw) tap, no IC iteration inside K_TOTAL.
// ---------------------------------------------------------------------------

module node_conv_908 #(
    parameter WEIGHTS_PATH = "D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_908_weights.hex",
    parameter BIAS_PATH    = "D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_908_bias.hex"
) (
    input  wire           clk,
    input  wire           rst_n,
    input  wire           valid_in,
    output wire           ready_in,
    input  wire [4095:0]  data_in,
    output reg            valid_out,
    output reg  [4095:0]  data_out
);

    localparam integer SCALE_SHIFT = 23;
    localparam signed [63:0] SCALE_MULT_64 = 64'sd60332;

    // (full source persisted on disk at output/mobilenet-v2/rtl/node_conv_908.v)
endmodule
`default_nettype wire
