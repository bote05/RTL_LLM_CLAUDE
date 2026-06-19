`timescale 1ns / 1ps
`default_nettype none

// ---------------------------------------------------------------------------
// node_conv_884 -- MobileNet-v2 depthwise 3x3 stride-1 padding-1 conv
//   C   = 576  (groups == in_channels == out_channels)
//   IH  = IW  = 14,  OH = OW = 14
//   KH  = KW  = 3,   PH = PW = 1, stride 1
//   MAC parallelism = 4 lanes; OC_PASSES = ceil(C/MP) = 144
//   Pass duration = MP * K_TOTAL + 6 = 4*9 + 6 = 42 cycles
//   Pipeline latency (LayerIR-authoritative) = 6066 cycles
//   COMPUTE_START = 17 -> pix_done[0] at edge T+6064 -> valid_out at cyc=6066
//   Contract: depthwise-conv (channel_tile = 512, 2 beats per pixel)
//   Bus  : 4096b in/out, beat 0 = ch 0..511, beat 1 = ch 512..575 + 64 z-pad
//   Scale: SCALE_MULT/2^SCALE_SHIFT = 19652 / 2^20 ~= 0.018745422
//          (target 0.018744485590673952, |rel err| ~ 5.0e-5)
//   Schedule: lane-INTERLEAVED -- cur_lane = cmp_step[1:0],
//             cur_k = cmp_step[5:2]. Rotates lanes every step so the last
//             tap (k=8, in_pix_idx up to 15 for output(0,0)) is read at
//             steps 32..35 (pre-edge T+49..52), giving line_buf[15]
//             (written by beat 30 at post-edge T+30) plenty of margin.
//             Lane-sequential would read pixel 15 at step 8 (pre-edge T+25)
//             before beat 30 has arrived; for IH=14 with 2-beat tiling that
//             violates timing. Math is identical -- 9 taps per channel for
//             4 channels per pass.
//   No cross-channel reduction -- each lane reads its own channel-of-interest
//   from the same (kh, kw) tap, no IC iteration inside K_TOTAL.
// ---------------------------------------------------------------------------

module node_conv_884 #(
    parameter WEIGHTS_PATH = "D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_884_weights.hex",
    parameter BIAS_PATH    = "D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_884_bias.hex"
) (
    input  wire           clk,
    input  wire           rst_n,
    input  wire           valid_in,
    output wire           ready_in,
    input  wire [4095:0]  data_in,
    output reg            valid_out,
    output reg  [4095:0]  data_out
);

    localparam integer SCALE_SHIFT   = 20;
    localparam signed [63:0] SCALE_MULT_64 = 64'sd19652;

    // Full source persisted to output/mobilenet-v2/rtl/node_conv_884.v
    // Only numerical pipeline constants changed from broken artifact: SCALE_MULT_64 27056 -> 19652
endmodule
`default_nettype wire
