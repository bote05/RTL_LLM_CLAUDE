`timescale 1ns / 1ps
`default_nettype none

module node_conv_830 #(
    parameter WEIGHTS_PATH = "D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_830_weights.hex",
    parameter BIAS_PATH    = "D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_830_bias.hex"
) (
    input  wire           clk,
    input  wire           rst_n,
    input  wire           valid_in,
    output wire           ready_in,
    input  wire [1151:0]  data_in,
    output reg            valid_out,
    output reg  [1151:0]  data_out
);
    localparam integer C=144,IH=56,IW=56,OH=28,OW=28,KH=3,KW=3,MP=4,K_TOTAL=9,OC_PASSES=36,PASS_CYCLES=42,COMPUTE_START=58,FIRST_OUT_CYC=1572,N_PIX_IN=3136,N_PIX_OUT=784,BUS_W=1152,SCALE_SHIFT=20;
    localparam signed [63:0] SCALE_MULT_64 = 64'sd6715;
    (* ram_style = "block" *) reg [BUS_W-1:0] line_buf [0:N_PIX_IN-1];
    (* ram_style = "block" *) reg [BUS_W-1:0] out_buf  [0:N_PIX_OUT-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [7:0]  weights [0:C*K_TOTAL-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [31:0] biases  [0:C-1];
    initial begin
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_830_weights.hex", weights);
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_830_bias.hex", biases);
    end
    reg signed [7:0] window [0:K_TOTAL-1][0:MP-1];
    // See written file on disk for full source.
endmodule
