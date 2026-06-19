// node_conv_910 — pointwise 1x1 conv2d, tiled-streaming contract.
// IC=960, OC=320, IH=IW=7, OH=OW=7, stride=1, MP=4.
// channel_tile=32  ->  IN_BEATS=30, OUT_BEATS=10 per pixel.
// Latency: IN_BEATS + OC_PASSES*(MP*K_TOTAL + 6) = 30 + 80*3846 = 307710.
module node_conv_910 (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              valid_in,
    output reg               ready_in,
    input  wire [255:0]      data_in,
    output reg               valid_out,
    output reg  [255:0]      data_out
);
    localparam IC=960, OC=320, IH=7, IW=7, OH=7, OW=7;
    localparam KH=1, KW=1, SH=1, SW=1, PH=0, PW=0;
    localparam K_TOTAL=IC*KH*KW, MP=4, OC_PASSES=(OC+MP-1)/MP;
    localparam CHANNEL_TILE=32, IN_BEATS=(IC+CHANNEL_TILE-1)/CHANNEL_TILE;
    localparam OUT_BEATS=(OC+CHANNEL_TILE-1)/CHANNEL_TILE, TILE_BITS=CHANNEL_TILE*8;
    localparam TOTAL_PIXELS=OH*OW, SCALE_MULT=21804, SCALE_SHIFT=22;
    // Full RTL persisted to output/mobilenet-v2/rtl/node_conv_910.v
endmodule
