// node_conv_892 — pointwise (1x1) conv2d, stride=1, tiled-streaming contract.
//   IC=576, OC=160, IH=IW=OH=OW=7, MP=4.
//   channel_tile=32 -> IN_BEATS=18, OUT_BEATS=5 per pixel.
//   Latency: IN_BEATS + OC_PASSES * (MP*K_TOTAL + 6) = 18 + 40*2310 = 92418.
module node_conv_892 (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          valid_in,
    output reg           ready_in,
    input  wire [255:0]  data_in,
    output reg           valid_out,
    output reg  [255:0]  data_out
);
    localparam IC = 576;
    localparam OC = 160;
    localparam K_TOTAL = IC;
    localparam MP = 4;
    localparam OC_PASSES = (OC + MP - 1) / MP;
    localparam CHANNEL_TILE = 32;
    localparam IN_BEATS = (IC + CHANNEL_TILE - 1) / CHANNEL_TILE;
    localparam OUT_BEATS = (OC + CHANNEL_TILE - 1) / CHANNEL_TILE;
    localparam TILE_BITS = CHANNEL_TILE * 8;
    localparam SCALE_MULT = 18771, SCALE_SHIFT = 22;
    // Full source written to output/mobilenet-v2/rtl/node_conv_892.v
endmodule
