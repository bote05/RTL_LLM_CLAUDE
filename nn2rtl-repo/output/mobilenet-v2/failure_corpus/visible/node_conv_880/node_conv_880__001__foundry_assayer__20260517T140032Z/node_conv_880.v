// node_conv_880 — pointwise 1x1 conv2d, tiled-streaming contract.
//   IC=576, OC=96, IH=IW=14 -> OH=OW=14 (stride=1), MP=4.
//   channel_tile=32 => IN_BEATS=18, OUT_BEATS=3.
//   Latency = IN_BEATS + OC_PASSES*(MP*K_TOTAL+6) = 18 + 24*2310 = 55458.
module node_conv_880 (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              valid_in,
    output reg               ready_in,
    input  wire [255:0]      data_in,
    output reg               valid_out,
    output reg  [255:0]      data_out
);

    localparam integer IC=576, OC=96, OH=14, OW=14, KH=1, KW=1;
    localparam integer K_TOTAL=IC*KH*KW, MP=4, OC_PASSES=(OC+MP-1)/MP;
    localparam integer CHANNEL_TILE=32, TILE_BITS=CHANNEL_TILE*8;
    localparam integer IN_BEATS=(IC+CHANNEL_TILE-1)/CHANNEL_TILE;
    localparam integer OUT_BEATS=(OC+CHANNEL_TILE-1)/CHANNEL_TILE;
    localparam integer TOTAL_PIXELS=OH*OW;
    localparam integer SCALE_MULT=30293, SCALE_SHIFT=22;
    localparam integer PROD_W=16, ACC_W=PROD_W+10, BIAS_W=32;
    localparam integer BIASED_W=((ACC_W>BIAS_W)?ACC_W:BIAS_W)+1;
    localparam integer SCALE_MAG_W=15, SCALE_CONST_W=SCALE_MAG_W+1;
    localparam integer SCALED_W=BIASED_W+SCALE_CONST_W;
    // Full source persisted to output/mobilenet-v2/rtl/node_conv_880.v
endmodule
