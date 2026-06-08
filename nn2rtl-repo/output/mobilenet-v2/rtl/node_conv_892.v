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
    localparam IH = 7, IW = 7, OH = 7, OW = 7;
    localparam K_TOTAL = IC;
    localparam MP = 4;
    localparam OC_PASSES = (OC + MP - 1) / MP;
    localparam CHANNEL_TILE = 32;
    localparam IN_BEATS  = (IC + CHANNEL_TILE - 1) / CHANNEL_TILE;
    localparam OUT_BEATS = (OC + CHANNEL_TILE - 1) / CHANNEL_TILE;
    localparam TILE_BITS = CHANNEL_TILE * 8;
    localparam SCALE_MULT  = 18771;
    localparam SCALE_SHIFT = 22;

    localparam integer PROD_W        = 16;
    localparam integer ACC_W         = PROD_W + 10;
    localparam integer BIAS_W        = 32;
    localparam integer BIASED_W      = ((ACC_W > BIAS_W) ? ACC_W : BIAS_W) + 1;
    localparam integer SCALE_CONST_W = 16;
    localparam integer SCALED_W      = BIASED_W + SCALE_CONST_W;

    localparam signed [SCALE_CONST_W-1:0] SCALE_MULT_CONST = SCALE_MULT;
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};

    (* rom_style = "block", ram_style = "block" *)
    reg signed [7:0]  weights [0:OC*K_TOTAL-1];
    (* rom_style = "block", ram_style = "block" *)
    reg signed [31:0] biases  [0:OC-1];

    initial begin
        $readmemh("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_892_weights.hex", weights);
        $readmemh("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_892_bias.hex",    biases);
    end

    reg signed [7:0] in_latch [0:IC-1];

    localparam [2:0] ST_LOAD=3'd0, ST_RUNNING=3'd1, ST_BIAS=3'd2, ST_SCALE=3'd3, ST_OUTPUT=3'd4, ST_EMIT=3'd5;
    reg [2:0] state;

    reg [4:0] in_beat_idx;
    reg [2:0] out_beat_idx;
    reg [5:0] oc_group;
    reg [1:0] lane_counter;
    reg [9:0] k_counter;
    reg [5:0] active_emit_count;
    reg [2:0] pixel_row;
    reg [2:0] pixel_col;

    reg signed [7:0]            weight_q;
    reg signed [7:0]            in_q;
    reg                         mac_valid_q1, mac_valid_q2;
    reg [1:0]                   mac_lane_q1, mac_lane_q2;
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] mul_q;
    reg                         mac_done_issuing;

    reg signed [ACC_W-1:0]    acc    [0:MP-1];
    reg signed [BIASED_W-1:0] biased [0:MP-1];
    reg signed [SCALED_W-1:0] scaled [0:MP-1];
    reg signed [SCALED_W-1:0] v_tmp;
    integer                   global_oc;

    reg [OC*8-1:0] out_pack;

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ready_in<=1'b1; valid_out<=1'b0; data_out<=256'd0;
            state<=ST_LOAD; in_beat_idx<=0; out_beat_idx<=0; oc_group<=0;
            lane_counter<=0; k_counter<=0; active_emit_count<=0;
            pixel_row<=0; pixel_col<=0;
            weight_q<=0; in_q<=0; mac_valid_q1<=0; mac_valid_q2<=0;
            mac_lane_q1<=0; mac_lane_q2<=0; mul_q<=0; mac_done_issuing<=0;
            v_tmp<=0;
            for (i=0;i<MP;i=i+1) begin acc[i]<=0; biased[i]<=0; scaled[i]<=0; end
            out_pack<=0;
        end else begin
            case (state)
                ST_LOAD: if (valid_in && ready_in) begin
                    for (i=0;i<CHANNEL_TILE;i=i+1) in_latch[in_beat_idx*CHANNEL_TILE+i]<=data_in[i*8 +: 8];
                    if (in_beat_idx==IN_BEATS-1) begin
                        ready_in<=1'b0; in_beat_idx<=0;
                        for (i=0;i<MP;i=i+1) acc[i]<=0;
                        oc_group<=0; lane_counter<=0; k_counter<=0;
                        mac_done_issuing<=0; mac_valid_q1<=0; mac_valid_q2<=0;
                        state<=ST_RUNNING;
                    end else in_beat_idx<=in_beat_idx+1;
                end
                ST_RUNNING: begin
                    if (!mac_done_issuing) begin
                        weight_q<=weights[(oc_group*MP+lane_counter)*K_TOTAL+k_counter];
                        in_q<=in_latch[k_counter];
                        mac_valid_q1<=1'b1; mac_lane_q1<=lane_counter;
                        if (k_counter==K_TOTAL-1) begin
                            k_counter<=0;
                            if (lane_counter==MP-1) begin lane_counter<=0; mac_done_issuing<=1'b1; end
                            else lane_counter<=lane_counter+1;
                        end else k_counter<=k_counter+1;
                    end else mac_valid_q1<=1'b0;
                    mul_q<=weight_q*in_q;
                    mac_valid_q2<=mac_valid_q1; mac_lane_q2<=mac_lane_q1;
                    if (mac_valid_q2) acc[mac_lane_q2]<=acc[mac_lane_q2]+mul_q;
                    if (mac_done_issuing && !mac_valid_q1 && !mac_valid_q2) state<=ST_BIAS;
                end
                ST_BIAS: begin
                    for (i=0;i<MP;i=i+1) biased[i]<=$signed(acc[i])+$signed(biases[oc_group*MP+i]);
                    state<=ST_SCALE;
                end
                ST_SCALE: begin
                    for (i=0;i<MP;i=i+1) scaled[i]<=$signed(biased[i])*$signed(SCALE_MULT_CONST);
                    state<=ST_OUTPUT;
                end
                ST_OUTPUT: begin
                    for (i=0;i<MP;i=i+1) begin
                        global_oc=oc_group*MP+i;
                        v_tmp=(scaled[i]+(scaled[i][SCALED_W-1]?SCALE_ROUND_HALF_M1:SCALE_ROUND_HALF))>>>SCALE_SHIFT;
                        out_pack[global_oc*8 +: 8]<=(v_tmp>127)?8'sd127:(v_tmp<-128)?-8'sd128:v_tmp[7:0];
                    end
                    if (oc_group==OC_PASSES-1) begin
                        valid_out<=1'b1; data_out<=out_pack[0 +: TILE_BITS]; out_beat_idx<=0; state<=ST_EMIT;
                    end else begin
                        oc_group<=oc_group+1; lane_counter<=0; k_counter<=0; mac_done_issuing<=0;
                        for (i=0;i<MP;i=i+1) acc[i]<=0;
                        state<=ST_RUNNING;
                    end
                end
                ST_EMIT: if (out_beat_idx==OUT_BEATS-1) begin
                    valid_out<=1'b0; ready_in<=1'b1; out_beat_idx<=0; oc_group<=0;
                    if (active_emit_count==OH*OW-1) begin active_emit_count<=0; pixel_row<=0; pixel_col<=0; end
                    else begin active_emit_count<=active_emit_count+1;
                        if (pixel_col==IW-1) begin pixel_col<=0; pixel_row<=pixel_row+1; end else pixel_col<=pixel_col+1;
                    end
                    state<=ST_LOAD;
                end else begin
                    valid_out<=1'b1; data_out<=out_pack[(out_beat_idx+1)*TILE_BITS +: TILE_BITS]; out_beat_idx<=out_beat_idx+1;
                end
                default: state<=ST_LOAD;
            endcase
        end
    end
endmodule
