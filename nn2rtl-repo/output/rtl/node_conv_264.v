// node_conv_252 - 1x1 pointwise conv2d, stride=1, tiled-streaming contract.
//   IC=1024, OC=256, IH=IW=14, OH=OW=14, MP=4, channel_tile=32.
//   IN_BEATS=32, OUT_BEATS=8, OC_PASSES=64.
//   Latency: IN_BEATS + OC_PASSES*(MP*K_TOTAL + 6) = 32 + 64*4102 = 262560.
module node_conv_264 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    output reg          ready_in,
    input  wire [255:0] data_in,
    output reg          valid_out,
    output reg  [255:0] data_out
);

    localparam IC = 1024;
    localparam OC = 256;
    localparam IH = 14;
    localparam IW = 14;
    localparam OH = 14;
    localparam OW = 14;
    localparam KH = 1;
    localparam KW = 1;
    localparam K_TOTAL = IC * KH * KW;
    localparam MP = 4;
    localparam OC_PASSES = (OC + MP - 1) / MP;
    localparam CHANNEL_TILE = 32;
    localparam IN_BEATS  = (IC + CHANNEL_TILE - 1) / CHANNEL_TILE;
    localparam OUT_BEATS = (OC + CHANNEL_TILE - 1) / CHANNEL_TILE;
    localparam TILE_BITS = CHANNEL_TILE * 8;
    localparam SCALE_MULT  = 32'd12775;
    localparam SCALE_SHIFT = 5'd20;
    localparam integer PROD_W        = 16;
    localparam integer ACC_W         = PROD_W + $clog2(K_TOTAL);
    localparam integer BIAS_W        = 32;
    localparam integer BIASED_W      = ((ACC_W > BIAS_W) ? ACC_W : BIAS_W) + 1;
    localparam integer SCALE_MAG_W   = $clog2(SCALE_MULT + 1);
    localparam integer SCALE_CONST_W = SCALE_MAG_W + 1;
    localparam integer SCALED_W      = BIASED_W + SCALE_CONST_W;
    localparam signed [SCALE_CONST_W-1:0] SCALE_MULT_CONST = SCALE_MULT;
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};
    (* rom_style = "block", ram_style = "block" *) reg signed [7:0]  weights [0:OC*K_TOTAL-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [31:0] biases  [0:OC-1];
    initial begin
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_264_weights_wide.hex", weights);
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_264_bias.hex", biases);
    end
    reg [IC*8-1:0] in_latch;
    reg [OC*8-1:0] out_pack;
    localparam ST_LOAD=3'd0, ST_RUNNING=3'd1, ST_BIAS=3'd2, ST_SCALE=3'd3, ST_OUTPUT=3'd4, ST_EMIT=3'd5;
    reg [2:0] state;
    reg [5:0]  in_beat_idx;
    reg [4:0]  out_beat_idx;
    reg [3:0]  pixel_row;
    reg [3:0]  pixel_col;
    reg [7:0]  active_emit_count;
    reg [6:0]  oc_group;
    reg [1:0]  lane_counter;
    reg [10:0] k_counter;
    reg signed [7:0] weight_q, data_q;
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] mul_q;
    reg [1:0] mac_lane_q1, mac_lane_q2;
    reg mac_valid_q1, mac_valid_q2, mac_done_issuing;
    reg [1:0] drain_count;
    reg signed [ACC_W-1:0]    acc    [0:MP-1];
    reg signed [BIASED_W-1:0] biased [0:MP-1];
    reg signed [SCALED_W-1:0] scaled [0:MP-1];
    reg signed [SCALED_W-1:0] v_tmp;
    integer i, lane, global_oc;
    reg [13:0] cur_global_oc_w;
    reg [23:0] weight_addr;
    reg signed [7:0] cur_data;
    always @* begin
        cur_global_oc_w = oc_group * MP + lane_counter;
        weight_addr     = cur_global_oc_w * K_TOTAL + k_counter;
        cur_data        = in_latch[k_counter*8 +: 8];
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state<=ST_LOAD; ready_in<=1'b1; valid_out<=1'b0; data_out<=256'd0;
            in_beat_idx<=0; out_beat_idx<=0; pixel_row<=0; pixel_col<=0;
            active_emit_count<=0; oc_group<=0; lane_counter<=0; k_counter<=0;
            in_latch<={(IC*8){1'b0}}; out_pack<={(OC*8){1'b0}};
            weight_q<=0; data_q<=0; mul_q<=0;
            mac_lane_q1<=0; mac_lane_q2<=0; mac_valid_q1<=0; mac_valid_q2<=0;
            mac_done_issuing<=0; drain_count<=0;
            for (i=0;i<MP;i=i+1) begin acc[i]<=0; biased[i]<=0; scaled[i]<=0; end
        end else begin
            mac_valid_q2 <= mac_valid_q1;
            mac_lane_q2  <= mac_lane_q1;
            mac_valid_q1 <= 1'b0;
            mul_q        <= weight_q * data_q;
            if (mac_valid_q2) acc[mac_lane_q2] <= acc[mac_lane_q2] + mul_q;
            case (state)
                ST_LOAD: if (valid_in && ready_in) begin
                    in_latch[in_beat_idx*TILE_BITS +: TILE_BITS] <= data_in;
                    if (in_beat_idx == IN_BEATS - 1) begin
                        in_beat_idx<=0; ready_in<=1'b0;
                        oc_group<=0; lane_counter<=0; k_counter<=0;
                        mac_done_issuing<=0; drain_count<=0;
                        for (i=0;i<MP;i=i+1) acc[i]<=0;
                        state<=ST_RUNNING;
                    end else in_beat_idx<=in_beat_idx+1'b1;
                end
                ST_RUNNING: begin
                    if (!mac_done_issuing) begin
                        weight_q<=weights[weight_addr]; data_q<=cur_data;
                        mac_valid_q1<=1'b1; mac_lane_q1<=lane_counter;
                        if (k_counter == K_TOTAL-1) begin
                            k_counter<=0;
                            if (lane_counter == MP-1) begin lane_counter<=0; mac_done_issuing<=1'b1; end
                            else lane_counter<=lane_counter+1'b1;
                        end else k_counter<=k_counter+1'b1;
                    end else begin
                        if (drain_count == 2'd2) state<=ST_BIAS;
                        else drain_count<=drain_count+1'b1;
                    end
                end
                ST_BIAS: begin
                    for (lane=0;lane<MP;lane=lane+1) biased[lane] <= acc[lane] + biases[oc_group*MP+lane];
                    state<=ST_SCALE;
                end
                ST_SCALE: begin
                    for (lane=0;lane<MP;lane=lane+1) scaled[lane] <= biased[lane] * SCALE_MULT_CONST;
                    state<=ST_OUTPUT;
                end
                ST_OUTPUT: begin
                    for (lane=0;lane<MP;lane=lane+1) begin
                        global_oc = oc_group*MP+lane;
                        v_tmp = (scaled[lane] + (scaled[lane][SCALED_W-1] ? SCALE_ROUND_HALF_M1 : SCALE_ROUND_HALF)) >>> SCALE_SHIFT;
                        if (v_tmp > 47'sd127) out_pack[global_oc*8 +: 8] <= 8'sd127;
                        else if (v_tmp < -47'sd128) out_pack[global_oc*8 +: 8] <= -8'sd128;
                        else out_pack[global_oc*8 +: 8] <= v_tmp[7:0];
                    end
                    if (oc_group == OC_PASSES-1) begin
                        valid_out<=1'b1; data_out<=out_pack[0 +: TILE_BITS];
                        out_beat_idx<=5'd1; state<=ST_EMIT;
                    end else begin
                        oc_group<=oc_group+1'b1; lane_counter<=0; k_counter<=0;
                        mac_done_issuing<=0; drain_count<=0;
                        for (i=0;i<MP;i=i+1) acc[i]<=0;
                        state<=ST_RUNNING;
                    end
                end
                ST_EMIT: begin
                    if (out_beat_idx < OUT_BEATS) begin
                        data_out<=out_pack[out_beat_idx*TILE_BITS +: TILE_BITS];
                        out_beat_idx<=out_beat_idx+1'b1;
                    end else begin
                        valid_out<=1'b0; ready_in<=1'b1; out_beat_idx<=0;
                        if (active_emit_count == OH*OW-1) begin
                            active_emit_count<=0; pixel_row<=0; pixel_col<=0;
                        end else begin
                            active_emit_count<=active_emit_count+1'b1;
                            if (pixel_col == OW-1) begin pixel_col<=0; pixel_row<=pixel_row+1'b1; end
                            else pixel_col<=pixel_col+1'b1;
                        end
                        state<=ST_LOAD;
                    end
                end
                default: state<=ST_LOAD;
            endcase
        end
    end
endmodule
