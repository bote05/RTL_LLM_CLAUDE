// Tiled-streaming pointwise (1x1) conv2d.
//   IC=512, OC=2048, IH=IW=7, stride=1, MP=4, channel_tile=32.
//   IN_BEATS=16, OUT_BEATS=64 per pixel. Latency=16+512*2054=1051664.
module node_conv_294 (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              valid_in,
    output reg               ready_in,
    input  wire [255:0]      data_in,
    output reg               valid_out,
    output reg  [255:0]      data_out
);
    localparam IC = 512;
    localparam OC = 2048;
    localparam IH = 7;
    localparam IW = 7;
    localparam OH = 7;
    localparam OW = 7;
    localparam KH = 1;
    localparam KW = 1;
    localparam SH = 1;
    localparam SW = 1;
    localparam PH = 0;
    localparam PW = 0;
    localparam K_TOTAL      = IC * KH * KW;
    localparam MP           = 4;
    localparam OC_PASSES    = (OC + MP - 1) / MP;
    localparam CHANNEL_TILE = 32;
    localparam TILE_BITS    = CHANNEL_TILE * 8;
    localparam IN_BEATS     = (IC + CHANNEL_TILE - 1) / CHANNEL_TILE;
    localparam OUT_BEATS    = (OC + CHANNEL_TILE - 1) / CHANNEL_TILE;
    localparam NUM_WEIGHTS   = OC * K_TOTAL;
    localparam WEIGHT_ADDR_W = $clog2(NUM_WEIGHTS);
    localparam OC_INDEX_W    = $clog2(OC + MP);
    localparam SCALE_MULT  = 32'd19837;
    localparam SCALE_SHIFT = 5'd21;
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
    localparam ST_LOAD    = 3'd0;
    localparam ST_RUNNING = 3'd1;
    localparam ST_BIAS    = 3'd2;
    localparam ST_SCALE   = 3'd3;
    localparam ST_OUTPUT  = 3'd4;
    localparam ST_EMIT    = 3'd5;
    (* rom_style = "block", ram_style = "block" *) reg signed [7:0]  weights [0:NUM_WEIGHTS-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [31:0] biases  [0:OC-1];
    initial begin
        $readmemh("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_294_weights_wide.hex", weights);
        $readmemh("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_294_bias.hex", biases);
    end
    reg signed [7:0] in_latch [0:IC-1];
    reg [OC*8-1:0] out_pack;
    reg signed [ACC_W-1:0]    acc    [0:MP-1];
    reg signed [BIASED_W-1:0] biased [0:MP-1];
    reg signed [SCALED_W-1:0] scaled [0:MP-1];
    reg signed [SCALED_W-1:0] v_tmp;
    reg [$clog2(K_TOTAL+1)-1:0]   k_counter;
    reg [$clog2(MP+1)-1:0]        lane_counter;
    reg [$clog2(OC_PASSES+1)-1:0] oc_group;
    reg [$clog2(IN_BEATS+1)-1:0]  in_beat_idx;
    reg [$clog2(OUT_BEATS+1)-1:0] out_beat_idx;
    reg [2:0] state;
    reg signed [7:0]              weight_q;
    wire [OC_INDEX_W-1:0]         current_global_oc;
    wire [WEIGHT_ADDR_W-1:0]      weight_read_addr;
    assign current_global_oc = oc_group * MP + lane_counter;
    assign weight_read_addr  = current_global_oc * K_TOTAL + k_counter;
    always @(posedge clk) begin
        weight_q <= weights[weight_read_addr];
    end
    reg                            mac_valid_q1;
    reg [$clog2(MP+1)-1:0]         mac_lane_q1;
    reg [$clog2(K_TOTAL+1)-1:0]    mac_k_q1;
    reg [OC_INDEX_W-1:0]           mac_global_oc_q1;
    reg                            mac_done_issuing;
    reg                            mac_valid_q2;
    reg [$clog2(MP+1)-1:0]         mac_lane_q2;
    reg [OC_INDEX_W-1:0]           mac_global_oc_q2;
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] mul_q;
    integer i, lane;
    integer bias_oc;
    integer out_oc;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= ST_LOAD;
            ready_in         <= 1'b1;
            valid_out        <= 1'b0;
            k_counter        <= 0;
            lane_counter     <= 0;
            oc_group         <= 0;
            in_beat_idx      <= 0;
            out_beat_idx     <= 0;
            data_out         <= {TILE_BITS{1'b0}};
            out_pack         <= {(OC*8){1'b0}};
            mac_valid_q1     <= 1'b0;
            mac_lane_q1      <= 0;
            mac_k_q1         <= 0;
            mac_global_oc_q1 <= 0;
            mac_valid_q2     <= 1'b0;
            mac_lane_q2      <= 0;
            mac_global_oc_q2 <= 0;
            mac_done_issuing <= 1'b0;
            mul_q            <= 0;
            for (i = 0; i < IC; i = i + 1) in_latch[i] <= 8'sd0;
            for (lane = 0; lane < MP; lane = lane + 1) begin acc[lane] <= 0; biased[lane] <= 0; scaled[lane] <= 0; end
        end else begin
            mul_q            <= $signed(weight_q) * $signed(in_latch[mac_k_q1]);
            mac_valid_q2     <= mac_valid_q1;
            mac_lane_q2      <= mac_lane_q1;
            mac_global_oc_q2 <= mac_global_oc_q1;
            if (mac_valid_q2 && mac_global_oc_q2 < OC) acc[mac_lane_q2] <= acc[mac_lane_q2] + $signed(mul_q);
            case (state)
            ST_LOAD: begin valid_out <= 1'b0; mac_valid_q1 <= 1'b0; mac_valid_q2 <= 1'b0; if (valid_in) begin for (i = 0; i < CHANNEL_TILE; i = i + 1) in_latch[in_beat_idx * CHANNEL_TILE + i] <= $signed(data_in[i*8 +: 8]); if (in_beat_idx == IN_BEATS - 1) begin in_beat_idx <= 0; ready_in <= 1'b0; k_counter <= 0; lane_counter <= 0; oc_group <= 0; out_beat_idx <= 0; mac_done_issuing <= 1'b0; for (lane = 0; lane < MP; lane = lane + 1) acc[lane] <= 0; state <= ST_RUNNING; end else in_beat_idx <= in_beat_idx + 1; end end
            ST_RUNNING: begin if (mac_done_issuing) begin mac_valid_q1 <= 1'b0; if (!mac_valid_q1 && !mac_valid_q2) begin mac_done_issuing <= 1'b0; state <= ST_BIAS; end end else begin mac_lane_q1 <= lane_counter; mac_k_q1 <= k_counter; mac_global_oc_q1 <= current_global_oc; mac_valid_q1 <= 1'b1; if (lane_counter == MP - 1) begin lane_counter <= 0; if (k_counter == K_TOTAL - 1) mac_done_issuing <= 1'b1; else k_counter <= k_counter + 1; end else lane_counter <= lane_counter + 1; end end
            ST_BIAS: begin for (lane = 0; lane < MP; lane = lane + 1) begin bias_oc = oc_group * MP + lane; if (bias_oc < OC) biased[lane] <= $signed(acc[lane]) + $signed(biases[bias_oc]); else biased[lane] <= 0; end state <= ST_SCALE; end
            ST_SCALE: begin for (lane = 0; lane < MP; lane = lane + 1) scaled[lane] <= $signed(biased[lane]) * $signed(SCALE_MULT_CONST); state <= ST_OUTPUT; end
            ST_OUTPUT: begin for (lane = 0; lane < MP; lane = lane + 1) begin out_oc = oc_group * MP + lane; if (out_oc < OC) begin v_tmp = (scaled[lane] + (scaled[lane][SCALED_W-1] ? SCALE_ROUND_HALF_M1 : SCALE_ROUND_HALF)) >>> SCALE_SHIFT; out_pack[out_oc*8 +: 8] <= (v_tmp > 127) ? 8'sd127 : (v_tmp < -128) ? -8'sd128 : v_tmp[7:0]; end end if (oc_group < OC_PASSES - 1) begin for (lane = 0; lane < MP; lane = lane + 1) acc[lane] <= 0; k_counter <= 0; lane_counter <= 0; oc_group <= oc_group + 1; state <= ST_RUNNING; end else begin valid_out <= 1'b1; data_out <= out_pack[0 +: TILE_BITS]; out_beat_idx <= 1; oc_group <= 0; state <= ST_EMIT; end end
            ST_EMIT: begin if (out_beat_idx == OUT_BEATS) begin valid_out <= 1'b0; ready_in <= 1'b1; out_beat_idx <= 0; state <= ST_LOAD; end else begin data_out <= out_pack[out_beat_idx * TILE_BITS +: TILE_BITS]; out_beat_idx <= out_beat_idx + 1; end end
            default: state <= ST_LOAD;
            endcase
        end
    end
endmodule
