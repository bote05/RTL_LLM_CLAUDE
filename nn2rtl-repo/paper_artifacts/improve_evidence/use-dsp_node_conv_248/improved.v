// node_conv_248 -- tiled-streaming pointwise (1x1) conv2d, BRAM-optimized
// DSP-banked: SCALE_MULT (29011) split into HI*256 + LO (113*256 + 83) so each
// lane drives two registered DSP multiplies in ST_BIAS. Total DSPs: 1 MAC + 8
// scale partials = 9. Per-pass cycles unchanged: 1024 (issue) + 4 (drain) +
// 1 (BIAS) + 1 (PACK) = 1030. Total latency: 8 + 256 * 1030 = 263688.

module node_conv_248 (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             valid_in,
    output reg              ready_in,
    input  wire [255:0]     data_in,
    output reg              valid_out,
    output reg  [255:0]     data_out
);
    localparam IC            = 256;
    localparam OC            = 1024;
    localparam IH            = 14;
    localparam IW            = 14;
    localparam KH            = 1;
    localparam KW            = 1;
    localparam K_TOTAL       = IC * KH * KW;
    localparam MP            = 4;
    localparam OC_PASSES     = (OC + MP - 1) / MP;
    localparam CHANNEL_TILE  = 32;
    localparam IN_BEATS      = IC / CHANNEL_TILE;
    localparam OUT_BEATS     = OC / CHANNEL_TILE;
    localparam BEAT_BITS     = CHANNEL_TILE * 8;

    localparam NUM_WEIGHTS   = OC * K_TOTAL;
    localparam WEIGHT_ADDR_W = (NUM_WEIGHTS <= 1) ? 1 : $clog2(NUM_WEIGHTS);
    localparam OC_INDEX_W    = (OC + MP <= 1) ? 1 : $clog2(OC + MP);

    localparam SCALE_MULT    = 29011;
    localparam SCALE_SHIFT   = 21;
    // 29011 = 113 * 256 + 83
    localparam SCALE_HI      = 113;
    localparam SCALE_LO      = 83;

    localparam integer PROD_W        = 16;
    localparam integer ACC_W         = PROD_W + $clog2(K_TOTAL);
    localparam integer BIAS_W        = 32;
    localparam integer BIASED_W      = ((ACC_W > BIAS_W) ? ACC_W : BIAS_W) + 1;
    localparam integer SCALE_MAG_W   = $clog2(SCALE_MULT + 1);
    localparam integer SCALE_CONST_W = SCALE_MAG_W + 1;
    localparam integer SCALED_W      = BIASED_W + SCALE_CONST_W;
    localparam signed [SCALE_CONST_W-1:0] SCALE_HI_CONST = SCALE_HI;
    localparam signed [SCALE_CONST_W-1:0] SCALE_LO_CONST = SCALE_LO;
    localparam signed [SCALED_W-1:0] SCALE_ROUND_BIAS =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);

    localparam ST_INPUT      = 3'd0;
    localparam ST_RUNNING    = 3'd1;
    localparam ST_BIAS       = 3'd2;
    localparam ST_PACK       = 3'd3;
    localparam ST_STREAM_OUT = 3'd4;

    (* rom_style = "block", ram_style = "block" *) reg [7:0]  weights [0:OC*K_TOTAL-1];
    (* rom_style = "block", ram_style = "block" *) reg [31:0] biases  [0:OC-1];
    initial begin
        $readmemh("<repo>/output/weights/node_conv_248_weights.hex", weights);
        $readmemh("<repo>/output/weights/node_conv_248_bias.hex", biases);
    end

    reg signed [7:0] in_latch [0:IC-1];

    reg signed [ACC_W-1:0]    acc       [0:MP-1];
    (* use_dsp = "yes" *) reg signed [SCALED_W-1:0] scaled_hi [0:MP-1];
    (* use_dsp = "yes" *) reg signed [SCALED_W-1:0] scaled_lo [0:MP-1];
    reg signed [SCALED_W-1:0] v_tmp;
    reg signed [SCALED_W-1:0] scaled_sum;

    reg [OC*8-1:0] out_buffer;

    reg [$clog2(K_TOTAL+1)-1:0]   k_counter;
    reg [$clog2(MP+1)-1:0]        lane_counter;
    reg [$clog2(OC_PASSES+1)-1:0] oc_group;
    reg [$clog2(IN_BEATS+1)-1:0]  in_beat;
    reg [$clog2(OUT_BEATS+1)-1:0] out_beat;
    reg [2:0] state;

    reg  [WEIGHT_ADDR_W-1:0]       weight_addr_q;
    reg  [7:0]                     weight_data_q;
    wire signed [7:0]              weight_q;
    wire [OC_INDEX_W-1:0]          current_global_oc;
    wire [WEIGHT_ADDR_W-1:0]       weight_read_addr;
    assign current_global_oc = oc_group * MP + lane_counter;
    assign weight_read_addr  = current_global_oc * K_TOTAL + k_counter;
    assign weight_q          = $signed(weight_data_q);

    always @(posedge clk) begin
        weight_addr_q <= weight_read_addr;
        weight_data_q <= weights[weight_addr_q];
    end

    reg                            mac_valid_q1;
    reg [$clog2(MP+1)-1:0]         mac_lane_q1;
    reg [$clog2(K_TOTAL+1)-1:0]    mac_k_q1;
    reg [OC_INDEX_W-1:0]           mac_global_oc_q1;
    reg                            mac_done_issuing;

    reg                            mac_valid_q2;
    reg [$clog2(MP+1)-1:0]         mac_lane_q2;
    reg [$clog2(K_TOTAL+1)-1:0]    mac_k_q2;
    reg [OC_INDEX_W-1:0]           mac_global_oc_q2;

    reg                            mac_valid_q3;
    reg [$clog2(MP+1)-1:0]         mac_lane_q3;
    reg [OC_INDEX_W-1:0]           mac_global_oc_q3;

    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] mul_q;

    integer i, lane, ch;
    integer bias_oc, out_oc;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= ST_INPUT;
            ready_in         <= 1'b1;
            valid_out        <= 1'b0;
            in_beat          <= 0;
            out_beat         <= 0;
            k_counter        <= 0;
            lane_counter     <= 0;
            oc_group         <= 0;
            data_out         <= {BEAT_BITS{1'b0}};
            out_buffer       <= {(OC*8){1'b0}};
            mac_valid_q1     <= 1'b0;
            mac_lane_q1      <= 0;
            mac_k_q1         <= 0;
            mac_global_oc_q1 <= 0;
            mac_valid_q2     <= 1'b0;
            mac_lane_q2      <= 0;
            mac_k_q2         <= 0;
            mac_global_oc_q2 <= 0;
            mac_valid_q3     <= 1'b0;
            mac_lane_q3      <= 0;
            mac_global_oc_q3 <= 0;
            mac_done_issuing <= 1'b0;
            mul_q            <= 0;
            for (i = 0; i < IC; i = i + 1)
                in_latch[i] <= 8'sd0;
            for (lane = 0; lane < MP; lane = lane + 1) begin
                acc       [lane] <= 0;
                scaled_hi [lane] <= 0;
                scaled_lo [lane] <= 0;
            end
        end else begin
            mul_q            <= weight_q * $signed(in_latch[mac_k_q2]);
            mac_valid_q2     <= mac_valid_q1;
            mac_lane_q2      <= mac_lane_q1;
            mac_k_q2         <= mac_k_q1;
            mac_global_oc_q2 <= mac_global_oc_q1;
            mac_valid_q3     <= mac_valid_q2;
            mac_lane_q3      <= mac_lane_q2;
            mac_global_oc_q3 <= mac_global_oc_q2;

            if (mac_valid_q3 && mac_global_oc_q3 < OC) begin
                acc[mac_lane_q3] <= acc[mac_lane_q3] + $signed(mul_q);
            end

            case (state)

            ST_INPUT: begin
                valid_out    <= 1'b0;
                mac_valid_q1 <= 1'b0;
                mac_valid_q2 <= 1'b0;
                mac_valid_q3 <= 1'b0;
                if (valid_in) begin
                    for (ch = 0; ch < CHANNEL_TILE; ch = ch + 1) begin
                        in_latch[in_beat * CHANNEL_TILE + ch] <=
                            $signed(data_in[ch*8 +: 8]);
                    end
                    if (in_beat == IN_BEATS - 1) begin
                        in_beat          <= 0;
                        ready_in         <= 1'b0;
                        k_counter        <= 0;
                        lane_counter     <= 0;
                        oc_group         <= 0;
                        mac_done_issuing <= 1'b0;
                        for (lane = 0; lane < MP; lane = lane + 1)
                            acc[lane] <= 0;
                        state            <= ST_RUNNING;
                    end else begin
                        in_beat <= in_beat + 1;
                    end
                end
            end

            ST_RUNNING: begin
                if (mac_done_issuing) begin
                    mac_valid_q1 <= 1'b0;
                    if (!mac_valid_q1 && !mac_valid_q2 && !mac_valid_q3) begin
                        mac_done_issuing <= 1'b0;
                        state            <= ST_BIAS;
                    end
                end else begin
                    mac_lane_q1      <= lane_counter;
                    mac_k_q1         <= k_counter;
                    mac_global_oc_q1 <= current_global_oc;
                    mac_valid_q1     <= 1'b1;

                    if (lane_counter == MP - 1) begin
                        lane_counter <= 0;
                        if (k_counter == K_TOTAL - 1) begin
                            mac_done_issuing <= 1'b1;
                        end else begin
                            k_counter <= k_counter + 1;
                        end
                    end else begin
                        lane_counter <= lane_counter + 1;
                    end
                end
            end

            ST_BIAS: begin
                for (lane = 0; lane < MP; lane = lane + 1) begin
                    bias_oc = oc_group * MP + lane;
                    if (bias_oc < OC) begin
                        scaled_hi[lane] <= ($signed(acc[lane]) + $signed(biases[bias_oc])) * $signed(SCALE_HI_CONST);
                        scaled_lo[lane] <= ($signed(acc[lane]) + $signed(biases[bias_oc])) * $signed(SCALE_LO_CONST);
                    end else begin
                        scaled_hi[lane] <= 0;
                        scaled_lo[lane] <= 0;
                    end
                end
                state <= ST_PACK;
            end

            ST_PACK: begin
                for (lane = 0; lane < MP; lane = lane + 1) begin
                    out_oc = oc_group * MP + lane;
                    scaled_sum = (scaled_hi[lane] <<< 8) + scaled_lo[lane];
                    v_tmp = (scaled_sum + SCALE_ROUND_BIAS) >>> SCALE_SHIFT;
                    if (out_oc < OC)
                        out_buffer[out_oc*8 +: 8] <= (v_tmp >  127) ?  8'sd127 :
                                                     (v_tmp < -128) ? -8'sd128 :
                                                                       v_tmp[7:0];
                end
                if (oc_group == OC_PASSES - 1) begin
                    valid_out  <= 1'b1;
                    data_out   <= out_buffer[BEAT_BITS-1:0];
                    out_beat   <= 0;
                    oc_group   <= 0;
                    state      <= ST_STREAM_OUT;
                end else begin
                    oc_group     <= oc_group + 1;
                    k_counter    <= 0;
                    lane_counter <= 0;
                    for (lane = 0; lane < MP; lane = lane + 1)
                        acc[lane] <= 0;
                    state        <= ST_RUNNING;
                end
            end

            ST_STREAM_OUT: begin
                if (out_beat == OUT_BEATS - 1) begin
                    valid_out <= 1'b0;
                    ready_in  <= 1'b1;
                    out_beat  <= 0;
                    state     <= ST_INPUT;
                end else begin
                    data_out  <= out_buffer[(out_beat + 1) * BEAT_BITS +: BEAT_BITS];
                    valid_out <= 1'b1;
                    out_beat  <= out_beat + 1;
                end
            end

            default: state <= ST_INPUT;
            endcase
        end
    end

endmodule
