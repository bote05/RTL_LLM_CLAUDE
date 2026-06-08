// node_conv_876 — pointwise (1x1) conv2d, tiled-streaming contract.
//   IC=96, OC=576, IH=IW=14, OH=OW=14, stride=1, MP=4.
//   channel_tile=32  =>  IN_BEATS=3, OUT_BEATS=18 per pixel.
//   Latency: IN_BEATS + OC_PASSES * (MP*K_TOTAL + 6) = 3 + 144*390 = 56163.
module node_conv_888 (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          valid_in,
    output reg           ready_in,
    input  wire [255:0]  data_in,
    output reg           valid_out,
    output reg  [255:0]  data_out
);

    localparam integer IC            = 96;
    localparam integer OC            = 576;
    localparam integer IH            = 14;
    localparam integer IW            = 14;
    localparam integer OH            = 14;
    localparam integer OW            = 14;
    localparam integer KH            = 1;
    localparam integer KW            = 1;
    localparam integer K_TOTAL       = IC * KH * KW;
    localparam integer MP            = 4;
    localparam integer OC_PASSES     = (OC + MP - 1) / MP;
    localparam integer CHANNEL_TILE  = 32;
    localparam integer IN_BEATS      = (IC + CHANNEL_TILE - 1) / CHANNEL_TILE;
    localparam integer OUT_BEATS     = (OC + CHANNEL_TILE - 1) / CHANNEL_TILE;
    localparam integer TILE_BITS     = CHANNEL_TILE * 8;
    localparam integer TOTAL_PIXELS  = OH * OW;

    localparam integer SCALE_MULT  = 32'd279;
    localparam integer SCALE_SHIFT = 5'd16;

    localparam integer PROD_W        = 16;
    localparam integer ACC_W         = PROD_W + 7;
    localparam integer BIAS_W        = 32;
    localparam integer BIASED_W      = BIAS_W + 1;
    localparam integer SCALE_MAG_W   = 14;
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
        $readmemh("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_888_weights.hex", weights);
        $readmemh("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_888_bias.hex",    biases);
    end

    reg signed [7:0] in_latch [0:IC-1];
    reg [OC*8-1:0] out_pack;

    localparam [2:0] ST_LOAD    = 3'd0;
    localparam [2:0] ST_RUNNING = 3'd1;
    localparam [2:0] ST_BIAS    = 3'd2;
    localparam [2:0] ST_SCALE   = 3'd3;
    localparam [2:0] ST_OUTPUT  = 3'd4;
    localparam [2:0] ST_EMIT    = 3'd5;
    reg [2:0] state;

    reg [3:0] in_beat_idx;
    reg [4:0] out_beat_idx;
    reg [6:0] k_counter;
    reg [2:0] lane_counter;
    reg [7:0] oc_group;
    reg [7:0] active_emit_count;
    reg [1:0] drain_count;
    reg       mac_done_issuing;

    reg signed [ACC_W-1:0]    acc    [0:MP-1];
    reg signed [BIASED_W-1:0] biased [0:MP-1];
    reg signed [SCALED_W-1:0] scaled [0:MP-1];
    reg signed [SCALED_W-1:0] v_tmp;

    reg                       mac_valid_q1;
    reg                       mac_valid_q2;
    reg [2:0]                 mac_lane_q1;
    reg [2:0]                 mac_lane_q2;
    reg signed [7:0]          weight_q;
    reg signed [7:0]          act_q;
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] mul_q;

    integer i;
    integer lane;
    integer global_oc;
    integer tile_start_channel;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state             <= ST_LOAD;
            ready_in          <= 1'b1;
            valid_out         <= 1'b0;
            data_out          <= 256'b0;
            in_beat_idx       <= 4'd0;
            out_beat_idx      <= 5'd0;
            k_counter         <= 7'd0;
            lane_counter      <= 3'd0;
            oc_group          <= 8'd0;
            active_emit_count <= 8'd0;
            drain_count       <= 2'd0;
            mac_done_issuing  <= 1'b0;
            mac_valid_q1      <= 1'b0;
            mac_valid_q2      <= 1'b0;
            mac_lane_q1       <= 3'd0;
            mac_lane_q2       <= 3'd0;
            weight_q          <= 8'sd0;
            act_q             <= 8'sd0;
            mul_q             <= {PROD_W{1'b0}};
            v_tmp             <= {SCALED_W{1'b0}};
            out_pack          <= {(OC*8){1'b0}};
            for (i = 0; i < MP; i = i + 1) begin
                acc[i]    <= {ACC_W{1'b0}};
                biased[i] <= {BIASED_W{1'b0}};
                scaled[i] <= {SCALED_W{1'b0}};
            end
            for (i = 0; i < IC; i = i + 1) begin
                in_latch[i] <= 8'sd0;
            end
        end else begin
            mac_valid_q1 <= 1'b0;
            mul_q        <= weight_q * act_q;
            mac_valid_q2 <= mac_valid_q1;
            mac_lane_q2  <= mac_lane_q1;
            if (mac_valid_q2) begin
                acc[mac_lane_q2] <= acc[mac_lane_q2] + mul_q;
            end

            case (state)
                ST_LOAD: begin
                    if (valid_in && ready_in) begin
                        tile_start_channel = in_beat_idx * CHANNEL_TILE;
                        for (i = 0; i < CHANNEL_TILE; i = i + 1) begin
                            if ((tile_start_channel + i) < IC) begin
                                in_latch[tile_start_channel + i] <= data_in[i*8 +: 8];
                            end
                        end
                        if (in_beat_idx == IN_BEATS - 1) begin
                            in_beat_idx      <= 4'd0;
                            ready_in         <= 1'b0;
                            state            <= ST_RUNNING;
                            k_counter        <= 7'd0;
                            lane_counter     <= 3'd0;
                            oc_group         <= 8'd0;
                            mac_done_issuing <= 1'b0;
                            drain_count      <= 2'd0;
                            for (i = 0; i < MP; i = i + 1) begin
                                acc[i] <= {ACC_W{1'b0}};
                            end
                        end else begin
                            in_beat_idx <= in_beat_idx + 4'd1;
                        end
                    end
                end

                ST_RUNNING: begin
                    if (!mac_done_issuing) begin
                        weight_q     <= weights[(oc_group * MP + lane_counter) * K_TOTAL + k_counter];
                        act_q        <= in_latch[k_counter];
                        mac_lane_q1  <= lane_counter;
                        mac_valid_q1 <= 1'b1;
                        if (k_counter == K_TOTAL - 1) begin
                            k_counter <= 7'd0;
                            if (lane_counter == MP - 1) begin
                                lane_counter     <= 3'd0;
                                mac_done_issuing <= 1'b1;
                                drain_count      <= 2'd0;
                            end else begin
                                lane_counter <= lane_counter + 3'd1;
                            end
                        end else begin
                            k_counter <= k_counter + 7'd1;
                        end
                    end else begin
                        if (drain_count == 2'd2) begin
                            state            <= ST_BIAS;
                            mac_done_issuing <= 1'b0;
                            drain_count      <= 2'd0;
                        end else begin
                            drain_count <= drain_count + 2'd1;
                        end
                    end
                end

                ST_BIAS: begin
                    for (lane = 0; lane < MP; lane = lane + 1) begin
                        global_oc = oc_group * MP + lane;
                        biased[lane] <= $signed(acc[lane]) + $signed(biases[global_oc]);
                    end
                    state <= ST_SCALE;
                end

                ST_SCALE: begin
                    for (lane = 0; lane < MP; lane = lane + 1) begin
                        scaled[lane] <= $signed(biased[lane]) * $signed(SCALE_MULT_CONST);
                    end
                    state <= ST_OUTPUT;
                end

                ST_OUTPUT: begin
                    for (lane = 0; lane < MP; lane = lane + 1) begin
                        global_oc = oc_group * MP + lane;
                        v_tmp = (scaled[lane] +
                                 (scaled[lane][SCALED_W-1] ? SCALE_ROUND_HALF_M1 : SCALE_ROUND_HALF)
                                ) >>> SCALE_SHIFT;
                        out_pack[global_oc*8 +: 8] <= (v_tmp > 127)  ?  8'sd127 :
                                                      (v_tmp < -128) ? -8'sd128 :
                                                                       v_tmp[7:0];
                    end
                    if (oc_group == OC_PASSES - 1) begin
                        oc_group     <= 8'd0;
                        valid_out    <= 1'b1;
                        data_out     <= out_pack[0 +: TILE_BITS];
                        out_beat_idx <= 5'd0;
                        state        <= ST_EMIT;
                    end else begin
                        oc_group         <= oc_group + 8'd1;
                        state            <= ST_RUNNING;
                        k_counter        <= 7'd0;
                        lane_counter     <= 3'd0;
                        mac_done_issuing <= 1'b0;
                        drain_count      <= 2'd0;
                        for (i = 0; i < MP; i = i + 1) begin
                            acc[i] <= {ACC_W{1'b0}};
                        end
                    end
                end

                ST_EMIT: begin
                    if (out_beat_idx == OUT_BEATS - 1) begin
                        valid_out    <= 1'b0;
                        out_beat_idx <= 5'd0;
                        ready_in     <= 1'b1;
                        state        <= ST_LOAD;
                        if (active_emit_count == TOTAL_PIXELS - 1) begin
                            active_emit_count <= 8'd0;
                        end else begin
                            active_emit_count <= active_emit_count + 8'd1;
                        end
                    end else begin
                        out_beat_idx <= out_beat_idx + 5'd1;
                        data_out     <= out_pack[(out_beat_idx + 5'd1) * TILE_BITS +: TILE_BITS];
                    end
                end

                default: state <= ST_LOAD;
            endcase
        end
    end

endmodule
