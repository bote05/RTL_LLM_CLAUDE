// node_conv_886 - 1x1 pointwise conv2d, tiled-streaming contract.
//   IC=576, OC=96, IH=IW=14, OH=OW=14, SH=SW=1 (every pixel active), MP=4.
//   channel_tile=32 -> IN_BEATS=18 input beats, OUT_BEATS=3 output beats.
// Latency: IN_BEATS + OC_PASSES * (MP*K_TOTAL + 6)
//        = 18 + 24 * (4*576 + 6) = 18 + 24*2310 = 55458 cycles.
module node_conv_886 (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              valid_in,
    output reg               ready_in,
    input  wire [255:0]      data_in,
    output reg               valid_out,
    output reg  [255:0]      data_out
);
    // ---- Geometry ----
    localparam integer IC = 576, OC = 96;
    localparam integer OH = 14, OW = 14;
    localparam integer K_TOTAL = IC;                       // KH*KW=1, so K_TOTAL=IC
    localparam integer MP = 4;
    localparam integer OC_PASSES = 24;                     // OC/MP
    localparam integer CHANNEL_TILE = 32;
    localparam integer IN_BEATS  = 18;                     // ceil(IC/CHANNEL_TILE)
    localparam integer OUT_BEATS = 3;                      // ceil(OC/CHANNEL_TILE)
    localparam integer TILE_BITS = 256;                    // CHANNEL_TILE*8

    // ---- Quantisation (scale = 0.008422413586800596 -> 17663/2^21) ----
    localparam integer SCALE_MULT  = 17663;
    localparam integer SCALE_SHIFT = 21;

    // ---- Datapath widths ----
    localparam integer PROD_W        = 16;                 // 8x8 signed
    localparam integer ACC_W         = 26;                 // PROD_W + clog2(K_TOTAL=576)
    localparam integer BIASED_W      = 33;                 // max(ACC_W,32)+1
    localparam integer SCALE_CONST_W = 16;
    localparam integer SCALED_W      = 49;                 // BIASED_W + SCALE_CONST_W

    localparam signed [SCALE_CONST_W-1:0] SCALE_MULT_CONST = SCALE_MULT;

    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};

    // ---- Weight / bias storage (BRAM-inferred ROM) ----
    (* rom_style = "block", ram_style = "block" *)
    reg signed [7:0]  weights [0:OC*K_TOTAL-1];
    (* rom_style = "block", ram_style = "block" *)
    reg signed [31:0] biases  [0:OC-1];

    initial begin
        $readmemh("output/mobilenet-v2/weights/node_conv_886_weights.hex", weights);
        $readmemh("output/mobilenet-v2/weights/node_conv_886_bias.hex", biases);
    end

    // ---- Activation latch (one full input pixel = IC INT8 channels) ----
    reg signed [7:0] in_latch [0:IC-1];

    // ---- FSM ----
    localparam [2:0] ST_LOAD    = 3'd0,
                     ST_RUNNING = 3'd1,
                     ST_BIAS    = 3'd2,
                     ST_SCALE   = 3'd3,
                     ST_OUTPUT  = 3'd4,
                     ST_EMIT    = 3'd5;
    reg [2:0] state;

    // Pixel coords (only used for inter-vector reset detection)
    reg [3:0] pixel_row;
    reg [3:0] pixel_col;
    reg [7:0] active_emit_count;

    // Beat counters (orchestrator preflight requires a `beat` name)
    reg [4:0] in_beat_idx;
    reg [1:0] out_beat_idx;

    // MAC scheduling
    reg [4:0] oc_group;
    reg [1:0] lane_counter;
    reg [9:0] k_counter;
    reg       mac_done_issuing;
    reg [1:0] drain_count;

    // 3-stage MAC pipeline registers
    reg                          mac_valid_q1, mac_valid_q2;
    reg [1:0]                    mac_lane_q1,  mac_lane_q2;
    reg signed [7:0]             weight_q;
    reg signed [7:0]             in_q;
    (* use_dsp = "yes" *)
    reg signed [PROD_W-1:0]      mul_q;

    // Accumulators / requantize pipeline regs
    reg signed [ACC_W-1:0]       acc    [0:MP-1];
    reg signed [BIASED_W-1:0]    biased [0:MP-1];
    reg signed [SCALED_W-1:0]    scaled [0:MP-1];
    reg signed [SCALED_W-1:0]    v_tmp;

    // Full-OC output staging buffer
    reg [OC*8-1:0] out_pack;

    integer lane;
    integer i;
    integer global_oc;
    integer wr_idx;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ready_in          <= 1'b1;
            valid_out         <= 1'b0;
            data_out          <= 256'b0;
            state             <= ST_LOAD;
            in_beat_idx       <= 5'd0;
            out_beat_idx      <= 2'd0;
            pixel_row         <= 4'd0;
            pixel_col         <= 4'd0;
            active_emit_count <= 8'd0;
            oc_group          <= 5'd0;
            lane_counter      <= 2'd0;
            k_counter         <= 10'd0;
            mac_done_issuing  <= 1'b0;
            drain_count       <= 2'd0;
            mac_valid_q1      <= 1'b0;
            mac_valid_q2      <= 1'b0;
            mac_lane_q1       <= 2'd0;
            mac_lane_q2       <= 2'd0;
            weight_q          <= 8'sd0;
            in_q              <= 8'sd0;
            mul_q             <= {PROD_W{1'b0}};
            out_pack          <= {(OC*8){1'b0}};
            v_tmp             <= {SCALED_W{1'b0}};
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

            mac_valid_q2 <= mac_valid_q1;
            mac_lane_q2  <= mac_lane_q1;
            mul_q        <= weight_q * in_q;

            if (mac_valid_q2) begin
                acc[mac_lane_q2] <= acc[mac_lane_q2] + mul_q;
            end

            case (state)
                ST_LOAD: begin
                    valid_out <= 1'b0;
                    if (valid_in && ready_in) begin
                        for (i = 0; i < CHANNEL_TILE; i = i + 1) begin
                            wr_idx = in_beat_idx * CHANNEL_TILE + i;
                            if (wr_idx < IC) begin
                                in_latch[wr_idx] <= data_in[i*8 +: 8];
                            end
                        end
                        if (in_beat_idx == IN_BEATS - 1) begin
                            in_beat_idx      <= 5'd0;
                            // [INVARIANT:READY_IN_GATING]
                            ready_in         <= 1'b0;
                            state            <= ST_RUNNING;
                            oc_group         <= 5'd0;
                            lane_counter     <= 2'd0;
                            k_counter        <= 10'd0;
                            mac_done_issuing <= 1'b0;
                            drain_count      <= 2'd0;
                            for (lane = 0; lane < MP; lane = lane + 1) begin
                                acc[lane] <= {ACC_W{1'b0}};
                            end
                        end else begin
                            in_beat_idx <= in_beat_idx + 5'd1;
                        end
                    end
                end

                ST_RUNNING: begin
                    valid_out <= 1'b0;
                    if (!mac_done_issuing) begin
                        weight_q     <= weights[(oc_group * MP + lane_counter) * K_TOTAL + k_counter];
                        in_q         <= in_latch[k_counter];
                        mac_valid_q1 <= 1'b1;
                        mac_lane_q1  <= lane_counter;

                        if (k_counter == K_TOTAL - 1) begin
                            k_counter <= 10'd0;
                            if (lane_counter == MP - 1) begin
                                lane_counter     <= 2'd0;
                                mac_done_issuing <= 1'b1;
                                drain_count      <= 2'd0;
                            end else begin
                                lane_counter <= lane_counter + 2'd1;
                            end
                        end else begin
                            k_counter <= k_counter + 10'd1;
                        end
                    end else begin
                        if (drain_count == 2'd2) begin
                            state <= ST_BIAS;
                        end
                        drain_count <= drain_count + 2'd1;
                    end
                end

                ST_BIAS: begin
                    valid_out <= 1'b0;
                    for (lane = 0; lane < MP; lane = lane + 1) begin
                        biased[lane] <= acc[lane] + biases[oc_group * MP + lane];
                    end
                    state <= ST_SCALE;
                end

                ST_SCALE: begin
                    valid_out <= 1'b0;
                    for (lane = 0; lane < MP; lane = lane + 1) begin
                        scaled[lane] <= biased[lane] * SCALE_MULT_CONST;
                    end
                    state <= ST_OUTPUT;
                end

                ST_OUTPUT: begin
                    for (lane = 0; lane < MP; lane = lane + 1) begin
                        global_oc = oc_group * MP + lane;
                        // [INVARIANT:ROUNDING]
                        // Golden requant = floor(value*MULT/2^SHIFT + 0.5):
                        // round-half-toward-+inf via UNCONDITIONAL +ROUND_BIAS then arith >>>.
                        v_tmp = (scaled[lane] + SCALE_ROUND_HALF) >>> SCALE_SHIFT;
                        out_pack[global_oc*8 +: 8] <= (v_tmp > 127)  ?  8'sd127 :
                                                       (v_tmp < -128) ? -8'sd128 :
                                                                          v_tmp[7:0];
                    end
                    if (oc_group == OC_PASSES - 1) begin
                        // [INVARIANT:VALID_OUT_LATENCY]
                        valid_out    <= 1'b1;
                        data_out     <= out_pack[0 +: TILE_BITS];
                        out_beat_idx <= 2'd1;
                        state        <= ST_EMIT;
                    end else begin
                        valid_out        <= 1'b0;
                        oc_group         <= oc_group + 5'd1;
                        lane_counter     <= 2'd0;
                        k_counter        <= 10'd0;
                        mac_done_issuing <= 1'b0;
                        drain_count      <= 2'd0;
                        for (lane = 0; lane < MP; lane = lane + 1) begin
                            acc[lane] <= {ACC_W{1'b0}};
                        end
                        state <= ST_RUNNING;
                    end
                end

                ST_EMIT: begin
                    // [INVARIANT:VALID_OUT_LATENCY]
                    valid_out <= 1'b1;
                    data_out  <= out_pack[out_beat_idx * TILE_BITS +: TILE_BITS];
                    if (out_beat_idx == OUT_BEATS - 1) begin
                        out_beat_idx <= 2'd0;
                        if (active_emit_count == (OH*OW) - 1) begin
                            active_emit_count <= 8'd0;
                            pixel_row         <= 4'd0;
                            pixel_col         <= 4'd0;
                        end else begin
                            active_emit_count <= active_emit_count + 8'd1;
                            if (pixel_col == OW - 1) begin
                                pixel_col <= 4'd0;
                                pixel_row <= pixel_row + 4'd1;
                            end else begin
                                pixel_col <= pixel_col + 4'd1;
                            end
                        end
                        state    <= ST_LOAD;
                        ready_in <= 1'b1;
                    end else begin
                        out_beat_idx <= out_beat_idx + 2'd1;
                    end
                end

                default: state <= ST_LOAD;
            endcase
        end
    end
endmodule
