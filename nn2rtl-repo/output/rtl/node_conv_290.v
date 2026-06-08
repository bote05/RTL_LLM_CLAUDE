// node_conv_290 -- pointwise (1x1) conv2d, stride=1, tiled-streaming.
//   IC=2048, OC=512, IH=IW=OH=OW=7, MP=4, channel_tile=32.
//   IN_BEATS=64, OUT_BEATS=16, OC_PASSES=128, K_TOTAL=2048.
//
// Improvement-mode variant (attempt 2):
//   - Attempt 1 isolated the weight read into a dedicated always block but
//     Vivado still chose LUTRAM for the 1M-deep weights array (likely
//     because cascading ~500 BRAM18s for a single ROM exceeds the BRAM
//     cost heuristic). The weight memory is retained with the same
//     attributes as the original — synthesis may still infer LUTRAM there
//     and that is acceptable.
//   - The lever flipped here is the bias memory. The original reads
//     biases[oc_group*MP + i] combinationally inside ST_BIAS, which is the
//     LUTRAM pattern. A dedicated always block now performs a synchronous
//     read every cycle into bias_q[0..MP-1] (canonical Vivado BRAM read
//     template). ram_style="block" forces BRAM18 inference; with 4 read
//     ports Vivado replicates the small 512x32 array across >=1 BRAM18.
//   - oc_group is stable for >=MP*K_TOTAL cycles per OC pass, so bias_q is
//     always loaded with biases[oc_group*MP+i] by the time ST_BIAS uses it.
//     Bit-exact equivalent to the original arithmetic.

module node_conv_290 (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              valid_in,
    output reg               ready_in,
    input  wire [255:0]      data_in,
    output reg               valid_out,
    output reg  [255:0]      data_out
);
    // ---- Geometry ----
    localparam integer IC           = 2048;
    localparam integer OC           = 512;
    localparam integer IH           = 7;
    localparam integer IW           = 7;
    localparam integer OH           = 7;
    localparam integer OW           = 7;
    localparam integer KH           = 1;
    localparam integer KW           = 1;
    localparam integer K_TOTAL      = IC * KH * KW;
    localparam integer MP           = 4;
    localparam integer OC_PASSES    = (OC + MP - 1) / MP;
    localparam integer CHANNEL_TILE = 32;
    localparam integer IN_BEATS     = (IC + CHANNEL_TILE - 1) / CHANNEL_TILE;
    localparam integer OUT_BEATS    = (OC + CHANNEL_TILE - 1) / CHANNEL_TILE;
    localparam integer TILE_BITS    = CHANNEL_TILE * 8;

    // ---- Quantisation ----
    localparam integer SCALE_MULT  = 23455;
    localparam integer SCALE_SHIFT = 21;

    // ---- Datapath widths ----
    localparam integer PROD_W        = 16;
    localparam integer ACC_W         = 28;
    localparam integer BIAS_W        = 32;
    localparam integer BIASED_W      = ((ACC_W > BIAS_W) ? ACC_W : BIAS_W) + 1;
    localparam integer SCALE_MAG_W   = 12;
    localparam integer SCALE_CONST_W = SCALE_MAG_W + 1;
    localparam integer SCALED_W      = BIASED_W + SCALE_CONST_W;

    localparam signed [SCALE_CONST_W-1:0] SCALE_MULT_CONST = SCALE_MULT;
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};

    // ---- FSM states ----
    localparam [2:0] ST_LOAD    = 3'd0;
    localparam [2:0] ST_RUNNING = 3'd1;
    localparam [2:0] ST_BIAS    = 3'd2;
    localparam [2:0] ST_SCALE   = 3'd3;
    localparam [2:0] ST_OUTPUT  = 3'd4;
    localparam [2:0] ST_EMIT    = 3'd5;

    // ---- Memories ----
    (* rom_style = "block", ram_style = "block" *) reg signed [7:0]  weights [0:OC*K_TOTAL-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [31:0] biases  [0:OC-1];

    initial begin
        $readmemh("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_290_weights_wide.hex", weights);
        $readmemh("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_290_bias.hex", biases);
    end

    // ---- State and counters ----
    reg [2:0]            state;
    reg [IC*8-1:0]       in_latch;
    reg [7:0]            in_beat_idx;
    reg [3:0]            pixel_row;
    reg [3:0]            pixel_col;
    reg [7:0]            active_emit_count;
    reg [7:0]            oc_group;
    reg [2:0]            lane_counter;
    reg [11:0]           k_counter;
    reg [1:0]            drain_count;
    reg [4:0]            out_beat_idx;
    reg                  mac_done_issuing;

    // ---- MAC pipeline registers (3-stage: issue / mul / acc) ----
    reg signed [7:0]                        weight_q;
    reg signed [7:0]                        act_q1;
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] mul_q;
    reg [1:0]                               mac_lane_q1;
    reg [1:0]                               mac_lane_q2;
    reg                                     mac_valid_q1;
    reg                                     mac_valid_q2;

    reg signed [ACC_W-1:0]    acc    [0:MP-1];
    reg signed [BIASED_W-1:0] biased [0:MP-1];
    reg signed [SCALED_W-1:0] scaled [0:MP-1];

    // ---- Registered bias read pipeline (canonical BRAM read template). ----
    // Each cycle, register the 4 bias values for the current oc_group into
    // bias_q[0..MP-1]. oc_group is stable for many thousands of cycles per
    // OC pass (MP*K_TOTAL + drain cycles in ST_RUNNING), so bias_q is fully
    // refreshed long before ST_BIAS consumes it. With ram_style="block"
    // forcing BRAM, Vivado replicates the 512x32 array to satisfy the four
    // read ports (>=1 BRAM18 inferred per replica).
    reg signed [31:0] bias_q [0:MP-1];

    // ---- Output staging buffer (full OC vector) ----
    reg [OC*8-1:0] out_pack;

    // ---- Module-scope temporaries ----
    integer                   i;
    integer                   bj;
    integer                   global_oc_int;
    reg signed [SCALED_W-1:0] v_tmp;

    // ---- Dedicated synchronous BRAM read block for biases. ----
    // No reset, no enable, no surrounding conditional — textbook Vivado BRAM
    // template. bias_q[j] is always one clock behind biases[oc_group*MP+j];
    // since oc_group is stable across MAC/drain/BIAS the registered value
    // matches the original combinational read exactly when ST_BIAS samples
    // it.
    always @(posedge clk) begin
        for (bj = 0; bj < MP; bj = bj + 1)
            bias_q[bj] <= biases[oc_group*MP + bj];
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state             <= ST_LOAD;
            ready_in          <= 1'b1;
            valid_out         <= 1'b0;
            data_out          <= 256'd0;
            in_latch          <= {(IC*8){1'b0}};
            in_beat_idx       <= 8'd0;
            pixel_row         <= 4'd0;
            pixel_col         <= 4'd0;
            active_emit_count <= 8'd0;
            oc_group          <= 8'd0;
            lane_counter      <= 3'd0;
            k_counter         <= 12'd0;
            drain_count       <= 2'd0;
            out_beat_idx      <= 5'd0;
            mac_done_issuing  <= 1'b0;
            weight_q          <= 8'sd0;
            act_q1            <= 8'sd0;
            mul_q             <= {PROD_W{1'b0}};
            mac_lane_q1       <= 2'd0;
            mac_lane_q2       <= 2'd0;
            mac_valid_q1      <= 1'b0;
            mac_valid_q2      <= 1'b0;
            out_pack          <= {(OC*8){1'b0}};
            for (i = 0; i < MP; i = i + 1) begin
                acc[i]    <= {ACC_W{1'b0}};
                biased[i] <= {BIASED_W{1'b0}};
                scaled[i] <= {SCALED_W{1'b0}};
            end
        end else begin
            // ---- Default pipeline propagation ----
            mac_valid_q1 <= 1'b0;
            mac_valid_q2 <= mac_valid_q1;
            mac_lane_q2  <= mac_lane_q1;
            if (mac_valid_q1) begin
                mul_q <= $signed(weight_q) * $signed(act_q1);
            end
            if (mac_valid_q2) begin
                acc[mac_lane_q2] <= acc[mac_lane_q2] + mul_q;
            end

            case (state)
                // -------------------------------------------------------------
                ST_LOAD: begin
                    valid_out <= 1'b0;
                    if (valid_in && ready_in) begin
                        in_latch[in_beat_idx*TILE_BITS +: TILE_BITS] <= data_in;
                        if (in_beat_idx == IN_BEATS - 1) begin
                            in_beat_idx      <= 8'd0;
                            ready_in         <= 1'b0;  // [INVARIANT:READY_IN_GATING]
                            oc_group         <= 8'd0;
                            lane_counter     <= 3'd0;
                            k_counter        <= 12'd0;
                            drain_count      <= 2'd0;
                            mac_done_issuing <= 1'b0;
                            for (i = 0; i < MP; i = i + 1) acc[i] <= {ACC_W{1'b0}};
                            state            <= ST_RUNNING;
                        end else begin
                            in_beat_idx <= in_beat_idx + 8'd1;
                        end
                    end
                end

                // -------------------------------------------------------------
                ST_RUNNING: begin
                    if (!mac_done_issuing) begin
                        weight_q     <= weights[(oc_group*MP + lane_counter)*K_TOTAL + k_counter];
                        act_q1       <= $signed(in_latch[k_counter*8 +: 8]);
                        mac_valid_q1 <= 1'b1;
                        mac_lane_q1  <= lane_counter[1:0];

                        if (k_counter == K_TOTAL - 1) begin
                            k_counter <= 12'd0;
                            if (lane_counter == MP - 1) begin
                                mac_done_issuing <= 1'b1;
                                lane_counter     <= 3'd0;
                            end else begin
                                lane_counter <= lane_counter + 3'd1;
                            end
                        end else begin
                            k_counter <= k_counter + 12'd1;
                        end
                    end else begin
                        drain_count <= drain_count + 2'd1;
                        if (drain_count == 2'd2) begin
                            state            <= ST_BIAS;
                            drain_count      <= 2'd0;
                            mac_done_issuing <= 1'b0;
                        end
                    end
                end

                // -------------------------------------------------------------
                ST_BIAS: begin
                    // Use the synchronously-registered bias values. bias_q[i]
                    // was loaded last cycle from biases[oc_group*MP+i]; since
                    // oc_group has been stable throughout ST_RUNNING/drain,
                    // this is bit-identical to the original
                    // acc[i] + biases[oc_group*MP+i].
                    for (i = 0; i < MP; i = i + 1) begin
                        biased[i] <= acc[i] + bias_q[i];
                    end
                    state <= ST_SCALE;
                end

                // -------------------------------------------------------------
                ST_SCALE: begin
                    for (i = 0; i < MP; i = i + 1) begin
                        scaled[i] <= biased[i] * SCALE_MULT_CONST;
                    end
                    state <= ST_OUTPUT;
                end

                // -------------------------------------------------------------
                ST_OUTPUT: begin
                    for (i = 0; i < MP; i = i + 1) begin
                        global_oc_int = oc_group*MP + i;
                        v_tmp = (scaled[i] +
                                 (scaled[i][SCALED_W-1] ? SCALE_ROUND_HALF_M1
                                                        : SCALE_ROUND_HALF)
                                ) >>> SCALE_SHIFT; // [INVARIANT:ROUNDING]
                        out_pack[global_oc_int*8 +: 8] <=
                            (v_tmp >  127) ?  8'sd127 :
                            (v_tmp < -128) ? -8'sd128 : v_tmp[7:0];
                    end

                    if (oc_group == OC_PASSES - 1) begin
                        valid_out    <= 1'b1; // [INVARIANT:VALID_OUT_LATENCY]
                        data_out     <= out_pack[0 +: TILE_BITS];
                        out_beat_idx <= 5'd1;
                        oc_group     <= 8'd0;
                        state        <= ST_EMIT;
                    end else begin
                        oc_group         <= oc_group + 8'd1;
                        lane_counter     <= 3'd0;
                        k_counter        <= 12'd0;
                        drain_count      <= 2'd0;
                        mac_done_issuing <= 1'b0;
                        for (i = 0; i < MP; i = i + 1) acc[i] <= {ACC_W{1'b0}};
                        state <= ST_RUNNING;
                    end
                end

                // -------------------------------------------------------------
                ST_EMIT: begin
                    valid_out <= 1'b1;
                    data_out  <= out_pack[out_beat_idx*TILE_BITS +: TILE_BITS];
                    if (out_beat_idx == OUT_BEATS - 1) begin
                        out_beat_idx <= 5'd0;
                        if (active_emit_count == OH*OW - 1) begin
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
                        ready_in <= 1'b1;  // [INVARIANT:READY_IN_GATING]
                        state    <= ST_LOAD;
                    end else begin
                        out_beat_idx <= out_beat_idx + 5'd1;
                    end
                end

                // -------------------------------------------------------------
                default: state <= ST_LOAD;
            endcase
        end
    end
endmodule
