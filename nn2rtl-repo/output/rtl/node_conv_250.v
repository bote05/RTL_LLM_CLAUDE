// Pointwise (1x1) conv2d, stride=2, tiled-streaming contract.
//   IC=512, OC=1024, IH=IW=28, OH=OW=14, SH=SW=2, MP=4.
//   channel_tile=32 -> IN_BEATS=16, OUT_BEATS=32 per pixel.
// Latency: 16 (input collection) + OC_PASSES * (MP*K_TOTAL + 6) = 525840.
//   First valid_out fires the cycle after the last OC pass's ST_OUTPUT;
//   beats 1..31 stream out in subsequent cycles from the OC staging buffer.
// Stride: accept every input pixel (TB drives all IH*IW pixels), MAC only
//   when (pixel_row[0]==0 && pixel_col[0]==0). Inactive pixels are consumed
//   and discarded.
// Inter-vector reset: TB stops driving once the OH*OW-th active output
//   pixel emits its last beat. Force pixel_row/pixel_col/active_emit_count
//   back to 0 in the cycle that emits the final output beat so the next
//   vector starts aligned at (0,0).
// Adapted from knowledge/references/protected/conv1x1_passing_reference.v
// and output/rtl/node_conv_224.v (sibling 1x1 stride-2 flat-bus module).

module node_conv_250 (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              valid_in,
    output reg               ready_in,
    input  wire [255:0]      data_in,
    output reg               valid_out,
    output reg  [255:0]      data_out
);
    localparam IC           = 512;
    localparam OC           = 1024;
    localparam IH           = 28;
    localparam IW           = 28;
    localparam OH           = 14;
    localparam OW           = 14;
    localparam KH           = 1;
    localparam KW           = 1;
    localparam SH           = 2;
    localparam SW           = 2;
    localparam PH           = 0;
    localparam PW           = 0;
    localparam K_TOTAL      = IC * KH * KW;
    localparam MP           = 4;
    localparam OC_PASSES    = (OC + MP - 1) / MP;
    localparam CHANNEL_TILE = 32;
    localparam IN_BEATS     = (IC + CHANNEL_TILE - 1) / CHANNEL_TILE;
    localparam OUT_BEATS    = (OC + CHANNEL_TILE - 1) / CHANNEL_TILE;
    localparam TILE_BITS    = CHANNEL_TILE * 8;

    localparam NUM_WEIGHTS    = OC * K_TOTAL;
    localparam WEIGHT_ADDR_W  = (NUM_WEIGHTS <= 1) ? 1 : $clog2(NUM_WEIGHTS);
    localparam OC_INDEX_W     = (OC + MP <= 1) ? 1 : $clog2(OC + MP);
    localparam ROW_W          = (IH <= 1) ? 1 : $clog2(IH);
    localparam COL_W          = (IW <= 1) ? 1 : $clog2(IW);
    localparam APIX_W         = $clog2(OH*OW + 1);
    localparam IN_BEAT_W      = $clog2(IN_BEATS + 1);
    localparam OUT_BEAT_W     = $clog2(OUT_BEATS + 1);

    localparam SCALE_MULT  = 6306;
    localparam SCALE_SHIFT = 19;

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
    localparam signed [SCALED_W-1:0] SCALE_ROUND_ONE =
        {{(SCALED_W-1){1'b0}}, 1'b1};
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - SCALE_ROUND_ONE;

    localparam ST_LOAD     = 3'd0;
    localparam ST_RUNNING  = 3'd1;
    localparam ST_BIAS     = 3'd2;
    localparam ST_SCALE    = 3'd3;
    localparam ST_OUTPUT   = 3'd4;
    localparam ST_EMIT     = 3'd5;

    (* rom_style = "block", ram_style = "block" *) reg signed [7:0]  weights [0:OC*K_TOTAL-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [31:0] biases  [0:OC-1];
    initial begin
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_250_weights_wide.hex", weights);
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_250_bias.hex", biases);
    end

    reg signed [7:0]   in_latch [0:IC-1];
    reg [OC*8-1:0]     out_pack;

    reg signed [ACC_W-1:0]    acc    [0:MP-1];
    reg signed [BIASED_W-1:0] biased [0:MP-1];
    reg signed [SCALED_W-1:0] scaled [0:MP-1];
    reg signed [SCALED_W-1:0] v_tmp;
    reg signed [7:0]          sat_tmp;

    reg [$clog2(K_TOTAL+1)-1:0]   k_counter;
    reg [$clog2(MP+1)-1:0]        lane_counter;
    reg [$clog2(OC_PASSES+1)-1:0] oc_group;
    reg [2:0]                     state;

    reg [IN_BEAT_W-1:0]   in_beat_idx;
    reg [OUT_BEAT_W-1:0]  out_beat_idx;
    reg [OUT_BEAT_W-1:0]  next_out_beat;
    reg [ROW_W-1:0]       pixel_row;
    reg [COL_W-1:0]       pixel_col;
    reg [APIX_W-1:0]      active_emit_count;

    reg  signed [7:0]              weight_q;
    wire [OC_INDEX_W-1:0]          current_global_oc;
    wire [WEIGHT_ADDR_W-1:0]       weight_read_addr;
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
            state             <= ST_LOAD;
            ready_in          <= 1'b1;  // [INVARIANT:READY_IN_GATING]
            valid_out         <= 1'b0;
            data_out          <= 256'd0;
            out_pack          <= {(OC*8){1'b0}};
            in_beat_idx       <= 0;
            out_beat_idx      <= 0;
            next_out_beat     <= 0;
            pixel_row         <= 0;
            pixel_col         <= 0;
            active_emit_count <= 0;
            k_counter         <= 0;
            lane_counter      <= 0;
            oc_group          <= 0;
            mac_valid_q1      <= 1'b0;
            mac_lane_q1       <= 0;
            mac_k_q1          <= 0;
            mac_global_oc_q1  <= 0;
            mac_valid_q2      <= 1'b0;
            mac_lane_q2       <= 0;
            mac_global_oc_q2  <= 0;
            mac_done_issuing  <= 1'b0;
            mul_q             <= 0;
            v_tmp             <= 0;
            sat_tmp           <= 0;
            for (i = 0; i < IC; i = i + 1)
                in_latch[i] <= 8'sd0;
            for (lane = 0; lane < MP; lane = lane + 1) begin
                acc   [lane] <= 0;
                biased[lane] <= 0;
                scaled[lane] <= 0;
            end
        end else begin
            mul_q            <= $signed(weight_q) * $signed(in_latch[mac_k_q1]);
            mac_valid_q2     <= mac_valid_q1;
            mac_lane_q2      <= mac_lane_q1;
            mac_global_oc_q2 <= mac_global_oc_q1;

            if (mac_valid_q2 && mac_global_oc_q2 < OC) begin
                acc[mac_lane_q2] <= acc[mac_lane_q2] + $signed(mul_q);
            end

            case (state)

            ST_LOAD: begin
                valid_out    <= 1'b0;
                mac_valid_q1 <= 1'b0;
                mac_valid_q2 <= 1'b0;
                if (valid_in) begin
                    for (i = 0; i < CHANNEL_TILE; i = i + 1)
                        in_latch[in_beat_idx * CHANNEL_TILE + i] <= $signed(data_in[i*8 +: 8]);

                    if (in_beat_idx == IN_BEATS - 1) begin
                        in_beat_idx <= 0;
                        if ((pixel_row[0] == 1'b0) && (pixel_col[0] == 1'b0)) begin
                            ready_in         <= 1'b0;  // [INVARIANT:READY_IN_GATING]
                            k_counter        <= 0;
                            lane_counter     <= 0;
                            oc_group         <= 0;
                            mac_done_issuing <= 1'b0;
                            for (lane = 0; lane < MP; lane = lane + 1)
                                acc[lane] <= 0;
                            state <= ST_RUNNING;
                        end else begin
                            if (pixel_col == IW - 1) begin
                                pixel_col <= 0;
                                if (pixel_row == IH - 1) pixel_row <= 0;
                                else                     pixel_row <= pixel_row + 1;
                            end else begin
                                pixel_col <= pixel_col + 1;
                            end
                        end
                    end else begin
                        in_beat_idx <= in_beat_idx + 1;
                    end
                end
            end

            ST_RUNNING: begin
                if (mac_done_issuing) begin
                    mac_valid_q1 <= 1'b0;
                    if (!mac_valid_q1 && !mac_valid_q2) begin
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
                    if (bias_oc < OC)
                        biased[lane] <= $signed(acc[lane]) + $signed(biases[bias_oc]);
                    else
                        biased[lane] <= 0;
                end
                state <= ST_SCALE;
            end

            ST_SCALE: begin
                for (lane = 0; lane < MP; lane = lane + 1)
                    scaled[lane] <= $signed(biased[lane]) * $signed(SCALE_MULT_CONST);
                state <= ST_OUTPUT;
            end

            ST_OUTPUT: begin
                for (lane = 0; lane < MP; lane = lane + 1) begin
                    out_oc = oc_group * MP + lane;
                    if (out_oc < OC) begin
                        v_tmp = (scaled[lane] +
                                 (scaled[lane][SCALED_W-1] ? SCALE_ROUND_HALF_M1
                                                           : SCALE_ROUND_HALF)
                                ) >>> SCALE_SHIFT;  // [INVARIANT:ROUNDING]
                        sat_tmp = (v_tmp >  127) ?  8'sd127 :
                                  (v_tmp < -128) ? -8'sd128 :
                                                   v_tmp[7:0];
                        out_pack[out_oc*8 +: 8] <= sat_tmp;
                    end
                end

                if (oc_group < OC_PASSES - 1) begin
                    for (lane = 0; lane < MP; lane = lane + 1) acc[lane] <= 0;
                    k_counter    <= 0;
                    lane_counter <= 0;
                    oc_group     <= oc_group + 1;
                    state        <= ST_RUNNING;
                end else begin
                    // Last OC pass: emit output beat 0 next cycle. The
                    // out_pack[0 +: TILE_BITS] read picks up the values
                    // written during oc_groups 0..(OC_PER_BEAT/MP - 1);
                    // this cycle's lane writes target out_pack[(OC-MP)*8
                    // +: MP*8] (last beat tail), which does not overlap.
                    valid_out     <= 1'b1;  // [INVARIANT:VALID_OUT_LATENCY]
                    data_out      <= out_pack[0 +: TILE_BITS];
                    out_beat_idx  <= 0;
                    oc_group      <= 0;
                    state         <= ST_EMIT;
                end
            end

            ST_EMIT: begin
                if (out_beat_idx == OUT_BEATS - 1) begin
                    valid_out    <= 1'b0;
                    ready_in     <= 1'b1;  // [INVARIANT:READY_IN_GATING]
                    out_beat_idx <= 0;
                    if (active_emit_count == OH*OW - 1) begin
                        // Final output of this vector. Force coords back
                        // to (0,0) so the next vector starts cleanly even
                        // if the TB cuts inputs at this same cycle.
                        active_emit_count <= 0;
                        pixel_row         <= 0;
                        pixel_col         <= 0;
                    end else begin
                        active_emit_count <= active_emit_count + 1;
                        if (pixel_col == IW - 1) begin
                            pixel_col <= 0;
                            if (pixel_row == IH - 1) pixel_row <= 0;
                            else                     pixel_row <= pixel_row + 1;
                        end else begin
                            pixel_col <= pixel_col + 1;
                        end
                    end
                    state <= ST_LOAD;
                end else begin
                    next_out_beat = out_beat_idx + 1;
                    data_out      <= out_pack[next_out_beat * TILE_BITS +: TILE_BITS];
                    valid_out     <= 1'b1;
                    out_beat_idx  <= next_out_beat;
                end
            end

            default: state <= ST_LOAD;
            endcase
        end
    end

endmodule
