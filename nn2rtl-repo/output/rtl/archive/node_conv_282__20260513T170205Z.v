// node_conv_282 -- 1x1 pointwise conv2d, IC=1024, OC=512, 14x14, stride=1.
// Contract: tiled-streaming. Bus: 256b in / 256b out. CHANNEL_TILE=32.
// IN_BEATS=32, OUT_BEATS=16, OC_PASSES=128, MP=4.
// First valid_out latency: IN_BEATS + OC_PASSES*(MP*K_TOTAL + 6)
//                        = 32 + 128*(4*1024 + 6) = 525088 cycles.
// use-bram rewrite: in_latch reorganized as 32x256 BRAM-inferred memory
// (was 1024x8 array implemented as ~8000 FFs + 1024:1 async mux).
module node_conv_282 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    output reg          ready_in,
    input  wire [255:0] data_in,
    output reg          valid_out,
    output reg  [255:0] data_out
);
    localparam IC           = 1024;
    localparam OC           = 512;
    localparam IH           = 14;
    localparam IW           = 14;
    localparam OH           = 14;
    localparam OW           = 14;
    localparam KH           = 1;
    localparam KW           = 1;
    localparam K_TOTAL      = IC*KH*KW;
    localparam MP           = 4;
    localparam OC_PASSES    = (OC + MP - 1) / MP;
    localparam CHANNEL_TILE = 32;
    localparam IN_BEATS     = (IC + CHANNEL_TILE - 1) / CHANNEL_TILE;
    localparam OUT_BEATS    = (OC + CHANNEL_TILE - 1) / CHANNEL_TILE;
    localparam TILE_BITS    = CHANNEL_TILE * 8;

    localparam integer SCALE_MULT  = 31926;
    localparam integer SCALE_SHIFT = 21;

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

    localparam NUM_WEIGHTS    = OC * K_TOTAL;
    localparam WEIGHT_ADDR_W  = (NUM_WEIGHTS <= 1) ? 1 : $clog2(NUM_WEIGHTS);
    localparam OC_INDEX_W     = (OC + MP <= 1) ? 1 : $clog2(OC + MP);
    localparam IN_ROW_ADDR_W  = (IN_BEATS <= 1) ? 1 : $clog2(IN_BEATS);

    (* rom_style = "block", ram_style = "block" *) reg signed [7:0]  weights [0:OC*K_TOTAL-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [31:0] biases  [0:OC-1];
    initial begin
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_282_weights.hex", weights);
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_282_bias.hex",   biases);
    end

    localparam [2:0] ST_LOAD    = 3'd0,
                     ST_RUNNING = 3'd1,
                     ST_BIAS    = 3'd2,
                     ST_SCALE   = 3'd3,
                     ST_OUTPUT  = 3'd4,
                     ST_EMIT    = 3'd5;

    reg [2:0] state;
    reg [3:0] pixel_row, pixel_col;
    reg [5:0] in_beat_idx;
    reg [4:0] out_beat_idx;

    (* ram_style = "block" *) reg [TILE_BITS-1:0] in_latch_mem [0:IN_BEATS-1];
    reg [TILE_BITS-1:0] in_latch_q;

    reg [$clog2(K_TOTAL+1)-1:0]   k_counter;
    reg [$clog2(MP+1)-1:0]        lane_counter;
    reg [$clog2(OC_PASSES+1)-1:0] oc_group;

    reg signed [ACC_W-1:0]    acc    [0:MP-1];
    reg signed [BIASED_W-1:0] biased [0:MP-1];
    reg signed [SCALED_W-1:0] scaled [0:MP-1];
    reg signed [SCALED_W-1:0] v_tmp;

    reg [OC*8-1:0] out_pack;

    reg                            mac_valid_q1;
    reg [$clog2(MP+1)-1:0]         mac_lane_q1;
    reg [$clog2(K_TOTAL+1)-1:0]    mac_k_q1;
    reg [OC_INDEX_W-1:0]           mac_global_oc_q1;
    reg                            mac_done_issuing;
    reg                            mac_valid_q2;
    reg [$clog2(MP+1)-1:0]         mac_lane_q2;
    reg [OC_INDEX_W-1:0]           mac_global_oc_q2;

    reg signed [7:0]               weight_q;
    wire [OC_INDEX_W-1:0]          current_global_oc;
    wire [WEIGHT_ADDR_W-1:0]       weight_read_addr;
    wire [IN_ROW_ADDR_W-1:0]       in_row_addr;
    wire [4:0]                     in_sub_idx_q1;
    wire [7:0]                     in_byte_sel;
    assign current_global_oc = oc_group * MP + lane_counter;
    assign weight_read_addr  = current_global_oc * K_TOTAL + k_counter;
    assign in_row_addr       = k_counter[IN_ROW_ADDR_W+5-1:5];
    assign in_sub_idx_q1     = mac_k_q1[4:0];
    assign in_byte_sel       = in_latch_q[in_sub_idx_q1*8 +: 8];

    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] mul_q;

    integer lane;
    integer bias_oc, out_oc;

    always @(posedge clk) begin
        weight_q <= weights[weight_read_addr];
    end

    always @(posedge clk) begin
        if (state == ST_LOAD && valid_in && ready_in)
            in_latch_mem[in_beat_idx[IN_ROW_ADDR_W-1:0]] <= data_in;
    end

    always @(posedge clk) begin
        in_latch_q <= in_latch_mem[in_row_addr];
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= ST_LOAD;
            ready_in         <= 1'b1;
            valid_out        <= 1'b0;
            data_out         <= 256'd0;
            pixel_row        <= 0;
            pixel_col        <= 0;
            in_beat_idx      <= 0;
            out_beat_idx     <= 0;
            k_counter        <= 0;
            lane_counter     <= 0;
            oc_group         <= 0;
            mac_valid_q1     <= 1'b0;
            mac_lane_q1      <= 0;
            mac_k_q1         <= 0;
            mac_global_oc_q1 <= 0;
            mac_valid_q2     <= 1'b0;
            mac_lane_q2      <= 0;
            mac_global_oc_q2 <= 0;
            mac_done_issuing <= 1'b0;
            mul_q            <= 0;
            v_tmp            <= 0;
            out_pack         <= {(OC*8){1'b0}};
            for (lane = 0; lane < MP; lane = lane + 1) begin
                acc   [lane] <= 0;
                biased[lane] <= 0;
                scaled[lane] <= 0;
            end
        end else begin
            mul_q            <= $signed(weight_q) * $signed(in_byte_sel);
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
                if (valid_in && ready_in) begin
                    if (in_beat_idx == IN_BEATS - 1) begin
                        in_beat_idx      <= 0;
                        ready_in         <= 1'b0;
                        k_counter        <= 0;
                        lane_counter     <= 0;
                        oc_group         <= 0;
                        mac_done_issuing <= 1'b0;
                        for (lane = 0; lane < MP; lane = lane + 1)
                            acc[lane] <= 0;
                        state <= ST_RUNNING;
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
                        // [INVARIANT:ROUNDING]
                        v_tmp = (scaled[lane] +
                                 (scaled[lane][SCALED_W-1] ? SCALE_ROUND_HALF_M1 : SCALE_ROUND_HALF)
                                ) >>> SCALE_SHIFT;
                        out_pack[out_oc*8 +: 8] <= (v_tmp >  127) ?  8'sd127 :
                                                   (v_tmp < -128) ? -8'sd128 :
                                                                    v_tmp[7:0];
                    end
                end

                if (oc_group < OC_PASSES - 1) begin
                    for (lane = 0; lane < MP; lane = lane + 1) acc[lane] <= 0;
                    k_counter    <= 0;
                    lane_counter <= 0;
                    oc_group     <= oc_group + 1;
                    state        <= ST_RUNNING;
                end else begin
                    valid_out    <= 1'b1;
                    data_out     <= out_pack[0 +: TILE_BITS];
                    out_beat_idx <= 5'd1;
                    oc_group     <= 0;
                    state        <= ST_EMIT;
                end
            end

            ST_EMIT: begin
                valid_out <= 1'b1;
                data_out  <= out_pack[out_beat_idx*TILE_BITS +: TILE_BITS];
                if (out_beat_idx == OUT_BEATS - 1) begin
                    out_beat_idx <= 0;
                    ready_in     <= 1'b1;
                    if (pixel_col == OW - 1) begin
                        pixel_col <= 0;
                        if (pixel_row == OH - 1)
                            pixel_row <= 0;
                        else
                            pixel_row <= pixel_row + 1;
                    end else begin
                        pixel_col <= pixel_col + 1;
                    end
                    state <= ST_LOAD;
                end else begin
                    out_beat_idx <= out_beat_idx + 1;
                end
            end

            default: state <= ST_LOAD;
            endcase
        end
    end
endmodule
