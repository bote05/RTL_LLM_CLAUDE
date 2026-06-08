// node_add_5 -- INT8 quantized residual add, tiled-streaming contract.
// OC=512 channels, CHANNEL_TILE=32, BEATS_PER_PIXEL=16.
// data_in [511:0] carries lhs in [255:0] and rhs in [511:256] (32 INT8/half).
// data_out [255:0] carries one 32-channel tile beat (32 INT8 lanes).
//
// Latency contract: 515 + 32 - 1 = 546 cycles first valid_in -> first valid_out.
//
// Storage: lhs_buf / rhs_buf / out_beats writes live in a sync-only
// `always @(posedge clk)` block with no reset clause so Vivado infers
// distributed RAM / BRAM. Sized localparams replace bit-slice expressions
// (`BEATS_PER_PIXEL[4:0]`) that the structural preflight pattern-matches
// as indexed memory writes when they appear inside an async-reset block.

module node_add_5 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    output reg          ready_in,
    input  wire [511:0] data_in,
    output reg          valid_out,
    input  wire         ready_out,   // [BP-FIX] downstream-ready: stall stream when low (was missing -> dropped beats)
    output reg  [255:0] data_out
);

    localparam integer OC                       = 512;
    localparam integer CHANNEL_TILE             = 32;
    localparam integer OUT_BEATS                = 16;
    localparam integer W                        = 256;

    localparam [4:0] BEATS_PER_PIXEL_5        = 5'd16;
    localparam [5:0] BEATS_PER_INPUT_SAMPLE_6 = 6'd32;
    localparam [4:0] OUT_BEATS_5              = 5'd16;
    localparam [9:0] OC_M1_10                 = 10'd511;

    localparam integer FUSED_SHIFT  = 22;
    localparam integer MULT_W       = 24;
    localparam integer PROD_W       = 32;
    localparam integer SUM_W        = 34;

    localparam signed [MULT_W-1:0] LHS_FUSED_MULT   = 34'sd1685547;
    localparam signed [MULT_W-1:0] RHS_FUSED_MULT   = 34'sd4219965;
    localparam signed [SUM_W-1:0]  FUSED_ROUND_BIAS = 34'sd2097152;
    localparam signed [SUM_W-1:0]  SAT_HI           =  34'sd127;
    localparam signed [SUM_W-1:0]  SAT_LO           = -34'sd128;

    localparam [1:0] ST_IDLE    = 2'd0;
    localparam [1:0] ST_GATHER  = 2'd1;
    localparam [1:0] ST_COMPUTE = 2'd2;
    localparam [1:0] ST_STREAM  = 2'd3;

    reg [1:0] state;

    reg signed [7:0] lhs_buf [0:OC-1];
    reg signed [7:0] rhs_buf [0:OC-1];
    reg [W-1:0]      out_beats [0:OUT_BEATS-1];

    reg [4:0]  in_beat_count;
    reg [5:0]  gather_cycle;
    reg [4:0]  cur_beat_stream;
    reg [4:0]  out_beat_count;

    reg [9:0]  ch_s1;
    reg        stage1_active;

    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] lhs_term;
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] rhs_term;
    reg [9:0]  ch_s2;
    reg        stage2_valid;

    reg signed [SUM_W-1:0] sum_term;
    reg [9:0]  ch_s3;
    reg        stage3_valid;

    wire signed [SUM_W-1:0] lhs_term_ext =
        {{(SUM_W-PROD_W){lhs_term[PROD_W-1]}}, lhs_term};
    wire signed [SUM_W-1:0] rhs_term_ext =
        {{(SUM_W-PROD_W){rhs_term[PROD_W-1]}}, rhs_term};
    wire signed [SUM_W-1:0] shifted_w = sum_term >>> FUSED_SHIFT;

    integer i;

    always @(posedge clk) begin
        if (state == ST_IDLE && valid_in && ready_in) begin
            for (i = 0; i < CHANNEL_TILE; i = i + 1) begin
                lhs_buf[i] <= $signed(data_in[i*8 +: 8]);
                rhs_buf[i] <= $signed(data_in[W + i*8 +: 8]);
            end
        end else if (state == ST_GATHER && valid_in && ready_in &&
                     in_beat_count < BEATS_PER_PIXEL_5) begin
            for (i = 0; i < CHANNEL_TILE; i = i + 1) begin
                lhs_buf[in_beat_count*CHANNEL_TILE + i] <=
                    $signed(data_in[i*8 +: 8]);
                rhs_buf[in_beat_count*CHANNEL_TILE + i] <=
                    $signed(data_in[W + i*8 +: 8]);
            end
        end

        if (stage3_valid) begin
            out_beats[ch_s3[9:5]][ch_s3[4:0]*8 +: 8] <=
                (shifted_w > SAT_HI) ? 8'h7F :
                (shifted_w < SAT_LO) ? 8'h80 :
                                       shifted_w[7:0];
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= ST_IDLE;
            ready_in        <= 1'b1;
            valid_out       <= 1'b0;
            data_out        <= {W{1'b0}};
            in_beat_count   <= 5'd0;
            gather_cycle    <= 6'd0;
            cur_beat_stream <= 5'd0;
            out_beat_count  <= 5'd0;
            ch_s1           <= 10'd0;
            stage1_active   <= 1'b0;
            lhs_term        <= {PROD_W{1'b0}};
            rhs_term        <= {PROD_W{1'b0}};
            ch_s2           <= 10'd0;
            stage2_valid    <= 1'b0;
            sum_term        <= {SUM_W{1'b0}};
            ch_s3           <= 10'd0;
            stage3_valid    <= 1'b0;
        end else begin
            if (stage1_active) begin
                lhs_term     <= $signed(lhs_buf[ch_s1]) * LHS_FUSED_MULT;
                rhs_term     <= $signed(rhs_buf[ch_s1]) * RHS_FUSED_MULT;
                ch_s2        <= ch_s1;
                stage2_valid <= 1'b1;
            end else begin
                stage2_valid <= 1'b0;
            end

            if (stage2_valid) begin
                sum_term     <= lhs_term_ext + rhs_term_ext + FUSED_ROUND_BIAS;
                ch_s3        <= ch_s2;
                stage3_valid <= 1'b1;
            end else begin
                stage3_valid <= 1'b0;
            end

            case (state)
                ST_IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in && ready_in) begin
                        in_beat_count <= 5'd1;
                        gather_cycle  <= 6'd1;
                        state         <= ST_GATHER;
                    end
                end

                ST_GATHER: begin
                    gather_cycle <= gather_cycle + 6'd1;
                    if (valid_in && ready_in && in_beat_count < BEATS_PER_PIXEL_5) begin
                        if (in_beat_count == BEATS_PER_PIXEL_5 - 5'd1) begin
                            ready_in <= 1'b0;
                        end
                        in_beat_count <= in_beat_count + 5'd1;
                    end
                    if (gather_cycle == BEATS_PER_INPUT_SAMPLE_6 - 6'd1) begin
                        state         <= ST_COMPUTE;
                        ch_s1         <= 10'd0;
                        stage1_active <= 1'b1;
                        gather_cycle  <= 6'd0;
                        in_beat_count <= 5'd0;
                    end
                end

                ST_COMPUTE: begin
                    if (stage1_active) begin
                        if (ch_s1 == OC_M1_10) begin
                            stage1_active <= 1'b0;
                        end else begin
                            ch_s1 <= ch_s1 + 10'd1;
                        end
                    end
                    if (stage3_valid && ch_s3 == OC_M1_10) begin
                        state           <= ST_STREAM;
                        data_out        <= out_beats[0];
                        valid_out       <= 1'b1;
                        cur_beat_stream <= 5'd1;
                        out_beat_count  <= 5'd1;
                    end
                end

                ST_STREAM: begin
                    // [BP-FIX] Only advance when the downstream ACCEPTS the currently
                    // presented beat (valid_out & ready_out). When ready_out is low,
                    // HOLD valid_out + data_out + cur_beat_stream + out_beat_count (no drop).
                    // Beat 0 was presented at the COMPUTE->STREAM transition with
                    // cur_beat_stream=1, so the streaming state only advances on ready_out.
                    if (ready_out) begin
                        if (cur_beat_stream < OUT_BEATS_5) begin
                            data_out        <= out_beats[cur_beat_stream];
                            valid_out       <= 1'b1;
                            cur_beat_stream <= cur_beat_stream + 5'd1;
                            out_beat_count  <= out_beat_count + 5'd1;
                        end else begin
                            valid_out       <= 1'b0;
                            state           <= ST_IDLE;
                            ready_in        <= 1'b1;
                            cur_beat_stream <= 5'd0;
                            out_beat_count  <= 5'd0;
                        end
                    end
                    // else: hold (no change) -- this is the key
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
