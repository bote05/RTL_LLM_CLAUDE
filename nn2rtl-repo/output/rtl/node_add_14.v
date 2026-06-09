`timescale 1ns / 1ps
// node_add_14 - tiled-streaming INT8 quantized residual ADD.
// spec_hash: add_2048x2048_s7x7_i512_o256_iotiled-streaming_tile32
// channel_tile = 32, BEATS_PER_PIXEL = 64, OC = 2048.
// Bus contract: data_in[255:0] = lhs tile, data_in[511:256] = rhs tile,
//               data_out[255:0] = one 32-channel INT8 output tile beat.
// Quantization: unconditional +HALF rounding, fused multipliers normalised
//   by out_scale; r_lhs = lhs_scale_factor / scale_factor,
//   r_rhs = rhs_scale_factor / scale_factor.
//   FUSED_SHIFT = 16. LHS_FUSED_MULT = round(r_lhs * 2^23) = 6881387.
//                     RHS_FUSED_MULT = round(r_rhs * 2^23) = 13882290.
// Latency first_valid_in -> first_valid_out = BEATS_PER_PIXEL + OC + 2 = 2114.

module node_add_14 (
    input  wire           clk,
    input  wire           rst_n,
    input  wire           valid_in,
    output reg            ready_in,
    input  wire [511:0]   data_in,
    output reg            valid_out,
    input  wire           ready_out,   // [BP-FIX] downstream-ready: stall stream when low (was missing -> dropped beats)
    output reg  [255:0]   data_out
);

    localparam integer OC              = 2048;
    localparam integer CHANNEL_TILE    = 32;
    localparam integer BEATS_PER_PIXEL = 64;
    localparam integer FUSED_SHIFT     = 16;

    localparam integer MULT_W  = 25;
    localparam integer PROD_W  = 8 + MULT_W;   // 33
    localparam integer SUM_W   = PROD_W + 1;   // 34
    localparam integer TERM_W  = SUM_W + 1;    // 35

    localparam signed [MULT_W-1:0] LHS_FUSED_MULT  = 34'sd44859;
    localparam signed [MULT_W-1:0] RHS_FUSED_MULT  = 34'sd78378;
    localparam signed [TERM_W-1:0] FUSED_ROUND_BIAS = 34'sd32768; // 1<<(FUSED_SHIFT-1)
    localparam signed [TERM_W-1:0] SAT_HI =  35'sd127;
    localparam signed [TERM_W-1:0] SAT_LO = -35'sd128;

    localparam [1:0] ST_IDLE    = 2'd0,
                     ST_GATHER  = 2'd1,
                     ST_COMPUTE = 2'd2,
                     ST_STREAM  = 2'd3;

    reg [1:0]  state;
    reg [6:0]  in_beat_count;
    reg [6:0]  out_beat_count;
    reg        cur_beat_stream;

    reg [11:0] ch_s1;
    reg [11:0] ch_s2;
    reg [11:0] ch_s3;
    reg        stage1_active;
    reg        stage2_valid;
    reg        stage3_valid;

    (* ram_style = "block" *) reg signed [7:0] lhs_buf  [0:OC-1];
    (* ram_style = "block" *) reg signed [7:0] rhs_buf  [0:OC-1];
    (* ram_style = "block" *) reg [255:0]      out_beats [0:BEATS_PER_PIXEL-1];

    reg signed [PROD_W-1:0] lhs_term;
    reg signed [PROD_W-1:0] rhs_term;
    reg signed [TERM_W-1:0] sum_term;

    wire signed [SUM_W-1:0]  sum_pre   = $signed(lhs_term) + $signed(rhs_term);
    wire signed [TERM_W-1:0] shifted_w = sum_term >>> FUSED_SHIFT;
    wire [7:0] sat_w = (shifted_w > SAT_HI) ? 8'h7F :
                       (shifted_w < SAT_LO) ? 8'h80 :
                                              shifted_w[7:0];

    wire [11:0] beat_idx  = ch_s3[11:5];           // ch_s3 / 32
    wire [4:0]  lane_idx  = ch_s3[4:0];            // ch_s3 % 32

    wire signed [7:0] lhs_rd = lhs_buf[ch_s1];
    wire signed [7:0] rhs_rd = rhs_buf[ch_s1];

    integer gi;
    integer bi;

    // [K1-FDCE] Block A: array/data writes (sync-only) -- node_add_1
    // precedent. lhs_buf/rhs_buf are fully rewritten during each pixel's
    // gather before ST_COMPUTE reads them; every out_beats byte is written
    // by the 3-stage pipe before ST_STREAM presents it under valid_out.
    // Sync-only writes also unblock RAM inference for lhs/rhs.
    always @(posedge clk) begin
        if (state == ST_IDLE && valid_in) begin
            for (gi = 0; gi < CHANNEL_TILE; gi = gi + 1) begin
                lhs_buf[gi] <= $signed(data_in[gi*8 +: 8]);
                rhs_buf[gi] <= $signed(data_in[256 + gi*8 +: 8]);
            end
        end
        if (state == ST_GATHER && valid_in) begin
            for (gi = 0; gi < CHANNEL_TILE; gi = gi + 1) begin
                lhs_buf[in_beat_count*CHANNEL_TILE + gi] <= $signed(data_in[gi*8 +: 8]);
                rhs_buf[in_beat_count*CHANNEL_TILE + gi] <= $signed(data_in[256 + gi*8 +: 8]);
            end
        end
        if (stage3_valid) begin
            out_beats[beat_idx][lane_idx*8 +: 8] <= sat_w;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= ST_IDLE;
            ready_in        <= 1'b1; // [INVARIANT:READY_IN_GATING]
            valid_out       <= 1'b0;
            data_out        <= 256'd0;
            in_beat_count   <= 7'd0;
            out_beat_count  <= 7'd0;
            cur_beat_stream <= 1'b0;
            ch_s1           <= 12'd0;
            ch_s2           <= 12'd0;
            ch_s3           <= 12'd0;
            stage1_active   <= 1'b0;
            stage2_valid    <= 1'b0;
            stage3_valid    <= 1'b0;
            lhs_term        <= {PROD_W{1'b0}};
            rhs_term        <= {PROD_W{1'b0}};
            sum_term        <= {TERM_W{1'b0}};
        end else begin
            valid_out <= 1'b0;

            // 3-stage MAC pipeline (advances unconditionally; gated by stageN_valid).
            lhs_term <= $signed(lhs_rd) * LHS_FUSED_MULT;
            rhs_term <= $signed(rhs_rd) * RHS_FUSED_MULT;
            ch_s2    <= ch_s1;
            stage2_valid <= stage1_active;

            sum_term     <= sum_pre + FUSED_ROUND_BIAS; // [INVARIANT:ROUNDING]
            ch_s3        <= ch_s2;
            stage3_valid <= stage2_valid;

            case (state)
                ST_IDLE: begin
                    ready_in <= 1'b1;
                    if (valid_in) begin
                        in_beat_count <= 7'd1;
                        state         <= ST_GATHER;
                    end
                end

                ST_GATHER: begin
                    if (valid_in) begin
                        if (in_beat_count == BEATS_PER_PIXEL-1) begin
                            ready_in      <= 1'b0;
                            in_beat_count <= 7'd0;
                            state         <= ST_COMPUTE;
                            ch_s1         <= 12'd0;
                            stage1_active <= 1'b1;
                        end else begin
                            in_beat_count <= in_beat_count + 7'd1;
                        end
                    end
                end

                ST_COMPUTE: begin
                    if (stage1_active) begin
                        if (ch_s1 == OC-1) begin
                            stage1_active <= 1'b0;
                        end else begin
                            ch_s1 <= ch_s1 + 12'd1;
                        end
                    end

                    if (stage3_valid && (ch_s3 == OC-1)) begin
                        state           <= ST_STREAM;
                        out_beat_count  <= 7'd1;
                        cur_beat_stream <= 1'b1;
                        valid_out       <= 1'b1; // [INVARIANT:VALID_OUT_LATENCY]
                        data_out        <= out_beats[0];
                    end
                end

                ST_STREAM: begin
                    // [BP-FIX] Only advance when the downstream ACCEPTS the currently
                    // presented beat (valid_out & ready_out). When ready_out is low,
                    // HOLD valid_out + data_out + out_beat_count (no drop). Beat 0 was
                    // presented at the COMPUTE->STREAM transition with out_beat_count=1.
                    if (ready_out) begin
                        if (out_beat_count == BEATS_PER_PIXEL) begin
                            state           <= ST_IDLE;
                            out_beat_count  <= 7'd0;
                            cur_beat_stream <= 1'b0;
                            valid_out       <= 1'b0;
                            ready_in        <= 1'b1;
                        end else begin
                            valid_out      <= 1'b1;
                            data_out       <= out_beats[out_beat_count];
                            out_beat_count <= out_beat_count + 7'd1;
                            cur_beat_stream <= 1'b1;
                        end
                    end else begin
                        // hold: re-assert valid_out (overrides the per-cycle default),
                        // keep data_out + out_beat_count unchanged -> no beat lost
                        valid_out <= 1'b1;
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
