// node_add_9 - tiled-streaming INT8 residual add
// OC=1024, CHANNEL_TILE=32, BEATS_PER_PIXEL=32
// data_in[511:0] = {rhs_tile[511:256], lhs_tile[255:0]}, data_out[255:0] = one tile beat
`timescale 1ns/1ps

module node_add_9 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    output reg          ready_in,
    input  wire [511:0] data_in,
    output reg          valid_out,
    input  wire         ready_out,   // [BP-FIX] downstream-ready: stall stream when low (was missing -> dropped beats)
    output reg  [255:0] data_out
);

    localparam integer OC              = 1024;
    localparam integer CHANNEL_TILE    = 32;
    localparam integer BEATS_PER_PIXEL = OC / CHANNEL_TILE; // 32

    localparam [5:0] BEATS_LAST6  = 6'd31;
    localparam [5:0] BEATS_TOTAL6 = 6'd32;

    localparam integer FUSED_SHIFT     = 10;
    localparam integer MULT_W          = 24;
    localparam integer PROD_W          = 32;
    localparam integer SUM_W           = 34;

    localparam signed [MULT_W-1:0] FUSED_LHS_MULT   = 24'sd1024;
    localparam signed [MULT_W-1:0] FUSED_RHS_MULT   = 24'sd1133;
    localparam signed [SUM_W-1:0]  FUSED_ROUND_BIAS = 34'sd512;
    localparam signed [SUM_W-1:0]  SAT_HI =  34'sd127;
    localparam signed [SUM_W-1:0]  SAT_LO = -34'sd128;

    localparam [1:0] ST_IDLE    = 2'd0,
                     ST_GATHER  = 2'd1,
                     ST_COMPUTE = 2'd2,
                     ST_STREAM  = 2'd3;

    reg [1:0] state;
    reg signed [7:0] lhs_buf [0:OC-1];
    reg signed [7:0] rhs_buf [0:OC-1];
    reg [255:0]      out_beats [0:BEATS_PER_PIXEL-1];
    reg [5:0]  in_beat_count;
    reg [5:0]  out_beat_count;
    reg [9:0]  ch_s1, ch_s2, ch_s3;
    reg stage1_active, stage2_valid, stage3_valid;
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] lhs_term;
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] rhs_term;
    reg  signed [SUM_W-1:0]  sum_term;
    wire signed [SUM_W-1:0] sum_pre = $signed(lhs_term) + $signed(rhs_term);
    wire signed [SUM_W-1:0] v_tmp   = sum_term >>> FUSED_SHIFT;
    integer i;
    integer j;

    always @(posedge clk) begin
        if (state == ST_IDLE) begin
            if (valid_in && ready_in) begin
                for (i = 0; i < CHANNEL_TILE; i = i + 1) begin
                    lhs_buf[i] <= $signed(data_in[i*8 +: 8]);
                    rhs_buf[i] <= $signed(data_in[256 + i*8 +: 8]);
                end
            end
        end else if (state == ST_GATHER) begin
            if (valid_in && ready_in) begin
                for (j = 0; j < CHANNEL_TILE; j = j + 1) begin
                    lhs_buf[in_beat_count*CHANNEL_TILE + j] <= $signed(data_in[j*8 +: 8]);
                    rhs_buf[in_beat_count*CHANNEL_TILE + j] <= $signed(data_in[256 + j*8 +: 8]);
                end
            end
        end
        if (stage3_valid) begin
            out_beats[ch_s3 / CHANNEL_TILE][(ch_s3 % CHANNEL_TILE)*8 +: 8] <=
                (v_tmp > SAT_HI) ? 8'sd127 :
                (v_tmp < SAT_LO) ? 8'h80   :
                                   v_tmp[7:0];
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE; ready_in <= 1'b1; valid_out <= 1'b0; data_out <= 256'd0;
            in_beat_count <= 6'd0; out_beat_count <= 6'd0;
            ch_s1 <= 10'd0; ch_s2 <= 10'd0; ch_s3 <= 10'd0;
            stage1_active <= 1'b0; stage2_valid <= 1'b0; stage3_valid <= 1'b0;
            lhs_term <= {PROD_W{1'b0}}; rhs_term <= {PROD_W{1'b0}}; sum_term <= {SUM_W{1'b0}};
        end else begin
            if (stage1_active) begin
                lhs_term <= $signed(lhs_buf[ch_s1]) * FUSED_LHS_MULT;
                rhs_term <= $signed(rhs_buf[ch_s1]) * FUSED_RHS_MULT;
                ch_s2 <= ch_s1; stage2_valid <= 1'b1;
            end else stage2_valid <= 1'b0;
            if (stage2_valid) begin
                sum_term <= sum_pre + FUSED_ROUND_BIAS; // [INVARIANT:ROUNDING]
                ch_s3 <= ch_s2; stage3_valid <= 1'b1;
            end else stage3_valid <= 1'b0;
            case (state)
                ST_IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in && ready_in) begin in_beat_count <= 6'd1; state <= ST_GATHER; end
                end
                ST_GATHER: begin
                    if (valid_in && ready_in) begin
                        if (in_beat_count == BEATS_LAST6) begin
                            ready_in <= 1'b0; // [INVARIANT:READY_IN_GATING]
                            state <= ST_COMPUTE; ch_s1 <= 10'd0; stage1_active <= 1'b1; in_beat_count <= 6'd0;
                        end else in_beat_count <= in_beat_count + 6'd1;
                    end
                end
                ST_COMPUTE: begin
                    if (stage1_active) begin
                        if (ch_s1 == (OC - 1)) stage1_active <= 1'b0;
                        else ch_s1 <= ch_s1 + 10'd1;
                    end
                    if (stage3_valid && ch_s3 == (OC - 1)) begin
                        state <= ST_STREAM;
                        data_out <= out_beats[0]; // [INVARIANT:VALID_OUT_LATENCY]
                        valid_out <= 1'b1; out_beat_count <= 6'd1;
                    end
                end
                ST_STREAM: begin
                    // [BP-FIX] Only advance when the downstream ACCEPTS the currently
                    // presented beat (valid_out & ready_out). When ready_out is low,
                    // HOLD valid_out + data_out + out_beat_count (no drop). Beat 0 was
                    // presented at the COMPUTE->STREAM transition with out_beat_count=1.
                    if (ready_out) begin
                        if (out_beat_count < BEATS_TOTAL6) begin
                            data_out <= out_beats[out_beat_count];
                            valid_out <= 1'b1; out_beat_count <= out_beat_count + 6'd1;
                        end else begin
                            valid_out <= 1'b0; state <= ST_IDLE; ready_in <= 1'b1; out_beat_count <= 6'd0;
                        end
                    end
                    // else: hold (no change) -- key to losslessness
                end
                default: state <= ST_IDLE;
            endcase
        end
    end
endmodule
