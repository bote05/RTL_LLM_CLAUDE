`timescale 1ns / 1ps

// node_add_87 -- INT8 residual add, flat-bus, OC=64, 8x8 spatial.
// data_in[ 511:  0] = lhs (64 INT8 channels packed)
// data_in[1023:512] = rhs (64 INT8 channels packed)
// data_out[511:0]   = saturated INT8 result, 64 channels packed
//
// out_q = sat( ((lhs_q * lhs_scale + rhs_q * rhs_scale) * scale) >>> shift, INT8 )
// implemented as fused 2-scale form:
//   r_lhs = lhs_scale_factor / scale_factor = 0.5059140848892504
//   r_rhs = rhs_scale_factor / scale_factor = 0.8768556548645362
//   out_q = sat( (lhs_q * R_LHS_MULT + rhs_q * R_RHS_MULT + ROUND_BIAS) >>> FUSED_SHIFT, INT8 )
// FUSED_SHIFT=23, R_LHS_MULT=4243915, R_RHS_MULT=7355598.
//
// Pipeline latency: 67 cycles from first valid_in&ready_in to first valid_out.
// 1 cycle capture + 64 channel serialize through 3-stage pipeline
// (mult / sum+round / shift+sat+pack) + drain = 67.
//
// Buffer writes are intentionally placed in a sync-only always block
// to allow Vivado LUTRAM/distributed-RAM inference for lhs_buf/rhs_buf
// (avoiding the async-reset preflight rejection that killed earlier
// node_add_25 / node_add_56 attempts).

module node_add_87_old (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          valid_in,
    output reg           ready_in,
    input  wire [1023:0] data_in,
    output reg           valid_out,
    output reg  [511:0]  data_out
);

    localparam integer OC          = 64;
    localparam integer FUSED_SHIFT = 23;
    localparam integer MULT_W      = 24;
    localparam integer PROD_W      = 8 + MULT_W;
    localparam integer SUM_W       = PROD_W + 2;

    localparam signed [MULT_W-1:0] FUSED_LHS_MULT = 24'sd4243915;
    localparam signed [MULT_W-1:0] FUSED_RHS_MULT = 24'sd7355598;
    localparam signed [SUM_W-1:0]  FUSED_ROUND_BIAS = 34'sd4194304;
    localparam signed [SUM_W-1:0]  SAT_HI =  34'sd127;
    localparam signed [SUM_W-1:0]  SAT_LO = -34'sd128;

    localparam [1:0] ST_IDLE    = 2'd0;
    localparam [1:0] ST_COMPUTE = 2'd1;
    localparam [1:0] ST_DONE    = 2'd2;
    reg [1:0] state;

    reg in_beat_idx;
    reg out_beat_idx;

    reg signed [7:0] lhs_buf [0:OC-1];
    reg signed [7:0] rhs_buf [0:OC-1];

    reg [6:0] ch_s1, ch_s2, ch_s3;
    reg       stage1_active, stage2_valid, stage3_valid;

    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] lhs_term;
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] rhs_term;
    reg signed [SUM_W-1:0]  sum_term;
    reg signed [SUM_W-1:0]  v_tmp;

    wire signed [SUM_W-1:0] sum_pre = $signed(lhs_term) + $signed(rhs_term);

    integer i_load;

    always @(posedge clk) begin
        if (state == ST_IDLE && valid_in && ready_in) begin
            for (i_load = 0; i_load < OC; i_load = i_load + 1) begin
                lhs_buf[i_load] <= $signed(data_in[i_load*8 +: 8]);
                rhs_buf[i_load] <= $signed(data_in[512 + i_load*8 +: 8]);
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= ST_IDLE;
            ready_in      <= 1'b1;
            valid_out     <= 1'b0;
            data_out      <= 512'd0;
            in_beat_idx   <= 1'b0;
            out_beat_idx  <= 1'b0;
            ch_s1         <= 7'd0;
            ch_s2         <= 7'd0;
            ch_s3         <= 7'd0;
            stage1_active <= 1'b0;
            stage2_valid  <= 1'b0;
            stage3_valid  <= 1'b0;
            lhs_term      <= {PROD_W{1'b0}};
            rhs_term      <= {PROD_W{1'b0}};
            sum_term      <= {SUM_W{1'b0}};
            v_tmp         <= {SUM_W{1'b0}};
        end else begin
            if (stage1_active) begin
                lhs_term     <= $signed(lhs_buf[ch_s1]) * FUSED_LHS_MULT;
                rhs_term     <= $signed(rhs_buf[ch_s1]) * FUSED_RHS_MULT;
                ch_s2        <= ch_s1;
                stage2_valid <= 1'b1;
            end else begin
                stage2_valid <= 1'b0;
            end

            if (stage2_valid) begin
                sum_term     <= sum_pre + FUSED_ROUND_BIAS;
                ch_s3        <= ch_s2;
                stage3_valid <= 1'b1;
            end else begin
                stage3_valid <= 1'b0;
            end

            if (stage3_valid) begin
                v_tmp = sum_term >>> FUSED_SHIFT;
                data_out[ch_s3*8 +: 8] <= (v_tmp > SAT_HI) ? 8'sd127 :
                                          (v_tmp < SAT_LO) ? 8'h80   :
                                          v_tmp[7:0];
            end

            case (state)
                ST_IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in && ready_in) begin
                        ready_in      <= 1'b0;
                        state         <= ST_COMPUTE;
                        stage1_active <= 1'b1;
                        ch_s1         <= 7'd0;
                        in_beat_idx   <= 1'b1;
                    end
                end
                ST_COMPUTE: begin
                    if (stage1_active) begin
                        if (ch_s1 == (OC - 1)) stage1_active <= 1'b0;
                        else                    ch_s1 <= ch_s1 + 7'd1;
                    end
                    if (stage3_valid && ch_s3 == (OC - 1)) begin
                        valid_out    <= 1'b1;
                        state        <= ST_DONE;
                        out_beat_idx <= 1'b1;
                    end
                end
                ST_DONE: begin
                    valid_out    <= 1'b0;
                    ready_in     <= 1'b1;
                    state        <= ST_IDLE;
                    in_beat_idx  <= 1'b0;
                    out_beat_idx <= 1'b0;
                end
                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
