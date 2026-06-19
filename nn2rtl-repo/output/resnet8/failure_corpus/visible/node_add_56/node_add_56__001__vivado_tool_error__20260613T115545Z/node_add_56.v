`timescale 1ns/1ps

// node_add_56 — flat-bus packed_full residual add
// Geometry: OC=32, IC=32, spatial=16x16, data_in=512 (lhs|rhs each 256b),
//           data_out=256 (32 INT8 channels). One beat in -> one beat out.
// pipeline_latency_cycles = OC + 3 = 35 cycles from first valid_in to first valid_out.

module node_add_56 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    output reg          ready_in,
    input  wire [511:0] data_in,
    output reg          valid_out,
    output reg  [255:0] data_out
);

    localparam integer OC           = 32;
    localparam integer FUSED_SHIFT  = 22;
    localparam integer MULT_W       = 24;
    localparam integer PROD_W       = 32;
    localparam integer SUM_W        = 34;

    // Fused requantisation constants — ratios pre-divided by out_scale:
    //   lhs_in = 0.04880561978798213, rhs_in = 0.11695987220824235,
    //   out    = 0.13737633472352517
    //   r_lhs  = lhs_in / out ~= 0.3552697  -> round(r_lhs * 2^22) = 1490108
    //   r_rhs  = rhs_in / out ~= 0.8513830  -> round(r_rhs * 2^22) = 3570959
    localparam signed [MULT_W-1:0] FUSED_LHS_MULT   = 24'sd1490108;
    localparam signed [MULT_W-1:0] FUSED_RHS_MULT   = 24'sd3570959;
    // round_half_up_toward_pos_inf — UNCONDITIONAL +HALF (matches golden_impl.py Int8Add).
    localparam signed [SUM_W-1:0]  FUSED_ROUND_BIAS = 34'sd2097152;  // 1 << (FUSED_SHIFT-1)
    localparam signed [SUM_W-1:0]  SAT_HI           =  34'sd127;
    localparam signed [SUM_W-1:0]  SAT_LO           = -34'sd128;

    localparam [1:0] ST_IDLE    = 2'd0,
                     ST_COMPUTE = 2'd1;
    reg [1:0] state;

    reg signed [7:0] lhs_buf [0:OC-1];
    reg signed [7:0] rhs_buf [0:OC-1];

    reg [5:0] ch_s1, ch_s2, ch_s3;
    reg       stage1_active, stage2_valid, stage3_valid;

    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] lhs_term;
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] rhs_term;
    reg signed [SUM_W-1:0]  sum_term;
    reg signed [SUM_W-1:0]  v_tmp;
    wire signed [SUM_W-1:0] sum_pre = $signed(lhs_term) + $signed(rhs_term);

    integer i;

    always @(posedge clk) begin
        if (state == ST_IDLE && valid_in && ready_in) begin
            for (i = 0; i < OC; i = i + 1) begin
                lhs_buf[i] <= $signed(data_in[i*8 +: 8]);
                rhs_buf[i] <= $signed(data_in[256 + i*8 +: 8]);
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= ST_IDLE;
            ready_in      <= 1'b1;
            valid_out     <= 1'b0;
            data_out      <= 256'd0;
            ch_s1         <= 6'd0;
            ch_s2         <= 6'd0;
            ch_s3         <= 6'd0;
            stage1_active <= 1'b0;
            stage2_valid  <= 1'b0;
            stage3_valid  <= 1'b0;
            lhs_term      <= 32'sd0;
            rhs_term      <= 32'sd0;
            sum_term      <= 34'sd0;
            v_tmp         <= 34'sd0;
        end else begin
            if (stage1_active) begin
                lhs_term     <= $signed(lhs_buf[ch_s1[4:0]]) * FUSED_LHS_MULT;
                rhs_term     <= $signed(rhs_buf[ch_s1[4:0]]) * FUSED_RHS_MULT;
                ch_s2        <= ch_s1;
                stage2_valid <= 1'b1;
            end else begin
                stage2_valid <= 1'b0;
            end

            if (stage2_valid) begin
                sum_term     <= sum_pre + FUSED_ROUND_BIAS;  // [INVARIANT:ROUNDING]
                ch_s3        <= ch_s2;
                stage3_valid <= 1'b1;
            end else begin
                stage3_valid <= 1'b0;
            end

            if (stage3_valid) begin
                v_tmp = sum_term >>> FUSED_SHIFT;
                data_out[ch_s3[4:0]*8 +: 8] <=
                    (v_tmp > SAT_HI) ? 8'sd127 :
                    (v_tmp < SAT_LO) ? 8'h80   :
                                       v_tmp[7:0];
                if (ch_s3 == (OC - 1)) begin
                    valid_out <= 1'b1;          // [INVARIANT:VALID_OUT_LATENCY]
                    ready_in  <= 1'b1;
                    state     <= ST_IDLE;
                end
            end

            case (state)
                ST_IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in && ready_in) begin
                        ready_in      <= 1'b0;   // [INVARIANT:READY_IN_GATING]
                        state         <= ST_COMPUTE;
                        ch_s1         <= 6'd0;
                        stage1_active <= 1'b1;
                    end
                end
                ST_COMPUTE: begin
                    if (stage1_active) begin
                        if (ch_s1 == (OC - 1)) stage1_active <= 1'b0;
                        else                   ch_s1 <= ch_s1 + 6'd1;
                    end
                end
                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
