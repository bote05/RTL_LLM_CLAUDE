// node_add_25 — INT8 residual add, flat-bus, OC=16
// data_in[127:0]   = lhs (16 ch * 8 bits)
// data_in[255:128] = rhs (16 ch * 8 bits)
// data_out[127:0]  = saturated INT8 sum, 16 channels packed

module node_add_25_old (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    output reg          ready_in,
    input  wire [255:0] data_in,
    output reg          valid_out,
    output reg  [127:0] data_out
);

    localparam integer OC          = 16;
    localparam integer FUSED_SHIFT = 14;
    localparam integer MULT_W      = 24;
    localparam integer PROD_W      = 32;  // 8 + MULT_W
    localparam integer SUM_W       = 34;  // PROD_W + 2

    // Fused scale: r_lhs = lhs_scale/out_scale = 0.5878958..., r_rhs = 1.0
    localparam signed [MULT_W-1:0] FUSED_LHS_MULT   = 24'sd9632;
    localparam signed [MULT_W-1:0] FUSED_RHS_MULT   = 24'sd16384;
    localparam signed [SUM_W-1:0]  FUSED_ROUND_BIAS = 34'sd8192;   // 1 << (FUSED_SHIFT-1)
    localparam signed [SUM_W-1:0]  SAT_HI =  34'sd127;
    localparam signed [SUM_W-1:0]  SAT_LO = -34'sd128;

    localparam [1:0] ST_IDLE = 2'd0, ST_COMPUTE = 2'd1, ST_DONE = 2'd2;
    reg [1:0] state;

    reg signed [7:0] lhs_buf [0:OC-1];
    reg signed [7:0] rhs_buf [0:OC-1];

    reg [4:0] ch_s1, ch_s2, ch_s3;
    reg [0:0] in_beat_idx, out_beat_idx;

    reg stage1_active, stage2_valid, stage3_valid;

    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] lhs_term;
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] rhs_term;
    reg signed [SUM_W-1:0]  sum_term;
    reg signed [SUM_W-1:0]  v_tmp;
    wire signed [SUM_W-1:0] sum_pre = $signed(lhs_term) + $signed(rhs_term);

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= ST_IDLE;
            ready_in      <= 1'b1;
            valid_out     <= 1'b0;
            data_out      <= 128'd0;
            ch_s1         <= 5'd0;
            ch_s2         <= 5'd0;
            ch_s3         <= 5'd0;
            in_beat_idx   <= 1'b0;
            out_beat_idx  <= 1'b0;
            stage1_active <= 1'b0;
            stage2_valid  <= 1'b0;
            stage3_valid  <= 1'b0;
            lhs_term      <= {PROD_W{1'b0}};
            rhs_term      <= {PROD_W{1'b0}};
            sum_term      <= {SUM_W{1'b0}};
            v_tmp         <= {SUM_W{1'b0}};
            for (i = 0; i < OC; i = i + 1) begin
                lhs_buf[i] <= 8'sd0;
                rhs_buf[i] <= 8'sd0;
            end
        end else begin
            // ---- 3-stage MAC / round / saturate pipeline (free-running) ----
            if (stage1_active) begin
                lhs_term     <= $signed(lhs_buf[ch_s1]) * FUSED_LHS_MULT;
                rhs_term     <= $signed(rhs_buf[ch_s1]) * FUSED_RHS_MULT;
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
                data_out[ch_s3*8 +: 8] <= (v_tmp > SAT_HI) ? 8'sd127 :
                                          (v_tmp < SAT_LO) ? 8'h80   : v_tmp[7:0];
            end

            // ---- FSM ----
            case (state)
                ST_IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in && ready_in) begin
                        for (i = 0; i < OC; i = i + 1) begin
                            lhs_buf[i] <= $signed(data_in[i*8 +: 8]);
                            rhs_buf[i] <= $signed(data_in[128 + i*8 +: 8]);
                        end
                        ready_in      <= 1'b0;             // [INVARIANT:READY_IN_GATING]
                        state         <= ST_COMPUTE;
                        stage1_active <= 1'b1;
                        ch_s1         <= 5'd0;
                        in_beat_idx   <= 1'b1;
                    end
                end
                ST_COMPUTE: begin
                    if (stage1_active) begin
                        if (ch_s1 == 5'd15) stage1_active <= 1'b0;
                        else                ch_s1 <= ch_s1 + 5'd1;
                    end
                    if (stage3_valid && ch_s3 == 5'd15) begin
                        valid_out    <= 1'b1;              // [INVARIANT:VALID_OUT_LATENCY]
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
