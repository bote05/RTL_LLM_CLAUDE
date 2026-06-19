// Foundry: node_add_546 - INT8 quantized residual add, OC=64, flat-bus.
// Channel-serialized 3-stage requantize pipeline per 05_add_quantized.md.
//   data_in[511:0]    = lhs (64 INT8 channels)
//   data_in[1023:512] = rhs (64 INT8 channels)
//   out_i = saturate( ( lhs_i * (lhs_scale/out_scale)
//                      + rhs_i * (rhs_scale/out_scale) )  >>> SHIFT )
// pipeline_latency_cycles = OC + 3 = 67.
module node_add_618 (
    input  wire           clk,
    input  wire           rst_n,
    input  wire           valid_in,
    output reg            ready_in,
    input  wire [1023:0]  data_in,
    output reg            valid_out,
    output reg  [511:0]   data_out
);

    // ---------- Geometry ----------
    localparam integer OC       = 64;
    localparam integer W        = 512;   // output_width_bits, = lhs slice width
    localparam integer CH_IDX_W = 7;     // ceil_log2(OC+1)

    // ---------- Fused per-side scale constants ----------
    // LHS_FUSED = lhs_scale / out_scale = 0.13125772.../0.16331340... ~ 0.80371678
    //           ~ 26336 / 2^15 = 0.80371094     (rel err ~ 7.3e-6)
    // RHS_FUSED = rhs_scale / out_scale = 0.13779990.../0.16331340... ~ 0.84377580
    //           ~ 27649 / 2^15 = 0.84378052     (rel err ~ 5.6e-6)
    localparam integer FUSED_SHIFT   = 22;
    localparam integer SCALE_CONST_W = 23;

    // ---------- Internal widths ----------
    localparam integer PROD_W = 8 + SCALE_CONST_W;   // 24-bit signed
    localparam integer SUM_W  = PROD_W + 2;          // 26-bit signed (absorb sum + round bias)

    // Multiplier constants pre-widened to PROD_W signed so the multiply is
    // unambiguously context-determined at PROD_W. Both fit in 24-bit signed
    // positive (max product magnitude = 128*27649 ~ 3.5M, well within +/-8.4M).
    localparam signed [PROD_W-1:0] LHS_M = 26336;
    localparam signed [PROD_W-1:0] RHS_M = 27649;

    // Sign-aware rounding bias. Verilog >>> always floors toward -inf, so the
    // negative branch must be (HALF - 1), NOT -HALF (subtracting HALF
    // over-rounds by one LSB on negatives).
    localparam signed [SUM_W-1:0] FUSED_HALF =
        {{(SUM_W-1){1'b0}}, 1'b1} <<< (FUSED_SHIFT - 1);
    localparam signed [SUM_W-1:0] FUSED_HALF_M1 =
        FUSED_HALF - {{(SUM_W-1){1'b0}}, 1'b1};

    localparam signed [SUM_W-1:0] SAT_HI =  127;
    localparam signed [SUM_W-1:0] SAT_LO = -128;

    // ---------- FSM ----------
    localparam ST_IDLE = 1'b0;
    localparam ST_RUN  = 1'b1;
    reg state;

    // ---------- Pipeline registers ----------
    reg  [1023:0]            input_buf;
    reg  [CH_IDX_W-1:0]      ch_idx;
    reg  [CH_IDX_W-1:0]      stage1_idx;
    reg  [CH_IDX_W-1:0]      stage2_idx;
    reg                      stage1_valid;
    reg                      stage2_valid;

    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] lhs_term;
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] rhs_term;
    reg signed [SUM_W-1:0]   sum_term;

    // ---------- Combinational helpers (module scope, Verilog-2001) ----------
    wire signed [7:0] lhs_ch;
    wire signed [7:0] rhs_ch;
    assign lhs_ch = $signed(input_buf[ch_idx*8     +: 8]);
    assign rhs_ch = $signed(input_buf[W + ch_idx*8 +: 8]);

    wire signed [PROD_W-1:0] lhs_op;
    wire signed [PROD_W-1:0] rhs_op;
    assign lhs_op = lhs_ch;
    assign rhs_op = rhs_ch;

    wire signed [SUM_W-1:0] sum_pre;
    assign sum_pre = $signed(lhs_term) + $signed(rhs_term);

    wire signed [SUM_W-1:0] out_pre;
    assign out_pre = sum_term >>> FUSED_SHIFT;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= ST_IDLE;
            ready_in     <= 1'b1;
            valid_out    <= 1'b0;
            data_out     <= {W{1'b0}};
            input_buf    <= {1024{1'b0}};
            ch_idx       <= {CH_IDX_W{1'b0}};
            stage1_idx   <= {CH_IDX_W{1'b0}};
            stage2_idx   <= {CH_IDX_W{1'b0}};
            stage1_valid <= 1'b0;
            stage2_valid <= 1'b0;
            lhs_term     <= {PROD_W{1'b0}};
            rhs_term     <= {PROD_W{1'b0}};
            sum_term     <= {SUM_W{1'b0}};
        end else begin
            valid_out <= 1'b0;

            case (state)
                ST_IDLE: begin
                    stage1_valid <= 1'b0;
                    stage2_valid <= 1'b0;
                    if (valid_in && ready_in) begin
                        input_buf <= data_in;
                        ready_in  <= 1'b0;    // [INVARIANT:READY_IN_GATING]
                        ch_idx    <= {CH_IDX_W{1'b0}};
                        state     <= ST_RUN;
                    end
                end

                ST_RUN: begin
                    if (ch_idx < OC) begin
                        lhs_term     <= lhs_op * LHS_M;
                        rhs_term     <= rhs_op * RHS_M;
                        stage1_valid <= 1'b1;
                        stage1_idx   <= ch_idx;
                        ch_idx       <= ch_idx + 1'b1;
                    end else begin
                        stage1_valid <= 1'b0;
                    end

                    if (stage1_valid) begin
                        // [INVARIANT:ROUNDING]
                        sum_term     <= sum_pre + (sum_pre[SUM_W-1] ? FUSED_HALF_M1 : FUSED_HALF);
                        stage2_valid <= 1'b1;
                        stage2_idx   <= stage1_idx;
                    end else begin
                        stage2_valid <= 1'b0;
                    end

                    if (stage2_valid) begin
                        if (out_pre > SAT_HI)
                            data_out[stage2_idx*8 +: 8] <= 8'sd127;
                        else if (out_pre < SAT_LO)
                            data_out[stage2_idx*8 +: 8] <= -8'sd128;
                        else
                            data_out[stage2_idx*8 +: 8] <= out_pre[7:0];

                        if (stage2_idx == (OC-1)) begin
                            valid_out <= 1'b1;    // [INVARIANT:VALID_OUT_LATENCY]
                            ready_in  <= 1'b1;    // [INVARIANT:READY_IN_GATING]
                            state     <= ST_IDLE;
                        end
                    end
                end
            endcase
        end
    end

endmodule
