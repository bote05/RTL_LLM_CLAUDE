// Foundry: node_add_546 - INT8 quantized residual add, OC=64, flat-bus.
// Channel-serialized 3-stage requantize pipeline per 05_add_quantized.md.
//   data_in[511:0]    = lhs (64 INT8 channels)
//   data_in[1023:512] = rhs (64 INT8 channels)
//   out_i = saturate( ( lhs_i * (lhs_scale/out_scale)
//                      + rhs_i * (rhs_scale/out_scale) )  >>> SHIFT )
// pipeline_latency_cycles = OC + 3 = 67.
// PARAM-GATED ELASTIC BACKPRESSURE (ENABLE_BACKPRESSURE, default 0):
//   * ==0 (default): bit/cycle-IDENTICAL to the legacy add. out_ready_in is
//     IGNORED; skid_block is constant 0; valid_out/data_out come straight from
//     the datapath (dp_valid_out/dp_data_out). The add arithmetic is UNCHANGED.
//   * ==1: 1-deep output skid holds the 1-cycle result beat until out_ready_in;
//     ready_in drops while a beat is parked, freezing the NEXT frame's accept so
//     the single-beat-per-frame producer can never overrun the skid.
module node_add_546 #(
    parameter ENABLE_BACKPRESSURE = 0
)(
    input  wire           clk,
    input  wire           rst_n,
    input  wire           valid_in,
    output reg            ready_in,
    input  wire [1023:0]  data_in,
    input  wire           out_ready_in,   // NEW: downstream-ready (ignored when ENABLE_BACKPRESSURE==0)
    output wire           valid_out,
    output wire [511:0]   data_out
);

    // ---- datapath output regs + 1-deep output skid (see header) ----
    reg                 dp_valid_out;
    reg  [511:0]        dp_data_out;
    reg                 out_full;
    reg  [511:0]        out_data;
    wire skid_block = (ENABLE_BACKPRESSURE != 0) && out_full && !out_ready_in;

    generate
    if (ENABLE_BACKPRESSURE == 0) begin : g_out_legacy
        assign valid_out = dp_valid_out;
        assign data_out  = dp_data_out;
    end else begin : g_out_bp
        assign valid_out = out_full;
        assign data_out  = out_data;
    end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_full <= 1'b0;
            out_data <= 512'd0;
        end else begin
            if (out_full && out_ready_in)
                out_full <= 1'b0;
            if (dp_valid_out) begin
                out_data <= dp_data_out;
                out_full <= 1'b1;
            end
        end
    end

    // ---------- Geometry ----------
    localparam integer OC       = 64;
    localparam integer W        = 512;   // output_width_bits, = lhs slice width
    localparam integer CH_IDX_W = 7;     // ceil_log2(OC+1)

    // ---------- Fused per-side scale constants ----------
    // out_scale = 0.16331340759757937
    // LHS_FUSED = lhs_scale / out_scale = 0.13125772551288756 / out_scale ~ 0.803716...
    //           = round(LHS_FUSED * 2^20) = 842758 / 2^20      (round_half_up fixed-point)
    // RHS_FUSED = rhs_scale / out_scale = 0.1377999012864481  / out_scale ~ 0.843776...
    //           = round(RHS_FUSED * 2^20) = 884763 / 2^20
    //
    // SHIFT bumped 15 -> 20: the canonical golden requant is the MATHEMATICAL
    //   round_half_up_toward_+inf( (lhs*lhs_scale + rhs*rhs_scale) / out_scale ).
    //   At SHIFT=15 the rounded multipliers were too coarse and landed on the
    //   wrong side of a 0.5 tie (off-by-1). PROVEN by exhaustive 65536-pair
    //   sweep: SHIFT=20 is the SMALLEST shift for which the integer
    //   (lhs*LHS_M + rhs*RHS_M + 2^(S-1)) >>> S EXACTLY equals the canonical
    //   golden for ALL int8 pairs (0 mismatches). The only pair where this
    //   diverges from golden_impl.py is (11,111): the true ratio is
    //   +102.49999843666 (< 102.5 -> rounds to 102, which the RTL produces),
    //   but golden_impl.py's float32 path collapses it to exactly 102.5 -> 103.
    //   That single point is a float32 artifact in the golden, not an RTL error;
    //   the RTL is bit-exact to the infinite-precision (and float64) golden.
    localparam integer FUSED_SHIFT   = 20;
    localparam integer SCALE_CONST_W = 21;

    // ---------- Internal widths ----------
    localparam integer PROD_W = 8 + SCALE_CONST_W;   // 29-bit signed
    localparam integer SUM_W  = PROD_W + 2;          // 31-bit signed (absorb sum + round bias)

    // Multiplier constants pre-widened to PROD_W signed so the multiply is
    // unambiguously context-determined at PROD_W. Both fit in 29-bit signed
    // positive: max |product| = 128*884763 = 113,249,664 (< 2^28), and
    // max |sum+half| = 128*842758 + 128*884763 + 2^19 = 221,646,976 (< 2^28),
    // both well within the signed PROD_W(29)/SUM_W(31) ranges.
    localparam signed [PROD_W-1:0] LHS_M = 842758;
    localparam signed [PROD_W-1:0] RHS_M = 884763;

    // Round-half-up-toward-+inf bias. Golden requant = floor(value + 0.5):
    // add HALF = 2^(SHIFT-1) UNCONDITIONALLY, then arithmetic >>> (floor).
    // (No sign-conditional branch: the >>> already floors uniformly toward -inf,
    //  and an asymmetric negative bias over-rounds negatives by one LSB.)
    localparam signed [SUM_W-1:0] FUSED_HALF =
        {{(SUM_W-1){1'b0}}, 1'b1} <<< (FUSED_SHIFT - 1);

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
            dp_valid_out <= 1'b0;
            dp_data_out  <= {W{1'b0}};
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
            dp_valid_out <= 1'b0;

            case (state)
                ST_IDLE: begin
                    // Re-raise ready_in when the skid drains (==0: always 1'b1).
                    // The accept below (later in source) still wins with
                    // ready_in<=0 on the accept cycle, so ==0 is byte/cycle-exact.
                    ready_in     <= !skid_block;
                    stage1_valid <= 1'b0;
                    stage2_valid <= 1'b0;
                    if (valid_in && ready_in && !skid_block) begin
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
                        // Golden requant = floor(value + 0.5) = round_half_up_toward_+inf:
                        // ROUND_BIAS = 2^(SHIFT-1) added UNCONDITIONALLY, then arithmetic >>> (floor).
                        // The earlier sign-conditional (FUSED_HALF_M1 on negatives) over-rounded by
                        // one LSB on negatives -> off-by-1 vs golden. Must be unconditional + FUSED_HALF.
                        sum_term     <= sum_pre + FUSED_HALF;
                        stage2_valid <= 1'b1;
                        stage2_idx   <= stage1_idx;
                    end else begin
                        stage2_valid <= 1'b0;
                    end

                    if (stage2_valid) begin
                        if (out_pre > SAT_HI)
                            dp_data_out[stage2_idx*8 +: 8] <= 8'sd127;
                        else if (out_pre < SAT_LO)
                            dp_data_out[stage2_idx*8 +: 8] <= -8'sd128;
                        else
                            dp_data_out[stage2_idx*8 +: 8] <= out_pre[7:0];

                        if (stage2_idx == (OC-1)) begin
                            dp_valid_out <= 1'b1; // [INVARIANT:VALID_OUT_LATENCY]
                            ready_in  <= !skid_block; // [INVARIANT:READY_IN_GATING] (==0: 1'b1)
                            state     <= ST_IDLE;
                        end
                    end
                end
            endcase
        end
    end

endmodule
