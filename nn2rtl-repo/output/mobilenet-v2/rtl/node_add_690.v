// Foundry: node_add_690 - INT8 quantized residual add, OC=64, flat-bus.
// Channel-serialized 3-stage requantize pipeline per 05_add_quantized.md.
//   data_in[511:0]    = lhs (64 INT8 channels)
//   data_in[1023:512] = rhs (64 INT8 channels)
//   out_i = saturate( ( lhs_i * (lhs_scale/out_scale)
//                      + rhs_i * (rhs_scale/out_scale) )  >>> SHIFT )
// pipeline_latency_cycles = OC + 3 = 67.
//
// Per-layer fused scale constants (this layer's LayerIR):
//   lhs_scale = 0.19701833049143394
//   rhs_scale = 0.22187153373177596
//   out_scale = 0.19505784642977977
//   LHS_FUSED = lhs_scale / out_scale ~ 1.01005149
//   RHS_FUSED = rhs_scale / out_scale ~ 1.13746609
//   At FUSED_SHIFT = 22 (2^22 = 4194304):
//     LHS_M = round(1.01005149 * 2^22) = 4236463  (rel err < 1e-7)
//     RHS_M = round(1.13746609 * 2^22) = 4770879  (rel err < 1e-7)
//   Both fit in 24-bit signed positive (< 2^23-1 = 8388607).
// PARAM-GATED ELASTIC BACKPRESSURE (ENABLE_BACKPRESSURE, default 0):
//   * ==0 (default): bit/cycle-IDENTICAL to the legacy add. out_ready_in is
//     IGNORED; skid_block is constant 0; valid_out/data_out come straight from
//     the datapath (dp_valid_out/dp_data_out). The add arithmetic is UNCHANGED.
//   * ==1: 1-deep output skid holds the 1-cycle result beat until out_ready_in;
//     ready_in drops while a beat is parked, freezing the NEXT frame's accept so
//     the single-beat-per-frame producer can never overrun the skid.
module node_add_690 #(
    parameter ENABLE_BACKPRESSURE = 0
)(
    input  wire           clk,
    input  wire           rst_n,
    input  wire           valid_in,
    output wire            ready_in,
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
        end else begin
            if (out_full && out_ready_in)
                out_full <= 1'b0;
            if (dp_valid_out) begin
                out_full <= 1'b1;
            end
        end
    end
    // [K1-MBV2] out_data is skid DATA: sampled downstream only under
    // out_full (reset-kept); written only under dp_valid_out (reset-kept).
    always @(posedge clk) begin
        if (dp_valid_out) out_data <= dp_data_out;
    end

    // ---------- Geometry ----------
    localparam integer OC       = 64;
    localparam integer W        = 512;   // output_width_bits, = lhs slice width
    localparam integer CH_IDX_W = 7;     // ceil_log2(OC+1)

    // ---------- Fused per-side scale constants ----------
    localparam integer FUSED_SHIFT   = 22;
    localparam integer SCALE_CONST_W = 24;

    // ---------- Internal widths ----------
    localparam integer PROD_W = 8 + SCALE_CONST_W;   // 32-bit signed
    localparam integer SUM_W  = PROD_W + 2;          // 34-bit signed (absorb sum + round bias)

    // Multiplier constants pre-widened to PROD_W signed so the multiply is
    // unambiguously context-determined at PROD_W. Max product magnitude =
    // 128 * 4770879 = 610,672,512, well within 32-bit signed (+/-2.1e9).
    localparam signed [PROD_W-1:0] LHS_M = 4236463;
    localparam signed [PROD_W-1:0] RHS_M = 4770879;

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

    // [K1-MBV2] Block A: array/data writes (sync-only) -- node_add_1
    // precedent (ResNet K1 P9/P10 analog). input_buf is fully rewritten on
    // the accept edge before the RUN pipe reads it; every consumed
    // dp_data_out byte is written by the 3-stage pipe (stage2_valid covers
    // ch 0..OC-1) before dp_valid_out pulses; both guards replicate the
    // original conditions on reset-kept control. lhs/rhs/sum MAC pipes and
    // all stage*_valid/idx control KEEP their async reset.
    always @(posedge clk) begin
        if (state == ST_IDLE && valid_in && ready_in && !skid_block) begin
            input_buf <= data_in;
        end
        if (state == ST_RUN && stage2_valid) begin
                        if (out_pre > SAT_HI)
                            dp_data_out[stage2_idx*8 +: 8] <= 8'sd127;
                        else if (out_pre < SAT_LO)
                            dp_data_out[stage2_idx*8 +: 8] <= -8'sd128;
                        else
                            dp_data_out[stage2_idx*8 +: 8] <= out_pre[7:0];
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= ST_IDLE;
            ready_in_r     <= 1'b1;
            dp_valid_out <= 1'b0;
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
                    // ready_in_r<=0 on the accept cycle, so ==0 is byte/cycle-exact.
                    ready_in_r     <= !skid_block;
                    stage1_valid <= 1'b0;
                    stage2_valid <= 1'b0;
                    if (valid_in && ready_in && !skid_block) begin
                        ready_in_r  <= 1'b0;    // [INVARIANT:READY_IN_GATING]
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

                        if (stage2_idx == (OC-1)) begin
                            dp_valid_out <= 1'b1; // [INVARIANT:VALID_OUT_LATENCY]
                            ready_in_r  <= !skid_block; // [INVARIANT:READY_IN_GATING] (==0: 1'b1)
                            state     <= ST_IDLE;
                        end
                    end
                end
            endcase
        end
    end

    // [ENG_PIPE 2026-06-10][ADD-JOIN FIX] ready_in is the SAME signal the
    // two input skid-FIFOs pop on, so it must be the COMBINATIONAL truth of
    // the accept predicate (the old registered ready_in was 1 cycle stale
    // vs the combinational skid_block -> accept/pop desync = duplicate or
    // stale pair processing when the downstream ready toggled). ready_in_r
    // keeps the legacy register writes (now shadow/dead) so every generated
    // FSM shape is patched uniformly. Cycle-identical when
    // ENABLE_BACKPRESSURE==0 (1 in IDLE, 0 in RUN, same edges).
    reg ready_in_r;
    /* verilator lint_off UNUSED */
    wire _unused_ready_in_r = ready_in_r;
    /* verilator lint_on UNUSED */
    assign ready_in = (state == ST_IDLE) && !skid_block;

endmodule
