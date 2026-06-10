`default_nettype none

// PARAM-GATED ELASTIC BACKPRESSURE (ENABLE_BACKPRESSURE, default 0):
//   * ==0 (default): bit/cycle-IDENTICAL to the legacy add. out_ready_in is
//     IGNORED; skid_block is constant 0; valid_out/data_out come straight from
//     the datapath (dp_valid_out/dp_data_out). The add arithmetic is UNCHANGED.
//   * ==1: 1-deep output skid holds the 1-cycle result beat until out_ready_in;
//     ready_in drops while a beat is parked, freezing the NEXT frame's accept so
//     the single-beat-per-frame producer can never overrun the skid.
module node_add_408 #(
    parameter ENABLE_BACKPRESSURE = 0
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 valid_in,
    output reg                  ready_in,
    input  wire [511:0]         data_in,
    input  wire                 out_ready_in,   // NEW: downstream-ready (ignored when ENABLE_BACKPRESSURE==0)
    output wire                 valid_out,
    output wire [255:0]         data_out
);

    // ---- datapath output regs + 1-deep output skid (see header) ----
    reg                 dp_valid_out;
    reg  [255:0]        dp_data_out;
    reg                 out_full;
    reg  [255:0]        out_data;
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

    localparam integer OC            = 32;
    localparam integer W_OUT         = 256;
    localparam integer CH_IDX_W      = 6;

    // Fused multipliers encode (input_scale / output_scale) so
    // out = (lhs*lhs_s + rhs*rhs_s) / out_s lands back in INT8 range.
    // lhs_scale/out_scale = 0.28508/0.27289 = 1.04467 -> 17116/2^14
    // rhs_scale/out_scale = 0.21698/0.27289 = 0.79511 -> 13027/2^14
    localparam integer SCALE_CONST_W = 24;
    localparam integer FUSED_SHIFT   = 22;
    localparam signed [SCALE_CONST_W-1:0] LHS_FUSED_MULT = 24'sd2816215;
    localparam signed [SCALE_CONST_W-1:0] RHS_FUSED_MULT = 24'sd4424446;

    localparam integer PROD_W = 8 + SCALE_CONST_W;
    localparam integer SUM_W  = PROD_W + 2;

    // [BITEXACT-FIX 2026-05-31] round bias = 2^(FUSED_SHIFT-1) = 2^21 (was 2^13=8192 -> truncation
    // => 49.89% off-by-1). Matches golden round_half_up_toward_pos_inf = floor(x+0.5), UNCONDITIONAL.
    localparam signed [SUM_W-1:0] FUSED_HALF    = 26'sd2097152;
    localparam signed [SUM_W-1:0] FUSED_HALF_M1 = 26'sd2097151;
    localparam signed [SUM_W-1:0] SAT_HI        =  26'sd127;
    localparam signed [SUM_W-1:0] SAT_LO        = -26'sd128;

    localparam ST_IDLE = 1'b0;
    localparam ST_RUN  = 1'b1;
    reg                       state;

    reg [511:0]               input_buf;
    reg [CH_IDX_W-1:0]        ch_idx;
    reg [CH_IDX_W-1:0]        stage1_idx;
    reg [CH_IDX_W-1:0]        stage2_idx;
    reg                       stage1_valid;
    reg                       stage2_valid;

    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] lhs_term;
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] rhs_term;
    reg signed [SUM_W-1:0]    sum_term;

    wire signed [7:0]         cur_lhs;
    wire signed [7:0]         cur_rhs;
    wire signed [SUM_W-1:0]   sum_pre;
    wire signed [SUM_W-1:0]   shifted;
    wire signed [7:0]         sat_byte;

    assign cur_lhs  = $signed(input_buf[ch_idx*8 +: 8]);
    assign cur_rhs  = $signed(input_buf[W_OUT + ch_idx*8 +: 8]);
    assign sum_pre  = lhs_term + rhs_term;
    assign shifted  = sum_term >>> FUSED_SHIFT;
    assign sat_byte = (shifted > SAT_HI) ?  8'sd127 :
                      (shifted < SAT_LO) ? -8'sd128 :
                       shifted[7:0];

    // [K1-MBV2] Block A: array/data writes (sync-only) -- node_add_1
    // precedent (ResNet K1 P9/P10 analog). input_buf is fully rewritten on
    // the accept edge before the RUN pipe reads it; every consumed
    // dp_data_out byte is written by the 3-stage pipe (stage2_valid covers
    // ch 0..OC-1) before dp_valid_out pulses; both guards replicate the
    // original conditions on reset-kept control. lhs/rhs/sum MAC pipes and
    // all stage*_valid/idx control KEEP their async reset.
    always @(posedge clk) begin
        if (state == ST_IDLE && valid_in && !skid_block) begin
            input_buf <= data_in;
        end
        if (stage2_valid) begin
                dp_data_out[stage2_idx*8 +: 8] <= sat_byte;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= ST_IDLE;
            ready_in     <= 1'b1;
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
            // Re-raise ready_in when the skid drains (==0: always 1'b1). The
            // accept block below (later in source) still wins with ready_in<=0
            // on the accept cycle, so ENABLE_BACKPRESSURE==0 is byte/cycle-exact.
            if (state == ST_IDLE) ready_in <= !skid_block;

            if (state == ST_RUN && ch_idx < 6'd32) begin
                lhs_term     <= cur_lhs * LHS_FUSED_MULT;
                rhs_term     <= cur_rhs * RHS_FUSED_MULT;
                stage1_idx   <= ch_idx;
                stage1_valid <= 1'b1;
                ch_idx       <= ch_idx + 6'd1;
            end else begin
                stage1_valid <= 1'b0;
            end

            if (stage1_valid) begin
                sum_term     <= sum_pre + FUSED_HALF; // [INVARIANT:ROUNDING] unconditional +2^(SHIFT-1) = golden round_half_up_toward_pos_inf
                stage2_idx   <= stage1_idx;
                stage2_valid <= 1'b1;
            end else begin
                stage2_valid <= 1'b0;
            end

            if (stage2_valid) begin
                if (stage2_idx == 6'd31) begin
                    dp_valid_out <= 1'b1;          // [INVARIANT:VALID_OUT_LATENCY]
                    state     <= ST_IDLE;
                    ready_in  <= !skid_block;      // [INVARIANT:READY_IN_GATING] (==0: 1'b1)
                end
            end

            if (state == ST_IDLE && valid_in && !skid_block) begin
                state        <= ST_RUN;
                ready_in     <= 1'b0;              // [INVARIANT:READY_IN_GATING]
                ch_idx       <= {CH_IDX_W{1'b0}};
                stage1_valid <= 1'b0;
                stage2_valid <= 1'b0;
            end
        end
    end

endmodule

`default_nettype wire
