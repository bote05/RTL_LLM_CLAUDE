`default_nettype none

// PARAM-GATED ELASTIC BACKPRESSURE (ENABLE_BACKPRESSURE, default 0):
//   * ==0 (default): bit/cycle-IDENTICAL to the legacy add. out_ready_in is
//     IGNORED; skid_block is constant 0; valid_out/data_out come straight from
//     the datapath (dp_valid_out/dp_data_out). The add arithmetic is UNCHANGED.
//   * ==1: 1-deep output skid holds the 1-cycle result beat until out_ready_in;
//     ready_in drops while a beat is parked, freezing the NEXT frame's accept so
//     the single-beat-per-frame producer can never overrun the skid.
module node_add_828 #(
    parameter ENABLE_BACKPRESSURE = 0
)(
    input  wire           clk,
    input  wire           rst_n,
    input  wire           valid_in,
    output reg            ready_in,
    input  wire [1535:0]  data_in,
    input  wire           out_ready_in,   // NEW: downstream-ready (ignored when ENABLE_BACKPRESSURE==0)
    output wire           valid_out,
    output wire [767:0]   data_out
);

    // ---- datapath output regs + 1-deep output skid (see header) ----
    reg                 dp_valid_out;
    reg  [767:0]        dp_data_out;
    reg                 out_full;
    reg  [767:0]        out_data;
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

    localparam integer OC            = 96;
    localparam integer INPUT_WIDTH   = 1536;
    localparam integer OUTPUT_WIDTH  = 768;
    localparam integer RHS_BASE      = OUTPUT_WIDTH;

    // [BITEXACT-FIX 2026-05-31] SHIFT 15->19, mults retuned, round-half-toward-+inf
    // made UNCONDITIONAL (was banker's ties-to-even via is_tie&floor_q[0] -> 387
    // off-by-1 mismatches vs golden). Golden Int8Add = round_half_up_toward_pos_inf
    // = floor(sum/out_scale + 0.5). Exact-match constants found by sweeping shift to
    // bit-equality over the full INT8xINT8 grid (apply_add_rescale.py method):
    //   lhs_scale/out_scale = 0.08310311/0.10058279 -> 433175/2^19
    //   rhs_scale/out_scale = 0.06375816/0.10058279 -> 332340/2^19
    localparam integer FUSED_SHIFT      = 19;
    localparam integer SCALE_CONST_W    = 24;
    localparam signed [SCALE_CONST_W-1:0] LHS_FUSED_MULT = 24'sd433175;
    localparam signed [SCALE_CONST_W-1:0] RHS_FUSED_MULT = 24'sd332340;

    localparam integer PROD_W           = 8 + SCALE_CONST_W;
    localparam integer SUM_W            = PROD_W + 2;

    // Unconditional round-half-toward-+inf bias = 2^(FUSED_SHIFT-1) = 2^18 = 262144.
    localparam signed [SUM_W-1:0] FUSED_HALF = 34'sd262144;

    localparam signed [SUM_W-1:0] SAT_HI =  34'sd127;
    localparam signed [SUM_W-1:0] SAT_LO = -34'sd128;

    localparam integer CH_IDX_W = 7;

    localparam ST_IDLE = 1'b0,
               ST_RUN  = 1'b1;
    reg                    state;

    reg [INPUT_WIDTH-1:0]  input_buf;
    reg [CH_IDX_W:0]       ch_idx;
    reg [CH_IDX_W-1:0]     stage1_idx;
    reg [CH_IDX_W-1:0]     stage2_idx;
    reg                    stage1_valid;
    reg                    stage2_valid;

    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] lhs_term;
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] rhs_term;
    reg signed [SUM_W-1:0] sum_pre_r;

    wire signed [7:0]       cur_lhs;
    wire signed [7:0]       cur_rhs;
    wire signed [SUM_W-1:0] sum_pre;

    assign cur_lhs = $signed(input_buf[ch_idx*8 +: 8]);
    assign cur_rhs = $signed(input_buf[RHS_BASE + ch_idx*8 +: 8]);
    assign sum_pre = lhs_term + rhs_term;

    // Round-half-toward-+infinity requantisation, matching the golden
    // Int8Add (round_half_up_toward_pos_inf = floor(x + 0.5)): add the
    // half-LSB bias UNCONDITIONALLY, then arithmetic-shift (floor), then
    // saturate. The previous banker's (ties-to-even) form disagreed with the
    // golden on every exact half-LSB tie -> off-by-1. [INVARIANT:ROUNDING]
    wire signed [SUM_W-1:0] rounded;
    wire signed [7:0]       sat_byte;

    assign rounded       = (sum_pre_r + FUSED_HALF) >>> FUSED_SHIFT;
    assign sat_byte      = (rounded > SAT_HI) ?  8'sd127 :
                           (rounded < SAT_LO) ? -8'sd128 :
                                                rounded[7:0];

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
        if (state == ST_RUN && stage2_valid) begin
                dp_data_out[stage2_idx*8 +: 8] <= sat_byte;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= ST_IDLE;
            ready_in     <= 1'b1;
            dp_valid_out <= 1'b0;
            ch_idx       <= {(CH_IDX_W+1){1'b0}};
            stage1_idx   <= {CH_IDX_W{1'b0}};
            stage2_idx   <= {CH_IDX_W{1'b0}};
            stage1_valid <= 1'b0;
            stage2_valid <= 1'b0;
            lhs_term     <= {PROD_W{1'b0}};
            rhs_term     <= {PROD_W{1'b0}};
            sum_pre_r    <= {SUM_W{1'b0}};
        end else begin
            dp_valid_out <= 1'b0;
            // Re-raise ready_in when the skid drains (==0: always 1'b1). The
            // ST_IDLE accept below (later in source) still wins with ready_in<=0
            // on the accept cycle, so ENABLE_BACKPRESSURE==0 is byte/cycle-exact.
            if (state == ST_IDLE) ready_in <= !skid_block;

            if (state == ST_RUN && ch_idx < OC) begin
                lhs_term     <= cur_lhs * LHS_FUSED_MULT;
                rhs_term     <= cur_rhs * RHS_FUSED_MULT;
                stage1_idx   <= ch_idx[CH_IDX_W-1:0];
                stage1_valid <= 1'b1;
                ch_idx       <= ch_idx + 1'b1;
            end else begin
                stage1_valid <= 1'b0;
            end

            if (stage1_valid) begin
                sum_pre_r    <= sum_pre;
                stage2_idx   <= stage1_idx;
                stage2_valid <= 1'b1;
            end else begin
                stage2_valid <= 1'b0;
            end

            if (stage2_valid) begin
            end

            case (state)
                ST_IDLE: begin
                    if (valid_in && !skid_block) begin
                        ch_idx    <= {(CH_IDX_W+1){1'b0}};
                        state     <= ST_RUN;
                        ready_in  <= 1'b0;
                    end
                end
                ST_RUN: begin
                    if (stage2_valid && stage2_idx == (OC - 1)) begin
                        dp_valid_out <= 1'b1;
                        ready_in  <= !skid_block;
                        state     <= ST_IDLE;
                    end
                end
                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
