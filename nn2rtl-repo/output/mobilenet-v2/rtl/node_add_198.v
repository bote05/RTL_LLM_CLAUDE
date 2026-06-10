// Module: node_add_198
// op_type=add, OC=24, packed [rhs|lhs] (lhs low 192b, rhs high 192b)
// pipeline_latency_cycles = OC + 3 = 27
// Fused scales: FUSED_SHIFT=20 (smallest S byte-exact over all 65536 int8 pairs)
//   LHS_FUSED_MULT/2^20 ~= lhs_scale/out_scale  (1083546/1048576 ~= 1.033348)
//   RHS_FUSED_MULT/2^20 ~= rhs_scale/out_scale  (992967/1048576  ~= 0.946967)
//   lhs_scale=0.4906998581773653 rhs_scale=0.44967984777735914 out_scale=0.47486300731268455
// PARAM-GATED ELASTIC BACKPRESSURE (ENABLE_BACKPRESSURE, default 0):
//   * ==0 (default): bit/cycle-IDENTICAL to the legacy add. out_ready_in is
//     IGNORED; skid_block is constant 0; valid_out/data_out come straight from
//     the datapath (dp_valid_out/dp_data_out). The add arithmetic is UNCHANGED.
//   * ==1: 1-deep output skid holds the 1-cycle result beat until out_ready_in;
//     ready_in drops while a beat is parked, freezing the NEXT frame's accept so
//     the single-beat-per-frame producer can never overrun the skid.
module node_add_198 #(
    parameter ENABLE_BACKPRESSURE = 0
)(
    input  wire                clk,
    input  wire                rst_n,
    input  wire                valid_in,
    output reg                 ready_in,
    input  wire [383:0]        data_in,
    input  wire                out_ready_in,   // NEW: downstream-ready (ignored when ENABLE_BACKPRESSURE==0)
    output wire                valid_out,
    output wire [191:0]        data_out
);

    // ---- datapath output regs + 1-deep output skid (see header) ----
    reg                 dp_valid_out;
    reg  [191:0]        dp_data_out;
    reg                 out_full;
    reg  [191:0]        out_data;
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

    localparam integer OC               = 24;
    localparam integer INPUT_WIDTH      = 384;
    localparam integer OUTPUT_WIDTH     = 192;
    localparam integer CH_IDX_W         = 5;

    localparam integer FUSED_SHIFT      = 20;
    localparam integer SCALE_CONST_W    = 24;
    localparam signed [SCALE_CONST_W-1:0] LHS_FUSED_MULT = 24'sd1083546;
    localparam signed [SCALE_CONST_W-1:0] RHS_FUSED_MULT = 24'sd992967;

    localparam integer PROD_W           = 8 + SCALE_CONST_W;   // 32
    localparam integer SUM_W            = PROD_W + 2;          // 34

    localparam signed [SUM_W-1:0] FUSED_HALF  =
        {{(SUM_W-FUSED_SHIFT){1'b0}}, 1'b1, {(FUSED_SHIFT-1){1'b0}}};

    localparam ST_IDLE = 1'b0;
    localparam ST_RUN  = 1'b1;

    reg                       state;
    reg [INPUT_WIDTH-1:0]     input_buf;
    reg [CH_IDX_W-1:0]        ch_idx;
    reg [CH_IDX_W-1:0]        stage1_idx;
    reg [CH_IDX_W-1:0]        stage2_idx;
    reg                       stage1_valid;
    reg                       stage2_valid;

    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] lhs_term;
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] rhs_term;
    reg signed [SUM_W-1:0]    sum_term;

    wire signed [7:0]         lhs_ch;
    wire signed [7:0]         rhs_ch;
    wire signed [SUM_W-1:0]   sum_pre;
    wire signed [SUM_W-1:0]   shifted;
    wire signed [7:0]         sat_out;

    assign lhs_ch  = $signed(input_buf[ch_idx*8 +: 8]);
    assign rhs_ch  = $signed(input_buf[OUTPUT_WIDTH + ch_idx*8 +: 8]);
    assign sum_pre = lhs_term + rhs_term;
    assign shifted = sum_term >>> FUSED_SHIFT;
    assign sat_out = (shifted >  127) ?  8'sd127 :
                     (shifted < -128) ? -8'sd128 :
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
        if (state == ST_RUN && stage2_valid) begin
                        // [INVARIANT:ROUNDING]
                        dp_data_out[stage2_idx*8 +: 8] <= sat_out;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= ST_IDLE;
            ready_in      <= 1'b1;                       // [INVARIANT:READY_IN_GATING]
            dp_valid_out  <= 1'b0;
            ch_idx        <= {CH_IDX_W{1'b0}};
            stage1_idx    <= {CH_IDX_W{1'b0}};
            stage2_idx    <= {CH_IDX_W{1'b0}};
            stage1_valid  <= 1'b0;
            stage2_valid  <= 1'b0;
            lhs_term      <= {PROD_W{1'b0}};
            rhs_term      <= {PROD_W{1'b0}};
            sum_term      <= {SUM_W{1'b0}};
        end else begin
            dp_valid_out <= 1'b0;
            case (state)
                ST_IDLE: begin
                    ready_in     <= !skid_block;         // [INVARIANT:READY_IN_GATING] (==0: 1'b1)
                    stage1_valid <= 1'b0;
                    stage2_valid <= 1'b0;
                    if (valid_in && !skid_block) begin
                        ch_idx    <= {CH_IDX_W{1'b0}};
                        state     <= ST_RUN;
                        ready_in  <= 1'b0;               // [INVARIANT:READY_IN_GATING]
                    end
                end
                ST_RUN: begin
                    // Stage 1 — fused multiplies (one channel per cycle).
                    if (ch_idx < OC) begin
                        lhs_term     <= lhs_ch * LHS_FUSED_MULT;
                        rhs_term     <= rhs_ch * RHS_FUSED_MULT;
                        stage1_idx   <= ch_idx;
                        stage1_valid <= 1'b1;
                        ch_idx       <= ch_idx + 1'b1;
                    end else begin
                        stage1_valid <= 1'b0;
                    end

                    // Stage 2 — sum + UNCONDITIONAL round-half-toward-+inf bias.
                    // Golden requant = floor(value + 0.5): add FUSED_HALF = 2^(SHIFT-1)
                    // ALWAYS (no sign-dependent FUSED_HALF_M1), then arithmetic >>> floors.
                    stage2_valid <= stage1_valid;
                    stage2_idx   <= stage1_idx;
                    if (stage1_valid) begin
                        sum_term <= sum_pre + FUSED_HALF;
                    end

                    // Stage 3 — arithmetic shift, saturate, pack into data_out.
                    if (stage2_valid) begin
                        if (stage2_idx == OC - 1) begin
                            dp_valid_out <= 1'b1;        // [INVARIANT:VALID_OUT_LATENCY]
                            state     <= ST_IDLE;
                            ready_in  <= !skid_block;    // [INVARIANT:READY_IN_GATING] (==0: 1'b1)
                        end
                    end
                end
                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
