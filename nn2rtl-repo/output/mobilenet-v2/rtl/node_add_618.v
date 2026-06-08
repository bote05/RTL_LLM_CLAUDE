// Surgeon: node_add_618 - INT8 quantized residual add, OC=64, flat-bus.
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
module node_add_618 #(
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

    localparam integer OC       = 64;
    localparam integer W        = 512;
    localparam integer CH_IDX_W = 7;

    localparam integer FUSED_SHIFT   = 22;
    localparam integer SCALE_CONST_W = 23;

    localparam integer PROD_W = 8 + SCALE_CONST_W;
    localparam integer SUM_W  = PROD_W + 2;

    localparam signed [PROD_W-1:0] LHS_M = 3476763;
    localparam signed [PROD_W-1:0] RHS_M = 2381718;

    localparam signed [SUM_W-1:0] FUSED_HALF =
        {{(SUM_W-1){1'b0}}, 1'b1} <<< (FUSED_SHIFT - 1);
    localparam signed [SUM_W-1:0] FUSED_HALF_M1 =
        FUSED_HALF - {{(SUM_W-1){1'b0}}, 1'b1};

    localparam signed [SUM_W-1:0] SAT_HI =  127;
    localparam signed [SUM_W-1:0] SAT_LO = -128;

    localparam ST_IDLE = 1'b0;
    localparam ST_RUN  = 1'b1;
    reg state;

    reg  [1023:0]            input_buf;
    reg  [CH_IDX_W-1:0]      ch_idx;
    reg  [CH_IDX_W-1:0]      stage1_idx;
    reg  [CH_IDX_W-1:0]      stage2_idx;
    reg                      stage1_valid;
    reg                      stage2_valid;

    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] lhs_term;
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] rhs_term;
    reg signed [SUM_W-1:0]   sum_term;

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
                        ready_in  <= 1'b0;
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
                        sum_term     <= sum_pre + (sum_pre[SUM_W-1] ? FUSED_HALF_M1 : FUSED_HALF);
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
                            dp_valid_out <= 1'b1;
                            ready_in  <= !skid_block;
                            state     <= ST_IDLE;
                        end
                    end
                end
            endcase
        end
    end

endmodule
