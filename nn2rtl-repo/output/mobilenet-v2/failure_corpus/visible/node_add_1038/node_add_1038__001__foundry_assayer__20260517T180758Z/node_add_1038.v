// node_add_1038 — INT8 residual add, flat-bus contract.
// OC = 160, IH = IW = 7, data_in = 2 * OUTPUT_WIDTH packed as [rhs|lhs].
// Channel-serialized 3-stage arithmetic pipeline:
//   stage 1  : per-channel multiplies (DSP48)
//   stage 2  : sum + sign-aware rounding bias
//   stage 3  : arithmetic right-shift, INT8 saturate, write channel slice
//
// Latency contract: pipeline_latency_cycles == OC + 3 == 163 cycles from
// the first valid_in to the first valid_out. Implemented by inserting a
// single S_CAPTURE state between S_IDLE and S_RUN so that stage1 first
// fires two cycles after valid_in is accepted, and stage3 first writes
// data_out at cycle 4 of the run, with channel OC-1 completing at
// cycle (4 + OC - 1) == OC + 3.
//
// Quantization: both operands carry their own INT8 scale; the output
// is `saturate(((lhs * lhs_scale + rhs * rhs_scale) / out_scale))`.
// We fold (operand_scale / out_scale) into a fused multiplier per side
// so a single shift performs the final requantize:
//   lhs_ratio = lhs_scale_factor / scale_factor ~= 0.695182
//   rhs_ratio = rhs_scale_factor / scale_factor ~= 0.897385
// Both are approximated at FUSED_SHIFT = 15 (2^15 = 32768) to keep the
// multipliers within signed 16-bit range:
//   LHS_FUSED_MULT = round(0.695182 * 32768) = 22780
//   RHS_FUSED_MULT = round(0.897385 * 32768) = 29406

module node_add_1038 (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                valid_in,
    output reg                 ready_in,
    input  wire [2559:0]       data_in,
    output reg                 valid_out,
    output reg  [1279:0]       data_out
);

    localparam integer OC           = 160;
    localparam integer OUTPUT_WIDTH = 1280;
    localparam integer INPUT_WIDTH  = 2560;
    localparam integer CH_IDX_W     = 8;

    localparam integer FUSED_SHIFT  = 15;
    localparam integer MULT_W       = 16;
    localparam integer PROD_W       = 24;
    localparam integer SUM_W        = 26;

    localparam signed [MULT_W-1:0] LHS_FUSED_MULT = 16'sd22780;
    localparam signed [MULT_W-1:0] RHS_FUSED_MULT = 16'sd29406;

    localparam signed [SUM_W-1:0] FUSED_HALF    = 26'sd16384;
    localparam signed [SUM_W-1:0] FUSED_HALF_M1 = 26'sd16383;

    localparam signed [SUM_W-1:0] SAT_HI =  26'sd127;
    localparam signed [SUM_W-1:0] SAT_LO = -26'sd128;

    localparam [CH_IDX_W-1:0] OC_LIMIT   = 8'd160;
    localparam [CH_IDX_W-1:0] OC_MINUS_1 = 8'd159;

    localparam [1:0] S_IDLE    = 2'd0;
    localparam [1:0] S_CAPTURE = 2'd1;
    localparam [1:0] S_RUN     = 2'd2;

    reg [1:0]              state;
    reg [INPUT_WIDTH-1:0]  input_buf;
    reg [CH_IDX_W-1:0]     ch_idx;
    reg [CH_IDX_W-1:0]     stage1_idx;
    reg [CH_IDX_W-1:0]     stage2_idx;
    reg [CH_IDX_W-1:0]     stage3_idx;
    reg                    stage1_valid;
    reg                    stage2_valid;
    reg                    stage3_valid;

    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] lhs_term;
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] rhs_term;
    reg signed [SUM_W-1:0] sum_term;

    wire signed [SUM_W-1:0] sum_pre;
    assign sum_pre = lhs_term + rhs_term;

    wire signed [SUM_W-1:0] shifted_val;
    assign shifted_val = sum_term >>> FUSED_SHIFT;

    wire signed [7:0] sat_val;
    assign sat_val = (shifted_val > SAT_HI) ?  8'sd127 :
                     (shifted_val < SAT_LO) ? -8'sd128 :
                                              shifted_val[7:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            ready_in     <= 1'b1;
            valid_out    <= 1'b0;
            data_out     <= {OUTPUT_WIDTH{1'b0}};
            input_buf    <= {INPUT_WIDTH{1'b0}};
            ch_idx       <= {CH_IDX_W{1'b0}};
            stage1_idx   <= {CH_IDX_W{1'b0}};
            stage2_idx   <= {CH_IDX_W{1'b0}};
            stage3_idx   <= {CH_IDX_W{1'b0}};
            stage1_valid <= 1'b0;
            stage2_valid <= 1'b0;
            stage3_valid <= 1'b0;
            lhs_term     <= {PROD_W{1'b0}};
            rhs_term     <= {PROD_W{1'b0}};
            sum_term     <= {SUM_W{1'b0}};
        end else begin
            stage1_valid <= 1'b0;
            stage2_valid <= 1'b0;
            stage3_valid <= 1'b0;
            valid_out    <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (valid_in && ready_in) begin
                        input_buf <= data_in;
                        ch_idx    <= {CH_IDX_W{1'b0}};
                        ready_in  <= 1'b0;
                        state     <= S_CAPTURE;
                    end
                end
                S_CAPTURE: begin
                    state <= S_RUN;
                end
                S_RUN: begin
                    if (ch_idx < OC_LIMIT) begin
                        lhs_term     <= $signed(input_buf[ch_idx*8 +: 8]) * LHS_FUSED_MULT;
                        rhs_term     <= $signed(input_buf[OUTPUT_WIDTH + ch_idx*8 +: 8]) * RHS_FUSED_MULT;
                        stage1_idx   <= ch_idx;
                        stage1_valid <= 1'b1;
                        ch_idx       <= ch_idx + 1'b1;
                    end
                    if (stage1_valid) begin
                        sum_term     <= sum_pre + (sum_pre[SUM_W-1] ? FUSED_HALF_M1 : FUSED_HALF);
                        stage2_idx   <= stage1_idx;
                        stage2_valid <= 1'b1;
                    end
                    if (stage2_valid) begin
                        data_out[stage2_idx*8 +: 8] <= sat_val;
                        stage3_idx   <= stage2_idx;
                        stage3_valid <= 1'b1;
                        if (stage2_idx == OC_MINUS_1) begin
                            valid_out <= 1'b1;
                            ready_in  <= 1'b1;
                            state     <= S_IDLE;
                        end
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
