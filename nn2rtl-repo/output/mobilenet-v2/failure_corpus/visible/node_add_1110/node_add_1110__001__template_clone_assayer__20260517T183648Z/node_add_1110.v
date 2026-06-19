// node_add_1038 — INT8 residual add, flat-bus contract.
// OC = 160, IH = IW = 7, data_in = 2 * OUTPUT_WIDTH packed as [rhs|lhs].
// Channel-serialized 3-stage arithmetic pipeline:
//   stage 1  : per-channel multiplies (DSP48)
//   stage 2  : sum + sign-aware rounding bias
//   stage 3  : arithmetic right-shift, INT8 saturate, write channel slice
//
// Latency contract: pipeline_latency_cycles == OC + 3 == 163 cycles from
// the first valid_in to the first valid_out.
//   - cycle T   : S_IDLE, valid_in accepted, input_buf <= data_in, state <= S_RUN
//   - cycle T+1 : S_RUN ch=0, stage1 (lhs_term/rhs_term register for ch0)
//   - cycle T+2 : S_RUN ch=1, stage1 ch1, stage2 ch0 (sum_term registers)
//   - cycle T+3 : S_RUN ch=2, stage1 ch2, stage2 ch1, stage3 writes ch0
//   - ...
//   - cycle T+3+(OC-1) = T+162 : stage3 writes ch159, valid_out <= 1
//   - cycle T+163 : valid_out visible to consumer  --> latency = 163.

module node_add_1110 (
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

    localparam integer FUSED_SHIFT  = 23;
    localparam integer MULT_W       = 16;
    localparam integer PROD_W       = 24;
    localparam integer SUM_W        = 26;

    localparam signed [MULT_W-1:0] LHS_FUSED_MULT = 24'sd5544062;
    localparam signed [MULT_W-1:0] RHS_FUSED_MULT = 24'sd5069535;

    localparam signed [SUM_W-1:0] FUSED_HALF    = 26'sd16384;
    localparam signed [SUM_W-1:0] FUSED_HALF_M1 = 26'sd16383;

    localparam signed [SUM_W-1:0] SAT_HI =  26'sd127;
    localparam signed [SUM_W-1:0] SAT_LO = -26'sd128;

    localparam [CH_IDX_W-1:0] OC_LIMIT   = 8'd160;
    localparam [CH_IDX_W-1:0] OC_MINUS_1 = 8'd159;

    localparam S_IDLE = 1'b0;
    localparam S_RUN  = 1'b1;

    reg                    state;
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
                        state     <= S_RUN;
                    end
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
