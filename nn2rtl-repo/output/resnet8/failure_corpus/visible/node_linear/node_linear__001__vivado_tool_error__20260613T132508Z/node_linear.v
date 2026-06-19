// node_linear — Gemm (K=64, M=10), flat-bus contract.
// Architecture: parallel accumulators across M output features (computed
// concurrently); MP=4 input lanes per cycle => 16 MAC cycles to consume K=64.
// Requantize tail (BIAS -> SCALE -> CLAMP) is a 3-stage scalar pipeline that
// feeds the M outputs sequentially, one per cycle. Outputs collected byte-by-
// byte into the packed 80-bit data_out; valid_out fires on the last lane.
//
// Latency (first valid_in -> first valid_out):
//   1 (latch) + 16 (MAC) + 3 (tail pipe) + (M-1)=9 (serial drain) = 29 cycles.
module node_linear (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          valid_in,
    output reg           ready_in,
    input  wire [511:0]  data_in,
    output reg           valid_out,
    output reg  [79:0]   data_out
);

    localparam integer K           = 64;
    localparam integer M           = 10;
    localparam integer MP          = 4;
    localparam integer MAC_CYCLES  = 16;

    localparam integer SCALE_MULT  = 3181;
    localparam integer SCALE_SHIFT = 19;

    localparam integer PROD_W        = 16;
    localparam integer ACC_W         = 24;
    localparam integer BIAS_W        = 32;
    localparam integer BIASED_W      = 33;
    localparam integer SCALE_CONST_W = 13;
    localparam integer SCALED_W      = BIASED_W + SCALE_CONST_W;

    localparam signed [SCALE_CONST_W-1:0] SCALE_MULT_CONST = SCALE_MULT;

    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF    =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};

    localparam signed [SCALED_W-1:0] OUT_MAX_S = {{(SCALED_W-8){1'b0}}, 8'sd127};
    localparam signed [SCALED_W-1:0] OUT_MIN_S = {{(SCALED_W-8){1'b1}}, 8'b10000000};

    (* rom_style = "block", ram_style = "block" *)
    reg signed [7:0]  weights [0:M*K-1];
    (* rom_style = "block", ram_style = "block" *)
    reg signed [31:0] biases  [0:M-1];

    initial begin
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/resnet8/weights/node_linear_weights.hex", weights);
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/resnet8/weights/node_linear_bias.hex",    biases);
    end

    localparam [1:0] ST_IDLE = 2'd0;
    localparam [1:0] ST_MAC  = 2'd1;
    localparam [1:0] ST_REQ  = 2'd2;

    reg [1:0] state;
    reg [4:0] mac_cnt;
    reg [4:0] req_cnt;

    reg [K*8-1:0]            in_reg;
    reg signed [M*ACC_W-1:0] acc_packed;

    reg                       stage1_valid;
    reg [3:0]                 stage1_oc;
    reg signed [BIASED_W-1:0] stage1_biased;

    reg                       stage2_valid;
    reg [3:0]                 stage2_oc;
    reg signed [SCALED_W-1:0] stage2_scaled;

    reg                       stage3_valid;
    reg [3:0]                 stage3_oc;
    reg signed [7:0]          stage3_out;

    reg signed [SCALED_W-1:0] v_tmp;
    reg signed [ACC_W-1:0]    acc_sel;
    reg signed [BIASED_W-1:0] acc_ext;
    reg signed [BIASED_W-1:0] bias_ext;
    integer m_i;
    integer k_base;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= ST_IDLE;
            ready_in      <= 1'b1;
            valid_out     <= 1'b0;
            data_out      <= 80'd0;
            mac_cnt       <= 5'd0;
            req_cnt       <= 5'd0;
            in_reg        <= {(K*8){1'b0}};
            acc_packed    <= {(M*ACC_W){1'b0}};
            stage1_valid  <= 1'b0;
            stage1_oc     <= 4'd0;
            stage1_biased <= {BIASED_W{1'b0}};
            stage2_valid  <= 1'b0;
            stage2_oc     <= 4'd0;
            stage2_scaled <= {SCALED_W{1'b0}};
            stage3_valid  <= 1'b0;
            stage3_oc     <= 4'd0;
            stage3_out    <= 8'd0;
        end else begin
            valid_out <= 1'b0;
            case (state)
                ST_IDLE: begin
                    if (valid_in && ready_in) begin
                        in_reg     <= data_in;
                        acc_packed <= {(M*ACC_W){1'b0}};
                        mac_cnt    <= 5'd0;
                        ready_in   <= 1'b0;
                        state      <= ST_MAC;
                    end
                end
                ST_MAC: begin
                    k_base = mac_cnt * MP;
                    for (m_i = 0; m_i < M; m_i = m_i + 1) begin
                        acc_packed[m_i*ACC_W +: ACC_W] <=
                              $signed(acc_packed[m_i*ACC_W +: ACC_W])
                            + $signed(in_reg[(k_base+0)*8 +: 8]) * $signed(weights[m_i*K + k_base + 0])
                            + $signed(in_reg[(k_base+1)*8 +: 8]) * $signed(weights[m_i*K + k_base + 1])
                            + $signed(in_reg[(k_base+2)*8 +: 8]) * $signed(weights[m_i*K + k_base + 2])
                            + $signed(in_reg[(k_base+3)*8 +: 8]) * $signed(weights[m_i*K + k_base + 3]);
                    end
                    if (mac_cnt == MAC_CYCLES - 1) begin
                        state   <= ST_REQ;
                        req_cnt <= 5'd0;
                    end else begin
                        mac_cnt <= mac_cnt + 5'd1;
                    end
                end
                ST_REQ: begin
                    if (req_cnt < M) begin
                        acc_sel      = $signed(acc_packed[req_cnt*ACC_W +: ACC_W]);
                        acc_ext      = {{(BIASED_W-ACC_W){acc_sel[ACC_W-1]}}, acc_sel};
                        bias_ext     = $signed(biases[req_cnt]);
                        stage1_valid <= 1'b1;
                        stage1_oc    <= req_cnt[3:0];
                        stage1_biased <= acc_ext + bias_ext;
                        req_cnt      <= req_cnt + 5'd1;
                    end else begin
                        stage1_valid <= 1'b0;
                    end
                    stage2_valid  <= stage1_valid;
                    stage2_oc     <= stage1_oc;
                    stage2_scaled <= stage1_biased * SCALE_MULT_CONST;
                    stage3_valid <= stage2_valid;
                    stage3_oc    <= stage2_oc;
                    v_tmp = (stage2_scaled +
                             (stage2_scaled[SCALED_W-1] ? SCALE_ROUND_HALF_M1 : SCALE_ROUND_HALF)
                            ) >>> SCALE_SHIFT;
                    stage3_out <= (v_tmp > OUT_MAX_S) ?  8'sd127 :
                                  (v_tmp < OUT_MIN_S) ? -8'sd128 : v_tmp[7:0];
                    if (stage3_valid) begin
                        data_out[stage3_oc*8 +: 8] <= stage3_out;
                        if (stage3_oc == 4'd9) begin
                            valid_out    <= 1'b1;
                            ready_in     <= 1'b1;
                            state        <= ST_IDLE;
                            stage1_valid <= 1'b0;
                            stage2_valid <= 1'b0;
                            stage3_valid <= 1'b0;
                        end
                    end
                end
                default: state <= ST_IDLE;
            endcase
        end
    end
endmodule
