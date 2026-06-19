// node_linear — gemm (fully-connected) layer for MobileNetV2 classifier.
// K=1280 input features, M=1000 output features.
// flat-bus contract: data_in=10240 bits (K INT8), data_out=8000 bits (M INT8),
// single-beat input, single-beat output.
// First valid_out fires exactly pipeline_latency_cycles (1323) after first valid_in:
//   1 (latch) + 1000 (per-output MAC) + 321 (tail wait) + 1 (emit) = emit at cycle 1323.

module node_linear (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                valid_in,
    output reg                 ready_in,
    input  wire [10239:0]      data_in,
    output reg                 valid_out,
    output reg  [7999:0]       data_out
);

    localparam integer K             = 1280;
    localparam integer M             = 1000;
    localparam integer KLOG2         = 11;
    localparam integer PROD_W        = 16;
    localparam integer ACC_W         = PROD_W + KLOG2;
    localparam integer BIAS_W        = 32;
    localparam integer BIASED_W      = ((ACC_W > BIAS_W) ? ACC_W : BIAS_W) + 1;
    localparam integer SCALE_MULT    = 32568;
    localparam integer SCALE_SHIFT   = 23;
    localparam integer SCALE_MAG_W   = 15;
    localparam integer SCALE_CONST_W = SCALE_MAG_W + 1;
    localparam integer SCALED_W      = BIASED_W + SCALE_CONST_W;

    localparam signed [SCALE_CONST_W-1:0] SCALE_MULT_CONST = 16'sd32568;
    localparam signed [SCALED_W-1:0]      SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    // Signedness footgun fix: explicit $signed cast prevents unsigned coercion
    // of the negative-branch rounding bias under Verilog mixed-sign arithmetic.
    localparam signed [SCALED_W-1:0]      SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - $signed({{(SCALED_W-1){1'b0}}, 1'b1});

    (* rom_style = "block", ram_style = "block" *) reg signed [7:0]  weights [0:M*K-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [31:0] biases  [0:M-1];

    initial begin
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_linear_weights.hex", weights);
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_linear_bias.hex", biases);
    end

    localparam [1:0] ST_IDLE    = 2'd0;
    localparam [1:0] ST_COMPUTE = 2'd1;
    localparam [1:0] ST_WAIT    = 2'd2;
    localparam [1:0] ST_EMIT    = 2'd3;

    reg [1:0]  state;
    reg [15:0] cycle_count;
    reg [15:0] m_counter;
    reg        emit_now;

    reg signed [7:0] in_buf  [0:K-1];
    reg signed [7:0] out_buf [0:M-1];

    integer i, k, m;
    reg signed [ACC_W-1:0]    acc_tmp;
    reg signed [BIASED_W-1:0] biased_tmp;
    reg signed [SCALED_W-1:0] scaled_tmp;
    reg signed [SCALED_W-1:0] v_tmp;
    reg signed [7:0]          clamped_tmp;

    always @(posedge clk) begin
        if (state == ST_IDLE && valid_in && ready_in) begin
            for (i = 0; i < K; i = i + 1) begin
                in_buf[i] <= $signed(data_in[i*8 +: 8]);
            end
        end

        if (state == ST_COMPUTE) begin
            acc_tmp = {ACC_W{1'b0}};
            for (k = 0; k < K; k = k + 1) begin
                acc_tmp = acc_tmp + $signed(in_buf[k]) * $signed(weights[m_counter * K + k]);
            end
            biased_tmp = acc_tmp + $signed(biases[m_counter]);
            scaled_tmp = biased_tmp * SCALE_MULT_CONST;
            // [INVARIANT:ROUNDING]
            v_tmp = (scaled_tmp +
                     (scaled_tmp[SCALED_W-1] ? SCALE_ROUND_HALF_M1 : SCALE_ROUND_HALF)
                    ) >>> SCALE_SHIFT;
            clamped_tmp = (v_tmp > 127)  ?  8'sd127 :
                          (v_tmp < -128) ? -8'sd128 : v_tmp[7:0];
            out_buf[m_counter] <= clamped_tmp;
        end

        if (emit_now) begin
            for (m = 0; m < M; m = m + 1) begin
                data_out[m*8 +: 8] <= out_buf[m];
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= ST_IDLE;
            cycle_count <= 16'd0;
            m_counter   <= 16'd0;
            // [INVARIANT:READY_IN_GATING]
            ready_in    <= 1'b1;
            valid_out   <= 1'b0;
            emit_now    <= 1'b0;
        end else begin
            valid_out <= 1'b0;
            emit_now  <= 1'b0;
            case (state)
                ST_IDLE: begin
                    if (valid_in && ready_in) begin
                        // [INVARIANT:READY_IN_GATING]
                        ready_in    <= 1'b0;
                        cycle_count <= 16'd1;
                        m_counter   <= 16'd0;
                        state       <= ST_COMPUTE;
                    end
                end
                ST_COMPUTE: begin
                    cycle_count <= cycle_count + 16'd1;
                    if (m_counter == M - 1) begin
                        state     <= ST_WAIT;
                        m_counter <= 16'd0;
                    end else begin
                        m_counter <= m_counter + 16'd1;
                    end
                end
                ST_WAIT: begin
                    cycle_count <= cycle_count + 16'd1;
                    if (cycle_count == 16'd1321) begin
                        emit_now <= 1'b1;
                        state    <= ST_EMIT;
                    end
                end
                ST_EMIT: begin
                    // [INVARIANT:VALID_OUT_LATENCY]
                    valid_out <= 1'b1;
                    // [INVARIANT:READY_IN_GATING]
                    ready_in  <= 1'b1;
                    state     <= ST_IDLE;
                end
                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
