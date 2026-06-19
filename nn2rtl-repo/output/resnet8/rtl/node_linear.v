// node_linear — GEMM K=64, M=10, flat-bus, MP_K=4
// Bus: data_in=512 bits (64 INT8 inputs), data_out=80 bits (10 INT8 outputs)
// Latency: 1 (latch) + 16 (MAC) + 12 (per-output pipeline) = 29 cycles to first valid_out.
// Quantisation: scale_factor=0.006067302138840834 -> SCALE_MULT=25448, SCALE_SHIFT=22.

module node_linear (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    output reg          ready_in,
    input  wire [511:0] data_in,
    output reg          valid_out,
    output reg  [79:0]  data_out
);

    localparam integer K              = 64;
    localparam integer M              = 10;
    localparam integer MP_K           = 4;
    localparam integer K_PASSES       = K / MP_K;       // 16
    localparam integer PROD_W         = 16;
    localparam integer ACC_W          = 22;             // 16 + ceil(log2(64))
    localparam integer BIAS_W         = 32;
    localparam integer BIASED_W       = 33;             // max(ACC_W, BIAS_W) + 1
    localparam integer SCALE_CONST_W  = 16;
    localparam integer SCALED_W       = 49;             // BIASED_W + SCALE_CONST_W
    localparam integer SCALE_SHIFT    = 22;

    localparam signed [SCALE_CONST_W-1:0] SCALE_MULT_CONST = 16'sd25448;
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};
    localparam signed [SCALED_W-1:0] SAT_MAX =  49'sd127;
    localparam signed [SCALED_W-1:0] SAT_MIN = -49'sd128;

    (* rom_style = "block", ram_style = "block" *)
    reg signed [7:0]  weights [0:M*K-1];
    (* rom_style = "block", ram_style = "block" *)
    reg signed [31:0] biases  [0:M-1];

    initial begin
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/resnet8/weights/node_linear_weights.hex", weights);
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/resnet8/weights/node_linear_bias.hex", biases);
    end

    localparam [1:0] ST_IDLE   = 2'd0;
    localparam [1:0] ST_MAC    = 2'd1;
    localparam [1:0] ST_OUTPUT = 2'd2;
    localparam [1:0] ST_DONE   = 2'd3;

    reg [1:0] state;
    reg [4:0] k_counter;           // 0..15
    reg [3:0] out_tick;            // 0..12

    reg signed [7:0]       in_latch [0:K-1];
    reg signed [ACC_W-1:0] acc      [0:M-1];

    reg signed [BIASED_W-1:0] biased_pipe;
    reg signed [SCALED_W-1:0] scaled_pipe;
    reg signed [SCALED_W-1:0] v_tmp;

    integer i, mi;

    always @(posedge clk) begin
        if (state == ST_IDLE && valid_in && ready_in) begin
            for (i = 0; i < K; i = i + 1)
                in_latch[i] <= data_in[i*8 +: 8];
            for (mi = 0; mi < M; mi = mi + 1)
                acc[mi] <= {ACC_W{1'b0}};
        end else if (state == ST_MAC) begin
            for (mi = 0; mi < M; mi = mi + 1) begin
                acc[mi] <= acc[mi]
                    + $signed(weights[mi*K + {k_counter, 2'b00}    ]) * $signed(in_latch[{k_counter, 2'b00}    ])
                    + $signed(weights[mi*K + {k_counter, 2'b00} + 1]) * $signed(in_latch[{k_counter, 2'b00} + 1])
                    + $signed(weights[mi*K + {k_counter, 2'b00} + 2]) * $signed(in_latch[{k_counter, 2'b00} + 2])
                    + $signed(weights[mi*K + {k_counter, 2'b00} + 3]) * $signed(in_latch[{k_counter, 2'b00} + 3]);
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= ST_IDLE;
            ready_in    <= 1'b1;
            valid_out   <= 1'b0;
            data_out    <= 80'b0;
            k_counter   <= 5'd0;
            out_tick    <= 4'd0;
            biased_pipe <= {BIASED_W{1'b0}};
            scaled_pipe <= {SCALED_W{1'b0}};
        end else begin
            valid_out <= 1'b0;
            case (state)
                ST_IDLE: begin
                    ready_in <= 1'b1;
                    if (valid_in) begin
                        ready_in  <= 1'b0;
                        k_counter <= 5'd0;
                        state     <= ST_MAC;
                    end
                end
                ST_MAC: begin
                    if (k_counter == K_PASSES - 1) begin
                        state    <= ST_OUTPUT;
                        out_tick <= 4'd0;
                    end else begin
                        k_counter <= k_counter + 5'd1;
                    end
                end
                ST_OUTPUT: begin
                    if (out_tick <= M - 1) begin
                        biased_pipe <= acc[out_tick[3:0]] + biases[out_tick[3:0]];
                    end
                    if (out_tick >= 1 && out_tick <= M) begin
                        scaled_pipe <= biased_pipe * SCALE_MULT_CONST;
                    end
                    if (out_tick >= 2 && out_tick <= M + 1) begin
                        v_tmp = (scaled_pipe + (scaled_pipe[SCALED_W-1] ? SCALE_ROUND_HALF_M1 : SCALE_ROUND_HALF)) >>> SCALE_SHIFT;
                        if (v_tmp > SAT_MAX)
                            data_out[(out_tick - 4'd2) * 8 +: 8] <= 8'h7F;
                        else if (v_tmp < SAT_MIN)
                            data_out[(out_tick - 4'd2) * 8 +: 8] <= 8'h80;
                        else
                            data_out[(out_tick - 4'd2) * 8 +: 8] <= v_tmp[7:0];
                    end
                    if (out_tick == M + 1) begin
                        valid_out <= 1'b1;
                        state     <= ST_DONE;
                    end
                    out_tick <= out_tick + 4'd1;
                end
                ST_DONE: begin
                    ready_in <= 1'b1;
                    state    <= ST_IDLE;
                end
                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
