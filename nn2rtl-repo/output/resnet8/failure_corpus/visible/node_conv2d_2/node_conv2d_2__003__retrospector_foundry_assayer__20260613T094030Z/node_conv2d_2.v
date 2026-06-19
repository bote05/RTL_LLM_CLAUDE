// node_conv2d_2 -- 3x3 stride-1 pad-1 INT8 conv. IC=16, OC=16, IH=IW=OH=OW=32.
// flat-bus contract: data_in=128b, data_out=128b (16 INT8 channels per beat).
// Fully-pipelined throughput-1 datapath (replaces split-arch serial template).
// scale_factor = 1.093115374452829 -> SCALE_MULT=17910, SCALE_SHIFT=14
//   (17910 / 16384 = 1.0931396484375; rel err ~2.22e-5 vs 1.0931153744528...)
// Pipeline-fill latency contract: pipeline_latency_cycles=124 (LayerIR).

module node_conv2d_2 (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       valid_in,
    output wire                       ready_in,
    input  wire [127:0]               data_in,
    output wire                       valid_out,
    output wire [127:0]               data_out
);
    // ---- Layer geometry ----
    localparam integer IC          = 16;
    localparam integer OC          = 16;
    localparam integer IH          = 32;
    localparam integer IW          = 32;
    localparam integer OH          = 32;
    localparam integer OW          = 32;
    localparam integer KH          = 3;
    localparam integer KW          = 3;
    localparam integer K_TOTAL     = IC * KH * KW;   // 144
    localparam integer IN_PIXELS   = IH * IW;        // 1024
    localparam integer OUT_PIXELS  = OH * OW;        // 1024

    // ---- Quantization scale. INVARIANT(ROUNDING) ----
    localparam signed [15:0] SCALE_MULT_S    = 16'sd17910;
    localparam integer       SCALE_SHIFT     = 14;
    localparam signed [48:0] SCALE_HALF_S    = 49'sd8192;   // 1 << (SCALE_SHIFT-1)
    localparam signed [48:0] SCALE_HALF_M1_S = 49'sd8191;

    // ---- Pipeline-fill latency: 124 cycles ----
    // 8 register stages between trigger and valid_out_q.
    // Trigger comb at cycle_count == 116 -> valid_out at 116 + 8 = 124.
    // INVARIANT(VALID_OUT_LATENCY=124)
    localparam integer TRIGGER_THRESH = 116;

    // ---- Weight + bias on-chip ROMs ($readmemh). INVARIANT(MEMORY) ----
    reg signed [7:0]  weights [0:OC*K_TOTAL-1];
    reg signed [31:0] biases  [0:OC-1];

    initial begin
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/resnet8/weights/node_conv2d_2_weights.hex", weights);
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/resnet8/weights/node_conv2d_2_bias.hex",    biases);
    end

    // ---- Activation memory: full 32x32 frame for back-to-back vector safety ----
    reg [127:0] line_buf [0:IN_PIXELS-1];

    // ---- Control state ----
    reg        started_q;
    reg [11:0] cycle_count_q;
    reg [11:0] in_count_q;
    reg [11:0] out_trigger_count_q;
    reg [11:0] out_count_q;
    reg [5:0]  out_pipe_row_q;
    reg [5:0]  out_pipe_col_q;

    // ---- Pipeline valid shift register ----
    reg pipe_v0_q, pipe_v1_q, pipe_v2_q, pipe_v3_q, pipe_v4_q, pipe_v5_q, pipe_v6_q;
    reg         valid_out_q;
    reg [127:0] data_out_q;

    // ---- Datapath pipeline registers (module-scope per Verilog-2001 scoping rule) ----
    reg signed [7:0]  window_reg  [0:K_TOTAL-1];      // 144 INT8 window bytes
    reg signed [15:0] prod_reg    [0:OC*K_TOTAL-1];   // 2304 signed products
    reg signed [18:0] sum8_reg    [0:OC*18-1];        // 288 partial sums of 8 products
    reg signed [23:0] sum_oc_reg  [0:OC-1];           // 16 per-OC final sums
    reg signed [32:0] biased_reg  [0:OC-1];           // 16 sum + bias
    reg signed [48:0] scaled_reg  [0:OC-1];           // 16 (sum+bias) * SCALE_MULT
    reg signed [48:0] rounded_reg [0:OC-1];           // 16 rounded + arith-shifted

    // ---- Loop indices and scratch (module scope) ----
    integer oc_i, ic_i, kh_i, kw_i, k_i, s8_i;
    integer wsel_r, wsel_c;
    reg signed [18:0] partial_sum8;
    reg signed [23:0] full_sum_oc;

    // ---- Combinational window selection from line_buf ----
    reg [127:0] window_sel [0:KH*KW-1];

    always @* begin
        for (kh_i = 0; kh_i < KH; kh_i = kh_i + 1) begin
            for (kw_i = 0; kw_i < KW; kw_i = kw_i + 1) begin
                wsel_r = out_pipe_row_q + kh_i - 1;
                wsel_c = out_pipe_col_q + kw_i - 1;
                if (wsel_r < 0 || wsel_r >= IH || wsel_c < 0 || wsel_c >= IW)
                    window_sel[kh_i*KW + kw_i] = 128'b0;
                else
                    window_sel[kh_i*KW + kw_i] = line_buf[wsel_r * IW + wsel_c];
            end
        end
    end

    // ---- Control signals ----
    wire input_fire        = valid_in && (in_count_q < IN_PIXELS);
    wire start_comb        = !started_q && input_fire;
    wire pipe_trigger      = started_q && (cycle_count_q >= TRIGGER_THRESH) && (out_trigger_count_q < OUT_PIXELS);
    wire frame_will_finish = pipe_v6_q && (out_count_q == (OUT_PIXELS - 1));

    assign ready_in  = 1'b1;  // INVARIANT(READY_IN_GATING=always_ready)
    assign valid_out = valid_out_q;
    assign data_out  = data_out_q;

    // ============================================================
    // Async-reset block: scalars + pipe-valid shift + data_out_q saturate/pack.
    // No unpacked-array writes here (activation_memory_in_async_reset_block rule).
    // ============================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            started_q           <= 1'b0;
            cycle_count_q       <= 12'd0;
            in_count_q          <= 12'd0;
            out_trigger_count_q <= 12'd0;
            out_count_q         <= 12'd0;
            out_pipe_row_q      <= 6'd0;
            out_pipe_col_q      <= 6'd0;
            pipe_v0_q           <= 1'b0;
            pipe_v1_q           <= 1'b0;
            pipe_v2_q           <= 1'b0;
            pipe_v3_q           <= 1'b0;
            pipe_v4_q           <= 1'b0;
            pipe_v5_q           <= 1'b0;
            pipe_v6_q           <= 1'b0;
            valid_out_q         <= 1'b0;
            data_out_q          <= 128'b0;
        end else begin
            if (start_comb) begin
                started_q <= 1'b1;
            end

            if (start_comb || started_q) begin
                cycle_count_q <= cycle_count_q + 12'd1;
            end

            if (input_fire) begin
                in_count_q <= in_count_q + 12'd1;
            end

            if (pipe_trigger) begin
                out_trigger_count_q <= out_trigger_count_q + 12'd1;
                if (out_pipe_col_q == OW - 1) begin
                    out_pipe_col_q <= 6'd0;
                    out_pipe_row_q <= out_pipe_row_q + 6'd1;
                end else begin
                    out_pipe_col_q <= out_pipe_col_q + 6'd1;
                end
            end

            pipe_v0_q   <= pipe_trigger;
            pipe_v1_q   <= pipe_v0_q;
            pipe_v2_q   <= pipe_v1_q;
            pipe_v3_q   <= pipe_v2_q;
            pipe_v4_q   <= pipe_v3_q;
            pipe_v5_q   <= pipe_v4_q;
            pipe_v6_q   <= pipe_v5_q;
            valid_out_q <= pipe_v6_q;

            if (pipe_v6_q) begin
                out_count_q <= out_count_q + 12'd1;
            end

            // Saturate rounded_reg to int8 and pack into data_out_q (16 slice writes).
            for (oc_i = 0; oc_i < OC; oc_i = oc_i + 1) begin
                if (rounded_reg[oc_i] > 49'sd127)
                    data_out_q[8*oc_i +: 8] <= 8'h7F;
                else if (rounded_reg[oc_i] < -49'sd128)
                    data_out_q[8*oc_i +: 8] <= 8'h80;
                else
                    data_out_q[8*oc_i +: 8] <= rounded_reg[oc_i][7:0];
            end

            // Re-arm for back-to-back vectors. Overrides earlier counter increments.
            if (frame_will_finish) begin
                started_q           <= 1'b0;
                cycle_count_q       <= 12'd0;
                in_count_q          <= 12'd0;
                out_trigger_count_q <= 12'd0;
                out_count_q         <= 12'd0;
                out_pipe_row_q      <= 6'd0;
                out_pipe_col_q      <= 6'd0;
            end
        end
    end

    // ============================================================
    // Sync-only block: activation memory + pipeline-stage array writes.
    // ============================================================
    always @(posedge clk) begin
        if (input_fire) begin
            line_buf[in_count_q] <= data_in;
        end

        // Stage 1: latch 144-byte 3x3xIC window.
        // Layout: window_reg[ic*KH*KW + kh*KW + kw] matches
        //         weights  [oc*K_TOTAL + ic*KH*KW + kh*KW + kw] (ic-outer).
        for (kh_i = 0; kh_i < KH; kh_i = kh_i + 1) begin
            for (kw_i = 0; kw_i < KW; kw_i = kw_i + 1) begin
                for (ic_i = 0; ic_i < IC; ic_i = ic_i + 1) begin
                    window_reg[ic_i*KH*KW + kh_i*KW + kw_i] <=
                        $signed(window_sel[kh_i*KW + kw_i][8*ic_i +: 8]);
                end
            end
        end

        // Stage 2: 2304 signed INT8 * INT8 -> 16b signed products.
        for (oc_i = 0; oc_i < OC; oc_i = oc_i + 1) begin
            for (k_i = 0; k_i < K_TOTAL; k_i = k_i + 1) begin
                prod_reg[oc_i*K_TOTAL + k_i] <=
                    window_reg[k_i] * weights[oc_i*K_TOTAL + k_i];
            end
        end

        // Stage 3: per-OC, 18 partial sums of 8 products each.
        for (oc_i = 0; oc_i < OC; oc_i = oc_i + 1) begin
            for (s8_i = 0; s8_i < 18; s8_i = s8_i + 1) begin
                partial_sum8 = 19'sd0;
                for (k_i = 0; k_i < 8; k_i = k_i + 1) begin
                    partial_sum8 = partial_sum8 + prod_reg[oc_i*K_TOTAL + s8_i*8 + k_i];
                end
                sum8_reg[oc_i*18 + s8_i] <= partial_sum8;
            end
        end

        // Stage 4: per-OC final sum of the 18 partial sums.
        for (oc_i = 0; oc_i < OC; oc_i = oc_i + 1) begin
            full_sum_oc = 24'sd0;
            for (s8_i = 0; s8_i < 18; s8_i = s8_i + 1) begin
                full_sum_oc = full_sum_oc + sum8_reg[oc_i*18 + s8_i];
            end
            sum_oc_reg[oc_i] <= full_sum_oc;
        end

        // Stage 5: + bias (signed widths via reg signed on both sides).
        for (oc_i = 0; oc_i < OC; oc_i = oc_i + 1) begin
            biased_reg[oc_i] <= sum_oc_reg[oc_i] + biases[oc_i];
        end

        // Stage 6: x SCALE_MULT (signed * signed -> signed).
        for (oc_i = 0; oc_i < OC; oc_i = oc_i + 1) begin
            scaled_reg[oc_i] <= biased_reg[oc_i] * SCALE_MULT_S;
        end

        // Stage 7: sign-aware rounding + arithmetic shift right by SCALE_SHIFT.
        // INVARIANT(ROUNDING)
        for (oc_i = 0; oc_i < OC; oc_i = oc_i + 1) begin
            if (scaled_reg[oc_i][48])
                rounded_reg[oc_i] <= (scaled_reg[oc_i] + SCALE_HALF_M1_S) >>> SCALE_SHIFT;
            else
                rounded_reg[oc_i] <= (scaled_reg[oc_i] + SCALE_HALF_S) >>> SCALE_SHIFT;
        end
    end

endmodule
