// node_conv2d -- 3x3 stride-1 pad-1 conv (IC=3, OC=16, IH=IW=32).
// Surgeon attempt 2: fix MAC window indexing.
//
// Foundry attempt 1 paired weights[k_i] directly with win_s1[k_i*8 +: 8].
// But line_buf_window flattens window_flat as
//   window_flat[(kh*KW*IC + kw*IC + ic)*8 +: 8]   -- ic INNERMOST
// while the PyTorch [oc, ic, kh, kw] weight file flattens as
//   weights[oc*K_TOTAL + (ic*KH*KW + kh*KW + kw)]  -- ic OUTERMOST
// The two axes are permuted. Decompose k_i with the weight-order
// formula and index window_flat with the corresponding (kh, kw, ic).
//
// Latency contract: pipeline_latency_cycles = 72.
// spatial_fill = max(KH-1-PH,0)*(IW+PW) + max(KW-PW,1) = 1*33 + 2 = 35.
// Internal pipeline = 5 registered stages. LATENCY_PAD = 32.

module node_conv2d (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       valid_in,
    output wire                       ready_in,
    input  wire [23:0]                data_in,
    output wire                       valid_out,
    output wire [127:0]               data_out
);
    localparam integer IC        = 3;
    localparam integer OC        = 16;
    localparam integer IH        = 32;
    localparam integer IW        = 32;
    localparam integer OH        = 32;
    localparam integer OW        = 32;
    localparam integer KH        = 3;
    localparam integer KW        = 3;
    localparam integer SH        = 1;
    localparam integer SW        = 1;
    localparam integer PH        = 1;
    localparam integer PW        = 1;
    localparam integer K_TOTAL   = IC * KH * KW;
    localparam integer WIN_BITS  = KH * KW * IC * 8;

    localparam integer SCALE_SHIFT   = 22;
    localparam integer PROD_W        = 16;
    localparam integer ACC_W         = 24;
    localparam integer BIAS_W        = 32;
    localparam integer BIASED_W      = ((ACC_W > BIAS_W) ? ACC_W : BIAS_W) + 1;
    localparam integer SCALE_CONST_W = 16;
    localparam integer SCALED_W      = BIASED_W + SCALE_CONST_W;
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};
    localparam integer LATENCY_PAD = 32;

    reg signed [SCALE_CONST_W-1:0] scale_mult_arr [0:OC-1];
    initial begin
        scale_mult_arr[0]  = 16'sd11812;
        scale_mult_arr[1]  = 16'sd3737;
        scale_mult_arr[2]  = 16'sd9312;
        scale_mult_arr[3]  = 16'sd12347;
        scale_mult_arr[4]  = 16'sd10010;
        scale_mult_arr[5]  = 16'sd7074;
        scale_mult_arr[6]  = 16'sd9983;
        scale_mult_arr[7]  = 16'sd15575;
        scale_mult_arr[8]  = 16'sd6913;
        scale_mult_arr[9]  = 16'sd6961;
        scale_mult_arr[10] = 16'sd3332;
        scale_mult_arr[11] = 16'sd6514;
        scale_mult_arr[12] = 16'sd10467;
        scale_mult_arr[13] = 16'sd1914;
        scale_mult_arr[14] = 16'sd3552;
        scale_mult_arr[15] = 16'sd22146;
    end

    wire                              sched_needs_real_input;
    wire                              sched_ready_in;
    wire                              sched_output_fires;
    wire                              sched_advance;
    wire [$clog2(IH + PH + 1)-1:0]    sched_in_row;
    wire [$clog2(IW + PW + 1)-1:0]    sched_in_col;
    wire [$clog2(OH * OW + 1)-1:0]    sched_outputs_emitted;
    wire [WIN_BITS-1:0]               window_flat;
    wire                              sched_out_frame_done;

    wire mac_busy = 1'b0;
    wire stall_in = mac_busy;

    reg started, start_pulse, pending_rearm;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            started       <= 1'b0;
            start_pulse   <= 1'b0;
            pending_rearm <= 1'b0;
        end else begin
            start_pulse <= 1'b0;
            if (sched_out_frame_done) pending_rearm <= 1'b1;
            if (!started) begin
                started     <= 1'b1;
                start_pulse <= 1'b1;
            end else if (pending_rearm && !mac_busy) begin
                started       <= 1'b0;
                pending_rearm <= 1'b0;
            end
        end
    end

    coord_scheduler #(
        .IH(IH), .IW(IW), .OH(OH), .OW(OW),
        .KH(KH), .KW(KW), .SH(SH), .SW(SW),
        .PH(PH), .PW(PW)
    ) scheduler (
        .clk(clk), .rst_n(rst_n),
        .start(start_pulse),
        .stall_in(stall_in),
        .valid_in(valid_in),
        .ready_in(sched_ready_in),
        .needs_real_input(sched_needs_real_input),
        .in_row(sched_in_row),
        .in_col(sched_in_col),
        .output_fires(sched_output_fires),
        .advance(sched_advance),
        .in_frame_done(),
        .out_frame_done(sched_out_frame_done),
        .outputs_emitted(sched_outputs_emitted)
    );

    line_buf_window #(
        .IC(IC), .IW(IW), .IH(IH),
        .KH(KH), .KW(KW), .PW(PW), .PH(PH)
    ) lbw (
        .clk(clk), .rst_n(rst_n),
        .frame_start(start_pulse),
        .sched_in_row(sched_in_row),
        .sched_in_col(sched_in_col),
        .sched_needs_real_input(sched_needs_real_input),
        .sched_advance(sched_advance),
        .sched_output_fires(sched_output_fires),
        .valid_in(valid_in),
        .data_in(data_in),
        .window_flat(window_flat)
    );

    (* rom_style = "block", ram_style = "block" *) reg signed [7:0]  weights [0:OC*K_TOTAL-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [31:0] biases  [0:OC-1];
    initial begin
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/resnet8/weights/node_conv2d_weights.hex", weights);
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/resnet8/weights/node_conv2d_bias.hex", biases);
    end

    reg [WIN_BITS-1:0]        win_s1;
    reg signed [ACC_W-1:0]    acc_s2    [0:OC-1];
    reg signed [BIASED_W-1:0] biased_s3 [0:OC-1];
    reg signed [SCALED_W-1:0] scaled_s4 [0:OC-1];
    reg [127:0]               data_s5;
    reg [127:0]               data_pipe [0:LATENCY_PAD-1];

    reg pv_s1, pv_s2, pv_s3, pv_s4, pv_s5;
    reg [LATENCY_PAD-1:0]     valid_pipe;
    reg [$clog2(OH*OW+1)-1:0] outputs_emitted;

    reg signed [ACC_W-1:0]    tmp_acc;
    reg signed [SCALED_W-1:0] v_tmp;
    integer oc_i, k_i, oc_b, oc_s, oc_r, pi;
    integer w_ic, w_kh, w_kw, win_byte_idx;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pv_s1 <= 1'b0; pv_s2 <= 1'b0; pv_s3 <= 1'b0; pv_s4 <= 1'b0; pv_s5 <= 1'b0;
            valid_pipe <= {LATENCY_PAD{1'b0}};
            outputs_emitted <= 0;
        end else begin
            pv_s1 <= sched_output_fires;
            pv_s2 <= pv_s1; pv_s3 <= pv_s2; pv_s4 <= pv_s3; pv_s5 <= pv_s4;
            valid_pipe[0] <= pv_s5;
            for (pi = 1; pi < LATENCY_PAD; pi = pi + 1)
                valid_pipe[pi] <= valid_pipe[pi-1];
            if (valid_pipe[LATENCY_PAD-1] && outputs_emitted < OH*OW)
                outputs_emitted <= outputs_emitted + 1;
        end
    end

    always @(posedge clk) begin
        if (sched_output_fires) win_s1 <= window_flat;
    end

    // MAC: decompose k_i using PyTorch weight order
    //   ic = k/(KH*KW), kh = (k%(KH*KW))/KW, kw = k%KW
    // then index window_flat (ic-innermost layout) at
    //   (kh*KW*IC + kw*IC + ic)*8 +: 8.
    always @(posedge clk) begin
        for (oc_i = 0; oc_i < OC; oc_i = oc_i + 1) begin
            tmp_acc = {ACC_W{1'b0}};
            for (k_i = 0; k_i < K_TOTAL; k_i = k_i + 1) begin
                w_ic = k_i / (KH*KW);
                w_kh = (k_i % (KH*KW)) / KW;
                w_kw = k_i % KW;
                win_byte_idx = w_kh*KW*IC + w_kw*IC + w_ic;
                tmp_acc = tmp_acc + $signed(win_s1[win_byte_idx*8 +: 8]) *
                                    $signed(weights[oc_i*K_TOTAL + k_i]);
            end
            acc_s2[oc_i] <= tmp_acc;
        end
    end

    always @(posedge clk) begin
        for (oc_b = 0; oc_b < OC; oc_b = oc_b + 1)
            biased_s3[oc_b] <= $signed(acc_s2[oc_b]) + $signed(biases[oc_b]);
    end

    always @(posedge clk) begin
        for (oc_s = 0; oc_s < OC; oc_s = oc_s + 1)
            scaled_s4[oc_s] <= $signed(biased_s3[oc_s]) *
                               $signed(scale_mult_arr[oc_s]);
    end

    always @(posedge clk) begin
        for (oc_r = 0; oc_r < OC; oc_r = oc_r + 1) begin
            // [INVARIANT:ROUNDING]
            v_tmp = (scaled_s4[oc_r] +
                     (scaled_s4[oc_r][SCALED_W-1] ? SCALE_ROUND_HALF_M1
                                                  : SCALE_ROUND_HALF)
                    ) >>> SCALE_SHIFT;
            data_s5[oc_r*8 +: 8] <= (v_tmp > 127)  ?  8'sd127 :
                                    (v_tmp < -128) ? -8'sd128 : v_tmp[7:0];
        end
    end

    always @(posedge clk) begin
        data_pipe[0] <= data_s5;
        for (pi = 1; pi < LATENCY_PAD; pi = pi + 1)
            data_pipe[pi] <= data_pipe[pi-1];
    end

    // [INVARIANT:VALID_OUT_LATENCY]
    assign valid_out = valid_pipe[LATENCY_PAD-1];
    assign data_out  = data_pipe[LATENCY_PAD-1];
    // [INVARIANT:READY_IN_GATING]
    assign ready_in  = sched_ready_in;

endmodule
