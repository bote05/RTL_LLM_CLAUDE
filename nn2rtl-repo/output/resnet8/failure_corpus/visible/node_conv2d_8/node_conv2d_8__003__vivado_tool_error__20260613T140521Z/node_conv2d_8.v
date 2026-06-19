// node_conv2d_8 -- 3x3 stride-1 pad-1 conv (IC=64, OC=64, IH=IW=8, OH=OW=8).
// Surgeon attempt 3: replace MP=4 serialized conv_datapath (per-pixel ~36960
// cyc) with a fully-parallel-OC 5-stage pipeline + LATENCY_PAD shift register
// to satisfy the pipeline_latency_cycles=1132 contract (hang_budget=4544).
// spatial_fill = 1*(IW+PW)+max(KW-PW,1) = 1*9+2 = 11.
// First valid_out cycle = sched_output_fires_cycle + 5 + LATENCY_PAD.
// LATENCY_PAD = 1132 - 11 - 5 = 1116.
// Per-OC requant: SCALE_SHIFT=22 fits every per-OC mult in 16-bit signed.

module node_conv2d_8 (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       valid_in,
    output wire                       ready_in,
    input  wire [511:0]               data_in,
    output wire                       valid_out,
    output wire [511:0]               data_out
);
    localparam integer IC        = 64;
    localparam integer OC        = 64;
    localparam integer IH        = 8;
    localparam integer IW        = 8;
    localparam integer OH        = 8;
    localparam integer OW        = 8;
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
    localparam integer ACC_W         = 26;   // 16 + clog2(576) = 26
    localparam integer BIAS_W        = 32;
    localparam integer BIASED_W      = ((ACC_W > BIAS_W) ? ACC_W : BIAS_W) + 1;
    localparam integer SCALE_CONST_W = 16;
    localparam integer SCALED_W      = BIASED_W + SCALE_CONST_W;
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};
    localparam integer LATENCY_PAD = 1116;

    reg signed [SCALE_CONST_W-1:0] scale_mult_arr [0:OC-1];
    initial begin
        scale_mult_arr[0]  = 16'sd12463;
        scale_mult_arr[1]  = 16'sd11251;
        scale_mult_arr[2]  = 16'sd13753;
        scale_mult_arr[3]  = 16'sd9215;
        scale_mult_arr[4]  = 16'sd12100;
        scale_mult_arr[5]  = 16'sd14463;
        scale_mult_arr[6]  = 16'sd11732;
        scale_mult_arr[7]  = 16'sd17633;
        scale_mult_arr[8]  = 16'sd11243;
        scale_mult_arr[9]  = 16'sd13354;
        scale_mult_arr[10] = 16'sd13491;
        scale_mult_arr[11] = 16'sd14784;
        scale_mult_arr[12] = 16'sd11589;
        scale_mult_arr[13] = 16'sd10948;
        scale_mult_arr[14] = 16'sd13640;
        scale_mult_arr[15] = 16'sd11720;
        scale_mult_arr[16] = 16'sd13254;
        scale_mult_arr[17] = 16'sd18255;
        scale_mult_arr[18] = 16'sd12095;
        scale_mult_arr[19] = 16'sd11495;
        scale_mult_arr[20] = 16'sd11470;
        scale_mult_arr[21] = 16'sd11755;
        scale_mult_arr[22] = 16'sd12076;
        scale_mult_arr[23] = 16'sd12117;
        scale_mult_arr[24] = 16'sd14128;
        scale_mult_arr[25] = 16'sd15078;
        scale_mult_arr[26] = 16'sd16217;
        scale_mult_arr[27] = 16'sd12249;
        scale_mult_arr[28] = 16'sd16472;
        scale_mult_arr[29] = 16'sd11140;
        scale_mult_arr[30] = 16'sd12231;
        scale_mult_arr[31] = 16'sd12252;
        scale_mult_arr[32] = 16'sd13650;
        scale_mult_arr[33] = 16'sd15184;
        scale_mult_arr[34] = 16'sd9308;
        scale_mult_arr[35] = 16'sd17820;
        scale_mult_arr[36] = 16'sd16000;
        scale_mult_arr[37] = 16'sd12218;
        scale_mult_arr[38] = 16'sd10837;
        scale_mult_arr[39] = 16'sd13660;
        scale_mult_arr[40] = 16'sd12657;
        scale_mult_arr[41] = 16'sd12756;
        scale_mult_arr[42] = 16'sd15097;
        scale_mult_arr[43] = 16'sd11708;
        scale_mult_arr[44] = 16'sd16126;
        scale_mult_arr[45] = 16'sd12660;
        scale_mult_arr[46] = 16'sd9861;
        scale_mult_arr[47] = 16'sd10214;
        scale_mult_arr[48] = 16'sd12255;
        scale_mult_arr[49] = 16'sd11491;
        scale_mult_arr[50] = 16'sd12671;
        scale_mult_arr[51] = 16'sd13520;
        scale_mult_arr[52] = 16'sd10032;
        scale_mult_arr[53] = 16'sd13884;
        scale_mult_arr[54] = 16'sd12889;
        scale_mult_arr[55] = 16'sd12205;
        scale_mult_arr[56] = 16'sd12231;
        scale_mult_arr[57] = 16'sd15865;
        scale_mult_arr[58] = 16'sd11153;
        scale_mult_arr[59] = 16'sd14451;
        scale_mult_arr[60] = 16'sd11229;
        scale_mult_arr[61] = 16'sd10713;
        scale_mult_arr[62] = 16'sd11954;
        scale_mult_arr[63] = 16'sd15077;
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
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/resnet8/weights/node_conv2d_8_weights.hex", weights);
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/resnet8/weights/node_conv2d_8_bias.hex", biases);
    end

    reg [WIN_BITS-1:0]        win_s1;
    reg signed [ACC_W-1:0]    acc_s2    [0:OC-1];
    reg signed [BIASED_W-1:0] biased_s3 [0:OC-1];
    reg signed [SCALED_W-1:0] scaled_s4 [0:OC-1];
    reg [511:0]               data_s5;
    reg [511:0]               data_pipe [0:LATENCY_PAD-1];

    reg pv_s1, pv_s2, pv_s3, pv_s4, pv_s5;
    reg [LATENCY_PAD-1:0]     valid_pipe;
    reg [$clog2(OH*OW+1)-1:0] outputs_emitted;

    reg signed [ACC_W-1:0]    tmp_acc;
    reg signed [SCALED_W-1:0] v_tmp;
    integer oc_i, k_i, oc_b, oc_s, oc_r, vpi, dpi;
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
            for (vpi = 1; vpi < LATENCY_PAD; vpi = vpi + 1)
                valid_pipe[vpi] <= valid_pipe[vpi-1];
            if (valid_pipe[LATENCY_PAD-1] && outputs_emitted < OH*OW)
                outputs_emitted <= outputs_emitted + 1;
        end
    end

    always @(posedge clk) begin
        if (sched_output_fires) win_s1 <= window_flat;
    end

    // MAC: decompose k_i with PyTorch weight order (ic OUTERMOST) and index
    // window_flat with line_buf_window's (kh, kw, ic INNERMOST) layout:
    //   ic = k/(KH*KW), kh = (k%(KH*KW))/KW, kw = k%KW
    //   window_flat[(kh*KW*IC + kw*IC + ic)*8 +: 8]
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
        for (dpi = 1; dpi < LATENCY_PAD; dpi = dpi + 1)
            data_pipe[dpi] <= data_pipe[dpi-1];
    end

    // [INVARIANT:VALID_OUT_LATENCY]
    assign valid_out = valid_pipe[LATENCY_PAD-1];
    assign data_out  = data_pipe[LATENCY_PAD-1];
    // [INVARIANT:READY_IN_GATING]
    assign ready_in  = sched_ready_in;

endmodule
