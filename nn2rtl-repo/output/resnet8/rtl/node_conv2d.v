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
    localparam integer SCALE_SHIFT_MAX = 23;
    localparam integer PROD_W        = 16;
    localparam integer ACC_W         = 24;
    localparam integer BIAS_W        = 32;
    localparam integer BIASED_W      = ((ACC_W > BIAS_W) ? ACC_W : BIAS_W) + 1;
    localparam integer SCALE_CONST_W = 16;
    localparam integer SCALED_W      = BIASED_W + SCALE_CONST_W;
    localparam integer LATENCY_PAD = 32;

    // ---- Per-OC requant ROMs: compute_scale_approx(scale_factor_per_oc[oc]) ----
    // Variable per-OC shift = the EXACT golden contract (replaces the prior
    // round(s*2^22) + uniform SCALE_SHIFT=22 approximation, which disagreed
    // with the golden at the +-1 clip boundary on 13/16 output channels).
    reg signed [SCALE_CONST_W-1:0] scale_mult_arr  [0:OC-1];
    reg        [5:0]               scale_shift_arr [0:OC-1];
    initial begin
        scale_mult_arr[0]  = 16'sd2953;
        scale_mult_arr[1]  = 16'sd7475;
        scale_mult_arr[2]  = 16'sd291;
        scale_mult_arr[3]  = 16'sd3087;
        scale_mult_arr[4]  = 16'sd10009;
        scale_mult_arr[5]  = 16'sd7075;
        scale_mult_arr[6]  = 16'sd19965;
        scale_mult_arr[7]  = 16'sd1947;
        scale_mult_arr[8]  = 16'sd3457;
        scale_mult_arr[9]  = 16'sd6961;
        scale_mult_arr[10]  = 16'sd6665;
        scale_mult_arr[11]  = 16'sd13031;
        scale_mult_arr[12]  = 16'sd2617;
        scale_mult_arr[13]  = 16'sd3829;
        scale_mult_arr[14]  = 16'sd111;
        scale_mult_arr[15]  = 16'sd11073;
        scale_shift_arr[0] = 6'd20;
        scale_shift_arr[1] = 6'd23;
        scale_shift_arr[2] = 6'd17;
        scale_shift_arr[3] = 6'd20;
        scale_shift_arr[4] = 6'd22;
        scale_shift_arr[5] = 6'd22;
        scale_shift_arr[6] = 6'd23;
        scale_shift_arr[7] = 6'd19;
        scale_shift_arr[8] = 6'd21;
        scale_shift_arr[9] = 6'd22;
        scale_shift_arr[10] = 6'd23;
        scale_shift_arr[11] = 6'd23;
        scale_shift_arr[12] = 6'd20;
        scale_shift_arr[13] = 6'd23;
        scale_shift_arr[14] = 6'd17;
        scale_shift_arr[15] = 6'd21;
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
    // TAIL_PIPE: split scale-multiply  scaled = biased*scale  into a pipelined
    // pair of partial products  (bhi*scale)<<16 + (blo*scale)  (byte-exact).
    // Stage s4a registers the two narrow products; the next stage combines.
    // bhi = biased>>>16 (signed, BIASED_W-16 bits); blo = biased[15:0] (unsigned
    // 16) -> the {1'b0,blo} feeds a 17-bit operand. Product widths bounded by
    // SCALED_W; declare wide enough then truncate into the SCALED_W combine.
    reg signed [SCALED_W:0]   pp_hi_s4a [0:OC-1];  // (biased>>>16)*scale
    reg signed [SCALED_W:0]   pp_lo_s4a [0:OC-1];  // biased[15:0]*scale (>=0 side)
    reg        [5:0]          shift_s4a [0:OC-1];
    reg signed [SCALED_W-1:0] scaled_s4 [0:OC-1];
    reg        [5:0]          shift_s4  [0:OC-1];
    reg [127:0]               data_s5;
    reg [127:0]               data_pipe [0:LATENCY_PAD-1];

    reg pv_s1, pv_s2a, pv_s2, pv_s3, pv_s4, pv_s4a, pv_s5;
    reg [LATENCY_PAD-1:0]     valid_pipe;
    reg [$clog2(OH*OW+1)-1:0] outputs_emitted;

    // [STEM-PIPE] Fmax fix: the 27-term MAC sum was a single-cycle 31-deep CARRY8
    // cone (the global critical path). Split into 3 chunks of 9 (stage s2a) then a
    // 3-term add (stage s2b) -> ~3x shorter logic depth. Byte-exact (associative
    // integer adds, same total); +1 pipeline stage (latency only, throughput II=1).
    reg signed [ACC_W-1:0]    p0_s2a [0:OC-1];
    reg signed [ACC_W-1:0]    p1_s2a [0:OC-1];
    reg signed [ACC_W-1:0]    p2_s2a [0:OC-1];
    reg signed [ACC_W-1:0]    c0, c1, c2;
    reg signed [SCALED_W-1:0] v_tmp;
    integer oc_i, k_i, oc_b, oc_s, oc_r, pi;
    integer w_ic, w_kh, w_kw, win_byte_idx;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pv_s1 <= 1'b0; pv_s2a <= 1'b0; pv_s2 <= 1'b0; pv_s3 <= 1'b0; pv_s4 <= 1'b0; pv_s4a <= 1'b0; pv_s5 <= 1'b0;
            valid_pipe <= {LATENCY_PAD{1'b0}};
            outputs_emitted <= 0;
        end else begin
            pv_s1 <= sched_output_fires;
            pv_s2a <= pv_s1; pv_s2 <= pv_s2a; pv_s3 <= pv_s2; pv_s4 <= pv_s3; pv_s4a <= pv_s4; pv_s5 <= pv_s4a;
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
    // [STEM-PIPE] stage s2a: 3 chunk partial sums (k=0..8 / 9..17 / 18..26).
    always @(posedge clk) begin
        for (oc_i = 0; oc_i < OC; oc_i = oc_i + 1) begin
            c0 = {ACC_W{1'b0}}; c1 = {ACC_W{1'b0}}; c2 = {ACC_W{1'b0}};
            for (k_i = 0; k_i < K_TOTAL; k_i = k_i + 1) begin
                w_ic = k_i / (KH*KW);
                w_kh = (k_i % (KH*KW)) / KW;
                w_kw = k_i % KW;
                win_byte_idx = w_kh*KW*IC + w_kw*IC + w_ic;
                if (k_i < 9)
                    c0 = c0 + $signed(win_s1[win_byte_idx*8 +: 8]) * $signed(weights[oc_i*K_TOTAL + k_i]);
                else if (k_i < 18)
                    c1 = c1 + $signed(win_s1[win_byte_idx*8 +: 8]) * $signed(weights[oc_i*K_TOTAL + k_i]);
                else
                    c2 = c2 + $signed(win_s1[win_byte_idx*8 +: 8]) * $signed(weights[oc_i*K_TOTAL + k_i]);
            end
            p0_s2a[oc_i] <= c0; p1_s2a[oc_i] <= c1; p2_s2a[oc_i] <= c2;
        end
    end
    // [STEM-PIPE] stage s2b: combine chunks -> full sum (byte-identical to the
    // single-cycle 27-term accumulate; just pipelined to break the CARRY8 cone).
    always @(posedge clk) begin
        for (oc_i = 0; oc_i < OC; oc_i = oc_i + 1)
            acc_s2[oc_i] <= $signed(p0_s2a[oc_i]) + $signed(p1_s2a[oc_i]) + $signed(p2_s2a[oc_i]);
    end

    always @(posedge clk) begin
        for (oc_b = 0; oc_b < OC; oc_b = oc_b + 1)
            biased_s3[oc_b] <= $signed(acc_s2[oc_b]) + $signed(biases[oc_b]);
    end

    // TAIL_PIPE stage s4a: the two partial products of the split multiply.
    //   biased = (biased>>>16)<<16 + biased[15:0]
    //   biased*scale = ((biased>>>16)*scale)<<16 + (biased[15:0]*scale)
    // Registering the two narrower products (vs one 33x16) halves the path.
    always @(posedge clk) begin
        for (oc_s = 0; oc_s < OC; oc_s = oc_s + 1) begin
            pp_hi_s4a[oc_s] <= $signed(biased_s3[oc_s] >>> 16) *
                               $signed(scale_mult_arr[oc_s]);
            pp_lo_s4a[oc_s] <= $signed({1'b0, biased_s3[oc_s][15:0]}) *
                               $signed(scale_mult_arr[oc_s]);
            shift_s4a[oc_s] <= scale_shift_arr[oc_s];
        end
    end
    // TAIL_PIPE combine (was the single-cycle scaled_s4 multiply):
    //   scaled = (pp_hi << 16) + pp_lo  == biased*scale  (byte-exact).
    always @(posedge clk) begin
        for (oc_s = 0; oc_s < OC; oc_s = oc_s + 1) begin
            scaled_s4[oc_s] <= ($signed(pp_hi_s4a[oc_s]) <<< 16) + $signed(pp_lo_s4a[oc_s]);
            shift_s4[oc_s]  <= shift_s4a[oc_s];
        end
    end

    always @(posedge clk) begin
        for (oc_r = 0; oc_r < OC; oc_r = oc_r + 1) begin
            // [INVARIANT:ROUNDING] single positive bias + arith >>> = golden floor.
            v_tmp = (scaled_s4[oc_r] +
                     ($signed({{(SCALED_W-1){1'b0}}, 1'b1}) <<< (shift_s4[oc_r] - 1))
                    ) >>> shift_s4[oc_r];
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
