// node_conv2d_4 -- conv2d 4x4 stride-2 pad-1 (IC=16, OC=32, IH=IW=32, OH=OW=16).
// resnet8 is per-OC quantized: scale_factor_per_oc[oc] ranges 0.00136..0.00355.
// The conv_datapath library only supports a single scalar SCALE_MULT/SCALE_SHIFT,
// so this top inlines the MAC/BIAS/SCALE/OUTPUT pipeline with a per-OC scale ROM
// (scale_mult_rom[oc] = round(scale_factor_per_oc[oc] * 2^23)) and uniform
// SCALE_SHIFT=23. FSM cadence is identical to conv_datapath: MP*K_TOTAL + 6
// cycles per OC-group pass, OC_PASSES=8 passes per output pixel, 70-cycle
// scheduler pre-roll => first valid_out at 8310 cycles after first valid_in.

module node_conv2d_4 (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       valid_in,
    output wire                       ready_in,
    input  wire [127:0]               data_in,
    output wire                       valid_out,
    output wire [255:0]               data_out
);
    localparam integer IC          = 16;
    localparam integer OC          = 32;
    localparam integer IH          = 32;
    localparam integer IW          = 32;
    localparam integer OH          = 16;
    localparam integer OW          = 16;
    localparam integer KH          = 4;
    localparam integer KW          = 4;
    localparam integer SH          = 2;
    localparam integer SW          = 2;
    localparam integer PH          = 1;
    localparam integer PW          = 1;
    localparam integer K_TOTAL     = IC * KH * KW; // 256
    localparam integer MP          = 4;
    localparam integer OC_PASSES   = OC / MP;       // 8
    localparam integer NUM_WEIGHTS = OC * K_TOTAL;  // 8192

    // Uniform shift; per-OC MULT chosen via round(s * 2^SCALE_SHIFT).
    localparam integer SCALE_SHIFT  = 23;

    localparam integer PROD_W       = 16;
    localparam integer ACC_W        = PROD_W + $clog2(K_TOTAL);                // 24
    localparam integer BIAS_W       = 32;
    localparam integer BIASED_W     = ((ACC_W > BIAS_W) ? ACC_W : BIAS_W) + 1; // 33
    localparam integer SCALE_MULT_W = 16;                                      // max 29742 < 2^15
    localparam integer SCALED_W     = BIASED_W + SCALE_MULT_W;                 // 49

    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};

    localparam integer WEIGHT_ADDR_W   = $clog2(NUM_WEIGHTS);
    localparam integer K_COUNTER_W     = $clog2(K_TOTAL);
    localparam integer LANE_COUNTER_W  = $clog2(MP);
    localparam integer OC_GROUP_W      = $clog2(OC_PASSES);
    localparam integer OC_INDEX_W      = $clog2(OC + MP);

    // ---------------- Per-OC scale ROM --------------------------------
    // scale_mult_rom[oc] = round(scale_factor_per_oc[oc] * 2^23). All 32
    // entries fit in 16-bit signed (max 29742). Replaces the single
    // scalar SCALE_MULT_CONST used by the conv_datapath library.
    reg signed [SCALE_MULT_W-1:0] scale_mult_rom [0:OC-1];
    initial begin
        scale_mult_rom[0]  = 16'sd21499;
        scale_mult_rom[1]  = 16'sd28939;
        scale_mult_rom[2]  = 16'sd19136;
        scale_mult_rom[3]  = 16'sd16565;
        scale_mult_rom[4]  = 16'sd18120;
        scale_mult_rom[5]  = 16'sd23332;
        scale_mult_rom[6]  = 16'sd19595;
        scale_mult_rom[7]  = 16'sd15066;
        scale_mult_rom[8]  = 16'sd23682;
        scale_mult_rom[9]  = 16'sd16897;
        scale_mult_rom[10] = 16'sd18329;
        scale_mult_rom[11] = 16'sd29742;
        scale_mult_rom[12] = 16'sd14576;
        scale_mult_rom[13] = 16'sd14351;
        scale_mult_rom[14] = 16'sd17823;
        scale_mult_rom[15] = 16'sd17423;
        scale_mult_rom[16] = 16'sd18109;
        scale_mult_rom[17] = 16'sd14104;
        scale_mult_rom[18] = 16'sd19385;
        scale_mult_rom[19] = 16'sd11419;
        scale_mult_rom[20] = 16'sd16696;
        scale_mult_rom[21] = 16'sd14459;
        scale_mult_rom[22] = 16'sd14978;
        scale_mult_rom[23] = 16'sd19241;
        scale_mult_rom[24] = 16'sd23059;
        scale_mult_rom[25] = 16'sd23222;
        scale_mult_rom[26] = 16'sd18183;
        scale_mult_rom[27] = 16'sd25111;
        scale_mult_rom[28] = 16'sd22045;
        scale_mult_rom[29] = 16'sd18697;
        scale_mult_rom[30] = 16'sd15010;
        scale_mult_rom[31] = 16'sd23009;
    end

    // ---------------- Frame-start handshake ---------------------------
    reg started, start_pulse, pending_rearm;
    wire sched_out_frame_done;
    wire                              sched_needs_real_input;
    wire                              sched_ready_in;
    wire                              sched_output_fires;
    wire                              sched_advance;
    wire [$clog2(IH + PH + 1)-1:0]    sched_in_row;
    wire [$clog2(IW + PW + 1)-1:0]    sched_in_col;
    wire [$clog2(OH * OW + 1)-1:0]    sched_outputs_emitted;
    wire [KH*KW*IC*8-1:0]             window_flat;
    wire                              mac_busy_w;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            started       <= 1'b0;
            start_pulse   <= 1'b0;
            pending_rearm <= 1'b0;
        end else begin
            start_pulse <= 1'b0;
            if (sched_out_frame_done) begin
                pending_rearm <= 1'b1;
            end
            if (!started) begin
                started     <= 1'b1;
                start_pulse <= 1'b1;
            end else if (pending_rearm && !mac_busy_w) begin
                started       <= 1'b0;
                pending_rearm <= 1'b0;
            end
        end
    end

    wire stall_in = mac_busy_w;

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

    // ---------------- Inlined MAC/BIAS/SCALE/OUTPUT FSM ---------------
    // Mirrors conv_datapath.v exactly except the SCALE stage uses
    // scale_mult_rom[global_oc] instead of a single SCALE_MULT_CONST.

    localparam ST_IDLE   = 3'd0;
    localparam ST_MAC    = 3'd1;
    localparam ST_BIAS   = 3'd2;
    localparam ST_SCALE  = 3'd3;
    localparam ST_OUTPUT = 3'd4;

    reg [2:0]   state;
    reg         valid_out_r;
    reg [255:0] data_out_r;

    (* rom_style = "block", ram_style = "block" *) reg signed [7:0]  weights_mem [0:NUM_WEIGHTS-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [31:0] biases_mem  [0:OC-1];
    initial begin
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/resnet8/weights/node_conv2d_4_weights.hex", weights_mem);
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/resnet8/weights/node_conv2d_4_bias.hex",    biases_mem);
    end

    reg signed [ACC_W-1:0]    acc    [0:MP-1];
    reg signed [BIASED_W-1:0] biased [0:MP-1];
    reg signed [SCALED_W-1:0] scaled [0:MP-1];
    reg signed [SCALED_W-1:0] v_tmp;

    reg [K_COUNTER_W-1:0]    k_counter;
    reg [LANE_COUNTER_W-1:0] lane_counter;
    reg [OC_GROUP_W-1:0]     oc_group;

    integer i, lane_i;
    integer bias_oc;
    integer out_oc;

    assign mac_busy_w = (state != ST_IDLE);
    assign valid_out  = valid_out_r;     // [INVARIANT:VALID_OUT_LATENCY]
    assign data_out   = data_out_r;
    assign ready_in   = sched_ready_in;  // [INVARIANT:READY_IN_GATING]

    wire [OC_INDEX_W-1:0]    current_global_oc = oc_group * MP + lane_counter;
    wire [WEIGHT_ADDR_W-1:0] weight_read_addr  = current_global_oc * K_TOTAL + k_counter;

    function [7:0] tap_at;
        input [K_COUNTER_W-1:0] k;
        integer kh_idx, kw_idx, ic_idx, flat_idx;
        begin
            kh_idx   = (k % (KH * KW)) / KW;
            kw_idx   = k % KW;
            ic_idx   = k / (KH * KW);
            flat_idx = kh_idx * KW * IC + kw_idx * IC + ic_idx;
            tap_at   = window_flat[flat_idx*8 +: 8];
        end
    endfunction

    reg signed [7:0] weight_q;
    reg signed [7:0] tap_q;
    always @(posedge clk) begin
        weight_q <= weights_mem[weight_read_addr];
        tap_q    <= $signed(tap_at(k_counter));
    end

    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] mul_q;

    reg                       mac_valid_q1;
    reg [LANE_COUNTER_W-1:0]  mac_lane_q1;
    reg [OC_INDEX_W-1:0]      mac_global_oc_q1;
    reg                       mac_done_issuing;

    reg                       mac_valid_q2;
    reg [LANE_COUNTER_W-1:0]  mac_lane_q2;
    reg [OC_INDEX_W-1:0]      mac_global_oc_q2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= ST_IDLE;
            valid_out_r      <= 1'b0;
            data_out_r       <= 256'd0;
            k_counter        <= 0;
            lane_counter     <= 0;
            oc_group         <= 0;
            mac_valid_q1     <= 1'b0;
            mac_lane_q1      <= 0;
            mac_global_oc_q1 <= 0;
            mac_valid_q2     <= 1'b0;
            mac_lane_q2      <= 0;
            mac_global_oc_q2 <= 0;
            mac_done_issuing <= 1'b0;
            mul_q            <= 0;
            for (i = 0; i < MP; i = i + 1) begin
                acc[i]    <= 0;
                biased[i] <= 0;
                scaled[i] <= 0;
            end
        end else begin
            valid_out_r <= 1'b0;

            mul_q            <= $signed(weight_q) * $signed(tap_q);
            mac_valid_q2     <= mac_valid_q1;
            mac_lane_q2      <= mac_lane_q1;
            mac_global_oc_q2 <= mac_global_oc_q1;

            if (mac_valid_q2 && mac_global_oc_q2 < OC) begin
                acc[mac_lane_q2] <= acc[mac_lane_q2] + $signed(mul_q);
            end

            case (state)
                ST_IDLE: begin
                    if (sched_output_fires) begin
                        state            <= ST_MAC;
                        k_counter        <= 0;
                        lane_counter     <= 0;
                        oc_group         <= 0;
                        mac_valid_q1     <= 1'b0;
                        mac_valid_q2     <= 1'b0;
                        mac_done_issuing <= 1'b0;
                        for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1)
                            acc[lane_i] <= 0;
                    end
                end

                ST_MAC: begin
                    if (mac_done_issuing) begin
                        mac_valid_q1 <= 1'b0;
                        if (!mac_valid_q1 && !mac_valid_q2) begin
                            mac_done_issuing <= 1'b0;
                            state            <= ST_BIAS;
                        end
                    end else begin
                        mac_lane_q1      <= lane_counter;
                        mac_global_oc_q1 <= current_global_oc;
                        mac_valid_q1     <= 1'b1;

                        if (lane_counter == MP - 1) begin
                            lane_counter <= 0;
                            if (k_counter == K_TOTAL - 1) begin
                                mac_done_issuing <= 1'b1;
                            end else begin
                                k_counter <= k_counter + 1'b1;
                            end
                        end else begin
                            lane_counter <= lane_counter + 1'b1;
                        end
                    end
                end

                ST_BIAS: begin
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                        bias_oc = oc_group * MP + lane_i;
                        biased[lane_i] <= $signed(acc[lane_i]) + $signed(biases_mem[bias_oc]);
                    end
                    state <= ST_SCALE;
                end

                ST_SCALE: begin
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                        scaled[lane_i] <= $signed(biased[lane_i]) *
                                          $signed(scale_mult_rom[oc_group * MP + lane_i]);
                    end
                    state <= ST_OUTPUT;
                end

                ST_OUTPUT: begin
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                        out_oc = oc_group * MP + lane_i;
                        // [INVARIANT:ROUNDING]
                        v_tmp = (scaled[lane_i] +
                                 (scaled[lane_i][SCALED_W-1] ? SCALE_ROUND_HALF_M1
                                                             : SCALE_ROUND_HALF)
                                ) >>> SCALE_SHIFT;
                        data_out_r[out_oc*8 +: 8] <=
                            (v_tmp >  127) ?  8'sd127 :
                            (v_tmp < -128) ? -8'sd128 : v_tmp[7:0];
                    end

                    if (oc_group == OC_PASSES - 1) begin
                        valid_out_r <= 1'b1;
                        state       <= ST_IDLE;
                    end else begin
                        oc_group     <= oc_group + 1'b1;
                        k_counter    <= 0;
                        lane_counter <= 0;
                        for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1)
                            acc[lane_i] <= 0;
                        state <= ST_MAC;
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
