// node_conv2d_5 -- 3x3 stride-1 pad-1 conv (IC=32, OC=32, IH=IW=16, OH=OW=16).
// RE-PARALLELIZED: MP=16 lanes x K_PAR=16 [DSP] TREE4 PACK2 taps = 256 INT8 multiplies/cycle.
// ST_MAC = K_GROUPS(18) * MP(16) cycles/pass; OC_PASSES=2 passes/pixel.
// Byte-exact vs the serial MP=4 FSM (same products, same accumulation order, same
// per-OC requant compute_scale_approx, same round/saturate). Weights repacked WIDE
// (MP*K_PAR bytes/word) read one word/cycle.
// DSP_PACK: WP487 dual-INT8-MACC -- TWO OCs share each DSP48E2 (A=(w_n<<<18)+w_m,
// B=a; P=(a*w_n)<<<18+(a*w_m)). The K_PAR packed products tree-sum to depth-4
// nodes, then UNPACK (signed lo=node[17:0], hi=node[..:18]+borrow) into the two
// per-OC signed partials -> halves the DSP multiplier count ((MP/2)*K_PAR packed
// products). Byte-exact (OFFSET=18 is the unique offset: A fits 27b @ S<=18, depth-4
// LO field fits @ S>=18); same data_latency as TREE4 (re-gated byte-exact).
// TREE_STAGES=4: K_PAR reduction is a 4-level PIPELINED balanced binary adder
// tree (breaks the 16-deep linear DSP cascade -> shorter critical path + frees
// global placement). Byte-exact (associative integer adds, +1 bit/level, no trunc);
// adds 4 cycles of MAC-issue latency (valid chain + drain deepened to match).


module node_conv2d_5 (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       valid_in,
    output wire                       ready_in,
    input  wire [255:0]               data_in,
    output wire                       valid_out,
    output wire [255:0]               data_out
);
    localparam integer IC          = 32;
    localparam integer OC          = 32;
    localparam integer IH          = 16;
    localparam integer IW          = 16;
    localparam integer OH          = 16;
    localparam integer OW          = 16;
    localparam integer KH          = 3;
    localparam integer KW          = 3;
    localparam integer SH          = 1;
    localparam integer SW          = 1;
    localparam integer PH          = 1;
    localparam integer PW          = 1;
    localparam integer K_TOTAL     = IC * KH * KW; // 288
    localparam integer MP          = 16;
    localparam integer K_PAR       = 16;
    localparam integer K_GROUPS    = K_TOTAL / K_PAR;  // 18
    localparam integer OC_PASSES   = OC / MP;          // 2
    localparam integer NUM_WIDE    = OC_PASSES * K_GROUPS; // 36
    localparam integer WIDE_W      = MP * K_PAR * 8;   // 2048

    localparam integer PROD_W       = 16;
    localparam integer TREE_W       = PROD_W + $clog2(K_PAR);
    localparam integer ACC_W        = TREE_W + $clog2(K_GROUPS > 1 ? K_GROUPS : 2);
    localparam integer BIAS_W       = 32;
    localparam integer BIASED_W     = ((ACC_W > BIAS_W) ? ACC_W : BIAS_W) + 1;
    localparam integer SCALE_MULT_W = 16;
    localparam integer SCALED_W     = BIASED_W + SCALE_MULT_W;

    localparam integer KGROUP_W     = (K_GROUPS <= 1) ? 1 : $clog2(K_GROUPS);
    localparam integer OC_GROUP_W   = (OC_PASSES <= 1) ? 1 : $clog2(OC_PASSES);

    // ---- Per-OC requant ROMs: compute_scale_approx(scale_factor_per_oc[oc]) ----
    reg signed [SCALE_MULT_W-1:0] scale_mult_rom  [0:OC-1];
    reg        [5:0]              scale_shift_rom [0:OC-1];
    initial begin
        scale_mult_rom[0]  = 16'sd23881;
        scale_mult_rom[1]  = 16'sd6465;
        scale_mult_rom[2]  = 16'sd14161;
        scale_mult_rom[3]  = 16'sd8323;
        scale_mult_rom[4]  = 16'sd22835;
        scale_mult_rom[5]  = 16'sd10123;
        scale_mult_rom[6]  = 16'sd807;
        scale_mult_rom[7]  = 16'sd23025;
        scale_mult_rom[8]  = 16'sd9213;
        scale_mult_rom[9]  = 16'sd17347;
        scale_mult_rom[10]  = 16'sd29027;
        scale_mult_rom[11]  = 16'sd9439;
        scale_mult_rom[12]  = 16'sd9917;
        scale_mult_rom[13]  = 16'sd12597;
        scale_mult_rom[14]  = 16'sd2685;
        scale_mult_rom[15]  = 16'sd10861;
        scale_mult_rom[16]  = 16'sd8423;
        scale_mult_rom[17]  = 16'sd4915;
        scale_mult_rom[18]  = 16'sd19551;
        scale_mult_rom[19]  = 16'sd9799;
        scale_mult_rom[20]  = 16'sd24913;
        scale_mult_rom[21]  = 16'sd18947;
        scale_mult_rom[22]  = 16'sd9205;
        scale_mult_rom[23]  = 16'sd25113;
        scale_mult_rom[24]  = 16'sd9163;
        scale_mult_rom[25]  = 16'sd6043;
        scale_mult_rom[26]  = 16'sd6159;
        scale_mult_rom[27]  = 16'sd22915;
        scale_mult_rom[28]  = 16'sd11983;
        scale_mult_rom[29]  = 16'sd21239;
        scale_mult_rom[30]  = 16'sd20855;
        scale_mult_rom[31]  = 16'sd10153;
        scale_shift_rom[0] = 6'd23;
        scale_shift_rom[1] = 6'd21;
        scale_shift_rom[2] = 6'd23;
        scale_shift_rom[3] = 6'd22;
        scale_shift_rom[4] = 6'd23;
        scale_shift_rom[5] = 6'd22;
        scale_shift_rom[6] = 6'd18;
        scale_shift_rom[7] = 6'd23;
        scale_shift_rom[8] = 6'd22;
        scale_shift_rom[9] = 6'd23;
        scale_shift_rom[10] = 6'd23;
        scale_shift_rom[11] = 6'd22;
        scale_shift_rom[12] = 6'd21;
        scale_shift_rom[13] = 6'd22;
        scale_shift_rom[14] = 6'd20;
        scale_shift_rom[15] = 6'd22;
        scale_shift_rom[16] = 6'd22;
        scale_shift_rom[17] = 6'd21;
        scale_shift_rom[18] = 6'd23;
        scale_shift_rom[19] = 6'd22;
        scale_shift_rom[20] = 6'd23;
        scale_shift_rom[21] = 6'd23;
        scale_shift_rom[22] = 6'd22;
        scale_shift_rom[23] = 6'd23;
        scale_shift_rom[24] = 6'd22;
        scale_shift_rom[25] = 6'd21;
        scale_shift_rom[26] = 6'd21;
        scale_shift_rom[27] = 6'd23;
        scale_shift_rom[28] = 6'd22;
        scale_shift_rom[29] = 6'd23;
        scale_shift_rom[30] = 6'd23;
        scale_shift_rom[31] = 6'd22;
    end

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

    localparam ST_IDLE   = 3'd0;
    localparam ST_MAC    = 3'd1;
    localparam ST_BIAS   = 3'd2;
    localparam ST_SCALE  = 3'd3;
    localparam ST_OUTPUT = 3'd4;

    reg [2:0]   state;
    reg         valid_out_r;
    reg [255:0] data_out_r;

    // ---- Wide weight ROM: MP*K_PAR bytes/word, [oc_group*K_GROUPS + k_group] ----
    (* rom_style = "block", ram_style = "block" *) reg [WIDE_W-1:0]  weights_wide [0:NUM_WIDE-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [31:0] biases_mem   [0:OC-1];
    initial begin
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/resnet8/weights/node_conv2d_5_weights_wide_mp16_kp16.hex", weights_wide);
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/resnet8/weights/node_conv2d_5_bias.hex",       biases_mem);
    end

    // [PIPELINE] Two accumulator banks (acc_b0/acc_b1) alternate per work-item.
    reg signed [ACC_W-1:0]    acc_b0 [0:MP-1];
    reg signed [ACC_W-1:0]    acc_b1 [0:MP-1];
    reg                       issuing;             // issuing K_GROUPS of a work-item
    reg                       pending;             // fired pixel awaiting first work-item
    reg                       pixel_active;        // window held across the pixel's OC_PASSES work-items
    reg                       ib;                  // issue bank (0/1)
    reg                       bank_busy0, bank_busy1;
    reg                       rq_v1, rq_bank1, rq_v2, rq_v3;        // 3-stage requant pipe
    reg [OC_GROUP_W-1:0]      rq_oc1, rq_oc2, rq_oc3;               // oc_group carried through requant
    reg signed [BIASED_W-1:0] biased [0:MP-1];
    reg signed [SCALED_W-1:0] scaled [0:MP-1];
    reg [5:0]                 shift_lane [0:MP-1];
    reg signed [SCALED_W-1:0] v_tmp;

    reg [KGROUP_W-1:0]   k_group;
    reg [OC_GROUP_W-1:0] oc_group;

    integer i, lane_i;
    integer bias_oc;
    integer out_oc;

    assign mac_busy_w = pixel_active || pending;  // [PIPELINE] hold window across pixel's work-items
    assign valid_out  = valid_out_r;     // [INVARIANT:VALID_OUT_LATENCY]
    assign data_out   = data_out_r;
    assign ready_in   = sched_ready_in;  // [INVARIANT:READY_IN_GATING]

    wire [$clog2(NUM_WIDE+1)-1:0] weight_read_addr = oc_group * K_GROUPS + k_group;

    // Window-tap indexer. Linear k index -> (kh,kw,ic) -> flat window slice.
    function [7:0] tap_at;
        input integer k_lin;
        integer kh_idx, kw_idx, ic_idx, flat_idx;
        begin
            kh_idx   = (k_lin % (KH * KW)) / KW;
            kw_idx   = k_lin % KW;
            ic_idx   = k_lin / (KH * KW);
            flat_idx = kh_idx * KW * IC + kw_idx * IC + ic_idx;
            tap_at   = window_flat[flat_idx*8 +: 8];
        end
    endfunction

    // ---- Stage 1: register wide weight word + K_PAR taps for current k_group ----
    (* max_fanout = 64 *) reg [WIDE_W-1:0] weight_word_q;
    (* max_fanout = 64 *) reg signed [7:0] tap_q [0:K_PAR-1];
    integer ld_i;
    always @(posedge clk) begin
        weight_word_q <= weights_wide[weight_read_addr];
        for (ld_i = 0; ld_i < K_PAR; ld_i = ld_i + 1)
            tap_q[ld_i] <= $signed(tap_at(k_group * K_PAR + ld_i));
    end

    // ---- Stage 2: MP*K_PAR multipliers + per-lane reduction (legacy linear OR
    //               pipelined balanced adder tree, selected by TREE_STAGES). ----
    // [DSP_PACK] WP487 dual-INT8-MACC: TWO OCs per DSP48E2 (shared activation).
    // OFFSET=18: A=(w_n<<<18)+w_m (27b), B=a (8b); P=(a*w_n)<<<18+(a*w_m).
    localparam integer PACK_OFFSET = 18;
    localparam integer PACK_A_W    = 27;          // DSP48E2 A port
    localparam integer PACK_PROD_W = PACK_A_W + 8; // 27x8 packed product
    localparam integer PACK_NODE_W = PACK_PROD_W + 4;
    localparam integer PACK_PAIRS  = MP / 2;
    localparam integer PACK_N4     = K_PAR / 4;   // depth-4 nodes per pair
    // Level 0a combinational: packed A operand + packed DSP product.
    reg signed [PACK_A_W-1:0]    pack_a_comb [0:PACK_PAIRS*K_PAR-1];
    reg signed [PACK_PROD_W-1:0] pp_comb     [0:PACK_PAIRS*K_PAR-1];
    (* use_dsp = "yes" *) reg signed [PACK_PROD_W-1:0] pp_q [0:PACK_PAIRS*K_PAR-1];
    reg signed [PACK_NODE_W-1:0] ptree_l1 [0:PACK_PAIRS*8-1];
    reg signed [PACK_NODE_W-1:0] ptree_l2 [0:PACK_PAIRS*4-1];
    reg signed [TREE_W-1:0] un_lo [0:PACK_PAIRS*PACK_N4-1];
    reg signed [TREE_W-1:0] un_hi [0:PACK_PAIRS*PACK_N4-1];
    reg signed [TREE_W-1:0] lane_partial [0:MP-1];
    integer cs_pair, cs_kpos;
    // [DSP_PACK] Level-0a: pack A=(w_n<<<OFFSET)+w_m, packed product = A*tap.
    always @* begin
        for (cs_pair = 0; cs_pair < PACK_PAIRS; cs_pair = cs_pair + 1)
            for (cs_kpos = 0; cs_kpos < K_PAR; cs_kpos = cs_kpos + 1) begin
                pack_a_comb[cs_pair*K_PAR + cs_kpos] =
                    ($signed(weight_word_q[((2*cs_pair+1) * K_PAR + cs_kpos) * 8 +: 8]) <<< PACK_OFFSET) +
                     $signed(weight_word_q[((2*cs_pair  ) * K_PAR + cs_kpos) * 8 +: 8]);
                pp_comb[cs_pair*K_PAR + cs_kpos] =
                    pack_a_comb[cs_pair*K_PAR + cs_kpos] * $signed(tap_q[cs_kpos]);
            end
    end
    // [DSP_PACK] Level-0b register (DSP packed mult) + tree->depth4 + unpack + sum.
    integer pk_pair, pk_i, un_j;
    always @(posedge clk) begin
        for (pk_i = 0; pk_i < PACK_PAIRS*K_PAR; pk_i = pk_i + 1)
            pp_q[pk_i] <= pp_comb[pk_i];
        for (pk_pair = 0; pk_pair < PACK_PAIRS; pk_pair = pk_pair + 1) begin
            ptree_l1[pk_pair*8 + 0] <= $signed(pp_q[pk_pair*16 + 0]) + $signed(pp_q[pk_pair*16 + 1]);
            ptree_l1[pk_pair*8 + 1] <= $signed(pp_q[pk_pair*16 + 2]) + $signed(pp_q[pk_pair*16 + 3]);
            ptree_l1[pk_pair*8 + 2] <= $signed(pp_q[pk_pair*16 + 4]) + $signed(pp_q[pk_pair*16 + 5]);
            ptree_l1[pk_pair*8 + 3] <= $signed(pp_q[pk_pair*16 + 6]) + $signed(pp_q[pk_pair*16 + 7]);
            ptree_l1[pk_pair*8 + 4] <= $signed(pp_q[pk_pair*16 + 8]) + $signed(pp_q[pk_pair*16 + 9]);
            ptree_l1[pk_pair*8 + 5] <= $signed(pp_q[pk_pair*16 + 10]) + $signed(pp_q[pk_pair*16 + 11]);
            ptree_l1[pk_pair*8 + 6] <= $signed(pp_q[pk_pair*16 + 12]) + $signed(pp_q[pk_pair*16 + 13]);
            ptree_l1[pk_pair*8 + 7] <= $signed(pp_q[pk_pair*16 + 14]) + $signed(pp_q[pk_pair*16 + 15]);
            ptree_l2[pk_pair*4 + 0] <= $signed(ptree_l1[pk_pair*8 + 0]) + $signed(ptree_l1[pk_pair*8 + 1]);
            ptree_l2[pk_pair*4 + 1] <= $signed(ptree_l1[pk_pair*8 + 2]) + $signed(ptree_l1[pk_pair*8 + 3]);
            ptree_l2[pk_pair*4 + 2] <= $signed(ptree_l1[pk_pair*8 + 4]) + $signed(ptree_l1[pk_pair*8 + 5]);
            ptree_l2[pk_pair*4 + 3] <= $signed(ptree_l1[pk_pair*8 + 6]) + $signed(ptree_l1[pk_pair*8 + 7]);
        end
        // [DSP_PACK] unpack each depth-4 packed node into signed (lo,hi).
        for (pk_pair = 0; pk_pair < PACK_PAIRS; pk_pair = pk_pair + 1)
            for (un_j = 0; un_j < PACK_N4; un_j = un_j + 1) begin
                un_lo[pk_pair*PACK_N4 + un_j] <= $signed(ptree_l2[pk_pair*PACK_N4 + un_j][PACK_OFFSET-1:0]);
                un_hi[pk_pair*PACK_N4 + un_j] <= $signed(ptree_l2[pk_pair*PACK_N4 + un_j][PACK_NODE_W-1:PACK_OFFSET]) + ptree_l2[pk_pair*PACK_N4 + un_j][PACK_OFFSET-1];
            end
        // [DSP_PACK] sum the PACK_N4 unpacked partials -> lane_partial (per OC).
        for (pk_pair = 0; pk_pair < PACK_PAIRS; pk_pair = pk_pair + 1) begin
            lane_partial[2*pk_pair    ] <= $signed(un_lo[pk_pair*PACK_N4 + 0]) + $signed(un_lo[pk_pair*PACK_N4 + 1]) + $signed(un_lo[pk_pair*PACK_N4 + 2]) + $signed(un_lo[pk_pair*PACK_N4 + 3]);
            lane_partial[2*pk_pair + 1] <= $signed(un_hi[pk_pair*PACK_N4 + 0]) + $signed(un_hi[pk_pair*PACK_N4 + 1]) + $signed(un_hi[pk_pair*PACK_N4 + 2]) + $signed(un_hi[pk_pair*PACK_N4 + 3]);
        end
    end

    // [PIPELINE] valid chain (depth n_valid=6) carries valid + bank + last + oc tags.
    reg mac_valid_q1; reg mac_bank_q1; reg mac_last_q1; reg [OC_GROUP_W-1:0] mac_oc_q1;
    reg mac_valid_q2; reg mac_bank_q2; reg mac_last_q2; reg [OC_GROUP_W-1:0] mac_oc_q2;
    reg mac_valid_q3; reg mac_bank_q3; reg mac_last_q3; reg [OC_GROUP_W-1:0] mac_oc_q3;
    reg mac_valid_q4; reg mac_bank_q4; reg mac_last_q4; reg [OC_GROUP_W-1:0] mac_oc_q4;
    reg mac_valid_q5; reg mac_bank_q5; reg mac_last_q5; reg [OC_GROUP_W-1:0] mac_oc_q5;
    reg mac_valid_q6; reg mac_bank_q6; reg mac_last_q6; reg [OC_GROUP_W-1:0] mac_oc_q6;
    integer p_i;

    // [PIPELINE] Banked pipeline (OC_PASSES>=1). A pixel = OC_PASSES work-items
    // (one per oc_group), all sharing one held window. Each work-item issues its
    // K_GROUPS into a bank; the next work-item issues into the idle bank while this
    // one drains + requants in the background. II -> ~OC_PASSES*K_GROUPS + overhead
    // (vs the serial OC_PASSES*(K_GROUPS + drain 5 + requant 3 + idle/sched)).
    // Data path + valid-chain depth (n_valid=6) UNCHANGED -> byte-exact.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out_r  <= 1'b0;
            data_out_r   <= 256'd0;
            k_group      <= 0;
            oc_group     <= 0;
            issuing      <= 1'b0;
            pending      <= 1'b0;
            pixel_active <= 1'b0;
            ib           <= 1'b0;
            bank_busy0   <= 1'b0;
            bank_busy1   <= 1'b0;
            rq_v1<=1'b0; rq_bank1<=1'b0; rq_oc1<=0;
            rq_v2<=1'b0; rq_oc2<=0;
            rq_v3<=1'b0; rq_oc3<=0;
            mac_valid_q1<=1'b0; mac_bank_q1<=1'b0; mac_last_q1<=1'b0; mac_oc_q1<=0;
            mac_valid_q2<=1'b0; mac_bank_q2<=1'b0; mac_last_q2<=1'b0; mac_oc_q2<=0;
            mac_valid_q3<=1'b0; mac_bank_q3<=1'b0; mac_last_q3<=1'b0; mac_oc_q3<=0;
            mac_valid_q4<=1'b0; mac_bank_q4<=1'b0; mac_last_q4<=1'b0; mac_oc_q4<=0;
            mac_valid_q5<=1'b0; mac_bank_q5<=1'b0; mac_last_q5<=1'b0; mac_oc_q5<=0;
            mac_valid_q6<=1'b0; mac_bank_q6<=1'b0; mac_last_q6<=1'b0; mac_oc_q6<=0;
            for (i = 0; i < MP; i = i + 1) begin
                acc_b0[i]     <= 0;
                acc_b1[i]     <= 0;
                biased[i]     <= 0;
                scaled[i]     <= 0;
                shift_lane[i] <= 0;
            end
        end else begin
            valid_out_r <= 1'b0;
            rq_v1       <= 1'b0;

            // latch a newly-fired output pixel (scheduler holds the window while busy).
            if (sched_output_fires) pending <= 1'b1;

            // ---- valid/bank/last/oc chain shift (depth = n_valid = 6) ----
            mac_valid_q2<=mac_valid_q1; mac_bank_q2<=mac_bank_q1; mac_last_q2<=mac_last_q1; mac_oc_q2<=mac_oc_q1;
            mac_valid_q3<=mac_valid_q2; mac_bank_q3<=mac_bank_q2; mac_last_q3<=mac_last_q2; mac_oc_q3<=mac_oc_q2;
            mac_valid_q4<=mac_valid_q3; mac_bank_q4<=mac_bank_q3; mac_last_q4<=mac_last_q3; mac_oc_q4<=mac_oc_q3;
            mac_valid_q5<=mac_valid_q4; mac_bank_q5<=mac_bank_q4; mac_last_q5<=mac_last_q4; mac_oc_q5<=mac_oc_q4;
            mac_valid_q6<=mac_valid_q5; mac_bank_q6<=mac_bank_q5; mac_last_q6<=mac_last_q5; mac_oc_q6<=mac_oc_q5;
            mac_valid_q1 <= 1'b0;   // default; issue re-asserts below

            // ---- banked accumulate on the last valid stage (routed by bank tag) ----
            if (mac_valid_q6) begin
                if (!mac_bank_q6) begin
                    for (p_i = 0; p_i < MP; p_i = p_i + 1)
                        acc_b0[p_i] <= acc_b0[p_i] + $signed(lane_partial[p_i]);
                end else begin
                    for (p_i = 0; p_i < MP; p_i = p_i + 1)
                        acc_b1[p_i] <= acc_b1[p_i] + $signed(lane_partial[p_i]);
                end
                if (mac_last_q6) begin   // last k_group accumulated -> bank complete
                    rq_v1    <= 1'b1;
                    rq_bank1 <= mac_bank_q6;
                    rq_oc1   <= mac_oc_q6;
                end
            end

            // ---- decoupled requant pipeline: BIAS -> SCALE -> OUTPUT (oc-indexed) ----
            if (rq_v1) begin
                for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1)
                    biased[lane_i] <= $signed(rq_bank1 ? acc_b1[lane_i] : acc_b0[lane_i])
                                      + $signed(biases_mem[rq_oc1 * MP + lane_i]);
                if (!rq_bank1) bank_busy0 <= 1'b0; else bank_busy1 <= 1'b0;  // acc consumed -> free
            end
            rq_v2 <= rq_v1; rq_oc2 <= rq_oc1;
            if (rq_v2) begin
                for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                    scaled[lane_i]     <= $signed(biased[lane_i]) *
                                          $signed(scale_mult_rom[rq_oc2 * MP + lane_i]);
                    shift_lane[lane_i] <= scale_shift_rom[rq_oc2 * MP + lane_i];
                end
            end
            rq_v3 <= rq_v2; rq_oc3 <= rq_oc2;
            if (rq_v3) begin
                for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                    out_oc = rq_oc3 * MP + lane_i;
                    // [INVARIANT:ROUNDING] single positive bias + arith >>> = golden floor.
                    v_tmp = (scaled[lane_i] +
                             ($signed({{(SCALED_W-1){1'b0}}, 1'b1}) <<< (shift_lane[lane_i] - 1))
                            ) >>> shift_lane[lane_i];
                    data_out_r[out_oc*8 +: 8] <=
                        (v_tmp >  127) ?  8'sd127 :
                        (v_tmp < -128) ? -8'sd128 : v_tmp[7:0];
                end
                if (rq_oc3 == OC_PASSES - 1) valid_out_r <= 1'b1;  // fire only after last oc_group
            end

            // ---- issue engine: one k_group/cycle; work-items chained via pixel_active ----
            if (issuing) begin
                mac_valid_q1 <= 1'b1;
                mac_bank_q1  <= ib;
                mac_oc_q1    <= oc_group;
                mac_last_q1  <= (k_group == K_GROUPS - 1);
                if (k_group == K_GROUPS - 1) begin
                    if (oc_group == OC_PASSES - 1) begin
                        issuing <= 1'b0; ib <= ~ib; pixel_active <= 1'b0;  // pixel done -> release window
                    end else if ((ib && !bank_busy0) || (!ib && !bank_busy1)) begin
                        // [PIPELINE] continue DIRECTLY into the other bank -- no inter-work-item bubble
                        oc_group <= oc_group + 1'b1;
                        ib       <= ~ib;
                        k_group  <= 0;
                        if (ib) begin   // next bank = ~ib = 0
                            bank_busy0 <= 1'b1;
                            for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) acc_b0[lane_i] <= 0;
                        end else begin  // next bank = ~ib = 1
                            bank_busy1 <= 1'b1;
                            for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) acc_b1[lane_i] <= 0;
                        end
                    end else begin
                        issuing <= 1'b0; ib <= ~ib;  // next bank busy -> fall back to pixel_active stall
                    end
                end else begin
                    k_group <= k_group + 1'b1;
                end
            end else if (pixel_active) begin
                // start the next work-item of the SAME pixel (oc_group+1) into bank ib.
                if ((!ib && !bank_busy0) || (ib && !bank_busy1)) begin
                    issuing  <= 1'b1;
                    oc_group <= oc_group + 1'b1;
                    k_group  <= 0;
                    if (!ib) begin
                        bank_busy0 <= 1'b1;
                        for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) acc_b0[lane_i] <= 0;
                    end else begin
                        bank_busy1 <= 1'b1;
                        for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) acc_b1[lane_i] <= 0;
                    end
                end
            end else if (pending) begin
                // start a new pixel: oc_group 0 into bank ib, when free.
                if ((!ib && !bank_busy0) || (ib && !bank_busy1)) begin
                    issuing      <= 1'b1;
                    pending      <= 1'b0;
                    pixel_active <= 1'b1;
                    oc_group     <= 0;
                    k_group      <= 0;
                    if (!ib) begin
                        bank_busy0 <= 1'b1;
                        for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) acc_b0[lane_i] <= 0;
                    end else begin
                        bank_busy1 <= 1'b1;
                        for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) acc_b1[lane_i] <= 0;
                    end
                end
            end
        end
    end

endmodule
