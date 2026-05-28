// conv_datapath_mp_k — parallel datapath with BOTH MP lane parallelism AND
// MP_K kernel parallelism. Per cycle: MP × MP_K multipliers compute MP × MP_K
// products. Per lane, MP_K products are tree-summed into one partial sum and
// added to that lane's accumulator. Total cycles per OC pass: K_TOTAL/MP_K + 5.
//
// Constraint: K_TOTAL must be divisible by MP_K.
//
// Weight layout (very wide):
//   weights_wide[oc_group * K_GROUPS + k_group] is MP * MP_K * 8 bits wide.
//   Bits [(lane * MP_K + kpos) * 8 +: 8] = weight at (oc=g*MP+lane,
//   k=k_group*MP_K+kpos).
//   scripts/repack_weights_wide.py with --mp-k=N produces this layout.
//
// Speedup vs MP-only parallel (which is MP*K_TOTAL+6 cycles per OC pass):
//   With MP_K: MP * K_TOTAL/MP_K + 6 cycles ≈ K_TOTAL/MP_K + 5 (compared to
//   K_TOTAL + 5 for MP_K=1). For 3x3 conv with MP_K=9: 9× speedup on the
//   inner-product depth.
//
// DSP cost: MP * MP_K multipliers per module (vs MP for MP_K=1).
//   conv_200 (MP=4, MP_K=9): 36 DSPs (vs 4). Across ~17 slow modules with
//   MP_K=9: +540 DSPs out of U250's 12288 budget = 4.4% added utilization.
//
// Byte-exactness: same products as serialized MAC, just reordered into a
// tree-sum per lane. Integer addition is associative; the final acc value
// per lane is identical (modulo carry; ACC_W has sufficient headroom).

module conv_datapath_mp_k #(
    parameter integer IC          = 64,
    parameter integer OC          = 64,
    parameter integer KH          = 3,
    parameter integer KW          = 3,
    parameter integer K_TOTAL     = IC * KH * KW,
    parameter integer MP          = 4,
    parameter integer MP_K        = 1,    // kernel-parallelism (1 = no K-parallel)
    parameter integer OC_PASSES   = (OC + MP - 1) / MP,
    parameter integer SCALE_MULT  = 1,    // legacy per-tensor (unused when SCALE_PATH set)
    parameter integer SCALE_SHIFT = 16,   // legacy per-tensor (unused when SCALE_PATH set)
    parameter         WEIGHTS_PATH = "",  // MP×MP_K-byte-packed .hex
    parameter         BIAS_PATH    = "",
    // Phase 2 INT4-GPTQ: per-OUTPUT-CHANNEL requant scale. When non-empty, one
    // 32-bit hex entry per OC: bits[15:0]=mult (15-bit compute_scale_approx),
    // bits[21:16]=shift. Overrides the per-tensor SCALE_MULT/SCALE_SHIFT.
    parameter         SCALE_PATH   = ""
) (
    input  wire                               clk,
    input  wire                               rst_n,

    input  wire [KH*KW*IC*8-1:0]              window_flat,
    input  wire                               start_mac,

    output reg                                valid_out,
    output reg  [OC*8-1:0]                    data_out,
    output wire                               mac_busy
);

    // ---------------- Derived widths ----------------------------------
    localparam integer K_GROUPS         = K_TOTAL / MP_K;  // assumes K_TOTAL % MP_K == 0
    localparam integer WIDE_W           = MP * MP_K * 8;   // bits per weight entry
    localparam integer PROD_W           = 16;
    localparam integer TREE_W           = PROD_W + ((MP_K <= 1) ? 0 : $clog2(MP_K));
    localparam integer ACC_W            = TREE_W + ((K_GROUPS <= 1) ? 0 : $clog2(K_GROUPS));
    localparam integer BIAS_W           = 32;
    localparam integer BIASED_W         = ((ACC_W > BIAS_W) ? ACC_W : BIAS_W) + 1;
    // Per-OC runtime mult is 15-bit (compute_scale_approx) -> signed 16-bit holds
    // it positive. Fixed width (not derived from the compile-time SCALE_MULT) so
    // SCALED_W is the same for per-OC and legacy per-tensor.
    localparam integer SCALE_CONST_W    = 16;
    localparam integer SCALED_W         = BIASED_W + SCALE_CONST_W;
    localparam integer NUM_WIDE_WORDS   = OC_PASSES * K_GROUPS;
    localparam integer KGROUP_COUNTER_W = (K_GROUPS <= 1) ? 1 : $clog2(K_GROUPS);
    localparam integer OC_GROUP_W       = (OC_PASSES <= 1) ? 1 : $clog2(OC_PASSES);
    localparam integer OC_INDEX_W       = (OC + MP <= 1) ? 1 : $clog2(OC + MP);
    // Per-OC scale removed the compile-time SCALE_MULT_CONST / SCALE_ROUND_BIAS;
    // each output channel's (mult,shift) now comes from scale_rom (below), and
    // the round bias is computed per-OC at the OUTPUT stage.

    // ---------------- FSM states --------------------------------------
    localparam ST_IDLE   = 3'd0;
    localparam ST_MAC    = 3'd1;
    localparam ST_BIAS   = 3'd2;
    localparam ST_SCALE  = 3'd3;
    localparam ST_OUTPUT = 3'd4;

    reg [2:0] state;

    // ---------------- Wide weight ROM + bias --------------------------
    (* rom_style = "block", ram_style = "block" *) reg [WIDE_W-1:0] weights_wide [0:NUM_WIDE_WORDS-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [31:0] biases [0:OC-1];
    // Per-OC requant scale: one 32-bit slot/OC {shift[21:16], mult[15:0]}.
    reg [31:0] scale_rom [0:OC-1];
    integer init_oc;
    initial begin
        if (WEIGHTS_PATH != "") $readmemh(WEIGHTS_PATH, weights_wide);
        if (BIAS_PATH    != "") $readmemh(BIAS_PATH,    biases);
        if (SCALE_PATH   != "") $readmemh(SCALE_PATH,   scale_rom);
        else // legacy per-tensor fallback: same (mult,shift) for every OC.
            for (init_oc = 0; init_oc < OC; init_oc = init_oc + 1)
                scale_rom[init_oc] = {10'd0, SCALE_SHIFT[5:0], SCALE_MULT[15:0]};
    end

    // ---------------- MAC pipeline registers --------------------------
    reg signed [ACC_W-1:0]    acc    [0:MP-1];
    reg signed [BIASED_W-1:0] biased [0:MP-1];
    reg signed [SCALED_W-1:0] scaled [0:MP-1];
    reg signed [SCALED_W-1:0] v_tmp;
    reg        [5:0]          out_shift;   // per-OC shift (OUTPUT stage)
    reg signed [SCALED_W-1:0] out_round;   // per-OC round bias (OUTPUT stage)
`ifdef DBG_SCALE
    integer dbg_n; initial dbg_n = 0;
`endif

    reg [KGROUP_COUNTER_W-1:0] k_group;
    reg [OC_GROUP_W-1:0]       oc_group;

    // Separate per-block loop variables. Sharing one `integer i` across
    // multiple always blocks creates a race because all blocks fire on the
    // same posedge clk and the inner for-loops mutate the shared variable.
    // Symptom: tap_q[i] is loaded with a tap_at(k_group * MP_K + i) where
    // `i` was clobbered to MP-1 by the FSM block — so every tap_q[i] gets
    // the same tap value, breaking equivalence for non-uniform inputs.
    integer ld_i;       // loop var for weight/tap load block
    integer cs_lane_i;  // loop var for combinational sum_lane_w
    integer cs_kpos;    // loop var for combinational sum_lane_w
    integer fsm_i;      // loop var for FSM block (reset, stage 2/3, OC pass reset)
    integer fsm_lane_i; // loop var for FSM block (bias/scale/output stages)
    integer bias_oc;
    integer out_oc;
    integer sc_oc;

    assign mac_busy = (state != ST_IDLE);

    wire [$clog2(NUM_WIDE_WORDS+1)-1:0] weight_read_addr = oc_group * K_GROUPS + k_group;

    // Window-tap indexer: same as conv_datapath_parallel. Linear k index k_lin
    // maps to (kh, kw, ic). Function called MP_K times per cycle for k positions
    // [k_group*MP_K, k_group*MP_K+MP_K-1].
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

    // ---- Stage 1: register weight word + MP_K taps ----
    reg [WIDE_W-1:0]    weight_word_q;
    reg signed [7:0]    tap_q [0:MP_K-1];
    always @(posedge clk) begin
        weight_word_q <= weights_wide[weight_read_addr];
        for (ld_i = 0; ld_i < MP_K; ld_i = ld_i + 1)
            tap_q[ld_i] <= $signed(tap_at(k_group * MP_K + ld_i));
    end

    // ---- Stage 2: MP × MP_K parallel multipliers, tree-sum per lane ----
    // partial_q[lane] = sum over kpos of (weight_word_q[lane,kpos] * tap_q[kpos]).
    // Verilog tools infer a DSP per multiplier; the tree sum collapses to an
    // adder cascade that Vivado retimes onto the DSPs' P-cascade where it can.
    (* use_dsp = "yes" *) reg signed [TREE_W-1:0] partial_q [0:MP-1];
    reg [OC_GROUP_W-1:0]       mac_oc_group_q1;
    reg                        mac_valid_q1;
    reg                        mac_valid_q2;
    reg [OC_GROUP_W-1:0]       mac_oc_group_q2;
    reg                        mac_done_issuing;

    // Compute MP × MP_K products combinationally, sum per lane.
    //
    // CRITICAL: Do NOT wrap the multiplication in an outer $signed() — that
    // makes the multiplication self-determined (width = max of operands =
    // 8 bits), truncating the 16-bit product before sign-extending it. The
    // bug shows up as (-2)*(-113) = -30 (= 226 mod 256) instead of 226,
    // breaking equivalence for any non-uniform input. We rely on:
    //   - signed-typed prod_w (per-product) for the multiplication's LHS
    //     context, so the multiplication is computed at signed 16 bits.
    //   - signed sum_w accumulator for the sum context.
    reg signed [TREE_W-1:0] sum_lane_w [0:MP-1];
    reg signed [PROD_W-1:0] prod_w;
    always @* begin
        for (cs_lane_i = 0; cs_lane_i < MP; cs_lane_i = cs_lane_i + 1) begin
            sum_lane_w[cs_lane_i] = {TREE_W{1'b0}};
            for (cs_kpos = 0; cs_kpos < MP_K; cs_kpos = cs_kpos + 1) begin
                prod_w = $signed(weight_word_q[(cs_lane_i * MP_K + cs_kpos) * 8 +: 8]) *
                         $signed(tap_q[cs_kpos]);
                sum_lane_w[cs_lane_i] = sum_lane_w[cs_lane_i] + prod_w;
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= ST_IDLE;
            valid_out        <= 1'b0;
            data_out         <= {OC*8{1'b0}};
            k_group          <= 0;
            oc_group         <= 0;
            mac_valid_q1     <= 1'b0;
            mac_valid_q2     <= 1'b0;
            mac_oc_group_q1  <= 0;
            mac_oc_group_q2  <= 0;
            mac_done_issuing <= 1'b0;
            for (fsm_i = 0; fsm_i < MP; fsm_i = fsm_i + 1) begin
                acc[fsm_i]      <= 0;
                biased[fsm_i]   <= 0;
                scaled[fsm_i]   <= 0;
                partial_q[fsm_i] <= 0;
            end
        end else begin
            valid_out <= 1'b0;

            // Stage 2: register the MP partial sums.
            for (fsm_i = 0; fsm_i < MP; fsm_i = fsm_i + 1)
                partial_q[fsm_i] <= sum_lane_w[fsm_i];
            mac_valid_q2     <= mac_valid_q1;
            mac_oc_group_q2  <= mac_oc_group_q1;

            // Stage 3: accumulate partial sums into MP lanes.
            if (mac_valid_q2) begin
                for (fsm_i = 0; fsm_i < MP; fsm_i = fsm_i + 1) begin
                    if (mac_oc_group_q2 * MP + fsm_i < OC)
                        acc[fsm_i] <= acc[fsm_i] + $signed(partial_q[fsm_i]);
                end
            end

            case (state)
                ST_IDLE: begin
                    if (start_mac) begin
                        state            <= ST_MAC;
                        k_group          <= 0;
                        oc_group         <= 0;
                        mac_valid_q1     <= 1'b0;
                        mac_valid_q2     <= 1'b0;
                        mac_done_issuing <= 1'b0;
                        for (fsm_lane_i = 0; fsm_lane_i < MP; fsm_lane_i = fsm_lane_i + 1)
                            acc[fsm_lane_i] <= 0;
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
                        mac_oc_group_q1 <= oc_group;
                        mac_valid_q1    <= 1'b1;
                        if (k_group == K_GROUPS - 1) begin
                            mac_done_issuing <= 1'b1;
                        end else begin
                            k_group <= k_group + 1'b1;
                        end
                    end
                end

                ST_BIAS: begin
                    for (fsm_lane_i = 0; fsm_lane_i < MP; fsm_lane_i = fsm_lane_i + 1) begin
                        bias_oc = oc_group * MP + fsm_lane_i;
                        if (bias_oc < OC)
                            biased[fsm_lane_i] <= $signed(acc[fsm_lane_i]) + $signed(biases[bias_oc]);
                        else
                            biased[fsm_lane_i] <= 0;
                    end
                    state <= ST_SCALE;
                end

                ST_SCALE: begin
                    for (fsm_lane_i = 0; fsm_lane_i < MP; fsm_lane_i = fsm_lane_i + 1) begin
                        sc_oc = oc_group * MP + fsm_lane_i;
                        // Per-OC mult (positive 15-bit in a 16-bit slot -> signed
                        // positive). out-of-range lanes don't matter (OUTPUT guards).
                        if (sc_oc < OC)
                            scaled[fsm_lane_i] <= $signed(biased[fsm_lane_i]) *
                                                  $signed(scale_rom[sc_oc][15:0]);
                        else
                            scaled[fsm_lane_i] <= 0;
                    end
                    state <= ST_OUTPUT;
                end

                ST_OUTPUT: begin
                    for (fsm_lane_i = 0; fsm_lane_i < MP; fsm_lane_i = fsm_lane_i + 1) begin
                        out_oc = oc_group * MP + fsm_lane_i;
                        if (out_oc < OC) begin
                            // Per-OC shift + round bias (shift==0 -> no rounding).
                            out_shift = scale_rom[out_oc][21:16];
                            out_round = (out_shift == 6'd0) ? {SCALED_W{1'b0}}
                                      : ({{(SCALED_W-1){1'b0}}, 1'b1} <<< (out_shift - 6'd1));
                            v_tmp = (scaled[fsm_lane_i] + out_round) >>> out_shift;
                            data_out[out_oc*8 +: 8] <=
                                (v_tmp >  127) ?  8'sd127 :
                                (v_tmp < -128) ? -8'sd128 : v_tmp[7:0];
`ifdef DBG_SCALE
                            if ((out_oc == 0 || out_oc == 32) && dbg_n < 8) begin
                                $display("[DBG_SCALE] oc=%0d biased=%0d scale_rom=%h mult=%0d shift=%0d out_round=%0d scaled=%0d v_tmp=%0d -> out=%0d",
                                    out_oc, $signed(biased[fsm_lane_i]), scale_rom[out_oc],
                                    $signed({1'b0,scale_rom[out_oc][15:0]}), out_shift, $signed(out_round),
                                    $signed(scaled[fsm_lane_i]), $signed(v_tmp),
                                    $signed((v_tmp>127)?8'sd127:(v_tmp<-128)?-8'sd128:v_tmp[7:0]));
                                dbg_n = dbg_n + 1;
                            end
`endif
                        end
                    end

                    if (oc_group == OC_PASSES - 1) begin
                        valid_out <= 1'b1;
                        state     <= ST_IDLE;
                    end else begin
                        oc_group     <= oc_group + 1'b1;
                        k_group      <= 0;
                        for (fsm_lane_i = 0; fsm_lane_i < MP; fsm_lane_i = fsm_lane_i + 1)
                            acc[fsm_lane_i] <= 0;
                        state <= ST_MAC;
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
