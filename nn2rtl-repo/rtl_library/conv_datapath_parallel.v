// conv_datapath_parallel - parallel-MAC variant of conv_datapath.
//
// Per cycle in ST_MAC: ONE weight-word read (containing MP packed 8-bit
// weights), ONE tap read, MP parallel multipliers, MP parallel accumulator
// writes. The lane_counter is eliminated; k_counter advances every cycle.
//
// Per OC-group pass cost: K_TOTAL issue cycles + 2 trailing drain cycles +
// ST_BIAS + ST_SCALE + ST_OUTPUT = K_TOTAL + 6.
//
// Speedup vs original conv_datapath: MP×. With MP=8 (typical), this drops
// per-pixel cycles from MP*K_TOTAL+6 to K_TOTAL+6. Same accumulator depth,
// same scaling/clamping logic → numerically byte-exact.
//
// Weight layout (WIDE):
//   weights_wide[oc_group * K_TOTAL + k] is a MP*8-bit packed word holding
//   weights for {OCs g*MP+0 .. g*MP+MP-1} at kernel index k.
//   This is a re-pack of the original flat (oc * K_TOTAL + k) layout;
//   scripts/repack_weights_wide.py produces the .hex file for $readmemh.
//
// DSP cost: MP multipliers per spatial conv (the (* use_dsp *) hint on the
// per-lane mul_q registers maps each to a DSP48E2). Previous version used
// 1 multiplier per module; this version uses MP. Trade-off: ~MP× DSP for
// ~MP× throughput. With MP=8 and 17 slow spatial convs, ~136 extra DSPs
// out of U250's 12288 budget (1.1% added utilization).
//
// Behavior is byte-exact equivalent to conv_datapath.v when the weights
// are correctly re-packed — same products in same accumulator, just done
// in parallel. Integer addition is associative; the order in which the
// MP products are added into separate accumulator lanes does not change
// the final per-lane sum (each lane accumulates independently).

module conv_datapath_parallel #(
    parameter integer IC          = 64,
    parameter integer OC          = 64,
    parameter integer KH          = 3,
    parameter integer KW          = 3,
    parameter integer K_TOTAL     = IC * KH * KW,
    parameter integer MP          = 4,
    parameter integer OC_PASSES   = (OC + MP - 1) / MP,
    parameter integer SCALE_MULT  = 1,    // legacy per-tensor (unused when SCALE_PATH set)
    parameter integer SCALE_SHIFT = 16,   // legacy per-tensor (unused when SCALE_PATH set)
    parameter         WEIGHTS_PATH = "",   // wide-packed .hex (OC_PASSES*K_TOTAL lines, MP bytes each)
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
    localparam integer PROD_W          = 16;
    localparam integer ACC_W           = PROD_W + $clog2(K_TOTAL);
    localparam integer BIAS_W          = 32;
    localparam integer BIASED_W        = ((ACC_W > BIAS_W) ? ACC_W : BIAS_W) + 1;
    // Per-OC runtime mult is 15-bit (compute_scale_approx) -> signed 16-bit holds
    // it positive. Fixed width (not derived from the compile-time SCALE_MULT) so
    // SCALED_W is the same for per-OC and legacy per-tensor.
    localparam integer SCALE_CONST_W   = 16;
    localparam integer SCALED_W        = BIASED_W + SCALE_CONST_W;
    localparam integer NUM_WIDE_WORDS  = OC_PASSES * K_TOTAL;
    localparam integer WIDE_ADDR_W     = (NUM_WIDE_WORDS <= 1) ? 1 : $clog2(NUM_WIDE_WORDS);
    localparam integer K_COUNTER_W     = (K_TOTAL <= 1) ? 1 : $clog2(K_TOTAL);
    localparam integer OC_GROUP_W      = (OC_PASSES <= 1) ? 1 : $clog2(OC_PASSES);
    localparam integer OC_INDEX_W      = (OC + MP <= 1) ? 1 : $clog2(OC + MP);
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
    // weights_wide[oc_group*K_TOTAL + k] contains MP packed 8-bit weights
    // for lanes 0..MP-1 at kernel index k of the current oc_group.
    (* rom_style = "block", ram_style = "block" *) reg [MP*8-1:0] weights_wide [0:NUM_WIDE_WORDS-1];
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

    reg [K_COUNTER_W-1:0]    k_counter;
    reg [OC_GROUP_W-1:0]     oc_group;

    integer i, lane_i;
    integer bias_oc;
    integer out_oc;
    integer sc_oc;

    assign mac_busy = (state != ST_IDLE);

    wire [WIDE_ADDR_W-1:0] weight_read_addr = oc_group * K_TOTAL + k_counter;

    // Window-tap indexer: kernel-index k -> flat window byte. Same as the
    // original conv_datapath (kept identical for byte-exact equivalence).
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

    // Stage 1: register wide weight word + single tap.
    reg [MP*8-1:0] weight_word_q;
    reg signed [7:0] tap_q;
    always @(posedge clk) begin
        weight_word_q <= weights_wide[weight_read_addr];
        tap_q         <= $signed(tap_at(k_counter));
    end

    // Pipeline metadata for MP parallel MACs.
    reg                        mac_valid_q1;
    reg                        mac_valid_q2;
    reg [OC_GROUP_W-1:0]       mac_oc_group_q1;
    reg [OC_GROUP_W-1:0]       mac_oc_group_q2;
    reg                        mac_done_issuing;

    // MP parallel multipliers (one DSP per lane).
    // (* use_dsp = "yes" *) maps each registered product to a DSP48E2 with
    // MREG=1. The MP-way pattern lets Vivado pack MP DSPs into a column.
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] mul_q [0:MP-1];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= ST_IDLE;
            valid_out        <= 1'b0;
            data_out         <= {OC*8{1'b0}};
            k_counter        <= 0;
            oc_group         <= 0;
            mac_valid_q1     <= 1'b0;
            mac_valid_q2     <= 1'b0;
            mac_oc_group_q1  <= 0;
            mac_oc_group_q2  <= 0;
            mac_done_issuing <= 1'b0;
            for (i = 0; i < MP; i = i + 1) begin
                acc[i]    <= 0;
                biased[i] <= 0;
                scaled[i] <= 0;
                mul_q[i]  <= 0;
            end
        end else begin
            valid_out <= 1'b0;

            // Stage 2: register MP products from MP parallel multipliers.
            // Each lane multiplies its slice of weight_word_q with the
            // shared tap_q. The slice is byte i (lane i): bits [i*8 +: 8].
            for (i = 0; i < MP; i = i + 1) begin
                mul_q[i] <= $signed(weight_word_q[i*8 +: 8]) * $signed(tap_q);
            end
            mac_valid_q2     <= mac_valid_q1;
            mac_oc_group_q2  <= mac_oc_group_q1;

            // Stage 3: accumulate MP products into MP separate lanes in
            // parallel. acc[i] += mul_q[i] for each lane i where the
            // global OC (oc_group*MP + i) is within OC.
            if (mac_valid_q2) begin
                for (i = 0; i < MP; i = i + 1) begin
                    if (mac_oc_group_q2 * MP + i < OC)
                        acc[i] <= acc[i] + $signed(mul_q[i]);
                end
            end

            case (state)
                ST_IDLE: begin
                    if (start_mac) begin
                        state            <= ST_MAC;
                        k_counter        <= 0;
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
                        // Stop issuing. Wait two more cycles for stages 2
                        // and 3 to drain. State transitions when both
                        // mac_valid_q1 and mac_valid_q2 are 0.
                        mac_valid_q1 <= 1'b0;
                        if (!mac_valid_q1 && !mac_valid_q2) begin
                            mac_done_issuing <= 1'b0;
                            state            <= ST_BIAS;
                        end
                    end else begin
                        mac_oc_group_q1 <= oc_group;
                        mac_valid_q1    <= 1'b1;

                        // k_counter advances EVERY cycle. No inner lane loop.
                        if (k_counter == K_TOTAL - 1) begin
                            mac_done_issuing <= 1'b1;
                        end else begin
                            k_counter <= k_counter + 1'b1;
                        end
                    end
                end

                ST_BIAS: begin
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                        bias_oc = oc_group * MP + lane_i;
                        if (bias_oc < OC)
                            biased[lane_i] <= $signed(acc[lane_i]) + $signed(biases[bias_oc]);
                        else
                            biased[lane_i] <= 0;
                    end
                    state <= ST_SCALE;
                end

                ST_SCALE: begin
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                        sc_oc = oc_group * MP + lane_i;
                        // Per-OC mult (positive 15-bit in a 16-bit slot -> signed
                        // positive). out-of-range lanes don't matter (OUTPUT guards).
                        if (sc_oc < OC)
                            scaled[lane_i] <= $signed(biased[lane_i]) *
                                              $signed(scale_rom[sc_oc][15:0]);
                        else
                            scaled[lane_i] <= 0;
                    end
                    state <= ST_OUTPUT;
                end

                ST_OUTPUT: begin
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                        out_oc = oc_group * MP + lane_i;
                        if (out_oc < OC) begin
                            // [INVARIANT:ROUNDING] per-OC shift + round bias
                            // (shift==0 -> no rounding).
                            out_shift = scale_rom[out_oc][21:16];
                            out_round = (out_shift == 6'd0) ? {SCALED_W{1'b0}}
                                      : ({{(SCALED_W-1){1'b0}}, 1'b1} <<< (out_shift - 6'd1));
                            v_tmp = (scaled[lane_i] + out_round) >>> out_shift;
                            data_out[out_oc*8 +: 8] <=
                                (v_tmp >  127) ?  8'sd127 :
                                (v_tmp < -128) ? -8'sd128 : v_tmp[7:0];
                        end
                    end

                    if (oc_group == OC_PASSES - 1) begin
                        // [INVARIANT:VALID_OUT_LATENCY]
                        valid_out <= 1'b1;
                        state     <= ST_IDLE;
                    end else begin
                        oc_group     <= oc_group + 1'b1;
                        k_counter    <= 0;
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
