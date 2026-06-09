`timescale 1ns/1ps

// requant_pipeline.v
// ---------------------------------------------------------------------------
// Wave 2 task 08 sub-block. Port list is locked by
// docs/agent_tasks/00_engine_skeleton_spec_PORTS.md `## SUBBLOCK: requant_pipeline`.
// Spec:   docs/agent_tasks/08_engine_requant_pipeline.md
// Seed:   output/rtl/node_conv_288.v (canonical bias-add → scale-multiply →
//                                     round + arith-shift + INT8 saturate)
//
// 256 parallel requantisation lanes, output-channel-parallel.
//
//   stage 1 (clk + 1): biased[lane]   <= $signed(acc_in_lane) + $signed(bias_in_lane)
//   stage 2 (clk + 2): scaled[lane]   <= biased[lane] * $signed(scale_mult)
//   stage 3 (clk + 3): data_out[lane] <= int8_saturate(
//                          (scaled[lane] + sign_bias) >>> scale_shift)
//
// valid_out asserts exactly 3 cycles after valid_in (the assertion that
// matches the latency contract in 00_engine_skeleton_spec_PORTS.md
// §SUBBLOCK: requant_pipeline).
//
// Sign-aware rounding bias (knowledge/patterns/protected/01_context.md
// §"Scale-shift rounding — MANDATORY"):
//
//   sign_bias = scaled[MSB] ? (HALF - 1) : HALF    with HALF = 1 << (scale_shift - 1)
//
// For negatives the bias is (HALF - 1), NOT (-HALF) — Verilog `>>>` already
// floors toward -inf, so subtracting HALF would over-round (see the worked
// example in 01_context.md). The two rounding constants are precomputed as
// SCALED_W-wide signed wires so the surrounding ternary stays signed and
// `>>>` performs an arithmetic shift on the negative branch (the signedness
// footgun in 01_context.md §"Verilog signedness footgun on `(SCALE_ROUND_HALF
// - 1)`" is avoided by Option A — pre-computed signed constants).
//
// Bias delivery contract (locked by docs/agent_tasks/09_engine_address_generator.md
// §"Address granularity — bias memory layout (LOCKED)"): bias_in arrives as
// one wide word of 256 INT32 biases on the same cycle as the matching
// valid_in pulse. Lane `i` reads its bias from `bias_in[i*32 +: 32]`. No
// per-channel-byte addressing is performed in this module.
//
// Universal-bugs rule (knowledge/patterns/protected/08_common_bugs.md
// §"Array memory write in async-reset block") does NOT fire: every per-lane
// state register is a SCALAR `reg` declared inside its own generate block,
// not an indexed `reg [..] mem [..:..]` array. Vivado infers DFF per lane
// independently of the reset clause (same precedent as
// output/rtl/engine/mac_array.v).
//
// Bit-exact-against-existing-modules contract: the arithmetic
// (acc + bias) * scale_mult and the sign-aware-round + arith-shift + clamp
// match the active dram-backed-weights reference at output/rtl/node_conv_288.v
// (see ST_BIAS_SCALE/ST_PACK there). Register boundaries differ — the seed
// collapses BIAS and SCALE into one stage; this sub-block separates them
// for Fmax — but the final 8-bit output per lane is byte-identical.
// ---------------------------------------------------------------------------

module requant_pipeline (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                valid_in,
    input  wire [8191:0]       acc_in,
    input  wire [8191:0]       bias_in,
    // PER-OUTPUT-CHANNEL scale (INT4-GPTQ, Phase 2 2026-05-28). One 32-bit slot
    // per lane (256 lanes = 8192 bits), aligned with bias_in (read from the
    // engine scale ROM at the SAME address as bias). Per-lane slot layout:
    //   bits[15:0]  = SCALE_MULT (15-bit, compute_scale_approx)
    //   bits[21:16] = SCALE_SHIFT (<=23)
    // Replaces the former shared scale_mult[31:0]/scale_shift[5:0] (per-tensor),
    // which could not represent GPTQ per-channel scales (naive per-tensor INT4
    // collapses ResNet accuracy to ~0%). Timing is byte-exact to the old design:
    // mult used 1-deep (scaled_q2), shift used 2-deep (v_tmp) — see scale_q1/q2.
    input  wire [8191:0]       scale_in,
    output reg                 valid_out,
    output wire [2047:0]       data_out
);

    // ----------------------------------------------------------------------
    // Width derivation (task 13a fix 2: scale widened 16 -> 32).
    //   ACC_W    = 32    (fixed by acc_in port)
    //   BIAS_W   = 32    (fixed by bias_in port)
    //   BIASED_W = max(ACC_W, BIAS_W) + 1 = 33
    //   SCALE_W  = 32    (fixed by scale_mult port; signed; real ResNet-50
    //                     scale_mult values reach ~30 bits, so the prior
    //                     16-bit width silently truncated layer scales)
    //   SCALED_W = BIASED_W + SCALE_W = 65
    // ----------------------------------------------------------------------
    localparam integer ACC_W    = 32;
    localparam integer BIAS_W   = 32;
    localparam integer BIASED_W = 33;
    localparam integer SCALE_W  = 32;
    localparam integer SCALED_W = 65;

    // [FIT-FIX 2026-06-07] Constant-shift requant. The per-OC scale shift is folded OFFLINE
    // into a pre-widened multiplier (mult' = mult << (FIXED_SHIFT - shift), in scale.mem via
    // build_scale_memory_map.py), so this module applies a SINGLE compile-time arithmetic
    // shift instead of 256 per-lane VARIABLE 65-bit barrel shifters (~70K LUT removed; the
    // multiply is already DSP-mapped and the U250 has 91% idle DSP). FIXED_SHIFT must match
    // build_scale_memory_map.py and be >= any compute_scale_approx shift (range [0,23]).
    // ROUND_CONST = 2^(FIXED_SHIFT-1) is the unconditional +HALF the old round_half_lane gave.
    // Byte-exact: floor((biased*mult*2^(FS-shift)+2^(FS-1))/2^FS)==floor((biased*mult+2^(shift-1))/2^shift).
    localparam integer FIXED_SHIFT = 23;
    localparam signed [SCALED_W-1:0] ROUND_CONST =
        $signed({{(SCALED_W-1){1'b0}}, 1'b1}) <<< (FIXED_SHIFT - 1);

    // ----------------------------------------------------------------------
    // Pipelined valid bit and scale parameters. valid_in → valid_q1 →
    // valid_q2 → valid_out gives the locked 3-cycle latency. scale_mult and
    // scale_shift are held stable by the scheduler across a layer dispatch,
    // but pipelining them along with the data keeps each stage's combinational
    // cone bounded and lets the synthesis tool retime per-lane independently.
    // ----------------------------------------------------------------------
    reg                      valid_q1;
    reg                      valid_q2;
    reg                      valid_q3;   // Lever 2 (2026-05-26): added q3 stage
    // Per-OC scale pipelined alongside the data. scale_q1 (1-deep) feeds the
    // per-lane multiply (matches old scale_mult_q1 timing); scale_q2 (2-deep)
    // feeds the per-lane round+shift (matches old scale_shift_q2 timing).
    reg [8191:0]             scale_q1;
    reg [8191:0]             scale_q2;
    // 2026-05-26 first-light slack diag: every worst-slack path in the routed
    // design (40 ns @ slow corner) had Source=scale_shift_q2_reg[1]/C with
    // route delay 29.367 ns (97.965% of 29.977 ns total path), logic delay
    // 0.610 ns, and net fanout 23217. Single register driving every one of
    // the 256 OC lanes' dynamic right-shift plus the shared round_half /
    // round_half_m1 / scale_shift_zero computation, all on long wires.
    //
    // Fix: split the q2 register and the derived control cone across
    // N_GROUPS local replicas (see g_ctrl generate block below). Soft hints
    // like (* max_fanout *) were insufficient because they only replicate
    // the source register; Vivado may still build a SHARED round_half cone
    // downstream and recreate the high-fanout broadcast on derived signals.
    // Explicit generate-instance replication forces Vivado to keep each
    // group's control logic distinct and locally placed.
    //
    // All N_GROUPS replicas latch the same scale_shift_q1 input so they
    // hold identical values cycle-by-cycle — byte-exact equivalent to the
    // pre-split single-register design. Engine TB goldens unaffected.

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_q1       <= 1'b0;
            valid_q2       <= 1'b0;
            valid_q3       <= 1'b0;
            valid_out      <= 1'b0;
        end else begin
            valid_q1       <= valid_in;
            valid_q2       <= valid_q1;
            valid_q3       <= valid_q2;    // Lever 2: extra stage
            valid_out      <= valid_q3;    // Lever 2: shifted from valid_q2
        end
    end

    // [K1-FDCE] scale_q1/scale_q2 are DATAPATH pipes (2 x 8192 FF): their
    // values reach data_out only through the per-lane pipeline, which the
    // engine samples strictly under valid_out (reset-gated above), and they
    // are rewritten every cycle -> the reset value is dead. No-reset => FDRE.
    always @(posedge clk) begin
        scale_q1       <= scale_in;
        scale_q2       <= scale_q1;
    end

    // ----------------------------------------------------------------------
    // Per-group control replicas (8 groups × 32 lanes = 256 lanes total).
    // Each group has its own scale_shift_q2_local register PLUS its own
    // round_half / round_half_m1 / scale_shift_zero combinational logic.
    // Vivado treats each generate instance as a distinct entity and will
    // not share their internal cones, so the high-fanout broadcast cannot
    // re-form on the derived signals. The `keep_hierarchy` attribute pins
    // the boundary so downstream opt passes don't dissolve it.
    //
    // 13a audit fix preserved: scale_shift_q2 - 1 underflows to 63 when
    // scale_shift_q2 = 0, so the rounding bias is clamped to 0 in that
    // case (see scale_shift_zero_local guard).
    // ----------------------------------------------------------------------
    // [Phase 2 per-OC] The former per-GROUP scale_shift replicas (g_ctrl) are
    // gone: per-OC requant gives each lane its OWN shift from scale_q2, so there
    // is no shared high-fanout scale_shift to replicate. Each lane derives its
    // round_half / round_half_m1 locally from its own shift (below).

    // ----------------------------------------------------------------------
    // 256 parallel lanes. Each lane is structurally identical and uses only
    // SCALAR per-lane registers (no indexed arrays inside the always block,
    // so the activation_memory_in_async_reset_block preflight rule does not
    // apply — same precedent as output/rtl/engine/mac_array.v).
    // ----------------------------------------------------------------------
    genvar lane;
    generate
        for (lane = 0; lane < 256; lane = lane + 1) begin : g_lane
            // Each lane reads its scale_shift / round_half / round_half_m1
            // from the LOCAL replica owned by its group (32 lanes per group).
            // This breaks the 23K-fanout single-register broadcast by
            // construction: each replica only drives ~32 lanes' worth of
            // shift+round logic.
            // Per-OC scale for THIS lane. mult from scale_q1 (1-deep, aligns
            // with the multiply); shift from scale_q2 (2-deep, aligns with the
            // round+shift). Slot layout: [15:0]=mult (positive 15-bit, zero-
            // extended to signed SCALE_W), [21:16]=shift.
            // [FIT-FIX 2026-06-07] mult_lane is the PRE-WIDENED multiplier
            // mult' = mult << (FIXED_SHIFT - shift) (low 31 bits of the slot, always
            // positive, < 2^31). The per-OC VARIABLE shift + variable round generator
            // are GONE -- replaced by the module-level constant ROUND_CONST + a
            // compile-time >>> FIXED_SHIFT below. scale_q2 is now unused (pruned).
            wire signed [SCALE_W-1:0] mult_lane =
                $signed({1'b0, scale_q1[lane*32 +: 31]});

            wire signed [ACC_W-1:0]    acc_lane;
            wire signed [BIAS_W-1:0]   bias_lane;
            reg  signed [BIASED_W-1:0] biased_q1;
            (* use_dsp = "yes" *)
            reg  signed [SCALED_W-1:0] scaled_q2;
            wire signed [SCALED_W-1:0] biased_round_sum;
            wire signed [SCALED_W-1:0] v_tmp;
            // Lever 2 (2026-05-26): split stage 3 into q3a + q4 to break the
            // shift+clamp combinational cone. Stage q3a registers the
            // saturation decisions and the low byte (10 bits total per lane,
            // not the full 65-bit shifted value — see Other AI's refinement).
            // Stage q4 (renamed from old q3) computes the clamp mux from the
            // already-registered q3a signals. Engine latency: 3 -> 4 cycles.
            // Output values unchanged (byte-exact), just emitted one cycle
            // later than before.
            reg                        sat_hi_q3a;
            reg                        sat_lo_q3a;
            reg  signed [7:0]          v_low_q3a;
            reg  signed [7:0]          data_out_q4;

            assign acc_lane  = $signed(acc_in [lane*32 +: 32]);
            assign bias_lane = $signed(bias_in[lane*32 +: 32]);

            // Stage-3a combinational cone: sign-aware add then arith-shift.
            // [INVARIANT:ROUNDING]
            //
            // Investigated 2026-05-24: tried replacing this with unconditional
            // +HALF to match scripts/golden_impl.py:round_half_up_toward_pos_inf,
            // hypothesizing the 2 off-by-one mismatches caught by
            // engine_one_layer_tb on node_conv_246 (pixel[1,4] ch124 & ch238)
            // were sign-aware-rounding ties. Re-ran TB after the change:
            // identical 2 mismatches. So this is NOT a rounding-tie bug —
            // it's a MAC-path / weight-slot / accumulator-order discrepancy
            // for those 2 specific output channels at that pixel. Kept the
            // original sign-aware bias since the actual fix lives elsewhere.
            // [ROUNDING — per-OC golden alignment 2026-05-29] Unconditional +HALF
            // (round-half-up toward +inf), matching scripts/golden_impl.py
            // requantize_tensor_with_scale_per_oc AND the byte-exact spatial datapath
            // (conv_datapath_mp_k). The OLD sign-aware bias (HALF for >=0, HALF-1 for <0)
            // rounds negatives half-toward-zero; the conv_282 engine sweep shows its
            // ±1 are ALL negative values with got = gold-1 (round-half-DOWN) — the exact
            // signature of HALF-1. +HALF fixes it. (The 2026-05-24 "+HALF no help" note
            // was for conv_246's DIFFERENT, since-fixed MAC-path ±1.)
            assign biased_round_sum = scaled_q2 + ROUND_CONST;
            assign v_tmp            = biased_round_sum >>> FIXED_SHIFT;

            // [K1-FDCE] per-lane requant pipe (116 FF x 256 lanes): pure
            // feed-forward datapath; data_out is sampled only under valid_out
            // (reset-gated valid chain above). Reset clause removed -> FDRE.
            always @(posedge clk) begin
                begin
                    // Stage 1: bias-add (signed-signed add; BIASED_W absorbs
                    // the sign bit).
                    biased_q1 <= acc_lane + bias_lane;

                    // Stage 2: scale-multiply (registered into a DSP-mapped
                    // SCALED_W-wide signed product). Per-OC mult for this lane.
                    scaled_q2 <= biased_q1 * mult_lane;

                    // Stage 3a (Lever 2): compute saturation flags + low byte
                    // from the FULL 65-bit v_tmp, register 10 bits per lane.
                    // Comparisons use bare 127 / -128 literals (matches the
                    // seed in node_conv_288.v ST_PACK byte-for-byte).
                    sat_hi_q3a <= (v_tmp >  127);
                    sat_lo_q3a <= (v_tmp < -128);
                    v_low_q3a  <= v_tmp[7:0];

                    // Stage 4 (Lever 2): final clamp mux from already-registered
                    // q3a signals. This cone is pure 3-input mux + 8-bit
                    // assign — minimal LUT depth, fast.
                    data_out_q4 <= sat_hi_q3a ? 8'sd127 :
                                   sat_lo_q3a ? -8'sd128 :
                                                 v_low_q3a;
                end
            end

            assign data_out[lane*8 +: 8] = data_out_q4;
        end
    endgenerate

endmodule
