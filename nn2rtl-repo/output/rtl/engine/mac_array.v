`timescale 1ns/1ps

// mac_array.v
// --------------------------------------------------------------------------
// Wave 2 task 07 sub-block. Port list is locked by
// docs/agent_tasks/00_engine_skeleton_spec_PORTS.md `## SUBBLOCK: mac_array`.
// Spec:  docs/agent_tasks/07_engine_mac_array.md
//
// 256 parallel signed-INT8 multiply-accumulate lanes, output-channel-parallel.
// Every cycle that mac_valid_in is high:
//   stage 1 (clk + 1): mul_q1[lane] <= act_byte * weight_bus[lane]
//   stage 2 (clk + 2): if mac_valid_q1 then acc[lane] <= acc[lane] + mul_q1[lane]
//
// So acc_out[lane] becomes final 2 cycles after the last mac_valid_in pulse
// of the current dot product. mac_clear synchronously zeroes all 256
// accumulators; the engine FSM pulses it for one cycle when entering ST_RUN
// at the start of each OC pass.
//
// `mac_busy` is high whenever ANY pipeline stage holds live data, so the
// engine FSM can wait two cycles past the last mac_valid_in before
// snapshotting acc_out into the requant pipeline.
//
// Universal-bugs rule (knowledge/patterns/protected/08_common_bugs.md
// §"Array memory write in async-reset block") does NOT fire here: each
// accumulator is a SCALAR `reg signed [31:0] acc` declared per generated
// lane, not an indexed `reg [..] mem [..:..]` array. Vivado infers DFF
// for each scalar lane register independent of the reset clause.
// --------------------------------------------------------------------------

module mac_array #(
    // [INT3-MIXED] engine weight bit-width. 4 = INT4 (default, nibble-packed),
    // 3 = INT3. weight_bus packs 256 lanes * WGT_W bits. The shared engine
    // serves all 14 dispatched convs, so WGT_W is UNIFORM across them.
    parameter integer WGT_W = 4,
    // [KPAR4 2026-06-10] K-tap parallelism. 1 (DEFAULT) elaborates the
    // ORIGINAL serial datapath via generate-if — every legacy instance
    // (all ResNet tops/harnesses never set K_PAR) is bit- and
    // cycle-identical. 4 = MBV2 engine top: 4 taps/cycle/lane; the 4
    // products are summed by a COMBINATIONAL 4:1 tree into the same 32b
    // accumulator (INT8xINT8 -> 32b accumulation is exact and
    // order-independent), so the accumulate latency — and the skeleton's
    // d5 requant drain — is UNCHANGED (TREE_STAGES=0).
    parameter integer K_PAR = 1
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          mac_clear,
    input  wire          mac_valid_in,
    input  wire [7:0]    act_byte,
    input  wire [K_PAR*256*WGT_W-1:0] weight_bus,  // WGT_W-packed: K_PAR taps x 256 lanes (tap-major, tap0 lowest)
    // [KPAR4] taps 1..K_PAR-1 broadcast act bytes (dense mode; tap0 reuses
    // the legacy act_byte port). The skeleton ties this 0 when K_PAR==1.
    // [KPAR8 2026-06-10] width = (max(K_PAR,4)-1) bytes: the K_PAR==1 and K_PAR==4
    // elaborations keep their ORIGINAL [23:0] port exactly; only K_PAR==8
    // widens to [55:0] (taps 1..7).
    input  wire [(((K_PAR > 4) ? K_PAR : 4)-1)*8-1:0] act_bytes_ext,
    // [KPAR4] per-tap valid mask aligned with weight_bus/act bytes
    // (fast group: all-ones / partial; serial fallback: bit0 only). The
    // skeleton ties bit0=1 when K_PAR==1. A masked tap's act byte is zeroed
    // before the multiply, so its contribution is EXACTLY 0.
    // [KPAR8 2026-06-10] width = max(K_PAR,4): [3:0] at K_PAR<=4 (unchanged), [7:0] at 8.
    input  wire [((K_PAR > 4) ? K_PAR : 4)-1:0] tap_mask,
    // [DW-ENGINE P1 2026-06-10] per-lane activation mode (MobileNetV2 wide
    // depthwise convs). dw_mode=1: lane L multiplies its OWN byte of the
    // already-aligned activation word (act_word[L*8 +: 8], channels map 1:1
    // to lanes) instead of the shared broadcast act_byte. dw_mode=0 (legacy
    // and every ResNet instance — the skeleton ties it 0 when
    // ENABLE_DEPTHWISE==0): identical to the original broadcast datapath.
    input  wire          dw_mode,
    input  wire [2047:0] act_word,
    output wire [8191:0] acc_out,
    output wire          mac_busy
);

    // ----------------------------------------------------------------------
    // Shared pipeline-valid registers. All 256 lanes accumulate in lockstep,
    // so we only need one set of valid bits (not 256).
    // ----------------------------------------------------------------------
    // [FMAX-FANOUT] mac_valid_q1 gates the accumulate in all 256 MAC lanes (256-way
    // broadcast). Replicate so each region drives a local copy. Synth-only attribute
    // (Verilator ignores it) -> byte-exact + latency-neutral. Prep for the 100MHz
    // target (the 256-DSP-column broadcast becomes a limiter once spatial path speeds up).
    (* max_fanout = 32 *) reg mac_valid_q1;
    reg mac_valid_q2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mac_valid_q1 <= 1'b0;
            mac_valid_q2 <= 1'b0;
        end else begin
            mac_valid_q1 <= mac_valid_in;
            mac_valid_q2 <= mac_valid_q1;
        end
    end

    // High the moment a multiplicand enters stage-1 and stays high until the
    // last accumulated product has retired from stage-2. The engine FSM uses
    // this to know when acc_out has settled for snapshotting.
    assign mac_busy = mac_valid_in | mac_valid_q1 | mac_valid_q2;

    // ----------------------------------------------------------------------
    // 256 lanes. Each lane:
    //   - extracts its signed-INT8 weight from weight_bus
    //   - registers act_byte * weight_byte into a DSP-mapped product reg
    //   - accumulates the registered product into a signed INT32 acc reg
    //   - exposes acc as a slice of acc_out
    // ----------------------------------------------------------------------
    genvar lane;
    generate
    if (K_PAR == 1) begin : g_p1
        // ---- [KPAR4] ORIGINAL serial datapath, VERBATIM (legacy default) ----
        for (lane = 0; lane < 256; lane = lane + 1) begin : g_mac
            wire signed [WGT_W-1:0]  w_byte;   // WGT_W-bit weight (sign-extended in the multiply)
            wire signed [7:0]  a_byte;
            (* use_dsp = "yes" *) reg signed [15:0] mul_q1;
            reg signed [31:0] acc;

            assign w_byte = $signed(weight_bus[lane*WGT_W +: WGT_W]);
            assign a_byte = $signed(act_byte);

            // [DW-ENGINE P1] per-lane activation select: broadcast byte in
            // dense mode, this lane's own channel byte in depthwise mode.
            // With dw_mode tied 0 (legacy) this is the original a_byte.
            wire signed [7:0] a_byte_lane =
                dw_mode ? $signed(act_word[lane*8 +: 8]) : a_byte;

            // Stage 1: signed 8×8 multiply, registered into the DSP block.
            always @(posedge clk) begin
                mul_q1 <= w_byte * a_byte_lane;
            end

            // Stage 2: gated accumulate. The accumulator stays at zero from
            // reset and only updates while mac_valid_q1 indicates a live
            // multiplicand is exiting stage 1. mac_clear takes priority over
            // mac_valid_q1 so the FSM can synchronously reset all lanes on
            // the same cycle it kicks the next OC pass.
            // [K1-FDCE] acc's async reset is dead: the engine FSM pulses
            // mac_clear on EVERY ST_RUN entry (run_entered), so acc is sync-
            // cleared before the first gated accumulate of every dot product
            // (incl. the first after power-on; mac_valid_q1 is reset-held 0
            // until then). FDCE -> FDRE on 256 x 32 accumulator bits.
            always @(posedge clk) begin
                if (mac_clear)
                    acc <= 32'sd0;
                else if (mac_valid_q1)
                    acc <= acc + $signed(mul_q1);
            end

            assign acc_out[lane*32 +: 32] = acc;
        end
        // [KPAR4] lint tie: ext ports are consumed only by the K_PAR>1 branch.
        /* verilator lint_off UNUSED */
        wire _unused_kpar_ext = &{1'b0, act_bytes_ext, tap_mask};
        /* verilator lint_on UNUSED */
    end else if (K_PAR == 8) begin : g_p8
        // ---- [KPAR8 2026-06-10] 8-tap datapath: 8 DSP products/lane/cycle + a
        // COMBINATIONAL 8:1 adder tree into the 32b accumulator. Same
        // pipeline SHAPE as the serial and 4-tap paths (stage-1 product
        // regs, stage-2 gated accumulate) -> mac_busy timing and the
        // skeleton's d5 requant capture are unchanged (TREE_STAGES=0).
        // Fmax note: stage-2 is now a 9-operand (acc + 8x16b) sum — the
        // deepest combinational adder in the engine; see KPAR8_ANALYSIS.md.
        for (lane = 0; lane < 256; lane = lane + 1) begin : g_mac
            wire signed [7:0]  a_byte = $signed(act_byte);
            // tap0: legacy broadcast byte (dense) or this lane's own
            // channel byte (depthwise) — same select as the serial path.
            wire signed [7:0]  a_lane0 = dw_mode ? $signed(act_word[lane*8 +: 8]) : a_byte;
            // per-tap act bytes, ZEROED when the tap is masked (partial
            // last group / serial-fallback dispatches): a 0 act byte makes
            // the tap's product exactly 0, so masked taps cannot perturb acc.
            wire signed [7:0]  a0 = tap_mask[0] ? a_lane0 : 8'sd0;
            wire signed [7:0]  a1 = tap_mask[1] ? $signed(act_bytes_ext[7:0]) : 8'sd0;
            wire signed [7:0]  a2 = tap_mask[2] ? $signed(act_bytes_ext[15:8]) : 8'sd0;
            wire signed [7:0]  a3 = tap_mask[3] ? $signed(act_bytes_ext[23:16]) : 8'sd0;
            wire signed [7:0]  a4 = tap_mask[4] ? $signed(act_bytes_ext[31:24]) : 8'sd0;
            wire signed [7:0]  a5 = tap_mask[5] ? $signed(act_bytes_ext[39:32]) : 8'sd0;
            wire signed [7:0]  a6 = tap_mask[6] ? $signed(act_bytes_ext[47:40]) : 8'sd0;
            wire signed [7:0]  a7 = tap_mask[7] ? $signed(act_bytes_ext[55:48]) : 8'sd0;
            // tap-major weight slices: tap j's 256-lane word at [j*256*WGT_W].
            wire signed [WGT_W-1:0] w0 = $signed(weight_bus[(0*256 + lane)*WGT_W +: WGT_W]);
            wire signed [WGT_W-1:0] w1 = $signed(weight_bus[(1*256 + lane)*WGT_W +: WGT_W]);
            wire signed [WGT_W-1:0] w2 = $signed(weight_bus[(2*256 + lane)*WGT_W +: WGT_W]);
            wire signed [WGT_W-1:0] w3 = $signed(weight_bus[(3*256 + lane)*WGT_W +: WGT_W]);
            wire signed [WGT_W-1:0] w4 = $signed(weight_bus[(4*256 + lane)*WGT_W +: WGT_W]);
            wire signed [WGT_W-1:0] w5 = $signed(weight_bus[(5*256 + lane)*WGT_W +: WGT_W]);
            wire signed [WGT_W-1:0] w6 = $signed(weight_bus[(6*256 + lane)*WGT_W +: WGT_W]);
            wire signed [WGT_W-1:0] w7 = $signed(weight_bus[(7*256 + lane)*WGT_W +: WGT_W]);
            (* use_dsp = "yes" *) reg signed [15:0] mul_q1_0;
            (* use_dsp = "yes" *) reg signed [15:0] mul_q1_1;
            (* use_dsp = "yes" *) reg signed [15:0] mul_q1_2;
            (* use_dsp = "yes" *) reg signed [15:0] mul_q1_3;
            (* use_dsp = "yes" *) reg signed [15:0] mul_q1_4;
            (* use_dsp = "yes" *) reg signed [15:0] mul_q1_5;
            (* use_dsp = "yes" *) reg signed [15:0] mul_q1_6;
            (* use_dsp = "yes" *) reg signed [15:0] mul_q1_7;
            reg signed [31:0] acc;

            always @(posedge clk) begin
                mul_q1_0 <= w0 * a0;
                mul_q1_1 <= w1 * a1;
                mul_q1_2 <= w2 * a2;
                mul_q1_3 <= w3 * a3;
                mul_q1_4 <= w4 * a4;
                mul_q1_5 <= w5 * a5;
                mul_q1_6 <= w6 * a6;
                mul_q1_7 <= w7 * a7;
            end

            // [K1-FDCE] same no-reset accumulate as the serial path
            // (mac_clear pulses on every ST_RUN entry). All operands are
            // signed; the 8-way sum is sign-extended into the 32b acc —
            // exact integer math, identical result to 8 serial adds.
            always @(posedge clk) begin
                if (mac_clear)
                    acc <= 32'sd0;
                else if (mac_valid_q1)
                    acc <= acc + mul_q1_0 + mul_q1_1 + mul_q1_2 + mul_q1_3 + mul_q1_4 + mul_q1_5 + mul_q1_6 + mul_q1_7;
            end

            assign acc_out[lane*32 +: 32] = acc;
        end
    end else begin : g_p4
        // ---- [KPAR4] 4-tap datapath: 4 DSP products/lane/cycle + a
        // COMBINATIONAL 4:1 adder tree into the 32b accumulator. The
        // pipeline SHAPE matches the serial path exactly (stage-1 product
        // regs, stage-2 gated accumulate), so mac_busy timing and the
        // skeleton's d5 requant capture are unchanged.
        for (lane = 0; lane < 256; lane = lane + 1) begin : g_mac
            wire signed [7:0]  a_byte = $signed(act_byte);
            // tap0: legacy broadcast byte (dense) or this lane's own
            // channel byte (depthwise) — same select as the serial path.
            wire signed [7:0]  a_lane0 = dw_mode ? $signed(act_word[lane*8 +: 8]) : a_byte;
            // per-tap act bytes, ZEROED when the tap is masked (partial
            // last group / serial-fallback dispatches): a 0 act byte makes
            // the tap's product exactly 0, so masked taps cannot perturb acc.
            wire signed [7:0]  a0 = tap_mask[0] ? a_lane0                       : 8'sd0;
            wire signed [7:0]  a1 = tap_mask[1] ? $signed(act_bytes_ext[7:0])   : 8'sd0;
            wire signed [7:0]  a2 = tap_mask[2] ? $signed(act_bytes_ext[15:8])  : 8'sd0;
            wire signed [7:0]  a3 = tap_mask[3] ? $signed(act_bytes_ext[23:16]) : 8'sd0;
            // tap-major weight slices: tap j's 256-lane word at [j*256*WGT_W].
            wire signed [WGT_W-1:0] w0 = $signed(weight_bus[(0*256 + lane)*WGT_W +: WGT_W]);
            wire signed [WGT_W-1:0] w1 = $signed(weight_bus[(1*256 + lane)*WGT_W +: WGT_W]);
            wire signed [WGT_W-1:0] w2 = $signed(weight_bus[(2*256 + lane)*WGT_W +: WGT_W]);
            wire signed [WGT_W-1:0] w3 = $signed(weight_bus[(3*256 + lane)*WGT_W +: WGT_W]);
            (* use_dsp = "yes" *) reg signed [15:0] mul_q1_0;
            (* use_dsp = "yes" *) reg signed [15:0] mul_q1_1;
            (* use_dsp = "yes" *) reg signed [15:0] mul_q1_2;
            (* use_dsp = "yes" *) reg signed [15:0] mul_q1_3;
            reg signed [31:0] acc;

            always @(posedge clk) begin
                mul_q1_0 <= w0 * a0;
                mul_q1_1 <= w1 * a1;
                mul_q1_2 <= w2 * a2;
                mul_q1_3 <= w3 * a3;
            end

            // [K1-FDCE] same no-reset accumulate as the serial path
            // (mac_clear pulses on every ST_RUN entry). All operands are
            // signed; the 4-way sum is sign-extended into the 32b acc —
            // exact integer math, identical result to 4 serial adds.
            always @(posedge clk) begin
                if (mac_clear)
                    acc <= 32'sd0;
                else if (mac_valid_q1)
                    acc <= acc + mul_q1_0 + mul_q1_1 + mul_q1_2 + mul_q1_3;
            end

            assign acc_out[lane*32 +: 32] = acc;
        end
    end
    endgenerate

endmodule
