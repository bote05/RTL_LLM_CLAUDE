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
    parameter integer WGT_W = 4
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          mac_clear,
    input  wire          mac_valid_in,
    input  wire [7:0]    act_byte,
    input  wire [256*WGT_W-1:0] weight_bus,  // WGT_W-packed: 256 lanes * WGT_W bits
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
        for (lane = 0; lane < 256; lane = lane + 1) begin : g_mac
            wire signed [WGT_W-1:0]  w_byte;   // WGT_W-bit weight (sign-extended in the multiply)
            wire signed [7:0]  a_byte;
            (* use_dsp = "yes" *) reg signed [15:0] mul_q1;
            reg signed [31:0] acc;

            assign w_byte = $signed(weight_bus[lane*WGT_W +: WGT_W]);
            assign a_byte = $signed(act_byte);

            // Stage 1: signed 8×8 multiply, registered into the DSP block.
            always @(posedge clk) begin
                mul_q1 <= w_byte * a_byte;
            end

            // Stage 2: gated accumulate. The accumulator stays at zero from
            // reset and only updates while mac_valid_q1 indicates a live
            // multiplicand is exiting stage 1. mac_clear takes priority over
            // mac_valid_q1 so the FSM can synchronously reset all lanes on
            // the same cycle it kicks the next OC pass.
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    acc <= 32'sd0;
                else if (mac_clear)
                    acc <= 32'sd0;
                else if (mac_valid_q1)
                    acc <= acc + $signed(mul_q1);
            end

            assign acc_out[lane*32 +: 32] = acc;
        end
    endgenerate

endmodule
