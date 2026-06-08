`timescale 1ns/1ps

// bram_to_stream_bridge.v
// --------------------------------------------------------------------------
// Wave 2 task 11 sub-block. Port list is LOCKED by
// docs/agent_tasks/00_engine_skeleton_spec_PORTS.md
// `## SUBBLOCK: bram_to_stream_bridge`. Drift fails check_subblock_ports.py.
// Spec: docs/agent_tasks/11_bram_to_stream_bridge.md
//
// Two-half bridge between the engine's parallel BRAM activation ports and
// its internal streaming MAC / requant pipeline.
//
// Read half (BRAM beat -> MAC byte):
//   - On every clock the caller drives a 2048-bit `act_in_rd_data` beat that
//     just came back from the activation BRAM (BRAM read latency = 1 cycle
//     on UltraScale+). Alongside the beat the caller drives
//     `act_in_ic_byte_idx` (the channel byte the MAC needs this step) and
//     `act_in_rd_data_valid` (1 when the beat is real).
//   - The bridge byte-selects bits [byte_idx*8 +: 8] from the beat and
//     forwards them as `mac_act_byte`, with `mac_act_byte_valid` riding
//     along. Both outputs are registered, giving the byte-select cone a
//     full clock period to settle and adding one cycle of latency between
//     `act_in_rd_data_valid` and `mac_act_byte_valid`.
//
// Write half (requant beat -> BRAM beat):
//   - When `requant_valid` pulses, the bridge latches the 2048-bit
//     `requant_data` beat and drives it onto `act_out_wr_data` the next
//     cycle with `act_out_wr_en` pulsed for exactly one cycle. No deeper
//     buffering (per task 11 "Out of scope: deep FIFO buffering").
//   - `bridge_busy` is the combinational OR of `requant_valid` and the
//     pulsed `act_out_wr_en`, so it stays high from the cycle the caller
//     hands the bridge a beat through the cycle the BRAM port actually
//     accepts it. The engine FSM uses this as a hold-off so it does not
//     advance the next OC pass while a write is still in flight.
//
// Out of scope (per task 11):
//   - No width conversion (the data buses are 2048-bit on both sides).
//   - No deep FIFO buffering beyond the single output register.
//
// Reset is canonical async-assert / sync-deassert; only scalar regs and
// vector regs (no indexed arrays) are written here, so a single
// `posedge clk or negedge rst_n` block per half is correct (no array-write
// universal-bug case from 08_common_bugs.md).
// --------------------------------------------------------------------------

module bram_to_stream_bridge (
    input  wire           clk,
    input  wire           rst_n,
    // Read half ---------------------------------------------------------
    input  wire [2047:0]  act_in_rd_data,
    input  wire           act_in_rd_data_valid,
    input  wire [7:0]     act_in_ic_byte_idx,
    output reg  [7:0]     mac_act_byte,
    output reg            mac_act_byte_valid,
    // Write half --------------------------------------------------------
    input  wire [2047:0]  requant_data,
    input  wire           requant_valid,
    // [ENGINE-OUTPUT BACKPRESSURE] downstream (engine_output_fifo) can accept a
    // beat THIS cycle. The shared_engine drives this from its own out_ready,
    // which is param-gated to a constant 1'b1 unless ENABLE_OUTPUT_BACKPRESSURE
    // is set. With out_ready==1'b1 the hold branch below is dead and the write
    // half is byte-identical to the original 1-cycle-pulse design (ResNet + the
    // engine-iso harnesses, which leave it gated to 1, are unchanged).
    input  wire           out_ready,
    output reg  [2047:0]  act_out_wr_data,
    output reg            act_out_wr_en,
    output reg            bridge_busy
);

    // ----------------------------------------------------------------------
    // Read half: register the selected byte and the valid bit. The
    // byte-select `act_in_rd_data[byte_idx*8 +: 8]` is a 256:1 mux over the
    // 2048-bit beat; the registered output gives Vivado a full clock period
    // to map it (relevant for the 400 MHz Fmax target).
    // ----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mac_act_byte       <= 8'd0;
            mac_act_byte_valid <= 1'b0;
        end else begin
            mac_act_byte       <= act_in_rd_data[act_in_ic_byte_idx * 8 +: 8];
            mac_act_byte_valid <= act_in_rd_data_valid;
        end
    end

    // ----------------------------------------------------------------------
    // Write half: latch the requant beat and present the BRAM/FIFO write the
    // next cycle. With backpressure (out_ready):
    //   * When out_ready is HIGH the beat is accepted the cycle act_out_wr_en
    //     is asserted, so act_out_wr_en degenerates to the original one-cycle
    //     pulse = registered requant_valid (BYTE-IDENTICAL to the original).
    //   * When a beat is presented (act_out_wr_en=1) and out_ready is LOW the
    //     beat CANNOT be accepted; HOLD it (keep act_out_wr_en=1 and the latched
    //     act_out_wr_data unchanged) instead of clobbering it with the next
    //     requant beat. The shared_engine FSM is frozen coherently while this
    //     hold is active (it does not start the next oc_pass's MAC run), so no
    //     new requant_valid pulse arrives during the hold -> no beat is lost.
    // bridge_busy (below) stays high through the hold, so the engine FSM's
    // ST_DRAIN / ST_REQUANT advance is naturally gated until the beat lands.
    // ----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            act_out_wr_data <= 2048'd0;
            act_out_wr_en   <= 1'b0;
        end else if (act_out_wr_en && !out_ready) begin
            // Held: beat presented but not yet accepted. Freeze the write half.
            act_out_wr_en   <= 1'b1;
            act_out_wr_data <= act_out_wr_data;
        end else begin
            act_out_wr_data <= requant_data;
            act_out_wr_en   <= requant_valid;
        end
    end

    // ----------------------------------------------------------------------
    // bridge_busy spans both the cycle the caller hands us a beat
    // (requant_valid=1, the latch cycle) and the cycle the BRAM port
    // actually accepts it (act_out_wr_en=1). Combinational so the engine
    // FSM sees "busy" on the same cycle it pulses requant_valid.
    // ----------------------------------------------------------------------
    always @(*) begin
        bridge_busy = requant_valid | act_out_wr_en;
    end

endmodule
