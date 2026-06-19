---
task_id: 07
title: Engine MAC array sub-block
type: Foundry RTL generation (LLM-driven)
status: review
depends_on: [00]
unblocks: [Phase 2 integration]
---

# Task 07 — Engine MAC array sub-block

## Goal

Generate the MAC array sub-block of the shared compute engine. This is the parallel multiplier-and-accumulator structure that performs the core dot-product arithmetic for any heavy convolutional layer dispatched through the engine.

The skeleton (from task 00) declares an instantiation:

```verilog
// SUBBLOCK: mac_array
mac_array u_mac_array (
    // (signals defined in 00_engine_skeleton_spec_PORTS.md)
);
```

This task fills that stub.

## Deliverable

A single Verilog file at `output/rtl/engine/mac_array.v` containing `module mac_array (...)` that drops into the skeleton's instantiation block.

The module:
- Implements **256 parallel signed INT8 × INT8 multipliers**.
- Feeds into **256 INT32 accumulators**, one per output channel lane.
- Output-channel-parallel: at each cycle, the array processes one (input-channel, kernel-position) pair across 256 output channels simultaneously.
- Supports a `clear_accumulators` strobe that zeroes all 256 accumulators at the start of each output pixel.
- Exposes `acc_out[256]` (an array of 256 INT32 accumulators) when the dot product for the current output pixel is complete. This is the input to the requantisation pipeline (task 08).
- Uses `(* use_dsp = "yes" *)` attributes so synthesis maps multipliers to DSP48E2 slices.

## Required ports (must match task 00's spec)

Read `docs/agent_tasks/00_engine_skeleton_spec_PORTS.md` for the canonical port list. Expected ports:

- `clk`, `rst_n`
- `clear_accumulators` (input, 1 bit, pulses high to zero accumulators)
- `weight_lane_data[256 * 8 - 1 : 0]` (input, packed 256 INT8 weights for the current MAC step)
- `act_value[7:0]` (input, signed INT8 — the single input activation broadcast across all 256 lanes)
- `mac_valid` (input, 1 bit — assert when weight + activation are valid)
- `acc_valid` (output, 1 bit — pulses high when accumulators hold the final dot-product for one output pixel)
- `acc_out[256 * 32 - 1 : 0]` (output, packed 256 INT32 accumulator values)

If task 00's spec uses slightly different names, follow that spec — these are illustrative.

## Architecture hints

- The 256 multipliers should be a generate-for loop:
  ```verilog
  genvar lane;
  generate
    for (lane = 0; lane < 256; lane = lane + 1) begin : g_mac
      (* use_dsp = "yes" *) reg signed [15:0] mul_q1;
      reg signed [31:0] acc;
      wire signed [7:0] w_lane = $signed(weight_lane_data[lane*8 +: 8]);
      always @(posedge clk) begin
        mul_q1 <= w_lane * $signed(act_value);
      end
      always @(posedge clk or negedge rst_n) begin
        if (!rst_n) acc <= 32'sd0;
        else if (clear_accumulators) acc <= 32'sd0;
        else if (mac_valid_q1) acc <= acc + $signed(mul_q1);
      end
    end
  endgenerate
  ```
- Apply the structural patterns from `knowledge/patterns/protected/08_common_bugs.md` (especially the "array memory write in async-reset block" rule — the accumulator array is allowed in an async-reset block because each accumulator is a scalar register, not an indexed memory; the universal-bugs rule applies to indexed arrays only).
- Pipeline depth: 1 cycle for the multiply, 1 cycle for the add. Acc_valid pulses 2 cycles after the last `mac_valid` of the dot product.

## Context to read before starting

- `docs/agent_tasks/00_engine_skeleton_spec_PORTS.md` — authoritative port spec.
- `docs/nn2rtl_u250_deployment_plan.md §6.1` — design commitments (256 MACs, output-channel-parallel, INT8 datapath).
- `knowledge/patterns/protected/02_conv1x1.md` — canonical conv MAC structure.
- `knowledge/patterns/protected/08_common_bugs.md` — universal RTL discipline.
- `output/rtl/node_conv_288.v` — the existing seed reference. The MAC structure inside is a smaller version of what this task generates.

## How to verify

The MAC array has no per-network golden vector (it is a sub-block, not a layer). Verification is:

0. **Port consistency (mandatory)**: `python scripts/check_subblock_ports.py --subblock=mac_array --rtl=output/rtl/engine/mac_array.v` exits 0. The sub-block's port list must match `docs/agent_tasks/00_engine_skeleton_spec_PORTS.md` exactly — copy the declarations verbatim from there, do not rename, widen, or restructure them.
1. **Compiles**: `iverilog -t null output/rtl/engine/mac_array.v` succeeds.
2. **Unit testbench**: a small C++ Verilator wrapper that drives `clear_accumulators`, feeds a known sequence of weight + activation byte pairs, and checks the output accumulators against a Python golden computed in software. The agent must produce this testbench at `output/rtl/engine/mac_array_tb.cpp`.
3. **DSP packing**: post-synth Vivado must report ≥ 200 DSPs for this module (256 multipliers; some may pack into LUT MULTs depending on fabric availability — accept anything ≥ 200).
4. **Standalone Fmax**: ≥ 250 MHz post-synth (engine should be the fastest path in the design, since heavy layers serialise through it).
5. **Bit-exact**: the unit testbench's max_error must be 0. There is no requantisation in this sub-block; outputs are full INT32 accumulators.

## Out of scope

- Do NOT add bias addition, scaling, or saturation — those are the requantisation pipeline's job (task 08).
- Do NOT add weight memory reads — task 09 owns the address generator and weight read.
- Do NOT instantiate this module inside the skeleton — task 00 has already done that. Your job is to fill in the body of `mac_array.v`.
- Do NOT touch other engine sub-blocks, the wrapper, the scheduler, or per-layer modules.

## Success criteria

- `output/rtl/engine/mac_array.v` exists, parses cleanly, compiles cleanly under iverilog.
- `output/rtl/engine/mac_array_tb.cpp` exists.
- Unit Verilator run reports `max_error = 0` on at least 100 random INT8 input pairs.
- Vivado synth on the standalone module reports ≥ 200 DSPs and Fmax ≥ 250 MHz.
