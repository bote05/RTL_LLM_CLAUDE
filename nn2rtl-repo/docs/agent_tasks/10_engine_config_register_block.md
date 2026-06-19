---
task_id: 10
title: Engine config register block (AXI4-Lite slave)
type: Foundry RTL generation (LLM-driven)
status: review
depends_on: [00]
unblocks: [Phase 2 integration]
---

# Task 10 — Engine config register block (AXI4-Lite slave)

## Goal

Generate the AXI4-Lite slave that holds the engine's per-layer configuration registers. The scheduler (task 03) writes these registers via the AXI4-Lite master interface before each dispatch. The address generator (task 09) reads them via the parallel output of this block.

The skeleton (task 00) declares the instantiation:

```verilog
// SUBBLOCK: config_register_block
config_register_block u_config_register_block (...);
```

This task fills that stub.

## Deliverable

A single Verilog file at `output/rtl/engine/config_register_block.v` containing `module config_register_block (...)`.

## Required ports

Read `docs/agent_tasks/00_engine_skeleton_spec_PORTS.md`. Expected ports include:

- `clk`, `rst_n`
- **AXI4-Lite slave interface** (the standard 5 channels):
  - `s_axil_awvalid` (in), `s_axil_awready` (out), `s_axil_awaddr[7:0]` (in, byte-address)
  - `s_axil_wvalid` (in), `s_axil_wready` (out), `s_axil_wdata[31:0]` (in), `s_axil_wstrb[3:0]` (in)
  - `s_axil_bvalid` (out), `s_axil_bready` (in), `s_axil_bresp[1:0]` (out)
  - `s_axil_arvalid` (in), `s_axil_arready` (out), `s_axil_araddr[7:0]` (in)
  - `s_axil_rvalid` (out), `s_axil_rready` (in), `s_axil_rdata[31:0]` (out), `s_axil_rresp[1:0]` (out)
- **Parallel output ports** (the registered values, consumed by other sub-blocks):
  - `cfg_input_channels[15:0]`
  - `cfg_output_channels[15:0]`
  - `cfg_kernel_h[3:0]`, `cfg_kernel_w[3:0]`
  - `cfg_stride_h[2:0]`, `cfg_stride_w[2:0]`
  - `cfg_padding_h[2:0]`, `cfg_padding_w[2:0]`
  - `cfg_input_h[8:0]`, `cfg_input_w[8:0]`
  - `cfg_output_h[8:0]`, `cfg_output_w[8:0]`
  - `cfg_weight_base_word[19:0]` (URAM word address)
  - `cfg_bias_base_word[15:0]`
  - `cfg_scale_mult[31:0]`
  - `cfg_scale_shift[5:0]`
  - `cfg_zero_point[7:0]` (signed)
- **Control**:
  - `engine_start` (out, pulses high for one cycle when the scheduler writes the START register and the engine is currently idle)
  - `engine_busy` (in, from the engine FSM)

## Register map

Choose a clean byte-addressed register map (32-bit registers). Suggested layout:

| Offset | Register | Width |
| --- | --- | --- |
| 0x00 | INPUT_CHANNELS  | 16 |
| 0x04 | OUTPUT_CHANNELS | 16 |
| 0x08 | KERNEL_H_W | {pad, kh, kw} packed |
| 0x0C | STRIDE_H_W | {pad, sh, sw} packed |
| 0x10 | PADDING_H_W | {pad, ph, pw} packed |
| 0x14 | INPUT_H_W | {ih[15:0], iw[15:0]} |
| 0x18 | OUTPUT_H_W | {oh[15:0], ow[15:0]} |
| 0x1C | WEIGHT_BASE_WORD | 32 (upper bits unused) |
| 0x20 | BIAS_BASE_WORD | 32 (upper bits unused; this is the wide-bias-word base, see task 09 §"Address granularity") |
| 0x24 | SCALE_MULT | 32 |
| 0x28 | SCALE_SHIFT_AND_ZP | {pad, zero_point[7:0], scale_shift[5:0]} packed |
| 0x2C | CONTROL | bit[0] = START (write-1 pulses engine_start), bit[1] = BUSY (read-only mirror of engine_busy) |
| 0x30 | STATUS | bit[0] = DONE (read-only, set by engine, cleared on START) |

(You can tweak the exact layout if it makes the scheduler simpler; the scheduler is also generated, so they stay in sync.)

## Behavioural requirements

- Single-cycle AXI4-Lite write response (`bvalid` follows immediately after `awvalid+wvalid`).
- Single-cycle read response (`rvalid` follows immediately after `arvalid`).
- Writing `1` to the START bit of CONTROL pulses `engine_start` for exactly one cycle on the next rising edge, *provided* `engine_busy` is low. If busy, the write completes but no pulse is emitted (and a status bit could record "ignored write" for debug; not required).
- All registers reset to 0 on `!rst_n`.
- `engine_start` is held low except for the 1-cycle pulse.

## Architecture hints

- The AXI4-Lite slave is a well-known structural pattern. There is no protected reference Verilog for it in nn2rtl today (this is the first task to need one), so write it cleanly and the result will likely become a reference for future engine variants.
- Use the canonical patterns from `knowledge/patterns/protected/08_common_bugs.md` — particularly: control regs stay in async-reset always blocks, never indexed array writes in async-reset blocks (n/a here, no arrays).
- Single-cycle handshake on both read and write channels is the simplest correct implementation; latency-fairness and back-to-back transactions are not required.

## How to verify

0. **Port consistency (mandatory)**: `python scripts/check_subblock_ports.py --subblock=config_register_block --rtl=output/rtl/engine/config_register_block.v` exits 0. Copy port declarations verbatim from `docs/agent_tasks/00_engine_skeleton_spec_PORTS.md`.
1. **Compiles**: `iverilog -t null output/rtl/engine/config_register_block.v` passes.
2. **AXI4-Lite scoreboard testbench**: a C++ Verilator harness at `output/rtl/engine/config_register_block_tb.cpp` that:
   - Writes all registers to known values.
   - Reads them back. Expected values match writes.
   - Writes START, observes `engine_start` pulses for exactly one cycle.
   - Holds `engine_busy` high and writes START; observes no pulse.
3. **All registers respond to read after write** (no register lost).
4. **`engine_start` pulse is exactly 1 cycle wide**.
5. **Standalone Fmax**: ≥ 400 MHz (this is small control logic; should not be on any critical path).

## Out of scope

- Do NOT include any compute logic, MAC pipeline, or convolution loops.
- Do NOT add interrupt support (this version is polled — the scheduler writes START and polls DONE).
- Do NOT add multi-master AXI arbitration. Single master is fine.

## Success criteria

- File exists, compiles, parses.
- AXI4-Lite scoreboard testbench reports zero mismatches across all 12 register read-after-write checks.
- `engine_start` pulse width is exactly 1 cycle in all tested transitions.
- Standalone synth Fmax ≥ 400 MHz.
