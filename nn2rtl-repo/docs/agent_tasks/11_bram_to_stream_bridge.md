---
task_id: 11
title: BRAM-to-stream bridge module
type: Foundry RTL generation (LLM-driven)
status: review
depends_on: [00]
unblocks: [Phase 2 integration]
---

# Task 11 — BRAM-to-stream bridge module

## Goal

Generate a small Verilog module that bridges between the engine's BRAM-based output (where the engine writes one BRAM word per output pixel) and the spatial chain's downstream streaming interface (which expects standard `valid_in/ready_in/data_in` handshakes).

The deployment plan §6.4 calls this out: *"A bridge module between them converts BRAM-access to the streaming handshake. The bridge is itself a small Verilog module — LLM-generable, bit-exact-verifiable."*

## Deliverable

A single Verilog file at `output/rtl/engine/bram_to_stream_bridge.v` containing `module bram_to_stream_bridge (...)`.

The module is symmetric — the same module can be used both:

1. **Engine output → spatial chain input**: the engine writes BRAM, the bridge reads and streams.
2. **Spatial chain output → engine input**: the chain streams, the bridge writes BRAM.

You may implement both directions in one module with a `MODE` parameter, or split into two sister modules — your call. Easier review if one module.

## Required ports

The bridge in **stream-out mode** (BRAM → stream):

- `clk`, `rst_n`
- `start` (input, pulses when the upstream BRAM is fully written and ready for the bridge to start streaming)
- `total_words` (input, how many BRAM words to stream)
- `bram_rd_addr` (output)
- `bram_rd_data` (input, width = `BUS_W`)
- `bram_rd_en` (output)
- `valid_out` (output)
- `ready_out` (input, downstream consumer's ready)
- `data_out[BUS_W-1:0]` (output)
- `done` (output, pulses when all words have been streamed)

The bridge in **stream-in mode** (stream → BRAM):

- `clk`, `rst_n`
- `start` (input)
- `bram_wr_addr` (output)
- `bram_wr_data[BUS_W-1:0]` (output)
- `bram_wr_en` (output)
- `valid_in` (input)
- `ready_in` (output)
- `data_in[BUS_W-1:0]` (input)
- `done` (output)

## Architecture hints

- BRAM read latency is 1 cycle on UltraScale+. Need to pipeline `bram_rd_en`/`addr` → `bram_rd_data` → `valid_out` accordingly. A simple two-stage pipeline (fetch, register) works.
- Backpressure: when `ready_out` is low, the bridge must hold its current data and not advance the read address. Implementing this cleanly often uses a small skid buffer (1-entry FIFO) to absorb the cycle of latency between deasserting `bram_rd_en` and the data already in flight from BRAM.
- For write mode: standard accept-when-ready pattern. `ready_in` goes high after `start`, stays high until `total_words` have been received, then goes low and `done` pulses.

## Context to read before starting

- `docs/agent_tasks/00_engine_skeleton_spec_PORTS.md` (the engine's BRAM activation ports — this bridge sits between those and the spatial chain).
- `docs/nn2rtl_u250_deployment_plan.md §6.4` (the design intent).
- `knowledge/patterns/protected/01_context.md` (the streaming handshake conventions used throughout nn2rtl).

## How to verify

0. **Port consistency (mandatory)**: `python scripts/check_subblock_ports.py --subblock=bram_to_stream_bridge --rtl=output/rtl/engine/bram_to_stream_bridge.v` exits 0. Copy port declarations verbatim from `docs/agent_tasks/00_engine_skeleton_spec_PORTS.md`.
1. **Compiles**: `iverilog -t null output/rtl/engine/bram_to_stream_bridge.v` passes.
2. **Read-mode testbench**: load a behavioural BRAM model with 64 known words. Trigger `start`. Drive `ready_out` always high. Confirm 64 words come out in the correct order with `valid_out` asserted on each.
3. **Read-mode with backpressure**: same as 2 but cycle `ready_out` randomly (sometimes deasserted). Confirm no word is lost, no word is repeated, order is preserved.
4. **Write-mode testbench**: stream 64 known words into the bridge with `valid_in` cycling realistically. Confirm BRAM is written in the right addresses with the right data.
5. **`done` pulses exactly once**, at the correct cycle (one cycle after the last word transfer completes).
6. **Standalone Fmax**: ≥ 400 MHz (small data-path module, should not be on critical path).

## Out of scope

- Do NOT include reordering or width-conversion logic. The bridge reads/writes one BRAM word per cycle at the same width as the streaming bus. If width-conversion is ever needed for a future contract, that is a separate task.
- Do NOT add FIFO depth beyond the 1-entry skid buffer needed for backpressure. Deeper buffering is the wrapper's concern.

## Success criteria

- File exists, compiles, parses.
- Read-mode and write-mode testbenches both pass with zero data corruption and zero lost cycles.
- Standalone Fmax ≥ 400 MHz.
- The bridge can be instantiated cleanly in the top wrapper's engine I/O paths.
