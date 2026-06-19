---
task_id: 08
title: Engine requantisation pipeline sub-block
type: Foundry RTL generation (LLM-driven)
status: review
depends_on: [00]
unblocks: [Phase 2 integration]
---

# Task 08 — Engine requantisation pipeline sub-block

## Goal

Generate the requantisation pipeline sub-block of the shared compute engine. This takes the 256 INT32 accumulators produced by the MAC array (task 07), adds the per-output-channel bias, multiplies by the per-layer scale, shifts and saturates back to 256 INT8 values.

The skeleton (task 00) declares the instantiation:

```verilog
// SUBBLOCK: requant_pipeline
requant_pipeline u_requant_pipeline (...);
```

This task fills that stub.

## Deliverable

A single Verilog file at `output/rtl/engine/requant_pipeline.v` containing `module requant_pipeline (...)`.

The module:
- Reads `acc_in[256 * 32 - 1 : 0]` — 256 INT32 accumulators from the MAC array.
- Reads `bias_in[256 * 32 - 1 : 0]` — 256 INT32 biases for the current 256 output channels. The bias memory layout is **one wide word = 256 INT32 biases**, addressed by `oc_pass` (locked in task 09 §"Address granularity"). Task 09 emits `bias_rd_addr = bias_base_word + oc_pass`; the bias memory delivers the wide word on the next cycle; this module consumes it as `bias_in`.
- Reads `scale_mult[31:0]` and `scale_shift[5:0]` — the per-layer composite scale, loaded by the scheduler via task 10's config registers.
- Outputs `data_out[256 * 8 - 1 : 0]` — 256 saturated INT8 values, ready to write back to the output activation BRAM.
- Pipeline depth: **3 stages** (bias-add → scale-multiply → scale-shift + saturate). This is fixed by the deployment plan §6.1.

## Required ports (must match task 00's spec)

Read `docs/agent_tasks/00_engine_skeleton_spec_PORTS.md`. Expected ports:

- `clk`, `rst_n`
- `acc_valid` (input — pulse from the MAC array)
- `acc_in[256 * 32 - 1 : 0]` (input)
- `bias_in[256 * 32 - 1 : 0]` (input)
- `scale_mult[31:0]` (input, the per-layer scale_mult value loaded into the config register block)
- `scale_shift[5:0]` (input, the per-layer scale_shift value)
- `out_valid` (output, pulses 3 cycles after `acc_valid`)
- `data_out[256 * 8 - 1 : 0]` (output, packed 256 INT8 saturated values)

## Algorithm (must match the per-layer requantisation in the existing modules)

For each output channel `lane` in 0..255:

```
biased[lane]   = acc_in[lane] + bias_in[lane]                          ; INT32 add
scaled[lane]   = round_half_up( biased[lane] * scale_mult >> scale_shift )
out[lane]      = clamp(scaled[lane], -128, +127)                       ; INT8 saturate
```

This must be **bit-exact identical** to the requantisation tail of the existing per-layer modules. Look at `output/rtl/node_conv_288.v` (the active reference for dram-backed-weights) for the canonical sequence:
- The scale-multiply stage uses a signed 32×32→64-bit multiply.
- The scale-shift uses round-half-up-toward-+infinity (add `1 << (scale_shift - 1)` before the right shift).
- The saturation is an explicit clamp: `> 127 → 127`, `< -128 → -128`, else lower 8 bits.

If any of these is wrong, the per-layer goldens will fail. Match the existing reference exactly.

## Architecture hints

- Use a generate-for loop to fan out 256 parallel requantisation pipelines.
- Apply `(* use_dsp = "yes" *)` to the scale-multiply registered output if the synthesis report shows DSPs being underutilised.
- Bias loading is sequenced by task 09 — this module just consumes `bias_in` on the cycle when `acc_valid` is high.
- 3-stage pipeline → `out_valid` is 3 cycles delayed from `acc_valid`. Track the valid bit through 3 flip-flop stages alongside the data.

## Context to read before starting

- `docs/agent_tasks/00_engine_skeleton_spec_PORTS.md` — authoritative port spec.
- `docs/nn2rtl_u250_deployment_plan.md §6.1` — design commitment for 3-stage requant.
- `output/rtl/node_conv_288.v` — canonical requantisation arithmetic. Match this byte-for-byte.
- `knowledge/patterns/protected/02_conv1x1.md` — explains the requantisation tail conceptually.

## How to verify

0. **Port consistency (mandatory)**: `python scripts/check_subblock_ports.py --subblock=requant_pipeline --rtl=output/rtl/engine/requant_pipeline.v` exits 0. Copy port declarations verbatim from `docs/agent_tasks/00_engine_skeleton_spec_PORTS.md`.
1. **Compiles**: `iverilog -t null output/rtl/engine/requant_pipeline.v` succeeds.
2. **Unit testbench**: a C++ Verilator harness at `output/rtl/engine/requant_pipeline_tb.cpp` that feeds known accumulator + bias + scale values and checks against a Python golden using the exact same arithmetic. Must report `max_error = 0` on at least 1,000 random INT32 accumulator values.
3. **Bit-exact cross-check against an existing layer**: take the requantisation of any heavy module (e.g. `node_conv_298`), extract the scale_mult, scale_shift, biases. Feed them into this module and compare against the existing module's intermediate signals. Same outputs → success.
4. **Standalone Fmax**: ≥ 300 MHz (requant is shorter than MAC; should not be on the critical path).

## Out of scope

- Do NOT do bias memory reads; task 09 handles those.
- Do NOT do output BRAM writes; task 09 handles those too.
- Do NOT add any precision other than INT8 in / INT8 out via INT32 accumulator.
- Do NOT touch other engine sub-blocks.

## Success criteria

- `output/rtl/engine/requant_pipeline.v` exists, parses + compiles cleanly.
- Unit testbench `max_error = 0` on 1,000+ random inputs.
- Cross-check against `node_conv_298`'s requant tail is bit-exact.
- Standalone Fmax ≥ 300 MHz post-synth.
