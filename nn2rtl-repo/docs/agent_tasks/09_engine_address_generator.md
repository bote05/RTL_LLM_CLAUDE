---
task_id: 09
title: Engine address generator sub-block
type: Foundry RTL generation (LLM-driven)
status: review
depends_on: [00, 01]
unblocks: [Phase 2 integration]
---

# Task 09 — Engine address generator sub-block

## Goal

Generate the address generator sub-block of the shared compute engine. This is the small state machine that walks the convolution loops (output pixels × kernel positions × input channels) and produces the right URAM weight address, BRAM activation address, and bias address each cycle.

The skeleton (task 00) declares the instantiation:

```verilog
// SUBBLOCK: address_generator
address_generator u_address_generator (...);
```

This task fills that stub.

## Deliverable

A single Verilog file at `output/rtl/engine/address_generator.v` containing `module address_generator (...)`.

The module:
- Reads per-layer parameters from the config register block (task 10): `input_channels`, `output_channels`, `kernel_h`, `kernel_w`, `stride_h`, `stride_w`, `padding_h`, `padding_w`, `output_h`, `output_w`, `weight_base_word`, `bias_base_word`.
- Walks the convolution loop: for each `(oc_pass, oh, ow, kh, kw, ic)` produces:
  - Activation BRAM read address (with padding bounds check)
  - URAM weight address (= `weight_base_word + offset(oc_pass, ic, kh, kw)`)
  - Bias address (= `bias_base_word + oc_pass`)
- Produces a `mac_valid` signal that pulses when all three addresses are valid and the weight/activation/bias have been fetched.
- Produces an `output_pixel_done` signal that pulses when all input channels and kernel positions have been accumulated for one output pixel — this signals the requantisation pipeline to consume.
- Produces a `layer_done` signal when all output pixels have been emitted.

## Required ports (must match task 00's spec)

Read `docs/agent_tasks/00_engine_skeleton_spec_PORTS.md`. Expected ports include:

- `clk`, `rst_n`, `engine_start`, `engine_done`
- Per-layer config inputs (input_channels, output_channels, kernels, strides, paddings, output spatial dims, base addresses) — all loaded from task 10's config register block before `engine_start`.
- Output activation BRAM read address: `act_rd_addr[15:0]`, `act_rd_en`, in-bounds flag `act_in_bounds` (for padded positions, force the multiplicand to 0 rather than reading out of bounds).
- URAM weight address: `weight_rd_addr[W:0]`, `weight_rd_en`.
- Bias address: `bias_rd_addr[B:0]`, `bias_rd_en`.
- `mac_valid`, `output_pixel_done`, `layer_done` outputs.

## Address granularity — bias memory layout (LOCKED)

**Decision (single source of truth, referenced by tasks 08 and 10):**

The bias memory is laid out as `ceil(OC / 256)` "wide bias words", where each word packs 256 INT32 biases (8,192 bits = ~22 BRAM18 columns or 2 URAM blocks). One wide bias word feeds exactly one `oc_pass` of the requantisation pipeline.

Therefore the bias address increments by **1** per `oc_pass`, not by 256:

```
bias_rd_addr = bias_base_word + oc_pass
```

`bias_base_word` is per-layer (loaded from task 10's config register block). `oc_pass` is the address generator's internal counter.

This matches the requant pipeline's input (task 08): `bias_in[256 * 32 - 1 : 0]` — one wide bias word per requantisation cycle. The bias memory module is sized to deliver this width in one read.

## Algorithm

The convolution loop in pseudocode:

```
for oc_pass in 0 .. ceil(OC / 256) - 1:        # 256 output channels per pass
  for oh in 0 .. OH-1:
    for ow in 0 .. OW-1:
      clear accumulators
      for kh in 0 .. KH-1:
        for kw in 0 .. KW-1:
          for ic in 0 .. IC-1:
            in_r = oh*SH + kh - PH
            in_c = ow*SW + kw - PW
            if in_r in [0, IH) and in_c in [0, IW):
              act_addr = in_r * IW + in_c   # channel ic is at offset ic within the BRAM word
              weight_addr = weight_base_word + (oc_pass * IC * KH * KW + ic * KH * KW + kh * KW + kw)
              emit (mac_valid=1, addresses set)
            else:
              emit (mac_valid=1, force activation to 0)
      output_pixel_done = 1
      # Requant pipeline consumes here. Bias for this oc_pass is fetched at:
      #     bias_rd_addr = bias_base_word + oc_pass       # ONE wide bias word = 256 INT32 biases
  layer_done = 1
```

## Architecture hints

- The convolution loop is a 6-deep counter: `oc_pass`, `oh`, `ow`, `kh`, `kw`, `ic`. Each counter increments under controlled-overflow chained signals (standard "wraparound + carry-to-next" pattern).
- URAM has 2-cycle read latency; pipeline the addresses so the MAC array sees `act_valid + weight_valid + mac_valid` aligned on the same cycle.
- The `in_bounds` check is a small combinational block — does not need to be registered.
- Apply patterns from `knowledge/patterns/protected/03_conv3x3_pad1.md` (the coord_scheduler structure). Re-use `coord_scheduler.v` if it cleanly fits, otherwise write the loop inline (the engine's address generator is allowed to be specific to the engine's parallelism, not a generic coord scheduler).

## Context to read before starting

- `docs/agent_tasks/00_engine_skeleton_spec_PORTS.md` and `00_engine_skeleton_spec_FSM.md`.
- `docs/nn2rtl_u250_deployment_plan.md §6.1, §6.7`.
- `output/rtl/node_conv_288.v` — the seed reference; its address generator is a smaller instance of what this task produces.
- `output/rtl/coord_scheduler.v` (if it exists from the conv library) — possible reuse.
- `knowledge/patterns/protected/03_conv3x3_pad1.md`.

## How to verify

0. **Port consistency (mandatory)**: `python scripts/check_subblock_ports.py --subblock=address_generator --rtl=output/rtl/engine/address_generator.v` exits 0. Copy port declarations verbatim from `docs/agent_tasks/00_engine_skeleton_spec_PORTS.md`.
1. **Compiles**: `iverilog -t null output/rtl/engine/address_generator.v` passes.
2. **Per-config unit testbench**: feed the address generator with one heavy module's config (e.g. `node_conv_298`: 512×512 3×3 stride 1 padding 1). Confirm the emitted (weight_addr, act_addr, bias_addr) sequence matches a Python golden walk through the same loop.
3. **Multi-layer dispatch test**: feed three different layer configs back-to-back. Confirm the generator resets cleanly between dispatches and produces the right addresses for each config.
4. **Padding bounds correctness**: feed a config with non-zero padding (e.g. `node_conv_196` with padding [3,3]). Confirm `act_in_bounds` goes low for all padded positions and the MAC array would see act_value=0 on those cycles.
5. **`layer_done` matches expected output count**: for a layer with OH × OW = 49 pixels and OC = 512 (= 2 oc_passes), `output_pixel_done` should pulse 49 × 2 = 98 times before `layer_done` pulses.

## Out of scope

- Do NOT actually read from URAM or BRAM. This module only emits addresses + enables. The URAM and BRAM are external blocks driven by these signals.
- Do NOT do the MAC arithmetic. Task 07 owns that.
- Do NOT do requantisation. Task 08 owns that.
- Do NOT touch the engine's top-level FSM. Task 00's skeleton sequences engine_start → address_generator activation → ... → engine_done.

## Success criteria

- `output/rtl/engine/address_generator.v` parses + compiles.
- Unit testbench on `node_conv_298` config matches Python golden bit-exact across the full address sequence.
- Multi-layer dispatch (3 configs back-to-back) emits the right addresses for each, with clean transitions.
- Padding bounds correctness: `act_in_bounds` goes low at every padded position.
- `layer_done` pulses exactly once per dispatched layer.
