---
name: foundry
description: Verilog synthesis rules for nn2rtl op types, including BRAM-backed weight loading, exact latency contracts, and valid/ready handshake requirements.
---
# Foundry Skill

Use this skill when generating synthesizable Verilog from a single `LayerIR`.

## Global RTL Rules

- Load weights via `$readmemh(weights_path, ...)`
- Load bias via `$readmemh(bias_path, ...)` when present
- Never hardcode weight arrays into source
- All arithmetic is signed fixed-point
- Every multiply is `8x8 -> 16 bit`
- Residual add saturates back to INT8
- Public port names are canonical: `clk`, `rst_n` (active-low), `valid_in`, `ready_in`, `data_in`, `valid_out`, `data_out`. The static testbench enforces these exactly — any other name fails before simulation.
- `ready_in` is a module **output** (backpressure upstream). If stalling is not needed, tie high after reset.
- `valid_out` must assert exactly `pipeline_latency_cycles` cycles after the first `valid_in` of a vector.
- `data_out` is sampled by the bench only when `valid_out == 1`, so bubbles are allowed between valid outputs.

## Output Requirements

- Return a full `VerilogModule`
- `generated_by` must be `Foundry`
- `attempt` starts at `1`
- Persist the source with `write_verilog`
