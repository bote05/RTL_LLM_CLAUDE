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
- Public interface must honor `clock_signal`, `reset_signal`, `valid_in_signal`, and `valid_out_signal`
- `valid_out` must assert exactly `pipeline_latency_cycles` cycles after `valid_in`

## Output Requirements

- Return a full `VerilogModule`
- `generated_by` must be `Foundry`
- `attempt` starts at `1`
- Persist the source with `write_verilog`
