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
- For `op_type=add` modules, `data_in` is a packed wide bus: `data_in[W-1:0] = lhs`, `data_in[2W-1:W] = rhs`, where `W = input_width_bits / 2`. Unpack internally, apply the INT8 quantized-add formula using `lhs_scale_factor`, `rhs_scale_factor`, and `scale_factor`, saturate to INT8, and drive the result on `data_out[W-1:0]`.
- For the module_id `layer0_0_conv1` specifically, implement Conv2d + folded BatchNorm + ReLU + `3x3` stride-2 MaxPool as a single fused pipelined unit. The MaxPool is a sliding-window max across the `3x3` neighborhood with stride 2 in both spatial dimensions, and `pipeline_latency_cycles` already reflects the fused contract.

## Output Requirements

- Return a full `VerilogModule`
- `generated_by` must be `Foundry`
- `attempt` starts at `1`
- Persist the source with `write_verilog`
