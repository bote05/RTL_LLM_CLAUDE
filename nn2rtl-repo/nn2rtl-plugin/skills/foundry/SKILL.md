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
- `data_in` is always packed by channel. For conv/relu, `data_in[i*8 +: 8]` is channel `i` and the port width must be `IC*8`; never use scalar `[7:0]` interfaces there. For add, `data_in[W-1:0]` is the packed lhs bus and `data_in[2W-1:W]` is the packed rhs bus.
- `data_out` is always packed by channel the same way: `data_out[i*8 +: 8]` is channel `i` and the port width must match the output channel count times 8.
- For `op_type=add` modules, `data_in` is a packed wide bus: `data_in[W-1:0] = lhs`, `data_in[2W-1:W] = rhs`, where `W = input_width_bits / 2`. Unpack internally, apply the INT8 quantized-add formula using `lhs_scale_factor`, `rhs_scale_factor`, and `scale_factor`, saturate to INT8, and drive the result on `data_out[W-1:0]`.
- For the module_id `layer0_0_conv1` specifically, implement Conv2d + folded BatchNorm + ReLU + `3x3` stride-2 MaxPool as a single fused pipelined unit. The MaxPool is a sliding-window max across the `3x3` neighborhood with stride 2 in both spatial dimensions, and `pipeline_latency_cycles` already reflects the fused contract.
- **Conv modules must use an output-stationary MAC array. Single-MAC designs are rejected.** Use `OC` parallel signed 8x8 MAC lanes that share the current packed input byte and update `acc[0:OC-1]` together every cycle while `k_counter` walks `ic*kh*kw`. `pipeline_latency_cycles = input_channels * kernel_h * kernel_w + 3` already budgets for this. Deassert `ready_in` while the state machine is busy and accept the next pixel only after `valid_out` fires.

Reference template:

```verilog
// Output-stationary MAC array: OC parallel 8x8 MAC units share the input byte
// each cycle. A k_counter walks (ic, kh, kw); after IC*KH*KW cycles the OC
// accumulators hold full dot products and get scaled/clamped/packed to data_out.
// pipeline_latency_cycles = IC*KH*KW + 3 (fetch, mul, acc, out).
input  wire [IC*8-1:0] data_in;
output reg  [OC*8-1:0] data_out;
reg signed [31:0] acc [0:OC-1];
```

## Output Requirements

- Return a full `VerilogModule`
- `generated_by` must be `Foundry`
- `attempt` starts at `1`
- Persist the source with `write_verilog`
