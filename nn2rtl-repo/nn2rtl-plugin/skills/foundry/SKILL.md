---
name: foundry
description: Verilog synthesis rules for nn2rtl op types, including BRAM-backed weight loading, exact latency contracts, and valid/ready handshake requirements.
---
# Foundry Skill

Use this skill when generating synthesizable Verilog from a single `LayerIR`.

Supported `op_type` values: `conv2d`, `relu`, `add`, `maxpool`.

## Global RTL Rules

- Load weights via `$readmemh(weights_path, ...)`
- Load bias via `$readmemh(bias_path, ...)` when present
- Never hardcode weight arrays into source
- All arithmetic is signed fixed-point
- Every multiply is `8x8 -> 16 bit`
- For conv modules, derive `ACC_W`, `BIASED_W`, `SCALED_W`, and the clamp temporary width from `K_TOTAL`, the fixed `BIAS_W=32`, and the chosen `SCALE_MULT`; never hardcode `acc` to `32` bits or `scaled`/`v_tmp` to `48` bits
- Residual add saturates back to INT8
- Public port names are canonical: `clk`, `rst_n` (active-low), `valid_in`, `ready_in`, `data_in`, `valid_out`, `data_out`. The static testbench enforces these exactly — any other name fails before simulation.
- `ready_in` is a module **output** (backpressure upstream). If stalling is not needed, tie high after reset.
- `valid_out` must assert exactly `pipeline_latency_cycles` cycles after the first `valid_in` of a vector.
- `pipeline_latency_cycles` from the LayerIR is authoritative. Do not override it with a hand-derived formula.
- `data_out` is sampled by the bench only when `valid_out == 1`, so bubbles are allowed between valid outputs.
- `data_in` is always packed by channel. For conv/relu, `data_in[i*8 +: 8]` is channel `i` and the port width must be `IC*8`; never use scalar `[7:0]` interfaces there. For add, `data_in[W-1:0]` is the packed lhs bus and `data_in[2W-1:W]` is the packed rhs bus.
- `data_out` is always packed by channel the same way: `data_out[i*8 +: 8]` is channel `i` and the port width must match the output channel count times 8.
- For `op_type=add` modules, `data_in` is a packed wide bus: `data_in[W-1:0] = lhs`, `data_in[2W-1:W] = rhs`, where `W = input_width_bits / 2`. Unpack internally, apply the INT8 quantized-add formula using `lhs_scale_factor`, `rhs_scale_factor`, and `scale_factor`, saturate to INT8, and drive the result on `data_out[W-1:0]`.
- For conv2d layers, if `stride` / `padding` are present in the LayerIR, use them exactly. Do not infer them when explicit geometry is already available.
- `layer0_0_conv1` must follow the current LayerIR / golden-vector contract. On the current legacy `.pth` path it is not a fused MaxPool stage; do not add extra fused stages unless the current LayerIR / goldens require them.
- **Conv modules must use the pattern-file conv architecture. Single-output-channel scalar designs are rejected.** In the current verified patterns, `mac_parallelism` is an accumulator-group size, not a promise of MP parallel memory reads: one `lane_counter`-selected lane issues one weight read / multiply / accumulate per cycle, and OC is covered by `OC_PASSES = ceil(OC / mac_parallelism)`. Deassert `ready_in` while the state machine is busy and accept the next pixel only after `valid_out` fires.

Reference template:

```verilog
// Serialized output-stationary MAC group: lane_counter selects one accumulator
// lane per cycle. A k_counter walks (ic, kh, kw); after MP*K_TOTAL issue cycles
// the current OC group holds full dot products and gets scaled/clamped/packed.
// Honour the exact pipeline_latency_cycles from LayerIR; do not recompute it here.
input  wire [IC*8-1:0] data_in;
output reg  [OC*8-1:0] data_out;
localparam integer ACC_W = 16 + $clog2(K_TOTAL);
reg signed [ACC_W-1:0] acc [0:MP-1];
```

## Output Requirements

- Return a full `VerilogModule`
- `generated_by` must be `Foundry`
- `attempt` starts at `1`
- Persist the source with `write_verilog`
