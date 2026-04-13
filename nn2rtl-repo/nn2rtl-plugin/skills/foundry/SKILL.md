---
name: foundry
description: Verilog synthesis rules for nn2rtl op types, including fixed-point conv2d, batchnorm, relu, add, and example RTL snippets.
---
# Foundry Skill

Use this skill when generating synthesizable Verilog from a single `LayerIR`.

## Global RTL Rules

- All arithmetic is fixed-point INT8 unless an accumulator requires wider width.
- Every multiply is `8x8 -> 16 bit`.
- Use signed math where appropriate.
- No simulation-only constructs in synthesizable modules.
- Residual adds must saturate to the target output width.

## `conv2d`

- Implement MAC accumulation with a widened signed accumulator.
- Apply scaling only after accumulation unless the spec explicitly requires per-product scaling.

Example snippet:

```verilog
wire signed [15:0] product;
reg  signed [31:0] acc;

assign product = $signed(input_val) * $signed(weight_val);
```

## `batchnorm`

- Lower batch normalization to fixed-point multiply-plus-shift.
- Keep shift direction explicit and signed.

Example snippet:

```verilog
wire signed [15:0] bn_scaled;
assign bn_scaled = $signed(in_val) * $signed(scale_q8_8);
assign out_val = bn_scaled >>> shift_amount;
```

## `relu`

- Clamp negative values to zero.

Example snippet:

```verilog
assign relu_out = in_val[7] ? 8'sd0 : in_val;
```

## `add`

- Use saturation arithmetic for residual merges.

Example snippet:

```verilog
wire signed [8:0] sum_ext;
assign sum_ext = $signed(a) + $signed(b);
assign sum_sat =
  (sum_ext > 9'sd127) ? 8'sd127 :
  (sum_ext < -9'sd128) ? -8'sd128 :
  sum_ext[7:0];
```

## Output Requirements

- Return a full `VerilogModule` JSON object.
- `generated_by` must be `Foundry`.
- `attempt` starts at `0`.
- Persist the source with `write_verilog`.
