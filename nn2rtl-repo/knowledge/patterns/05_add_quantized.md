# 05 — INT8 quantized residual add

## When to use

`op_type == "add"`. This is ResNet's residual add: two tensors of the same
shape and channel count combined, each with its own scale factor.

## Input layout

`data_in` is packed as `[rhs | lhs]`:

```
data_in[W-1:0]     = lhs  (OC channels * 8 bits)
data_in[2W-1:W]    = rhs  (OC channels * 8 bits)
```

where `W = output_width_bits`. `input_width_bits == 2 * output_width_bits`.

The orchestrator validates this packing on LayerIR load; if the LayerIR
violates it, the pipeline fails fast before Foundry dispatch.

## Quantization formula

For each channel `i`:

```
lhs_i  = $signed(data_in[i*8 +: 8])                                  // INT8
rhs_i  = $signed(data_in[W + i*8 +: 8])                              // INT8
sum32  = lhs_i * LHS_SCALE_MULT + rhs_i * RHS_SCALE_MULT              // INT32
out32  = (sum32 + OUT_SCALE_ROUND_BIAS) >>> OUT_SCALE_SHIFT
out8   = saturate_int8(out32)
data_out[i*8 +: 8] = out8
```

`LHS_SCALE_MULT / 2^LHS_SCALE_SHIFT ≈ lhs_scale_factor`, similarly for rhs
and the output. Use `computeScaleApprox` in `sdk/orchestrate.ts` as the
reference for picking constants.

## Latency contract

Usually `pipeline_latency_cycles == 2` or `3` — a single-pixel combinational
path plus the output register. Use the LayerIR value; do not re-derive.

## Required FSM states

An add is essentially combinational. Typical structure:

- `ST_WAIT` — wait for `valid_in`; latch lhs/rhs channels.
- `ST_COMPUTE` — compute `scaled[i]` for all i in one cycle (combinational
  multiply-add-shift) and register the result.
- `ST_OUTPUT` — assert `valid_out`, return to ST_WAIT.

For small OC this can be collapsed to a single pipelined stage without an
explicit state register. The output-counter preflight rule does not apply
to `add` (it's triggered by op_type; add is excluded).

## Required registers

- `reg signed [7:0]  lhs_latch [0:OC-1];`
- `reg signed [7:0]  rhs_latch [0:OC-1];`
- `reg signed [31:0] scaled    [0:OC-1];`
- No weights, no biases, no $readmemh.

## Rules

- Every arithmetic signal on the datapath MUST be declared `signed` or
  wrapped with `$signed(...)`. A silently unsigned residual add produces
  mass saturation at ±127 on negative residuals — a classic symptom of
  `sign_extension_error`.
- Saturate to INT8 at the output, not internally. Internal widening is
  free; internal truncation discards information.

## Known failure modes

- `sign_extension_error` — missing `$signed()` around one of the operands
  of the multiply. Outputs saturate to 127 for negative residuals.
- `scale_factor_misapplied` — using `scale_factor` where `lhs_scale_factor`
  or `rhs_scale_factor` was meant, or vice versa.
- `port_width_mismatch` — `data_in` declared as `OC*8` instead of
  `OC*16`, because Foundry missed the lhs|rhs packing.
