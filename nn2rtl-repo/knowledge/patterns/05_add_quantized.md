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

`pipeline_latency_cycles == 3`. Use the LayerIR value; do not re-derive.
The cycle budget breaks down as one pipeline stage per registered hop:

1. **Stage 1 — multiplies.** Per channel,
   `lhs_term <= lhs_ch * LHS_FUSED_MULT` and
   `rhs_term <= rhs_ch * RHS_FUSED_MULT`. With `(* use_dsp = "yes" *)`
   on each registered product, Vivado infers the DSP48E1 MREG=1 path.
2. **Stage 2 — sum + round bias.**
   `sum_term <= lhs_term + rhs_term + FUSED_ROUND_BIAS`.
3. **Stage 3 — shift + saturate.** `(sum_term >>> FUSED_SHIFT)`,
   clamp to INT8, pack into `data_out`. `valid_out` registers along
   the same chain (`valid_out <= valid_q2`).

The single-cycle combinational implementation that the legacy
`computeScaleApprox` / Foundry templates produced does NOT close
timing on Artix-7 100T at OC=256: a 256-channel residual add saturates
all 240 DSP slices and pushes the rest of the multiplies into a wide
LUT-mul cone whose end-to-end depth blows past the 20 ns clock. The
3-stage pipeline restores Fmax without changing functional behavior.

## Required FSM

No explicit state register is needed — every stage is unconditionally
clocked, valid propagates through `valid_q1 -> valid_q2 -> valid_out`
with the same depth as the data path. `ready_in` is held high (the
pipe is always ready to accept the next sample); the output-counter
preflight rule does not apply to `add` (it's triggered by op_type;
add is excluded).

## Required registers

- `(* use_dsp = "yes" *) reg signed [PROD_W-1:0] lhs_term [0:OC-1];`
- `(* use_dsp = "yes" *) reg signed [PROD_W-1:0] rhs_term [0:OC-1];`
- `reg signed [SUM_W-1:0] sum_term [0:OC-1];`
- `reg valid_q1, valid_q2;`
- No weights, no biases, no `$readmemh`.

`PROD_W = 8 + SCALE_CONST_W` (8-bit input × `SCALE_CONST_W`-bit fused
multiplier). `SUM_W = PROD_W + 1` to absorb the lhs+rhs sum.

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
