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
// Sign-aware rounding (same rule as conv2d). Verilog >>> always floors
// toward -inf, so the bias for negatives is (HALF - 1), NOT -HALF.
half          = 1 << (OUT_SCALE_SHIFT - 1)
sign_bias     = (sum32 >= 0) ? half : (half - 1)
out32         = (sum32 + sign_bias) >>> OUT_SCALE_SHIFT
out8          = saturate_int8(out32)
data_out[i*8 +: 8] = out8
```

`LHS_SCALE_MULT / 2^LHS_SCALE_SHIFT ≈ lhs_scale_factor`, similarly for rhs
and the output. Use `computeScaleApprox` in `sdk/orchestrate.ts` as the
reference for picking constants.

## Latency contract

Use the LayerIR value; do not re-derive. The current resource-bounded add
contract is serialized over channels:

```
pipeline_latency_cycles = output_channels + 3
```

The cycle budget is one packed-pixel capture cycle, then one channel per
cycle through a three-stage arithmetic pipe:

1. **Stage 1 — multiplies.** For the current channel only,
   `lhs_term <= lhs_ch * LHS_FUSED_MULT` and
   `rhs_term <= rhs_ch * RHS_FUSED_MULT`. With `(* use_dsp = "yes" *)`
   on each registered product, Vivado infers two DSP48E1 multipliers total.
2. **Stage 2 — sum + sign-aware round bias.** Sum the two terms and add
   the sign-aware bias in the same registered stage (single cycle to keep
   the 3-stage latency contract). Declare the unbiased combinational sum
   as a **module-scope `wire`** (Verilog-2001 forbids `wire` decls inside
   an `always` block); reference it from inside the always:
   ```verilog
   // module scope, alongside other wire decls:
   wire signed [SUM_W-1:0] sum_pre = lhs_term + rhs_term;

   // inside the stage-2 always block:
   sum_term <= sum_pre + (sum_pre[SUM_W-1] ? (FUSED_HALF - 1) : FUSED_HALF);
   ```
   where `FUSED_HALF = 1 << (FUSED_SHIFT - 1)`. Adding `-FUSED_HALF` for
   negatives over-rounds (Verilog `>>>` already floors toward -inf).
3. **Stage 3 — shift + saturate.** `(sum_term >>> FUSED_SHIFT)`,
   clamp to INT8, and write only the current channel slice of `data_out`.

Do not instantiate one multiplier per channel. The old 3-cycle fully parallel
implementation is numerically correct but architecturally bad at OC=256: it
creates 512 constant multipliers, consumes all 240 Artix-7 DSPs, spills the
remaining multipliers into LUTs, and turns residual add into the largest LUT
consumer in layer 1.

## Required FSM

Use a two-state `IDLE/RUN` controller:

- `IDLE`: `ready_in=1`; when `valid_in` is high, capture the full packed
  `data_in` into `input_buf`, clear `ch_idx`, deassert `ready_in`, enter `RUN`.
- `RUN`: stream channels `0..OC-1` through the three arithmetic stages,
  writing `data_out[ch_idx*8 +: 8]` as each channel exits stage 3.
- Assert `valid_out` for one cycle when channel `OC-1` is written, then
  return to `IDLE` and reassert `ready_in`.

The output-counter preflight rule does not apply to `add` (it is triggered by
op_type; add is excluded).

## Required registers

- `reg [INPUT_WIDTH-1:0] input_buf;`
- `reg [OUTPUT_WIDTH-1:0] data_out;`
- `reg [CH_IDX_W-1:0] ch_idx, stage1_idx, stage2_idx, stage3_idx;`
- `reg stage1_valid, stage2_valid, stage3_valid;`
- `(* use_dsp = "yes" *) reg signed [PROD_W-1:0] lhs_term;`
- `(* use_dsp = "yes" *) reg signed [PROD_W-1:0] rhs_term;`
- `reg signed [SUM_W-1:0] sum_term;`
- No weights, no biases, no `$readmemh`.

`PROD_W = 8 + SCALE_CONST_W` (8-bit input x `SCALE_CONST_W`-bit fused
multiplier). `SUM_W = PROD_W + 2` to absorb the lhs+rhs sum and round bias.

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
