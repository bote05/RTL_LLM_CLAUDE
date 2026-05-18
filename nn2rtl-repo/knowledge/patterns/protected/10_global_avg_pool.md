# 10 — Global Average Pool

## When to use

`op_type == "global_avg_pool"`. LayerIR carries `gap_spatial = [H, W]` — the
spatial dims of the input feature map. Output is `[N, C, 1, 1]` (the per-
channel mean over the H·W cells), expressed as INT8.

## Semantics

Per channel `c`:

```
acc[c]    = Σ_{h, w} in[c, h, w]                    # INT32 accumulator
mean[c]   = acc[c] / (H · W)                        # not done explicitly
out[c]    = clamp(round( acc[c] · SCALE_MULT >> SCALE_SHIFT ), -128, 127)
```

The 1/(H·W) divisor is **folded into SCALE_MULT / SCALE_SHIFT** by the
golden generator. The RTL does NOT need a divider. Read
`scale_factor` from the LayerIR and convert to (SCALE_MULT, SCALE_SHIFT)
the same way conv does. The composite is:

```
SCALE_FLOAT = input_scale / output_scale / (H · W)
```

The accumulator width has to be wide enough to hold the worst case:
`max(|in|) · H · W` over INT8 inputs is `127 · H · W`. For MobileNetV2's
`H·W = 49` (7×7 spatial), 127·49 = 6,223 — fits in INT16. For larger spatial
(`H·W ≤ 2^15`), use INT32 to stay safe.

## Latency contract

```
fill_cycles    = 1                                  # input-latch only
reduce_cycles  = H · W                              # one cell per cycle
tail_cycles    = 3                                  # SCALE + CLAMP + OUTPUT
latency        = fill_cycles + reduce_cycles + tail_cycles
```

For 7×7: latency ≈ 53 cycles. Read `pipeline_latency_cycles` from the
LayerIR — Foundry must not re-derive a different number.

## Architecture hints

- One INT32 accumulator per channel. With `mac_parallelism = MP`, declare
  the accumulator as `MP` independent lanes; iterate over `C / MP` passes.
- A single spatial counter drives the "consume one cell" loop. When the
  counter reaches `H · W`, emit the channel-c result via the requantize
  tail and reset.
- The requantize tail mirrors the conv tail exactly: BIAS (no bias for GAP,
  pass-through), SCALE (mul by SCALE_MULT, right-shift by SCALE_SHIFT,
  round-half-toward-+inf), CLAMP to INT8.
- Inputs arrive on the standard channel-bus (`data_in[C*8-1:0]`); the bus
  carries one spatial cell per beat, all `C` channels in parallel. Do not
  declare a line buffer — GAP needs no spatial window.

## Required public interface

Same canonical interface as every other op: `clk, rst_n, valid_in, ready_in,
data_in, valid_out, data_out`. Width fields come from the LayerIR.

## Known failure modes

- `accumulator_width_too_small` — using INT16 when H·W exceeds the range.
  Inflate to INT32 unless you can prove the worst case fits.
- `divider_present` — somebody added `acc / (H*W)` thinking the layer needs
  a runtime divide. The 1/(H·W) is in SCALE_MULT. Adding the divider
  doubles the LUT cost and adds latency for nothing.
- `output_shape_dropped` — emitting `[N, C]` instead of `[N, C, 1, 1]` on
  the bus. The downstream Gemm (if present) absorbs either layout, but the
  Verilator goldens are packed as one beat per output, so any extra spatial
  must be flat-packed in the same beat order. Keep the bus width = `C*8`.
- `divide_truncates_toward_zero` — rounding the SCALE result toward zero
  instead of half-toward-+infinity. The conv requantize tail's rounding
  semantics apply identically. Use the same round-half-up primitive.
