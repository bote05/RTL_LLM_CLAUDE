# 06 — ReLU

> **Tile-ABI addendum (canonical for `io_mode == "channel_tiled"`)**: under the
> `tiled-streaming` contract, `input_width_bits == output_width_bits ==
> channel_tile*8` (default 256 for `channel_tile=32`). Each `data_in` beat
> carries one tile of `channel_tile` INT8 channels; each `data_out` beat
> carries the same `channel_tile` channels after element-wise `max(0, x)`.
> 1:1 tile cadence — emit exactly one output tile beat per input tile
> beat, no channel collapsing, no buffering across tiles. Pixel and tile
> ordering preserved from input. See `knowledge/patterns/protected/01_context.md`
> §"Bus convention — CANONICAL tiled-streaming ABI" for full rules.

## When to use

`op_type == "relu"`.

## Semantics

Per channel:

```
out_i = (in_i > 0) ? in_i : 8'sd0
```

No scale factor. No saturation (negative clamps to 0, positive is already
INT8). Combinational except for the output register.

## Latency contract

Typically `pipeline_latency_cycles == 1`. Use the LayerIR value.

## Required FSM

A bare register on the output with a ready/valid pipeline stage:

```verilog
always @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    valid_out <= 1'b0;
    ready_in  <= 1'b1;
    data_out  <= 0;
  end else begin
    valid_out <= valid_in;
    if (valid_in) begin
      for (i = 0; i < OC; i = i + 1) begin
        // [INVARIANT:ROUNDING]  -- no rounding needed but left marker-free
        data_out[i*8 +: 8] <= ($signed(data_in[i*8 +: 8]) > 0)
                               ? data_in[i*8 +: 8]
                               : 8'sd0;
      end
    end
  end
end
```

No weights, no biases, no window, no line buffer. The output-counter
preflight rule does not apply.

## Known failure modes

- `sign_extension_error` — comparison performed in unsigned context, so
  all negative channels pass through as large positives. Always use
  `$signed(...)`.
- `saturation_missing` — somebody added redundant min/max clamps and
  introduced a bug. The output of ReLU on INT8 is by construction in
  `[0, 127]`; no extra saturation is needed.

## ReLU6 / clipped activations

Some networks (MobileNetV2 et al.) use ReLU6: `out = clamp(x, 0, 6)` in
float domain. When the ONNX frontend imports such a model, the LayerIR
records `clip_max == 6.0` on the relu layer. Calibration anchors
`output_scale = max(observed, clip_max) / 128`. Critically, the upstream
Conv's output_scale is typically LARGER than this (because the Conv's
pre-clip range exceeds 6). So when the Conv emits INT8 stream X, the
float interpretation is `X · conv_output_scale`, and ReLU6's job is to:

1. Apply the relu non-linearity (negatives → 0).
2. Requantize the stream from the Conv's output_scale to the ReLU6's
   tighter output_scale. The composite is `input_scale / output_scale`,
   identical to the conv requantize tail style.
3. Clamp to INT8 [-128, 127]. With output_scale = 6/128, this clamp is
   exactly the float clip-at-6 (and clip-at-(-6), which relu has already
   ruled out).

The RTL form mirrors the relu body PLUS a SCALE_MULT / SCALE_SHIFT stage
when `clip_max` is present on the LayerIR:

```verilog
// Standard ReLU body
relu_out[i*8 +: 8] = ($signed(data_in[i*8 +: 8]) > 0)
                       ? data_in[i*8 +: 8]
                       : 8'sd0;
// ReLU6 requantize (only when LayerIR.clip_max is finite)
scaled_int32 = ($signed(relu_out) * SCALE_MULT) >>> SCALE_SHIFT;   // round half toward +inf
data_out[i*8 +: 8] = clamp(scaled_int32, -128, 127);
```

`SCALE_MULT` / `SCALE_SHIFT` are derived from `scale_factor` in the
LayerIR the same way conv does. When `clip_max` is absent (plain ReLU),
the scale_factor equals 1.0 and the requantize stage is a passthrough —
the body is exactly the historical `(in > 0) ? in : 0` form.
