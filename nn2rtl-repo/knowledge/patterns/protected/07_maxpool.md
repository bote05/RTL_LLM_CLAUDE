# 07 — MaxPool

## When to use

`op_type == "maxpool"`. LayerIR carries `kernel_size`, `pool_stride`, and
`pool_padding` (each a 2-element `[H, W]` array). ResNet's typical config
is 3×3 kernel, stride=2, padding=1.

## Latency contract

Same formula family as spatial conv, minus the MAC pipeline — each
output is just a compare tree across `KH * KW * OC` values, which takes
constant cycles:

```
fill_rows = max(KH - 1 - PH, 0)
fill_cols = max(KW - PW, 1)
pipeline_stages = 3                    # LATCH + COMPARE + OUTPUT
latency = fill_rows * (IW + PW) + fill_cols + pipeline_stages
```

Consult the LayerIR's `pipeline_latency_cycles` for the authoritative
value — do not re-derive.

## Architecture — split-module like spatial conv

Maxpool uses the SAME line-buffer + window machinery as spatial conv.
Do not declare a `line_buf` array or write your own row/col FSM in the
maxpool top-level. Instantiate two of the three library modules:

- `rtl_library/coord_scheduler.v` — same wrap/stride/padding/drain
  contract as spatial conv. Parameter mapping for maxpool:
  `KH=kernel_size[0], KW=kernel_size[1]`,
  `SH=pool_stride[0], SW=pool_stride[1]`,
  `PH=pool_padding[0], PW=pool_padding[1]`.
- `rtl_library/line_buf_window.v` — KH per-slot BRAM-inferred line
  buffer + KH×KW×OC registered shift-window (parameterise with
  `IC = OC` since maxpool's "channels" are the input channels carried
  through to the output). Exposes `window_flat`, takes
  `frame_start = start_pulse`.

The third library module (`conv_datapath.v`) is REPLACED by a
maxpool-specific compare tree because there is no MAC accumulation.
The compare tree consumes `window_flat` and produces `data_out` on
`sched_output_fires`.

## Required FSM

- `start_pulse` cycle-after-reset, identical to spatial conv. Wire
  it to `coord_scheduler.start` and `line_buf_window.frame_start`.
- On each `sched_output_fires`, latch the per-channel max across
  the window into a small ST_LATCH register, then emit `data_out`
  and `valid_out` over the next 1–2 cycles. Drive
  `stall_in = compare_busy` so the scheduler freezes through the
  compare-and-emit window the same way conv freezes through its
  MAC pass.

## Required registers (top-level)

- A few-stage compare-tree register array (sized OC × KH*KW slots
  → log2(KH*KW) compare stages), all in fabric flops.
- The `start_pulse` reg + (optional) `pending_rearm` per the
  conv references' multi-frame contract.
- `outputs_emitted` is an OUTPUT of the scheduler instance; do not
  declare a separate reg for it.

The structural preflight rules `line_buffer_missing` /
`window_not_registered` are skipped when `line_buf_window` is
instantiated (the library owns those concerns).

## Compare tree

The window is consumed via `window_flat` from the library module,
indexed the same way `conv_datapath::tap_at()` does:
`window_flat[(kh*KW*OC + kw*OC + c)*8 +: 8]` is `window[kh][kw][c]`
as a signed INT8.

```verilog
// Per-channel max across the window. Drive the compare tree from
// window_flat; do NOT declare a separate `window` reg array.
function [7:0] tap_at;
    input integer kh_idx, kw_idx, c_idx;
    begin
        tap_at = window_flat[(kh_idx*KW*OC + kw_idx*OC + c_idx)*8 +: 8];
    end
endfunction

// Per-channel max (single-cycle combinational on small KH*KW).
for (c = 0; c < OC; c = c + 1) begin
    max_val = $signed(tap_at(0, 0, c));
    for (i = 0; i < KH; i = i + 1)
        for (j = 0; j < KW; j = j + 1)
            if ($signed(tap_at(i, j, c)) > max_val)
                max_val = $signed(tap_at(i, j, c));
    data_out[c*8 +: 8] <= max_val;
end
```

## Reference status

There is no proven-passing maxpool reference in the readable
`knowledge/references/{protected,active,probationary}/` tiers yet; the
`layer0_0_maxpool` module of ResNet-50 was excluded from the current
LayerIR set and so has not exercised the library. Promote a passing
maxpool to `probationary/maxpool_passing_reference.v`, then to `active/`
once validated, so Foundry can adapt it the same way the conv references
work.

## Known failure modes

- `loop_bounds_incorrect` — iterating `KH*KW-1` instead of `KH*KW`, dropping
  a pixel from the max.
- `sign_extension_error` — compare in unsigned context, so a negative
  pixel's two's-complement representation looks "larger" than positives.
- `output_counter_missing` — solved by the coord_scheduler instantiation;
  its `outputs_emitted` output bounds the stream deterministically. Rolling
  a custom counter reintroduces the hang class.
