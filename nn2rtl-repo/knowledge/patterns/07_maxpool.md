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

## Required FSM

- `ST_STREAM` — on the scheduler's `ready_in && valid_in` handshake, write
  the input pixel into `line_buf[cur_row][sched_in_col]`. When
  `sched_output_fires` asserts, transition to `ST_COMPARE`.
- `ST_COMPARE` — for each of OC channels, compute max across the KH*KW
  window. Single-cycle combinational; no MAC pipeline, no OC_PASSES
  (there are no weights to iterate).
- `ST_OUTPUT` — emit `data_out` and drop `stall_in` so the scheduler
  advances past the firing coord and its `outputs_emitted` increments.

## coord_scheduler is MANDATORY

Maxpool uses the same coordinate FSM as spatial conv. Instantiate
`rtl_library/coord_scheduler.v` per the contract in
`01_context.md § coord_scheduler contract`; do not roll your own
row/col/wrap/stride/drain logic. Parameter mapping:

- `KH=kernel_size[0], KW=kernel_size[1]`
- `SH=pool_stride[0], SW=pool_stride[1]`
- `PH=pool_padding[0], PW=pool_padding[1]`

The scheduler's `outputs_emitted` is the bounded output counter —
functional code, not just a preflight appeaser.

## Required registers

- `reg signed [7:0] line_buf [0:KH-1][0:IW-1][0:OC-1];`
- `reg signed [7:0] window   [0:KH-1][0:KW-1][0:OC-1];`  (reg, updated in clocked block)
- No weights, no biases, no $readmemh.
- `outputs_emitted` is an OUTPUT of the scheduler instance; do not
  declare a separate reg for it.

Structural preflight's `line_buffer_missing` and `window_not_registered`
rules currently trigger on conv2d only, but maxpool functionally needs
both — keep them.

## Compare tree

```verilog
// Per-channel max across the window
for (c = 0; c < OC; c = c + 1) begin
  max_val = window[0][0][c];
  for (i = 0; i < KH; i = i + 1)
    for (j = 0; j < KW; j = j + 1)
      if ($signed(window[i][j][c]) > $signed(max_val))
        max_val = window[i][j][c];
  data_out[c*8 +: 8] <= max_val;
end
```

## Known failure modes

- `loop_bounds_incorrect` — iterating `KH*KW-1` instead of `KH*KW`, dropping
  a pixel from the max.
- `sign_extension_error` — compare in unsigned context, so a negative
  pixel's two's-complement representation looks "larger" than positives.
- `output_counter_missing` — solved by the coord_scheduler instantiation;
  its `outputs_emitted` output bounds the stream deterministically. Rolling
  a custom counter reintroduces the hang class.
