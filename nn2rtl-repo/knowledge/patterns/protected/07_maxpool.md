# 07 — MaxPool

> **Tile-ABI addendum (canonical for `io_mode == "channel_tiled"`)**: under the
> `tiled-streaming` contract, `input_width_bits == output_width_bits ==
> channel_tile*8` (default 256 for `channel_tile=32`). Maxpool with stride
> > 1 still emits ALL tiles of every KEPT output pixel — it never collapses
> the channel dimension. Per kept pixel the layer emits `ceil(C /
> channel_tile)` output tile beats (matching the input cadence). The
> public wrapper sees `channel_tile` lanes per beat, but the internal
> `line_buf_window` stores the full assembled channel count per spatial
> row, identical to the spatial-conv geometry from `03_conv3x3_pad1.md`. See
> `knowledge/patterns/protected/01_context.md` §"Bus convention —
> CANONICAL tiled-streaming ABI" for full rules.

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

## Architecture — split-module like spatial conv (single-lbw IC=full)

Maxpool uses the SAME line-buffer + window machinery as spatial conv.
Do not declare a `line_buf` array or write your own row/col FSM in the
maxpool top-level. Instantiate two of the three library modules:

- `rtl_library/coord_scheduler.v` — same wrap/stride/padding/drain
  contract as spatial conv. Parameter mapping for maxpool:
  `KH=kernel_size[0], KW=kernel_size[1]`,
  `SH=pool_stride[0], SW=pool_stride[1]`,
  `PH=pool_padding[0], PW=pool_padding[1]`.
- `rtl_library/line_buf_window.v` — **EXACTLY ONE INSTANCE** with
  `IC = full_channel_count` (e.g. IC=64 for ResNet's `node_max_pool2d`).
  `data_in` is `IC*8` bits wide (the full assembled pixel). Exposes
  `window_flat`, takes `frame_start = start_pulse`.

The third library module (`conv_datapath.v`) is REPLACED by a
maxpool-specific compare tree because there is no MAC accumulation.
The compare tree consumes `window_flat` and produces `data_out` on
`sched_output_fires`.

> ⛔ **DO NOT** instantiate one `line_buf_window` per tile (i.e. multiple
> instances with `IC=channel_tile` each, fed in parallel from
> tile-latched data). That topology has a coordination bug in the
> two lbws' `bypass_reg` / `q_array` data path when one sees live
> `data_in` and the other sees a registered tile latch — it produces
> ~25% byte-correct output that looks superficially close but is
> systematically wrong on tile 0. The proven shape is the
> SINGLE-LBW-IC=FULL pattern shown below, modelled after
> `node_conv_200` / `conv3x3_passing_reference.v`.

## Tile-32 ABI wrapper layout (REQUIRED for `io_mode==channel_tiled`)

The tiled-streaming wrapper around the single-lbw core has three required
rules:

**Input beat-aggregator** (combine 2 input beats into one full pixel
before feeding `line_buf_window`):

```verilog
reg                  in_beat_idx;
reg [TILE_BITS-1:0]  pixel_low_r;        // captured beat 0
// Beat 0 (in_beat_idx==0): pixel_low_r <= data_in; in_beat_idx <= 1.
// Beat 1 (in_beat_idx==1): combinational lib_data_in_w = {data_in, pixel_low_r}.
wire lib_valid_in_w               = beat1_now;
wire [IN_PIXEL_BITS-1:0] lib_data_in_w = {data_in, pixel_low_r};
```

**ready_in gating** — hold `ready_in` LOW during the `start_pulse`
cycle so the scheduler/window frame reset is not mixed with a public-bus
handshake. The wrapper may still capture beat 0 one cycle before
`start_pulse`; the input beat-aggregator rule below is what preserves
that beat correctly.

```verilog
assign ready_in = start_pulse              ? 1'b0
                : (in_beat_idx == 1'b0)    ? 1'b1
                :                            sched_ready_in;
```

The wrapper may also accept beat 0 one cycle BEFORE the scheduler sees
`start_pulse` (this is the same cadence used by `node_conv_200`): after
reset, `ready_in` can be high while the wrapper is arming. Therefore the
input beat-aggregator MUST NOT reset `in_beat_idx` or clear
`pixel_low_r` merely because `start_pulse` is high. If it does, the TB
has already advanced past beat 0, so beat 1 becomes the next "low" tile;
the first output then shows the exact signature
`got[0..31] == expected[32..63]`, followed by widespread partial
corruption.

```verilog
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        in_beat_idx <= 1'b0;
        pixel_low_r <= {TILE_BITS{1'b0}};
    end else if (input_handshake) begin
        if (in_beat_idx == 1'b0) begin
            pixel_low_r <= data_in;
            in_beat_idx <= 1'b1;
        end else begin
            in_beat_idx <= 1'b0;
        end
    end
end
```

> ⛔ **DO NOT** add `else if (start_pulse) in_beat_idx <= 1'b0` to the
> input beat-aggregator. That discards a pre-start captured beat 0 and
> creates the same tile-swap signature as the forbidden `ST_LATCH`
> output FSM.

**Output beat-splitter** (split the 512-bit `max_pack` into 2 × 256-bit
beats over 2 cycles):

```verilog
reg                       out_beat1_pending_r;
reg [TILE_BITS-1:0]       out_pixel_high_r;
// On sched_output_fires:
//   data_out         <= max_pack_w[TILE_BITS-1:0];        // beat 0 (low tile)
//   out_pixel_high_r <= max_pack_w[OUT_PIXEL_BITS-1:TILE_BITS]; // hold high tile
//   valid_out        <= 1; out_beat1_pending_r <= 1; compute_busy <= 1;
// Next cycle (out_beat1_pending_r):
//   data_out  <= out_pixel_high_r;                         // beat 1 (high tile)
//   valid_out <= 1; out_beat1_pending_r <= 0; compute_busy <= 0;
// Drive stall_in = compute_busy so scheduler stays frozen during the
// 2-cycle drain.
```

## Required FSM

- `start_pulse` cycle-after-reset, identical to spatial conv. Wire
  it to `coord_scheduler.start` and `line_buf_window.frame_start`.
- On each `sched_output_fires`, **emit beat 0 in the SAME cycle**
  (data_out NBA-assigned from `max_pack_w[TILE_BITS-1:0]`, valid_out
  NBA-set to 1, and latch the high tile into `out_pixel_high_r`).
  Next cycle: emit beat 1 from `out_pixel_high_r`. **2-STATE FSM**
  (`ST_IDLE` + `out_beat1_pending_r` flag) — see the "Output
  beat-splitter" block above for the exact recipe. Drive
  `stall_in = (sched_output_fires || out_beat1_pending_r)` so the
  scheduler freezes through the 2-cycle drain.

> ⛔ **DO NOT** introduce an intermediate `ST_LATCH` state that
> consumes `sched_output_fires` and only emits beat 0 the NEXT
> cycle. That 3-state FSM shifts the beat sequence by one cycle
> relative to what the testbench expects and produces the
> "got[0..31] == expected[32..63]" tile-swap pattern. Beat 0 must
> emit the SAME cycle as `sched_output_fires=1`.

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
`knowledge/references/{protected,active,probationary}/` tiers yet — for
`op_type=maxpool` the `get_rtl_patterns` tool returns
`conv3x3_passing_reference.v` as the closest anchor (see
`mcp/tools.ts:get_rtl_patterns`). Adapt the conv3x3 reference's
**single-instance** `line_buf_window` + `coord_scheduler` wiring and
replace `conv_datapath` with the compare tree shown below. The
beat-aggregator and beat-splitter wrappers are required for
`io_mode=channel_tiled` — see the "Tile-32 ABI wrapper layout"
section above.

Once a maxpool dispatch passes byte-exact sim under tiled-streaming,
promote that `.v` to `probationary/maxpool_passing_reference.v` and
update `mcp/tools.ts:get_rtl_patterns` to prefer it over the conv3x3
fallback for `op_type=maxpool`.

## Known failure modes

- `loop_bounds_incorrect` — iterating `KH*KW-1` instead of `KH*KW`, dropping
  a pixel from the max.
- `sign_extension_error` — compare in unsigned context, so a negative
  pixel's two's-complement representation looks "larger" than positives.
- `output_counter_missing` — solved by the coord_scheduler instantiation;
  its `outputs_emitted` output bounds the stream deterministically. Rolling
  a custom counter reintroduces the hang class.
