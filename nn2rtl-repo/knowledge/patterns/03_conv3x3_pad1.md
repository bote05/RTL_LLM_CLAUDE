# 03 — Spatial 3×3 conv, padding=1

## When to use

`op_type == "conv2d"` with `weight_shape[2] == 3 && weight_shape[3] == 3`.

## Latency contract

From `scripts/golden_impl.py::compute_conv2d_latency_cycles` spatial branch:

```
IW = input_shape[3]
PH = padding[0], PW = padding[1]
K_TOTAL = IC * KH * KW = IC * 9
MP = mac_parallelism
OC_PASSES = ceil(OC / MP)
pass_cycles = MP * K_TOTAL + 3

fill_rows = max(KH - 1 - PH, 0)        # = max(2 - PH, 0)
fill_cols = max(KW - PW,   1)          # = max(3 - PW, 1)
latency   = fill_rows * (IW + PW) + fill_cols + OC_PASSES * pass_cycles
```

Critical: the fill-row step is `(IW + PW)`, not `IW`. `ST_STREAM` wraps
`in_col` at `IW - 1 + PW` so the right-edge padding outputs are produced
inline during the stream, not in a separate drain state. If your formula
uses `IW` you will undershoot latency and the testbench will report a
timing mismatch.

## Required FSM states

- `ST_STREAM` — accept input; increment `in_row, in_col` with the
  `IW-1+PW` wrap; fire `output_fires` whenever the receptive-field window
  is valid.
- `ST_RUNNING`, `ST_BIAS`, `ST_SCALE`, `ST_OUTPUT` — same as pointwise.

**No `ST_DRAIN` state.** The historically buggy "drain-row" design checked
`in_row > IH-1+PH` as an exit condition and broke on every iteration. The
correct design terminates on `outputs_emitted == OH*OW`, as enforced by the
output-counter preflight and the handwritten `coord_scheduler` component
(see below).

## Required registers

- All registers from `02_conv1x1.md` (acc/biased/scaled/k_counter/lane_counter/
  oc_group/state).
- `reg signed [7:0] line_buf [0:LB_ROWS-1][0:IW-1][0:IC-1];` — 2D line buffer
  holding the last `LB_ROWS = KH` input rows. The structural preflight
  rule `line_buffer_missing` fires if this declaration is absent.
- `reg signed [7:0] window [0:KH-1][0:KW-1][0:IC-1];` — the receptive-field
  window. Must be declared as `reg` and updated via `<=` inside an
  `always @(posedge clk)` block. The structural preflight rule
  `window_not_registered` fires otherwise.
- `reg cur_row [$clog2(LB_ROWS+1)-1:0];` — which line_buf row is the
  newest.
- `reg [$clog2(OH*OW+1)-1:0] outputs_emitted;` — bounded output counter.

## Window update rule

On every input cycle, shift the window and load the new column from the
line buffer:

```verilog
always @(posedge clk) begin
  // shift window columns left
  for (i = 0; i < KH; i = i + 1)
    for (j = 0; j < KW-1; j = j + 1)
      window[i][j] <= window[i][j+1];
  // load rightmost column from line_buf
  for (i = 0; i < KH; i = i + 1)
    window[i][KW-1] <= line_buf[wrap(cur_row + 1 + i)][in_col];
end
```

Do not rebuild `window` combinationally from `line_buf` — that blows up
synth cones and loses the per-cycle invariant.

## Padding drain

Padded pixels are produced by the wrap-at-IW-1+PW behaviour in ST_STREAM:
when `in_col >= IW`, the "column" is the right padding region and the
window rightmost column is 0. No separate drain state.

## output_fires condition

```
row_num = in_row + PH - (KH - 1)
col_num = in_col + PW - (KW - 1)
fires   = (row_num >= 0) && (row_num < OH * SH) && (row_num % SH == 0)
       && (col_num >= 0) && (col_num < OW * SW) && (col_num % SW == 0)
       && outputs_emitted < OH * OW
```

This coordinate logic is the most bug-prone part of the design, which is
why `rtl_library/coord_scheduler.v` owns it. **Instantiate coord_scheduler;
do not roll your own row/col/wrap/stride/drain logic.** The full contract
(region handshake, `stall_in` derivation, instantiation form) lives in
`01_context.md § coord_scheduler contract`. Consume the scheduler's
`output_fires`, `outputs_emitted`, `in_row`, `in_col`, `ready_in`,
`needs_real_input`, and `out_frame_done` signals.

## Known failure modes

See `08_common_bugs.md`. Spatially-specific bugs:

- `output_counter_missing` — infinite output stream → Verilator timeout.
- `window_not_registered` — window rebuilt combinationally → synth cone.
- `loop_bounds_incorrect` — off-by-one in the `fill_rows / fill_cols`
  comparison; first_mismatch_index=0 or sample_count too small.
- `array_indexing_error` — window accessed as `window[ic][kh][kw]` instead
  of `window[kh][kw][ic]`.

## Reference skeleton

A canonical 3×3 spatial conv is ~350 lines. Adapt `02_conv1x1.md`'s skeleton
and add:

```verilog
    localparam KH = 3;
    localparam KW = 3;
    localparam PH = <padding[0]>;
    localparam PW = <padding[1]>;
    localparam LB_ROWS = KH;  // 3

    reg signed [7:0] line_buf [0:LB_ROWS-1][0:IW-1][0:IC-1];
    reg signed [7:0] window   [0:KH-1][0:KW-1][0:IC-1];
    reg [$clog2(LB_ROWS)-1:0] cur_row;
    reg [$clog2(IH+PH+1)-1:0] in_row;
    reg [$clog2(IW+PW+1)-1:0] in_col;
    reg [$clog2(OH*OW+1)-1:0] outputs_emitted;

    // ... ST_STREAM reads IC channels from data_in into
    //     line_buf[cur_row][in_col], then shifts the window, then advances
    //     in_col (wrapping at IW-1+PW) and in_row on col wrap.
    // ... output_fires gate triggers ST_RUNNING with the window frozen
    //     for all OC_PASSES (window-freeze rule).
```

Termination is bounded by `sched_outputs_emitted == OH*OW` (equivalently,
by pulsing on `sched_out_frame_done`). Never terminate on row-counter
comparisons like `in_row > IH-1+PH` — that's the drain-exit bug.

## Reference to adapt

`knowledge/references/conv3x3_passing_reference.v` — handwritten 3×3 s1 p1
reference. Written for `layer1_0_conv2` (IC=OC=64, IH=IW=112, MP=4). Adapt
the localparams (IC / OC / IH / IW / OH / OW / MP / SCALE_MULT / SCALE_SHIFT
/ `$readmemh` paths) to the current LayerIR; keep the FSM, the line-buffer
and window structure, the coord_scheduler instantiation, and the
serialized-MAC loop exactly as shown. It already satisfies:

- structural preflight (line_buf + registered window + coord_scheduler)
- the region handshake and combinational `stall_in` contract
- window-freeze across `OC_PASSES`
- round-to-nearest with `SCALE_ROUND_BIAS`
- multi-frame re-arm (scheduler re-starts on `sched_out_frame_done`)

Do not regenerate the FSM from scratch from this markdown if the reference
is available — that path historically produces multi-frame-reset bugs and
off-by-one window indexing. Copy the reference's structure, change the
parameters.
