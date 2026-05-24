# 03 — Spatial 3×3 conv, padding=1 (instantiation-only pattern)

> **Tile-ABI addendum (canonical for `io_mode == "channel_tiled"`)**: under the
> `tiled-streaming` contract, `input_width_bits == output_width_bits ==
> channel_tile*8` (default 256 for `channel_tile=32`). The 3×3 kernel walks
> `kh ∈ [0,3) × kw ∈ [0,3)`; for each (oh, ow) output pixel and (kh, kw)
> kernel position, the module reads `ceil(IC / channel_tile)` input tiles
> from the line-buffer window, MAC-accumulates them into the per-output-
> channel accumulators, and only emits the output tile beats after the
> final (kh, kw, ic_tile) iteration. The line-buffer + window logic
> (`coord_scheduler`, `line_buf_window`, `conv_datapath`) is shared across
> all 3×3 convs; the per-layer wrapper only sets `IC`, `OC`, `channel_tile`,
> and the bias/scale localparams. Receptive-field bounds with padding=1
> still substitute zero for out-of-bounds input bytes (no change). See
> `knowledge/patterns/protected/01_context.md` §"Bus convention —
> CANONICAL tiled-streaming ABI" for full ABI rules.

## When to use

`op_type == "conv2d"` with `weight_shape[2] == 3 && weight_shape[3] == 3`.

## Architecture

Spatial convs do NOT have a monolithic Foundry-generated FSM. They use the
split-module architecture in `rtl_library/SPLIT_ARCHITECTURE.md`:

```
                   ┌──────────────────┐
  valid_in ───────▶│                  │
  data_in  ───────▶│ line_buf_window  │────▶ window_flat ┐
                   │                  │                    │
                   └─▲────────────────┘                    ▼
                     │                          ┌──────────────────┐
                     │ in_row/in_col/advance    │                  │
                     │ needs_real/output_fires  │  conv_datapath   │──▶ data_out
                     │                          │                  │──▶ valid_out
                   ┌─┴────────────────┐         └──────▲───────────┘
                   │                  │ start_mac      │
                   │ coord_scheduler  │──────────────── (= output_fires)
                   │                  │◀── stall_in (= mac_busy)
                   │                  │
                   └──────────────────┘
                      ▲
                      │
            start ────┘
```

All three library modules are bundled into every iverilog / Verilator /
Vivado invocation (see `RTL_LIBRARY_SOURCES` in `mcp/tools.ts`), so they
are always in scope — just instantiate them.

Foundry's job for a 3×3 layer is **~60 lines of structural wiring**: set
the localparams from LayerIR, instantiate the three modules, connect the
canonical 7-signal top-level interface, generate a one-cycle `start_pulse`
on first `valid_in`. No FSM. No window management. No MAC pipeline.

## Scheduler contract (pixel-delivery-safe)

`coord_scheduler` emits `output_fires` as a REGISTERED one-cycle pulse
the cycle AFTER it advances past a firing coord. In that same advance
cycle, the pixel at the firing coord is handshaked into `line_buf_window`
via `valid_in && ready_in`, so the window has the correct rightmost
column *before* the MAC sees `output_fires`. `stall_in` is simply
`mac_busy` — no `output_fires` or `mac_done` plumbing.

`coord_scheduler` exposes its internal `advance` wire as an output
signal; `line_buf_window` consumes it directly to know when to shift
the window and write `line_buf`. No manual `sched_advance` replication
is needed.

## Top-level wiring template

```verilog
module <module_id> (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       valid_in,
    output wire                       ready_in,
    input  wire [<IC*8 - 1>:0]        data_in,
    output wire                       valid_out,
    output wire [<OC*8 - 1>:0]        data_out
);
    // --- Parameters from LayerIR ---
    localparam integer IC        = <input_shape[1]>;
    localparam integer OC        = <output_shape[1]>;
    localparam integer IH        = <input_shape[2]>;
    localparam integer IW        = <input_shape[3]>;
    localparam integer OH        = <output_shape[2]>;
    localparam integer OW        = <output_shape[3]>;
    localparam integer KH        = 3;
    localparam integer KW        = 3;
    localparam integer SH        = <stride[0]>;
    localparam integer SW        = <stride[1]>;
    localparam integer PH        = <padding[0]>;
    localparam integer PW        = <padding[1]>;
    localparam integer K_TOTAL   = IC * KH * KW;
    localparam integer MP        = <mac_parallelism>;
    localparam integer SCALE_MULT  = <derived from scale_factor>;
    localparam integer SCALE_SHIFT = <derived from scale_factor>;

    // --- One-cycle start pulse on reset deassertion; re-arms ONLY
    //     after both sched_out_frame_done has fired AND the last
    //     pixel's MAC pipeline has drained (mac_busy back to 0).
    //     Without the mac_busy gate the re-arm clears line_buf_window
    //     mid-MAC of the last pixel and corrupts that output. Critically,
    //     start does NOT wait on valid_in: the static TB waits for
    //     ready_in before asserting valid_in, and ready_in stays low
    //     until the scheduler is running, which requires start.
    //     Pulsing start on !started breaks that circular wait.
    reg started, start_pulse, pending_rearm;
    wire sched_out_frame_done;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            started       <= 1'b0;
            start_pulse   <= 1'b0;
            pending_rearm <= 1'b0;
        end else begin
            start_pulse <= 1'b0;
            if (sched_out_frame_done) pending_rearm <= 1'b1;
            if (!started) begin
                started       <= 1'b1;
                start_pulse   <= 1'b1;
            end else if (pending_rearm && !mac_busy) begin
                started       <= 1'b0;
                pending_rearm <= 1'b0;
            end
        end
    end

    // --- Scheduler ↔ datapath wires ---
    wire                              sched_needs_real_input;
    wire                              sched_ready_in;
    wire                              sched_output_fires;
    wire                              sched_advance;
    wire [$clog2(IH + PH + 1)-1:0]    sched_in_row;
    wire [$clog2(IW + PW + 1)-1:0]    sched_in_col;
    wire [$clog2(OH * OW + 1)-1:0]    sched_outputs_emitted;

    wire [KH*KW*IC*8-1:0]             window_flat;
    wire                              mac_busy;

    // stall_in is just mac_busy. No output_fires or mac_done needed —
    // scheduler's registered output_fires pulse + internal eff_stall
    // handle the firing-coord freeze on its own.
    wire stall_in = mac_busy;

    // --- Coord scheduler ---
    coord_scheduler #(
        .IH(IH), .IW(IW), .OH(OH), .OW(OW),
        .KH(KH), .KW(KW), .SH(SH), .SW(SW),
        .PH(PH), .PW(PW)
    ) scheduler (
        .clk(clk), .rst_n(rst_n),
        .start(start_pulse),
        .stall_in(stall_in),
        .valid_in(valid_in),
        .ready_in(sched_ready_in),
        .needs_real_input(sched_needs_real_input),
        .in_row(sched_in_row),
        .in_col(sched_in_col),
        .output_fires(sched_output_fires),
        .advance(sched_advance),
        .in_frame_done(),
        .out_frame_done(sched_out_frame_done),
        .outputs_emitted(sched_outputs_emitted)
    );

    // --- Line buffer + shift-register window ---
    line_buf_window #(
        .IC(IC), .IW(IW), .IH(IH),
        .KH(KH), .KW(KW), .PW(PW), .PH(PH)
    ) lbw (
        .clk(clk), .rst_n(rst_n),
        .frame_start(start_pulse),              // clears line_buf + window
                                                 // between input frames
        .sched_in_row(sched_in_row),
        .sched_in_col(sched_in_col),
        .sched_needs_real_input(sched_needs_real_input),
        .sched_advance(sched_advance),
        .sched_output_fires(sched_output_fires),
        .valid_in(valid_in),
        .data_in(data_in),
        .window_flat(window_flat)
    );

    // --- Datapath: MAC / bias / scale / output packing ---
    conv_datapath #(
        .IC(IC), .OC(OC), .KH(KH), .KW(KW),
        .K_TOTAL(K_TOTAL), .MP(MP),
        .SCALE_MULT(SCALE_MULT), .SCALE_SHIFT(SCALE_SHIFT),
        .WEIGHTS_PATH("<absolute path from LayerIR.weights_path>"),
        .BIAS_PATH("<absolute path from LayerIR.bias_path>")
    ) dp (
        .clk(clk), .rst_n(rst_n),
        .window_flat(window_flat),
        .start_mac(sched_output_fires),
        .valid_out(valid_out),
        .data_out(data_out),
        .mac_busy(mac_busy)
    );

    // --- Top-level ready_in passes through the scheduler's handshake. ---
    assign ready_in = sched_ready_in;

endmodule
```

## What Foundry must get right

- Every `<placeholder>` replaced with the exact value from the LayerIR or
  computed deterministically (`K_TOTAL = IC * KH * KW`, etc.).
- `SCALE_MULT`/`SCALE_SHIFT` derived via the algorithm in
  `01_context.md § Scale factor derivation`.
- `$readmemh` paths from LayerIR's `weights_path` / `bias_path` passed as
  `WEIGHTS_PATH`/`BIAS_PATH` parameters to `conv_datapath`. Absolute paths
  only.
- The 7 canonical top-level ports exactly. No extra signals, no renames.
- Bus widths: `data_in` is `IC*8` bits, `data_out` is `OC*8` bits.

## What Foundry must NOT do

- Do **not** write a line buffer, window, or MAC FSM. Those live in
  `rtl_library/`.
- Do **not** add `always @(posedge clk)` blocks except the single one for
  `start_pulse` shown above.
- Do **not** regenerate the coord_scheduler / line_buf_window /
  conv_datapath source files — they are fixed library infrastructure.
- Do **not** add a "preflight shim" with dummy `line_buf` / `window` /
  `weights` / `biases` regs to satisfy the structural preflight. The
  preflight recognizes the split architecture: when the top-level
  instantiates `line_buf_window` and `conv_datapath`, the rules for
  `line_buf` / `window` / `$readmemh` declarations are skipped.

## Latency

`pipeline_latency_cycles` from the LayerIR is authoritative. The exact
formula is in `scripts/golden_impl.py::compute_conv2d_latency_cycles` and
is derived from the scheduler's fill-row / fill-col startup plus
`OC_PASSES * (MP * K_TOTAL + 6)` cycles per firing coord — the 6 covers
the 3-stage MAC pipeline (weight ROM, registered DSP multiply, indexed
accumulate) and the post-MAC ST_BIAS / ST_SCALE / ST_OUTPUT stages. Here
`MP` is the number of accumulator lanes in an OC group; the current
contract still issues one weight read / one MAC per cycle. Do not
re-derive; trust LayerIR.

## Known failure modes

See `08_common_bugs.md` for the general catalog. With the split
architecture, the previously most-bug-prone classes (drain-exit, window
rotation, multi-frame reset, pixel-loss at firing coord) are owned by
the library modules and are not accessible to the generated top-level.

## BRAM-backed line buffer

`line_buf_window.v` uses KH per-slot BRAMs with a rotating-pointer
schedule (see `rtl_library/SPLIT_ARCHITECTURE.md` for the full
architecture). The implementation detail is hidden from Foundry —
the canonical wiring template above does not change — but it has two
observable consequences:

1. **Synthesis area is comparable to hls4ml.** A 3×3 conv on Artix-7
   100T uses tens of BRAMs for line storage instead of ~150K
   flip-flops, so PPA numbers compare fairly against hls4ml's
   BRAM-backed reference designs.
2. **Multi-frame correctness depends on `frame_start`.** The rotating
   pointer + `row_valid` mask are reset on `frame_start`; without
   that pulse, frame 2 will read stale frame-1 data from the BRAM
   cells. The pulse is generated by the canonical `start_pulse`
   block at the top level, so as long as that block is wired to
   `lbw.frame_start` and `scheduler.start`, multi-frame works.
