# 01 — Shared interface contract

Every module generated for nn2rtl MUST satisfy the contract below. This file
is prepended to every `get_rtl_patterns` response regardless of op_type, so the
op-specific files below can assume it already applies.

## Canonical top-level ports

Exactly these seven signals. No more, no fewer. Names and directions are fixed.

```verilog
module <module_id> (
    input  wire                               clk,
    input  wire                               rst_n,       // active-low reset
    input  wire                               valid_in,
    output reg                                ready_in,    // OUTPUT — backpressure
    input  wire [input_width_bits-1:0]        data_in,
    output reg                                valid_out,
    output reg  [output_width_bits-1:0]       data_out
);
```

`input_width_bits` and `output_width_bits` come from the `LayerIR`. Do not
invent other width sources.

## Packed-channel bus convention

- **conv2d / relu / maxpool**: `data_in` is `IC * 8` bits. `data_in[i*8 +: 8]`
  is channel `i` as a signed INT8. `data_out` is `OC * 8` bits, same layout.
- **add**: `data_in` is `2 * W` bits where `W = output_width_bits`. The low
  half `data_in[W-1:0]` is the packed lhs channels; the high half
  `data_in[2W-1:W]` is the packed rhs channels. `data_out` is `OC * 8` bits.

## INT8 quantization

Per-tensor symmetric with `scale_factor` provided in the LayerIR. The
canonical requantisation is:

```
scaled = (biased * SCALE_MULT + SCALE_ROUND_BIAS) >>> SCALE_SHIFT
out    = saturate_int8(scaled)
```

where `SCALE_MULT` / `SCALE_SHIFT` are derived so
`SCALE_MULT / 2^SCALE_SHIFT ≈ scale_factor` with minimal relative error.
Mark this line `// [INVARIANT:ROUNDING]`.

For `add`, use `lhs_scale_factor`, `rhs_scale_factor`, and `scale_factor`
together — the output is `saturate(((lhs * lhs_scale + rhs * rhs_scale) * scale)
>>> shift)`.

## Weight and bias loading

- Declare `reg signed [7:0]  weights [0:OC*K_TOTAL-1];` — flat INT8 array.
- Declare `reg signed [31:0] biases  [0:OC-1];` — flat INT32 array.
- Inside an `initial` block, load via `$readmemh("<weights_path>", weights);`
  and `$readmemh("<bias_path>", biases);` using the LayerIR-provided paths.
- Index as `weights[oc * K_TOTAL + k]` where `k` is a flat kernel index.

Mark each declaration and each $readmemh line `// [INVARIANT:WEIGHT_ARRAY]`.
NEVER introduce `weights_packed`, `initial weights[...] = expr`, or
`assign weights[...] = ...` — yosys `OPT_MEM` rejects non-constant memory
initializers and the pipeline's structural preflight will reject these
constructs before simulation.

## Valid / ready handshake

- `valid_in` asserts when `data_in` carries a sample.
- `ready_in` is the module's own output, raised when it can accept input.
  Deassert while computing an output; re-assert when computation completes
  for the current output pixel. Mark this `// [INVARIANT:READY_IN_GATING]`.
- `valid_out` asserts when `data_out` carries a valid sample. The first
  `valid_out` fires exactly `pipeline_latency_cycles` after the first
  `valid_in` for the current vector. Mark that assertion
  `// [INVARIANT:VALID_OUT_LATENCY]`.
- `pipeline_latency_cycles` is authoritative — do not re-derive it. A module
  whose measured latency disagrees with this value fails verification.

## Persistence

The Foundry/Surgeon path persists RTL via the `write_verilog` MCP tool into
`output/rtl/<module_id>.v` plus a sibling `.meta.json`. Do not write the
file by hand; the tool handles atomic persistence and the .meta sidecar.

## Reset behaviour

- All top-level output registers must have a non-X reset value — either
  `<= 0` or `<= 1'b1` as appropriate. A `ready_in` that comes up X at reset
  holds the testbench forever.
- Reset is synchronous-deassert, asynchronous-assert: `always @(posedge clk
  or negedge rst_n)`.

---

## Scale factor derivation

Convert the LayerIR's `scale_factor` to `SCALE_MULT` / `SCALE_SHIFT` via:

```
For SHIFT in 8..23:
    MULT = round(scale_factor * 2^SHIFT)
    if 1 <= MULT <= 32767:
        err = |MULT / 2^SHIFT - scale_factor| / scale_factor
        keep (MULT, SHIFT) with smallest err
```

Emit both as `localparam`. For `op_type == "add"`, apply the same algorithm
independently to `lhs_scale_factor`, `rhs_scale_factor`, and `scale_factor`.

### Scale-shift rounding — MANDATORY

The golden model uses round-to-nearest. A bare `>>> SCALE_SHIFT` in Verilog
is arithmetic right-shift (floor), which biases every output toward `-inf`
by up to one LSB per sample. Add a half-LSB bias before the shift:

```verilog
// WRONG — floor; every output biased by up to -1.
v_tmp = scaled[oc] >>> SCALE_SHIFT;

// CORRECT — round-half-up via +0.5 LSB bias, then arithmetic shift.
localparam signed [SCALED_W-1:0] SCALE_ROUND_BIAS =
    {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
v_tmp = (scaled[oc] + SCALE_ROUND_BIAS) >>> SCALE_SHIFT;
```

Applies to every op that requantises (conv2d, add, any future scaled op).
Mark the rounding expression `// [INVARIANT:ROUNDING]`.

---

## Internal width derivation

Do NOT use fixed `32` / `48`-bit internal registers. Derive widths from the
layer bounds — this is mandatory, not cosmetic:

```verilog
localparam integer PROD_W        = 16;                            // 8×8 signed
localparam integer ACC_W         = PROD_W + $clog2(K_TOTAL);
localparam integer BIAS_W        = 32;                            // INT32 bias file
localparam integer BIASED_W      = ((ACC_W > BIAS_W) ? ACC_W : BIAS_W) + 1;
localparam integer SCALE_MAG_W   = $clog2(SCALE_MULT + 1);
localparam integer SCALE_CONST_W = SCALE_MAG_W + 1;               // signed positive
localparam integer SCALED_W      = BIASED_W + SCALE_CONST_W;
localparam signed [SCALE_CONST_W-1:0] SCALE_MULT_CONST = SCALE_MULT;
```

Canonical register declarations:

```verilog
reg signed [ACC_W-1:0]    acc    [0:MP-1];
reg signed [BIASED_W-1:0] biased [0:MP-1];
reg signed [SCALED_W-1:0] scaled [0:MP-1];
reg signed [SCALED_W-1:0] v_tmp;
```

Use `SCALE_MULT_CONST` (not the raw literal) in the SCALE stage so the
multiply width tracks the layer: `scaled[lane] <= $signed(biased[lane]) *
$signed(SCALE_MULT_CONST);`.

---

## Memory inference

Weight and bias arrays carry a `(* ram_style = "block" *)` hint:

```verilog
(* ram_style = "block" *) reg signed [7:0]  weights [0:OC*K_TOTAL-1];
(* ram_style = "block" *) reg signed [31:0] biases  [0:OC-1];
```

Sky130 has no dedicated BRAM macros, so this is a portability hint that
maps to flip-flops on this PDK. Keep it regardless.

---

## Invariant markers

`// [INVARIANT:TAG] ...` comments tell Surgeon that a line embodies a
mathematically fixed formula, correct by construction. Only mark a line
invariant when its correctness is derivable from the spec alone,
independent of simulation results.

### Tags you MAY mark

- `ROUNDING` — the requantisation line `(scaled + SCALE_ROUND_BIAS) >>>
  SCALE_SHIFT`.
- `READY_IN_GATING` — the exact assertion/deassertion points for `ready_in`.
- `VALID_OUT_LATENCY` — the line that drives `valid_out` high for the
  current pixel.
- `WEIGHT_ARRAY` — the `weights` / `biases` declarations and their
  `$readmemh` loader lines. Surgeon MUST NOT pack, reshape, or merge these.

### Tags you MUST NOT mark

Speculative per-module control logic is not invariant. Never mark:

- State-transition conditions (any next-state assignment)
- Counter comparisons (`k_counter == K_TOTAL - 1`, row/col bound checks)
- Loop termination conditions (e.g. `in_row > IH - 1 + PH`)

The retired tags `DRAIN_EXIT` and `INTER_VECTOR_RESET` must not be used.

---

## Variable-declaration scoping — Verilog-2001

All `reg` and `wire` signals must be declared at **module scope**, before
any `always` block. Verilog-2001 rejects procedural declarations, and
Yosys will error out.

```verilog
// WRONG
always @(posedge clk) begin
    for (i = 0; i < N; i = i + 1) begin
        reg signed [63:0] tmp;   // ILLEGAL in Verilog-2001
        tmp = foo[i];
    end
end

// CORRECT
reg signed [63:0] tmp;
always @(posedge clk) begin
    for (i = 0; i < N; i = i + 1)
        tmp = foo[i];
end
```

Also avoid SystemVerilog-only casts like `7'(K_TOTAL - 1)`; use plain
expressions or sized literals (`7'd63`).

---

## Concatenation-based sign extension — FORBIDDEN

Verilog concatenation `{...}` is always unsigned. Sign-extending via
`{{N{sign_bit}}, value}` and then adding a signed value coerces the add
to unsigned context, turning negative accumulators into large positives
and saturating outputs to `+127`. Classic silent sign bug.

```verilog
// WRONG — unsigned coercion, negative values blow up.
biased[lane] <= {{1{acc[lane][ACC_W-1]}}, acc[lane]} + $signed(biases[oc]);

// CORRECT — both operands are `reg signed`; the wider destination
// sign-extends automatically.
biased[lane] <= acc[lane] + biases[oc];

// Also CORRECT — be explicit:
biased[lane] <= $signed(acc[lane]) + $signed(biases[oc]);
```

---

## coord_scheduler contract — spatial conv and maxpool

For any `op_type == "conv2d"` with `KH*KW > 1`, and for any
`op_type == "maxpool"`, the generated module MUST instantiate
`coord_scheduler` from `rtl_library/coord_scheduler.v`. Do not roll your
own row/col counters, stride-divisibility gate, `IW-1+PW` wrap, or
drain-row exit — the scheduler owns all of them. The library file is
bundled into every iverilog / Verilator / Yosys invocation, so the module
is always in scope.

### Region handshake

The scheduler classifies each coordinate as one of two regions and imposes
an upstream handshake only in the REAL region:

- **REAL region** (`in_row < IH && in_col < IW`): scheduler asserts
  `needs_real_input = 1` and raises `ready_in` combinationally when it can
  accept a pixel. It advances ONLY on a `valid_in && ready_in` cycle. Your
  top-level must forward the external `valid_in` in, route the scheduler's
  `ready_in` back out as the module's `ready_in`, and write `data_in` into
  `line_buf` on the same handshake cycle.
- **PADDED region** (`in_row >= IH || in_col >= IW`): no upstream
  handshake; the scheduler free-runs under `!stall_in && running`.

### `stall_in` contract — combinational

`stall_in` MUST be driven combinationally (not registered) from the union
of `output_fires` and any MAC-busy state, so the scheduler freezes on the
firing coord for the entire MAC pipeline and advances past it on the last
ST_OUTPUT cycle:

```verilog
wire stall_in = sched_output_fires
             || (state == ST_RUNNING)
             || (state == ST_BIAS)
             || (state == ST_SCALE)
             || (state == ST_OUTPUT);
```

### Instantiation form

```verilog
wire        sched_needs_real_input;
wire        sched_ready_in;
wire        sched_output_fires;
wire        sched_out_frame_done;
wire [$clog2(IH+PH+1)-1:0]     sched_in_row;
wire [$clog2(IW+PW+1)-1:0]     sched_in_col;
wire [$clog2(OH*OW+1)-1:0]     sched_outputs_emitted;

coord_scheduler #(
    .IH(<IH>), .IW(<IW>), .OH(<OH>), .OW(<OW>),
    .KH(<KH>), .KW(<KW>), .SH(<stride[0]>), .SW(<stride[1]>),
    .PH(<padding[0]>), .PW(<padding[1]>)
) scheduler (
    .clk(clk), .rst_n(rst_n),
    .start(<one-cycle pulse when the FSM accepts the first frame>),
    .stall_in(stall_in),
    .valid_in(valid_in),
    .ready_in(sched_ready_in),
    .needs_real_input(sched_needs_real_input),
    .in_row(sched_in_row), .in_col(sched_in_col),
    .output_fires(sched_output_fires),
    .in_frame_done(),
    .out_frame_done(sched_out_frame_done),
    .outputs_emitted(sched_outputs_emitted)
);

// Top-level ready_in is simply the scheduler's ready_in.
assign ready_in = sched_ready_in;
```

The FSM terminates on `sched_out_frame_done` (equivalently
`sched_outputs_emitted == OH*OW`), never on `in_row > IH-1+PH`. Writing
`line_buf[cur_row][sched_in_col]` on every `valid_in && sched_ready_in`
handshake keeps the scheduler's coord stream and the line-buffer fill in
lockstep.

---

## Output-stage packing rule

Clamp and pack directly into `data_out` in a single registered stage. No
intermediate `out_byte[]` registers. Every bit of `data_out` set in the
same always-block cycle that asserts `valid_out`:

```verilog
reg signed [SCALED_W-1:0] v_tmp;  // module scope

// ST_OUTPUT body — one pass of the current oc_group's MP lanes:
for (lane = 0; lane < MP; lane = lane + 1) begin
    integer global_oc;
    global_oc = oc_group * MP + lane;
    v_tmp = (scaled[lane] + SCALE_ROUND_BIAS) >>> SCALE_SHIFT;
    data_out[global_oc*8 +: 8] <= (v_tmp > 127)  ?  8'sd127 :
                                  (v_tmp < -128) ? -8'sd128 : v_tmp[7:0];
end
// On the last oc_group:
// valid_out <= 1'b1;
// ready_in  <= 1'b1;
// state     <= ST_STREAM;
```

Never combine BIAS and SCALE in the same registered stage — that collapses
a wide integer add and a wide multiply into one combinational cone and
hurts Fmax.

