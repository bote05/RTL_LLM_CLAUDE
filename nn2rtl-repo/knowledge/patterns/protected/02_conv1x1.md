# 02 — Pointwise (1×1) conv2d

Canonical reference: `knowledge/references/protected/conv1x1_passing_reference.v`
(Foundry first-shot, 0 Surgeon retries). The FSM is monolithic — pointwise
convs do not use the split-architecture library because they have no line
buffer / window to share with spatial convs. The reference embeds the full
serialized MAC pipeline; Foundry only adapts the localparam block + `$readmemh`
paths from the LayerIR.

Weight ROM uses a clean unconditional address path
(`current_global_oc * K_TOTAL + k_counter`). Do NOT wrap it in
`(current_global_oc < OC) ? addr : 0` — that conditional mux blocks Vivado
BRAM inference and forces the weight memory into LUT logic. The accumulator
gate `mac_global_oc_q2 < OC` later in the pipeline already prevents
out-of-range reads from contaminating results.

## When to use

`op_type == "conv2d"` with `weight_shape[2] == 1 && weight_shape[3] == 1`.

## Latency contract

From `scripts/golden_impl.py::compute_conv2d_latency_cycles`:

```
IC = weight_shape[1]
MP = mac_parallelism
OC_PASSES = ceil(OC / MP)
pass_cycles = MP * K_TOTAL + 6        # 3-stage MAC pipeline (weight ROM, registered DSP mul, indexed acc)
                                      #   + ST_BIAS + ST_SCALE + ST_OUTPUT
latency = 1 + OC_PASSES * pass_cycles
```

No line buffer. No window fill. The latency is the input-accept cycle plus
the serialized OC-group passes.

Important naming note: in the current verified 1x1 architecture, `MP` is
the number of accumulator lanes in an OC group, not the number of
cycle-parallel BRAM reads. `lane_counter` serializes those lanes, so a pass
issues `MP * K_TOTAL` distinct weight reads: for each `k_counter`, it visits
lane 0, lane 1, ..., lane MP-1. This is not redundant work; those reads are
for different output channels in the current OC group. A future true
MP-read-per-cycle banked datapath would need a different latency contract
(`K_TOTAL + overhead` per pass) and updated LayerIR generation.

## Required FSM states

- `ST_STREAM` — wait for `valid_in`, latch channels of the input pixel.
- `ST_RUNNING` — one MAC per cycle on the `lane_counter`-selected lane.
- `ST_BIAS` — add bias for the current OC group.
- `ST_SCALE` — apply `SCALE_MULT / SCALE_SHIFT` requantisation.
- `ST_OUTPUT` — emit one slice of `data_out`, advance `oc_group`, or return
  to `ST_STREAM` when all OC_PASSES are done.

Allowed transitions: STREAM → RUNNING → BIAS → SCALE → OUTPUT → (RUNNING | STREAM).

## Required registers

- `acc [0:MP-1]` — accumulator per lane in the current OC group, width
  `PROD_W + $clog2(K_TOTAL)`
  where `PROD_W = 16`.
- `biased [0:MP-1]` — width = max(ACC_W, 32) + 1.
- `scaled [0:MP-1]` — width = BIASED_W + SCALE_CONST_W.
- `in_latch [0:IC-1]` — the captured pixel, reused across all OC_PASSES.
- `k_counter` — kernel-index counter, 0..K_TOTAL-1.
- `lane_counter` — rotating 0..MP-1; advances `k_counter` when it wraps.
- `oc_group` — 0..OC_PASSES-1.
- `state` — FSM state.
- **No `line_buf`. No `window`. No `cur_row`.** The structural preflight rule
  for line_buffer only fires on KH*KW > 1.

## Serialized weight reads

ONE read from `weights[]` per cycle in the current verified contract. The
`lane_counter` rotates through 0..MP-1 before `k_counter` advances, so each
ST_RUNNING cycle does exactly:

```
weights[global_oc * K_TOTAL + k_counter]   // single memory read
```

Never read MP weights per cycle from one flat async array — that produces MP
parallel mux trees that Vivado cannot map into legal BRAM ports.

If `LayerIR.weight_bank_paths` is present, the frontend has emitted one bank
file per lane. Bank `lane` contains, for each `oc_group`, the full
`K_TOTAL` vector for output channel `oc_group*MP + lane`, zero-padded for
the tail group. That layout is ready for a future true lane-parallel BRAM
datapath, but the current reference below still uses the flat `weights_path`
and serialized one-read-per-cycle latency. Do not switch to parallel bank
reads unless the LayerIR latency contract has also been generated for that
banked datapath.

## Known failure modes

See `08_common_bugs.md`. The historically most common pointwise bugs:

- `rounding_mode_wrong` — arithmetic right-shift without the canonical
  sign-aware fixed-point bias term. A bare `>>>` floors toward `-inf`;
  an unconditional `+0.5 LSB` bias pushes negatives toward `+inf`.
  Subtracting HALF for negatives ALSO over-rounds (since `>>>` already
  floors). Use:
  `(scaled + (scaled[MSB] ? (HALF-1) : HALF)) >>> SHIFT`
  with `HALF = 1 << (SHIFT-1)`. Error is symmetric around zero and
  matches `torch.round` for non-tie values. See `01_context.md`
  "Scale-shift rounding — MANDATORY".
- `sign_extension_error` — the bias adder inferred unsigned context.
- `weights_packed_forbidden` — Surgeon attempted to pack weights after synth
  timed out; the right fix is serialized reads, not packing.

## Reference skeleton (~50 lines — see full file in references/)

```verilog
module <module_id> (
    input  wire                               clk,
    input  wire                               rst_n,
    input  wire                               valid_in,
    output reg                                ready_in,
    input  wire [IC*8-1:0]                    data_in,
    output reg                                valid_out,
    output reg  [OC*8-1:0]                    data_out
);
    localparam IC      = <from weight_shape[1]>;
    localparam OC      = <from weight_shape[0]>;
    localparam K_TOTAL = IC;            // 1*1 kernel
    localparam MP      = <mac_parallelism>;
    localparam OC_PASSES = (OC + MP - 1) / MP;

    localparam SCALE_MULT  = <from scale_factor>;
    localparam SCALE_SHIFT = <from scale_factor>;

    (* rom_style = "block", ram_style = "block" *)
    reg signed [7:0]  weights [0:OC*K_TOTAL-1];
    (* rom_style = "block", ram_style = "block" *)
    reg signed [31:0] biases  [0:OC-1];
    initial begin
        $readmemh("<weights_path>", weights);
        $readmemh("<bias_path>",   biases);
    end

    reg signed [7:0] in_latch [0:IC-1];
    reg signed [ACC_W-1:0] acc [0:MP-1];
    // ... biased / scaled / v_tmp declarations ...

    reg [$clog2(K_TOTAL+1)-1:0]   k_counter;
    reg [$clog2(MP+1)-1:0]        lane_counter;
    reg [$clog2(OC_PASSES+1)-1:0] oc_group;
    reg [2:0] state;

    // NOTE: do NOT declare `outputs_emitted` / `out_row` / `out_col` here.
    // Pointwise 1x1 is 1:1 pixel-in-to-pixel-out; the FSM terminates
    // naturally when `oc_group == OC_PASSES-1` completes for the current
    // pixel, then the module accepts the next input on the same handshake.
    // A frame-level counter would latch the FSM into a terminal state
    // after the first OH*OW outputs and break back-to-back frames.
    // The `output_counter_missing` structural preflight rule does NOT
    // fire on pointwise conv2d — it is scoped to spatial conv and maxpool.

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_STREAM;
            ready_in <= 1'b1;  // [INVARIANT:READY_IN_GATING]
            valid_out <= 1'b0;
            // ... reset lane/k/oc counters ...
        end else begin
            case (state)
                ST_STREAM: /* latch channels on valid_in */
                ST_RUNNING: begin
                    // ONE weight read, ONE MAC, ONE lane accumulated per cycle
                    acc[lane_counter] <= acc[lane_counter]
                        + $signed(weights[global_oc*K_TOTAL + k_counter])
                        * $signed(in_latch[k_counter]);
                    if (lane_counter == MP-1) begin
                        lane_counter <= 0;
                        k_counter <= k_counter + 1;
                    end else lane_counter <= lane_counter + 1;
                    if (k_counter == K_TOTAL-1 && lane_counter == MP-1)
                        state <= ST_BIAS;
                end
                ST_BIAS:   /* biased[i] <= acc[i] + biases[global_oc + i] */
                ST_SCALE:  /* scaled[i] <= biased[i] * SCALE_MULT_CONST */
                ST_OUTPUT: begin
                    // [INVARIANT:VALID_OUT_LATENCY]
                    for (i = 0; i < MP; i = i + 1)
                        data_out[(global_oc+i)*8 +: 8] <= saturate_int8(scaled[i]);
                    if (oc_group == OC_PASSES-1) begin
                        valid_out <= 1'b1;
                        state <= ST_STREAM;
                        ready_in <= 1'b1;
                    end else begin
                        oc_group <= oc_group + 1;
                        state <= ST_RUNNING;
                    end
                end
            endcase
        end
    end
endmodule
```

The reference file in `references/protected/conv1x1_passing_reference.v` is 245 lines
with all the boilerplate filled in. Adapt its parameter block (IC, OC, IH,
IW, MP, SCALE_MULT, SCALE_SHIFT, weights_path, bias_path) to the new layer's
LayerIR — do not regenerate the surrounding FSM from scratch.

## Reference to adapt

`knowledge/references/protected/conv1x1_passing_reference.v` — proven-passing 1×1
RTL from this repo (historical first-shot pass for `layer1_0_conv1`). Adapt
its parameter block (IC / OC / IH / IW / MP / SCALE_MULT / SCALE_SHIFT /
`$readmemh` paths) to the current LayerIR; do not regenerate the FSM from
scratch.
