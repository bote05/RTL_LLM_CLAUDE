# 02 — Pointwise (1×1) conv2d

Canonical legacy reference: `knowledge/references/conv1x1_passing_reference.v`
(Foundry first-shot, 0 Surgeon retries in the older flow). For new Vivado
work, preserve the FSM shape but use synchronous ROM reads and BRAM attributes.

## When to use

`op_type == "conv2d"` with `weight_shape[2] == 1 && weight_shape[3] == 1`.

## Latency contract

From `scripts/golden_impl.py::compute_conv2d_latency_cycles`:

```
IC = weight_shape[1]
MP = mac_parallelism
OC_PASSES = ceil(OC / MP)
pass_cycles = MP * K_TOTAL + 4        # +1 sync ROM read, then bias/scale/output
latency = 1 + OC_PASSES * pass_cycles
```

No line buffer. No window fill. The first input cycle produces the first
output immediately (after OC_PASSES).

## Required FSM states

- `ST_STREAM` — wait for `valid_in`, latch channels of the input pixel.
- `ST_RUNNING` — one MAC per cycle on the `lane_counter`-selected lane.
- `ST_BIAS` — add bias for the current OC group.
- `ST_SCALE` — apply `SCALE_MULT / SCALE_SHIFT` requantisation.
- `ST_OUTPUT` — emit one slice of `data_out`, advance `oc_group`, or return
  to `ST_STREAM` when all OC_PASSES are done.

Allowed transitions: STREAM → RUNNING → BIAS → SCALE → OUTPUT → (RUNNING | STREAM).

## Required registers

- `acc [0:MP-1]` — accumulator per lane, width `PROD_W + $clog2(K_TOTAL)`
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

ONE read from `weights[]` per cycle. The `lane_counter` rotates through
0..MP-1 before `k_counter` advances, so each ST_RUNNING cycle does exactly:

```
weights[global_oc * K_TOTAL + k_counter]   // single memory read
```

Never read MP weights per cycle from one flat async array — that produces MP
parallel mux trees that Vivado cannot map into legal BRAM ports.

## Known failure modes

See `08_common_bugs.md`. The historically most common pointwise bugs:

- `rounding_mode_wrong` — arithmetic right-shift without the round-half-up
  bias term.
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
            outputs_emitted <= 0;
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
                ST_SCALE:  /* scaled[i] <= (biased[i]*SCALE_MULT + SCALE_ROUND_BIAS) >>> SCALE_SHIFT */
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

The reference file in `references/conv1x1_passing_reference.v` is 245 lines
with all the boilerplate filled in. Adapt its parameter block (IC, OC, IH,
IW, MP, SCALE_MULT, SCALE_SHIFT, weights_path, bias_path) to the new layer's
LayerIR — do not regenerate the surrounding FSM from scratch.

## Reference to adapt

`knowledge/references/conv1x1_passing_reference.v` — proven-passing 1×1
RTL from this repo (historical first-shot pass for `layer1_0_conv1`). Adapt
its parameter block (IC / OC / IH / IW / MP / SCALE_MULT / SCALE_SHIFT /
`$readmemh` paths) to the current LayerIR; do not regenerate the FSM from
scratch.
