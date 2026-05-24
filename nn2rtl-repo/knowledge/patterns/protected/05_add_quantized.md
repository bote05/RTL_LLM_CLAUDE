# 05 — INT8 quantized residual add

> **Tile-ABI addendum (canonical for `io_mode == "channel_tiled"`)**: under the
> `tiled-streaming` contract, `input_width_bits == 2*channel_tile*8 == 512`
> (default for `channel_tile=32`) and `output_width_bits == channel_tile*8
> == 256`. Each `data_in` beat packs two `channel_tile`-wide INT8 tile
> halves: bits `[255:0]` are the lhs tile (channels from `mainSource`),
> bits `[511:256]` are the rhs tile (channels from `skipSource`). The two
> halves MUST present the same `(pixel, tile_idx)` in lock-step; `valid_in`
> fires only when both halves are present. The output emits one tile beat
> per input tile beat (1:1 cadence). Total beats per pixel = `ceil(C /
> channel_tile)`. See `knowledge/patterns/protected/01_context.md`
> §"Bus convention — CANONICAL tiled-streaming ABI" for full rules.
>
> ⛔ **DO NOT** override `channel_tile` from the LayerIR. Use **exactly**
> `channel_tile == LayerIR.channel_tile` (always 32 for ResNet-50 under
> the canonical ABI). `data_out` MUST be `channel_tile*8` bits wide
> (e.g. `[255:0]` for tile=32). DO NOT pick a smaller tile (e.g.
> `tile=16` with `data_out[127:0]`) even if the channel count is large
> — that breaks the chain because downstream relu/conv consumers are
> built at tile=32 and expect 256-bit beats. The `auto_add_tiled-streaming_node_add_7_..._i256_o128_..._tile16.v`
> probationary reference is an ANTI-PATTERN; do not crib its bus
> geometry. The correct shape: `data_in[511:0]` (512-bit, packed
> lhs|rhs of 256 each), `data_out[255:0]` (single 256-bit tile beat).

## ⛔ Anti-pattern: do NOT use channel_tile=16 (128-bit data_out)

The probationary `auto_add_*_tile16` reference picks `channel_tile=16`
so that `BEATS_PER_PIXEL = OC/16` is a power of two convenient for the
gather FSM. **This breaks the chain.** Downstream relu/conv modules are
built at `tile=32` (256-bit beats); a 128-bit add output cannot drive
a 256-bit relu input without a serializer-shim in the wrapper.

Use `CHANNEL_TILE = 32`, `BEATS_PER_PIXEL = OC/32`. For OC=1024, that's
32 beats per pixel — not 64 — and `data_out [255:0]` carries 32 channels
per beat. The internal accumulator `out_beats[]` array shrinks by half
correspondingly.

## When to use

`op_type == "add"`. This is ResNet's residual add: two tensors of the same
shape and channel count combined, each with its own scale factor.

## Input layout

`data_in` is packed as `[rhs | lhs]`:

```
data_in[W-1:0]     = lhs  (OC channels * 8 bits)
data_in[2W-1:W]    = rhs  (OC channels * 8 bits)
```

where `W = output_width_bits`. `input_width_bits == 2 * output_width_bits`.

The orchestrator validates this packing on LayerIR load; if the LayerIR
violates it, the pipeline fails fast before Foundry dispatch.

## Quantization formula

The golden (`scripts/golden_impl.py:Int8Add.forward`) computes:

```
summed_f = lhs_q * lhs_in_scale + rhs_q * rhs_in_scale     # float
out_q    = clamp(round_half_up_toward_pos_inf(summed_f / out_scale), -128, 127)
```

`round_half_up_toward_pos_inf` means: ties round toward +∞ ALWAYS, regardless
of sign. The Verilog equivalent is **unconditional `+HALF` then `>>>`**, NOT
sign-aware rounding. Sign-aware rounding (`(neg ? HALF-1 : HALF)`) diverges
from the golden at tie values and produces a ~22% mismatch rate on
non-degenerate residuals — that is the `scale_factor_misapplied` /
`unconditional_half_round_required` failure mode.

The two scales are FUSED in the RTL: factor out `out_scale` once so the
shift+saturate stage runs against an INT32 sum instead of doing a
floating-point divide. Define the fused ratios:

```
r_lhs = lhs_in_scale / out_scale     # ratio, may be < 1 or > 1
r_rhs = rhs_in_scale / out_scale
```

Then pick `FUSED_SHIFT` (typically 22) and:

```
LHS_FUSED_MULT = round(r_lhs * 2^FUSED_SHIFT)     # signed const, up to 23 bits
RHS_FUSED_MULT = round(r_rhs * 2^FUSED_SHIFT)
FUSED_HALF     = 1 << (FUSED_SHIFT - 1)            # 2^(FUSED_SHIFT-1)
```

⛔ **DO NOT** put raw `lhs_scale_factor`/`rhs_scale_factor` into the fused
multipliers without dividing by `out_scale` first. That over-scales every
output by `out_scale` and produces `max_error≈119, mismatch_count≈3.3M/6.4M`
on the first attempt — a classic `scale_factor_misapplied` regression.

`sdk/orchestrate.ts:computeAddFusedScaleApprox` is the reference: it
returns `(mult, shift)` for the ratio `input_scale / output_scale` and is
guaranteed to keep `mult < 2^23`. Use the same shift for both lhs and rhs by
picking `FUSED_SHIFT = max(lhs.shift, rhs.shift)` and rescaling the
narrower constant.

Per-channel datapath (one channel per cycle, three registered stages):

```
lhs_i   = $signed(lhs_buf[ch])                              // INT8
rhs_i   = $signed(rhs_buf[ch])                              // INT8
lhs_term <= lhs_i * LHS_FUSED_MULT                          // PROD_W bits
rhs_term <= rhs_i * RHS_FUSED_MULT                          // PROD_W bits
sum_pre   = $signed(lhs_term) + $signed(rhs_term)           // SUM_W combinational wire
sum_term <= sum_pre + FUSED_ROUND_BIAS                      // [INVARIANT:ROUNDING]  unconditional +HALF
v_tmp     = sum_term >>> FUSED_SHIFT                        // arithmetic right shift
out8      = (v_tmp >  127) ? 8'sd127
          : (v_tmp < -128) ? 8'h80
          :                  v_tmp[7:0]
```

`PROD_W = 8 + SCALE_CONST_W` where `SCALE_CONST_W` is the bit-width of the
fused multipliers (24 covers `mult < 2^23` signed). `SUM_W = PROD_W + 2`
absorbs the lhs+rhs sum plus the round bias without overflow.

## Latency contract

Use the LayerIR value as the BASELINE; do not re-derive. The current
serialized add contract uses one multiplier pair (two DSPs total) and
processes one channel per cycle:

```
LayerIR.pipeline_latency_cycles = output_channels + 3   # flat-bus baseline
```

Under the tile-32 ABI, the actual valid_out cycle is shifted by the gather
and stream phases:

```
effective_latency = BEATS_PER_PIXEL (gather) + OC (compute) + 2 (drain)
                  = OC/32 + OC + 2
```

For OC=256 this is `8 + 256 + 2 = 266`. The Assayer accepts drift up to 1%
(or 256 cycles, whichever is larger) over the IR baseline when sim is
byte-exact, so this 7-cycle drift is within tolerance — do not hand-edit
LayerIR to chase it.

The three pipeline stages run unconditionally (advance every cycle while
`stage1_active`, `stage2_valid`, or `stage3_valid` are true):

1. **Stage 1 — multiplies.** For the current channel only,
   `lhs_term <= $signed(lhs_buf[ch_s1]) * LHS_FUSED_MULT` and
   `rhs_term <= $signed(rhs_buf[ch_s1]) * RHS_FUSED_MULT`. With
   `(* use_dsp = "yes" *)` on each registered product, Vivado infers two
   DSP48E1 multipliers total.
2. **Stage 2 — sum + UNCONDITIONAL +HALF round bias.** Sum the two products
   and add `FUSED_ROUND_BIAS = 1 << (FUSED_SHIFT-1)` in the same registered
   stage. Declare the unbiased combinational sum as a **module-scope
   `wire`** (Verilog-2001 forbids `wire` decls inside an `always` block):
   ```verilog
   // module scope, alongside other wire decls:
   wire signed [SUM_W-1:0] sum_pre = $signed(lhs_term) + $signed(rhs_term);

   // inside the stage-2 always block:
   sum_term <= sum_pre + FUSED_ROUND_BIAS;   // [INVARIANT:ROUNDING]
   ```
   ⛔ **DO NOT** make this sign-aware
   (`(sum_pre[SUM_W-1] ? FUSED_HALF-1 : FUSED_HALF)`). The golden uses
   `round_half_up_toward_pos_inf`; sign-aware rounding diverges at ties,
   producing ~22% mismatch — the original tile=16 references
   (`node_add_9` first attempt) failed exactly here.
3. **Stage 3 — shift + saturate.** `(sum_term >>> FUSED_SHIFT)`, clamp to
   INT8, and write the saturated byte into
   `out_beats[ch_s3 / CHANNEL_TILE][(ch_s3 % CHANNEL_TILE)*8 +: 8]`.

Do not instantiate one multiplier per channel. The old 3-cycle fully parallel
implementation is numerically correct but architecturally bad at OC=256: it
creates 512 constant multipliers, consumes all 240 Artix-7 DSPs, spills the
remaining multipliers into LUTs, and turns residual add into the largest LUT
consumer in layer 1.

## Required FSM (tile-32 ABI — REQUIRED for `io_mode==channel_tiled`)

This is a four-state controller `IDLE / GATHER / COMPUTE / STREAM`. The
skeleton below is structurally identical to the proven-passing tile=16 add
(`node_add_1.v` first-light), just with `CHANNEL_TILE=32`, 512-bit
`data_in`, and 256-bit `data_out`. Treat it as the canonical shape — your
generated module should differ ONLY in scale-constant values, parameter
widths derived from OC, and timestamps. Do not invent a different FSM
topology.

### Port and parameter declarations

```verilog
module <module_id> (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    output reg          ready_in,
    input  wire [511:0] data_in,   // [255:0] = lhs tile, [511:256] = rhs tile
    output reg          valid_out,
    output reg  [255:0] data_out
);

    localparam integer OC              = <OC>;          // e.g. 256
    localparam integer CHANNEL_TILE    = 32;            // FIXED for tile-32 ABI
    localparam integer BEATS_PER_PIXEL = OC / 32;        // e.g. 8 for OC=256

    localparam integer FUSED_SHIFT     = <shift>;       // e.g. 22 from computeAddFusedScaleApprox
    localparam integer MULT_W          = 24;            // signed: holds mult < 2^23
    localparam integer PROD_W          = 32;            // 8 + MULT_W
    localparam integer SUM_W           = 34;            // PROD_W + 2

    // Fused multipliers come from r_lhs = lhs_scale / out_scale, r_rhs = rhs_scale / out_scale:
    localparam signed [MULT_W-1:0] FUSED_LHS_MULT   = 24'sd<round(r_lhs * 2^FUSED_SHIFT)>;
    localparam signed [MULT_W-1:0] FUSED_RHS_MULT   = 24'sd<round(r_rhs * 2^FUSED_SHIFT)>;
    localparam signed [SUM_W-1:0]  FUSED_ROUND_BIAS = 34'sd<1 << (FUSED_SHIFT - 1)>;  // unconditional +HALF
    localparam signed [SUM_W-1:0]  SAT_HI =  34'sd127;
    localparam signed [SUM_W-1:0]  SAT_LO = -34'sd128;

    localparam [1:0] ST_IDLE = 2'd0, ST_GATHER = 2'd1, ST_COMPUTE = 2'd2, ST_STREAM = 2'd3;
    reg [1:0] state;

    // OC-deep tile-channel buffers (gathered over BEATS_PER_PIXEL input beats):
    reg signed [7:0] lhs_buf [0:OC-1];
    reg signed [7:0] rhs_buf [0:OC-1];
    // BEATS_PER_PIXEL-deep output staging (filled during COMPUTE, drained in STREAM):
    reg [255:0]      out_beats [0:BEATS_PER_PIXEL-1];

    // Counter widths sized so MAX value fits exactly (BPP < 16 fits in 4 bits, OC <= 1024 in 10 bits).
    reg [3:0]  in_beat_count, out_beat_count;       // size for max BEATS_PER_PIXEL
    reg [9:0]  ch_s1, ch_s2, ch_s3;             // size for max OC; OC=256 fits in 8 bits

    reg stage1_active, stage2_valid, stage3_valid;

    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] lhs_term;
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] rhs_term;
    reg signed [SUM_W-1:0]  sum_term, v_tmp;
    // sum_pre is combinational; declare at MODULE SCOPE (Verilog-2001 forbids wire decl inside always):
    wire signed [SUM_W-1:0] sum_pre = $signed(lhs_term) + $signed(rhs_term);

    integer i;
```

### Sequential body (single always block, mirrors the proven tile=16 reference)

```verilog
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= ST_IDLE; ready_in <= 1'b1; valid_out <= 1'b0; data_out <= 256'd0;
        in_beat_count <= 0; out_beat_count <= 0;
        ch_s1 <= 0; ch_s2 <= 0; ch_s3 <= 0;
        stage1_active <= 0; stage2_valid <= 0; stage3_valid <= 0;
        lhs_term <= 0; rhs_term <= 0; sum_term <= 0; v_tmp <= 0;
    end else begin
        // ----- 3-stage MAC pipeline (advances every cycle independent of FSM state) -----
        if (stage1_active) begin
            lhs_term <= $signed(lhs_buf[ch_s1]) * FUSED_LHS_MULT;
            rhs_term <= $signed(rhs_buf[ch_s1]) * FUSED_RHS_MULT;
            ch_s2 <= ch_s1; stage2_valid <= 1'b1;
        end else stage2_valid <= 1'b0;

        if (stage2_valid) begin
            sum_term <= sum_pre + FUSED_ROUND_BIAS;     // [INVARIANT:ROUNDING] unconditional +HALF
            ch_s3 <= ch_s2; stage3_valid <= 1'b1;
        end else stage3_valid <= 1'b0;

        if (stage3_valid) begin
            v_tmp = sum_term >>> FUSED_SHIFT;            // blocking: same-cycle saturate
            out_beats[ch_s3 / CHANNEL_TILE][(ch_s3 % CHANNEL_TILE)*8 +: 8] <=
                (v_tmp > SAT_HI) ? 8'sd127 :
                (v_tmp < SAT_LO) ? 8'h80   : v_tmp[7:0];
        end

        // ----- FSM controlling stage1_active arming and IO handshakes -----
        case (state)
            ST_IDLE: begin
                valid_out <= 1'b0;
                if (valid_in && ready_in) begin
                    // beat 0: lhs tile in [255:0], rhs tile in [511:256], lanes 0..31
                    for (i = 0; i < CHANNEL_TILE; i = i + 1) begin
                        lhs_buf[i] <= $signed(data_in[i*8 +: 8]);
                        rhs_buf[i] <= $signed(data_in[256 + i*8 +: 8]);
                    end
                    in_beat_count <= 4'd1;
                    state       <= ST_GATHER;
                end
            end
            ST_GATHER: begin
                if (valid_in && ready_in) begin
                    // beat in_beat_count: lanes (in_beat_count*32)..(in_beat_count*32+31)
                    for (i = 0; i < CHANNEL_TILE; i = i + 1) begin
                        lhs_buf[in_beat_count*CHANNEL_TILE + i] <= $signed(data_in[i*8 +: 8]);
                        rhs_buf[in_beat_count*CHANNEL_TILE + i] <= $signed(data_in[256 + i*8 +: 8]);
                    end
                    if (in_beat_count == (BEATS_PER_PIXEL[3:0] - 4'd1)) begin
                        ready_in      <= 1'b0;             // [INVARIANT:READY_IN_GATING]
                        state         <= ST_COMPUTE;
                        ch_s1         <= 0;
                        stage1_active <= 1'b1;
                        in_beat_count   <= 0;
                    end else
                        in_beat_count <= in_beat_count + 4'd1;
                end
            end
            ST_COMPUTE: begin
                // Drive stage1_active for OC cycles, then let the 2-stage tail drain.
                if (stage1_active) begin
                    if (ch_s1 == (OC - 1)) stage1_active <= 1'b0;
                    else                    ch_s1 <= ch_s1 + 10'd1;
                end
                // When stage3 writes the LAST channel of the LAST tile, emit beat 0
                // IN THE SAME CYCLE and transition. The "ch_s3 == OC-1" cycle is the
                // one where out_beats[BEATS_PER_PIXEL-1] just got its last byte.
                if (stage3_valid && ch_s3 == (OC - 1)) begin
                    state        <= ST_STREAM;
                    data_out     <= out_beats[0];          // [INVARIANT:VALID_OUT_LATENCY]
                    valid_out    <= 1'b1;
                    out_beat_count <= 4'd1;
                end
            end
            ST_STREAM: begin
                if (out_beat_count < BEATS_PER_PIXEL[3:0]) begin
                    data_out     <= out_beats[out_beat_count];
                    valid_out    <= 1'b1;
                    out_beat_count <= out_beat_count + 4'd1;
                end else begin
                    valid_out    <= 1'b0;
                    state        <= ST_IDLE;
                    ready_in     <= 1'b1;
                    out_beat_count <= 0;
                end
            end
            default: state <= ST_IDLE;
        endcase
    end
end
endmodule
```

### Register naming rule (preflight gate)

The deterministic contract-conformance preflight scans for at least one
register/wire/integer declaration containing the substring `beat` in its
identifier (case-insensitive). The proven-passing tile=16 references use
`in_beat_idx` / `out_beat_idx` — keep that exact pattern, or use the
canonical `in_beat_count` / `out_beat_count` / `cur_beat_stream` names
the fix_hint suggests. ⛔ **DO NOT** rename these to `gather_count`,
`stream_count`, `tile_idx`, etc. The preflight (`hasBeatCounterDeclaration`
in `sdk/orchestrate.ts`) requires the literal `beat` token, and renaming
trips `contract_tiled_streaming_beat_counter_missing` before iverilog
even runs.

### Bus and timing contract (STRICT)

- `input  wire [511:0] data_in` — `[255:0]` = lhs tile, `[511:256]` = rhs tile, 32 INT8 channels per half.
- `output reg  [255:0] data_out` — one 256-bit output tile beat (32 INT8 channels).
- `BEATS_PER_PIXEL = OC / 32` input beats consumed (gather), then `BEATS_PER_PIXEL`
  output beats produced (stream). 1:1 cadence overall, but NOT cycle-for-cycle —
  the OC-cycle COMPUTE phase sits between them.
- `ready_in` LOW for the entire COMPUTE phase (`[INVARIANT:READY_IN_GATING]`).
- `valid_out` rises on the same cycle that `data_out` first equals `out_beats[0]`,
  in COMPUTE when `stage3_valid && ch_s3 == OC-1` (`[INVARIANT:VALID_OUT_LATENCY]`).
- Effective latency from first `valid_in` handshake to first `valid_out` =
  `BEATS_PER_PIXEL + OC + 2` cycles (gather + compute + 2-stage drain).

The output-counter preflight rule does not apply to `add` (it is triggered by
op_type; add is excluded).

## Required registers

The skeleton above already enumerates the canonical set. Required:

- `reg signed [7:0] lhs_buf [0:OC-1];` / `reg signed [7:0] rhs_buf [0:OC-1];` — tile buffers gathered over BEATS_PER_PIXEL beats.
- `reg [255:0]      out_beats [0:BEATS_PER_PIXEL-1];` — output tile staging.
- `reg [...]        in_beat_count, out_beat_count;` — gather/stream beat counters (4 bits covers BEATS_PER_PIXEL up to 16).
- `reg [...]        ch_s1, ch_s2, ch_s3;` — per-stage channel index (sized to hold OC).
- `reg              stage1_active, stage2_valid, stage3_valid;` — pipeline flags.
- `(* use_dsp = "yes" *) reg signed [PROD_W-1:0] lhs_term;` / `rhs_term;` — stage-1 products.
- `reg signed [SUM_W-1:0] sum_term;` — stage-2 biased sum.
- `wire signed [SUM_W-1:0] sum_pre = $signed(lhs_term) + $signed(rhs_term);` at MODULE SCOPE.
- No weights, no biases, no `$readmemh`.

`PROD_W = 8 + MULT_W` (8-bit channel × signed fused multiplier). `MULT_W = 24`
suffices because `computeAddFusedScaleApprox` caps `mult < 2^23` signed.
`SUM_W = PROD_W + 2` absorbs the lhs+rhs sum plus the round bias.

## Rules

- Every arithmetic signal on the datapath MUST be declared `signed` or
  wrapped with `$signed(...)`. A silently unsigned residual add produces
  mass saturation at ±127 on negative residuals — a classic symptom of
  `sign_extension_error`.
- Saturate to INT8 at the output, not internally. Internal widening is
  free; internal truncation discards information.

## Known failure modes

- `sign_extension_error` — missing `$signed()` around one of the operands
  of the multiply. Outputs saturate to 127 for negative residuals.
- `scale_factor_misapplied` — using `scale_factor` where `lhs_scale_factor`
  or `rhs_scale_factor` was meant, or vice versa.
- `port_width_mismatch` — `data_in` declared as `OC*8` instead of
  `OC*16`, because Foundry missed the lhs|rhs packing.
