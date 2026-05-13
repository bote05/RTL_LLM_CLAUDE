# 09 - DRAM-backed weights contract

This is contract guidance, not a passing RTL reference. Use it when
`contract_id == "dram-backed-weights"`.

## FORBIDDEN PATTERNS — read first

The structural preflight gate fails the build immediately if it sees any of
these. Do not write them, do not "stub them and revisit", do not justify them
in a comment. If you find yourself reaching for one of these, stop that design
path and implement the required AXI prefetch FSM instead.

1. `assign weights_arvalid = 1'b0;` — or any other tie-off of the AXI
   read-address valid. The contract requires a real AR-channel FSM that
   issues reads and tracks `arready` handshakes.
2. `$readmemh(... weights ...)` of the full weight tensor (or any subset
   larger than the active per-pass window). Weights enter this module only
   over the AXI R-channel. There is no on-chip ROM for the layer tensor.
3. `reg [W_W-1:0] weights_mem [0:OC*IC-1];` (or any storage sized
   `OC*K_TOTAL`). Only `MP*K_TOTAL` bytes — the active pass window — may be
   stored on chip at one time, optionally double-buffered (`2*MP*K_TOTAL`).
4. Comments asserting that "AXI is out of scope for the latency contract"
   or "per-tile streaming is deferred". The latency contract REQUIRES AXI;
   prefetching pass 0 before raising `ready_in` is exactly how you meet it
   (see "Exact-latency rule" below).
5. Negative rounding-bias constants such as `ROUND_BIAS_NEG = -...` or
   ternaries such as `scaled[MSB] ? -SCALE_ROUND_HALF : SCALE_ROUND_HALF`.
   Verilog `>>>` already floors negative values toward -inf; the negative
   branch must add `(HALF - 1)`, not subtract `HALF`.

If the contract_id passed to you is `dram-backed-weights` then the on-chip
weight ROM design is wrong, not just suboptimal. The orchestrator selected
this contract because the layer's `OC*K_TOTAL` exceeds the on-chip weight
budget; reverting to `$readmemh` puts you over budget and invalidates the
synthesis target.

## Multi-vector test sequencing — CRITICAL

The static Verilator testbench drives **multiple distinct input vectors**
through the DUT in a single simulation run (typically 4-8 per layer). The
goldin file's header carries `num_vectors` ≥ 1. Per-vector latency is
re-measured from each vector's first accepted `valid_in` to its first
`valid_out` and must equal `pipeline_latency_cycles` for EVERY vector.

This means:

1. **Do NOT use a terminal `ST_DONE`** that holds `valid_out=0; ready_in=0;`
   forever. Foundry's instinct is to write a "done" state that locks the FSM;
   that locks out vector N+1's input and the testbench hangs. The structural
   preflight gate `dram_backed_weights_terminal_done_lock` rejects this.
2. After the last active output pixel of vector N is emitted, the FSM must
   reset all per-vector state (in_row, in_col, in_pixel_counter,
   active_cache_sel, ar prefetch counters, MAC pipeline, accumulators,
   out_buffer) and re-enter `ST_INIT_BOOT` for vector N+1. The module must
   present the same shape vector N+1 sees as it would after `!rst_n`.
3. The **per-vector latency contract is exact, not approximate**. The
   Assayer reports `worst_vector_actual_cycles` when any vector misses;
   off-by-1 fails timing.

## Stop after the last active output pixel — CRITICAL

For stride > 1, only `OH × OW` of the `IH × IW` input-grid positions are
active. The remaining `IH×IW − OH×OW` positions are inactive and produce
no output. **Track an `active_pixel_counter` and force ST_DONE when it
reaches `OH × OW`. Do NOT iterate trailing inactive input pixels.**

The reason is a TB/RTL handshake-protocol pitfall: the testbench advances
to vector N+1 as soon as `output_idx >= outputs.size()` (= `OH × OW × OUT_BEATS`
for vector N). At that moment, vector N's last active pixel has been
emitted. If the FSM keeps iterating inactive trailing pixels (e.g. for
stride-2 with IH=14, pixels (12,13) through (13,13) — 15 inactive pixels
× 32 input beats = 480 beats), it raises `ready_in` for those pixels and
the testbench drives **vector N+1's first 480 beats into RTL's vector-N
tail**. Vector N+1's nominal "first input pixel" then receives data that
actually came from pixel-15 of vector N+1, not pixel-0.

```verilog
// CORRECT: stop on output-pixel count, not input-pixel count.
if (pix_active && active_pixel_counter + 1 == OH*OW) begin
    state <= ST_DONE;
end else if (in_pixel_counter + 1 == TOTAL_IN_PIXELS) begin
    // Defensive fallback only.
    state <= ST_DONE;
end else begin
    state <= ST_INIT_BOOT;  // next pixel
end
```

Failure signature when this rule is violated: vector 0 passes 100% (TB
exits v=0 cleanly; v=0 has no preceding tail). Vectors 1..N each match
~30% with small ±1-3 LSB errors AND, more diagnostically, the
**accumulated MAC values for vectors 1+'s "first pixel" match the int64
reference for an OFFSET pixel** (e.g. pixel (1,1) for stride-2 with the
15-pixel tail, or pixel (offset/IW, offset%IW) more generally where
`offset = floor(trailing_inactive_count)`).

## $readmemh paths must be absolute or come from a parameter — CRITICAL

`$readmemh("output/weights/foo.hex", rom)` with a **relative** path
silently fails at simulation start: Verilator (and iverilog under the
Assayer) compile in a temporary build directory with no `output/` subtree,
so the file is not found and the array stays at its default zero value.
**No warning or error is emitted by Verilator.** Bias/weight ROMs that
should hold real data instead read all-zero, and the RTL produces
plausible-but-wrong outputs (zero-bias outputs that test as "close but
not bit-exact").

The structural preflight gate `readmemh_relative_path_forbidden` rejects
any `$readmemh` whose first argument is a string literal that does not
start with `/` (POSIX absolute), `[A-Z]:` (Windows drive), or that is not
a top-level `parameter`. Use either:

```verilog
// Option A: top-level parameter (preferred — overridable per-instance).
parameter BIAS_PATH = "C:/abs/path/to/bias.hex";
initial $readmemh(BIAS_PATH, bias_rom);

// Option B: absolute path string from the LayerIR's bias_path field
// (which the orchestrator already provides as an absolute POSIX path).
initial $readmemh("/abs/path/to/bias.hex", bias_rom);
```

## Interface

The top level includes the seven base activation-stream ports from
`01_context.md` plus the AXI read-channel ports declared in
`contracts/dram-backed-weights/metadata.json`:

- `weights_arvalid`, `weights_arready`, `weights_araddr`, `weights_arlen`
- `weights_rvalid`, `weights_rready`, `weights_rdata`, `weights_rlast`

Do not collapse this contract back to the seven-port flat-bus interface.
Do not store the full `OC*K_TOTAL` weight tensor on chip.

## Exact-latency rule

The verifier measures `pipeline_latency_cycles` from the first accepted
`valid_in` beat. Therefore any memory warm-up needed for output-channel pass 0
must complete before the module raises `ready_in` for the first input beat.

Required sequence:

1. After reset, keep `ready_in = 0` while prefetching pass-0 weights.
2. Issue AXI reads for the pass-0 weight window.
3. Cache only the active weight window, not the full layer tensor.
4. Raise `ready_in = 1` only after pass-0 weights are available.
5. Once the first input beat is accepted, the first `valid_out` must occur at
   exactly `pipeline_latency_cycles`.

Do not add first-pass DRAM latency after `first_valid_in`; that violates the
LayerIR timing contract.

## Weight window formulas

Use formulas, not module-specific constants:

```
K_TOTAL        = (IC / groups) * KH * KW
OC_PASSES      = ceil(OC / MP)
PASS_WEIGHTS   = MP * K_TOTAL
PASS_BYTES     = PASS_WEIGHTS
AXI_BYTES      = 8
BEATS_PER_PASS = ceil(PASS_BYTES / AXI_BYTES)
pass_base_addr = oc_pass * PASS_BYTES
```

Each 64-bit AXI data beat carries eight INT8 weights in little-endian byte
order. Address units are bytes into `weights_path`.

## Progress rule

For an exact-latency design, overlap next-pass weight fetch with current-pass
compute whenever possible:

- compute pass `N` from cache A
- prefetch pass `N+1` into cache B
- swap caches at the pass boundary

If a simpler first version fetches pass `N+1` only after pass `N` finishes, it
must either still meet the declared latency or fail deterministically. Do not
hide the added cycles by changing `pipeline_latency_cycles`.

### Final-pass prefetch guard — `<=`, not `<` [INVARIANT:DRAM_PREFETCH_GUARD]

The end-of-pass kick that schedules the *next* pass's prefetch must use `<=`
on `OC_PASSES`, not `<`:

```verilog
// CORRECT — kick prefetch for pass N+1 whenever N+1 < OC_PASSES (i.e. there
// IS a next pass to compute). With ping-pong double-buffering, the LAST
// pass needs its weights too: when oc_pass = OC_PASSES - 2, oc_pass + 2 =
// OC_PASSES, and we still need to fire ar_kick to load the cache the LAST
// pass will read.
if (oc_pass + 8'd2 <= OC_PASSES)
    ar_kick <= 1'b1;
```

```verilog
// WRONG — `<` skips the prefetch when oc_pass = OC_PASSES - 2, so the cache
// the LAST pass reads keeps STALE contents from oc_pass - 2 (whichever
// half of the ping-pong was loaded two passes ago). Surfaces as a clean
// `sim_completed_mismatch` with timing_pass=true: ALL outputs received,
// 1/OC_PASSES of channels per pixel are wrong (they used the stale OC's
// weights). Bug pattern: max_error ≈ 6, ~0.5–1% mismatch rate, signed-
// error skew toward whatever sign the stale OCs encode.
if (oc_pass + 8'd2 < OC_PASSES)   // BUG
    ar_kick <= 1'b1;
```

Diagnosed on node_conv_292 (3×3 stride=1 pad=1, OC=512, OC_PASSES=128): the
`<` form left pass 127's cache holding pass 125's weights, so OC=508–511
were computed against OC=500–503's weights. Probe at `oc_pass=127`
showed `acc[0]=15` for pixel 1, vs golden's expected `−178` — exactly the
value a pass-125-weights × pixel-1-inputs MAC produces.

## Output coordinates (stride > 1) — CRITICAL

dram-backed-weights layers in this codebase routinely use stride > 1
(e.g. node_conv_288 is 1×1 stride 2, IH=14 → OH=7). The stream-side
coordinate counter that drives output emission MUST use `OH` / `OW` for
its bounds, never `IH` / `IW`. Mixing them is a deadlock trap:

```verilog
// WRONG — uses input dims; FSM keeps emitting (or waiting to emit) past
// the real output frame end, then deadlocks waiting for a non-existent
// next input beat.
if (out_w == IW-1) begin out_w <= 0; ... end
else out_w <= out_w + 1;

// CORRECT — uses output dims. Better still: drive a `done` flag from the
// compute counter that ticks once per emitted output pixel.
if (out_w == OW-1) begin out_w <= 0; ... end
else out_w <= out_w + 1;
```

Failure signature when this rule is violated: `outputs_received` falls
short of `outputs_expected` by an exact multiple of `OUT_BEATS_PER_PIXEL`,
`output_gap_histogram` shows the missing range packed into one tail
bucket, `status_class` is `sim_stalled`. See `08_common_bugs.md` →
"Stream-side coordinates confused with input coordinates (stride > 1)"
for the full diagnosis recipe.

## Rounding (sign-aware)

Every dram-backed conv requantises after the bias add, same as flat-bus
and tiled-streaming. Use the canonical sign-aware bias (see
`01_context.md` "Scale-shift rounding — MANDATORY"):

### FORBIDDEN ROUNDING PATTERNS — read before coding

The structural preflight gate fails the build immediately if it sees any of
these. Do not write them, do not mark them `[INVARIANT:ROUNDING]`, and do not
justify them as "sign-aware":

1. `ROUND_BIAS_NEG = -...`
2. `scaled[MSB] ? -HALF : HALF`
3. `scaled[MSB] ? -SCALE_ROUND_HALF : SCALE_ROUND_HALF`

For dram-backed layers this matters more than usual because high `K_TOTAL`
turns a one-LSB rounding skew into a large signed-diff drift. Use this shape
instead:

```verilog
v_tmp = (scaled[lane] +
         (scaled[lane][SCALED_W-1] ? (SCALE_ROUND_HALF - 1)
                                   : SCALE_ROUND_HALF)
        ) >>> SCALE_SHIFT;
```

Subtracting `SCALE_ROUND_HALF` for negatives over-rounds (Verilog `>>>`
already floors toward -inf). High-fan-in dram-backed layers
(K_TOTAL ≥ 1024) accumulate the asymmetric drift faster than flat-bus
layers and trip bit-exact verification with `max_error ≥ 8` and a
positive-skewed signed-diff distribution.

## Line-buffer storage — bank into LUT-RAM-friendly arrays under the Vivado variable-size cap [INVARIANT:ACTIVATION_BUFFER_BANKING]

The structural preflight gate
`activationMemoryBitLimitViolations` (in `sdk/orchestrate.ts`) fails the
build before Vivado if any single wide-word activation reg variable
(e.g. `line_buf`) totals more than 900,000 bits. Vivado hard-errors
with `[Synth 8-4556]` on any single unpacked reg above ~1,048,576 bits.

Two failure modes the gate prevents — both must be considered together:

1. **Vivado variable-size cap (hard error).** A single
   `reg [W-1:0] line_buf [0:D-1][0:M-1]` with `D*M*W > 1_048_576` is
   rejected outright by `synth_design`. Rounding to the next pow2 makes
   this WORSE, not better — e.g. `[0:255][0:15]` of 256-bit beats is
   1,048,576 bits exactly and trips the cap, while the layer's actual
   need was 196 pixels (under cap as one variable, but mapped to flops).
2. **LUT-RAM granularity (slow-synth).** Vivado's distributed RAM
   granule is 32- or 64-deep. A depth that aligns to neither (e.g. 196)
   forces flip-flop mapping for the entire array, ballooning post-synth
   `report_timing_summary -check_timing_verbose` wall time to 45+ min
   on 800k+ FFs without affecting PPA usefully.

**Solution — bank the memory AND keep each bank 1D-unpacked.** Split
one logical line buffer into multiple `BANK_DEPTH=64` variables
(LUT-RAM-friendly granule), AND collapse the inner unpacked dim into
the packed width. The 1D-unpacked × wide-packed shape is the ONLY one
Vivado reliably infers as distributed LUT-RAM for this contract — the
`cache_a` / `cache_b` AXI prefetch buffers use this exact shape and
get cleanly mapped to `RAM64M8` primitives.

```verilog
// IH=IW=14 (stride-2 input) → TOTAL_IN_PIXELS = 196 actual pixels.
// preferred_bank_depth = 64 (LUT-RAM granule)
// bank_count = ceil(196 / 64) = 4
// bits_per_bank = 64 * IN_BEATS * BEAT_BITS = 64 * 16 * 256 = 262,144  ✓
localparam BANK_DEPTH      = 64;
localparam BANK_COUNT      = (TOTAL_IN_PIXELS + BANK_DEPTH - 1) / BANK_DEPTH; // 4
localparam LINE_WORD_BITS  = IN_BEATS * BEAT_BITS;   // 16 * 256 = 4096

// 1D unpacked × wide packed — Vivado infers LUT-RAM cleanly.
(* ram_style = "distributed" *)
reg [LINE_WORD_BITS-1:0] line_buf_b0 [0:BANK_DEPTH-1];
(* ram_style = "distributed" *)
reg [LINE_WORD_BITS-1:0] line_buf_b1 [0:BANK_DEPTH-1];
(* ram_style = "distributed" *)
reg [LINE_WORD_BITS-1:0] line_buf_b2 [0:BANK_DEPTH-1];
(* ram_style = "distributed" *)
reg [LINE_WORD_BITS-1:0] line_buf_b3 [0:BANK_DEPTH-1];

wire [$clog2(BANK_COUNT)-1:0] bank_idx  = in_pixel_counter[$clog2(TOTAL_IN_PIXELS)-1:$clog2(BANK_DEPTH)];
wire [$clog2(BANK_DEPTH)-1:0] bank_addr = in_pixel_counter[$clog2(BANK_DEPTH)-1:0];

// Write: insert one beat into the wide word using a bit-select.
// Vivado understands `wide_reg[var*W +: W] <= narrow` as a partial RAM
// write (the unwritten bits are read-modify-write, inferred properly).
always @(posedge clk) begin
    case (bank_idx)
        2'd0: line_buf_b0[bank_addr][in_beat_index*BEAT_BITS +: BEAT_BITS] <= data_in;
        2'd1: line_buf_b1[bank_addr][in_beat_index*BEAT_BITS +: BEAT_BITS] <= data_in;
        2'd2: line_buf_b2[bank_addr][in_beat_index*BEAT_BITS +: BEAT_BITS] <= data_in;
        2'd3: line_buf_b3[bank_addr][in_beat_index*BEAT_BITS +: BEAT_BITS] <= data_in;
    endcase
end

// Read: fetch the whole wide word, then beat-select downstream.
reg [LINE_WORD_BITS-1:0] line_word_q1;
reg [$clog2(IN_BEATS)-1:0] beat_sel_q1;
always @(posedge clk) begin
    case (mac_bank_idx_q1)
        2'd0: line_word_q1 <= line_buf_b0[mac_bank_addr_q1];
        2'd1: line_word_q1 <= line_buf_b1[mac_bank_addr_q1];
        2'd2: line_word_q1 <= line_buf_b2[mac_bank_addr_q1];
        2'd3: line_word_q1 <= line_buf_b3[mac_bank_addr_q1];
    endcase
    beat_sel_q1 <= mac_in_beat_q1;
end

reg [BEAT_BITS-1:0] line_buf_word_q2;
always @(posedge clk) begin
    line_buf_word_q2 <= line_word_q1[beat_sel_q1*BEAT_BITS +: BEAT_BITS];
end
```

**Why this matters — empirical evidence (conv_284, 196 pixels):**

| Storage shape | LUT-RAM? | Total FDRE | synth_design | post-synth wall |
|---|---|---|---|---|
| `[0:195][0:511]` per-byte | no (giant mux) | — | aborted at 4h | n/a |
| `[0:255][0:15]` 2D wide-word (pow2 round) | no, hits 1 Mb cap | — | rejected | n/a |
| `[0:63][0:15]` 2D × 4 banks | **no, FF-mapped** | 1,048,904 | 8m 58s | 40+ min (stuck in `report_timing_verbose`) |
| `[0:63]` 1D × 4 banks of LINE_WORD_BITS | **yes (RAM64M8)** | ~250k | expected ~6 min | expected ~2 min |

The 4× FDRE count from the 2D shape doesn't change PPA but it cripples
`report_timing_summary -check_timing_verbose` because the verbose
analysis walks every endpoint. 250k vs 1.05M endpoints is the
difference between "synth finishes in 10 min" and "stuck for an hour".

**Rule of thumb (architecture-agnostic):**

```
bits_per_array = depth * inner_dims_product * element_bits
if bits_per_array >= 900_000:
    bank_depth  = 64   (drop to 32 if 64 * inner * width still busts the cap)
    bank_count  = ceil(logical_depth / bank_depth)
else:
    one variable is fine, just keep `depth` aligned to a LUT-RAM granule
    (16, 32, 64, 128 are clean) when possible.
```

**Memory writes must live in a posedge-clk-only block — NO async reset
on the activation memory itself [INVARIANT:RAM_NO_ASYNC_RESET].**

Vivado refuses to infer block-RAM or distributed-RAM for any reg array
that is ALSO written inside an `always @(posedge clk or negedge rst_n)`
block. The synth log emits these tells:

```
WARNING: [Synth 8-4767] Trying to implement RAM 'X' in registers.
   Block RAM or DRAM implementation is not possible
   Reason: RAM is sensitive to asynchronous reset signal.
ERROR:   [Synth 8-3391] Unable to infer a block/distributed RAM ...
   Failed to dissolve the memory into bits because the number of bits
   (...) is too large
```

Vivado then can't fall back to flip-flops because the dissolved-bit count
exceeds the elaboration limit, and synth_design HARD-FAILS in ~8 seconds.

Even if the always block's reset branch never assigns to `line_buf`
(e.g. only resets neighbouring FSM state), Vivado's RAM-inference checker
sees the storage in a reset-sensitive block and refuses. The reset has
to be syntactically absent from the memory's owning always.

**Do this** — put activation memory writes in a dedicated always block
with `@(posedge clk)` only, no `or negedge rst_n`, no reset clause. The
cache_a/cache_b AXI-prefetch arrays in this same module already use
exactly this pattern and Vivado maps them to `RAM64M8` primitives.

```verilog
// RIGHT — line_buf banks have their own posedge-only always, no reset.
// (The "stale data on power-up" is harmless: ST_INPUT pre-fills every
//  active address before the MAC pipeline ever reads from line_buf.)
always @(posedge clk) begin
    case (bank_idx)
        2'd0: line_buf_b0[bank_addr][in_beat_index*BEAT_BITS +: BEAT_BITS] <= data_in;
        2'd1: line_buf_b1[bank_addr][in_beat_index*BEAT_BITS +: BEAT_BITS] <= data_in;
        // ... other banks
    endcase
end

// WRONG — touching line_buf inside the async-reset always voids RAM inference
// even if the reset branch only clears unrelated state:
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= ST_INIT;
        // ... resets for OTHER state ...
    end else begin
        case (state)
            ST_INPUT: if (valid_in && ready_in)
                line_buf_b0[bank_addr][in_beat_index*BEAT_BITS +: BEAT_BITS] <= data_in;
        endcase
    end
end
```

The FSM state machine, counters, and handshake registers KEEP their
`posedge clk or negedge rst_n` async-reset block — only the memory
writes move to their own posedge-only block. The bank-select signals
(`bank_idx`, `bank_addr`, `in_beat_index`, the gating condition like
`valid_in && ready_in`) come in as combinational wires or registered
values from the main FSM block, exactly like cache_a/cache_b receive
their `cache_we`, `cache_we_addr`, `cache_we_data` from the AR
state machine.

**Why 64 banks instead of one rounded-pow2 variable**

For `TOTAL_IN_PIXELS = 49` (conv_292): one 64-deep variable holds the
whole buffer at 64*16*256 = 262 Kb — under the cap, and the depth-64
matches a LUT-RAM granule. No banking required.

For `TOTAL_IN_PIXELS = 196` (conv_284): rounding to 256 gives one
1,048,576-bit variable that Vivado HARD-rejects. Banking into 4×64
gives 4 variables of 262 Kb each — each maps into LUT-RAM cleanly,
together they cover 256 logical entries (60 unused), and none
trips the variable-size cap.

For larger layers (e.g. `TOTAL_IN_PIXELS = 784`): 13 banks of 64.
Generic addressing scales without per-layer hardcoding.

**Do NOT** declare a single monolithic `line_buf` and rely on Vivado to
shard it; the variable-size cap applies to the source RTL declaration,
not the post-synth physical mapping. The banking must be in the Verilog
the agent emits.
