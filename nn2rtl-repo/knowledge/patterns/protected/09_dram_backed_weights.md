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
