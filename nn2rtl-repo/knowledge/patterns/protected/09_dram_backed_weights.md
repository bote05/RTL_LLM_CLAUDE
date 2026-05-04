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
