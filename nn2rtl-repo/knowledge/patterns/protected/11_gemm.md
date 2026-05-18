# 11 — Gemm (Fully-Connected / Linear)

## When to use

`op_type == "gemm"`. LayerIR carries:

- `weight_shape = [M, K]` — row-major, M output features, K input features.
  Per-output weights are contiguous in memory: weight row `m` starts at
  offset `m * K` in the hex file.
- `gemm_in_features = K`, `gemm_out_features = M`.
- `bias_path` — INT32 bias per output (or absent if no bias).
- `scale_factor` — composite `(input_scale · weight_scale / output_scale)`
  identical to the conv requantize multiplier.

## Semantics

For each output feature `m ∈ [0, M)`:

```
acc[m]   = Σ_{k=0..K-1} in[k] · weight[m * K + k]         # INT32 accumulator
biased   = acc[m] + bias[m]                               # INT32
scaled   = round_half_up( biased · SCALE_MULT >> SCALE_SHIFT )
out[m]   = clamp(scaled, -128, 127)
```

`in` is the layer's input vector (INT8, length K). For MobileNetV2 the
input arrives as `[N, K]` after the upstream GlobalAveragePool/Flatten —
on the bus it's one beat of K bytes (the channel-bus packing in
`_shape_as_nchw` flattens [N, K] to [N, K, 1, 1]).

The requantize tail is the same shape as conv: BIAS → SCALE → CLAMP →
OUTPUT. Re-use the conv tail style and the round-half-toward-+inf
primitive; goldens depend on this rounding.

## Latency contract

```
fill_cycles   = 1                                # input latch
mac_cycles    = ceil(K / mac_parallelism)        # serialised dot product per output
output_cycles = M                                 # one output per cycle in steady state
tail_cycles   = 3                                # BIAS + SCALE + CLAMP
latency       = fill_cycles + mac_cycles + output_cycles + tail_cycles
```

For MobileNetV2's classifier (K=1280, M=1000, MP=4):
- mac_cycles = 320 per output, M=1000 outputs → 320 + 1000 = 1320 in steady
  state if you can overlap the MAC of output `m+1` with the requantize of
  output `m`.
- Without overlap: `M · (mac_cycles + tail_cycles)` ≈ 1000·323 = 323k cycles.

Read `pipeline_latency_cycles` from the LayerIR. Don't redo the math —
Python is the source of truth.

## Architecture notes (no reference module provided)

Foundry must produce this layer from the pattern alone. A few hints
without prescribing a specific datapath:

- The weight matrix is large (M·K = 1.28 M entries for MobileNetV2's
  classifier — ≈10 Mbit). It does NOT fit in a single BRAM36 (~36 Kbit
  usable). You will need either (a) banked BRAM weights addressed by an
  outer M-index and inner K-index, or (b) the `dram-backed-weights`
  contract if M·K exceeds the per-layer BRAM budget. Pick whichever the
  preflight gates and the LayerIR's contract selection imply.
- There is no spatial reuse — every output sees every input. The MAC loop
  is K-deep per output; you cannot amortise an input across outputs the
  way conv does. This is qualitatively different from the conv datapath.
- `mac_parallelism = MP` (typically 4) splits each per-output K-deep MAC
  into MP parallel lanes; each lane handles K/MP weights. This is the
  same lane-parallelism pattern conv uses, just along the K axis instead
  of the OC axis.
- Bias is optional. When `bias_path` is null, the BIAS stage is a
  pass-through (no add).
- The output bus is `M · 8 = M-byte` wide. For MobileNetV2's 1000-class
  output that's 8000 bits — large but not gargantuan. Channel-tile if the
  contract demands.

## Required public interface

Same canonical interface as every other op. The LayerIR's
`input_width_bits` (= K·8) and `output_width_bits` (= M·8) drive the bus
widths. Do not widen, narrow, or reorder the public signals.

## Known failure modes (anticipated — there is no proven reference yet)

- `weight_addressing_wrong` — using `weight[k * M + m]` instead of
  `weight[m * K + k]`. The hex file is row-major in `[M, K]`. Confirm
  with a 2×2 toy first if you're unsure.
- `signedness_dropped` — multiplying as unsigned. INT8 weights and inputs
  must be `$signed(...)` in every multiply.
- `accumulator_overflow` — INT8·INT8 → INT16, summed K times → up to
  INT16 + log2(K) bits. For K=1280, ⌈log2(1280)⌉ = 11; INT16 + 11 = INT27.
  Use INT32 to be safe across all networks.
- `requantize_rounds_toward_zero` — same round-half-up issue conv hits.
- `pipelining_breaks_timing_contract` — overlapping the requantize of
  output `m` with the MAC of output `m+1` is allowed only if the first
  `valid_out` still lands exactly on `pipeline_latency_cycles`. The
  deterministic assayer rejects timing drift.
