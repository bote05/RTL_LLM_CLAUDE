# 13 — On-chip-weights contract

This is contract guidance, not a passing RTL reference. Use it when
`contract_id == "on-chip-weights"`.

## When to choose this contract

Pick `on-chip-weights` when ALL of the following hold:

- The deployment target has a sizeable UltraRAM region (UltraScale+ devices:
  ZCU102 ≈ 25 Mbit URAM, U250 ≈ 90 Mbit URAM) and the orchestrator's URAM
  planner has reserved space for the layer's weight tensor.
- The layer's weight tensor would overflow the flat-bus on-chip weight
  budget (`max_on_chip_weight_bytes` on `flat-bus`), so it cannot use a
  per-module weight ROM, BUT the FPGA itself has enough URAM that the
  whole layer can live there.
- The deployment plan has elected a *shared* compute engine that runs
  several heavy layers sequentially (Phase 2 of the U250 plan, see
  `docs/agent_tasks/00_engine_skeleton_spec.md`).

`on-chip-weights` is the on-chip analogue of `dram-backed-weights`: same
streaming activation interface, same conv2d-only op set, same
exact-latency rule. The difference is purely on the weight side — there
is **no** AXI4-MM master, **no** DDR controller, and **no** per-pass
prefetch FSM. Reads complete in a fixed two-cycle URAM pipeline.

If the layer's weights fit comfortably on chip but the target has no URAM
(7-series, or anything below UltraScale+), stay on `flat-bus`. If the
weights do **not** fit in the device's URAM region but DDR is wired up,
use `dram-backed-weights` instead.

## FORBIDDEN PATTERNS — read first

The structural preflight gate fails the build immediately if it sees any
of these. Do not write them, do not "stub them and revisit", do not
justify them in a comment.

1. Any AXI4-MM read-channel port on the module (`weights_arvalid`,
   `weights_arready`, `weights_araddr`, `weights_rvalid`, `weights_rdata`
   …). This contract has no external memory interface. If the layer
   needs DRAM, you picked the wrong contract — use
   `dram-backed-weights`.
2. `$readmemh(... weights ...)` of the full weight tensor *inside* the
   module body. The URAM region is initialised at bitfile load by the
   top-level wrapper, not by an in-module `$readmemh`. The DUT must
   reach weights only through `weight_rd_en` + `weight_rd_addr` +
   `weight_rd_data`.
3. `reg [W_W-1:0] weights_mem [0:OC*K_TOTAL-1];` (or any storage sized
   `OC*K_TOTAL`) declared inside the module. Weight storage belongs in
   the shared URAM region, not duplicated in the layer body.
4. Negative rounding-bias constants such as `ROUND_BIAS_NEG = -...` or
   ternaries such as `scaled[MSB] ? -SCALE_ROUND_HALF : SCALE_ROUND_HALF`.
   Verilog `>>>` already floors negative values toward -inf; the
   negative branch must add `(HALF - 1)`, not subtract `HALF`. Same
   rule as every other conv contract.

## Required public interface

The top level includes the seven base activation-stream ports from
`01_context.md` plus the URAM read-port ports declared in
`contracts/on-chip-weights/metadata.json`:

- `weight_rd_addr`  — output, `weight_addr_bits` wide
- `weight_rd_en`    — output, 1 bit
- `weight_rd_data`  — input, `uram_word_bits` wide (288 bits on UltraScale+)
- `weight_base_word` — input, `weight_addr_bits` wide, latched once per layer

Do not collapse this contract back to the seven-port flat-bus interface.
Do not add AXI ports.

## URAM word geometry — CRITICAL

UltraScale+ UltraRAM (URAM288) stores **288-bit words**. With INT8
weights, that is **36 weights per URAM word**, not 32 and not a power of
two. Common bugs:

- Treating the URAM word as 256 bits (32 weights/word). This is a
  BRAM18 layout; URAM has a different aspect ratio and the address
  arithmetic for "which URAM word holds weight index `i`" does NOT
  collapse to a power-of-two shift.
- Using a byte address on `weight_rd_addr`. The interface is **word**
  addressed: `weight_rd_addr` units are URAM words, not bytes. The
  scheduler's `weight_base_word` is also in URAM words. If you compute
  `byte_offset = oc_pass * PASS_BYTES` you must then convert with
  `word_offset = byte_offset / 36` (handling the remainder by reading
  one extra word and masking).

Recommended addressing pattern:

```
WEIGHTS_PER_URAM_WORD = uram_word_bits / 8       // 36 for URAM288
weight_idx            = oc * K_TOTAL + ic * KH * KW + kh * KW + kw
uram_word_idx         = weight_base_word + (weight_idx / WEIGHTS_PER_URAM_WORD)
byte_in_word          = weight_idx % WEIGHTS_PER_URAM_WORD
weight_byte           = weight_rd_data[byte_in_word * 8 +: 8]
```

The divider is a constant — synthesise as a divide-by-36 (4-cycle
multiply-by-reciprocal) or, more cleanly, ensure the orchestrator pads
each layer's weight region up to a multiple of 36 weights so the address
generator can iterate one URAM word at a time without remainder logic.

## URAM read latency — 2 cycles, pipeline it

URAM288 has a fixed two-cycle read latency (address-register stage +
memory-output-register stage). The `latency.ts` contract adds these
two cycles on top of the flat-bus conv latency:

```
pipeline_latency = flat_bus_conv_latency + URAM_READ_LATENCY_CYCLES   // 2
```

To hit that, fire `weight_rd_en` and `weight_rd_addr` **two cycles ahead**
of the MAC stage that consumes the read data. The MAC pipeline naturally
provides two stages of slack (address-prep stage + window-shift stage),
so the URAM read fits without extending the overall pipeline depth.

If you serialise weight reads behind the MAC (read → wait 2 → MAC), you
overshoot the contract's expected latency by `2 * K_TOTAL` cycles per
output pixel; the deterministic Assayer rejects the result with a
`first_valid_out` measurement off by a large multiple of two.

## weight_base_word loading discipline

`weight_base_word` is an input from the scheduler. It must be:

1. **Stable** from at least one cycle before `engine_start` through the
   layer's final `valid_out` beat. The scheduler enforces this externally,
   so the module does NOT need to register it on rising `engine_start`,
   but DOES need to read it through the entire layer.
2. **Latched on the first** `weight_rd_en` of the layer — re-sampling it
   mid-layer is wrong because, for engines that overlap layer N's tail
   with layer N+1's config write, the scheduler may have already
   overwritten the input. A single internal register loaded from the
   input on first use is the safe pattern.
3. **Independent of byte/word confusion**. The CSR block exposes
   `cfg_weight_base_word[19:0]` in URAM-word units (see task 10's
   register map). Treating it as a byte address shifts the read window
   by a factor of 36 and produces an "all-bias outputs" failure
   signature (weights read from the bias region or from uninitialised
   URAM).

## Multi-layer dispatch — no terminal lock

The shared engine is dispatched on layer N, runs to completion, then is
re-dispatched on layer N+1 against a *different* `weight_base_word`. The
FSM rule from `09_dram_backed_weights.md` ("Multi-vector test sequencing")
applies here as well, but at the *layer* boundary rather than the
*vector* boundary:

- Do NOT use a terminal `ST_DONE` that holds `valid_out=0; ready_in=0`
  forever. The scheduler writes a new config block, re-pulses
  `engine_start`, and the engine must return to `ST_INIT_BOOT` cleanly.
- All per-layer state (weight read counters, accumulators, output buffer,
  MAC pipeline) must reset on `engine_start` even when `!rst_n` is not
  asserted between layers. The bitfile-loaded URAM contents do NOT
  reset — only the in-engine control state does.

The structural preflight gate `on_chip_weights_terminal_done_lock` will
reject any FSM whose `ST_DONE` does not transition out on a fresh
`engine_start` rising edge.

## Common bugs (anticipated — there is no proven reference yet)

- `axi_master_present` — accidentally emitting `weights_arvalid` /
  `weights_araddr` / `weights_rvalid` because the layer body was forked
  from a `dram-backed-weights` reference. Strip the AXI FSM, replace
  with a single `weight_rd_en` / `weight_rd_addr` pulse pair fired two
  cycles ahead of the MAC consumer.
- `uram_word_width_mismatch` — assuming 256-bit URAM words and packing
  32 weights/word. Always size the layer body off the contract's
  `uram_word_bits` (288 on UltraScale+); never hard-code 256.
- `weight_base_word_units_wrong` — using `weight_base_word` as a byte
  offset instead of a URAM-word offset. Failure signature: outputs
  match a *deterministic shifted* version of the golden (the read
  window is offset by a factor of 36), not random garbage.
- `read_fence_missing` — letting the address generator step past
  `weight_base_word + ceil(WSIZE_bytes / 36)` and pull URAM contents
  that belong to layer N+1's weights. Defensive fence: gate
  `weight_rd_en` with `(addr_offset < ceil(WSIZE_bytes / 36))`.
- `early_valid_in_crash` — receiving a `valid_in` beat before the
  scheduler has loaded `weight_base_word`. The scheduler enforces the
  ordering externally, but the module should not deadlock or emit
  X-propagated data if this is violated; back-pressure with
  `ready_in = 0` until `engine_start` has been observed.
- `uram_read_serialised_behind_mac` — issuing the URAM read inline with
  the MAC stage instead of two cycles ahead. Costs `2 * K_TOTAL` extra
  cycles per output pixel; the Assayer reports a clean
  `first_valid_out_late` with a delta matching the K_TOTAL geometry.
- `bitfile_init_in_module` — using `$readmemh` inside the module to
  initialise an internal `weights_mem` array. The structural preflight
  gate rejects this; URAM is initialised by the top-level wrapper from
  a `.mem` image, not by the layer body.
- `rounding_negative_bias_subtracts` — the same negative-half-round
  trap as every other conv contract (see `01_context.md`
  "Scale-shift rounding — MANDATORY"). The negative branch ADDS
  `(SCALE_ROUND_HALF - 1)`; it does not SUBTRACT `SCALE_ROUND_HALF`.

## Successful references

(None yet — this contract is the spec for Phase 2's shared engine. The
auto-promote machinery will populate this section as the engine
sub-blocks land and their integrated layer body passes verification.)
