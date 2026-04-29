# 08 — Common bugs catalog

Every entry below was observed in a real pipeline run. Symptom / diagnosis /
fix, using the raw `VerifResult` evidence fields. The structural preflight
rules in `sdk/orchestrate.ts::structuralPreflightViolations` catch the
simplest form of some of these before simulation even runs.

## single-pixel-MAC on a spatial conv (KH*KW > 1)

- **Symptom**: `first_mismatch_index` small, `mean_error` large across the
  full stream, and the output pattern is mathematically the sum-reduced
  1×1 approximation of the real 2D convolution (i.e.
  `output[oc,h,w] = sum_ic in[ic,h,w] * sum_{kh,kw} w[oc,ic,kh,kw]`).
- **Diagnosis**: the MAC reads only the current pixel (or a 1-D latch
  indexed `in_latch[k_counter % IC]` / `in_latch[k / (KH*KW)]`) instead
  of a true `KH × KW × IC` receptive field from the line buffer.
- **Fix**: for any `KH*KW > 1` conv, the MAC must read
  `window[kh][kw][ic]` with the index decomposition
  `ic = k / (KH*KW)`, `kh = (k % (KH*KW)) / KW`, `kw = k % KW`. Window
  must be a registered array (not a combinational rebuild) populated by
  the line-buffer / shift discipline described in `03_conv3x3_pad1.md`.
  Wrong and right forms:

  ```verilog
  // WRONG — spatially-summed 1×1 approximation for a KH*KW > 1 conv:
  acc[oc] <= acc[oc] + weights[oc*K_TOTAL + k_counter] * in_latch[k_counter % IC];
  acc[oc] <= acc[oc] + weights[oc*K_TOTAL + k_counter] * in_latch[k_counter / (KH*KW)];

  // CORRECT — true 2D MAC against the registered window:
  acc[oc] <= acc[oc] + weights[oc*K_TOTAL + k_counter] *
             window[ (k_counter % (KH*KW)) / KW ]  // kh
                   [ k_counter % KW ]              // kw
                   [ k_counter / (KH*KW) ];        // ic
  ```

  Pointwise (KH=KW=1) keeps the simpler `in_latch[k_counter]` MAC —
  the spatial index collapses.

## drain-exit bug (spatial conv)

- **Symptom**: `output_gap_histogram = [0, 0, 0, N]` (all missing samples
  at tail). `outputs_received < OH*OW`. `status_class: sim_stalled`.
- **Diagnosis**: The FSM has a state comparing `in_row > IH-1+PH` to exit,
  instead of terminating on `outputs_emitted == OH*OW`. The comparison
  fires one iteration too early and swallows the last row's outputs.
- **Fix**: remove the drain-row comparison entirely; gate termination on
  `outputs_emitted == OH*OW`. The handwritten `coord_scheduler` module
  implements this correctly; prefer it over rolling your own.

## right-edge padding off-by-one (fmm=7122 on layer0_0_conv1) — RESOLVED

- **Status**: Resolved by the split-architecture rework. Spatial conv
  top-levels no longer roll their own wrap math; `coord_scheduler.v`
  owns the `IW-1+PW` constant, and `line_buf_window.v`'s right-pad
  reads are sourced from BRAM cells that are never written (so they
  return zero by construction). `layer0_0_conv1` now passes first-shot
  (`max_error=0`) via `knowledge/references/protected/conv7x7_passing_reference.v`.
- **Historical record**: The bug repeatedly produced
  `first_mismatch_index` in the `OH*OW/2` range with `max_error` 1–3,
  histogram concentrated in the last quarter. Diagnosis was the
  `in_col` wrap constant off by one. The repair was structural, not
  literal: stop hand-rolling the FSM. If a future regression brings
  this symptom back on a spatial layer, do not patch a constant; check
  whether the generated module is bypassing the library.

## MAC window indexing swap

- **Symptom**: `first_mismatch_index` small, periodic pattern, `max_error`
  large (saturated-range).
- **Diagnosis**: Accessing the window as `window[ic][kh][kw]` when it was
  declared as `window[kh][kw][ic]` (or vice versa). Channels and kernel
  taps get permuted.
- **Fix**: match the window declaration order exactly. Convention:
  `window [0:KH-1][0:KW-1][0:IC-1]`, read as `window[kh][kw][ic]`.

## weights_packed BRAM/ROM inference rejection

- **Symptom**: Vivado synthesis fails or infers a huge LUT mux around the
  weight memory. `status: fail, failure_class: synthesis_failed`.
- **Diagnosis**: Someone — usually Surgeon responding to a synth timeout —
  introduced a `weights_packed` memory that packs multiple weights into a
  wide word via `for ... weights_packed[i] = {weights[i*4+3], ...}`.
  Vivado cannot infer a clean ROM/BRAM from this dynamic initializer.
- **Fix**: use `$readmemh`-initialized ROMs and read them through registered
  addresses. The current verified conv contract serializes the lanes through
  one read port. `LayerIR.weight_bank_paths` provides one bank file per lane
  for the future banked datapath, but do not switch to MP parallel reads
  unless the LayerIR latency contract was generated for that mode. The
  `weights_packed_forbidden` structural preflight rule catches the bad packed
  forms.

## non-constant $readmemh initialization (missing initial block)

- **Symptom**: Vivado synth fails with "non-constant memory initializer" or
  Verilator fails with "syntax error near weights[...]".
- **Diagnosis**: `$readmemh` call is outside an `initial` block, or the
  weights are being assigned through continuous `assign`.
- **Fix**: wrap every $readmemh in a single `initial begin ... end`
  block. The `readmemh_missing` preflight rule catches the absence.

## Verilator hangs on partial outputs (no hang_budget trigger)

- **Symptom**: Verilator simulation hits `VERILATOR_SIM_TIMEOUT_MS` (the
  10-minute cap added in CX-1). `status_class: sim_stalled,
  failure_class: verilator_timeout`.
- **Diagnosis**: FSM produces `valid_out` pulses intermittently forever
  because the output counter is missing or its bound is wrong. The
  testbench's `hang_budget` only fires on total silence, so any pulsing
  keeps the sim alive.
- **Fix**: add / fix `outputs_emitted`; ensure the FSM returns to a
  terminal state when `outputs_emitted == OH * OW`. The
  `output_counter_missing` structural preflight rule catches the absence
  for spatial conv and maxpool. It intentionally does not fire for pointwise
  1x1 conv, ReLU, or add, where each accepted input corresponds to one
  output and a frame-level counter would break back-to-back frames.

## port direction mismatch on ready_in

- **Symptom**: `port_width_mismatch` returned by deterministic preflight
  before iverilog/verilator runs.
- **Diagnosis**: `ready_in` declared as `input` instead of `output`.
  Foundry sometimes treats it as an upstream-driven handshake signal when
  it is actually the module's backpressure output.
- **Fix**: `output reg ready_in` — always. The module drives it.

## sign_extension_error in bias add

- **Symptom**: large `max_error`, many outputs saturated to ±127.
- **Diagnosis**: `biased = acc + biases[oc]` where one operand is treated
  unsigned. `$signed(acc) + $signed(biases[oc])` is required; without
  both, a negative bias wraps at 2^N-1 and saturates.
- **Fix**: wrap the bias add in `$signed(...)` both sides; widen
  `biased_w` to `max(acc_w, bias_w) + 1` to absorb the sign bit.

## BRAM line_buf — bottom-pad row_valid leak

- **Symptom**: First several output ROWS of a frame are bit-exact; the
  last few rows are wrong with `max_error` small-to-moderate and
  `first_mismatch_index` deep in the stream. Most visible on layers
  with `PH > 1` (e.g. the 7×7 stem with PH=3).
- **Diagnosis**: `coord_scheduler` walks `in_row` past `IH-1` into the
  bottom-pad fringe; `row_wrap_this_cycle` fires on those rows even
  though no real writes landed in `line_buf_window`'s currently-writing
  slot. If `row_valid[current_write_slot] := 1` is asserted
  unconditionally at row_wrap, that slot's `row_valid` becomes 1 while
  it still holds STALE data from KH rows ago (slots cycle through KH
  values). Subsequent reads for output rows whose receptive field
  includes the bottom-pad position return that stale data instead of
  zero.
- **Fix**: gate the row_valid set on `!bottom_padded` —
  `row_valid[current_write_slot] <= !bottom_padded`. The slot becomes
  "valid history" only when a real input row has actually written into
  it.

## BRAM line_buf — q_reg phase corruption mid-MAC

- **Symptom**: Bit-exact timing (`timing_actual == timing_expected`)
  but values disagree from `first_mismatch_index = 0`. `max_error`
  large (saturation-range), `mean_error` modest, diffs scattered.
  Hand-written legacy line_buf passes the same layer cleanly. Tiny
  geometry tests don't reproduce because they drive `sched_in_col`
  directly without modeling the scheduler's "advance, then output_fires
  next cycle while coord already advanced" phase.
- **Diagnosis**: `coord_scheduler` emits `output_fires` the cycle
  AFTER it advances past the firing coord, with `sched_in_col`
  already pointing at the NEXT coord. If `line_buf_window`'s per-slot
  BRAM output register `q_reg` free-runs off live `sched_in_col`
  every clock, the rightmost window column changes mid-MAC. The
  `conv_datapath` pipeline reads `window_flat[...]` at every
  `k_counter` step, so a moving `q` corrupts every k≥1 contribution
  of the MAC accumulation — the first weight is correct, the
  remainder are off.
- **Fix**: gate the `q_reg <= mem[sched_in_col]` update on
  `sched_advance`. The scheduler holds `sched_advance = 0` from
  `output_fires` through the entire MAC pass (via
  `eff_stall = stall_in || output_fires`), so the BRAM output stays
  stable as long as the scheduler is stalled. Writes (`mem[..] <= ..`)
  are independently gated and don't fire during stall anyway, so this
  is read-side only.

## Surgeon regression by scale rounding

- **Symptom**: `first_mismatch_index` regresses from a high value to 0
  after a Surgeon attempt; `max_error` only slightly changed; timing
  remains exact. The orchestrator triggers `surgeon_regression_reverted`
  and rolls back.
- **Diagnosis**: Surgeon changed the rounding line from
  `(x + SCALE_ROUND_BIAS) >>> SCALE_SHIFT` to a bare `>>> SCALE_SHIFT`,
  which truncates instead of using the current RTL half-up/toward-positive
  tie approximation. Many pixels drift by ~1 LSB.
- **Fix**: restore the `+ SCALE_ROUND_BIAS` before the shift. Mark the
  rounding line `// [INVARIANT:ROUNDING]` to protect it from future
  regressions. This is not exact PyTorch tie-even rounding; do not chase a
  remaining +/-1 exact-half tie as a datapath bug unless the project changes
  the golden/RTL rounding contract together.
