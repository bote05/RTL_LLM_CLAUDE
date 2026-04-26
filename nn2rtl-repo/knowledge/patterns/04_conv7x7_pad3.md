# 04 — Stem 7×7 conv, stride=2, padding=3

## When to use

`op_type == "conv2d"` with `weight_shape[2] == 7 && weight_shape[3] == 7`.
This is the ResNet-50 stem layer `layer0_0_conv1`. Stride is typically 2
and padding 3.

## Historical warning

As of the current implementation baseline, **this pattern has never closed
reliably via Foundry first-shot on a 7×7 stride-2 pad-3 conv**. The
`fmm=7122` right-edge padding bug (first_mismatch_index=7122) has survived
every Foundry/Surgeon attempt on `layer0_0_conv1`. See
`ARCHITECTURE.md § "Known Bottleneck"` for the full failure history.

Pay extra attention to:

1. The `ST_STREAM` wrap at `IW - 1 + PW = 113 + 3 = 116`, so the visible
   `in_col` range is `0..116` — not `0..111`.
2. The stride-2 row trigger: `in_row + PH - (KH - 1) >= 0 && % SH == 0`
   with SH=2. The first row that fires is `in_row = KH - 1 - PH = 3`;
   subsequent rows fire every 2.
3. The stride-2 col trigger: first firing col is `KW - 1 - PW = 3`;
   subsequent cols fire every 2.

## Latency contract

Same formula as 03_conv3x3_pad1.md (the spatial-conv formula is kernel-
agnostic):

```
fill_rows = max(KH - 1 - PH, 0) = max(6 - 3, 0) = 3
fill_cols = max(KW - PW,   1) = max(7 - 3, 1) = 4
latency   = 3 * (IW + PW) + 4 + OC_PASSES * pass_cycles
          = 3 * (224 + 3) + 4 + OC_PASSES * (MP * 7*IC*7 + 6)
```

For the current stem contract (IC=3, OC=64, IW=224, MP=4):
`OC_PASSES=16, K_TOTAL=147`, `pass_cycles = 4*147 + 6 = 594`,
`latency = 3*227 + 4 + 1 + 16*594 = 10190`. The `+6` per pass is the
3-stage MAC pipeline (weight ROM read, registered DSP multiply,
indexed accumulate) plus ST_BIAS + ST_SCALE + ST_OUTPUT, matching
`CONV_PIPELINE_STAGES = 6` in `scripts/golden_impl.py`. The extra
`+1` is the spatial-only coord_scheduler→`output_fires`→ST_IDLE→
ST_MAC transition latency that is absent in the pointwise path.

## Required FSM, registers, coordinate logic

Identical to `03_conv3x3_pad1.md` with larger window:

- `reg signed [7:0] line_buf [0:6][0:IW-1][0:IC-1];`  (7 rows)
- `reg signed [7:0] window   [0:6][0:6][0:IC-1];`

All other rules from `03_conv3x3_pad1.md` apply verbatim. The two rules
below are reproduced here in full so Foundry does not have to cross-open
`03_conv3x3_pad1.md` when generating a 7×7 layer:

- **No `ST_DRAIN` state.** Terminate on `coord_scheduler`'s
  `out_frame_done` (equivalently `outputs_emitted == OH*OW`). Never
  terminate on `in_row > IH-1+PH` — that's the drain-exit bug class.

- **Serialized weight reads (MANDATORY).** One read from the `weights`
  array per cycle, **not `MP`**. Each ST_RUNNING cycle selects a single
  lane via a `lane_counter` register that rotates `0 → 1 → ... → MP-1 → 0`;
  that lane performs one `weights[global_oc*K_TOTAL + k_counter]` read,
  one multiply, one accumulate into `acc[lane_counter]`. After `MP`
  cycles all lanes of the current `k_counter` step are done — then
  `k_counter` advances. Per output pixel: `MP * K_TOTAL * OC_PASSES`
  MAC cycles. For the current 7×7 stem contract (IC=3, OC=64, MP=4) this is
  `4 * 147 * 16 = 9408` MAC cycles per pixel. `MP` parallel reads from
  one flat async weight array become illegal BRAM port pressure and wide LUT
  mux trees in Vivado. The current verified contract removes that blocker
  by serialization; `weight_bank_paths` are emitted for the future banked
  datapath and must not be used to change latency unless LayerIR changes too.

- **Window-freeze during OC group iteration (MANDATORY).** While
  iterating the `OC_PASSES` groups for one output pixel, input capture
  MUST be frozen. The top-level `ready_in` stays low from the first MAC
  cycle of output pixel N until `valid_out` fires for pixel N. The line
  buffer, sliding window, and `cur_row` must hold the same receptive-field
  contents across all `OC_PASSES` passes. If later groups run against a
  shifted window, their accumulations corrupt later OC channels.

- **No `WEIGHT_ARRAY` invariant markers.** Weight memories may need Vivado
  BRAM banking or synchronous ROM rewrites; protect behavior with preflight
  rules and tests, not invariant comments.

## Use coord_scheduler — MANDATORY

Given this pattern's poor empirical convergence when Foundry rolls its
own coordinate logic, `rtl_library/coord_scheduler.v` is the **only**
supported way to emit the row/col/wrap/stride/drain FSM. Parameters for
the 7×7 stride-2 pad-3 stem are:

- `IH=IW=224, OH=OW=112, KH=KW=7, SH=SW=2, PH=PW=3`

The full instantiation template, region-handshake semantics, and the
combinational `stall_in` contract live in
`01_context.md § coord_scheduler contract` — follow it exactly. Your
MAC / window / data_out logic consumes the scheduler's `output_fires`,
`in_row`, `in_col`, `needs_real_input`, `ready_in`, `outputs_emitted`,
and `out_frame_done`. Do not re-derive the wrap/stride/padding math.

## Reference status

A proven-passing 7×7 reference now exists at
`knowledge/references/conv7x7_passing_reference.v`. It is a direct
adaptation of the 3×3 reference -- same split-architecture skeleton
(`coord_scheduler` + `line_buf_window` + `conv_datapath` instantiation,
`pending_rearm`+`mac_busy` re-arm gate), only the localparam block
and the asymmetric bus widths differ. The stride / padding / wrap /
right-edge-pad logic that historically failed under hand-rolled
attempts (`fmm=7122` class) lives entirely inside `coord_scheduler.v`
and `line_buf_window.v`, so Foundry's job is structural wiring only.

## Known failure modes

All spatial-conv bugs from `08_common_bugs.md`, plus a 7×7-specific bug
that has never been resolved:

- **`fmm=7122`** (the right-edge padding off-by-one). Symptoms:
  first_mismatch_index is very deep in the stream (just past OH*OW/2
  for a 112x112 output), `max_error` is small but non-zero, and the
  histogram shows errors concentrated in the last quarter. Diagnosis:
  the wrap point on the `in_col` register is off by one, so the right
  padding produces one stale column per row. Fix requires re-deriving
  `IW-1+PW` (not `IW+PW-1`, not `IW-1`, not `IW+PW`).
