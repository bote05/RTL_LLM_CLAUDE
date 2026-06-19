# Engine Debug — LARGE cluster (conv_286, 8578 mismatches)

## TL;DR

- **Root cause**: in `output/rtl/engine/address_generator.v`, the per-cycle
  `if (run_active)` walk's `else` branch (`ic_cnt <= ic_cnt + 1; k_cnt <=
  k_cnt + 1;`) fires on the cycle AFTER `k_at_last`. At that cycle the FSM
  is still in ST_RUN (it transitions to ST_REQUANT one cycle later when
  it samples the registered `ag_mac_done`), so `run_active` is still 1
  AND `mac_done` is now registered = 1. The spurious bump sets
  `ic_cnt` from the just-reset 0 to 1 (and `k_cnt` likewise). Through
  ST_REQUANT and ST_DRAIN nothing touches these counters, so they enter
  the next OC pass's first ST_RUN cycle at `ic_cnt = 1`. At that
  rising edge the rising-edge block tries to reset them to 0, but the
  per-cycle else block ALSO fires on the same NBA edge and (per
  "last NBA in source order wins" semantics) overrides the reset with
  `ic_cnt <= ic_cnt + 1`. As a result, **the activation byte index
  pipeline starts at 1 instead of 0**, so the very first MAC of EVERY OC
  pass uses `a[ic=1]` instead of `a[ic=0]`. The ic=0 contribution is
  never accumulated.
- **Fix**: wrap the advance `if/else if/else` chain in
  `if (!mac_done) begin ... end` so that the cycle after `k_at_last`
  becomes a no-op for the counters. See
  `output/rtl/engine/address_generator.v` lines 300-342 (the
  `if (!mac_done) begin` wrapper is at line 315).
- **conv_286 (this task's layer) sweep result**: PASS, 0 mismatches,
  max_error 0.
- **Cluster fix coverage**: the same bug was responsible for failures
  across the entire failing set of 9 layers. After the fix, ALL 14 heavy
  dispatches pass byte-exact.

## Why the bug only showed up on SOME layers

The bug "skip ic=0 in every dot product" is mathematically equivalent to
subtracting `a[ic=0] * w[lane, 0]` from every output accumulator. When
the upstream layer's input activation byte 0 is zero at every pixel, the
subtraction is zero and the layer accidentally passes. We verified this
maps the observed `a[ic=0]` histograms to the mismatch counts:

| layer | pixels with a[ic=0]≠0 | mismatches (pre-fix) | mismatches (post-fix) |
|------:|----------------------:|---------------------:|----------------------:|
| node_conv_246 | 1/784       | 0 (rounding hides single hit) | 0 |
| node_conv_250 | 0/784       | 0                              | 0 |
| node_conv_254 | 0/196       | 0                              | 0 |
| node_conv_260 | 29/196      | 27                             | **0** |
| node_conv_264 | (1024-IC)   | 663                            | **0** |
| node_conv_266 | 2/196       | 2                              | **0** |
| node_conv_272 | many/196    | 62                             | **0** |
| node_conv_278 | many/196    | 95                             | **0** |
| node_conv_282 | 52/196      | 2859                           | **0** |
| node_conv_286 | 41/49       | 8578                           | **0** |
| node_conv_290 | 8/49        | 184                            | **0** |
| node_conv_294 | 0/49        | 0                              | 0 |
| node_conv_296 | (mixed)/49  | 413                            | **0** |
| node_conv_300 | 0/49        | 0                              | 0 |

## Diagnostic methodology

1. **Mismatch pattern grid** — counted mismatches per output pixel in a
   7×7 grid for conv_286. Pixels with zero mismatches matched EXACTLY the
   pixels where the input activation byte 0 is zero. This pointed at the
   ic=0 contribution being dropped uniformly across all OC channels.

2. **Engine vs golden accumulator comparison** — instrumented the TB with
   a `DEBUG_PIXEL` ifdef (R/C/oc_pass plusargs) that dumps the
   mac_array's `acc[N]` register at `mac_done` and the surrounding state
   (ic_cnt, k_cnt, weight_rd_addr, ic_byte_idx_d, mac_valid_in,
   mac_valid_q1, mac_clear) every ST_RUN cycle of the tracked region.
   This showed the engine's running accumulator was wrong by exactly
   `a[ic=0] * w[lane, 0]` for several channels at pixel [0,6].

3. **Single-IC-step elimination** — wrote a Python helper
   (`scripts/diag_engine_pixel_286.py`) that for each candidate dropped
   `ic` index, predicts the per-lane diff `-a[ic] * w[lane, ic]` and
   compares to the observed `engine_acc - golden_acc` diffs across
   {ch 10, 13, 42, 49, 64}. Only `ic = 0` produced a consistent match.
   That's the signature of "first MAC of the pass is missing".

4. **Cycle-by-cycle TB trace** — printed `ic_cnt`, `k_cnt`, `wraddr`,
   `wren`, `byteidx`, `byteidx_d`, `mac_valid_in`, `mac_valid_q1` for
   each ST_RUN cycle of pixel 6 pass 0. The very first ST_RUN cycle
   (where `mclr=1`) showed `ic_cnt=1, k_cnt=1, wraddr=base+pass_offset`
   — stale values inherited from the previous pass's k_at_last cycle.
   This localised the bug to the in-between cycle after `k_at_last` where
   the FSM hasn't transitioned to ST_REQUANT yet but `mac_done=1`.

## The fix

`output/rtl/engine/address_generator.v` lines 300-342, wrap the counter
advance in `if (!mac_done)`:

```verilog
if (!mac_done) begin
    if (k_at_last) begin
        mac_done <= 1'b1;
        ic_cnt   <= 12'd0;
        kw_cnt   <= 3'd0;
        kh_cnt   <= 3'd0;
        k_cnt    <= 16'd0;
        ...
    end else if (ic_cnt == (cfg_ic - 12'd1)) begin
        ic_cnt <= 12'd0;
        ...
        k_cnt <= k_cnt + 16'd1;
    end else begin
        ic_cnt <= ic_cnt + 12'd1;
        k_cnt  <= k_cnt + 16'd1;
    end
end
```

This blocks the spurious increment at the one cycle between `k_at_last`
and the FSM's transition to ST_REQUANT, so the next OC pass starts
cleanly at `ic_cnt = 0`. The address/enable updates above the wrapped
block (weight_rd_addr / weight_rd_en / act_in_rd_addr / act_in_rd_en /
byteidx / k_index / act_out_wr_addr) remain ungated — they're already
gated by `~mac_done` where needed (weight_rd_en, act_in_rd_en) and the
remaining ungated address-update at the in-between cycle is overwritten
at the next ST_RUN rising edge before it's ever read.

## Sweep verdict

After applying the fix and re-running
`scripts/engine_sweep_driver.py --workers 4 --timeout-cycles 50000000`:

```
node_conv_246  PASS  mism=0
node_conv_250  PASS  mism=0
node_conv_254  PASS  mism=0
node_conv_260  PASS  mism=0
node_conv_264  PASS  mism=0
node_conv_266  PASS  mism=0
node_conv_272  PASS  mism=0
node_conv_278  PASS  mism=0
node_conv_282  PASS  mism=0
node_conv_286  PASS  mism=0   <- this task's assigned layer
node_conv_290  PASS  mism=0
node_conv_294  PASS  mism=0
node_conv_296  PASS  mism=0
node_conv_300  PASS  mism=0
```

14/14 PASS, byte-exact, max_error 0 on every layer. No regression on
the previously-passing layers (246/250/254/294/300), and the 9
previously-failing layers (260/264/266/272/278/282/286/290/296) all now
pass.

## Files modified

- `output/rtl/engine/address_generator.v` — added the `if (!mac_done)`
  wrapper around the inner-loop counter advance block (and a comment
  block explaining why).
- `tb/engine_one_layer_tb.v` — added a `DEBUG_PIXEL` ifdef block that
  prints per-cycle DUT state when `pixel_h_r/pixel_w_r/oc_pass_idx_r`
  match `+DEBUG_R / +DEBUG_C / +DEBUG_P` plusargs. The block is opt-in
  via `-DDEBUG_PIXEL` at compile; the default sweep build does NOT
  define it so the production comparison flow is unchanged.
- `scripts/diag_engine_pixel_286.py` — new diagnostic helper that
  computes the golden accumulator for conv_286 at a given pixel and
  identifies which `ic` step is missing.
