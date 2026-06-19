# Task 13 — Engine two-mismatch root cause (node_conv_246)

## Symptom

`tb/engine_one_layer_tb.v` ran node_conv_246 through the shared engine
and produced 99.996% byte-exact output vs `output/goldens/node_conv_246.goldout`
vector 0 — but with exactly 2 mismatches, both off by -1, both at pixel
[1, 4]:

* `byte[4732]` = pixel [1,4] channel 124: golden = +1, engine = 0
* `byte[4846]` = pixel [1,4] channel 238: golden =  0, engine = -1

A prior attempt to fix this in `requant_pipeline.v` (replacing the
sign-aware rounding bias with an unconditional +HALF) produced the
identical 2 mismatches, ruling out rounding-tie semantics.

## Root cause

`output/rtl/engine/address_generator.v` was suppressing the LAST
legitimate weight + activation read of every output-pixel OC pass.

The walk emits one URAM weight read per cycle while the FSM is in
`ST_RUN`. With `K_TOTAL = IC * KH * KW = 2304` for this layer, the
inner-loop counter `k_cnt` runs 0..2303. The legit reads happen at
cycle `T+1` for each `weight_rd_addr` registered during the walk at
cycle `T`. The address for the very last MAC tuple
`(ic=255, kh=2, kw=2)` is computed at the walk cycle where `k_cnt==2303`
(i.e. `k_at_last==1`), so it is presented to URAM the cycle AFTER —
which is exactly the cycle the prior "Fix B" was suppressing with:

```verilog
weight_rd_en <= ~k_at_last;
act_in_rd_en <= in_bounds & ~k_at_last;
```

Trace (let `T_last` be the cycle where `k_cnt==2303`):

| cycle     | event                                                   |
|-----------|---------------------------------------------------------|
| T_last    | walk sets `weight_rd_addr <= BASE + 2303` (last weight) and `weight_rd_en <= ~k_at_last = 0`. mac_done is asserted next-cycle. |
| T_last+1  | weight_rd_addr = BASE+2303, but weight_rd_en = 0. URAM read SUPPRESSED. mac_done now visible. ic_cnt, kh_cnt, kw_cnt, k_cnt have wrapped to 0. |
| T_last+2  | FSM transitions ST_RUN → ST_REQUANT; `else` branch in ag forces weight_rd_en <= 0. ag_weight_rd_en_d (in shared_engine) goes high for this cycle (delayed from T_last+1 when weight_rd_en should have been 1). Since the read at T_last+1 was suppressed, the URAM data on this cycle is still the previous mem[BASE+2294], NOT mem[BASE+2303]. |

The mac_array therefore accumulated only 2303 of the 2304 products.
The dropped product was `act[in_r=3, in_c=9, ic=255] * weight[OC, ic=255, kh=2, kw=2]`
for every OC channel of every pixel. For 49,994 of the 50,176 output
bytes the dropped contribution did not change the requantized INT8
result, because the accumulator landed comfortably inside a requant
bucket. For the 2 mismatching bytes the accumulator landed exactly on
the requant rounding boundary:

* pixel[1,4] ch124: gold acc = 214, dropped MAC = +4 → engine acc = 210.
  biased = 213 vs 217. requant(213) = 0, requant(217) = 1. Off by -1.
* pixel[1,4] ch238: gold acc = -211, dropped MAC = +12 → engine acc = -223.
  biased = -220 vs -208. requant(-220) = -1, requant(-208) = 0. Off by -1.

Python diagnostic at `scripts/diag_engine_pixel.py` confirms these
intermediates. Also: scanning all 50,176 (pixel, OC) cells, only the
two reported mismatches are predicted by the "drop last MAC" model —
matching the testbench's observation exactly.

## Fix

Replace the `~k_at_last` gating with `~mac_done` on both
`weight_rd_en` and `act_in_rd_en` in the per-cycle walk. `mac_done` is
the REGISTERED `pulse` that fires exactly one cycle AFTER `k_at_last`,
so:

* On the cycle where `k_at_last` fires (the cycle the LAST legit
  address is being registered), `mac_done` is still 0 — the read
  enable goes high next cycle, allowing the URAM read of BASE+K_TOTAL-1
  to happen.
* On the cycle AFTER `k_at_last` (when `mac_done` is high and the
  counters have wrapped), `~mac_done` forces the next-cycle read enable
  to 0, suppressing the stray read that the original "Fix B" was
  guarding against.

```verilog
// output/rtl/engine/address_generator.v, ~line 271
weight_rd_en        <= ~mac_done;
act_in_rd_en        <= in_bounds & ~mac_done;
```

This restores the original (pre-Fix-B) read pattern for cycles
1..K_TOTAL+1, AND still suppresses the stray read at K_TOTAL+2 that
would have produced a spurious 2305th MAC.

## Verification

* `bash scripts/run_engine_one_layer_tb.sh` (compile + 13-minute sim + compare)
  exits 0 with:
  ```
  PASS: 50176 bytes match (n_samples=196, bytes_per_sample=256, max_error=0, mismatch_count=0)
  ```
* Engine total runtime for the layer is 453,157 cycles — slightly
  different from the prior 453,156 because the legit last MAC now
  happens, but well within the timeout budget.
* The diagnostic script `scripts/diag_engine_pixel.py` (golden-side
  intermediates for ch 124 and ch 238 at pixel [1,4]) is left on disk
  for future debug sessions.

## Files touched

* `output/rtl/engine/address_generator.v` — the gating change above.
* `scripts/diag_engine_pixel.py` — new diagnostic helper (golden
  acc/bias/scaled/clamped per channel; also enumerates predicted
  mismatches under the "drop last MAC" hypothesis).
