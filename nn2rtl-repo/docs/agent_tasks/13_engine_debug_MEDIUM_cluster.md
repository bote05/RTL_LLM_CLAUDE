# Task 13 — Engine debug: MEDIUM cluster (conv_290, 184 mismatches)

## Status: FIXED — 14/14 dispatches PASS post-fix

## Root cause

The address_generator's per-cycle walk continued to advance `ic_cnt` /
`k_cnt` for one cycle AFTER `k_at_last` fired — the "stray cycle"
between the last legitimate weight read and the FSM transition to
ST_REQUANT. The previous 13a fix gated `weight_rd_en` / `act_in_rd_en`
on `~mac_done` to suppress a stray URAM/BRAM read in that cycle, but
the counter advance was NOT gated, so it kept hitting the walk's
`else` branch (`ic_cnt <= ic_cnt + 1`). `k_at_last` already had reset
`k_cnt` to 0, so the cycle right after fell into the unconditional
`else`. Net effect: after every OC pass, `ic_cnt` was left at 1 when
the FSM came back into ST_RUN for the next OC pass.

When ST_RUN was re-entered, the rising-edge block AND the walk's
per-cycle body both fired at the same posedge (`run_active` was now 1,
`run_active_d` was still 0 — the rising-edge condition fires; the
walk fires because `if (run_active)`). They both scheduled
`ic_cnt <= ...`:
- rising-edge: `ic_cnt <= 0`
- walk:        `ic_cnt <= ic_cnt_pre + 1 = 1 + 1 = 2`

Non-blocking semantics make the LATER assignment win, so `ic_cnt`
became 2 (or 1 if the leftover PRE was 0), and the walk's address
emission used the PRE value of 1 for its outputs. Consequence: the
engine SKIPPED `ic = 0` of every OC pass after the layer's very first
one (pass 0 of pixel (0,0) was unaffected because run_active was 0
at PRE the first time ST_RUN was entered, so the walk did not fire
that cycle).

## Evidence

`scripts/diag_engine_pixel_290.py` showed that for every mismatched
channel of conv_290 at pixel (0,0) pass 1 (lanes 21, 33, 47, 75,
120, 146, 173, 185), the engine accumulator was off by exactly
`-product[ic=0, oc=ch]`. The same delta pattern held for 16/16 lanes
sampled. The 8 output pixels with mismatches in conv_290 are exactly
the pixels where `a[r,c,0] != 0`:
```
input[r,c,0]:
   1    0    0    0    0    1    0      <- (0,0), (0,5) affected
   0    0    0    0    0    0    0
   1    0    0    0    0    0    0      <- (2,0)
   1    0    0    0    1    0    0      <- (3,0), (3,4)
   0    0    0    0    0    1    0      <- (4,5)
   0    0    0    1    2    0    0      <- (5,3), (5,4)
   0    0    0    0    0    0    0
```
Cycle-by-cycle TB dump (`DBG_DUMP_MAC`/`DBG_DUMP_ACC` flags in
`tb/engine_one_layer_tb.v`) on cycles 2510..2550 confirmed:
- **before fix**: at posedge after k_at_last, ic_cnt advanced from 0
  to 1 and persisted through ST_REQUANT until the next ST_RUN entry;
  the first MAC of pass 1 saw `byte_idx_d = 1` and processed ic=1's
  activation.
- **after fix**: ic_cnt stays at 0 through ST_REQUANT; the first MAC
  of pass 1 sees `byte_idx_d = 0`, `act = 1` (matches `input[0,0,0]`)
  and the product enters the accumulator.

## Fix

`output/rtl/engine/address_generator.v`: wrap the per-cycle counter
advance block (the `if (k_at_last) ... else if (ic_cnt == cfg_ic-1)
... else ic_cnt <= ic_cnt+1` chain) in `if (!mac_done) begin ... end`.

This preserves:
- the LAST legitimate MAC (which fires from `weight_rd_addr` /
  `act_in_rd_addr` LATCHED in cycle T(k_at_last) — the address-emission
  block remains ungated), and
- the existing `~mac_done` gating of `weight_rd_en` / `act_in_rd_en`
  that suppresses a stray URAM/BRAM read at T(k_at_last)+1,
while preventing the counter leakage into subsequent OC passes.

The fix is one 3-line wrap (`if (!mac_done) begin` ... `end`) around
lines 301–326 of the existing always block.

## Verdict

`py scripts/engine_sweep_driver.py --workers 4` after fix:

| dispatch | module       | before     | after  |
|---------:|:-------------|:-----------|:-------|
| 0        | node_conv_246| PASS       | PASS   |
| 1        | node_conv_250| PASS       | PASS   |
| 2        | node_conv_254| PASS       | PASS   |
| 3        | node_conv_260| FAIL 27    | PASS   |
| 4        | node_conv_264| FAIL 663   | PASS   |
| 5        | node_conv_266| FAIL 2     | PASS   |
| 6        | node_conv_272| FAIL 62    | PASS   |
| 7        | node_conv_278| FAIL 95    | PASS   |
| 8        | node_conv_282| FAIL 2859  | PASS   |
| 9        | node_conv_286| FAIL 8578  | PASS   |
| 10       | node_conv_290| **FAIL 184** | **PASS** (target) |
| 11       | node_conv_294| PASS       | PASS   |
| 12       | node_conv_296| FAIL 413   | PASS   |
| 13       | node_conv_300| PASS       | PASS   |

**14/14 PASS, byte-exact (max_error=0 across all 14 layers).** The
same root cause fix resolved every other failing dispatch in the
sweep. No regressions on the 5 previously-passing layers.

`output/engine_sweep_results.json` -> `n_pass=14, n_fail=0`.
