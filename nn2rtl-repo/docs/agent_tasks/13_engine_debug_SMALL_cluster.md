# 13_engine_debug_SMALL_cluster — conv_266

## Summary

Layer: `node_conv_266` (dispatch_index=5, IC=256, OC=256, K=3×3, S=1, P=1, IH=IW=14).

Failure on entry (before this session): 2 mismatches at pixel (3,5), channels 9 and 58 — both engine_output = gold − 1.

Outcome: **0 mismatches** (PASS) after re-running the sweep. No further RTL changes were needed in this session — the fix that resolves conv_266 had already been committed to `output/rtl/engine/address_generator.v` by the prior conv_290-cluster debug agent. The TB instrumentation added here (`DBG_TRACE_PIXEL` block in `tb/engine_one_layer_tb.v`) confirms the engine now does the full K_TOTAL=2304 MAC walk for the failing pixel.

## Root cause

The address generator's per-cycle walk (the ic/kw/kh advance block) was running on the cycle immediately AFTER `k_at_last` fired. That cycle keeps `run_active=1` for one more clock because the FSM ST_RUN→ST_REQUANT transition is registered. With the advance still ungated, the `else: ic_cnt <= ic_cnt + 1` branch bumped `ic_cnt` from `0` (just reset by the k_at_last branch) to `1`. The leftover `ic_cnt=1` persisted through ST_REQUANT and ST_DRAIN, and at the DRAIN→RUN rising edge of the NEXT pixel the walk ran a second time (the rising-edge `ic_cnt <= 0` is overridden by the same-cycle `ic_cnt <= ic_cnt + 1` because the later non-blocking assign wins). The net effect: every output pixel after the first one started its MAC stream at `ic=1` instead of `ic=0`, silently dropping the very first weight × activation product of `(kh=0, kw=0, ic=0)`.

For most pixels the dropped product was small enough that the requantised INT8 output landed in the same bucket as the golden — so the bug was invisible. Pixel (3,5) channels 9 and 58 of conv_266 are the only positions in the failing set where the dropped product crosses the round-half-toward-+inf boundary. The lane-9 drop was `p = a*w = 1*5 = 5` (engine acc −140 vs gold −135, biased −136 vs −131, scaled −1 vs 0). The lane-58 drop was `p = 1*6 = 6` (engine acc 130 vs gold 136, biased 131 vs 137, scaled 0 vs 1).

## Fix

Already in place at `output/rtl/engine/address_generator.v` lines 300–342 (comment dated `2026-05-24 fix (conv_290 cluster)`):

```verilog
// ---- advance (ic, kw, kh) — ic innermost ----
if (!mac_done) begin
    if (k_at_last) begin
        mac_done <= 1'b1;
        ic_cnt   <= 12'd0;
        ...
    end else if (ic_cnt == (cfg_ic - 12'd1)) begin
        ...
    end else begin
        ic_cnt <= ic_cnt + 12'd1;
        k_cnt  <= k_cnt + 16'd1;
    end
end
```

Gating the entire advance block on `~mac_done` keeps `ic_cnt` at 0 across the one-cycle ST_RUN tail that follows `k_at_last`, so the next pass enters its walk with a clean counter state.

## Verdict

- `output/engine_sweep/result_dispatch05_node_conv_266.json` → `"status": "PASS"`, `"n_mismatches": 0`.
- Pixel (3,5) trace confirms the engine now consumes all 2304 MAC products including `(kh=0, kw=0, ic=0)`; final `acc[9] = -135`, `acc[58] = 136` (both match the python golden bit-exact).

## Side effects on other layers

The same root cause was the conv_290 cluster bug fixed by the previous agent. With the fix in place, conv_266 no longer needs any separate change. Other layers in the failing set (conv_260, conv_264, conv_272, conv_278, conv_282, conv_286, conv_290, conv_296) may benefit from the same fix — a full sweep is the source of truth.

## Files touched in this session

- `tb/engine_one_layer_tb.v` — added a `DBG_TRACE_PIXEL` instrumentation block that dumps the MAC sequence for a chosen output pixel. Read-only on the DUT; controlled by `+DBG_PIX_R=`, `+DBG_PIX_C=`, `+DBG_LANE_A=`, `+DBG_LANE_B=` plusargs. Wrapped in `\`ifdef DBG_TRACE_PIXEL` so the default build is unchanged.
- `scripts/diag_conv266_pixel35.py` — standalone python diagnostic that builds the golden accumulator from `.goldin` + `.hex` weights + bias and prints (a) the per-channel golden value at pixel (3,5) and (b) the set of candidate single-MAC drops whose engine output would match the observed mismatch.
- `scripts/diff_engine_trace.py` — parses the TB trace log and diffs the engine MAC sequence against the python golden walk; reports per-cycle `byte_idx_d / act / weight / product` mismatches.
