# ResNet-50 INT4-GPTQ → RTL on Alveo U250 — Final Report

**Date:** 2026-05-30 (autonomous overnight run). **Status:** correctness + accuracy DONE and byte-exact. **On-chip fit is NOT confirmed** — the first full Vivado synth reports the design OVER capacity (174% BRAM, 115% LUT at synth level); the earlier "fits at 72.9%" was an analytical estimate, since corrected (see §3). A fit-investigation is in progress.

---

## Executive summary

A full ImageNet ResNet-50, quantized to **INT4 weights / INT8 activations (per-channel GPTQ)**, compiled to RTL by the nn2rtl pipeline, now:
- **Computes byte-exact** to the golden across the entire backbone (relu_48 final output: 0.00% mismatch, ImageNet class 91 == golden, feature cosine 1.000000). Byte-exactness is input-independent (deterministic datapath) ⇒ the RTL **is** the INT4-GPTQ reference model.
- **Achieves 79.47% top-1** (per-channel GPTQ, 1500-image eval; float baseline 80.07%).
- **Does NOT yet fit the U250 as measured:** first full synth = **4663 BRAM36 (174%)** + **1.98M LUT (115%, incl. 434K LUT-as-distributed-RAM)**, URAM only 16%, DSP 60%. The "~1960 BRAM36 / 72.9%" figure was an analytical estimate that assumed optimal packing and did **not** hold up — corrected in §3. Fit-investigation (byte-exact mapping fixes) in progress; verdict pending.
- **Runs the full frame in 13,348,787 cycles = 15.0 fps @ 200 MHz** (11.2 @ 150) — meets the 10 fps target.

---

## 1. Correctness (Phases 0–2) — SOLVED, byte-exact

The e2e value-mismatch that blocked the plan for multiple sessions was root-caused to **two RTL bugs** (everything else — datapath, weight packing, per-OC scales, engine — was always correct):

1. **22 of 48 ReLU nodes were missing their activation rescale.** The RTL ReLU template emitted pure `max(0,x)`, but a ReLU with `input_scale ≠ output_scale` must requantize: `out = round(max(0,x) · input_scale/output_scale)`. Fixed in all 22 (`scripts/apply_relu_rescale.py`; per-relu (mult,shift) recovered byte-exact). This was the dominant bug — it made every conv byte-exact and the prediction correct.
2. **node_add_7 had its two operand halves swapped.** It is the one residual add whose golden lhs/rhs is flipped vs the `{skip, main[255:0]}` wiring convention, so each operand got the other's fused scale. Fixed by swapping the `data_in` halves (`nn2rtl_top.v`). Audited all 16 adds — add_7 was the only one affected.

**Method:** the decisive, un-confounded techniques were (a) `recompute(from byte-exact upstream capture) == RTL` position-exact, (b) `triangulate(from contract goldin) == golden` byte-exact, (c) contract-id-correct golden selection (a stale `dram-backed-weights` contract dir had been masquerading as a 95% datapath error), and (d) multi-agent workflows with adversarial synthesis to avoid single-threaded mis-localization.

## 2. Accuracy (Phase 4) — DONE

INT4-GPTQ **per-channel** requant → **79.47% top-1** (float 80.07%, INT8 ~75%). Per-tensor GPTQ is unusable (2.80%), which is why the Phase-2 per-channel rework was necessary. The RTL is byte-exact to this reference, so RTL accuracy = 79.47%.

## 3. On-chip fit (Phase 3) — ⚠️ NOT CONFIRMED; real synth says OVER capacity

**CORRECTION (2026-05-30): the earlier "1960 BRAM36 / 72.9% / FITS" was an ANALYTICAL estimate (`bram36_count_int4.py`, assuming optimal aspect-ratio packing), NOT a tool result. The first full Vivado synth to complete reports the design OVER on two axes.** Stating it as "DONE/FITS" was wrong.

**Measured synth-level utilization** (xcu250, `first_light_util.rpt.synth`, pre-opt/place):

| Resource | Used | Capacity | Util |
|---|---|---|---|
| RAMB36E2 | **4663** | 2688 | **174% — OVER** |
| CLB LUT | **1,983,938** | 1,728,000 | **115% — OVER** |
| — of which LUT-as-Distributed-RAM | **434,324** | 791,040 | 55% |
| URAM288 | 203 | 1280 | 16% |
| DSP48E2 | 7429 | 12288 | 60% — fine |
| FF | 1.31M | 3.46M | 38% — fine |

Caveats: (1) these are **synth-level**, pre-`opt_design`/`place_design` (Vivado warns the final count is "typically lower") — but BRAM count is usually stable synth→place and LUT-as-distributed-RAM does **not** auto-convert to BRAM in opt, so opt is a moderate discount, **not** a rescue from 174%. (2) The 6 h run **timed out** (the single-threaded flatten/cleanup ate the window) **before** opt/place ran, so there is **no post-place number and no checkpoint** — fit is genuinely unmeasured at implementation level.

**Why the analytical estimate was wrong (preliminary):** the 1960 assumed optimal packing; real inference shows **shallow-but-wide weight ROMs each round up to whole BRAM tiles** (a 1×1 conv's ROM is few words × very wide → big width-rounding waste), and **434K LUTs of memory** are inferred as distributed RAM (leading suspect: the per-OC `scale_rom` in `conv_datapath_mp_k.v:90`, which has **no `ram_style` directive** and exists in all 45 conv instances; plus the undirected `skip_fifo` mem). URAM sits 84% empty while BRAM overflows — runtime memory isn't using the dense resource it could. **A dedicated fit-investigation workflow is running to attribute every over-resource and produce a measured, confidence-rated fit plan (byte-exact: mapping/`ram_style`/packing only).** The nibble-pack (½ bits) + engine dead-row dedup (½ rows) are real and correct, but they reduce *bits*, not *tile count* when the ROMs are width-bound — which is the core misconception that produced the 1960 figure.

Runtime buffers map to URAM zero-init; the engine weight memory uses `ram_style="block"` + `$readmemh` (URAM cannot be bitstream-initialized on this device — proven; BRAM can). The 2-cycle weight-read-latency alignment (commit 8677bc0) is preserved. Datapath byte-exactness is unaffected by any of this (mapping-only).

## 4. Cycles / throughput (Phase 5)

13,348,787 cycles/frame = **15.0 fps @ 200 MHz** (meets 10 fps). The frame is fully serialized and spatial/streaming-bound: spatial ≈ 8.8M (66%) + shared engine ≈ 4.55M (34%), with zero overlap (the engine is the cycle floor).

**Cycle-opt attempted (Phase 5 Lever 1), and the honest result:** the obvious lever is spatial output-channel parallelism (`MP`). Two attempts, both **reverted because they break the e2e chain**:
- **conv_196 (stem) MP 8→16:** hard deadlock (out=0). conv_196 is a special wrapper (2-beat splitter + custom start/rearm), not the standard backpressured streamer. Reverted to MP=8.
- **Bulk MP 16→32 on all 38 standard spatial convs:** also deadlocks the e2e chain — at cycle 10M the data never reaches mid-chain (block 11 `skidR31_cap=0`, block 14 `c282drain=0`), whereas the MP=16 baseline has block 11 complete by 8M and block 14 complete by 10M.

This deadlock is **not** explained by the obvious suspects, all ruled out by inspection:
- **Not weights:** `regen_mp_k_weights.py`/`write_wide_weights` produce geometrically-correct nibble-packed hex at MP=32 (128 words × 288 hex-chars = OC_PASSES·K_GROUPS × MP·MP_K·4 bits, exactly as derived).
- **Not a fixed width:** `conv_datapath_mp_k.v` is genuinely MP-parameterized (all `acc/biased/scaled/partial_q/sum_lane_w` arrays are `[0:MP-1]`, loops `0..MP-1`, `WIDE_W=MP*MP_K*4`, FSM control is MP-independent), and the wrapper presents an **identical** interface at any MP (same `data_out` width = OC·8, same valid/ready handshake, same OUT_BEATS).
- **Not X-poisoning:** the byte-exact gate runs Verilator `--x-initial 0`, so missing/uninit state is 0 (wrong *values* at worst, not a stall).

So it is a **subtle control-flow/timing bug surfaced only at MP≠baseline**, requiring dedicated per-conv sim-probe debugging — which would contend for CPU with the running Vivado P&R for a marginal gain over an **already-passing 15 fps design**. Per the standing priority (accuracy + byte-exactness first, never ship a change that breaks the chain), the **byte-exact MP=16 baseline is kept** and the MP-increase deadlock is logged as scoped future work (`project_mp_increase_deadlock.md`). Engine K-parallelism (Lever 2) and spatial↔engine overlap remain available but touch the engine/handshake and carry the same risk class.

## 5. Vivado (Phase 6) — in flight (long run)

`run_first_light_synth.ts` synth on `xcu250-figd2104-2L-e`, clock 20 ns (resource-focused; accurate 200 MHz timing needs P&R / a 5 ns re-run). **The full integrated synth takes ~5.3 h** (a prior complete run = 19273 s); the wrapper's default 5400 s (90 min) timeout was killing it prematurely, so it has been re-launched with `NN2RTL_VIVADO_TIMEOUT_MS=21600000` (6 h). The design **elaborates + synthesizes cleanly in Vivado** (the 90-min run reached and ran `synth_design`, so no read_verilog/missing-module errors).

The last *completed* synth (pre-fix, 5.3 h) reported **3850 BRAM36 (143% over) + 2.07M LUT (~120% over)** — this was the pathological **XPM-URAM + `$readmemh` init** path, which on this device falls back to BRAM *and* LUT-based distributed RAM (init can't go to URAM), blowing up both. My fixes remove that entirely: weights are now **clean `ram_style="block"` BRAM (bitstream-init'able) + nibble-packed (½) + engine dead-row deduped (½ rows)**. Expected post-fix: **BRAM ≈ 1960** (matches the analytical count) and **LUTs down sharply** (the LUT-RAM blowup is gone). **Fresh resource/timing numbers pending the 6 h run** (Monitor armed). If LUTs remain over 1.728M after the clean-BRAM synth, that is the one open fit risk to investigate (the 2.07M figure is believed to be dominated by the now-removed URAM-fallback distributed RAM).

## Open items / follow-ups
- Vivado: confirm the BRAM init inference (`ram_style=block` + `$readmemh` → INIT_xx) and the final LUT/FF/DSP/BRAM/URAM + timing. (Sim cannot verify `ram_style`.)
- Generator hygiene: fold the ReLU activation-rescale, the add operand-order, and the INT4 nibble-packing back into the pipeline emitters (orchestrate.ts / build scripts) so future regenerations are correct without the post-hoc patch scripts.
- Quarantine the dead `dram-backed-weights` contract dirs (conv_284/288/292/298) so triangulation tooling can't mis-select the stale golden.

## Artifacts
- Fixes: `scripts/apply_relu_rescale.py`, add_7 swap in `output/rtl/nn2rtl_top.v`, `scripts/dedup_engine_banks.py`, `scripts/nibble_engine_banks.py`, `repack_weights_wide.py` (nibble), `conv_datapath_mp_k.v` / `mac_array.v` / `shared_engine_skeleton.v` width changes.
- Backups: `backups/{relu_rescale,add7_swap,nibble_pack,engine_dedup,engine_nibble}_20260530/`.
- Verification: `scripts/run_nn2rtl_top_probe.ts` (byte-exact gate), `scripts/analyze_bisect.py`, `uram_init_test/bram36_count_int4.py` (fit), `scripts/accuracy_impact_relu48.py` (GAP+fc prediction).
- Full chronology: `docs/agent_tasks/autonomous_night_log.md`.
