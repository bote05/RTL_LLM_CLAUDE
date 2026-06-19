# nn2rtl on Alveo U250: deployment plan for MobileNetV2

Companion to [nn2rtl_u250_deployment_plan.md](./nn2rtl_u250_deployment_plan.md) (the ResNet-50 plan). Reads downstream of the ResNet-50 deployment campaign, which produced the infrastructure (engine RTL + sub-blocks, parallel sweep driver, byte-exact engine TB, parallel improve sweep, parallel synth driver) that the MobileNetV2 retarget reuses.

This document is the forward-looking engineering plan for taking the 97 already-passing MobileNetV2 INT8 modules to a single working FPGA design on AMD Alveo U250. The intent is to be deliberate about what carries over from ResNet-50 and what doesn't — particularly the depthwise convolution path, which is the central architectural question of this retarget.

## 1. Target and scope

- **FPGA target**: AMD Alveo U250 (`xcu250-figd2104-2L-e`, silicon is XCVU13P). Same chip as the ResNet-50 deployment.
  - 1.73 M LUTs, 12,288 DSPs, ~3.46 M flip-flops, 2,688 BRAM36 (= 5,376 BRAM18-equivalents), 1,280 UltraRAM blocks, 64 GB DDR4 across 4 SLRs.
- **Network**: MobileNetV2 INT8, imported from `checkpoints/mobilenet_v2.onnx` via the universal ONNX frontend (the same one used after the legacy ResNet-50 path).
- **Output**: a working FPGA design with measured PPA. Same supervisor-blessed simulation-only acceptance: post-route Vivado output plus end-to-end Verilator verification on 50k ImageNet validation images is sufficient.
- **Memory policy**: on-chip only. No external DDR. MobileNetV2 INT8 weight footprint is ~3.5 MB (vs ResNet-50's 22.4 MB), so URAM pressure is much lower. The ResNet-50 lesson that U250 URAM does NOT support non-zero `$readmemh` init still applies — weight tensors will fall back to BRAM unless loaded at runtime via AXI.
- **Out of scope**: training, FP32 inference, multi-FPGA partitioning, dynamic shape support.

## 2. The hard fact

The per-module LUT sum on **U250 (Phase 0 baseline, 97 modules)** is **2,042,651 LUTs**. The U250 has 1.73 M LUTs. **A fully spatial deployment of MobileNetV2 does not fit U250 either** — at 118% of LUT budget pre-integration, the design overshoots without any cross-boundary slack.

But the distribution is different from ResNet-50's:

- **17 depthwise convolutions** (3×3, `groups == channels`) sum **1,313,679 LUTs (76.0% of U250 alone)**. The single heaviest layer is `node_conv_818` (depthwise 3×3, IC=OC=96, IH=IW=112) at 336,522 LUTs — almost 20% of U250 in one module.
- **35 pointwise convolutions** (1×1, ordinary dense) sum approximately 500 kLUT. The heaviest, `node_conv_912` (1280×320×1×1, the final expansion before global average pool), is 82,686 LUTs.
- **35 ReLUs** sum 220,456 LUTs.
- **10 residual adds** sum ~10 kLUT.
- **1 global average pool (`node_mean`) and 1 Gemm (`node_linear`)** are present in the LayerIR but their RTL has NOT yet been generated (`pipeline_state.json` shows 97/99 passing; the missing two are exactly these).

This distribution is the OPPOSITE of ResNet-50's. ResNet-50's heavies were the dense 3×3 convolutions in stages 3 and 4. MobileNetV2's heavies are the **depthwise** layers. The ResNet-50 shared engine is *output-channel-parallel with cross-channel reduction* — built explicitly to amortize dense conv MAC cost. Depthwise convolutions have **no cross-channel reduction**: each output channel `c` depends only on input channel `c` over the 3×3 spatial window. The engine in its current form cannot process them efficiently.

This is the load-bearing architectural decision of the MobileNetV2 plan. §3 commits to it.

## 3. Architecture commitment: hybrid, on-chip-only memory, **depthwise stays spatial**

- **All weights live on-chip.** MobileNetV2's 3.5 MB INT8 weight footprint comfortably fits BRAM (the URAM init limitation found during ResNet-50 deployment means weights default to BRAM anyway).
- **All activations live on-chip in BRAM.** Largest single activation tensor in MobileNetV2 is 0.6 MB; budget unchanged from ResNet-50.
- **Pointwise convolutions** with significant LUT cost go to the existing **shared engine** (the same `output/rtl/shared_engine_skeleton.v` + 5 sub-blocks already byte-exact-verified on 14 ResNet-50 heavy convs). Engine handles 1×1 conv as a degenerate case of its existing K=KH×KW MAC walk (KH=KW=1, single kernel position per pixel).
- **Depthwise convolutions stay spatial.** With heavy Phase 1 compression they fit, and they do not benefit from the engine's cross-channel parallelism (each lane is independent). This is the realistic commit; an alternative depthwise engine variant is named in §6.8 as future work.
- **ReLUs, residual adds, GAP, Gemm** all stay spatial. GAP and Gemm need new per-layer RTL via Foundry (their contract docs exist but no .v generated yet).
- **Scheduler + skip-FIFOs + top wrapper**: all mechanically regenerated from MobileNetV2's LayerIR using the existing scripts (`scripts/build_top_wrapper.ts`, `scripts/build_scheduler.py`, `scripts/build_weight_memory_map.py`).

### Memory budget summary (rough, MobileNetV2)

| | Required | Available on U250 | Utilisation |
| --- | ---: | ---: | ---: |
| Weight bytes (INT8, all conv2d + Gemm summed) | ~5 MB | 11.8 MB BRAM (after URAM init fallback) | ~42% |
| Bias bytes (INT32 per output channel) | <0.1 MB | included above | negligible |
| Activation ping-pong + skip buffers | ~1 MB | 11.8 MB BRAM | ~8-10% |
| **Total on-chip memory needed** | **~6 MB** | **~12 MB BRAM + 45 MB URAM (unused)** | **~50% BRAM** |

This is far less BRAM pressure than ResNet-50's 93% post-synth (where the weight banks alone overwhelmed BRAM after the URAM fallback). MobileNetV2 has 6× fewer weights, so the same BRAM-fallback path is comfortable.

## 4. Phase 0 — Finalize per-module baseline (1 week)

- 97/99 modules already pass per-module byte-exact + per-module Vivado on U250. Re-confirm none regressed since they were last touched.
- **Generate the two missing modules** (`node_mean` global_avg_pool and `node_linear` gemm) via Foundry. New op_types under the pattern docs that already exist in the failure corpus (the docs were authored after ResNet-50 but no MobileNetV2 module exercised them yet).
- Re-run `py scripts/lint_boundaries.py --check-meta` after adding the two modules — confirm the chain (PIXEL_IN → … → node_linear → CLASS_OUT) is coherent.

**Deliverable**: 99/99 modules passing, lint clean, U250 area + Fmax baseline table.

**LLM contribution**: 2 Foundry dispatches (the two missing ops). ~$15-30.

## 5. Phase 1 — Targeted compression (2–3 weeks; the load-bearing phase)

This phase carries more weight than it did for ResNet-50 because MobileNetV2 depends on it to fit at all. Without heavy Phase 1 wins on the depthwise side, the design simply doesn't close the LUT budget.

### 5.1 Spatial compression targets

Order by expected impact:

1. **17 depthwise convs** (target: −40% LUT each on average, driven by `reduce-lut` and `use-bram`). The largest individual ones (`node_conv_818` at 337 kLUT, `node_conv_824` at 195 kLUT) have the most absolute room to compress; even −30% on the top two saves ~160 kLUT. The improve loop has not seen depthwise specifically before, so expect 2-3 iterations per heavy depthwise.
2. **6 pointwise convs ≥40 kLUT**: `conv_912`, `conv_910`, `conv_894/900/906`, `conv_852`. Target: −30% via `reduce-lut` and `use-bram`. Each LUT-ROM → BRAM tag move yields the ResNet-style large win (89% on `conv_252`).
3. **35 ReLUs**: skip. Each is <10 kLUT, improve cost > savings.

Planning assumption summed: **−40% on depthwise (~525 kLUT saved) + −30% on top 6 pointwise (~120 kLUT saved) = ~645 kLUT freed.** Post-compression sum ≈ 1.40 M LUT before integration overhead. Targeting under 70% utilisation post-route requires the conservative 30%/40% rates to be at least met; the optimistic case (the `−89%` LUT-ROM → BRAM transform we got on `conv_252`) repeated on each MobileNet pointwise would land us closer to 1.1 M LUT before overhead.

### 5.2 Workflow per module

Same as the ResNet-50 Phase 1 sweep:

1. `reduce-lut` → check
2. If LUT still > 30 kLUT → `use-bram` → check
3. If FF > 50 kLUT → `reduce-ff` → check
4. Promote any successful variant to `improved/` tier.

Use `scripts/run_improve_parallel.py` (it already supports MobileNetV2 via the network registry) with `--workers 4`. Budget: ~$200-300 LLM across the 17 depthwise + ~6 pointwise heavy candidates.

### 5.3 Expected outcome

Spatial side drops from ~2.04 M raw → ~1.40 M LUT after Phase 1. The 14-ish modules dispatched to the engine (see §6.1) drop out of the spatial budget entirely.

## 6. Phase 2 — Shared engine + integration (2-3 weeks; mostly reuse)

**This is the lightest phase compared to ResNet-50**, because the engine + sub-blocks already exist and have been byte-exact-verified.

### 6.1 Engine dispatch list

Pointwise convolutions ≥ 40 kLUT raw, the same selection logic as ResNet-50's heavy list, applied to MobileNetV2:

- `node_conv_912` (1280 × 320 × 1×1, 7×7 → 7×7)
- `node_conv_910` (320 × 960 × 1×1, 7×7 → 7×7)
- `node_conv_898, 900, 904, 906, 894` (each 160/960 × 1×1, 7×7)
- `node_conv_892` (160 × 576 × 1×1, 7×7)
- `node_conv_876, 880, 882, 886, 888` (each 96/576 × 1×1, 14×14)
- `node_conv_874` (96 × 384 × 1×1)
- ~14 modules total (final count after Phase 1 may shrink if compression succeeds enough).

### 6.2 What changes vs ResNet-50's engine path

- **Per-layer config** — different `cfg_input_channels`, `cfg_output_channels`, `cfg_kernel_h`, `cfg_kernel_w`, `cfg_stride_*`, `cfg_padding_*` for each MobileNet dispatch. `scripts/build_scheduler.py` regenerates the schedule deterministically from the MobileNetV2 LayerIR.
- **Smaller per-layer weight footprint** — MobileNet's largest pointwise weight slice is ~409 kB (vs ResNet-50's 2.36 MB per 512×512×3×3). Fewer BRAM cascaded per bank.
- **Engine verification per heavy layer** — re-run `scripts/engine_sweep_driver.py --workers 4` over MobileNet's dispatch list. The engine RTL itself is unchanged; the verification re-confirms it on the new shapes (1×1 K=1×1 with various IC/OC values). Expected to PASS at first run because the engine is bit-exact on the K=1 degenerate case (the `address_generator` walks `(ic, kh, kw)` and K=1 just means kh=kw=0 always).
- **Skip-FIFOs**: 10 residual adds in MobileNetV2 (vs ResNet-50's 16), inverted-residual topology. Each add's skip path runs through a `skip_fifo` whose depth is sized per §6.5 of the ResNet-50 plan. Methodology unchanged; the latencies are smaller because MobileNetV2 layers are individually less expensive.

### 6.3 Top wrapper regeneration

`npx tsx scripts/build_top_wrapper.ts` after `setActiveNetwork mobilenet-v2`. Same generator. Different graph.

### 6.4 Engine bit-exact verification (the critical gate)

For each MobileNet dispatch in `output/mobilenet-v2/rtl/nn2rtl_scheduler_schedule.json`:

1. Configure engine for that layer's shape + scales.
2. Feed its `.goldin` through the engine.
3. Compare output BRAM range to `.goldout` byte-for-byte.

Same TB (`tb/engine_one_layer_tb.v`), same sweep driver, just a different `dispatch_idx → module_id` table. Verifies the engine fix from ResNet-50 (the `~k_at_last` → `~mac_done` address_generator bug fix) generalises to K=1×1 dispatches (it should: the bug is independent of K and was always per-OC-pass).

### 6.5 No engine changes expected, but…

If the bit-exact sweep surfaces a new bug specific to K=1 dispatches or to the smaller IC/OC values, dispatch a debug agent per the ResNet-50 playbook. The address_generator bug took one Foundry agent ~2 hours to root-cause; a similar latency budget applies if anything else surfaces.

### 6.6 Wrapper / scheduler integration audit

Run the equivalent of ResNet-50's Task 13a after the top wrapper regenerates:

1. `iverilog -t null` strict full elaboration of `output/mobilenet-v2/rtl/nn2rtl_top.v + scheduler.v + engine sub-blocks + node_*.v + rtl_library/*.v` → exit 0.
2. `verify_weight_memory_map.py` on MobileNetV2 layers — should be 0 mismatches across all dispatches.

### 6.7 What's brand-new for MobileNetV2 specifically

- The `gemm` op_type (final 1280→1000 classifier). Pattern doc exists; no module yet. Foundry dispatch; should pass first or second attempt given the simple shape.
- The `global_avg_pool` op_type (1280×7×7 → 1280×1). Pattern doc exists; no module yet.
- Inverted-residual block topology in the top wrapper. The skip-FIFO sizing methodology is unchanged but the address arithmetic differs slightly (skip span ≈ 5 layers per block vs ResNet-50's 3).

### 6.8 Depthwise engine variant — explicitly named future work, not in this plan

The depthwise convolutions could be moved into an engine variant that:
- Walks the 3×3 spatial window with one MAC per output channel
- No cross-channel reduction (`mac_array` simplifies dramatically — 256 independent INT8 multiplies, no accumulator tree across channels)
- Reuses the address_generator, config_register_block, and scheduler — only `mac_array.v` and `requant_pipeline.v` need a depthwise variant

This is ~2-3 weeks of new sub-block design + bit-exact verification. **Out of scope for this plan.** If post-compression depthwise still dominates LUT and timing closure fails, this is the documented next move.

## 7. Phase 3 — End-to-end empirical verification (1-2 weeks)

50,000 ImageNet validation images through:

- The quantised PyTorch reference (the ground truth).
- The Verilator-simulated integrated nn2rtl design.

Reports identical to ResNet-50 §7: per-output-logit max abs error distribution, Top-1 accuracy of both with gap, Top-5 accuracy of both with gap. Same 1.0 percentage point tolerance.

MobileNetV2 baseline Top-1 INT8: ~71.8% (vs FP32 72.1%). The deployment target is therefore Top-1 within [70.8%, 72.8%].

**LLM contribution**: low. Reuses the integrated harness built for ResNet-50 (which is itself pending — Phase 3 of the ResNet-50 deployment will produce this infrastructure first).

## 8. Phase 4a — Timing closure on U250 (3-5 weeks)

Same iterative loop as ResNet-50 §8. Expectations:
- Initial post-route Fmax estimate: 80-150 MHz (smaller modules → shorter critical paths → higher than ResNet-50's expected 100-150 MHz)
- Worst paths likely in depthwise convs' wide MAC trees (each lane is independent but the trees are still on-chip), and in the engine→spatial bridge transitioning from the heavy pointwise outputs.
- Pipelining iterations expected: 3-6 (less than ResNet-50's likely 5-10, because design is smaller).

## 9. Phase 4b — Measurement (1 week)

Post-route Vivado reports + Verilator throughput estimate + power via SAIF-driven `report_power`. PPA table reports: LUT, FF, DSP, BRAM18-eq, Fmax, fps end-to-end, ms latency per image, power (watts), **GOPS/W**, Top-1 ImageNet accuracy.

## 10. Intermediate deliverables (de-risking)

Same pattern as ResNet-50 §10:

- **Mini-deliverable A** — stem + first inverted-residual block (~6 layers) on U250. All-spatial, no engine. Validates the inverted-residual topology + skip-FIFO sizing on a small example. Achievable by end of Phase 1.
- **Mini-deliverable B** — stem + first three inverted-residual blocks + engine integration on the largest pointwise layer in those blocks (~15 layers). Validates that MobileNetV2's pointwise→engine dispatch path works. Achievable by end of Phase 2.

Both are first-class deliverables — measurable PPA + integration methodology even if the full 99-layer design has trouble closing timing.

## 11. Success criterion (pre-committed)

Design is called successful if it meets *all*:

- ≥ 20 fps end-to-end throughput on U250. (Higher than ResNet-50's 10 fps because MobileNetV2 is much smaller — ~300 MFLOPS vs ResNet-50's ~4 GFLOPS, so even at the same Fmax we should see significantly higher fps.)
- Top-1 ImageNet accuracy within 1.0 percentage point of the quantised PyTorch reference (target: ≥ 70.8%).
- Total LUT post-route ≤ 70% of U250 budget (≤ 1.21 M LUTs). Tighter than ResNet-50's 95% because the smaller network should give more headroom; if it doesn't, that's a sign Phase 1 underperformed.
- Fmax post-route ≥ 150 MHz. (Tighter than ResNet-50's 100 MHz because smaller modules should clock higher.)
- GOPS/W within 0.3× of the best published Alveo U250 INT8 MobileNetV2 result. Primary source: same Vitis AI 2.5 DPU baseline as ResNet-50; the DPU runs MobileNetV2 with its own publicly-documented PPA.

## 12. Comparison strategy (pre-committed)

**Whole-network external baseline: Vitis AI 2.5 DPU on U250 (DPUCADF8H), MobileNetV2 INT8.**

Same deprecation caveat as the ResNet-50 plan §12: Vitis AI 3.0+ has dropped U250 support. The comparison is still valid (published numbers, same chip, same precision, same network) but should be explicitly documented as against a legacy production target, not a current one.

**Layer-level external baselines**: nn2rtl per-module Vivado numbers vs hls4ml and FINN on the same MobileNetV2 layers, in the same format as `comparison/tier_a/compare_three_way.csv` for ResNet-50. Particularly interesting for the depthwise convolutions, which historically expose tool-quality differences (hls4ml struggles with depthwise; FINN handles it natively).

## 13. Contribution claim (headline)

The methodology claim is identical to the ResNet-50 one. The MobileNetV2 retarget specifically demonstrates:

- The infrastructure (Foundry/Surgeon/Retrospector loop, pattern + reference + failure-corpus library, parallel improve sweep, parallel synth driver, byte-exact engine TB) carries over to a structurally different network — not just a same-family network — with bounded incremental effort.
- The hybrid engine + spatial architecture handles two distinct heavy-layer regimes (ResNet's dense 3×3 and MobileNet's depthwise+pointwise) with appropriate placement of each.
- The deployment plan's planning-assumption compression rates were validated on ResNet-50 (`conv_252 −89%`, `conv_288 −84%`, `conv_284 −82%`); the same improve-loop machinery applied to MobileNetV2's depthwise layers should hit similar wins on the LUTRAM→BRAM transitions.

## 14. Risks and known unknowns

- **Phase 1 compression on depthwise may underperform.** Depthwise convs have less structural slack than dense ones — fewer DSPs absorbed per MAC, less LUTRAM consolidation opportunity. If actual compression is closer to −20% instead of −40%, the LUT budget tightens; mitigation is the depthwise engine variant (§6.8) at the cost of ~2-3 extra weeks.
- **Engine bit-exact on K=1×1 dispatches.** The engine was verified on K=3×3 dispatches in ResNet-50. K=1×1 is a degenerate case that *should* work — the address_generator iterates 1 kernel position per pixel — but has not been explicitly tested. Verification in §6.4 is the gate.
- **`node_linear` (Gemm 1280→1000) RTL has not been generated yet.** The contract docs exist; the failure-corpus knowledge from earlier networks should accelerate Foundry's first attempt. Plan adds 1-3 dispatch days as buffer.
- **Inverted-residual skip topology** changes the FIFO sizing math from ResNet-50's basic residual blocks. Skip spans are still small (~5 layers), but the methodology was validated on ResNet-50's topology; MobileNet's may surface edge cases the analytical formula misses. Cycle-accurate Verilator simulation per residual block (the same gate ResNet-50 used) catches deadlocks.
- **Vitis AI DPU MobileNetV2 baseline.** Less commonly cited than the ResNet-50 baseline. Verify a primary AMD source before Phase 4 starts (Vitis AI 2.5 model zoo).
- **GOPS/W denominator.** MobileNetV2 INT8 ops/inference is ~600 MOPS — a small denominator. fps drives the numerator heavily, so timing closure shortcomings hurt GOPS/W more than they do for ResNet-50.

## 15. Timeline + cost

| Phase | Calendar | Notes |
| --- | --- | --- |
| 0 — Finalize per-module baseline (Gemm + GAP) | 1 week | 2 Foundry dispatches + re-baseline |
| 1 — Targeted compression | 2–3 weeks | LLM-driven improve sweep on 17 depthwise + 6 pointwise heavies |
| 2 — Engine integration + verification | 2–3 weeks | Mostly reuse; new contracts (Gemm, GAP) + skip-FIFO topology + bit-exact gate |
| 3 — Empirical end-to-end | 1–2 weeks | 50k ImageNet validation |
| 4a — Timing closure | 3–5 weeks | LLM-driven iterative repair |
| 4b — Measurement | 1 week | Post-route + Verilator + DPU comparison |
| **Total** | **10–15 weeks** | ~60-70% of the ResNet-50 calendar |

LLM API cost estimate:

- Phase 0 new ops: ~$15-30
- Phase 1 improve sweep: ~$200-300
- Phase 2 verification + any bug-hunt agents: ~$30-100
- Phase 4a timing closure: ~$50-150
- **Total LLM cost: ~$300-580.** Approximately equal to ResNet-50's $190-420 because the depthwise compression cohort is larger.

## 16. Reusability — what we learn from doing this

The MobileNetV2 retarget is itself a test of the reusability claims in `nn2rtl_u250_deployment_plan.md` §16. If we land Phases 0+1+2 in fewer than 4 calendar weeks, the methodology claim is well-supported. If it takes longer than 8 weeks, the system has hidden ResNet-50 coupling that needs to be lifted.

Specific carryover audit checkpoints during the retarget:

- **Pattern library**: does each pattern doc apply unchanged? (Expected: 95%+. The 1×1 patterns are network-agnostic; the depthwise pattern doc exists from prior work but was never exercised by a passing module — its first real test is here.)
- **Failure corpus**: do entries from ResNet-50 reduce Foundry attempt count on MobileNetV2's analogous layers? Quantify: for each conv2d shape in MobileNetV2, measure first-attempt PASS rate. Compare to ResNet-50's first-attempt PASS rate on similar shapes. Target: ≥ 80% first-attempt PASS on shapes already present in the failure corpus.
- **Build scripts**: do `build_weight_memory_map.py`, `build_bias_memory_map.py`, `build_scheduler.py`, `build_top_wrapper.ts` produce valid output for MobileNetV2 without any hand-tuning? (Expected: yes, except possibly the Gemm + GAP integration in the top wrapper, which may need a small generator update.)
- **Engine sub-blocks**: do any of the 5 sub-blocks need MobileNetV2-specific changes? (Expected: no. If yes, that's a research finding worth documenting.)

## 17. What this plan is not

- It is not a depthwise-engine-variant plan. That work is named in §6.8 as future direction; this plan commits to keeping depthwise spatial with Phase 1 compression as the load-bearing strategy.
- It is not a fresh start. It explicitly leverages every infrastructure investment made for ResNet-50, including unverified-by-MobileNetV2 ones (e.g. the engine's K=1×1 path).
- It is not a same-effort-as-ResNet-50 claim. The 60-70% calendar reduction is the validated working hypothesis; if the depthwise compression hits the −40% target, the actual figure could be closer to 50%. If depthwise compression flat-lines at −20%, the figure could be 80-90%.
- It is not contingent on the ResNet-50 deployment finishing successfully. The MobileNetV2 work can begin in parallel with ResNet-50's Phase 3+ if the user wants two streams running. The ResNet-50 deployment provides validated infrastructure; it doesn't gate MobileNetV2.
