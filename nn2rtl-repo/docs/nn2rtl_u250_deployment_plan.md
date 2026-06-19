# nn2rtl on Alveo U250: deployment plan for ResNet-50

This document is the forward-looking engineering plan for taking the verified per-layer ResNet-50 modules that nn2rtl already produces and turning them into a single working FPGA design on AMD Alveo U250. It is written to be read after [nn2rtl_supervisor_explanation.md](./nn2rtl_supervisor_explanation.md), which describes the system as it stands today.

The plan is deliberately concrete and pre-committed. Earlier drafts dodged architectural decisions with phrases like "try spatial first, fall back if needed." Those soft fallbacks always become the actual plan eventually, so this version commits to the decisions up front and names the risks honestly.

## 1. Target and scope

- **FPGA target**: AMD Alveo U250 (`xcu250-figd2104-2L-e`, silicon is XCVU13P).
  - 1.73 M LUTs, 12,288 DSPs, ~3.46 M flip-flops, **2,688 BRAM36** (= 5,376 BRAM18-equivalents) of block RAM, plus 1,280 UltraRAM blocks (~360 Mbit) as a separate memory tier, 64 GB DDR4 across 4 SLRs. The BRAM number is the one I'll quote in the rest of this plan; UltraRAM is treated as overflow memory rather than primary BRAM and is not counted toward the BRAM18 utilisation rows.
- **Network**: ResNet-50 INT8, the same quantised model nn2rtl already produces 119 passing per-layer modules for.
- **Output**: a working FPGA design with measured PPA. Real-silicon deployment is desirable but not required — supervisor has explicitly blessed simulation-only results if the design does not fit silicon at acceptable utilisation. The primary deliverable is post-route-clean Vivado output plus end-to-end Verilator verification on 50k ImageNet validation images.
- **Memory policy**: on-chip only. No external DDR. All 22.4 MB of weights live in UltraRAM, all activations live in BRAM. This is a constraint from the supervisor and is reflected throughout the architecture in §3, §6.1, and §6.6.
- **Out of scope**: training, FP32 inference, multi-FPGA partitioning, dynamic shape support.

## 2. The hard fact

The per-module LUT sum on **U250 (Phase 0 baseline, all 119 modules)** is **2,917,911 LUTs**. (ZCU102 baseline was 2,917,729 LUTs — essentially identical, confirming the toolchain switch is clean. Median per-module ΔLUT% = 0.00% across the chip change.) The U250 has 1.73 M LUTs. Even after aggressive compression and adding integration overhead, a fully spatial deployment of every module simultaneously does not fit U250.

The distribution is heavy-tailed:
- **12 heavy modules** sit at LUT ≥ 85,000 each, summing 1.66 M LUTs (56.7% of the network). These go to the shared engine.
- **13 medium modules** sit between 30,000 and 85,000 LUTs, summing 0.68 M LUTs (23.2%). These stay spatial and go through Phase 1 compression.
- **94 small modules** sit below 30,000 LUTs each, summing 0.58 M LUTs (20.0%). These stay spatial as-is.

(See [docs/agent_tasks/06_phase1_compression_candidates_REPORT.md](agent_tasks/06_phase1_compression_candidates_REPORT.md) for the full breakdown.)

The threshold analysis:

| Heavy-module compression | Medium-module compression | Compressed sum | + 15% integration | + 10% P&R | Final | Fits U250 (≤1.21 M, 70% util)? |
| --- | --- | ---: | ---: | ---: | ---: | :---: |
| 0% | 0% | 2.92 M | 3.36 M | 3.69 M | 3.69 M | No |
| 15% | 10% | 2.58 M | 2.97 M | 3.27 M | 3.27 M | No |
| 30% (likely) | 15% (likely) | 2.33 M | 2.68 M | 2.94 M | 2.94 M | No |
| 50% (optimistic) | 30% (optimistic) | 1.90 M | 2.18 M | 2.40 M | 2.40 M | No |
| 70% | 50% | 1.42 M | 1.63 M | 1.79 M | 1.79 M | Borderline only |
| 80% (unrealistic) | 70% (unrealistic) | 1.18 M | 1.36 M | 1.49 M | 1.49 M | Tight (86%) |

The "comfortable fit" threshold (≤70% utilisation = 1.21 M LUTs post-route) requires average heavy-module compression of at least 85% and medium-module compression of at least 75%. That is not achievable at scale. The best single improve result observed so far is `node_conv_284 reduce-lut` at −71.7%, and that was one module. As an average across forty modules it is fantasy.

**This is the reason the plan is hybrid, not spatial.** The sensitivity analysis is the load-bearing argument; everything that follows is downstream of it.

## 3. Architecture commitment: hybrid, on-chip-only memory

- **All weights live in on-chip UltraRAM.** No DDR streaming. ResNet-50 INT8 weight footprint is 22.5 MB; U250 has ~45 MB of UltraRAM, so the entire network's weights fit on-chip at roughly 50% URAM utilisation. This is a constraint from the supervisor (do not use external DDR) and matches the thesis claim that the deployment is self-contained on the FPGA.
- **All activations live in on-chip BRAM.** Largest single activation tensor in ResNet-50 is 0.77 MB; U250 has ~11.8 MB of BRAM, so activation ping-pong buffers fit comfortably under 8% BRAM utilisation.
- **Spatial dataflow** for the ~95 small and medium modules (most ReLUs, all Adds, the MaxPool, the small convs). These already fit individually and benefit from per-layer parallelism.
- **One shared compute engine** for the ~10 heavy modules (the large 1×1 convs and a small number of 3×3s). The engine processes one heavy layer at a time, sequentially, **reading weights from a pre-loaded UltraRAM region rather than from external memory**.
- **A scheduler / control plane** that holds the activations in BRAM banks, dispatches each heavy layer to the engine in order, and coordinates the handoff with the spatial chain.
- **Per-skip synchronisation FIFOs** for the 16 residual adds, sized empirically.

This is closer in spirit to Vitis AI DPU than to FINN, with the additional constraint that everything is on-chip. That is acceptable — the nn2rtl thesis claim is methodological, not architectural-novelty.

### Memory budget summary

| | Required | Available on U250 | Utilisation |
| --- | ---: | ---: | ---: |
| Weight bytes (INT8, all conv2d weights summed) | 22.4 MB | 45 MB (URAM) | ~50% |
| Bias bytes (INT32 per output channel) | 0.1 MB | included above | negligible |
| Activation ping-pong buffer (largest tensor × 2) | 1.54 MB | 11.8 MB (BRAM) | ~13% |
| Skip-path FIFOs (sized in §6.5) | ~1-2 MB estimated | 11.8 MB (BRAM) | ~10-15% |
| **Total on-chip memory needed** | **~26 MB** | **~57 MB** | **~46%** |

Headroom of ~31 MB is enough room for engine-internal staging buffers and any future expansion (e.g. a second engine variant for depthwise convolutions when MobileNetV2 is retargeted).

## 4. Phase 0 — Re-baseline on U250 (1 week)

Re-run Vivado for all 119 passing modules with the U250 part. Most modules will resynthesise without change (the silicon family is still UltraScale+). The 22 modules whose ZCU102 runs returned `fmax = 0` should be re-verified — these are tool-side flakes, not design problems.

**Deliverable**: U250-baselined area + Fmax table replacing the ZCU102 numbers in the supervisor doc.

**LLM contribution**: minimal. Existing `vivado_resynth_failed.ts` plus a part-target flag.

## 5. Phase 1 — Targeted compression (3–4 weeks)

Not blanket. Only compress the modules that will live in the **spatial** part of the hybrid. The ~10 heavy modules will be re-implemented inside the shared engine and do not need their per-module variants compressed.

Workflow per spatial-targeted module:

1. `reduce-lut` → check
2. If LUT still > 30k → `use-bram` → check
3. If FF > 50k → `reduce-ff` → check
4. Promote the variant to `improved/` tier; the orchestrator picks it up automatically on the next integration build.

Planning assumption: −30% LUT on the 5–8 spatially-routed medium convs that remain expensive. Small modules (ReLU, Add, MaxPool, 1-channel pieces) are not improved further — they are small enough that the compression cost is not worth it.

**Expected outcome**: spatial side drops from ~1.0 M LUTs raw → ~0.75 M LUTs after compression. Heavy modules will be replaced by the engine, so their raw LUT cost is moot for the integrated build.

**LLM contribution**: central. The improve loop is already proven to produce real wins on this exact class of module. Verified bit-exact by Verilator plus Vivado plus the deterministic target checker.

## 6. Phase 2 — Shared engine + integration (5–6 weeks)

This is the new core engineering work and the main thesis artefact for the deployment chapter.

### 6.1 Shared engine design

This is its own sub-project, not a one-shot Foundry call. The plan commits to **option 1**: human-designed skeleton with LLM-generated sub-blocks. The engine is not a single LayerIR module; it is a multi-shape parameterised datapath. Asking the existing Foundry agent to produce it in one prompt is not realistic given how Foundry currently works.

Open design parameters that must be decided before sub-block generation begins:

- **MAC array shape.** The largest heavy module (e.g. `node_conv_298`, 512×512 1×1) needs enough parallel multipliers to be processed in a bounded number of passes. A 256-MAC array sequentialises the 512×512 layer across 2 passes per output pixel; a 512-MAC array is one pass but doubles the multiplier and routing cost. **Tentative commitment**: 256 MACs, balancing engine area against engine throughput, leaving room for the spatial side. To be re-evaluated against U250 DSP budget (12,288) and against engine-only Vivado synthesis after the first iteration.
- **Parallelism axis.** Output-channel-parallel (process MAC-count output channels at one spatial position simultaneously) versus spatial-parallel (process MAC-count spatial positions at one channel pair simultaneously). Output-channel-parallel matches the existing `node_conv_288` seed and the BRAM access pattern of the spatial chain. **Tentative commitment**: output-channel-parallel.
- **Requantisation pipeline depth.** Each heavy layer has its own scale and zero-point; the engine must load these per dispatched layer and apply them in the final stage. **Tentative commitment**: 3 pipeline stages (bias-add, scale-multiply, scale-shift + saturate), matching the existing nn2rtl per-layer tail.
- **Weight port (NEW after on-chip constraint).** The engine reads weights from a pre-loaded UltraRAM region rather than from AXI4-MM to DDR. Per dispatched layer, the scheduler hands the engine the layer's URAM base address and shape parameters; the engine streams its own reads. Eliminates the entire AXI4-MM weight DMA controller, the burst-read state machine, and the prefetch double-buffer that were in the original `node_conv_288` seed.

The engine is functionally a *simplified-and-generalised* version of the existing `node_conv_288` reference (1024-channel 1×1 with DRAM-backed weights). The seed is in `output/rtl/node_conv_288.v`. The DRAM-backed-weights mechanics in that seed are now treated as a separate research artefact (see §16); for the U250 deployment, the engine uses an on-chip URAM weight port instead.

### 6.2 How the engine is actually built

Three honest options, picking one:

1. **Human-designed skeleton, LLM-generated sub-blocks.** The author writes the engine's top-level Verilog skeleton (interfaces, parameter ports, MAC array shell, requantisation pipeline shell, AXI4-MM control). Foundry generates the MAC array body, the requantisation arithmetic, the address-generation logic, the per-layer configuration register block — each as a small bounded prompt with its own goldens. Each sub-block is bit-exact verified against a contract-style golden vector.
2. **Engine LLM-generated from a detailed spec.** Write a spec document describing the engine and ask Foundry to produce it in one shot. Possible but the prompt is larger than anything Foundry has handled to date and the verification surface is wider.
3. **Generalised by repeated improve passes on the seed.** Take `node_conv_288`, parameterise it by running it through a series of improve-like prompts that progressively widen its shape support. The cleanest LLM story but the highest-risk to actually pull off.

**Committed choice: option 1.** It matches what nn2rtl has demonstrated. The LLM contribution stays real but bounded.

### 6.3 Engine verification strategy

The engine does not have its own goldens. It is verified by reusing every heavy layer's existing `.goldin` / `.goldout`:

- **Functional verification** (per heavy layer): configure the engine for layer L's shape and scales, feed L's `.goldin` through it, check that the engine's output matches L's `.goldout` byte-for-byte (with the same `max_error ≤ 3` tolerance as per-layer testing). This runs in the existing static Verilator harness with a small wrapper that programs the engine's config registers before driving inputs.
- **Integration verification**: a sequence of heavy layers dispatched back-to-back. Confirms the scheduler correctly switches the engine between layer configs, that BRAM ownership is correct, that no activation data is dropped between layers, and that the engine drains cleanly between dispatches.

Both must pass before the engine is wired into the full integration.

### 6.4 Scheduler / control plane

The scheduler is **not** just a state machine that picks the next layer. It coordinates four moving parts:

- **Activation memory ownership**. For a linear chain of N layers, two ping-pong buffers suffice (layer N writes A, layer N+1 reads A and writes B, layer N+2 reads B). Residual paths break this: the output of a residual block's *first* layer must be held until both the consuming convolution chain and the corresponding Add have run. That requires more than two buffers, sized by the longest skip span the engine sees.
- **Dispatch interface to spatial modules**. The spatial chain uses valid/ready handshakes on packed-pixel buses. The engine writes BRAM and signals completion. A *bridge module* between them converts BRAM-access to the streaming handshake. The bridge is itself a small Verilog module — LLM-generable, bit-exact-verifiable.
- **Backpressure across the boundary**. When the spatial chain is busy, the engine must stall before overwriting BRAM regions that have not been consumed yet. When the engine is busy, the spatial chain upstream must stall before delivering data that has nowhere to land. Implemented as conventional ready signals; the complication is making sure no deadlock is possible across the engine + scheduler + spatial chain triangle.
- **Per-layer configuration loading**. Before each heavy-layer dispatch, the scheduler loads (input/output channel counts, kernel, stride, scale, zero-point, weight DRAM base address) into the engine's config registers. This is a small AXI4-Lite write sequence per dispatch.

The scheduler itself is mechanically generated from the LayerIR graph (which layers go to the engine, what order, what activations they consume and produce). It is not LLM-generated. The author writes the scheduler generator script.

### 6.5 Skip-connection FIFOs

ResNet-50 has 16 residual adds. Each one needs a synchronisation FIFO on the skip path so the main-path activation and the skip-path activation arrive at the Add module aligned in time.

Sizing methodology, in order:

1. **Analytical first pass**. For each Add: `FIFO_depth_initial = (main_path_latency_cycles − skip_path_latency_cycles) + 1.5× backpressure_margin`. Latencies come from each module's `pipeline_latency_cycles` in LayerIR.
2. **Adjust for engine sequentialisation**. The shared engine runs heavy layers serially. If a residual block's main path contains an engine-dispatched layer, the skip-path activation may need to wait much longer than the steady-state formula suggests, because the spatial path stalls while the engine runs. Increase FIFO depth by the worst-case engine occupancy time observed for that residual block.
3. **Verify in Verilator under representative workload**. Run the full residual stage in cycle-accurate simulation with the engine actually dispatched in sequence. Confirm no deadlock, no FIFO underflow / overflow. This is what FINN's `auto_fifosize` does for the same reason.
4. **Iterate** if deadlock or overflow is observed. A failing residual-block simulation becomes a failure-corpus entry; the existing nn2rtl Surgeon loop fires one level up, indicting whichever module's handshake or FIFO depth is wrong.

The analytical formula is a starting point, not a final answer. The cycle-accurate verification step is non-negotiable.

### 6.6 Top-level wrapper

The wrapper wires:

- The ~95 spatial modules in their dataflow chain (LLM-generated from the LayerIR graph; this is a mechanical wiring exercise).
- The shared engine as one block, with its URAM weight port, BRAM activation ports, and AXI4-Lite control port.
- The 16 skip-FIFOs sized per §6.5.
- A pre-loaded URAM region holding all 22.4 MB of network weights. Initialised at bitfile load via `$readmemh`-style `.mem` files generated from the existing per-layer `_weights.hex` artefacts. No runtime memory traffic to the host.
- An AXI4-Lite slave for host control (start, image-input, status, output-collection).

The original plan had an AXI4-MM master to U250's DDR4 for weight DMA and activation staging; that block is removed. Roughly 1-2 weeks of Phase 2 engineering disappears with it, and a non-trivial source of post-route timing risk (the DDR PHY interface) goes away.

### 6.7 Weight memory layout

Concrete numbers from the LayerIR audit:

- 53 conv2d layers total. Per-layer weight footprint ranges from a few KB (early small convs) to 2.36 MB each (`node_conv_284`, `node_conv_292`, `node_conv_298` — the 512×512 3×3s) and 2.1 MB (`node_conv_288`, the 2048×1024 1×1).
- Sum of all weights: 22.4 MB INT8 + 0.1 MB INT32 bias.
- URAM block size: 288 Kbit = 36 KB. Largest single layer therefore uses ~65 URAM blocks; total network uses ~625 URAM blocks out of 1,280 available.
- Each layer's weight region has a base address known at compile time (from the LayerIR weight layout). The scheduler dispatches the engine by writing that base address into the engine's config register before each layer.

The weight memory is not LLM-generated — it is a deterministic memory-map generator script that reads the LayerIR and emits `.mem` files plus a generated Verilog header (`weight_memory_map.vh`) of base addresses. About 100 lines of Python.

**LLM contribution in Phase 2**: high but circumscribed. Foundry generates the engine sub-blocks (§6.2) and the BRAM-to-stream bridges (§6.4). Surgeon repairs them when Verilator catches a handshake bug. The wrapper and scheduler are mechanically generated from LayerIR. The engine skeleton, the BRAM banking layout, the URAM weight memory map, and the architectural decisions in §6.1 are author-designed.

## 7. Phase 3 — End-to-end empirical verification (2 weeks)

Run 50,000 ImageNet validation images through:

- The quantised PyTorch reference (the ground truth).
- The Verilator-simulated integrated nn2rtl design.

Report:

- Per-output-logit max absolute error distribution.
- Top-1 ImageNet accuracy of both, and the gap.
- Top-5 accuracy of both, and the gap.

**Tolerance is empirical, not theoretical.** Error does compound through INT8 requantisation: a 1-LSB error in layer N's output can grow when multiplied by layer N+1's weights. Theoretical worst-case bounds on composed error are loose; the honest reportable number is the observed empirical distribution.

Before Phase 3 starts, the 9 ResNet-50 modules currently passing only by tolerance (`max_error ≤ 3`) each get one improve pass aimed at strict bit-exact. Modules that won't tighten get named explicitly in the limitations section as known sources of composed end-to-end error.

**LLM contribution**: low. Existing testbench infrastructure. One network-level sidecar JSON.

## 8. Phase 4a — Timing closure on U250 (4–6 weeks)

The plan does not assume timing closes in 3–5 iterations. A 100-module design across an entire U250 die can easily take 10+ iterations of place-route-fix, each iteration multiple hours of Vivado runtime.

Iterative loop:

1. Run P&R.
2. Read `*_post_route_timing_summary.rpt`. Extract worst-slack paths.
3. The LLM (Surgeon-style call) identifies which module each worst path crosses, and proposes a targeted RTL change: retiming, fanout buffering, register replication, pipeline-stage insertion at a specific module output.
4. Re-run Verilator on the changed module to confirm it remains bit-exact.
5. Re-run Vivado P&R.
6. Repeat until WNS ≥ 0 and Fmax ≥ 100 MHz.

Some modules will need to be regenerated with deeper pipelining, which feeds back into Phase 1 numbers. That is expected, not a failure. Add a flat 20% calendar buffer here.

**LLM contribution**: central. This is the existing Surgeon loop applied to integration-level timing, not per-module functional failures.

## 9. Phase 4b — Measurement (1-2 weeks)

Supervisor has explicitly approved simulation-only as a valid deliverable. So Phase 4b is now structured around the post-route + Verilator path as the primary, with real silicon as a "if convenient" bonus.

**Primary path — post-route + Verilator**:

- All PPA numbers come from post-route Vivado reports.
- Power: `report_power` activity-aware, using SAIF captured from a representative Verilator run on real ImageNet inputs. Flag the methodology explicitly as vectorless-equivalent.
- Throughput estimate: post-route Fmax × cycles-per-frame from cycle-accurate Verilator.
- Accuracy: 50k ImageNet validation images through the Verilator-simulated design (§7).

**Optional extension — real silicon**:

- If a U250 board is available (university hardware or short-term cloud), additionally synthesise the bitfile and flash.
- Power: XRT board telemetry, per-rail sensors.
- Throughput: timed runs over the same 50k images via the host driver.
- Real-silicon numbers replace post-route estimates if both are available; otherwise the post-route numbers stand.

The thesis is defensible from the primary path alone.

PPA table reports:

- LUT, FF, DSP, BRAM18-equiv (post-route).
- Fmax achieved.
- fps end-to-end and ms latency per image.
- Power (watts).
- **GOPS/W** (the headline efficiency metric).
- Top-1 ImageNet accuracy.

## 10. Intermediate deliverables (de-risking; parallel to the full attempt)

The full integration is high-risk. Two intermediate artefacts give a thesis result regardless of how the final integration goes:

- **Mini-deliverable A — stem + stage 1 on U250** (17 layers, no shared engine yet). All-spatial deployment of conv1 + the first residual stage of ResNet-50, targeted at the same `xcu250-figd2104-2L-e` part as the final design. Achievable by end of Phase 1. Validates the spatial wiring, the residual-add FIFO sizing on a small example, and the LayerIR-to-wrapper generator on a small graph. This deliverable does not require the shared engine to work.
- **Mini-deliverable B — stem + stage 1 + stage 2 with engine on U250** (32 layers, engine handles stage 2's heavies). Achievable by end of Phase 2. Validates the full hybrid architecture on a manageable scope: small enough to debug, large enough to exercise every component (spatial chain, shared engine, scheduler, skip FIFOs, AXI shells). Functionally equivalent to the full design's Phase 2 build but with 32 layers instead of 119.

Both deliverables target the same U250 part as the final design, so timing-closure techniques, P&R settings, and area accounting are directly reusable when scaling to the full 119-layer build.

Both are first-class deliverables. They are real measured artefacts that pin down PPA numbers and the integration methodology even if the full 119-layer integration does not close timing.

## 11. Success criterion (pre-committed)

Design is called successful if it meets *all*:

- ≥ 10 fps end-to-end throughput on U250.
- Top-1 ImageNet accuracy within 1.0 percentage points of the quantised PyTorch reference.
- Total LUT post-route ≤ 95% of U250 budget (≤ 1.64 M LUTs).
- Fmax post-route ≥ 100 MHz.
- GOPS/W within 0.3× of the best published Alveo U250 INT8 ResNet-50 result. **The published baseline value is currently TBD** — the plan commits to pinning down a primary source for this number before Phase 4 starts. Two specific verifications are needed:
  - The Vitis AI DPU (DPUCADF8H on U250) published GOPS/W for ResNet-50 INT8 from a Vitis AI 2.5 release note or equivalent primary AMD source.
  - Any earlier FINN paper figure must be checked for precision — FINN's headline U250 ResNet-50 result is W1A2 (binary weights, 2-bit activations), which trades roughly 7 percentage points of top-1 accuracy for the higher throughput. W1A2 versus INT8 is not a like-for-like comparison and must not be used as the floor.

Defining these before Phase 4 prevents the trap of finishing the work and arguing about whether the numbers count.

## 12. Comparison strategy (pre-committed)

**Whole-network external baseline: Vitis AI 2.5 DPU on U250 (DPUCADF8H).** Reasons:

- AMD publishes detailed DPU PPA + GOPS/W on U250 for ResNet-50 INT8.
- It is the production tool an industrial reviewer will care about.
- Same chip, same precision, same network — no normalisation games.

**Important caveat that the plan does not hide**: Vitis AI 3.0 and later have discontinued U250 support. The DPUCADF8H baseline is from Vitis AI 2.5 (the last U250-supported release), and AMD has flagged the entire U200 / U250 / U280 line for broader deprecation starting with the 2025.2 toolchain. The comparison is still defensible — the baseline numbers are real and published — but it is against a legacy production target, not a current one. The thesis text should state this explicitly. An industrial reviewer will know about the deprecation; not naming it would read as unawareness.

**Layer-level external baselines**: the existing nn2rtl vs FINN vs hls4ml three-way per-layer comparison in `comparison/tier_a/compare_three_way.csv` stays as supplementary evidence. It demonstrates the layer-level PPA position. The DPU comparison is the whole-network one.

Quoting FINN or hls4ml *whole-network* numbers from their papers is weaker because those papers chose different chips and configurations. In particular, FINN's headline U250 ResNet-50 result is W1A2 — binary weights, 2-bit activations — not INT8, and at ~68.9% top-1 accuracy instead of the ~76% typical of INT8. Comparing nn2rtl INT8 against FINN W1A2 on throughput would be apples-to-oranges and the thesis must not do that. Per-layer comparisons are apples-to-apples in the current data because both tools were run at the same precision; whole-network FINN figures need a precision-matched re-run before they enter the comparison.

## 13. Contribution claim (headline)

> "An LLM-agent pipeline can produce per-layer RTL that is bit-exact-verified, integrates into a working hybrid whole-network FPGA design on Alveo U250, and reaches PPA within a defined envelope of an established production baseline (Vitis AI DPU). The improve loop achieves layer-level area compression approaching hand-optimised quality on individual layers. The failure corpus produced during the process is a research artefact that improves subsequent generations on related networks."

The comparative numbers are *evidence* for this claim. The claim is independent of whether nn2rtl beats DPU on any specific metric. The methodology, the verified artefacts, and the failure corpus are what neither DPU nor FINN nor hls4ml produces in the same form.

## 14. Risks and known unknowns

Named, not buried.

- **Phase 1 compression may underperform.** Planning assumption is −30% heavy / −15% medium. If actual is closer to −15% / −5%, the spatial side stays bulky and the integration margin shrinks. Mitigation: the engine handles the heavy modules regardless, so spatial under-compression mainly shrinks the LUT headroom for the wrapper, not the headline architecture.
- **Engine design parameters in §6.1 are tentative commitments.** MAC count, parallelism axis, requant depth — each will be revisited after the first engine synthesis. The plan does not pretend these are settled.
- **Engine LLM-generation may produce broken sub-blocks.** Option 1 (skeleton + sub-blocks) is the most realistic LLM contribution, but the sub-blocks (MAC array, requant pipeline, AXI4 interface) are each individually large enough that Foundry may need multiple attempts and Surgeon repair. This is the existing nn2rtl loop and is expected to work, but iteration count is unknown.
- **Skip-FIFO sizing under engine sequentialisation may require buffers larger than analytical formula predicts.** Mitigation: cycle-accurate Verilator simulation of each residual block. If a residual block fails to converge on a stable FIFO depth, that block is a candidate for refactoring (e.g. running both its main and skip paths through the engine sequentially, eliminating the parallel-path FIFO entirely).
- **Phase 4a timing closure may spiral.** A 100-module design on U250 is a large target. Calendar estimate is 4–6 weeks; could realistically be 8 weeks if congestion forces multiple modules to be re-pipelined. Mitigation: the intermediate deliverable B (stem + stage 1 + stage 2) closes timing on a 32-module scope first, surfacing the worst paths early.
- **Hardware procurement is a separate question from the plan.** The plan assumes "U250 available" but is structured so that Phase 4b can be replaced by post-route-only measurement without invalidating any earlier phase. AWS F1 instances use VU9P silicon, which is the U200, not the U250 — F1 is therefore not a drop-in cloud replacement for the planned target.
- **Vitis AI DPU baseline GOPS/W needs verification.** The success-criterion floor depends on this number. To be pinned down before Phase 4 using a primary AMD source (Vitis AI 2.5 release notes or equivalent).
- **Vitis AI on U250 is a deprecated toolchain target.** Vitis AI 3.0+ does not support U250; the baseline comparison is therefore against Vitis AI 2.5. The comparison is still valid (the numbers are published and real) but the thesis text must name the deprecation status, not hide it.
- **Precision-matched comparison.** FINN's headline U250 ResNet-50 numbers are W1A2, not INT8. Any FINN-vs-nn2rtl throughput comparison at the whole-network level requires a precision-matched FINN re-run that the current `comparison/results/finn/` data does not contain. Until then, FINN comparisons stay layer-level only.
- **BRAM unit ambiguity.** Vendor literature switches between BRAM18 and BRAM36 without always saying which. This plan quotes BRAM36 counts to match the Xilinx tool reports (`report_utilization` defaults). BRAM18-equivalent is `2 × BRAM36` when needed for cross-document comparison. To be confirmed once with `report_property [get_parts xcu250-figd2104-2L-e]` inside Vivado before any number is locked into the thesis text.
- **URAM utilisation may exceed 50% if assumptions are wrong.** The 22.4 MB weight footprint is an INT8 count from LayerIR; if the deployment requires extra metadata bytes (e.g. per-layer scale tables, alignment padding to URAM word boundaries), real URAM usage could climb. Mitigation: per-layer URAM usage will be computed from the deterministic memory-map script (§6.7) before Phase 2 starts; if total exceeds ~80% URAM, revisit whether a small subset of large layers stays in BRAM-distributed rather than URAM.
- **No DDR fallback.** Supervisor constraint is on-chip-only. If a layer's weights cannot fit URAM at acceptable utilisation, the plan does not have a "spill to DDR" escape — that layer must be handled either by sharper compression (move to Phase 1 improve loop) or by accepting it as a known limitation. Worth naming this risk explicitly; the original DDR-streaming plan had elasticity that this plan does not.

## 15. Timeline + cost

| Phase | Calendar | Notes |
| --- | --- | --- |
| 0 — Re-baseline on U250 | 1 week | Mechanical re-runs |
| 1 — Targeted compression | 3–4 weeks | LLM-driven improve sweep |
| 2 — Engine + on-chip integration | 4–5 weeks | The new core work; reduced from 5–6 weeks because the DDR controller / AXI4-MM weight DMA is now removed |
| 3 — Empirical end-to-end | 2 weeks | 50k ImageNet validation |
| 4a — Timing closure | 4–6 weeks | LLM-driven iterative repair |
| 4b — Measurement | 1–2 weeks | Post-route + Verilator primary; silicon optional |
| **Total** | **15–20 weeks** | Roughly 1 week saved by dropping DDR |

Calendar buffer for unexpected debug (skip-FIFO deadlocks, engine sub-block regressions, timing closure spirals): assume the upper bound.

LLM API cost estimate, based on existing per-module rates:

- Phase 1 improve sweep: ~$80–150 (10-20 targeted compressions at $5–15 each).
- Phase 2 engine sub-blocks: ~$40–80 (4–5 sub-blocks; the AXI4-MM weight DMA sub-block is no longer needed since weights are on-chip URAM).
- Phase 2 wrapper + bridges: ~$20–40.
- Phase 4a timing closure: ~$50–150 (variable; depends on iteration count).
- **Total LLM cost: ~$190–420.** Comfortably under the supervisor-approved €300 budget once euro/dollar conversion is applied (~€175–390).

## 16. Reusability — what carries over to other networks and other FPGAs

This plan targets ResNet-50 on Alveo U250, but the whole point of nn2rtl is that the methodology should not be ResNet-50-on-U250-specific. The plan should be read as one instance of a process that retargets to other networks and other chips with bounded incremental work. Where it isn't, that is a fixable system shortfall, not a fixed architectural property — and the plan names where the leakage is so the system can be tightened over time.

### 16.1 Retargeting to another network (e.g. MobileNetV2 → U250)

What carries over unchanged:

- Phase 0 re-baseline scripts. Just point them at the other network's `output/` tree.
- Phase 1 improve loop. The improve targets (`reduce-lut`, `use-bram`, `reduce-ff`, `use-dsp`, `improve-fmax`, `reduce-latency`, `increase-throughput`) are network-agnostic. The deterministic target checker is network-agnostic. The Verilator harness reuses the network's own goldens.
- Phase 3 empirical verification. Different `.goldin` / `.goldout`, same testbench.
- Phase 4a timing-closure repair loop. The Surgeon-style call reads the post-route timing report and proposes RTL changes; nothing in that loop knows what network the modules came from.
- Phase 4b deployment. Same XRT / post-route methodology.
- Failure corpus, signature ladder, lifecycle tiers. These structures are network-agnostic and the existing MobileNetV2 entries already prove they work cross-network.

What needs new design work:

- Phase 2 shared-engine geometry. The engine's MAC array shape, parallelism axis, and requantisation pipeline are sized for whichever network's heavy modules will dispatch through it. MobileNetV2's heavy modules are different from ResNet-50's: the big 1×1 expand / project convs are similar in shape so the existing engine geometry probably maps over, but depthwise convolutions need a different compute pattern (no cross-channel reduction). MobileNetV2 deployment therefore needs either a second engine variant for depthwise, or accepts that depthwise layers stay spatial despite their high LUT cost.
- Per-network scheduler. The scheduler is mechanically generated from each network's LayerIR graph, so it is "easy" in the sense that no human writes it for each network — but a new graph is a new scheduler.
- Skip-connection topology. MobileNetV2 has 10 residual adds versus ResNet-50's 16, and the inverted-residual block layout is different. Skip FIFO sizes are re-derived from the new latency data; the methodology in §6.5 carries over verbatim.

Honest effort estimate, for MobileNetV2 on the same U250 after ResNet-50 is finished: 10-13 calendar weeks instead of 16-21. Phases 0, 1, 3, 4a, 4b carry over for free; Phase 2 takes ~70% of its ResNet-50 cost because the architectural pattern is the same but the engine geometry needs reassessment.

The system makes this easier than starting from scratch because:

- Per-module bit-exact verification means retargeting can be debugged one module at a time.
- The auto-promoted reference docs from earlier networks remain available; an engine designed for ResNet-50's 1×1 convs is a useful starting point for MobileNetV2's expand / project convs even when the parameters change.

### 16.2 Retargeting to another FPGA in the same vendor family

Hardest of these is the smallest in calendar terms.

| Move | Effort | Why |
| --- | --- | --- |
| U250 → bigger UltraScale+ (VP1802, U280) | days–1 week | Change `--part`, re-run, sensitivity table reshuffles. Architecture decisions may relax (full-spatial may even become feasible on VP1802). |
| U250 → smaller UltraScale+ (U200, ZCU102) | 2–4 weeks | More aggressive compression and/or more layers move into the shared engine. Plan structure unchanged. Sensitivity table re-derived. |
| UltraScale+ → Versal AI Core (VCK190 etc.) | 6–10 weeks | The AI Engines change the deployment model — they are hardened tensor units, not LUT/DSP fabric. Whether nn2rtl-RTL belongs at all on VCK190 is itself a research question. |

For Xilinx-to-Xilinx in the same family, the system is almost-modular by construction. The contracts and pattern docs encode UltraScale+ assumptions but those assumptions hold across U200/U250/U280/VCU118/VP1802.

### 16.3 Retargeting to a different vendor (Intel Agilex / Stratix 10)

This is the case where the architecture leaks Xilinx assumptions. Honest accounting:

What stays:

- LayerIR, goldens, the testbench, the improve loop targets, the failure-corpus structure, the lifecycle tiers, the signature ladder.
- The RTL bodies are largely portable synthesisable Verilog.
- The verification methodology (Verilator first, vendor synth second) is FPGA-vendor-agnostic.

What needs vendor-specific work:

- Synthesis attributes. Patterns reference `(* use_dsp = "yes" *)` and `(* ram_style = "block" *)`. Intel Quartus uses `multstyle` and `ramstyle = "M20K"`. About fifty references in the protected pattern docs would need a parallel Intel variant.
- The `dram-backed-weights` contract uses AXI4. Intel native is Avalon-MM, and while AXI4-to-Avalon bridges exist, the contract is currently AXI4-shaped.
- The failure-corpus knowledge is Vivado-specific. Preflight gates like `activation_memory_in_async_reset_block` exist because Vivado specifically refuses BRAM inference for async-reset memories; Quartus has different rules and a different set of synthesis failure modes. The corpus would need new entries for Intel failure patterns.
- The `run_vivado` MCP tool. Needs a `run_quartus` sibling. The orchestrator dispatching it is vendor-agnostic already.
- The `vivado_resynth_failed.ts` script needs a Quartus counterpart.

Honest effort estimate: 8-12 calendar weeks to make the system Intel-capable, plus the per-network deployment cost on top. That sounds large because it is — but it is a one-time refactor that makes every subsequent Intel target cheap.

### 16.4 What I would change in the system to improve reusability

This is the candid list, in priority order:

1. **Split contracts into "logical" and "vendor-specific" layers.** Each contract is currently one `metadata.json` plus a Verilog testbench template plus a Python golden generator. The Verilog template and the synthesis-attribute conventions inside it are Xilinx-coded. A clean split would put the logical interface (ports, protocol, supported ops, fit constraints) in one file and the vendor-specific synthesis hints (`ram_style`, `use_dsp`, AXI vs Avalon) in a sibling file. About 2 weeks of refactor; makes every future Intel or ASIC target cheap.
2. **Vendor-tag the failure corpus.** Each failure entry currently records `network id, module id, signature data, contraindications`. Adding `vendor: xilinx` and `toolchain: vivado-2024.x` would let the corpus serve cross-vendor work — Intel failures and Vivado failures stay separately retrievable.
3. **Promote the FPGA target to a first-class config object.** Today the part name is a flag passed to `run_vivado`. A first-class target config (with budgets per resource, vendor, toolchain, attribute syntax) would let the orchestrator make compression decisions automatically against that target.
4. **A `network_template.md` per network.** The MobileNetV2 retarget would have been faster if the design choices that *depend on the network* (which heavy modules go to the engine, how skip topology composes) were captured in a per-network template rather than in the deployment-plan prose for that network. Each new network would fill in its own template.

These are improvements that the thesis writeup can frame as the *next* development direction. The deployment for ResNet-50 itself does not need them.

### 16.5 The `dram-backed-weights` contract under the on-chip-only constraint

This deployment does not use external DDR (supervisor constraint, §3). That makes the existing `dram-backed-weights` contract a research artefact rather than the deployment mechanism: the contract still exists in the codebase, has its own pattern doc, has a passing reference (`node_conv_288`), and remains useful for any future deployment that *does* want DDR — for example, retargeting to a network too large to fit U250's on-chip memory (a hypothetical ResNet-152 or YOLO-style network where weights exceed ~57 MB).

Concretely:

- The `dram-backed-weights` contract metadata and pattern doc stay in `contracts/dram-backed-weights/` and `knowledge/patterns/protected/09_dram_backed_weights.md`. They are not removed.
- The `node_conv_288` reference remains in the lifecycle registry as the active reference for that contract. Other networks (e.g. a future MobileNetV3 with bigger weights) can still adopt it.
- For the ResNet-50 U250 deployment specifically, the engine uses an on-chip URAM weight port instead. A separate "on-chip-weights" contract variant may be promoted to a first-class artefact if it stabilises during Phase 2.

This is consistent with the thesis framing that contracts are reusable abstractions, not deployment-specific glue.

### 16.6 Summary

The methodology is universal. The toolchain integration is Xilinx-specific by accident, not by design. Retargeting to a new network is easier than retargeting to a new vendor. Retargeting to a bigger chip in the same family is essentially free. The biggest single reusability gain available is the contract-metadata split (item 1 above); the rest is incremental work that the thesis can name as future direction.

## 17. What this plan is not

It is not an architectural-novelty claim. The hybrid architecture is close to what Vitis AI DPU already does. The novelty is in the methodology: every module is bit-exact verified, the engine sub-blocks are LLM-generated and verified, the failure corpus accumulates across the project, and the whole pipeline is reproducible from the LayerIR.

It is not a "we beat DPU on every metric" claim. The success criterion deliberately allows being within 0.3× of DPU's GOPS/W, not beating it. The thesis result is that the methodology produces a working design within an envelope of an industrial baseline, not that it dominates.

It is not contingent on hardware procurement. The thesis is defensible from post-route Vivado measurements alone. Real silicon adds credibility, not the claim.
