# Thesis Source Map

Master's thesis: **Agentic LLM-driven neural-network-to-RTL generation with INT4/INT8 quantization, bit-exact (byte-exact) verification, and FPGA deployment (Xilinx Alveo U250) of ResNet-50 and MobileNetV2** — covering accuracy, resource fit, place-and-route, and Fmax/timing.

---

## 1. Overview

This document is the navigation map for the entire research corpus behind the thesis. The corpus consists of **~75 structured findings/diary/report/plan/spec documents** spanning two sources: (1) the persistent project-memory notes under `~/.claude/projects/.../memory/` (root-cause findings, status reports, plan-roadmaps, memory-lessons, agent-specs), and (2) the in-repo engineering docs under `nn2rtl-repo/` (`README.md`, `ARCHITECTURE.md`, `MILESTONES.md`, the `docs/` deployment plans, and the `docs/agent_tasks/` wave/task specs and root-cause logs). Behind these sit large raw-evidence buckets — ~1689 Claude session `*.jsonl` transcripts and in-repo `agent_tool_use.jsonl` / `failure_corpus` / Vivado `*.rpt` logs — which serve as the appendix.

**How to use it for the thesis.** Each chapter of the thesis maps to a small set of primary documents (see §6). Use §2 to find every document relevant to a theme with its abstract and headline numbers; use §3 as the flat index to locate any document by path; use §4 as the single consolidated table of thesis-grade quantitative results (every number is attributed to a source doc, so claims are citable); use §5 to point readers at the raw appendix material without inlining it. The corpus is faithful to measured results: where a number was an *estimate* that was later refuted by real synthesis (e.g. the "1960 BRAM36 / 72.9% fits" projection vs the first real synth's 174% BRAM), both are recorded so the thesis can present the honest-reporting arc.

> **Note on paths.** Repo-relative links resolve from `nn2rtl-repo/`. Project-memory notes live outside the repo at `~/.claude/projects/c--Users-User-Desktop-RTL-LLM-CLAUDE/memory/` and are shown with that absolute prefix abbreviated as `{memory}/`.

---

## 2. Documents grouped by thesis theme

A document is given its full abstract under its **most relevant** theme and cross-referenced elsewhere.

### (A) Project scope & architecture

**[`README.md`](../README.md)** — *Canonical design spec, 2026-04-29.*
The canonical nn2rtl design specification (University of Twente thesis). Describes an autonomous multi-agent system that compiles a quantized PyTorch ResNet-50 (stem + 16 residual blocks) into synthesizable INT8 Verilog. Three-layer system: Claude Code plugin (agent roles), TypeScript orchestrator on `@anthropic-ai/claude-agent-sdk` (deterministic PipelineStateManager), local MCP server (5 tools). Weights never passed to LLMs (written to `.hex`, loaded via `$readmemh`; goldens in NN2V binary sidecars). All modules fully pipelined with exact latency contracts.
- Latency contract: 1×1 conv = 3, 3×3 conv = 5, folded-BN = 2, ReLU = 1, residual add = OC+3 cycles
- Pass thresholds: max_error ≤ 1 expected / ≤ 3 accept; INT8 symmetric per-tensor (scale = max|w|/127, zero_point = 0)
- Wide-bus packed-channel interface: data_in = IC×8 (conv/relu) or IC×16 (add), data_out = OC×8
- Baseline target 50 MHz on ZCU102 xczu9eg (274,080 LUT / 548,160 FF / 2,520 DSP48E2 / 1,824 BRAM18 / no UltraRAM)
- Research goal: LLMs automate NN→RTL end-to-end at production scale (50,000+ lines); out of scope: GAP, FC, SoC, ASIC

**[`ARCHITECTURE.md`](../ARCHITECTURE.md)** — *File-by-file code tour, 2026-04-29 (+ 2026-04-26 Vivado migration log).*
782-line file-level reference complementing the README. Maps the three layers (plugin agents, `sdk/` deterministic orchestrator, `mcp/` tool server) to code; documents Zod validation at every JSON trust boundary, the deterministic Assayer (`runAssayerDeterministic`: sidecar write + iverilog lint + Verilator sim, no LLM), PPA gates, failure classifier, and Retrospector. Includes the DSP-inference saga and the documented reason RTL-from-scratch generation does not scale to deep spatial convs.
- Pipeline run #3: max_error = 0 across all 6,422,528 samples, first_mismatch_index = −1, $0.62 first-shot (12× cheaper than broken run #1)
- Run #2 PPA: LUT 1790, FF 1431, DSP 0, BRAM18 0, WNS 9.71 ns, Fmax 97.15 MHz
- Registered-mul_q DSP refactor: pass cycles MP·K_TOTAL+4 → +6; layer1_0_conv1 4161 → 4193 cycles (1 DSP vs 0)
- Spatial-conv scaling bottleneck: layer4_0_conv2 = 2,359,296 INT8 weights = 18.9 Mbit; bus-width gate MAX_SUPPORTED_BUS_BITS = 4096

**[`CLAUDE.md`](../CLAUDE.md)** — *Operational rule sheet.*
Session-start operational rules. Never write `.v` directly (persist via `write_verilog` MCP tool); the static Verilator TB is handwritten infrastructure, never agent-generated; verification is deterministic (no LLM Assayer).
- 3 LLM agents only: cartographer (sonnet-4-6), foundry (opus-4-7), surgeon (opus-4-7)
- 6 MCP tools (run_iverilog/verilator/vivado, read_weights, write_verilog, get_rtl_patterns); models pinned to full IDs in `sdk/config.ts`

**[`rtl_library/SPLIT_ARCHITECTURE.md`](../../rtl_library/SPLIT_ARCHITECTURE.md)** — *Split-architecture contract.*
Pins the contract that splits monolithic spatial-conv/maxpool generation into a thin LLM-generated wiring wrapper over three handwritten library modules; explains why LLM-generated cycle-aligned spatial RTL fails and why the split is the honest scaling path.

**[`{memory}/project_onnx_frontend.md`]** — *Universal ONNX frontend, April 2026.*
`scripts/onnx_frontend.py` is the universal pipeline frontend replacing the ResNet-50-specific FX path; reads the ONNX graph directly, fixing channel-indexing/MaxPool/topology bugs. Supports Conv2d (BN folded by onnxsim), ReLU, Add, MaxPool2d.
- Tool versions: onnx 1.21.0, onnxsim 0.6.2, onnxruntime 1.23.0; `torch.onnx.export` requires opset 18

**[`docs/nn2rtl_supervisor_explanation.md`](nn2rtl_supervisor_explanation.md)** — *Full results dump for a supervisor.*
Plain-language end-to-end explanation carrying the most complete measured results for **both** networks (cross-referenced heavily in (I)). Multi-network layout: ResNet-50 at `output/`, MobileNetV2 at `output/mobilenet-v2/`. Defines bit-exact carefully.
- ResNet-50: 119/119 pass, 1.5702 fps, $170.61 ($1.43/module); median module 5687 LUT; max node_conv_296 188,568 LUT; median Fmax 313.8 MHz
- MobileNetV2: 97/99 pass, 10.1424 fps, $196.39 ($2.02/module); max node_conv_818 depthwise 336,522 LUT; median Fmax 268.9 MHz
- Three-way avg (8 layers): nn2rtl 3,992 LUT vs hls4ml 32,174 vs FINN 49,079 (~8×/12× fewer); Fmax 339 vs 71 vs 109 MHz; but fps 9.50 vs 1,407 vs 183

Cross-refs: contract/skeleton specs in (C); engine architecture in (C)/(E); deployment plans in (E).

### (B) Quantization & accuracy (INT4/INT8, GPTQ, per-channel vs per-tensor)

**[`docs/agent_tasks/int4_imagenet_FINAL_REPORT.md`](agent_tasks/int4_imagenet_FINAL_REPORT.md)** — *Final ResNet INT4 report, 2026-05-30.*
Final report on INT4-weight/INT8-activation per-channel-GPTQ ResNet-50 → RTL: correctness and accuracy DONE and byte-exact, on-chip fit NOT confirmed (first full synth OVER capacity). The multi-session e2e bug root-caused to 22/48 ReLU nodes missing activation rescale + node_add_7 swapped operand halves.
- relu_48 final output: 0.00% mismatch, ImageNet class 91 == golden, feature cosine 1.000000 → RTL **is** the INT4-GPTQ reference
- **Accuracy 79.47% top-1** (per-channel GPTQ, 1500-image eval; float 80.07%, INT8 ~75%); per-tensor GPTQ unusable at 2.80%
- First full synth OVER: RAMB36 4663 (174%), LUT 1,983,938 (115%, incl 434K LUT-as-dist-RAM), URAM 203 (16%), DSP 7429 (60%), FF 1.31M (38%)
- Throughput 13,348,787 cyc/frame = 15.0 fps @200MHz (meets 10 fps)

**[`{memory}/project_int4_accuracy_and_deployment_gap.md`]** — *INT4 accuracy sound; live weights already INT4, 2026-05-29.*
INT4-GPTQ accuracy is sound and deployable; corrects an earlier "deployed weights are INT8" claim — everything live is already INT4; the 8 `_wide` INT8 files are instantiated but DEAD. Fit is packing-only.
- INT4 GPTQ weight-only 77.73%; W4A8 (Scheme A′) 77.54%; ~float 78.52%; QMAX/QMIN = 7/−8
- `val_int4.log` 0.20% is a broken naive-RTN red herring
- Proof `_wide` dead: conv_286 golden recompute 0% vs INT4 bank, 99.15% vs `_wide` INT8

**[`{memory}/project_int4_fit_analysis.md`]** — *INT4 fit + precision analysis + Phase-2 per-OC rework, 2026-05-28→30.*
Decisive evidence that per-output-channel GPTQ is **required** for usable INT4 (per-tensor INT4 = chance-level), plus the bit-budget math. (Fit side cross-referenced in (E).)
- Weights 187.6 Mbit INT8 → 93.8 Mbit INT4 (spatial 53.4 + engine 40.4); biases 0.85 Mbit
- INT8 full-pipeline 75.20% top-1 / 93.16% top-5; INT4-per-tensor 0.20% (chance)
- Weight-only: float 81.05 / INT8-tensor 80.47 / INT8-chan 80.86 / INT4-tensor 0.00 / INT4-chan 32.42
- GPTQ per-ch w-only 77.7% / +INT8 act 77.5%; GPTQ per-TENSOR 39% w-only, 1.8% W4A8 (fails)

**[`{memory}/feedback_accuracy_measure_bn_folded.md`]** — *Accuracy measurement must be BN-fold-aware, 2026-05-31.*
PROVEN (corr = 1.0000): `resnet50_full.onnx` = 53 Conv / 0 BatchNorm (BN folded). The deployed per-OC weight scale is correct/self-consistent; injecting dequant weights into a torchvision resnet50 with live BN double-counts BN and collapses top-1 to ~0% or a misleading ~73% — both artifacts.
- weight_scale_per_oc = max_abs(W_FOLDED[oc])/qmax; qmax = 7 (INT4), 3 (INT3); differs from raw scale by exactly per-channel BN factor
- Config B reported 77.60% via `gptq_buildable_configs.py` with NN2RTL_WEIGHT_BITS=4, NN2RTL_IMAGENET_CALIB=256, 18 INT3 layers

**[`{memory}/project_deploy_vs_measure_calibration.md`]** — *Deploy vs measure paths diverge, 2026-05-30→31.*
Two distinct quantization paths produce DIFFERENT weights: deployment (`generate_golden`, synthetic calib default 8 samples) vs accuracy-measurement (`gptq_int4.py`, real ImageNet 256 calib).
- 79.47% INT4 / 77.60% Config-B from the real-ImageNet 256-calib path
- conv_284: 1.9M of 2.36M weight hex lines differ between default-synthetic and deployed
- Deployment env knobs: NN2RTL_WEIGHT_BITS=4 + NN2RTL_IMAGENET_CALIB=256

**[`docs/agent_tasks/08_engine_requant_pipeline.md`](agent_tasks/08_engine_requant_pipeline.md)** — *Engine requant arithmetic spec.*
The exact INT8 requantisation arithmetic (round-half-up, saturate, per-OC scale) that must be byte-identical to the spatial goldens.
- Algorithm: biased = acc + bias; scaled = round_half_up(biased × scale_mult >> scale_shift); clamp [−128,127]; round-half-up = add 1<<(scale_shift−1)
- Verification: max_error 0 on ≥1,000 random INT32; bit-exact vs node_conv_298 requant tail; standalone Fmax ≥300 MHz

**[`docs/agent_tasks/phase2_int4_per_oc_plan.md`](agent_tasks/phase2_int4_per_oc_plan.md)** — *INT4 nibble-pack + per-OC requant plan, 2026-05-28.*
Two orthogonal Phase-2 changes: Change B (nibble-pack weight width 8→4) and Change A (per-output-channel requant, 256 per-OC (mult,shift)). Nibble-packing deferred to Phase 3 (byte-transparent); Phase 2's only correctness change is per-OC requant.
- e2e in-chain: per-tensor engine 11793 mismatch → per-OC engine 8464 (mean|err| 35)
- Scheme A blast radius ~9 shared files; Scheme B (full INT4) ~60+ files

**[`scripts/PLAN_E_pointwise_per_oc.md`](../scripts/PLAN_E_pointwise_per_oc.md)** — *MBV2 pointwise per-OC plan, 2026-06-09.*
Plan-only: per-OC requant for the MBV2 1×1 engine; verdict that the engine RTL is already per-OC (the easy path). (See (J) for the executed outcome — broke 2/8 vecs, deferred.)

**[`scripts/PLAN_B_golden_hardening.md`](../scripts/PLAN_B_golden_hardening.md)** — *MBV2 golden hardening, design-only.*
Hardening the MobileNetV2 golden requant to exact integer fixed-point (eliminating the FLOAT round_half_up tie risk that bit vec3/vec6).

Cross-refs: per-OC requant accuracy on MBV2 in (I); BN-fold validation underpins all accuracy claims.

### (C) RTL generation methodology & agentic tooling

**[`SYSTEM_REVIEW_FINDINGS.md`](../SYSTEM_REVIEW_FINDINGS.md)** — *Static system review, 2026-05-02.*
The biggest problems are feedback-loop hazards where the system learns from the wrong signal, not single RTL bugs. Root finding: Windows iverilog crashes (exit 3221225794 = 0xC0000002) misclassified as RTL bugs, burning Foundry/Surgeon calls.
- Contract registry advertised 5 contracts but only 3 executable (now all 5)
- LayerIR lacked groups/dilation/full padding → blocked depthwise/MobileNet
- Fix pass added `toolchain_infra` class, executable contract plans, contract-filtered knowledge lookup

**[`PROJECT_AUDIT.md`](../PROJECT_AUDIT.md)** — *Code-quality audit, 2026-04-15.*
Audit log enumerating ~32 fixes applied in one pass plus deferred items (orchestrator hardening: signed-output handling, atomic state, toolchain-infra vs RTL-bug distinction).
- yosys LUT regex now sums any `*LUT*` row (was LUT4-only); add-module packing contract asserted at LayerIR load (input_width = 2 × output_width)

**[`knowledge/IMPLEMENTATION_PLAN.md`](../knowledge/IMPLEMENTATION_PLAN.md)** — *Pattern-library / self-improvement plan.*
The pattern-library MCP tool implementation plan for Foundry/Surgeon (self-improvement Phases 1–3: classifier, Retrospector). Core architecture-of-the-agent-system doc.

**[`{memory}/project_failure_corpus_works.md`]** — *Failure-corpus convergence, 2026-05-04.*
The visible failure corpus + scored entries + `get_failure_corpus` tool is a working convergence mechanism. Backfilling 11 prior node_conv_288 attempts collapsed latency error from a +480..+3588 range to +1 cycle.
- status_class shifted sim_completed_mismatch → sim_stalled (genuinely different design tried)

**[`{memory}/feedback_universal_diagnostics.md`]** — *Universal diagnostics over hardcoded hints, 2026-04-17.*
The testbench/orchestrator must emit raw facts (cycle counts, missing-index ranges, histograms, first-mismatch values) and let the model diagnose — NOT pre-written hypotheses. A "likely missing drain" hint sent Surgeon to add MORE drain when the real bug was the existing drain's off-by-one counter.

**[`{memory}/feedback_atomic_arch_changes.md`]** — *Atomic architecture changes, 2026-04-26.*
RTL + `compute_conv2d_latency_cycles` + pattern docs + preflight regexes + regenerated goldens must change in the same commit; splitting risks LayerIR claiming one latency while RTL produces another.

**[`{memory}/feedback_no_reference_for_new_contracts.md`]** — *No reference Verilog for new contracts, 2026-05.*
For a never-seen contract (depthwise, GAP, Gemm) ship NO hand-authored reference Verilog — the experimental control that measures whether Foundry can build correct RTL from pattern doc + LayerIR + prompt alone.

**[`{memory}/feedback_self_improve_required_for_contract_walk.md`]** — *self_improve gates the contract walker, 2026-05-04.*
For a layer without an explicit contract_id, `NN2RTL_SELF_IMPROVE=1` must be on or the contract walker is skipped and the layer dies at the flat-bus gate (`architectural_unsupported`).

**[`{memory}/project_top_v_is_patched_not_regenerated.md`]** — *Top is patched, not regenerated, 2026-05-28.*
`output/rtl/nn2rtl_top.v` is generated THEN hand-patched (base ~113 handshake markers → working ~1205). A blind re-run of `build_top_wrapper.ts` reverted it to base and deadlocked the e2e (out = 0/3136, 50M-cycle timeout).

**[`{memory}/feedback_fix_everything_no_defer.md`]** — *Fix everything means everything, 2026-05-29.*
Byte-exact e2e correctness GATES all fit/perf/Vivado work; a Vivado synth on a values-broken design is a 5-hour measurement of an unwanted build. "Slow to localize" ≠ "skip".

**[`{memory}/feedback_autonomous_night_directive.md`]** — *Autonomous overnight directive, 2026-05-29.*
The operator's autonomous-overnight working contract: finish the plan, make it FIT and run Vivado, performance priority, only halt for a genuine fork with no recommended option; keep a background task pending so the session never idles.

**[`{memory}/project_overnight_chain_20260606.md`]** — *Standing overnight task chain, 2026-06-06.*
ResNet route_only → MobileNet synth (90% RAM watchdog) → fit-fix reshapes → ResNet Fmax RTL (no Vivado). Documents Vivado-serialization discipline and the BRAM-friendly reshape method.

**The Wave/Task agent-spec suite (`docs/agent_tasks/`)** — the orchestrated multi-agent build of the U250 shared engine:
- **[`docs/agent_tasks/README.md`](agent_tasks/README.md)** — Parallel agent-task dispatch (Waves 1–4) gated by a Wave-1 review gate on the engine skeleton.
- **[`docs/agent_tasks/00_engine_skeleton_spec.md`](agent_tasks/00_engine_skeleton_spec.md)** — Human-written engine skeleton (the linchpin); 256 MACs OC-parallel, 3-stage requant, URAM weights, AXI4-Lite control. Largest layer node_conv_298 = 2.36 MB.
- **[`docs/agent_tasks/00_engine_skeleton_spec_FSM.md`](agent_tasks/00_engine_skeleton_spec_FSM.md)** — 6-state, 3-bit FSM (IDLE/LOAD_CONFIG/RUN/REQUANT/DRAIN/DONE); mac_clear timing; OC-pass loop.
- **[`docs/agent_tasks/00_engine_skeleton_spec_PORTS.md`](agent_tasks/00_engine_skeleton_spec_PORTS.md)** — Port single-source-of-truth; 2048-bit act buses, 22-bit URAM addr, 8192-bit bias/acc, 14-entry config register map.
- **[`docs/agent_tasks/01_weight_memory_map_generator.md`](agent_tasks/01_weight_memory_map_generator.md)** — Deterministic URAM `.mem` packing (288-bit word = 36 INT8 bytes); 22.4 MB across 53 convs.
- **[`docs/agent_tasks/02_layerir_to_wrapper_generator.md`](agent_tasks/02_layerir_to_wrapper_generator.md)** — `build_top_wrapper.ts`: LayerIR → streaming top with engine-hole.
- **[`docs/agent_tasks/03_scheduler_generator.md`](agent_tasks/03_scheduler_generator.md)** — Scheduler FSM (AXI4-Lite master, spatial_stall, BRAM ping-pong).
- **[`docs/agent_tasks/04_skip_fifo_sizing_tool.md`](agent_tasks/04_skip_fifo_sizing_tool.md)** — Two-phase skip-FIFO sizing (analytical + Verilator); the deterministic analogue of FINN `auto_fifosize`.
- **[`docs/agent_tasks/05_on_chip_weights_contract.md`](agent_tasks/05_on_chip_weights_contract.md)** — New first-class `on-chip-weights` (URAM) contract; latency = flat_bus + 2 cycles.
- **[`docs/agent_tasks/07_engine_mac_array.md`](agent_tasks/07_engine_mac_array.md)** — 256 INT8×INT8 lanes (LLM-generated); max_error 0 on ≥100 pairs, ≥200 DSP, ≥250 MHz.
- **[`docs/agent_tasks/09_engine_address_generator.md`](agent_tasks/09_engine_address_generator.md)** — 6-deep conv loop; bias granularity LOCKED; URAM 2-cycle latency pipelined.
- **[`docs/agent_tasks/10_engine_config_register_block.md`](agent_tasks/10_engine_config_register_block.md)** — First AXI4-Lite slave in nn2rtl (no prior reference); scoreboard TB, ≥400 MHz.
- **[`docs/agent_tasks/11_bram_to_stream_bridge.md`](agent_tasks/11_bram_to_stream_bridge.md)** — BRAM↔stream handshake bridge with 1-entry skid.
- **[`docs/agent_tasks/12_phase1_improve_sweep.md`](agent_tasks/12_phase1_improve_sweep.md)** — Multi-agent budget-capped compression sweep; worker isolation + serial merge.
- **[`docs/agent_tasks/13_integration_first_light.md`](agent_tasks/13_integration_first_light.md)** — Integration/first-light gate (structural soundness before byte-exact + timing).

Cross-refs: the integration root-cause logs (13a, engine-debug clusters) are in (D); the engine-as-fit-play and synth numbers in (E).

### (D) Bit/byte-exact verification & root-cause debugging sagas

**[`{memory}/project_relu_rescale_bug.md`]** — *TRUE root cause of ResNet e2e corruption, 2026-05-30.*
The RTL ReLU template emitted pure max(0,x) but 22 of 48 ReLU nodes must REQUANTIZE when input_scale ≠ output_scale. This supersedes the entire datapath/line_buf/handshake/xinit saga. Plus one residual add_7 operand half-swap.
- relu_1 was ×3 too small → conv_200 93.9% wrong → cascade to relu_48
- After fix: relu_48 POSITION 0.00% / MULTISET 0.00% / cosine 1.000000 / top-1 == golden → backbone byte-exact → 79.47%

**[`{memory}/feedback_regen_must_rebuild_engine_maps.md`]** — *Regen must rebuild engine maps, 2026-05-31.*
The multi-day ~16% ResNet bug was NOT RTL — it was 7 regen steps `generate_golden` does not perform. Stale/per-tensor engine bias (~0.43× too small) made all 14 engine convs systematically LOW.
- 7 mandatory post-generate_golden steps (build_bias_memory_map, build_scale_memory_map, build_spatial_scale_mems, repack 4 spatial convs, engine banks, refresh_final_golden, rebuild_contract_goldens)

**[`{memory}/project_phase2_e2e_localization.md`]** — *Phase-2 INT4 localization → stale bias map, 2026-05-29→31.*
Localization log; TRUE root cause was a stale `output/weights/bias.mem` never rebuilt, masked by stale contract goldens. Engine RTL was CORRECT throughout (iso harness: conv_246 MAC acc == dot product byte-exact). Also documents the all-INT4 AND Config B byte-exact PASS.
- Saturation fixed (4 fixes): e2e relu_48 mismatch 11793 → 2728
- conv_246 oc0-3 got [6,101,87,87] vs correct [15,236,203,203] (~0.43×) → fix = re-run build_bias_memory_map → mismatch 0

**[`{memory}/project_e2e_value_verification.md`]** — *e2e value verification: 3 bugs incl URAM 2-cycle latency, 2026-05-28.*
Step-2 value verification. Fixed all-zero output ($readmemh cwd) + wrong spatial scale constants, then root-caused a REAL engine weight-read-latency bug: the engine pipeline assumed 1-cycle URAM but deployed URAM is 2-cycle (READ_LATENCY_A=2) → each MAC multiplies a stale weight.
- Engine-iso WLAT=1 byte-exact; WLAT=2 = 15058/50176 wrong, max|err| 3
- After 2-cyc align: e2e mismatch 19126 → 11793; exposed BUG4 (conv_248 27.5%)

**[`{memory}/project_xinit_artifact_conv200.md`]** — *conv_200 "94%" was an X-init artifact, 2026-05-30.*
The conv_200 "94–96% wrong" was a simulation X-init artifact, not an RTL bug. Only Verilator `--x-initial 0` (models FPGA GSR power-on) is hardware-faithful → relu_48 = 2.72% diffuse residual; iverilog / no-x-init = ~95.74% X-poisoned.

**[`{memory}/project_spatialrun_handshake_bug.md`]** — *spatial_run handshake asymmetry, 2026-05-30.*
83 skid-fed nodes had out_ready spatial_run-gated but consuming valid_in ungated → beat dup/drop on engine_busy rise. Originally proposed as the 2.7% root cause; SUPERSEDED by the relu-rescale bug — kept as a ~12% cycle improvement.

**[`{memory}/project_add_tile_retile_fix.md`]** — *Add golden tile-pair retile fix.*
Tiled-streaming residual adds (OC≥1024) failed ~22% mismatch (max_error 15) due to a golden-vector retile bug in `sdk/orchestrate.ts`, not the RTL. Adds need each beat as lhs_tile | rhs_tile interleaved.
- node_add_9 attempt1 passes 1605632/1605632 channels, max_error 0 after fix

**[`docs/E2E_SIM_DEBUG_HANDOFF.md`](E2E_SIM_DEBUG_HANDOFF.md)** — *e2e freeze root-cause handoff, 2026-05-27.*
The integrated ResNet top had NEVER produced an output frame. Root cause: pulse-style producers don't honor downstream ready → beats silently dropped when a skid fills → fast/slow imbalance → chain freeze. Fix is comprehensive backpressure.
- Engine de-risked: forced dispatch completed all 14 engine convs; wall is 100% spatial-chain backpressure
- 12 pointwise 1×1 convs parallelized MP=16/MP_K=8 (~128× faster), byte-exact; inventory gap 19 more serial 1×1 convs missed

**[`docs/agent_tasks/13a_system_wiring.md`](agent_tasks/13a_system_wiring.md)** — *System wiring: 12 cross-piece bugs, 3 audit rounds.*
Wave 1/2/04c each passed local gates but system integration was incoherent: 12 cross-piece bugs (7 BUG-severity) — AXI deadlock, scale_mult 32→16-bit truncation, 288-vs-2048-bit URAM mismatch, missing bias memory, etc.
- Path D banked weight memory: 0 mismatches across ~10.6M slot comparisons (all 14 heavy modules)
- FSM ST_REQUANT exit hardcoded to 7; fixed to cfg_oc[11:8]−1; bias byte-order `<i` → `>i`

**[`docs/agent_tasks/13_engine_one_layer_verification.md`](agent_tasks/13_engine_one_layer_verification.md)** — *node_conv_246 2-byte off-by-one.*
Engine-isolation TB on one heavy layer: FAIL with 2 mismatches of 50,176 bytes (99.9960% byte-exact), both at pixel [1,4]. The "leave the FAIL in place" rule.

**[`docs/agent_tasks/13_engine_two_mismatch_rootcause.md`](agent_tasks/13_engine_two_mismatch_rootcause.md)** — *Dropped-last-MAC root cause.*
The 2 mismatches root-caused to `address_generator.v` suppressing the LAST weight read (k_cnt 2303) via `~k_at_last` gating → 2303/2304 products. Fix: gate on `~mac_done`. Verified PASS (max_error 0).

**[`docs/agent_tasks/13_engine_debug_SMALL_cluster.md`](agent_tasks/13_engine_debug_SMALL_cluster.md)** / **[`...MEDIUM_cluster.md`](agent_tasks/13_engine_debug_MEDIUM_cluster.md)** / **[`...LARGE_cluster.md`](agent_tasks/13_engine_debug_LARGE_cluster.md)** — *Counter-leak cluster (node_conv_266/290/286).*
One root cause (ic_cnt counter leak across OC passes; non-blocking last-assign-wins overrides the rising-edge reset) explains a 9-layer failure spectrum (2 → 8578 mismatches) correlated with input-activation byte-0 sparsity. Fix: wrap the counter advance in `if(!mac_done)`.
- node_conv_286: 8578 → PASS; full sweep 14/14 PASS byte-exact, max_error 0
- Mismatch count proportional to a[ic=0]≠0 pixels (validated across 14 layers)

**[`{memory}/project_mbv2_e2e_backpressure.md`]** — *MBV2 engine-top e2e: 5 root causes → BYTE-EXACT, 2026-06-02→04.*
The definitive log of getting the MBV2 engine-top byte-exact. Genuine root: spatial chain was push-only; then 5 streaming-integration root causes (wsel-dup deadlock, activation-layout, D1 act_out collision, 10-add operand swap, g_tiled off-by-one).
- **e2e PASS mismatch_bytes = 0** (all 1000 logits, 8.86M cyc); vec3 = 1 byte = node_linear float32-acc golden artifact
- Dispatch wedge moved 1 → 4 → 22 → cleared all 34; integrated synth OOM'd ~88 GB

**[`{memory}/project_mbv2_engine_p1_proven.md`]** / **[`docs/agent_tasks/mbv2_engine_p1_correctness.md`](agent_tasks/mbv2_engine_p1_correctness.md)** — *MBV2 engine 34/34 byte-exact, 2026-06-02.*
All 34 engine-dispatched pointwise convs (node_conv_814..912) byte-exact through the real shared_engine + real URAM/bias/scale at WLAT=2. Documents the load-bearing INT8/URAM facts and the WGT_W=4 inheritance bug.
- 34/34 PASS, mismatch 0, max|err| 0; WLAT=1 catastrophic (1.15M/1.2M, max|err| 252)
- mbv2 URAM = INT8 bytes, 288-bit lines, 32 OC/bank, 2048-bit bus; engine must be WGT_W=8/URAM_DATA_W=2048

**[`docs/agent_tasks/mbv2_engine_top_deadlock.md`](agent_tasks/mbv2_engine_top_deadlock.md)** — *Loader sizing-units bug + FIFO-overflow next blocker.*
Deadlock at dispatch 0: bridge counts 2048-bit words but the generator used predecessor output BEATS (off by 2048/BUS_W). Fix cleared it; next blocker = engine-output FIFO overflow / no backpressure.
- Loader word_count plateaus at 1568 = 12544/8; after fix LDR0_LOADED @11.5M cyc; FIFO (DEPTH 4096) dropped 8446/12544 beats

**[`docs/agent_tasks/mbv2_engine_concurrent_drain.md`](agent_tasks/mbv2_engine_concurrent_drain.md)** — *Concurrent-drain: refutes a hazard claim.*
RTL ground-truthing that REFUTES the roadmap's act-BRAM write-write hazard claim (engine already wins the arbiter; both bank-1 writers carry identical data). Converts a "HIGH-risk engine change" into a 4-line wrapper fix.

**[`{memory}/MEMORY.md`]** — *Master memory index, 2026-06-09.*
Master index of ~40 project/feedback notes; the navigation map into every per-topic memory note (full cross-cutting status of both networks).

Cross-refs: the relu-rescale and per-OC requant arithmetic ties to (B); the backpressure / handshake bugs tie to (H).

### (E) FPGA resource fit (BRAM/URAM/LUT)

**[`{memory}/project_fit_not_confirmed_synth_over.md`]** — *Fit NOT confirmed: 174% synth → congestion → chan_window fix, 2026-05-30→06-04.*
The "1960 BRAM36/72.9% FITS" claim was an unverified analytical estimate; the first real synth measured 4663 RAMB36 (174% OVER) + 1.98M LUT (115%). Later the chan_window mux-collapse fully routes.
- BRAM buckets: engine banks 2048, spatial weights_wide 1622, line buffers 765 RAMB36
- Earlier ROUTED design (May 26): 2496 RAMB36 + 37 RAMB18 = 93.55% BRAM, 203 URAM, WNS +10.119 @40ns
- Mixed-INT3 sweep: INT4 79.47% / INT3-on-4-biggest 79.47% (zero loss) / 8-layer 78.13% / full-INT3 69.67% / INT2 0.20% DEAD

**[`{memory}/project_mbv2_synth_oom.md`]** — *Why MBV2 OOMs synth + byte-exact fixes to fit, 2026-06-05→06.*
Proven counterintuitive root cause: the smaller MBV2 (3.5M params) OOMs 96 GB synth while bigger ResNet (25M) completes — synth RAM = elaborated-netlist size + RAM-inference failures, not param count. node_mean (GAP) dissolves into 20480 registers; 17 depthwise window_reg dissolve to 46080 regs each.
- Fixes byte-exact (equiv-TB mismatch 0): node_mean/node_linear serialized; ROM banked 5×262144×8
- Final fit: **LUT 78.1% / BRAM 45.8% / FF 38.5% / DSP 16.5%** — all <80%, no DRAM
- LUT forensics: 17 depthwise 746,665 LUT (44.7%), node_linear 366,248 (21.9%)

**[`{memory}/project_mbv2_synth_oom.md` companion: `{memory}/project_mbv2_synth_oom.md`]** — *(see above; same doc covers the keystone)* line_buf TILE_STORAGE keystone: depthwise 2394 → ~204 RAMB36, BRAM 94.7% → ~24.9%, byte-exact.

**[`docs/agent_tasks/mbv2_u250_fit_projection.md`](agent_tasks/mbv2_u250_fit_projection.md)** — *MBV2 fit projection (analytical), 2026-06-02.*
All six resources project under 80%. Two former risks closed: depthwise line buffers off ram_style=ultra (URAM 197%→10%); residual skip FIFOs right-sized (BRAM 829%→37.7%).
- Projected: LUT ~1,064,000 (61.6%), FF ~760,000 (22.0%), DSP 1,345 (10.9%), RAMB36 ~1,013 (37.7%), URAM 128 (10.0%)
- shared_engine MEASURED on xcu250: 107,268 LUT / 30,979 FF / 1,283 DSP / 0 BRAM / 0 URAM

**[`docs/agent_tasks/00_engine_only_synth_REPORT.md`](agent_tasks/00_engine_only_synth_REPORT.md)** — *Engine-only synth (xcu250).*
Standalone shared_engine OOC synth: overwhelmingly LUT/FF based (256-MAC array maps to LUT multipliers), zero BRAM.
- LUT 107,268; FF 30,979; DSP 1,283; BRAM 0; WNS setup 14.737 ns; Fmax 190.01 MHz (post-synth estimate)

**[`{memory}/project_uram_no_init.md`]** — *URAM cannot be non-zero initialized, 2026-05-28.*
Empirically proven 3 ways on Vivado 2025.2 / xcu250: no way to bake non-zero contents into URAM — it always powers up to zero. A silicon constraint (UG573). `weights→URAM` via `$readmemh` silently falls back to BRAM and overflows the budget; URAM capacity needs a RUNTIME loader.
- ram_style=ultra+$readmemh → [Synth 8-10226] falls back to 1 RAMB36E2; XPM ultra+init → [Synth 8-12183]

**[`docs/nn2rtl_u250_deployment_plan.md`](nn2rtl_u250_deployment_plan.md)** — *ResNet-50 U250 deployment plan.*
The hybrid spatial + shared-engine on-chip-only architecture and the quantified fit argument.
- Per-module LUT sum 2,917,911 vs 1.73M budget → full-spatial does not fit; 12 heavy modules (≥85k LUT) = 1.66M LUT (56.7%) → engine
- Weights 22.4 MB INT8 + 0.1 MB bias fit ~50% URAM (~625 of 1,280); total on-chip ~26 MB of ~57 MB (~46%)
- Success criteria: ≥10 fps, top-1 within 1.0 pp, LUT ≤95% (1.64M), Fmax ≥100 MHz

**[`docs/nn2rtl_u250_deployment_plan_mobilenetv2.md`](nn2rtl_u250_deployment_plan_mobilenetv2.md)** — *MobileNetV2 U250 deployment plan.*
MBV2's heavy-layer distribution is the OPPOSITE of ResNet's — 17 depthwise convs sum 1,313,679 LUT (76.0%). Depthwise stays spatial (no cross-channel reduction); pointwise heavies → engine as K=1×1 degenerate.
- Per-module LUT sum 2,042,651 (118% of budget); heaviest node_conv_818 depthwise 336,522 LUT (~20% of chip)
- Success: ≥20 fps, top-1 ≥70.8%, LUT ≤70% (1.21M), Fmax ≥150 MHz

**[`docs/agent_tasks/06_phase1_compression_candidates_REPORT.md`](agent_tasks/06_phase1_compression_candidates_REPORT.md)** — *Actual U250 LUT distribution.*
The Phase-0 ResNet-50 U250 baseline (119/119 synthesised). Primary quantitative evidence for area concentration.
- Network LUT sum 2,917,911; Heavy 14 = 1,803,388 (61.8%); Medium 16 = 639,094 (21.9%); Small 89 = 475,429 (16.3%)
- Top: node_conv_296 188,568 LUT; node_conv_290 188,388 @429.9 MHz; 6 BRAM-heavy 1×1 ~105k LUT + 171 BRAM18 each all @218.7 MHz

**[`docs/agent_tasks/06_phase1_compression_candidates.md`](agent_tasks/06_phase1_compression_candidates.md)** — *Candidate-selection method (analysis spec).*
The data-driven cutoff method (≥60% cumulative LUT → engine; ≥20k LUT → improve sweep).

**[`docs/agent_tasks/04c_skip_fifo_resize_throttled.md`](agent_tasks/04c_skip_fifo_resize_throttled.md)** — *2 GB FIFO root cause.*
The Phase-A analytical FIFO depths verified clean but were physically infeasible: node_add_3 demanded 4,194,304 entries (~2 GB), FIFO sum ~50× the whole U250 memory. Cause = unthrottled model. Fix re-models the throttled producer.

**[`docs/agent_tasks/13a_system_wiring.md`](agent_tasks/13a_system_wiring.md)** — *(see (D))* URAM accounting: weight 768 + activation 174 = 942/1280 = 73.6%.

**[`{memory}/project_int4_fit_analysis.md`]** — *(full abstract in (B))* INT4 takes weights from 1.9× over (impossible at INT8) to ~0.95–1.05× BRAM (borderline).

**[`docs/agent_tasks/int4_imagenet_timemux_autonomous_plan.md`](agent_tasks/int4_imagenet_timemux_autonomous_plan.md)** — *7-phase autonomous INT4/fit/Vivado plan, 2026-05-28.*
Key correction: engine time-mux gives ZERO BRAM saving here because URAM can't be init'd and there is no DRAM — all weights resident in BRAM regardless.
- FIT (analytical): spatial 1528 + engine 1131 + biases 57 = 2716 BRAM36 = 101%; biases→LUTRAM → 98.9% → fits
- U250 BRAM bitstream-init capacity ~99.1 Mbit (2688 BRAM36)

**[`docs/agent_tasks/01_weight_memory_map_generator.md`](agent_tasks/01_weight_memory_map_generator.md)** — *(see (C))* the URAM packing scheme (36 weights/word) underpinning the no-DRAM design.

Cross-refs: per-OC requant is the fit MANDATE (B); the fit-vs-route distinction is in (F).

### (F) Place-and-route & routing congestion

**[`{memory}/project_resnet_route_logic_bound.md`]** — *ResNet FULLY ROUTES via chan_window collapse; route was logic/slice-bound, 2026-06-04→06.*
SOLVED: ResNet-50 INT4 Config B fully closes on U250 by extending the chan_window mux-collapse to all 7 remaining eligible 3×3 convs. The route was logic/congestion-bound (lbw MUX fanout), NOT BRAM-density-bound.
- **0 overlaps / 0 unrouted, timing MET @40ns (WNS +11.714), routed Fmax ~35 MHz**
- Util LUT 60.70% / BRAM 94.64% / URAM 90.86% / DSP 66.71% / FF 37.82%; power 6.607 W
- INT3 density is a WEAK lever (75% buys ~110 RAMB36 = 4.3%); binding resource = CLB slices 97.09% (csel fo=11487 routed 15.9 ns)
- DSP_INPUT_PIPE + max_fanout improved placement-est (64–92 MHz) but ADDED congestion → route FAILED

**[`{memory}/project_fit_not_confirmed_synth_over.md`]** — *(full abstract in (E))* Config B place-fit LUT 80.14% / BRAM 94.64% / URAM 90.86%, setup WNS +23ns @40ns but initial route FAILED at congestion 6–7; chan_window cut route failures 354817 → 3065 signals (99%).

**[`{memory}/project_overnight_mbv2_improvements_then_vivado_20260609.md`]** — *MBV2 synth fits 71.1%, no routed Fmax, 2026-06-09.*
Synth CLEAN + FITS U250 (`--flatten=none`): LUT 71.1% / FF 35.2% / DSP 17.6% / BRAM 53.0%, no OOM. NO validated routed Fmax for any MBV2 version (c8 route never closed). Critical path route/SLR-crossing-bound.
- 12.5ns = 1.15ns logic + 11.4ns route; ~10.06ns = two unregistered SLR crossings in conv_866
- CLB-site saturation 95.78% (SLR1 100%, SLR2 99.9%) while raw LUT 76.5% → 6ns (167 MHz) architecturally unreachable
- DCP-corruption bug fixed (atomic .tmp+rename persist)

**[`docs/agent_tasks/13_integration_first_light_REPORT.md`](agent_tasks/13_integration_first_light_REPORT.md)** — *Route-only resume post-route report.*
Auto-generated post-route report (resume from `first_light_placed.dcp`, 16 ns clock). route_design + reports took 21,620.5 s.
- success: false; Setup WNS +0.863 ns; Hold WNS −0.165 ns; timing_met true; Fmax estimate 66.06 MHz
- Post-route util counters all read 0 (extraction not populated)

Cross-refs: the fit-vs-routability distinction (F is the 4th gate beyond E); Fmax campaign in (G).

### (G) Timing / Fmax campaign

**[`{memory}/project_resnet_fmax_campaign_20260609.md`]** — *ResNet Fmax campaign: 35.35 MHz routed, congestion-gated DSP pipe, 2026-06-09.*
Confirmed routed Fmax = 35.35 MHz (the "54.5 MHz" was a wrong pre-route estimate). Wall is route-bound (97.76% route): conv_298 channel_select broadcast fanout 11,487 crossing SLR1↔SLR3. DSP_INPUT_PIPE is byte-exact and raises the timing ceiling but does NOT route.
- **Routed Fmax 35.35 MHz** (WNS +11.714 ns @40ns, worst path 27.694 ns, 0 failing)
- BRAM 94.64% / URAM 90.86% → CLB slice packing 97.09% while raw LUT only 60%
- DSP_INPUT_PIPE byte-exact (mismatch 0/100352) raises ceiling 28.3 → ~16ns (~62 MHz, 1.8×) BUT route FAILED (10488 nets, congestion 6)
- A′ enabler: LINE_BUF_USE_URAM==2 frees ~513 RAMB36 (94.12% → ~75%), byte-exact

**[`{memory}/project_resnet_route_logic_bound.md`]** — *(full abstract in (F))* realistic Fmax ceiling ~35 → ~45 MHz; conv_298 round-trips 3 SLRs with 0 Laguna regs.

**[`{memory}/project_overnight_mbv2_fit_fmax_20260607.md`]** — *MBV2 fit <80% + Fmax campaign, 2026-06-07→09.*
FIT gate green (synth LUT 78.1% / BRAM 45.8%). Accuracy +4.00% top-1 via per-channel depthwise quant. Critical path is 91% route / 9% logic; timing FAILS @8ns; c8 route NEVER closed.
- **Accuracy 67.27% → 71.27%** (float ceiling 72.73%), byte-exact e2e 8/8
- Design is SLICE-bound: CLB 95.78% (SLR1 100%, SLR2 99.9%) vs LUT-as-logic 78.1%
- Timing FAILS @8ns: WNS −4.56 ns → ~79.6 MHz; logic floor 1.15 ns
- Verilator `--threads-4` mismatch 688 = MT-scheduler artifact; `--threads-1` byte-exact

**[`{memory}/project_overnight_mbv2_improvements_then_vivado_20260609.md`]** — *(full abstract in (F))* the "70.7 MHz" was a stale pre-route estimate; no validated routed Fmax; 6ns architecturally unreachable.

**[`scripts/PLAN_D_overlap.md`](../scripts/PLAN_D_overlap.md)** — *(see (H))* overlap is a throughput, not Fmax, lever.

**[`MILESTONES.md`](../MILESTONES.md)** — *(full abstract in (I))* per-module post-synth Fmax (63.7–106.6 MHz on Artix-7); the serialized-add timing-closure decision.

Cross-refs: Fmax is coupled to congestion (F) and density (E).

### (H) Throughput / cycles / fps

**[`{memory}/project_e2e_sim_debug.md`]** — *ResNet e2e working + optimized + FIFO-right-sized, 2026-05-28.*
The authoritative ResNet-50 INT8 integrated-top e2e reference. Full frame at 13,348,787 cyc, byte-exact, optimized from 21,887,261. Frame is spatial/streaming-bound.
- e2e 13,348,787 cyc; **7.49 @100MHz / 11.24 @150MHz / 14.98 @200MHz** (meets ≥10 fps)
- FIFO BRAM36 6848 → 728 (−89%), cycles unchanged, 0/106 FIFOs reached full
- Partition: eng_busy 4.55M (34%) / spatial_run 8.45M (65%) / stall ~0
- BRAM-only w/ weights→URAM ~1123 BRAM36 (42%) + 637 URAM (50%) FITS U250

**[`{memory}/project_mbv2_throughput_corrected.md`]** — *MBV2 throughput corrected; spatial-3×3 is the limiter, 2026-06-02→03.*
Adversarial review overturned the "17.5 fps" story: the engine is NOT the bottleneck (3.79M cyc, 3× below the stem); engine and spatial SERIALIZE; clock is 50 MHz. After A2 tap-parallel + overlap/backpressure: byte-exact trajectory.
- Engine-serial all 34 pointwise = 3.79M cyc; real limiter = spatial 3×3 (1 stem 11.44M + 17 DW MP=4 = 35.6M)
- Trajectory: 39.4M → 8.86M → 8.22M → 7.48M → 6.82M → 6.45M cyc (−27.3% vs 8.86M)
- Honest fps: P=1 ~5.08 @200MHz / 1.27 @50MHz

**[`docs/agent_tasks/mbv2_spatial_throughput_roadmap.md`](agent_tasks/mbv2_spatial_throughput_roadmap.md)** — *A1 overlap + A2 MP_K=9 tap-parallel, 2026-06-02.*
The two levers that actually move MBV2 throughput. A2 (the real lever) tap-parallelizes the spatial 3×3 at MP_K=9. PoC proved byte-exact (node_conv_812 mismatch 0).
- A2 MP_K=9: stem 11.44M → 1.81M, DW 24.17M → 5.75M, spatial 35.61M → 7.56M = **4.71× speedup**
- fps: baseline 1.27 @50MHz; A1+A2 6.61; A1+A2(MP=16) 11.06; DSP MP_K=9 = 648, design ~16.1%

**[`docs/agent_tasks/mbv2_kparallel_plan.md`](agent_tasks/mbv2_kparallel_plan.md)** — *Engine K-parallelism plan + adversarial NO-GO, 2026-06-02.*
Read-only plan for P-lane IC reduction-tree parallelism; an adversarial review (4 passes) overturns it as a NO-GO as written (engine already a 3× sub-limiter). Conditional GO for P=4 only after the spatial path is parallelized.
- conv_912: P=1 81,340 cyc → P=2 42,140 → P=4 22,540 → P=8 12,740; P=4 e2e ~+6% (NOT 1.75×)

**[`{memory}/project_mp_increase_deadlock.md`]** — *ResNet MP-increase deadlocks the e2e chain, 2026-05-30.*
Cycle-opt via spatial MP deadlocks: conv_196 8→16 and bulk 16→32 both hard-deadlock. Root cause = beat-misalignment at a residual-ADD join (conv_202 lhs has no skid; rhs has DEPTH=512). Byte-exact MP=16 baseline kept (15 fps).

**[`scripts/PLAN_D_overlap.md`](../scripts/PLAN_D_overlap.md)** — *MBV2 spatial↔engine overlap, read-only analysis.*
Overlapping SPATIAL (depthwise 3×3) with the ENGINE (1×1 pointwise) in the MBV2 scheduler; cites live RTL line counts (`nn2rtl_top_engine.v` 4097 lines, `nn2rtl_scheduler.v` 1153 lines).

Cross-refs: the backpressure root cause that enabled throughput is in (D); FIFO sizing in (E).

### (I) ResNet-50 vs MobileNetV2 case studies

**[`docs/nn2rtl_supervisor_explanation.md`](nn2rtl_supervisor_explanation.md)** — *(full abstract in (A))* the consolidated both-networks results dump (pass rate, cost, fps, area, Fmax, three-way comparison).

**[`MILESTONES.md`](../MILESTONES.md)** — *Layer-1 milestone (17 modules, Artix-7), 2026-04-28.*
The first passing end-to-end pipeline run: all 17 ResNet-50 stage-1 modules pass on Artix-7 at 50 MHz, Verilator bit-exact (max_error ≤ 2). The serialized-add architecture decision.
- 17/17 pass, 0/17 Surgeon retries after refresh; sum 79,878 LUT / 58,864 FF / 20 DSP / 142 BRAM18 = 126% LUT of 100T
- Serialized add: DSP util 306% → 8%, LUT 245% → 126%; line_buf BRAM rewrite layer1_0_conv2 162K → 10.7K LUT
- Total LLM cost $11.94

**[`{memory}/project_mobilenet_u250_status.md`]** — *MBV2 early-stage snapshot, 2026-05-31.*
MBV2 much earlier-stage than ResNet: not byte-exact yet (node_conv_818 off-by-1; node_conv_908 badly wrong max_error 20). LUT (not BRAM) is the MBV2 constraint; depthwise dominate (17 convs = 1.31M LUT).
- Fully-spatial baseline 2.04M LUT = 118% U250 (uncompressed); node_conv_912.compressed.v 82686 → 9767 LUT (−88%) on ZCU102

**[`{memory}/project_mbv2_remaining_improvements_backlog.md`]** — *MBV2 backlog after the 6-item push, 2026-06-09.*
Verified post-push backlog: stem node_conv_810 still per-TENSOR; Int8ReLU/Add/node_mean goldens still FLOAT-requant; MP 4→16 on 6 final DW (−444K cyc); A1 spatial↔engine overlap (~46% of e2e cycles).

**[`docs/agent_tasks/mbv2_PRE_VIVADO_DELIVERABLE.md`](agent_tasks/mbv2_PRE_VIVADO_DELIVERABLE.md)** — *MBV2 pre-Vivado deliverable, 2026-06-02.*
Morning summary: all six resources projected <80%, engine 34/34 byte-exact, 67.27% INT8 top-1, A2 MP_K=9 applied + byte-exact (18/18). One open item: full-system e2e (blocked on the final-stage contract mismatch).

**[`{memory}/project_overnight_mbv2_vivado_ready.md`]** — *Do NOT claim Vivado-ready prematurely, 2026-06-03.*
A self-correction: MBV2 was NOT Vivado-ready (e2e never passed, integrated synth OOM'd, 10240-bit Gemm-bus synth blocker). The 3rd over-optimistic claim caught. (Also (J).)

**MBV2 contract / blocker specs:**
- **[`docs/agent_tasks/mbv2_engine_top_roadmap.md`](agent_tasks/mbv2_engine_top_roadmap.md)** — Ordered blocker roadmap (risk-classed, safe-to-auto vs user-gated).
- **[`docs/agent_tasks/mbv2_blocker3_design.md`](agent_tasks/mbv2_blocker3_design.md)** — The tiled↔packed contract mismatch (576/960/1280 ch > 4096b flat-bus cap); 8-parameter fix.
- **[`docs/agent_tasks/mbv2_highoc_path_b_spec.md`](agent_tasks/mbv2_highoc_path_b_spec.md)** — HIGH-OC (OC>256) coherent on-disk patching spec; 3-mode engine_output_bridge.

Cross-refs: MBV2 fit (E), throughput (H), accuracy (B), engine byte-exactness (D).

### (J) Methodology lessons / what-worked

**[`{memory}/feedback_vivado_only_when_proven.md`]** — *HARD RULE: Vivado only when proven, 2026-05-31.*
Never run Vivado until bit-exact AND accurate AND fit-confirmed (all MEASURED, never estimated). The "1960 BRAM36 / fits" estimate was WRONG (first real synth 174% BRAM + 115% LUT). If any gate is unproven, STOP ResNet and pivot to MobileNetV2.

**[`{memory}/feedback_vivado_serialize_ram.md`]** — *Vivado must be serialized (RAM), 2026-05-30.*
Never run >1 Vivado synth at once: one MBV2 depthwise OOC ballooned to ~75 GB; 4 parallel OOC drove RAM to 95%. Verilator coexists fine; Vivado does not.

**[`{memory}/project_overnight_mbv2_vivado_ready.md`]** — *(full abstract in (I))* the explicit lesson to state projections as projections; retile-bridge per-bit loops → wide shift = ~15× faster sim.

**[`{memory}/feedback_fix_everything_no_defer.md`]** / **[`{memory}/feedback_autonomous_night_directive.md`]** / **[`{memory}/feedback_universal_diagnostics.md`]** / **[`{memory}/feedback_atomic_arch_changes.md`]** / **[`{memory}/feedback_no_reference_for_new_contracts.md`]** — *(full abstracts in (C))* the operator's autonomous-overnight contract, the correctness-gates-everything rule, the raw-facts-not-hypotheses diagnostic principle, the atomic-change invariant, and the withheld-reference experimental control.

**[`{memory}/project_failure_corpus_works.md`]** — *(full abstract in (C))* quantitative evidence the in-loop failure corpus accelerates convergence (+480..+3588 → +1 cycle).

**[`{memory}/feedback_hls4ml_bias_quant.md`]** — *hls4ml + scalar-Quant QONNX unsynthesizable, 2026-04.*
Do NOT build QONNX with Quant nodes for hls4ml: each scalar scale broadcasts to a per-element array (3.2M–12.8M float constants in one `.cpp`), hanging the Vitis HLS clang frontend indefinitely. Identical hang on 2025.2 and 2024.2. Fix: plain float ONNX + per-layer `ap_fixed`.

**[`{memory}/project_hls4ml_finn_comparison.md`]** — *(see (J)/comparison)* FINN can't ingest INT8 PTQ (needs Brevitas QAT, only W1A2 ResNet-50 public); hls4ml QONNX needs zero_point==0 + pow2 scale. Two-tier experimental design; standardize on ZCU102.

**[`{memory}/project_tier_a_complete.md`]** — *Tier A nn2rtl vs hls4ml, 2026-04-29.*
14/17 ResNet-50 stem+layer1 modules compared on Artix-7. nn2rtl ~2–10× less LUT/FF; all three 3×3 conv2 layers UNSYNTHESIZABLE by hls4ml.
- Worst gap layer0_0_conv1 +1878% LUT; nn2rtl 1 DSP/conv vs hls4ml 0 DSP; Fmax nn2rtl 60–280 MHz vs hls4ml ~70 MHz est

**[`docs/agent_tasks/13_engine_sweep_REPORT.md`](agent_tasks/13_engine_sweep_REPORT.md)** — *2-dispatch byte-exact sweep.*
node_conv_246 (453,744 cyc) + node_conv_300 (204,334 cyc) both PASS byte-exact, max_error 0.

---

## 3. Master index table

Sorted by category. Path is repo-relative where possible; project-memory notes use the `{memory}/` prefix.

| Category | Path | One-line hook |
|---|---|---|
| agent-spec | [`CLAUDE.md`](../CLAUDE.md) | Session-start rules: 3 LLM agents, deterministic Assayer, never write .v directly |
| agent-spec | [`docs/agent_tasks/00_engine_skeleton_spec.md`](agent_tasks/00_engine_skeleton_spec.md) | Human-written engine skeleton (the linchpin), 256-MAC OC-parallel |
| agent-spec | [`docs/agent_tasks/01_weight_memory_map_generator.md`](agent_tasks/01_weight_memory_map_generator.md) | Deterministic URAM .mem packing (36 INT8/288-bit word) |
| agent-spec | [`docs/agent_tasks/02_layerir_to_wrapper_generator.md`](agent_tasks/02_layerir_to_wrapper_generator.md) | LayerIR → top wrapper with engine-hole |
| agent-spec | [`docs/agent_tasks/03_scheduler_generator.md`](agent_tasks/03_scheduler_generator.md) | Scheduler FSM (AXI4-Lite master, spatial_stall, BRAM ping-pong) |
| agent-spec | [`docs/agent_tasks/04_skip_fifo_sizing_tool.md`](agent_tasks/04_skip_fifo_sizing_tool.md) | Two-phase skip-FIFO sizing; deterministic FINN auto_fifosize analogue |
| agent-spec | [`docs/agent_tasks/05_on_chip_weights_contract.md`](agent_tasks/05_on_chip_weights_contract.md) | New on-chip-weights (URAM) contract; +2-cycle latency |
| agent-spec | [`docs/agent_tasks/06_phase1_compression_candidates.md`](agent_tasks/06_phase1_compression_candidates.md) | Heavy/spatial candidate-selection cutoff method |
| agent-spec | [`docs/agent_tasks/07_engine_mac_array.md`](agent_tasks/07_engine_mac_array.md) | 256 INT8×INT8 MAC array sub-block spec |
| agent-spec | [`docs/agent_tasks/08_engine_requant_pipeline.md`](agent_tasks/08_engine_requant_pipeline.md) | 3-stage 256-lane requant, round-half-up + saturate |
| agent-spec | [`docs/agent_tasks/09_engine_address_generator.md`](agent_tasks/09_engine_address_generator.md) | 6-deep conv loop; URAM 2-cycle latency pipelined |
| agent-spec | [`docs/agent_tasks/10_engine_config_register_block.md`](agent_tasks/10_engine_config_register_block.md) | First AXI4-Lite slave in nn2rtl (no prior reference) |
| agent-spec | [`docs/agent_tasks/11_bram_to_stream_bridge.md`](agent_tasks/11_bram_to_stream_bridge.md) | BRAM↔stream handshake bridge, 1-entry skid |
| agent-spec | [`docs/agent_tasks/12_phase1_improve_sweep.md`](agent_tasks/12_phase1_improve_sweep.md) | Multi-agent budget-capped compression sweep |
| agent-spec | [`docs/agent_tasks/13_integration_first_light.md`](agent_tasks/13_integration_first_light.md) | Integration/first-light gate (structural before byte-exact) |
| agent-spec | [`docs/agent_tasks/mbv2_highoc_path_b_spec.md`](agent_tasks/mbv2_highoc_path_b_spec.md) | MBV2 HIGH-OC on-disk patching; 3-mode engine bridge |
| agent-spec | [`{memory}/feedback_autonomous_night_directive.md`] | Autonomous-overnight working contract |
| agent-spec | [`{memory}/feedback_no_reference_for_new_contracts.md`] | Withhold reference Verilog = experimental control |
| agent-spec | [`{memory}/feedback_self_improve_required_for_contract_walk.md`] | NN2RTL_SELF_IMPROVE=1 gates the contract walker |
| architecture-doc | [`README.md`](../README.md) | Canonical nn2rtl design spec (latency contracts, quant scheme, RQs) |
| architecture-doc | [`ARCHITECTURE.md`](../ARCHITECTURE.md) | File-by-file code tour + Vivado-migration log |
| architecture-doc | [`rtl_library/SPLIT_ARCHITECTURE.md`](../../rtl_library/SPLIT_ARCHITECTURE.md) | Split spatial-conv = wrapper over 3 library modules |
| architecture-doc | [`knowledge/IMPLEMENTATION_PLAN.md`](../knowledge/IMPLEMENTATION_PLAN.md) | Pattern-library / self-improvement implementation plan |
| architecture-doc | [`docs/agent_tasks/00_engine_skeleton_spec_FSM.md`](agent_tasks/00_engine_skeleton_spec_FSM.md) | 6-state engine FSM contract |
| architecture-doc | [`docs/agent_tasks/00_engine_skeleton_spec_PORTS.md`](agent_tasks/00_engine_skeleton_spec_PORTS.md) | Engine port single-source-of-truth (2048/8192-bit buses) |
| architecture-doc | [`{memory}/project_onnx_frontend.md`] | Universal ONNX frontend (conv/relu/add/maxpool) |
| memory-diary | [`{memory}/MEMORY.md`] | Master index of ~40 project/feedback memory notes |
| memory-lesson | [`{memory}/feedback_accuracy_measure_bn_folded.md`] | Accuracy must be BatchNorm-fold-aware |
| memory-lesson | [`{memory}/feedback_atomic_arch_changes.md`] | RTL + latency formula + TB + goldens change atomically |
| memory-lesson | [`{memory}/feedback_fix_everything_no_defer.md`] | Correctness gates fit/perf/Vivado |
| memory-lesson | [`{memory}/feedback_universal_diagnostics.md`] | Emit raw facts, not hypotheses |
| memory-lesson | [`{memory}/feedback_vivado_only_when_proven.md`] | 3-gate discipline before Vivado |
| memory-lesson | [`{memory}/feedback_vivado_serialize_ram.md`] | Serialize Vivado (one OOC = 75 GB) |
| memory-lesson | [`{memory}/project_failure_corpus_works.md`] | Failure corpus → measurable convergence |
| memory-lesson | [`{memory}/project_overnight_mbv2_vivado_ready.md`] | Don't claim Vivado-ready prematurely |
| memory-lesson | [`{memory}/project_top_v_is_patched_not_regenerated.md`] | Top is patched, not regenerated |
| plan-roadmap | [`docs/agent_tasks/README.md`](agent_tasks/README.md) | Parallel agent-task dispatch (Waves 1–4) |
| plan-roadmap | [`docs/agent_tasks/int4_imagenet_timemux_autonomous_plan.md`](agent_tasks/int4_imagenet_timemux_autonomous_plan.md) | 7-phase autonomous INT4/fit/Vivado plan |
| plan-roadmap | [`docs/agent_tasks/mbv2_engine_top_roadmap.md`](agent_tasks/mbv2_engine_top_roadmap.md) | MBV2 engine-top blocker roadmap |
| plan-roadmap | [`docs/agent_tasks/mbv2_kparallel_plan.md`](agent_tasks/mbv2_kparallel_plan.md) | Engine K-parallelism plan + adversarial NO-GO |
| plan-roadmap | [`docs/agent_tasks/mbv2_spatial_throughput_roadmap.md`](agent_tasks/mbv2_spatial_throughput_roadmap.md) | A1 overlap + A2 MP_K=9 tap-parallel (4.71×) |
| plan-roadmap | [`docs/agent_tasks/phase2_int4_per_oc_plan.md`](agent_tasks/phase2_int4_per_oc_plan.md) | INT4 nibble-pack + per-OC requant plan |
| plan-roadmap | [`docs/nn2rtl_u250_deployment_plan.md`](nn2rtl_u250_deployment_plan.md) | ResNet-50 U250 hybrid deployment plan |
| plan-roadmap | [`docs/nn2rtl_u250_deployment_plan_mobilenetv2.md`](nn2rtl_u250_deployment_plan_mobilenetv2.md) | MobileNetV2 U250 deployment plan |
| plan-roadmap | [`scripts/PLAN_B_golden_hardening.md`](../scripts/PLAN_B_golden_hardening.md) | MBV2 golden integer fixed-point hardening |
| plan-roadmap | [`scripts/PLAN_D_overlap.md`](../scripts/PLAN_D_overlap.md) | MBV2 spatial↔engine overlap analysis |
| plan-roadmap | [`scripts/PLAN_E_pointwise_per_oc.md`](../scripts/PLAN_E_pointwise_per_oc.md) | MBV2 pointwise per-OC plan |
| plan-roadmap | [`{memory}/project_mbv2_remaining_improvements_backlog.md`] | MBV2 post-push improvement backlog |
| plan-roadmap | [`{memory}/project_overnight_chain_20260606.md`] | Standing overnight task chain |
| root-cause-finding | [`SYSTEM_REVIEW_FINDINGS.md`](../SYSTEM_REVIEW_FINDINGS.md) | Feedback-loop hazards (iverilog crash misclassified) |
| root-cause-finding | [`docs/E2E_SIM_DEBUG_HANDOFF.md`](E2E_SIM_DEBUG_HANDOFF.md) | e2e freeze = pulse producers ignore ready |
| root-cause-finding | [`docs/agent_tasks/04c_skip_fifo_resize_throttled.md`](agent_tasks/04c_skip_fifo_resize_throttled.md) | 2 GB FIFO from unthrottled model |
| root-cause-finding | [`docs/agent_tasks/13_engine_debug_LARGE_cluster.md`](agent_tasks/13_engine_debug_LARGE_cluster.md) | node_conv_286 8578 → PASS (counter leak) |
| root-cause-finding | [`docs/agent_tasks/13_engine_debug_MEDIUM_cluster.md`](agent_tasks/13_engine_debug_MEDIUM_cluster.md) | node_conv_290 184 → 14/14 PASS |
| root-cause-finding | [`docs/agent_tasks/13_engine_debug_SMALL_cluster.md`](agent_tasks/13_engine_debug_SMALL_cluster.md) | node_conv_266 2 → PASS (shared fix) |
| root-cause-finding | [`docs/agent_tasks/13_engine_one_layer_verification.md`](agent_tasks/13_engine_one_layer_verification.md) | node_conv_246 2/50176-byte off-by-one |
| root-cause-finding | [`docs/agent_tasks/13_engine_two_mismatch_rootcause.md`](agent_tasks/13_engine_two_mismatch_rootcause.md) | Dropped-last-MAC (~k_at_last vs ~mac_done) |
| root-cause-finding | [`docs/agent_tasks/13a_system_wiring.md`](agent_tasks/13a_system_wiring.md) | 12 cross-piece integration bugs, 3 audit rounds |
| root-cause-finding | [`docs/agent_tasks/mbv2_blocker3_design.md`](agent_tasks/mbv2_blocker3_design.md) | tiled↔packed contract mismatch (>4096b) |
| root-cause-finding | [`docs/agent_tasks/mbv2_engine_concurrent_drain.md`](agent_tasks/mbv2_engine_concurrent_drain.md) | Refutes act-BRAM write-write hazard claim |
| root-cause-finding | [`docs/agent_tasks/mbv2_engine_p1_correctness.md`](agent_tasks/mbv2_engine_p1_correctness.md) | MBV2 engine 34/34 byte-exact + WGT_W bug |
| root-cause-finding | [`docs/agent_tasks/mbv2_engine_top_deadlock.md`](agent_tasks/mbv2_engine_top_deadlock.md) | Loader sizing-units bug + FIFO overflow |
| root-cause-finding | [`{memory}/feedback_hls4ml_bias_quant.md`] | QONNX scalar-Quant unsynthesizable by hls4ml |
| root-cause-finding | [`{memory}/feedback_regen_must_rebuild_engine_maps.md`] | 7 mandatory regen steps after generate_golden |
| root-cause-finding | [`{memory}/project_add_tile_retile_fix.md`] | Add golden tile-pair retile bug (not RTL) |
| root-cause-finding | [`{memory}/project_deploy_vs_measure_calibration.md`] | Deploy vs measure quantization paths diverge |
| root-cause-finding | [`{memory}/project_e2e_value_verification.md`] | URAM 2-cycle weight-read-latency bug |
| root-cause-finding | [`{memory}/project_int4_accuracy_and_deployment_gap.md`] | INT4 accuracy sound; live weights already INT4 |
| root-cause-finding | [`{memory}/project_int4_fit_analysis.md`] | Per-OC GPTQ required for usable INT4 |
| root-cause-finding | [`{memory}/project_mbv2_e2e_backpressure.md`] | MBV2 e2e 5 root causes → byte-exact |
| root-cause-finding | [`{memory}/project_mbv2_engine_p1_proven.md`] | MBV2 engine datapath byte-exact (34/34) |
| root-cause-finding | [`{memory}/project_mbv2_synth_oom.md`] | Smaller MBV2 OOMs synth (netlist, not params) |
| root-cause-finding | [`{memory}/project_mbv2_throughput_corrected.md`] | Engine is NOT the bottleneck; spatial 3×3 is |
| root-cause-finding | [`{memory}/project_mp_increase_deadlock.md`] | ResNet MP-increase deadlocks at add-join |
| root-cause-finding | [`{memory}/project_phase2_e2e_localization.md`] | Root cause = stale engine bias map |
| root-cause-finding | [`{memory}/project_relu_rescale_bug.md`] | 22 ReLUs missing activation rescale |
| root-cause-finding | [`{memory}/project_resnet_route_logic_bound.md`] | ResNet fully routes; route was logic/slice-bound |
| root-cause-finding | [`{memory}/project_spatialrun_handshake_bug.md`] | spatial_run handshake asymmetry (superseded) |
| root-cause-finding | [`{memory}/project_uram_no_init.md`] | URAM cannot be non-zero initialized |
| root-cause-finding | [`{memory}/project_xinit_artifact_conv200.md`] | conv_200 "94%" = X-init sim artifact |
| status-report | [`MILESTONES.md`](../MILESTONES.md) | Layer-1 milestone (17 modules, Artix-7, 50 MHz) |
| status-report | [`PROJECT_AUDIT.md`](../PROJECT_AUDIT.md) | Code-quality audit (~32 fixes) |
| status-report | [`docs/agent_tasks/00_engine_only_synth_REPORT.md`](agent_tasks/00_engine_only_synth_REPORT.md) | Engine-only synth (107k LUT, 190 MHz) |
| status-report | [`docs/agent_tasks/06_phase1_compression_candidates_REPORT.md`](agent_tasks/06_phase1_compression_candidates_REPORT.md) | Actual U250 LUT distribution (2.9M sum) |
| status-report | [`docs/agent_tasks/13_engine_sweep_REPORT.md`](agent_tasks/13_engine_sweep_REPORT.md) | 2-dispatch byte-exact sweep (max_error 0) |
| status-report | [`docs/agent_tasks/13_integration_first_light_REPORT.md`](agent_tasks/13_integration_first_light_REPORT.md) | Route-only resume (Fmax 66.06 MHz est) |
| status-report | [`docs/agent_tasks/int4_imagenet_FINAL_REPORT.md`](agent_tasks/int4_imagenet_FINAL_REPORT.md) | ResNet INT4 final (79.47%, fit not confirmed) |
| status-report | [`docs/agent_tasks/mbv2_PRE_VIVADO_DELIVERABLE.md`](agent_tasks/mbv2_PRE_VIVADO_DELIVERABLE.md) | MBV2 pre-Vivado (67.27%, <80% fit, 4.71×) |
| status-report | [`docs/agent_tasks/mbv2_u250_fit_projection.md`](agent_tasks/mbv2_u250_fit_projection.md) | MBV2 fit projection (all six <80%) |
| status-report | [`docs/nn2rtl_supervisor_explanation.md`](nn2rtl_supervisor_explanation.md) | Full both-networks results dump |
| status-report | [`{memory}/project_fit_not_confirmed_synth_over.md`] | 174% synth → congestion → chan_window fix |
| status-report | [`{memory}/project_mobilenet_u250_status.md`] | MBV2 early-stage snapshot |
| status-report | [`{memory}/project_overnight_mbv2_fit_fmax_20260607.md`] | MBV2 fit <80% + Fmax (78.1%, +4% acc) |
| status-report | [`{memory}/project_overnight_mbv2_improvements_then_vivado_20260609.md`] | MBV2 synth fits 71.1%, no routed Fmax |
| status-report | [`{memory}/project_pipeline_status.md`] | ResNet-50 phase map + post-Vivado roadmap |
| status-report | [`{memory}/project_resnet_fmax_campaign_20260609.md`] | ResNet Fmax 35.35 MHz routed, congestion-gated |
| status-report | [`{memory}/project_tier_a_complete.md`] | Tier A nn2rtl vs hls4ml (14/17 modules) |
| runtime-log | [`docs/agent_tasks/autonomous_night_log.md`](agent_tasks/autonomous_night_log.md) | The autonomous overnight decision/findings journal |
| runtime-log | [`{memory}/project_e2e_sim_debug.md`] | ResNet e2e working/optimized/FIFO-right-sized |
| (plan) | [`{memory}/project_hls4ml_finn_comparison.md`] | hls4ml/FINN comparison constraints + 2-tier design |
| (other) | [`{memory}/project_pipeline_status.md`] | (see status-report) |

---

## 4. Headline quantitative results

Every number is attributed to its source. (`{memory}/` = project-memory note.)

**Accuracy (top-1):**
- ResNet-50 INT4 per-channel GPTQ = **79.47%** (float 80.07%, INT8 ~75%); per-tensor GPTQ unusable 2.80% — [`int4_imagenet_FINAL_REPORT.md`](agent_tasks/int4_imagenet_FINAL_REPORT.md)
- ResNet-50 Config B (18 INT3 + 35 INT4) = **77.60% measured / 77.07% deployed** — `{memory}/feedback_accuracy_measure_bn_folded.md`, `{memory}/project_fit_not_confirmed_synth_over.md`
- Weight-only quant ladder: float 81.05 / INT8-tensor 80.47 / INT8-chan 80.86 / INT4-tensor 0.00 / INT4-chan 32.42 / GPTQ per-ch 77.7% — `{memory}/project_int4_fit_analysis.md`
- INT8 full-pipeline ResNet = 75.20% top-1 / 93.16% top-5 — `{memory}/project_int4_fit_analysis.md`
- MobileNetV2 INT8 = **67.27%** (float 72.67%); per-channel depthwise → **71.27%** (+4.00%, ceiling 72.73%) — `{memory}/project_overnight_mbv2_fit_fmax_20260607.md`
- Mixed-INT3 sweep (1500 img): INT8 80.27 / INT5 79.40 / INT4 79.47 / INT3 69.67 / INT2 0.20 (dead) — `{memory}/autonomous_night_log` via `{memory}/project_resnet_route_logic_bound.md`

**Byte-exact / mismatch=0 milestones:**
- ResNet relu_48 final: POSITION 0.00% / MULTISET 0.00% / cosine 1.000000 → backbone byte-exact — `{memory}/project_relu_rescale_bug.md`
- ResNet all-INT4 AND Config B e2e: result PASS, mismatch 0, 3136/3136 beats — `{memory}/project_phase2_e2e_localization.md`
- ResNet engine cluster sweep: 14/14 PASS byte-exact, max_error 0 (node_conv_286 8578 → 0) — [`13_engine_debug_LARGE_cluster.md`](agent_tasks/13_engine_debug_LARGE_cluster.md)
- Pipeline run #3: max_error 0 across 6,422,528 samples, first_mismatch_index −1 — [`ARCHITECTURE.md`](../ARCHITECTURE.md)
- MobileNetV2 engine-top e2e: PASS mismatch_bytes 0 (all 1000 logits, 8.86M cyc) — `{memory}/project_mbv2_e2e_backpressure.md`
- MobileNetV2 engine datapath: 34/34 dispatches mismatch 0, max|err| 0 at WLAT=2 — `{memory}/project_mbv2_engine_p1_proven.md`
- DSP_INPUT_PIPE byte-exact e2e mismatch 0/100352 — `{memory}/project_resnet_fmax_campaign_20260609.md`

**FPGA fit (utilization):**
- ResNet first real synth (over): RAMB36 **4663 (174%)**, LUT 1,983,938 (115%), URAM 203 (16%), DSP 7429 (60%), FF 38% — [`int4_imagenet_FINAL_REPORT.md`](agent_tasks/int4_imagenet_FINAL_REPORT.md)
- ResNet routed util: LUT 60.70% / BRAM 94.64% / URAM 90.86% / DSP 66.71% / FF 37.82%; power 6.607 W — `{memory}/project_resnet_route_logic_bound.md`
- ResNet network LUT sum = 2,917,911 (Heavy 14 = 61.8%, Medium 16 = 21.9%, Small 89 = 16.3%) — [`06_..._REPORT.md`](agent_tasks/06_phase1_compression_candidates_REPORT.md)
- MobileNetV2 final synth: **LUT 78.1% / BRAM 45.8% / FF 38.5% / DSP 16.5%** — all <80% — `{memory}/project_mbv2_synth_oom.md`
- MobileNetV2 synth (`--flatten=none`): LUT 71.1% / FF 35.2% / DSP 17.6% / BRAM 53.0% — `{memory}/project_overnight_mbv2_improvements_then_vivado_20260609.md`
- MobileNetV2 per-module LUT sum 2,042,651 (118%); 17 depthwise = 1,313,679 LUT (76.0%); node_conv_818 = 336,522 LUT — [`..._mobilenetv2.md`](nn2rtl_u250_deployment_plan_mobilenetv2.md)
- shared_engine OOC (xcu250): 107,268 LUT / 30,979 FF / 1,283 DSP / 0 BRAM — [`00_engine_only_synth_REPORT.md`](agent_tasks/00_engine_only_synth_REPORT.md)

**Place-and-route / congestion:**
- ResNet Config B FULLY ROUTES: 0 overlaps / 0 unrouted, timing MET @40ns — `{memory}/project_resnet_route_logic_bound.md`
- chan_window fix cut route failures 354,817 → 3,065 signals (99%); overlaps 532,004 → 2,022 — `{memory}/project_fit_not_confirmed_synth_over.md`
- Binding resource = CLB slices 97.09% (csel broadcast fanout 11,487 routed 15.9 ns, conv_298 across 3 SLRs, 0 Laguna regs) — `{memory}/project_resnet_route_logic_bound.md`
- MobileNetV2 CLB-site saturation 95.78% (SLR1 100%, SLR2 99.9%) while raw LUT 76.5% — `{memory}/project_overnight_mbv2_improvements_then_vivado_20260609.md`

**Timing / Fmax:**
- ResNet **confirmed routed Fmax = 35.35 MHz** (WNS +11.714 ns @40ns, 0 failing) — `{memory}/project_resnet_fmax_campaign_20260609.md`
- ResNet DSP_INPUT_PIPE timing ceiling ~62 MHz (~1.8×) but congestion-gated (won't route) — `{memory}/project_resnet_fmax_campaign_20260609.md`
- MobileNetV2 timing FAILS @8ns: WNS −4.56 ns → ~79.6 MHz; NO validated routed Fmax (c8 never closed) — `{memory}/project_overnight_mbv2_fit_fmax_20260607.md`
- MobileNetV2 critical path 12.5ns = 1.15ns logic + 11.4ns route (~10ns from two SLR crossings in conv_866) — `{memory}/project_overnight_mbv2_improvements_then_vivado_20260609.md`
- Engine standalone post-synth Fmax 190.01 MHz — [`00_engine_only_synth_REPORT.md`](agent_tasks/00_engine_only_synth_REPORT.md)
- ResNet route-only resume estimate 66.06 MHz @16ns (setup met, hold failed) — [`13_integration_first_light_REPORT.md`](agent_tasks/13_integration_first_light_REPORT.md)

**Throughput / cycles / fps:**
- ResNet full frame **13,348,787 cyc = 14.98 fps @200MHz / 11.24 @150 / 7.49 @100** (meets ≥10 fps) — `{memory}/project_e2e_sim_debug.md`
- ResNet partition: eng_busy 4.55M (34%) / spatial_run 8.45M (65%) / stall ~0 — `{memory}/project_e2e_sim_debug.md`
- ResNet FIFO BRAM36 6848 → 728 (−89%), cycles unchanged — `{memory}/project_e2e_sim_debug.md`
- MobileNetV2 throughput trajectory 39.4M → 6.45M cyc (−27.3% vs 8.86M baseline) — `{memory}/project_mbv2_throughput_corrected.md`
- MobileNetV2 A2 MP_K=9: spatial 35.61M → 7.56M cyc = **4.71× speedup**; honest fps P=1 5.08 @200MHz / 1.27 @50MHz — [`mbv2_spatial_throughput_roadmap.md`](agent_tasks/mbv2_spatial_throughput_roadmap.md)
- Engine per-layer cycles: node_conv_246 453,744; node_conv_300 204,334 (both byte-exact) — [`13_engine_sweep_REPORT.md`](agent_tasks/13_engine_sweep_REPORT.md)

**Methodology / cost / comparison:**
- ResNet 119/119 pass, $170.61 ($1.43/module); MobileNetV2 97/99 pass, $196.39 ($2.02/module) — [`nn2rtl_supervisor_explanation.md`](nn2rtl_supervisor_explanation.md)
- nn2rtl vs hls4ml vs FINN (8 layers): 3,992 vs 32,174 vs 49,079 LUT (~8×/12× fewer); Fmax 339 vs 71 vs 109 MHz — [`nn2rtl_supervisor_explanation.md`](nn2rtl_supervisor_explanation.md)
- hls4ml 3×3 conv2 layers UNSYNTHESIZABLE (3.2M–12.8M float constants hang the clang frontend) — `{memory}/feedback_hls4ml_bias_quant.md`, `{memory}/project_tier_a_complete.md`
- Failure-corpus convergence: latency error +480..+3588 → +1 cycle — `{memory}/project_failure_corpus_works.md`
- One Vivado depthwise OOC = ~75 GB RAM; 4 parallel → 95% — `{memory}/feedback_vivado_serialize_ram.md`

---

## 5. Raw appendix material

Pointers only — these buckets are NOT inlined. Locations and counts:

**Claude session transcripts (`*.jsonl`)** — `~/.claude/projects/` — 1689 total:
- `C--Users-User-Desktop-RTL-LLM-CLAUDE-nn2rtl-repo/` — 661 flat session files
- `c--Users-User-Desktop-RTL-LLM-CLAUDE/.../subagents/workflows/wf_*/` — ~960 across ~150 workflow subdirs (largest: wf_517c278e 73, wf_564c9743 37, wf_cf2ceac1 37, wf_c76f145c 35)

**In-repo structured agent tool-use transcripts (`agent_tool_use.jsonl`):**
- `output/reports/agent_tool_use.jsonl` (1592 lines, ResNet)
- `output/mobilenet-v2/reports/agent_tool_use.jsonl` (1080 lines, MBV2)
- `output/_worker_{0..3}/reports/agent_tool_use.jsonl`

**Failure corpus (structured JSON-lines, human-inspectable):**
- `output/failure_corpus/visible/index.jsonl` (135 entries)
- `output/mobilenet-v2/failure_corpus/visible/index.jsonl` (74 entries)

**Pipeline event/job logs:**
- `output/{reports,mobilenet-v2/reports}/run_log.jsonl` (17), `tool_calls.jsonl` (140), `output/dashboard/jobs.jsonl` (266)

**Runtime stdout logs (`*.log`)** — `output/*.log` (66 files: synth/improve/redispatch/route_only/vivado_baseline captures) + per-module `output/logs/improve_worker_*.log`

**Vivado/Yosys synth reports** — `output/reports_integrated/` — 18 `*.rpt` (`configB_placed_timing.rpt`, `configB_placed_util.rpt`, `congestion_analysis.rpt`, `fmax_top15_paths.rpt`, `fmax_worst_detail.rpt`, `hier_util.rpt`, `diag_routed_slack.rpt`, …) + 1 `.txt` summary

**HLS/Vivado machine logs (comparison appendix)** — `comparison/tier_a/` — 55 `vivado.log` + 1 `vitis_hls.log` + thousands of clang/autopilot `*.out.log`/`*.err.log` under `hls4ml_out/.../.autopilot/db/`

**Protected pattern library (reference knowledge, semi-readable)** — `knowledge/patterns/protected/01_context.md` … `13_on_chip_weights.md` (13 files); agent role defs at `nn2rtl-plugin/agents/*.md` and `nn2rtl-plugin/skills/*/SKILL.md`

---

## 6. Suggested thesis chapter mapping

| Thesis chapter | Primary source docs |
|---|---|
| 1. Introduction & research questions | [`README.md`](../README.md), [`docs/nn2rtl_supervisor_explanation.md`](nn2rtl_supervisor_explanation.md) |
| 2. System architecture (agentic pipeline) | [`ARCHITECTURE.md`](../ARCHITECTURE.md), [`README.md`](../README.md), [`CLAUDE.md`](../CLAUDE.md), [`knowledge/IMPLEMENTATION_PLAN.md`](../knowledge/IMPLEMENTATION_PLAN.md), `{memory}/project_onnx_frontend.md` |
| 3. RTL-generation methodology & contracts | [`rtl_library/SPLIT_ARCHITECTURE.md`](../../rtl_library/SPLIT_ARCHITECTURE.md), `docs/agent_tasks/00–13_*`, [`05_on_chip_weights_contract.md`](agent_tasks/05_on_chip_weights_contract.md), `{memory}/feedback_atomic_arch_changes.md`, `{memory}/feedback_no_reference_for_new_contracts.md` |
| 4. Quantization & accuracy | [`int4_imagenet_FINAL_REPORT.md`](agent_tasks/int4_imagenet_FINAL_REPORT.md), `{memory}/project_int4_fit_analysis.md`, `{memory}/project_int4_accuracy_and_deployment_gap.md`, `{memory}/feedback_accuracy_measure_bn_folded.md`, `{memory}/project_deploy_vs_measure_calibration.md`, [`08_engine_requant_pipeline.md`](agent_tasks/08_engine_requant_pipeline.md) |
| 5. Bit/byte-exact verification & debugging | `{memory}/project_relu_rescale_bug.md`, `{memory}/feedback_regen_must_rebuild_engine_maps.md`, `{memory}/project_e2e_value_verification.md`, `{memory}/project_xinit_artifact_conv200.md`, `13_engine_debug_*_cluster.md`, [`13a_system_wiring.md`](agent_tasks/13a_system_wiring.md), `{memory}/project_mbv2_e2e_backpressure.md` |
| 6. FPGA resource fit | `{memory}/project_fit_not_confirmed_synth_over.md`, `{memory}/project_mbv2_synth_oom.md`, [`mbv2_u250_fit_projection.md`](agent_tasks/mbv2_u250_fit_projection.md), `{memory}/project_uram_no_init.md`, [`nn2rtl_u250_deployment_plan.md`](nn2rtl_u250_deployment_plan.md), [`06_..._REPORT.md`](agent_tasks/06_phase1_compression_candidates_REPORT.md) |
| 7. Place-and-route & congestion | `{memory}/project_resnet_route_logic_bound.md`, `{memory}/project_fit_not_confirmed_synth_over.md`, `{memory}/project_overnight_mbv2_improvements_then_vivado_20260609.md`, [`13_integration_first_light_REPORT.md`](agent_tasks/13_integration_first_light_REPORT.md) |
| 8. Timing / Fmax | `{memory}/project_resnet_fmax_campaign_20260609.md`, `{memory}/project_overnight_mbv2_fit_fmax_20260607.md`, [`00_engine_only_synth_REPORT.md`](agent_tasks/00_engine_only_synth_REPORT.md), [`MILESTONES.md`](../MILESTONES.md) |
| 9. Throughput / cycles / fps | `{memory}/project_e2e_sim_debug.md`, `{memory}/project_mbv2_throughput_corrected.md`, [`mbv2_spatial_throughput_roadmap.md`](agent_tasks/mbv2_spatial_throughput_roadmap.md), `{memory}/project_mp_increase_deadlock.md` |
| 10. ResNet vs MobileNetV2 case study | [`nn2rtl_supervisor_explanation.md`](nn2rtl_supervisor_explanation.md), [`MILESTONES.md`](../MILESTONES.md), [`nn2rtl_u250_deployment_plan_mobilenetv2.md`](nn2rtl_u250_deployment_plan_mobilenetv2.md), [`mbv2_PRE_VIVADO_DELIVERABLE.md`](agent_tasks/mbv2_PRE_VIVADO_DELIVERABLE.md) |
| 11. Baseline comparison (hls4ml / FINN) | `{memory}/project_tier_a_complete.md`, `{memory}/project_hls4ml_finn_comparison.md`, `{memory}/feedback_hls4ml_bias_quant.md` |
| 12. Agentic-methodology lessons | `{memory}/feedback_vivado_only_when_proven.md`, `{memory}/feedback_fix_everything_no_defer.md`, `{memory}/feedback_universal_diagnostics.md`, `{memory}/project_failure_corpus_works.md`, `{memory}/project_overnight_mbv2_vivado_ready.md`, [`SYSTEM_REVIEW_FINDINGS.md`](../SYSTEM_REVIEW_FINDINGS.md), [`docs/agent_tasks/autonomous_night_log.md`](agent_tasks/autonomous_night_log.md) |

---

*Generated as a thesis navigation aid. All quantitative results are faithful to the source documents; estimates that were later refuted by measurement (e.g. the "1960 BRAM36 / 72.9% fits" projection vs the 174% real synth) are recorded as such to preserve the honest-reporting arc.*
