# TScIT 2026 — nn2rtl Paper Handoff (writer copy)

*Lean, decision-locked extract for the paper author. Generated 2026-06-16 from the full archive (`TSCIT2026_FINDINGS.md`), which retains the changelog, full per-bug prose, run-identity records, and the supersedes trail. Every number here is verbatim with its (source: path); results are tagged DONE/PENDING with the board. Both systems are AI; the human author wrote no RTL and ran no commands.*

---

# nn2rtl — Paper Writer Handoff (lean) — Part 1: Framing & Contributions

*Lean handoff copy for the paper writer. Full archive (unchanged): D:/RTL_LLM_CLAUDE/nn2rtl-repo/docs/TSCIT2026_FINDINGS.md. Every real number is verbatim with its (source: path), tagged DONE/PENDING + board. Two-systems / autonomy framing and the locked RQs are non-negotiable.*

Tag legend: [DONE] complete/verified, [PENDING] not yet measured, [PARTIAL] partially verified. Board = the FPGA the fact is attributed to (Alveo U250 / ZCU104 / n-a).

---

## ★ 2026-06-18 — FINAL HARDWARE STATE & PAPER DECISIONS (authoritative; supersedes the per-section readings below)

Current source of truth after the power measurement and the paper-assembly pass. Full per-network breakdown incl. the dynamic/static power split: `docs/NETWORK_STATS_FINAL_20260618.md`. Assembled paper: `Paper template/Bachelor_s Student Conference Proceedings Paper in LaTeX Template/tscit2026_paper.tex` (8-page body, compiles clean).

**Fmax convention (PAPER decision, D3).** The PAPER reports the **guaranteed** clock for routes that meet timing (Fmax = 1000/period). `NETWORK_STATS_FINAL_20260618.md` and the per-section readings below use the **effective** Fmax = 1000/(period−WNS). Both are on disk; the paper's numbers are the guaranteed ones:
- ResNet-50 (U250): **83.33 MHz** guaranteed (12 ns MET, WNS +0.102) / **14.71 fps**.  [effective 84.05 / 14.84]
- ResNet-8 nn2rtl (ZCU104): **142.86 MHz** guaranteed (7 ns MET, WNS +0.009) / **~9,670 fps**.  [effective 143.04 / ~9,682]
- ResNet-8 FINN max-fold (ZCU104): **300.03 MHz** guaranteed (3.333 ns MET, WNS +0.047) / **~32,555 fps**.  [effective 304.3 / ~33,021; FINN's own analytical estimate 36,169 fps assumes 333 MHz]
- MobileNetV2 (U250): **110.90 MHz** = 1000/(7+2.017); its 7 ns route did NOT meet timing (WNS −2.017), so this achievable value holds under either convention / **93.61 fps**.

**Power (NEW — vectorless Vivado `report_power`, no SAIF → RELATIVE comparison only; confidence Low for nn2rtl, Medium for FINN).** Verified on disk at `output/power/*.rpt`:
- ResNet-50 **16.014 W** (dyn 12.698 / static 3.316) · MobileNetV2 **11.077 W** (dyn 7.932 / static 3.145) · ResNet-8 nn2rtl **7.252 W** (dyn 6.612 / static 0.640) · ResNet-8 FINN max-fold **9.106 W** (dyn 8.372 / static 0.735, build-host).
- In the paper, rounding to confidence ("~7.3 W", "~16.0 W") would match the Low/Medium labels; the table currently prints 4 s.f.

**Efficiency (NEW — derived, no extra synthesis; on the guaranteed-clock fps):**

| Design | fps/W | mJ/inf | fps/kLUT |
|---|---|---|---|
| ResNet-50 (U250) | 0.92 | 1088.7 | 0.012 |
| MobileNetV2 (U250) | 8.45 | 118.3 | 0.290 |
| ResNet-8 nn2rtl (ZCU104) | 1,333 | 0.75 | 62.7 |
| ResNet-8 FINN max-fold (ZCU104) | 3,575 | 0.28 | 510.8 (uses the 63,739-LUT max-fold config) |

Stories: ZCU104 — FINN ≈ **2.68×** more energy-efficient and **8.14×** more area-efficient than nn2rtl (dataflow vs time-mux), and ≈ **3.37×** the throughput for modestly higher power (9.106 vs 7.252 W; II 9,216 vs latency 14,774). U250 — MobileNetV2 ≈ **9.2×** more energy-efficient than the BRAM-bound INT4 ResNet-50. CROSS-DEVICE energy is NOT comparable (U250 ~3.2 W static vs ZCU104 ~0.6 W).

**FINN max-fold is a ROUTED result** (not just analytical): util LUT 63,739 (27.66%), DSP 569 (32.93%), BRAM 64.5 (20.67%), FF 62,499 (13.56%); bottleneck **II = 9,216 cyc** (MVAU_rtl_0). The "36,169 fps @ 333 MHz" is FINN's own analytical estimate; scaled to the achieved clock it is ~32,555 (guaranteed) / ~33,021 (effective) fps. NB: max-fold LUT/DSP/power are a DIFFERENT config from the baseline 25,760-LUT / 1,017-fps resource columns — keep configs separate; the paper labels the cycles cell "9,216 (max-fold II)".

**hls4ml accuracy = 89.10% on the full 10,000-image CIFAR-10 TEST set.** The **89.11%** below is the QKeras **validation** accuracy (1-image difference). All three flows on the same test-set basis: nn2rtl **87.19%**, FINN **86.68%**, hls4ml **89.10%**.

**Accuracy chain re-derived & CONFIRMED 2026-06-18** (independent GPU re-score; logs `output/reports/rederive_*_20260618.log`): ResNet-50 float **80.07** > all-INT4 **79.47** > deployed Config-B **77.07** (per-tensor INT4 collapses to 2.80%; the generic torchvision-V1 **76.1%** is the WRONG reference). MobileNetV2 float **72.67** / deployed **71.27**. ResNet-8 nn2rtl **87.19**. Unchanged from below — confirmed.

**Quantiser (state once; fixes a per-tensor/per-OC ambiguity):** the default is **per-tensor INT8 PTQ**; the DEPLOYED networks use **per-output-channel** scaling — ResNet-8 on 7 of 9 convs, MobileNetV2 on its depthwise layers, ResNet-50 mixed INT3/INT4 per-OC.

**Engine area collapse (precise count):** the **14 heaviest** convolutions as spatial datapaths = **1,803,388 LUT**; the shared time-multiplexed engine that replaces them (and serves all **17** heavy-layer dispatches) = **107,268 LUT** (~17×). Keep the 14 (baseline bucket) and 17 (engine dispatches) distinct.

---

## 0. Locked research framing (paste verbatim, do NOT mark TODO)

**Overarching question (verbatim, locked):** Whether a multi-agent large-language-model pipeline can autonomously generate synthesisable Verilog for deep neural networks, under a hard constraint that all weights remain in on-chip memory with no external DRAM.

**RQ1 (verbatim, locked):** Which stages of model-to-RTL generation — including architecture mapping, RTL synthesis from the network specification, and closed-loop repair — benefit most from LLM assistance, and which still require deterministic or hand-written components?

**RQ2 (verbatim, locked):** What is the resulting implementation quality, in terms of power, performance, and area (PPA), compared with deterministic FPGA-targeted workflows such as FINN and hls4ml?

**RQ3 (verbatim, locked):** What verification is needed to trust LLM-generated hardware, independent of how it was produced: what classes of integration and correctness bugs arise, and what methodology detects them, given that individually-correct modules can still fail once composed into a full design?

---

## 1. Project & contributions

### 1.1 Identity
[DONE] [board: Alveo U250] nn2rtl is an autonomous multi-agent AI system that takes a trained, quantized PyTorch CNN and produces synthesizable Verilog RTL for FPGA, coordinating LLM code-generation/repair agents plus deterministic verification around a TypeScript orchestrator (source: README.md lines 1-20). Thesis subject: agentic LLM-driven NN-to-RTL generation with INT4/INT8 quantization, byte-exact verification, and FPGA deployment on Xilinx Alveo U250 of ResNet-50 and MobileNetV2 — accuracy, fit, P&R, Fmax (source: docs/THESIS_SOURCE_MAP.md:3). The central claim is intentionally strong: LLMs can automate the NN-to-RTL workflow end to end, not merely assist a human with isolated snippets (source: README.md line 18). The RQ2 ResNet-8 / ZCU104 three-way comparison is a separate, later strand.

### 1.2 Autonomy boundary (RQ1 answer — state precisely)
[DONE] [board: n-a] **End-to-end AI authorship (first-class contribution, verbatim).** The human author **wrote no RTL and ran no commands**; the role was research direction, goal-setting, and milestone approval only. Two cooperating AI systems produced everything: **System 1 = the autonomous pipeline** (orchestrator + Cartographer/Foundry/Surgeon agents + self-improve Failure-Classifier/Retrospector + deterministic Assayer) generates and bit-exact-verifies the **per-layer RTL modules**. **System 2 = the integration agent** — a separate **Claude Opus 4.8 Code-agent instance** (the interactive coding-agent class, not the pipeline) — built the shared compute engine, wired/patched the top wrapper, ran all end-to-end byte-exact debugging, executed every command (Verilator/iverilog, Vivado, Python, git), and ran the entire synthesis + place-and-route campaign. System 2 operated **largely** autonomously (long unattended stretches under standing/overnight directives, orchestrating multi-agent workflows incl. adversarial cross-verification) — "largely" not "fully" because the human set goals, approved milestones, and resolved decision forks. Net: AI-built end-to-end, human as director not implementer (source: docs/agent_tasks/autonomous_night_log.md; memory/feedback_autonomous_night_directive.md; memory/project_overnight_directive_20260610.md; docs/THESIS_SOURCE_MAP.md:9,24,32; project framing per author 2026-06-16). Adversarial-workflow evidence: the 8-hypothesis adversarial code-review that isolated bug B21; the 3-skeptic adversarial verification of the ResNet-50 `_final_c14` route failure (source: memory/project_spatialrun_handshake_bug.md L10-24; memory/project_routes_20260612_results.md:14).

### 1.3 Other contributions
[DONE] [board: n-a]
- **Scale:** 50,000+ lines of autonomously generated RTL (source: README.md lines 639-666; line 643).
- **Methodology:** multi-agent orchestration + real MCP tool integration + sim/synth feedback closed-loop autonomous repair + structured Surgeon repair loop and failure taxonomy as contributions in their own right (source: README.md lines 639-666).
- **On-chip-only / no-DRAM result:** a trained quantized CNN taken to a placed-and-routed on-chip FPGA design, weights never leaving on-chip memory.
- **U250 deployment headline:** an LLM-agent pipeline produces per-layer RTL that is bit-exact-verified, integrates into a working hybrid whole-network FPGA design on Alveo U250, and reaches PPA within a defined envelope of an established production baseline (Vitis AI DPU); the improve loop achieves layer-level area compression approaching hand-optimized quality on individual layers; the failure corpus is a research artefact that improves subsequent generations (source: docs/nn2rtl_u250_deployment_plan.md lines 279-283).
- **RQ2 three-flow baseline:** nn2rtl vs FINN vs hls4ml on ResNet-8 / ZCU104.
- **Honesty caveat (keep):** this is **NOT an architectural-novelty** claim (the hybrid is close to the Vitis AI DPU) and **NOT a "we beat DPU on every metric"** claim — the novelty is methodological (source: docs/nn2rtl_u250_deployment_plan.md lines 279-283). Supervisor-approved budget ~EUR 300; estimated LLM API cost ~$190-420 (ResNet-50) / ~$300-580 (MobileNetV2) (source: docs/nn2rtl_u250_deployment_plan.md lines 279-283).

### 1.4 Scope
[DONE] [board: n-a] **Scope IN:** ResNet-50 stem + residual block stack; the legacy .pth export is constrained to stem conv + layer1 = 17 modules (source: README.md lines 51-58; ARCHITECTURE.md lines 320-331). Full network (layer2/3/4/avgpool/fc) noted outstanding (ARCHITECTURE.md lines 469-474). NOTE: later supervisor/deployment docs DO add global_avg_pool/gemm and full-network deployment (esp. MobileNetV2), so README scope is narrower than the current project.
[DONE] [board: n-a] **Scope OUT (per README):** global average pooling, the FC classifier layer, full-chip SoC integration, ASIC backend and tapeout; validation target is FPGA, not ASIC (source: README.md lines 53-58, 72-74).

### 1.5 Devices & budgets — CANONICAL DEVICE TABLE (single copy; §3 will not repeat it)
[DONE] [board: Alveo U250] **Alveo U250** = xcu250-figd2104-2L-e (UltraScale+ VU13P, 4 SLRs): LUT 1,728,000 / FF 3,456,000 / DSP48E2 12,288 / BRAM36 2,688 (= 99.09 Mbit) / URAM 1,280 (source: task brief — U250 budget from project brief, not re-verified on disk this pass).
[DONE] [board: ZCU104] **ZCU104** = xczu7ev-ffvc1156-2-e (Zynq UltraScale+ ZU7EV, speed -2, Temperature Grade E): LUT 230,400 / FF 460,800 / DSP 1,728 / BRAM36 312 / URAM 96 (source: output/resnet8/reports/synth/resnet8_postroute_util.rpt:11,36,41,109,114,125; timing header xczu7ev-ffvc1156, Speed File -2 PRODUCTION 1.30 05-15-2022, Temperature Grade E at output/resnet8/reports/synth/resnet8_postroute_timing.rpt:8-11; newestDate 2026-06-16).

**Pre-RQ2 comparison constraints (2 lines):** FINN cannot ingest PTQ resnet50_int8.pth (needs Brevitas QAT; only public FINN ResNet-50 is W1A2 on U250); hls4ml's QONNX Quant op requires zero_point==0 and scale=power-of-2, and has no published ResNet-50 (biggest documented = ResNet-8 MLPerf Tiny) (source: memory/project_hls4ml_finn_comparison.md L9-16). These constraints drove the native-quant-as-variable RQ2 design; ZCU104 superseded the original ZCU102 (xczu9eg) recommendation. Reference benchmark arXiv:2206.11791.

---

## 2. System & methodology

nn2rtl is an LLM-assisted multi-agent pipeline that autonomously generates verified, on-chip-only (no external DRAM) synthesizable RTL for full CNNs on AMD FPGAs (source: memory/project_thesis_source_map.md; memory/MEMORY.md). Three structural layers (source: README.md lines 39-46, 208-249; ARCHITECTURE.md lines 68-289): (1) Claude Code plugin (`nn2rtl-plugin/`) = agent roles + skills; (2) Agent SDK orchestrator (`sdk/`) = the deterministic TypeScript control plane (@anthropic-ai/claude-agent-sdk); (3) local MCP server (`mcp/`) = exposes synthesis/verification tools to agents.

**Two systems (autonomy framing).** system (1) = the autonomous per-module pipeline (3 LLM agents + deterministic orchestrator + 6 MCP tools); system (2) = the integration agent, a separate Claude Opus 4.8 Code-agent instance that built the engine, wired/patched the top wrapper, and ran all debugging + Vivado via multi-agent workflows — the human wrote no RTL and ran no commands (source: memory/pipeline-current-status.md; docs/THESIS_SOURCE_MAP.md:40-41,24,39). Verification is deterministic (no LLM Assayer); weights are never passed to LLMs (.hex + $readmemh).

### 2.1 Orchestrator (deterministic control plane — system 1)
The pipeline-coordinator is deterministic TypeScript (`sdk/orchestrate.ts`), NOT an LLM. It maintains pipeline_state.json, selects the next action, dispatches the LLM agents, enforces retry limits, runs Vivado on every Assayer-passed module, writes summary reports (source: README.md lines 252-266; ARCHITECTURE.md lines 179-235). Authoritative model assignments (`sdk/config.ts`, full IDs pinned to avoid tier-alias drift): Cartographer = claude-sonnet-4-6; Foundry = claude-opus-4-7; Surgeon = claude-opus-4-7; Failure Classifier = claude-sonnet-4-6; Retrospector = claude-opus-4-7 (source: sdk/config.ts lines 28-53). config.ts is the single source of truth (frontmatter model aliases are ignored by the orchestrator, ARCHITECTURE.md line 82).

### 2.2 Per-agent input -> job -> output

| Agent (system) | Input | Job | Output | Model / cap | Implementing files |
|---|---|---|---|---|---|
| Cartographer (1) | quantized ResNet-50 checkpoint | trace via torch.fx, fold BN into conv, write weight/bias .hex, emit Layer IR | PipelineIR (`output/layer_ir.json`); runs once at startup, bypassed on ONNX path | claude-sonnet-4-6, maxTurns 30 (config.ts line 29) | `nn2rtl-plugin/agents/cartographer.md` + `scripts/generate_golden.py` (via read_weights MCP) |
| Foundry (1) | exactly one LayerIR (+ contract metadata) | generate one synthesizable Verilog module per spec (INT8, signed, $readmemh weights, valid/ready, exact latency, no sim-only constructs) | metadata-only VerilogModule JSON; RTL persisted via write_verilog MCP | claude-opus-4-7, maxTurns 40 (config.ts line 30) | `nn2rtl-plugin/agents/foundry.md` |
| Surgeon (1) | broken module + VerifResult + LayerIR + prior_attempts (failures only) | classify failure, locate faulty lines, rewrite only those, preserve interface | minimally repaired VerilogModule (metadata) | claude-opus-4-7, maxTurns 20; max_retries 2 (config.ts line 31, line 104) | `nn2rtl-plugin/agents/surgeon.md` |
| Failure Classifier (1) | each failed module's evidence | classify into retry-policy category: code_bug, architectural_fit, toolchain_infra, verification_env, unknown | category verdict gating retry policy | claude-sonnet-4-6, maxTurns 4 (config.ts lines 36-40) | FAILURE_CLASSIFIER_CONFIG (LLM call, not a plugin agent) |
| Retrospector (1) | full failure history + Foundry RTL versions + knowledge doc + spec (self_improve only) | inject advisory JSON into existing Foundry session via SDK resume for one final attempt; does NOT write RTL | advisory JSON | claude-opus-4-7, maxTurns 10 (config.ts lines 42-53) | RETROSPECTOR_CONFIG (LLM call, not a plugin agent) |
| Deterministic Assayer (1, NOT LLM) | generated module + LayerIR | write sidecar, run run_iverilog (lint) then run_verilator (full sim, static C++ TB) | Zod-validated VerifResult | deterministic code | `runAssayerDeterministic` in `sdk/orchestrate.ts` (no agents/assayer.md) |
| Improve Foundry (1, improve flow only) | a module already passing Verilator + Vivado | functionally-equivalent rewrite for one target (use-dsp/use-bram/reduce-lut/reduce-ff/improve-fmax/reduce-latency/increase-throughput) | inlined verilog_source JSON | claude-opus-4-7, maxTurns 40 | `nn2rtl-plugin/agents/improve_foundry.md` |

### 2.3 Orchestration order & feedback edges
Flow: start orchestrator -> if no layer_ir.json invoke Cartographer -> init pipeline state (all modules pending) -> loop tick(): invoke_foundry (Foundry then deterministic Assayer) or invoke_surgeon (load broken module, Surgeon, then Assayer) -> classify any failure -> applyVerifResult -> on pass run Vivado synth-only -> Vivado failure feeds back as synthesis_failed -> on terminal write pipeline_summary.json (source: README.md lines 334-388; ARCHITECTURE.md lines 184-201). State machine: pending -> generating -> verifying -> pass | fail_retry (-> generating) | fail_abort; authoritative record `output/pipeline_state.json`. Feedback edges: failure-classifier verdict gates retry policy (code_bug -> retry budget; architectural_fit & unknown -> fail_abort); retry-exhausted code_bug/architectural_fit -> Retrospector advisory -> one final Foundry attempt via SDK resume; the contract walk falls back flat-bus -> tiled-streaming -> dram-backed on exhaustion. self_improve default ON (source: README.md lines 354-367, 554-562; sdk/config.ts lines 87-104; ARCHITECTURE.md lines 196-223).

### 2.4 Self-improving knowledge system & failure corpus
Failure-mode taxonomy = 16 base classes the Surgeon must classify into (integer overflow, sign extension error, bit shift wrong, rounding mode wrong, saturation missing, loop bounds incorrect, array indexing error, port width mismatch, residual addition overflow, missing pipeline register, pipeline latency wrong, reset logic broken, enable signal ignored, scale factor misapplied, bias term missing, batch norm not folded) plus 4 infra classes (synthesis_failed, verilator_timeout, structural_preflight_failed, architectural_unsupported) (source: README.md lines 564-596; nn2rtl-plugin/agents/surgeon.md lines 151-160). Knowledge-doc lifecycle (4-tier): passing modules in self-improve mode write probationary docs, promoted to active after 3 successful users (NN2RTL_DOC_PROMOTION_SUCCESSES default 3), archived on failure; protected patterns pinned (e.g. `knowledge/patterns/protected/13_on_chip_weights.md`) (source: README.md lines 354-367, 554-562; sdk/config.ts lines 87-104). The failure corpus stores broken RTL + failure history so later generations on related networks improve. **Measurable cross-run learning:** backfilling 11 prior failures cut latency error from +480 to +1 cycle on the next attempt (source: memory/project_failure_corpus_works.md).

### 2.5 Deterministic bit-exact Verilator verification loop
A static handwritten C++ testbench (`tb/static_verilator_tb.cpp`) is the reference checker — never agent-generated (avoids the two-bug problem). It reads a JSON sidecar + NN2V binary golden vectors, drives the DUT, checks values, exact pipeline latency, and valid/ready timing (source: README.md lines 151-168, 520-538; ARCHITECTURE.md lines 373-399). **Reference** = the QUANTIZED model's activations (.goldin/.goldout, NN2V v2 binary: 20-byte header + int32 LE words), NOT float32. **Pass gate:** numerical max_error <= 3 (a well-implemented module expects <= 1) AND exact timing (timing_actual_cycles == pipeline_latency_cycles). A 2026-04-26 bit-exact refactor (compute_scale_approx + requantize_fixed_point_int) dropped layer1_0_conv1 max_error from 1 to 0 across all 6,422,528 samples (first_mismatch_index = -1) (source: ARCHITECTURE.md lines 596-606). Weights are never passed to LLMs: Cartographer writes `output/weights/<id>_weights.hex`/`<id>_bias.hex`; Foundry RTL loads via $readmemh (source: README.md lines 102-124; ARCHITECTURE.md lines 341, 352, 520).

### 2.6 Data contracts between agents
Data contracts typed in `sdk/types.ts`, Zod-validated in `schemas.ts`: LayerIR (7 canonical signal names clk/rst_n/valid_in/valid_out/ready_in/data_in/data_out), PipelineIR, VerilogModule, VerifResult, VerificationSidecar, PipelineState; SDK outputFormat and MCP schemas derived from the same Zod via z.toJSONSchema(). Widths: input_width_bits = in_channels*8 (conv/relu), = in_channels*16 (add lhs+rhs packed); output_width_bits = out_channels*8. Contract set (complexity order): flat-bus, tiled-streaming, dram-backed-weights, activation-double-buffering, weight-tiling — all five now have executable orchestrator plans (source: README.md lines 389-481; ARCHITECTURE.md lines 121-162, 443-461; SYSTEM_REVIEW_FINDINGS line 25).

### 2.7 INT4-GPTQ / quantization recipe & precision blast radius
Scheme A (INT4 weights / INT8 activations) edits ~9 shared files vs 60+ for full INT4 (Scheme B), leaving requant_pipeline.v untouched; weights nibble-packed 2/byte; accumulators kept 32-bit (source: docs/agent_tasks/phase2_int4_per_oc_plan.md; int4_imagenet_timemux_autonomous_plan.md). The engine weight-latency bug (1-cyc pipeline vs 2-cyc deployment URAM) was root-caused + fixed WEIGHT_RD_LATENCY=2 (commit 8677bc0); per-OC requant landed (scale base_words == bias base_words). Stale-derived-artifact bug class (INT4 regen): source weights/adds updated but derived artifacts (wide mp_k packings, Style-A and Style-B residual-add fusion constants) left stale — 3 instances fixed (source: docs/agent_tasks/autonomous_night_log.md).

### 2.8 On-chip-only / no-DRAM enforcement (and why URAM is excluded)
Memory policy is on-chip only, no external DDR (supervisor constraint): all weights live in on-chip UltraRAM/BRAM and all activations in BRAM; the dram-backed-weights contract is a separate research artefact, not the deployment mechanism (source: docs/nn2rtl_u250_deployment_plan.md lines 12-13, 42-64; ..._mobilenetv2.md lines 12-13, 32-50). U250 board memory = 64 GB DDR4 (4 x 16 GB), no HBM (source: memory/pipeline-current-status.md L25). **Why URAM is excluded from weights:** UltraRAM CANNOT be initialized with non-zero values on the U250 (xcu250 / VU13P), proven on Vivado 2025.2 three independent ways — ram_style=ultra + $readmemh -> WARNING [Synth 8-10226] falls back to BRAM (0 URAM / 1 RAMB36E2); XPM ultra + MEMORY_INIT_FILE -> [Synth 8-12183] ignored (0 URAM / 8 RAMB36E2); URAM288_BASE has 0 content-init params vs RAMB36E2's 384 INIT_xx params (source: memory/project-uram-no-init.md L12-21). This hardware constraint (UG573) forces the design BRAM-bound. INT4 weight footprint is the binding fit lever: BRAM bitstream-initializable capacity ~99.1 Mbit (2688 BRAM36); INT4 weights total 93.8 Mbit (spatial 53.4 + engine 40.4); nibble-packing is binding; engine time-mux gives ZERO BRAM saving (no DRAM, URAM can't init). Runtime buffers (FIFO/act/line-buf ~40 Mbit) map to URAM zero-init and don't compete for BRAM (source: docs/agent_tasks/int4_imagenet_timemux_autonomous_plan.md).

### 2.9 Per-network architecture facts (system 2 — Claude Opus 4.8 integration agent)
> These describe the agent-built shared engine, top wrapper, and deployed netlists — all produced by system (2), not a human. Detailed P&R/accuracy results live in later RQ sections.

**ResNet-50 INT8 design (system 2):** 119 layer modules; 17 engine dispatches (9x 3x3 + 8x 1x1, LAST_DISPATCH=5'd16); K_PAR=8; ENGINE_WGT_W=3 (INT3 engine weights); 8 weight banks (768-bit lines = 8 taps x 96 b tap-major, depth 8,384, weight bus 6,144 b) (source: docs/NETWORKS_DATA_ANATOMY.md:199-204 + output/rtl/nn2rtl_scheduler.v:3 + output/rtl/nn2rtl_top.v:447,456).

**MobileNetV2 design (system 2):** 99 layer modules; 51 engine dispatches = 34 dense 1x1 pointwise + 16 depthwise (12 stride-1 + 4 stride-2 quartet-fill) + 1 FC (node_linear @ dispatch 50, LAST_DISPATCH=6'd50); depthwise-on-engine (16/17 DW on engine, one spatial DW = node_conv_812); engine K_PAR=8, ENG_PIPE=1, ENABLE_DEPTHWISE=1 (source: docs/NETWORKS_DATA_ANATOMY.md §3.7).

**Shared compute engine** = author-described wiring skeleton + LLM-generated sub-blocks (MAC array, requant pipeline, address generator, config register block, BRAM-to-stream bridge): 256 MACs (256 signed INT8xINT8 lanes feeding 256 INT32 accumulators, output-channel-parallel, 2-cycle pipeline); 3-stage requant (bias-add -> scale-multiply -> scale-shift+saturate, valid_out +3 cycles, scale_mult 32b, scale_shift 6b); weights in UltraRAM, no AXI4-MM to DDR (source: docs/agent_tasks/00_engine_skeleton_spec.md lines 16, 38-47; 07_engine_mac_array.md lines 31-38, 75; 08_engine_requant_pipeline.md lines 33-47).

### 2.10 On-chip weight memory generation
Weight-memory map generator + on-chip-weights contract (288-bit URAM word, URAM_READ_LATENCY_CYCLES = 2, conv2d only) + skip-FIFO sizing: see archive §2.10.

### 2.11 Deployment decomposition & repro entry points
Wave-grouped agent task briefs (tasks 00-13), pre-pipeline human-run steps, multi-network ONNX-frontend layout: see archive §2.11/§2.12.

### 2.12 Run-identity records
ResNet-50 and MobileNetV2 synth run identities (Vivado v2025.2, part xcu250-figd2104-2L-e), DCP/cycle records: see archive §2.13.

---

## 3. Experimental setup

Tool stack, the LLM-driven multi-agent pipeline, networks and roles, per-network quantization + datasets, and metric definitions as actually implemented. Every number carries its (source: ...). Tagged DONE/PENDING/PARTIAL with board.

### 3.1 Tool stack and versions

**nn2rtl sim/synth tooling (both systems).** Icarus Verilog `iverilog -g2012` (syntax/lint); Verilator + static C++ testbench (`--cc --exe --build`, `--threads`) for functional + timing verification; Vivado `synth_design` for synthesis validation (LUT/FF/DSP/BRAM + Fmax/WNS) (source: README.md lines 482-622; ARCHITECTURE.md lines 275-280). MCP exposes five tools — `run_iverilog`, `run_verilator`, `run_vivado`, `read_weights`, `write_verilog` (plus `get_rtl_patterns` added later) (source: README.md lines 482-622; ARCHITECTURE.md lines 275-280). **DONE. Board: n-a.** Exact iverilog/Verilator version strings TODO — author to supply.

**nn2rtl ONNX frontend.** Universal ONNX-graph frontend (`scripts/onnx_frontend.py`, April 2026) pinned to onnx 1.21.0, onnxsim 0.6.2, onnxruntime 1.23.0, `torch.onnx.export` opset 18; supports Conv2d (BN folded by onnxsim), ReLU, Add (residual), MaxPool2d (source: memory/project_onnx_frontend.md). **DONE. Board: n-a. newestDate: 2026-04.**

**Cross-flow tool versions (RQ2 ResNet-8)** (source: resnet8_postroute_timing.rpt line 3; rq2_resnet8/finn/build_resnet8_zcu104.py line 1; rq2_resnet8/hls4ml/FLOW_NOTES.md lines 5-7; rq2_resnet8/training/qkeras/requirements-pin.txt lines 14-26):
- **nn2rtl:** Vivado v2025.2 (Build 6299465, 2025-11-14) synth + P&R.
- **FINN:** FINN v0.10.1 bare-metal, Vivado/Zynq flow.
- **hls4ml:** hls4ml 1.3.0 (Vitis backend) + Vitis HLS 2024.2 + Vivado 2024.2.
- **Training (hls4ml leg):** QKeras 0.9.0 / tensorflow-cpu 2.15.1.

**DONE. Board: ZCU104. newestDate: 2026-06-16.** *Fairness caveat:* Vivado version differs across tools — nn2rtl 2025.2 vs FINN/hls4ml 2024.2; flag in thesis for strict fairness.

### 3.2 LLM model IDs (per pipeline agent) and the two-systems split

**Per-agent model IDs.** Autonomous pipeline (system 1): Cartographer = `claude-sonnet-4-6`; Foundry, Surgeon, Retrospector = `claude-opus-4-7`; Failure-Classifier = `claude-sonnet-4-6`. Integration agent (system 2) = `Claude Opus 4.8`.

**Two-system attribution (fixed).** (1) The autonomous pipeline — orchestrator + Cartographer/Foundry/Surgeon agents + self-improve Failure-Classifier/Retrospector + deterministic Assayer — generates and bit-exact-verifies per-LAYER RTL modules. (2) The integration agent — a separate Claude Opus 4.8 Code-agent instance working largely autonomously via multi-agent workflows — built the shared engine, wired/patched the top wrapper, did all end-to-end byte-exact debugging, and ran the Vivado campaign (the human author wrote no RTL and ran no commands).

**Parallel-agent deployment contract.** Wave-2 sub-blocks copy port declarations verbatim from `00_engine_skeleton_spec_PORTS.md` (no rename/widen/refactor); `scripts/check_subblock_ports.py` diffs each sub-block `.v` against the PORTS table, any mismatch = hard fail (source: docs/agent_tasks/README.md lines 79-85; 00_engine_skeleton_spec_PORTS.md lines 4-7). **DONE. Board: Alveo U250.** Gate validates names/widths/directions but not spec correctness (e.g. scale width should have been 32 not 16) — motivates a system-level co-sim gate.

### 3.3 Devices and budgets

Device budgets: see §1.5.

### 3.4 Networks and their roles

**Primary — ResNet-50 (full backbone on-chip, U250).** Top interface: single clock; 256-bit AXI4-Stream input (50,176 beats/frame = 224x224; RGB in `[23:0]`, rest ignored); 256-bit output feature map (3,136 beats/frame = 7x7x2048 INT8 pre-pool, 32 INT8 ch/beat); AXI4-Lite control (`s_axil` 32 b addr/data, 4 b wstrb, 2 b resp) (source: docs/NETWORKS_DATA_ANATOMY.md lines 171-195 + output/rtl/nn2rtl_top.v lines 15-26, 52, 4601, 4614). GAP/FC are NOT on-chip — chip emits the pre-pool feature map. **DONE. Board: n-a. newestDate: 2026-06-12.**

Pipeline baseline (system 1): all 119 ResNet-50 modules pass byte-exact Verilator + per-module Vivado on U250; 14/14 heavy convs byte-exact through the shared engine; engine standalone 107k LUT / 1283 DSP / Fmax 190 MHz on U250 (source: memory/pipeline-current-status.md lines 15-17, 32). Architectural win: 14 heavy convs spatial = 1.81M LUT vs shared engine 107k LUT = 17x area collapse via time-multiplexing; without the hybrid the design does not fit U250. **DONE. Board: Alveo U250. newestDate: 2026-05-25.**

**Architectural contrast — MobileNetV2 (U250).** INT8 per-output-channel requantization; depthwise-dominated architectural-contrast network (accuracy/utilization in their own sections). **DONE. Board: Alveo U250.**

**Cross-flow baseline — ResNet-8 (ZCU104, RQ2).** MLCommons Tiny pretrained ResNet-8 (`pretrainedResnet.h5`, sha256 5f938a8e..., commit 1afd2c98... of github.com/mlcommons/tiny), loaded directly; 16/32/64 filters, 78,666 params (source: rq2_resnet8/ACQUISITION.md; rq2_resnet8/LEG_A_STATUS.md lines 110-126). nn2rtl required one model transformation: the two stride-2 3x3 convs with keras-'same' asymmetric pad [0,0,1,1] reformulated as zero-embedded 4x4 kernels with symmetric pad=1 — bit-exact on all 10k images (max logit diff 0.0 vs `F.pad`) at ~1.78x weight cost for those 2 convs (source: rq2_resnet8/ACQUISITION.md; rq2_resnet8/LEG_A_STATUS.md lines 110-126). **DONE. Board: n-a (model). newestDate: 2026-06-12.**

### 3.5 Per-network quantization and datasets

**Number format (default flow).** INT8 symmetric per-tensor PTQ; PyTorch float32 weights quantized before the pipeline starts: zero_point = 0, scale = max(|w|)/127 (source: README.md lines 76-78; ARCHITECTURE.md line 326). BN folded into the preceding conv during extraction; generated hardware never implements standalone BN (source: README.md lines 80-82). **DONE. Board: n-a. newestDate: 2026-04-29.**

**Per-flow quantization (RQ2 ResNet-8).** nn2rtl = INT8 PTQ per-output-channel (no retrain); FINN = W4A4 QAT (Brevitas); hls4ml = W8A8 QAT (QKeras 8-bit quantized_bits) (source: memory/project_rq2_resnet8_results.md line 16; rq2_resnet8/LEG_A_STATUS.md lines 67-80). nn2rtl detail: NN2RTL_WEIGHT_BITS=8, 7/9 convs per-OC, the 2 1x1 shortcut convs + Gemm per-tensor. **DONE. Board: ZCU104. newestDate: 2026-06-16.**

**nn2rtl ResNet-8 import config (Leg A).** NN2RTL_WEIGHT_BITS=8 (INT8 W/A, GPTQ auto-OFF at 8 bits); NN2RTL_STEM_PER_CHANNEL=1 (7 of 9 convs per-OC); NN2RTL_PW_PER_CHANNEL=0 (the two 1x1 stride-2 shortcuts per-TENSOR — per-OC broke byte-exact e2e); NN2RTL_IMAGENET_CALIB=256; NN2RTL_GOLDEN_VECTORS=8. `layer_ir.json` = 21 layers (9 conv2d + 7 relu + 3 add + 1 global_avg_pool + 1 gemm); input_scale 255/127 = 2.0079 (source: rq2_resnet8/LEG_A_STATUS.md lines 67-86, 150-175). `resnet8.onnx` opset 18, input [batch,3,32,32] raw 0..255 NCHW. **DONE. Board: ZCU104. newestDate: 2026-06-12.**

**ResNet-8 dataset / calibration (documented deviation).** INT8 calibration uses CIFAR-10 TEST-set images, mirroring the official MLPerf Tiny scheme via `calibration_samples_idxs.npy` on the test set (500 calib indices in [18,9950]); NN2RTL_IMAGENET_CALIB=256, 8 golden vectors. The 8 golden images necessarily also appear in calibration stats by frontend design (source: rq2_resnet8/LEG_A_STATUS.md lines 88-108). Identical in kind to the upstream MLPerf Tiny reference flow — state explicitly in thesis. **DONE. Board: n-a. newestDate: 2026-06-12.**

**ResNet-50 dataset.** ImageNet validation, 1500-image subset for the headline GPTQ accuracy points (some earlier figures used 512-image val — different eval-set sizes; see accuracy section). CIFAR-10 full test set (10,000 images) is the RQ2 ResNet-8 evaluation set.

### 3.6 Latency contract (as implemented)

README minimum latencies by op: 1x1 conv 3 cycles; 3x3 conv 5 cycles; folded batchnorm stage 2 cycles; ReLU 1 cycle; residual add output_channels+3 cycles (source: README.md lines 132-148). Authoritative latency = `compute_conv2d_latency_cycles` (golden_impl.py); pointwise conv = 1 + OC_PASSES*(mac_parallelism*K_TOTAL + 4); registered-mul_q MAC pass cost = MP*K_TOTAL + 6 (source: README.md lines 132-148; ARCHITECTURE.md lines 343, 638-653). **DONE. Board: n-a. newestDate: 2026-04-26.** *Supersedes:* pre-DSP-refactor pass cost was MP*K_TOTAL + 4; after the registered-mul_q DSP48E1 MREG=1 refactor it is MP*K_TOTAL + 6 (two extra drain cycles); `layer1_0_conv1` latency went 4161 -> 4193 cycles (source: ARCHITECTURE.md lines 646-653). MAX_PARALLEL_MACS default 4 (sdk/config.ts line 115); `mac_parallelism` is an accumulator-group size, not cycle-parallel throughput.

### 3.7 Metric definitions (as implemented)

System-surfaced metrics: `timing_met` / WNS / `fmax_mhz` extracted by `parseVivadoReport` (Fmax from the requested clock); numerical pass = `max_error <= 3`; timing pass = first `valid_out` fires exactly at `pipeline_latency_cycles`; `total_cost_usd` accumulated across all agent calls; per-model `model_usage` token tracking; attempts per module counted for the retry budget (source: README.md lines 504, 534-538; ARCHITECTURE.md lines 213, 233). **DONE. Board: n-a. newestDate: 2026-04-29.**

PPA gates (`evaluateSynthesis`): (1) `timing_met` true; (2) Fmax >= 50 MHz target else `missing_pipeline_register`; (3) `wns_ns` not null and `fmax_mhz` > 0 else `synthesis_failed`. LUT/FF/DSP/BRAM recorded but not hard-failed (source: README.md lines 504, 534-538; ARCHITECTURE.md lines 213, 233). *TODO — author to supply:* formal definitions of "first-shot rate" and "$/run" (derivable from milestones/cost tables).

**fps measurement classes (cross-flow honesty).** The three fps numbers are NOT the same class — nn2rtl fps = **M3** (cycle-accurate Verilator cycles × routed Fmax); FINN fps = its own analytical estimate (clk/max_cycles) / driver throughput; hls4ml = no route (csynth latency estimate only). **M1** = EEMBC on-board; **M2** = on-board Python driver throughput; **M3** = cycle-accurate × routed Fmax (source: memory/project_rq2_resnet8_results.md L50; project_resnet8_3way_plan.md L18).

**Bit-exact pass criterion (end-to-end gates).** All engine-side ResNet/MBV2 optimizations gate on a Verilator `--x-initial 0` e2e byte-exact check; ResNet uses vec0+vec1 (0/100352 = 100352-byte frame), MBV2 uses 8 vectors (8/8 mismatch 0) (source: docs/agent_tasks/RESNET_FINAL_BUNDLE_ANALYSIS.md GATE MATRIX; MBV2_FINAL_BUNDLE_ANALYSIS.md §3). Shared-engine changes additionally require ResNet inertness (cycle-exact 5,664,715 / 5,299,588) and MBV2 inertness (1,184,731 exact) re-gates. **DONE. Board: n-a. newestDate: 2026-06-11.**

---

## 4. Results & data (RQ1, RQ2, accuracy)

> Framing (non-negotiable): evidence separates the **autonomous pipeline** (Cartographer/Foundry/Surgeon + deterministic Assayer) from the **interactive integration agent** (a separate Claude Opus 4.8 Code-agent instance). Both are AI; the human author wrote no RTL and ran no commands. Provenance rule: every number carries its `(source: …)`, tagged DONE/PENDING/PARTIAL with the board. Per repo convention, **post-route `.rpt` files beat route `.json` toplines**.

---

### 4.1 RQ1 — which pipeline stages benefit from LLM assistance

**Per-stage first-shot quality (DONE):**
- ResNet-50 average attempts/state entry **1.09** (≈91% one Foundry attempt, no Surgeon); MobileNetV2 **1.30** (harder DW/ReLU6) (source: docs/nn2rtl_supervisor_explanation.md L423, L451).
- Layer-1 first-shot rate = **17/17 (100%)** after the post-fix refresh, **0% retry** (was 12% original) (source: nn2rtl-repo/MILESTONES.md L29-L34, L82-L88, L136-L139).

**Single-module cost progression (DONE — board Artix-7, layer1_0_conv1):**

| Run | What changed | LLM attempts | Cost | Outcome |
|---|---|---|---|---|
| #1 (broken) | initial | 4 attempts | **$7.50** | fail-abort |
| #2 | hardening (commit decea66) | 1 Foundry + 1 Surgeon | **$1.36** (~5.5× cheaper) | Vivado success |
| #3 | bit-exact + use_dsp + doc honesty | first-shot, no Surgeon | **$0.62** (12× cheaper than #1) | bit-exact pass |

(source: nn2rtl-repo/ARCHITECTURE.md L530-L549, L625-L637). Run #3 bit-identical: `max_error = 0` across 6,422,528 samples, `first_mismatch_index = -1` (source: ARCHITECTURE.md L625-L630). Run #1's wasted attempts were phantom failures (Windows `run_iverilog` non-zero exit with empty stderr) eliminated by hardening commit decea66.

**Full-network LLM cost (autonomous pipeline, DONE):**

| Network | Total LLM cost | $/passing module | Avg attempts/entry | Pass count | Board |
|---|---|---|---|---|---|
| ResNet-50 | **$170.61** (Opus $163.57 / Sonnet $3.49 / Haiku $3.55) | **$1.43** | 1.09 | 119 pass / 0 fail_abort | Alveo U250 |
| MobileNetV2 | **$196.39** (Opus $186.75 / Sonnet $3.65 / Haiku $5.99) | **$2.02** | 1.30 | 97 pass | Alveo U250 |

(source: docs/nn2rtl_supervisor_explanation.md L409-L423, L434-L451). MobileNetV2 = 97 of 99 layers (GAP `node_mean` + Gemm `node_linear` deferred).

**Bit-exactness (PARTIAL — board Alveo U250):** ResNet-50 **119/119 pass in pipeline state**, of which **117/119 carry fresh on-disk evidence** (108 strict bit-exact + 9 within the `<=3` tolerance; 2 stale fail results superseded, 2 missing `.results.json`) (source: docs/nn2rtl_supervisor_explanation.md L130, L414, L721). MobileNetV2: 97 pass = 82 strict + 15 tolerance.

**Failure-corpus measurable learning (DONE — board n-a):** after backfilling 11 prior `node_conv_288` attempts, the next fresh Foundry attempt cut latency error from **+480..+3,588 cycles** (prior 8 attempts) to **+1 cycle**, and shifted `status_class` to a genuinely different design (source: memory/project_failure_corpus_works.md). Improve-command finding: one combined multi-target request is weaker than a controlled single-target sequence (`node_conv_298` reduce-ff **267,918 → 6,254 FF (-97.7%)**; reduce-lut **999,999 sentinel → 89,499 (-91.1%)**, per-module ZCU102) (source: docs/nn2rtl_supervisor_explanation.md L519-L537).

**Tier-A (DONE — board Artix-7):** nn2rtl vs hls4ml, 14/17 stem+layer1 modules, INT8 per-tensor, 50 MHz: nn2rtl ~2-10× less LUT/FF; all three 3×3 conv2 layers unsynthesizable by hls4ml (`[HLS 200-1715]`) (source: memory/project_tier_a_complete.md L7-18).

---

### 4.2 RQ2 — implementation quality vs deterministic flows

**Verified-status, side by side (writer: state each correctly).**

| Network (final netlist) | Simulation byte-exact | Synthesized (U250) | Routed / implemented | Citable Fmax / fps |
|---|---|---|---|---|
| ResNet-50 (U250, INT4/INT3 Config-B) | YES — 5,299,588 cyc, 0/100352 | YES — BRAM 98.81% (binding) | **Prev netlist: YES — timing MET @12 ns; final netlist: route pending (congestion)** | **83.33 MHz / 14.71 fps** (prev netlist, routed); final netlist pending |
| MobileNetV2 (U250, INT8 per-channel) | YES — 1,184,731 cyc, 8/8 | YES | **YES — routed signoff** | 110.90 MHz / 93.61 fps |
| ResNet-8 nn2rtl (ZCU104, INT8 PTQ) | YES — 14,774 cyc, 8/8 | YES | **YES — routed, timing MET @7 ns** | 143.04 MHz / ~9,682 fps |

(MBV2 deployed accuracy = **71.27% top-1 per-channel** — confirmed deployed figure. FINN ResNet-8 numbers are FINN's own analytical estimates from the WSL build host (bitfiles built); hls4ml ResNet-8 is csynth-only (no route, does not fit). nn2rtl ResNet-8 is the only cycle-accurate routed (M3) result.)

#### (a) ResNet-8 three-flow head-to-head on ZCU104 — HEADLINE

**Design (DONE — board ZCU104):** MLPerf-Tiny **ResNet-8** (CIFAR-10, residual adds intact, **78,666 params**, 16/32/64 filters) through three flows on one device (`xczu7ev-ffvc1156-2-e`, budgets LUT 230,400 / FF 460,800 / DSP48E2 1,728 / BRAM Tile 312 / URAM288 96), each at native quantization: nn2rtl **INT8 PTQ per-OC no retrain**, FINN **W4A4 QAT (Brevitas)**, hls4ml **W8A8 QAT (QKeras)** (source: memory/project_resnet8_3way_plan.md L10; project_rq2_resnet8_results.md L4,L10,L48). MLPerf reference 87.0% top-1, closed floor 85%.

| Metric | **nn2rtl** (INT8 PTQ per-OC) | **FINN** (W4A4 QAT) | **hls4ml** (W8A8 QAT) |
|---|---|---|---|
| Accuracy (top-1, CIFAR-10) | **87.19%** (== float ref, PTQ no-retrain) | **86.68%** (W4A4 QAT) | **89.11%** (W8A8 QAT) |
| LUT | **154,188 (66.92%)** | 25,760 (11%) *baseline* | 200,938 (87%) *csynth* |
| FF | **64,728 (14.05%)** | 30,824 (7%) | 100,239 *csynth* |
| DSP48E2 | **1,717 / 1,728 (99.36%)** | 74 (4%) | 488 (28%) *csynth* |
| BRAM | **199 tiles (63.78%)** (RAMB36 194 / RAMB18 10) | ~40 36K-eq (13%) | **1,216 BRAM18 (~194%)** *csynth* |
| URAM | 75 / 96 (78.13%) | 0 | 0 |
| Fmax | **143.04 MHz routed** (7 ns MET, WNS +0.009) | 100 MHz MET (333 MHz at max-fold) | 137.32 MHz *est* (no route) |
| Cycles/frame | **14,774** | ~98,304 (II) | 175,714 *csynth* |
| fps | **~9,682** (M3) | ~1,017 → 3,052 (matched) → ~36,169 (ceiling) *FINN est* | none (no route) |
| Fits ZCU104? | **YES — ROUTED & timing MET** | **YES — bitfile MET** | **NO — over BRAM budget** |
| Status | **DONE (routed, M3)** | **DONE (FINN est, build host)** | **PARTIAL (csynth only)** |

(sources: output/resnet8/reports/synth/resnet8_postroute_{timing,util}.rpt (Jun 16 14:21); rq2_resnet8/hls4ml/CSYNTH_SUMMARY.txt:80-97; memory/project_rq2_resnet8_results.md (2026-06-16); rq2_resnet8/LEG_A_STATUS.md:11)

**nn2rtl routed signoff (DONE — board ZCU104), newest 2026-06-16 14:21:** timing **MET at 7.000 ns**, post-route; setup **WNS +0.009 ns**, hold **WHS +0.011 ns** (both MET; "all user specified timing constraints are met") → **Fmax = 1000/(7.000−0.009) = 143.04 MHz → ~9,682 fps** (= 143.04 MHz / 14,774 cyc) (source: output/resnet8/reports/synth/resnet8_postroute_timing.rpt + resnet8_postroute_util.rpt, Jun 16 14:21). Byte-exact e2e Verilator: PASS, mismatch_bytes 0, beats 8/8, e2e_cycles 14,774 (source: output/resnet8/reports/verilator_resnet8_top_value/result.json). **DSP at 99.36% (1,717/1,728) is the binding resource** — the INT8 all-spatial DSP wall. *Supersedes the earlier 10 ns / 105.49 MHz / ~7,140 fps rung: the FSM pixel-pipeline + stem-accumulator-pipeline + weight-fanout-replication levers lifted Fmax 73.86 → 143.04 MHz at the same 14,774 cycles.*

**nn2rtl optimization trajectory (DONE — board ZCU104), throughput 14.45 → ~9,682 fps (≈670×):** first fitting build 14.45 fps (7,486,125 cyc, MP=4 serialized — full-spatial was 574% LUT / 1.32M LUT) → K_PAR=8 + layer overlap → DSP-pack → TREE4 balanced-adder-tree → **FSM pixel-pipelining** (→ 15,096 cyc, then 14,774) → **stem-accumulator pipelining** (Fmax 77.75 → 122.3 MHz) → **weight-broadcast fanout-replication** (→ 143.04 MHz routed @ 7 ns); byte-exact 8/8 at 14,774 cyc throughout (source: memory/project_rq2_resnet8_results.md, 2026-06-16). The wall is **DSP at 99.36%** (INT8 all-spatial): higher needs W4A4 (FINN's trick), a shared engine, or a bigger device.

**FINN leg (DONE — board ZCU104; FINN analytical estimates, bitfiles built on the WSL host):** accuracy **86.68%** (W4A4). Three operating points, all FIT and timing-MET: **baseline** ~1,017 fps @ 100 MHz — LUT 25,760 (11%), FF 30,824 (7%), DSP 74 (4%), BRAM ~40 36K-eq (13%), URAM 0; **matched-throughput** (target_fps=3000) **3,052 fps** @ 100 MHz — LUT 30,824 (13.4%), DSP 111 (6.4%); **structural ceiling (MAXFOLD)** **~36,169 fps @ 333 MHz** (WNS +0.047) — LUT 63,739 (28%), DSP 569 (33%), BRAM ~28% (input layer in_ch=3 caps further folding). FINN's W4A4 dataflow is ~5–15× more area/DSP-efficient than nn2rtl at equal-or-higher fps (source: memory/project_rq2_resnet8_results.md, 2026-06-16; on-host /root/rq2_training/finn_resnet8/). NB: FINN fps is its own analytical throughput (clk / max_cycles), not cycle-accurate M3.

**hls4ml leg (DONE characterization — board ZCU104; csynth only, no route, DOES NOT FIT):** highest accuracy **89.11%** (QKeras 8-bit; beats nn2rtl's 87.19% float ref) but over budget. Final fitted-attempt csynth: **BRAM ~194% (1,216 BRAM18)**, LUT 200,938 (87%), FF 100,239, DSP 488 (28%), latency 175,714 cyc (source: memory/project_rq2_resnet8_results.md, 2026-06-16). An earlier in-repo csynth config was heavier still — BRAM 234% / LUT 339,196 (147%) / 90,146 cyc, est Fmax 137.32 MHz (source: rq2_resnet8/hls4ml/CSYNTH_SUMMARY.txt:80-97) — both over budget. Root cause = io_stream skip-FIFOs forced to frame depth by Vitis; cosim FIFO-opt empirically impractical (one-frame run hung ~9.5 h with zero progress, killed). No routed result exists for any config.

**Measurement-class honesty (DONE):** the three fps numbers are NOT the same class — nn2rtl fps = M3 (cycle-accurate Verilator cycles × routed Fmax); FINN = analytical estimate; hls4ml = no route (csynth latency only) (source: memory/project_rq2_resnet8_results.md L50).

**Prior-art correction (DONE):** the unqualified "no published same-network ResNet-8 FPGA baseline" is FALSE — fpgaConvNet MLPerf-Tiny v1.1 CLOSED ran the exact ResNet-8 (86.0%); Minnella arXiv:2309.15631 (88.7%); Tailor arXiv:2301.07247 (87.4%). Safe claim: no published FINN ResNet-8; hls4ml's published entries removed skip connections (source: memory/project_resnet8_3way_plan.md L13).

#### (b) Full-network on-chip P&R on U250

**ResNet-50 (Config-B INT4/INT3, 5,299,588-cyc final netlist):**

**ResNet-50 routed-Fmax — canonical statement (use once; lead with the success, don't blur into "the final design runs on hardware"):** ResNet-50 (Config-B INT4/INT3) **did route and close timing on the Alveo U250**. The last successful place-and-route — the previous-netlist build (**5,664,715 cycles/frame**) — **met timing at 12 ns** (setup WNS +0.102 ns, hold WHS +0.010 ns, all constraints met) = **83.33 MHz → 14.71 fps** (routed util LUT 69.23% / BRAM 98.81% / DSP 56.83% / URAM 51.72%; dcp `first_light_routed_kp4mp32_c16.dcp`, source: first_light_postroute_timing_kp4mp32_c16.rpt). The subsequent **FINAL** sealed netlist — the faster **5,299,588-cycle** design — is **byte-exact in simulation** (vec0+vec1, 0/100352) but **has not yet routed**: its P&R failed organically from congestion (22,199 node overlaps), so no routed Fmax exists for that netlist. **Citable result: 83.33 MHz / 14.71 fps routed (previous netlist); the final, faster netlist is simulation-verified, route pending.**

- **Byte-exact DONE:** 5,299,588 cycles/frame, 0/100352 mismatch on vec0+vec1, both PASS; sealed commit 50c3054 (source: docs/NETWORKS_DATA_ANATOMY.md:32,146).
- **Synth banked DONE — board Alveo U250:** CLB LUTs **1,209,699 (70.01%)**; CLB Registers 1,215,675 (35.18%); **BRAM36 tiles 2,656 (98.81%)** — the **binding resource**; URAM288 662 (51.72%); DSP48E2 8,007 (65.16%) (source: output/reports_integrated/first_light_synth.json; docs/NETWORKS_DATA_ANATOMY.md:58-69).
- **ROUTE FAILED — board Alveo U250 (DONE):** final-netlist `_final_c14` failed **organically** — congestion-infeasible, `[Route 35-162]` 24,830 signals failed, **ERROR `[Route 35-2]` 22,199 node overlaps** (source: failed_route_final_c14/vivado_full.log:1288,1370,1393; resume_from_synth.json success=false; docs/NETWORKS_DATA_ANATOMY.md:108-123). Adversarially verified NOT a timeout artifact (vivado.exe routed 85.6 min orphaned past the kill). Forensics: SLR0↔SLR1 SLL columns oversubscribed 114%/104%; 8 of 10 top contended nodes = `u_uram_weight_bank{1,2,6,7}/weight_bus[*]`.
- **NEW route-unblock netlist (DONE synth — board Alveo U250, 2026-06-17): `conv_288→engine` directly attacks the congestion.** Re-synthesized byte-exact: **BRAM 2,464 tiles ≈ 91.7%** (down from 98.81%), LUT **1,129,355 (65.4%)**, DSP 7,879 (64.1%), URAM 658 (source: output/reports_integrated/first_light_synth.json, Jun 17 06:01; verilator_nn2rtl_top_value/result.json PASS, Jun 17 04:33). **Synth-only — not yet routed**; this is the candidate to break the BRAM-density / SLR-congestion wall that failed `_final_c14`.
- **Last MEASURED-routed ResNet (DONE — board Alveo U250) = PREVIOUS netlist `kp4mp32_c16` (5,664,715 cyc):** timing **MET at 12.000 ns**, setup **WNS +0.102 ns** (0 failing); hold WHS +0.010 (MET) → **Fmax 83.33 MHz → 14.71 fps** (source: first_light_postroute_timing_kp4mp32_c16.rpt; docs/NETWORKS_DATA_ANATOMY.md:125-133,155). Routed util: LUT 1,196,343 (69.23%); BRAM Tile 2,656 (98.81%); DSP 6,983 (56.83%); per-SLR BRAM saturated (SLR0 100% / SLR1 99.40% / SLR2 99.78% / SLR3 96.06%).
- **Weight footprint (DONE):** total weights 74.55 Mbit = 75.2% of chip BRAM (spatial ROMs 23.04 Mbit + engine banks 51.51 Mbit); all-INT4 ref = 93.8 Mbit (~94.6%); Config-B's 18 INT3 layers make it fit with margin (source: docs/NETWORKS_DATA_ANATOMY.md:94-102). 119 modules = 36 spatial convs + 17 engine dispatches + 49 ReLUs + 16 adds + 1 maxpool; engine K_PAR=8, ENGINE_WGT_W=3 (INT3).

**MobileNetV2 (INT8 per-channel, 1,184,731-cyc final netlist):**
- **Byte-exact DONE — board Alveo U250:** 1,184,731 cycles/frame, 8/8 vectors, mismatch 0; sealed commit 50c3054. Ladder 7,592,966 → 1,184,731 (-84.4%) (source: docs/NETWORKS_DATA_ANATOMY.md §3 L208,317,372).
- **Routed signoff DONE — board Alveo U250 (NEWEST/BEST = `physopt_aggr_c7`, Jun 15, 7.000 ns):** setup **WNS -2.017 ns**; hold WHS +0.004 (MET) → **Fmax = 1000/(7.000+2.017) = 110.90 MHz → 93.61 fps** (= 110.90 MHz / 1,184,731 cyc). `timing_met=false` at the 7 ns constraint, so 110.90 MHz is the achievable signoff Fmax. **Citable MobileNetV2 routed Fmax** (source: mbv2_route_postroute_timing_physopt_aggr_c7.rpt). A later SSI-SpreadSLLs re-place attempt (`spreadslls_c7`, Jun 15) did **not** improve on this (route returned no closing result, success=false) — aggr_c7 stands.
  - *Prior rung (superseded):* `final_c8` (8.000 ns): WNS -2.199 ns, hold WHS +0.006 (MET) → 1000/(8.000+2.199) = **98.05 MHz → 82.76 fps** (source: mbv2_route_postroute_timing_final_c8.rpt; docs/NETWORKS_DATA_ANATOMY.md:301).
- **Routed util (final_c8) DONE:** CLB LUTs **322,628 (18.67%)**; BRAM Tile 1,812.5 (67.43%); DSP48E2 3,345 (27.22%) (source: mbv2_route_postroute_util_final_c8.rpt; docs/NETWORKS_DATA_ANATOMY.md:302). 99 modules; 51 engine dispatches (34 dense 1×1 + 16 DW + 1 FC); K_PAR=8, ENABLE_DEPTHWISE=1. 8000-bit FC logit output serialized to a 256-bit AXI stream (32 beats × 32 logits) to fit U250's I/O pins.

#### (c) Fmax congestion analysis (routing-bound, not logic-bound)

- **ResNet-50 (DONE — board Alveo U250):** `kp4mp32_c16` routed critical path is **~99% routing-delay-bound with 1 logic level** — worst setup path Data Path Delay 11.960 ns = **logic 0.130 ns (1.087%) + route 11.830 ns (98.913%)**, clock net fanout 1,279,001 (source: first_light_postroute_timing_kp4mp32_c16.rpt:6-54). Logic floor ~0.130 ns ⇒ the wall is placement/routing congestion (weight-bank broadcast), not the datapath.
- **MobileNetV2 (PARTIAL — superseded denser pre-DW-on-engine netlist):** intermediate c8 worst path 12.5 ns = **1.15 ns logic (9%) + 11.4 ns route (91%)**, ~10.06 ns from two unregistered SLR crossings; CLB-site util 95.78% while raw LUT ~76.5% (source: memory/project_overnight_mbv2_fit_fmax_20260607.md). The final leaner netlist (LUT 19.06%) routed to 110.90 MHz; this congestion diagnosis applies to the superseded denser netlist. Both logic floors (~0.130 ns ResNet, ~1.15 ns MBV2) support a routing-bound Fmax thesis — Fmax is gated by congestion, not the datapath.
- **ResNet-8 / ZCU104:** the binding resource is DSP at **99.36% (1,717/1,728)**, not routing; this is the INT8 all-spatial DSP wall (source: output/resnet8/reports/synth/resnet8_postroute_util.rpt, Jun 16 14:21).

#### (d) URAM-not-bitstream-initialisable finding

- **DONE.** PROVEN (Vivado 2025.2): `ram_style=ultra` / XPM ultra + init silently falls back to BRAM; URAM288 has no content-INIT params, so weights → URAM needs runtime load, not `$readmemh` (source: memory/project_uram_no_init.md). Consequence on U250: the 8 MobileNetV2 KPAR8 weight banks fell back to 1,376 RAMB36 (76.1% of all BRAM36); URAM holds only runtime-written activation/FIFO memories. This is why BRAM (not URAM) is the binding ResNet-50 resource at 98.81% (source: docs/NETWORKS_DATA_ANATOMY.md:252-261, 83-92).

#### (e) Historical per-module aggregates

Per-module ZCU102 aggregates (LUT sums etc.) are superseded by the U250 integrated results above — see archive §4.2(e).

> **Misfiled-artifact caution:** the 2026-06-14 `first_light_postroute_*.rpt` files in the U250 checkpoints dir are NOT ResNet-50/U250 — they are a ZCU104 ResNet-8 build misfiled there (Device `xczu7ev-ffvc1156-2-e`) and do not supersede any U250 ResNet-50 data (source: output/reports_integrated/checkpoints/first_light_postroute_util.rpt:8).

---

### 4.3 Accuracy

#### 4.3.1 ResNet-50 (ImageNet)
- **DONE — board n-a (RTL byte-exact ⇒ accuracy = reference).** ResNet-50 **INT4-GPTQ per-output-channel** top-1 = **79.47%** (1500-image ImageNet val + INT8 acts); float baseline **80.07%**; per-tensor GPTQ unusable at **2.80%** (source: docs/agent_tasks/int4_imagenet_FINAL_REPORT.md §2; autonomous_night_log.md:679; THESIS_SOURCE_MAP.md:63,564).
- **DONE — board Alveo U250 (deployed netlist precision mix).** **Config-B (18 INT3 + 35 INT4 conv layers)** top-1 = **77.60% measured** (GPU 1500-img) / **77.07% deployed** (BN-fold harness) — consistent; this is the precision mix of the synthesized 5,299,588-cyc netlist (source: docs/NETWORKS_DATA_ANATOMY.md:32,384; feedback_accuracy_measure_bn_folded.md).
- Accuracy-vs-BRAM sweep: all-INT4 = 79.47%; Config-B = 77.60%; all-INT3 = 69.67% (cliff) (source: memory/project_resnet_route_logic_bound.md:18-19). Per-OC scale is mandatory (per-tensor INT4 = 32.42%/chance).

#### 4.3.2 MobileNetV2 (ImageNet)
- **DONE — board Alveo U250.** Deployed **INT8 per-channel/per-OC** top-1 = **71.27%** (float ceiling 72.67-72.73%), bit-identical at every byte-exact gate; **+4.00%** over the prior per-tensor 67.27% via per-channel depthwise quant (source: docs/NETWORKS_DATA_ANATOMY.md:210,384; memory/project_overnight_mbv2_fit_fmax_20260607.md). Confirmed deployed figure.

#### 4.3.3 ResNet-8 (CIFAR-10) — per flow

| Flow | Quantization | Top-1 | Status | Source |
|---|---|---|---|---|
| nn2rtl | INT8 PTQ per-OC (no retrain) | **87.19%** (8719/10000, == float ref) | DONE | rq2_resnet8/LEG_A_STATUS.md:11; memory/project_rq2_resnet8_results.md |
| FINN | W4A4 QAT (Brevitas) | **86.68%** | DONE (W4A4 trained + bitfile on WSL build host) | memory/project_rq2_resnet8_results.md |
| hls4ml | W8A8 QAT (QKeras) | **89.11%** (epoch 488) | DONE | rq2_resnet8/hls4ml_final/convert_resnet8_final.py:6 |

- **DONE — board n-a.** nn2rtl key finding = **accuracy without retraining** (pure PTQ, 87.19% = float). Argmax torch-vs-ORT agreement 10000/10000, export bit-exact (max logit diff 0.0) (source: rq2_resnet8/LEG_A_STATUS.md:11).

#### 4.3.4 Headline correctness
- **DONE.** Because the datapath is deterministic, **the design is its own reference model** — RTL accuracy = reference accuracy. Both U250 deployment netlists byte-exact: ResNet-50 5,299,588 cyc 0/100352, MobileNetV2 1,184,731 cyc 8/8 (source: docs/NETWORKS_DATA_ANATOMY.md:32,146,210,317). Root cause of the multi-session accuracy bug = the relu-rescale defect (22 of 48 ReLU nodes missing activation requantize + node_add_7 operand-half swap), both fixed (source: docs/agent_tasks/autonomous_night_log.md, 2026-05-30).

---

## 4.4 RQ3 — bug catalogue (appendix-grade, compressed)

RQ3 establishes two cross-cutting methodology claims. **Claim 1 — component byte-exactness does NOT imply end-to-end byte-exactness:** every per-module gate can pass while the composed design is wrong, because isolation goldens are generated *consistent with* the buggy integration assumptions (concat order, activation layout, weight-read latency) and local gates cannot judge spec correctness. **Claim 2 — summary statistics (mismatch %, pass/fail) HIDE outcome-relevant bugs:** a 99.9960% byte-exact layer hid a systematic dropped-MAC error, the same counter-leak gave 0 mismatches on some layers and 8,578 on others (data-dependent on input sparsity), and a 94% "mismatch" was a pure X-init sim artifact while the true residual was 2.72%.

Most catalogued bugs originate from the **integration agent** (the separate, largely-autonomous Claude Opus 4.8 Code-agent instance that built the shared engine, wired the top wrapper, ran all e2e debugging), not the autonomous per-module pipeline — both are AI; the distinction is which AI system produced the artifact. Board context: all simulation-level bugs are tool-level (board = n-a); Vivado synthesizability/route observations are Alveo U250 unless tagged otherwise (ZCU102 improve-sweep, Artix-7 Layer-1). ★ = SPINE bug. 77 distinct bugs (B1..B77); B51 appears under two category lenses.

### The 12 categories (name — count — what works to detect it)

1. **Stale-artifact / regeneration-omission** — 7 (B1,B2,B3,B4,B5,B6,B60) — compare against a freshly regenerated self-consistent golden and check artifact mtimes; run the full ordered regen chain before trusting any e2e number.
2. **Missing / wrong activation-rescale arithmetic** — 2 (B7,B8) — four-step un-confounded bracket (byte-exact 1×1 control, Python recompute = RTL position-exact, goldin triangulation) and fit the fixed-point transfer over the entire input domain, never one vector.
3. **Operand half-swap, scaling & tiling-layout** — 8 (B9,B10,B11,B12,B13,B14,B15,B16) — operand probe on the exact transfer handshake (both inputs byte-exact, output wrong), then offline replay of on-disk goldens with the actual RTL params in both orientations.
4. **Handshake / backpressure / deadlock / lockstep-freeze** — 17 (B17,B18,B19,B20,B21,B22,B23,B24,B25,B26,B27,B28,B29,B30,B31,B32,B33) — live simulation-state inspection (custom Verilator probe of scheduler/bridge/loader regs); freeze signature ready_out=1 ∧ valid_out=0 and impossible occupancy ratios; a self-consistency test (lossless module emits identical values under varying backpressure) separates real datapath errors from handshake artifacts. Static count audits = mostly false positives.
5. **Latency-contract & memory-read-latency** — 4 (B34,B35,B36,B51) — engine-isolation harness with controllable weight-read latency: byte-exact at WLAT=1, exact in-chain error count at WLAT=2, de-confounding the false "simulator artifact" verdict.
6. **Simulation-artifact false-positives** — 4 (B37,B38,B39,B40) — only Verilator `--x-initial 0` (models FPGA GSR) is hardware-faithful; single-threaded Verilator is the only trustworthy gate; lint catches undriven wires; inspect mapped primitive count not the readback.
7. **Contract / bus-width / packing mismatches** — 6 (B41,B42,B43,B44,B45,B46) — RTL port-width read against the contract map (SELRANGE flags width slicing); end-to-end coherent multi-stage patching (stages are mutually tuned to each other's lossy contract).
8. **Engine datapath arithmetic / counter-leak (high byte-exactness masks a systematic error)** — 5 (B47,B48,B49,B50,B51) — single-pixel-isolation TB vs `.goldout`, per-pixel mismatch grid, drop-which-MAC elimination model; map mismatch count to input-activation sparsity to explain why one root cause looks like a multi-layer spectrum.
9. **Component-byte-exact-but-e2e-wrong & verification-method** — 9 (B52,B53,B54,B55,B56,B57,B58,B59,B61) — treat e2e mismatch as the oracle and march a per-node probe bisection downstream from the first wrong node with a known-good control; Python triangulation (recompute layer from goldin+weights+scale = logical goldout) is the load-bearing de-confounder; `equiv_one` unreliable for spatial convs.
10. **Golden-reference float-accumulation artifacts (RTL is more correct)** — 2 (B62,B63) — try two different scale choices (persistent identical residual implicates the golden); integer-hardening only safe where the helper provably reproduces RTL constants over the full domain.
11. **Autonomous-pipeline self-improvement & toolchain hazards** — 12 (B64,B65,B66,B67,B68,B69,B70,B71,B72,B73,B74,B77) — deterministic preflight/structural gates, separate `toolchain_infra` from RTL failures, gate cache promotion on the final Vivado/PPA gate, surface raw numeric evidence with no pre-written hypothesis.
12. **Frontend / accuracy-measurement** — 2 (B75,B76) — measure through the deployment integer path or BN-set-to-identity; confirm the scale ratio equals the per-channel BN fold factor (corr 1.0).

### The 77-bug index (ID | category | net | symptom | fix)

| ID | Cat | Net | Symptom | Fix |
|----|-----|-----|---------|-----|
| B1 | A stale-artifact | RN50 | 14 engine convs systematically low (~16% e2e err), small +bias at final ReLU | rebuild bias_memory_map (stale per-tensor bias.mem); engine real-mem iso harness exonerated RTL |
| B2 | A stale-artifact | RN50 | e2e numbers fluctuated, byte-exact milestones not reproducible, ~16% err hidden | refresh_final_golden.py retiles fresh logical golden into the stale contract golden |
| B3 | A stale-artifact | RN50 | windowed 3×3 convs saturate (max 7f) while 1×1 clean | regen_mp_k_weights.py regenerated stale wide tiled packings ($readmemh'd) |
| B4 | A stale-artifact | RN50 | saturation explosion (add_9 0→7625), relu_48 sat | extend add-rescale patcher to Style-B (FUSED_LHS_MULT) adds it silently skipped |
| B5 | A stale-artifact | both | engine wrong addresses / range-assert under mixed INT3 packing | reorder build_weight_memory_map → dedup_engine_banks → nibble_int3 [SUPERSEDED] |
| B6 | A stale-artifact | RN50 | part of multi-day ~16% e2e bug (test-only) | single-file repack `--wgt-bits 4` (INT3 wide hex read under WGT_BITS(4) wrappers) |
| B7 ★ | B act-rescale | RN50 | diffuse e2e err, conv_200 93.9% wrong, near-orthogonal final map | add per-layer requant to the 22/48 rescaling ReLUs (template emitted pure max(0,x)) |
| B8 | B act-rescale | RN50 | byte-exact on vec0, mismatch bytes on vec1 | re-fit each relu transfer over all 128 inputs via bounding-interval (was vector-overfit) |
| B9 | C halfswap/tiling | RN50 | residual err mean\|d\|≈2.7 onset conv_252 → ~4% at final ReLU | swap reversed data_in halves of node_add_7 (lone offender of 16 adds) |
| B10 ★ | C halfswap/tiling | MBV2 | all 10 residual adds wrong despite both operands byte-exact + aligned | fix concat order to lhs=low=skip / rhs=high=main in build_top_wrapper (was {skip,main}) |
| B11 | C halfswap/tiling | RN50 | tiled adds (OC≥1024) fail ~22%, RTL passes pure-Python sim | add to isTiledAdd + per-beat lhs\|rhs interleave retile (tooling bug, not RTL) |
| B12 | C halfswap/tiling | MBV2 | conv_818 (first DW after dispatch) 100% wrong, stem byte-exact | rewrite g_w_lt to one 2048-bit word/beat + deepen act mem + remap D0/D1 bases |
| B13 | C halfswap/tiling | MBV2 | conv_878 (576-ch) ok to ch319 then wrong from ch320; same on 960-ch | add pull_idx reg decoupling resident-beat index from pull advance (off-by-one is_last) |
| B14 | C halfswap/tiling | RN50 | every weight read wrong, engine garbage | Path D: 8 native-288-bit URAM banks (288-vs-2048 width, 8×288≠2048, >>3 skip) |
| B15 | C halfswap/tiling | RN50 | IC>256 layers re-read ch0; 8/14 heavy layers wrong dot products | read addr = base + (in_r·IW+in_c)·ic_chunks + ic_chunk_idx (was no IC stride) |
| B16 | C halfswap/tiling | RN50 | every bias byte-swapped to huge wrong value (−4→−50,331,649) | struct.pack('>i') big-endian (was '<i', LSByte landed high) |
| B17 ★ | D handshake | RN50 | chain-wide lockstep freeze, no frame ever emitted | comprehensive backpressure (pulse producers asserted valid, ignored ready) |
| B18 | D handshake | RN50 | after backpressuring 25 relus, freeze byte-identical | backpressure must cover every producer (loss relocated to pulse-style conv_200) |
| B19 | D handshake | RN50 | subset of pointwise convs sped up → fast/slow imbalance, beat loss | fix param-grep (31 serial 1×1 convs existed, only 12 parallelized) |
| B20 ★ | D handshake | RN50 | frame never completes; wide 2048-ch ReLUs drop last beat (3135/3136) | drop spatial_run from ReLU→skid transfer so always-ready skids accept every beat |
| B21 | D handshake | RN50 | sparse ~2.7% broad-channel residual, stages 3-4, in-chain only | gate valid_in by spatial_run across 83 skid-fed nodes [SUPERSEDED by B7; kept ~12% cyc] |
| B22 | D handshake | RN50 | MP 16→32 on 38 convs hard-deadlocks e2e (out stuck 0) | proposed symmetric output skid on conv_202→add LHS; kept byte-exact MP=16 baseline |
| B23 | D handshake | RN50 | deadlock entering final stage (skip FIFOs 0-6 full, 7-15 empty) | revert one-sided tail-pipe (+3 spatial vs +0 engine desynced the add join) |
| B24 | D handshake | MBV2 | dispatch-1 deadlock, bridge waits inflated tile count | set engine_output_bridge TILES_PER_BEAT=1 (engine writes 1 position/2048-bit beat) |
| B25 | D handshake | MBV2 | dispatch 21 deadlock, loader at full cap from ~17 real positions | expose wsel_empty as wr_accept, re-gate 19 retile-bridge producers (duplication) |
| B26 | D handshake | MBV2 | engine-top parks S_WAIT_DRAIN, drain_complete never asserts | param-gated output-backpressure primitive (store-and-drop FIFO overflowed, dropped beats) |
| B27 | D handshake | MBV2 | engine-top deadlock dispatch 0, scheduler stuck S_WAIT_LOAD | apply_loader_word_resize.py (capacity in output beats vs 2048-bit words, off by 2048/BUS_W) |
| B28 | D handshake | MBV2 | e2e deadlock with backpressure retile bridges | per-bridge spatial_run_drain_br_i mask (shared global any_retile_stall dropped beats) |
| B29 | D handshake | MBV2 | e2e timeout when all 17 DW at MP=16 (also MP=8) | revert conv_884/908 to MP=4 [SUPERSEDED — MP16 later byte-exact once bridges deleted] |
| B30 | D handshake | RN50 | engine FSM enters ST_REQUANT, never advances, deadlock | remove 8 datapath tie-offs outside the ifndef guard, wire from addr-gen + bias mem |
| B31 | D handshake | RN50 | AXI4-Lite write never completes, engine never configured | collapse S_WRITE_ADDR+S_WRITE_DATA into one S_WRITE asserting awvalid+wvalid same cycle |
| B32 | D handshake | RN50 | engine never starts, design does nothing on input | add scheduler instance, wire AXI master→slave, drive engine_start (was tied 0) |
| B33 | D handshake | MBV2 | stem freeze / e2e never completes, ~50% beats dropped | elastic-pipeline rollout across all spatial producers (was push-only, no backpressure) |
| B34 ★ | E latency-contract | RN50 | small ±1..±12 in-chain errors on engine convs, iso sims passed | align engine to 2-cycle weight read (URAM READ_LATENCY_A=2 vs 1-cyc pipeline = stale weight) |
| B35 | E latency-contract | MBV2 | engine output catastrophically wrong, 2048-bit bus truncated to 1024 | override engine params WGT_W=8/URAM_DATA_W=2048/WLAT=2 (defaulted to ResNet INT4) |
| B36 | E latency-contract | RN50 | clean mismatch w/ timing_pass=true, ~0.5-1% sign-skewed | change prefetch guard `oc_pass+2 < OC_PASSES` to `<=` (stale double-buffered cache) |
| B37 ★ | F sim-artifact | RN50 | deep conv reads ~94-96% wrong vs golden in many harnesses | run Verilator `--x-initial 0` (uninit FF X-poisoning; true residual only 2.72%) |
| B38 | F sim-artifact | MBV2 | ~688 wrong logit bytes under `--threads 4`, cycle count unchanged | pin runner to `--threads 1` (multithread scheduler cross-partition hazard) |
| B39 | F sim-artifact | RN50 | long-standing ±1 error on spatial conv_216 | drive undriven window_kwm1_wire in line_buf_window.v (lint-flagged UNDRIVEN) |
| B40 | F sim-artifact | RN50 | functional sim reads correct URAM, falsely implies init works | inspect mapped primitive count (synth fell back to RAMB36E2, sim exercised BRAM not URAM) |
| B41 ★ | G contract/buswidth | MBV2 | both tops consume all input, produce no output (final-stage deadlock) | contract-regen major phase: retile bridges (576/960/1280-ch > 4096-bit flat-bus cap) |
| B42 | G contract/buswidth | MBV2 | correcting one high-OC stage deadlocks the next; chain regresses | coherent multi-stage contract patching [SUPERSEDED — region later byte-exact] |
| B43 | G contract/buswidth | RN50 | every layer's requant output garbage (scale hi bits discarded) | widen SCALE_MULT_W 16→32 (ResNet scales ~30 bits, e.g. 1284434803) |
| B44 | G contract/buswidth | RN50 | ±1 errors on rounding boundaries accumulating through depth | apply_scale_constant_fix.py re-derive 21 convs' mult/shift [SUPERSEDED engine ROM; spatial kept] |
| B45 | G contract/buswidth | RN50 | engine FSM never leaves requant for layers narrower than max OC | change ST_REQUANT exit to oc_pass_idx == oc_pass_total_m1[2:0] (was hardcoded ==7) |
| B46 | G contract/buswidth | RN50 | SELRANGE warnings (×10) on 1×1 / non-MP_K==9 conv instances | gate channel_select assign in generate (USE_CHAN_WINDOW=1 else tie 0) |
| B47 ★ | H counter-leak | RN50 | layer 99.9960% byte-exact, 2/50,176 wrong at pixel [1,4] (off by −1) | re-gate weight/act read on `~mac_done` not `~k_at_last` (suppressed last MAC tuple) |
| B48 ★ | H counter-leak | RN50 | OCs off by ic=0 contribution, mismatch 2..8578 on 9/14 layers | wrap counter-advance in `if(!mac_done)` (ic_cnt leaked 0→1, dropped first MAC) |
| B49 ★ | H counter-leak | RN50 | same bug = 0 mismatches on some layers, thousands on others | (B48 fix) — visibility is data-dependent: mismatch tracks count of a[ic=0]≠0 pixels |
| B50 | H counter-leak | RN50 | engine convs off-by-one (got=gold−1) on negative half-boundaries | unconditional `+HALF` round-half-up (was sign-aware HALF / HALF−1) |
| B51 | E/H latency+counter | RN50 | engine dropped last MAC of every output-channel pass | re-gate counter-advance by `~mac_done` not `~k_at_last` (one bug, two category lenses) |
| B52 ★ | I local-pass/e2e-wrong | both | sub-blocks each pass local gates but don't work wired together | 3 audit rounds fixed 12 cross-piece bugs (AXI/scale-trunc/URAM-bus/bias/FSM-exit/byte-order) |
| B53 ★ | I local-pass/e2e-wrong | MBV2 | modules byte-exact in isolation, integrated chain corrupts from 1st dispatch | march per-node e2e probe (iso goldens were consistent-with-the-bug); e2e mismatch is oracle |
| B54 | I local-pass/e2e-wrong | RN50 | byte-correct per-OC conv still reports mismatch via equiv_one | DBG_SCALE dump + Python triangulation localized to tiling path; mandate in-chain e2e |
| B55 | I local-pass/e2e-wrong | RN50 | all conv outputs zero (got=0) in sim | make SCALE_PATH absolute (relative path didn't resolve in sim cwd → scale ROM zero) |
| B56 | I local-pass/e2e-wrong | RN50 | hard all-zero from first engine-fed residual add onward | run executable with cwd at repo root ($readmemh relative paths not found → mems zero) |
| B57 | I local-pass/e2e-wrong | RN50 | known-good conv_198 equiv_one max_error 75, design ok in-chain | triangulate_conv198.py recompute matched logical goldout → fault is RTL path only |
| B58 | I local-pass/e2e-wrong | RN50 | earlier review concluded deployed weights were INT8 | adversarial re-scan: _wide INT8 files are dead (engine computes from INT4 URAM bank) |
| B59 | I local-pass/e2e-wrong | MBV2 | conv_820 wrong despite byte-exact spatial input, ~99% downstream | set scheduler act_out_base[D1]=12544 (dead act_out write collided w/ ldr2/ldr3 windows) |
| B60 | A/I stale-golden | RN50 | conv_248 stage-3 residual off ±1-6 (equiv_one max_error up to 51) | verify known-good stage-1/2 conv vs fresh INT4 goldens; deferred pending consistent regen |
| B61 | I local-pass/e2e-wrong | MBV2 | per-module off-by-one (max_error=1) on 17 of ~17 flagged | canonical compute_scale_approx + unconditional bias + shift bumps |
| B62 | J golden-float-acc | MBV2 | node_linear 2/8000 off-by-one logits | harden golden_impl Int8Gemm to integer requantize (golden cast acc to float32, RTL is correct) |
| B63 | J golden-float-acc | MBV2 | generic integer-hardened golden emits bytes RTL won't (119 vs 118) | harden ONLY node_mean (7619,18); flag+skip relu/add (per-module agent-chosen constants) |
| B64 | K self-improve/toolchain | both | working contract permanently flagged manual_correction_needed | add toolchain_infra class + retryable fallbacks (transient infra ≠ RTL failure) |
| B65 | K self-improve/toolchain | both | contract switched but verified against flat-bus goldens (wrong protocol) | executable plans + per-contract golden retile/repack + latency [SUPERSEDED — now executable] |
| B66 | K self-improve/toolchain | both | later degraded repair becomes cached state, better design lost | gate cache promotion on module still in pass after Vivado/PPA gate |
| B67 | K self-improve/toolchain | MBV2 | grouped/depthwise/dilated conv verifies against wrong math | add groups/dilation fields to LayerIR + depthwise detection + asymmetric-pad rejection |
| B68 | K self-improve/toolchain | both | self-improvement can't tell which learned docs a module used | deterministic pattern injection + hard-fail/retry when required lookup absent |
| B69 | K self-improve/toolchain | both | valid MaxPool omitted by stale Cartographer instructions | route extraction through deterministic read_weights [SUPERSEDED] |
| B70 | K self-improve/toolchain | both | file looks stale/missing depending on Win/WSL/native opener | normalize all path forms at tool boundaries + content-hash fingerprint identity |
| B71 | K self-improve/toolchain | RN50 | pipeline repeatedly repairs correct RTL (phantom failures, $7.50) | harden run_iverilog to emit structured no-diagnostic; classify as tb_setup_error/toolchain_infra |
| B72 ★ | K self-improve/toolchain | both | Verilator passes but synth preflight fails async-reset array write | split memory write into dedicated `always @(posedge clk)`-only block (restore RAM inference) |
| B73 | K self-improve/toolchain | both | Verilator never terminates, hits wall-clock cap | add VERILATOR_SIM_TIMEOUT_MS + output_counter_missing structural preflight rule |
| B74 | K self-improve/toolchain | RN50 | stem first_mismatch ≈7122, max_error 1-3, errors in last quarter | move wrap math (IW−1+PW) into coord_scheduler.v + right-pad from zero BRAM cells |
| B75 | L frontend/accuracy | RN50 | measured top-1 ~0% or misleading ~73% → false "design broken" | measure via onnx_frontend int path / BN-identity injection (ONNX is BN-folded, double-counts) |
| B76 | L frontend/accuracy | both | Vitis HLS launches, idles indefinitely at ~0% CPU, no csynth | plain float ONNX + per-layer ap_fixed (move_scales explodes scalar→per-element constants) |
| B77 | K self-improve/toolchain | both | repair agent steered to add drain logic, real bug = off-by-one exit | replace TB prose hypothesis with raw mismatch numbers + general interpretation rubric |

---

# TScIT 2026 — Paper-Writer Handoff (Status / Reproducibility / Citations)

Lean, decision-locked extract for the paper writer. Full archive (unchanged): D:/RTL_LLM_CLAUDE/nn2rtl-repo/docs/TSCIT2026_FINDINGS.md. Every number is verbatim with its (source: path); DONE/PENDING + board tagged. Two-systems / autonomy framing and the §0-locked RQs are non-negotiable.

Two-systems framing (attribution rule): System (1) = the autonomous deterministic pipeline (orchestrator + Cartographer/Foundry/Surgeon + Failure-Classifier/Retrospector + deterministic Assayer) that generates per-module RTL. System (2) = a separate Claude Opus 4.8 Code-agent instance that built the shared engine, integrated the full network, ran all debugging + every command + the whole Vivado campaign, largely autonomously via multi-agent workflows under standing human direction. Both systems are AI; the human author wrote NO RTL and ran NO commands.

---

## 5. Status (sim-verified vs routed-implemented, per RQ)

**Headline:** RQ1 DONE (autonomous per-module generation + integration). RQ2 ResNet-8 three-flow ZCU104 — nn2rtl DONE/routed (143.04 MHz, ~9,682 fps); FINN DONE (bitfiles, analytical fps ~1,017→36,169); hls4ml DONE characterization (csynth-only, does not fit). RQ2 U250 — MobileNetV2 routed (110.90 MHz); ResNet-50 simulation-verified, best routed = previous netlist 83.33 MHz / 14.71 fps, final netlist route FAILED (congestion; new conv_288→engine netlist re-synthesized at ~92% BRAM, route pending). RQ3 DONE (77-bug catalogue + methodology).

**ResNet-50 routed-Fmax — canonical statement (use once; lead with the success, don't blur into "the final design runs on hardware"):** ResNet-50 (Config-B INT4/INT3) **did route and close timing on the Alveo U250**. The last successful place-and-route — the previous-netlist build (**5,664,715 cycles/frame**) — **met timing at 12 ns** (setup WNS +0.102 ns, hold WHS +0.010 ns, all constraints met) = **83.33 MHz → 14.71 fps** (routed util LUT 69.23% / BRAM 98.81% / DSP 56.83% / URAM 51.72%; dcp `first_light_routed_kp4mp32_c16.dcp`, source: first_light_postroute_timing_kp4mp32_c16.rpt). The subsequent **FINAL** sealed netlist — the faster **5,299,588-cycle** design — is **byte-exact in simulation** (vec0+vec1, 0/100352) but **has not yet routed**: its P&R failed organically from congestion (22,199 node overlaps), so no routed Fmax exists for that netlist. **Citable result: 83.33 MHz / 14.71 fps routed (previous netlist); the final, faster netlist is simulation-verified, route pending.**

### RQ1 — autonomous pipeline + full ResNet-50 INT4 on-chip
**DONE:**
- Pipeline generated + verified per-layer RTL for full ResNet-50: 119 layer modules = 36 spatial convs + 17 engine-dispatched convs + 49 ReLUs + 16 residual adds + 1 maxpool (source: docs/NETWORKS_DATA_ANATOMY.md:199-204). Board: Alveo U250 (modules), Artix-7 (Layer-1 gate).
- Layer-1 (17-module first stage) e2e: 17/17 pass, 0/17 Surgeon retries after the add refresh, Verilator bit-exact (max_error ≤ 2), Vivado synth/timing_met at 50 MHz (source: MILESTONES.md:17-117). Board: Artix-7 (xc7a100tcsg324-1).
- ResNet-50 FINAL netlist byte-exact at 5,299,588 cycles/frame (vec0+vec1, 0/100352 mismatching bytes, both PASS), sealed commit 50c3054 (source: docs/NETWORKS_DATA_ANATOMY.md:32,146; resnet_final_bundle/e2e_waddr_rep_vec{0,1}.log). Board: Alveo U250.
- Final-netlist post-synth util: CLB LUTs 1,209,699 (70.01%), CLB Registers 1,215,675 (35.18%), BRAM36 tiles 2,656 (98.81%), URAM288 662 (51.72%), DSP48E2 8,007 (65.16%) (source: docs/NETWORKS_DATA_ANATOMY.md:58-69). Board: Alveo U250. (BRAM36 98.81% is the binding resource.)
- Last MEASURED-routed result (PREVIOUS netlist kp4mp32_c16, 5,664,715 cyc): timing MET at 12.000 ns, setup WNS +0.102 ns, hold WHS +0.010 ns (MET) → 83.33 MHz = 14.71 fps (source: docs/NETWORKS_DATA_ANATOMY.md:5,125-133,140,155; first_light_postroute_timing_kp4mp32_c16.rpt). DONE [ROUTED, superseded netlist]. Board: Alveo U250.

**PENDING / PARTIAL:**
- FINAL netlist (5,299,588 cyc) route attempt _final_c14 FAILED — congestion-infeasible (organic, 22,199 node overlaps; Route 35-447 congestion surrender; adversarially verified NOT a timeout) (source: output/reports_integrated/failed_route_final_c14/vivado_full.log:815,1165,1288,1370,1393; resume_from_synth.json success=false; docs/NETWORKS_DATA_ANATOMY.md:118-123). No routed Fmax for the final netlist. Board: Alveo U250.
- AggressiveExplore re-place left no surviving routed artifact; the 2026-06-14 first_light_postroute_*.rpt in the U250 checkpoints dir are a MISFILED ZCU104 ResNet-8 build (Device xczu7ev-ffvc1156-2-e), NOT U250 ResNet-50 (source: output/reports_integrated/checkpoints/first_light_postroute_util.rpt:8; first_light_postroute_timing.rpt:8,306,314). Board: ZCU104 (misfiled).
- Final-netlist throughput PARTIAL (routed Fmax unknown): 5,299,588 cyc → 15.72 fps @83.33 MHz [TARGET], 13.48 fps @71.43 MHz (14 ns), 11.79 fps @62.50 MHz (16 ns); only measured-routed throughput is 14.71 fps @5,664,715 cyc on the previous netlist (source: docs/NETWORKS_DATA_ANATOMY.md:146-156; resnet_final_bundle/e2e_waddr_rep_vec0.log). Board: Alveo U250.

### RQ2 — MobileNetV2 (U250) + ResNet-8 three-flow baseline (ZCU104, CIFAR-10)
**MobileNetV2 U250 — DONE:**
- Byte-exact at 1,184,731 cycles/frame (8/8 vectors, total mismatch 0, out_beats 32), sealed commit 50c3054 (source: docs/NETWORKS_DATA_ANATOMY.md:210,317). Board: Alveo U250.
- ROUTED + signed off. NEWEST/BEST = physopt_aggr_c7 (Jun 15) @7 ns: setup WNS −2.017 ns, hold WHS +0.004 ns (MET) → Fmax 110.90 MHz = 93.61 fps (source: output/mobilenet-v2/reports/synth/checkpoints/mbv2_route_postroute_timing_physopt_aggr_c7.rpt). (Superseded final_c8: 98.05 MHz / 82.76 fps @8 ns, source: docs/NETWORKS_DATA_ANATOMY.md:6,301,313,321.) Board: Alveo U250.
- Synth util LUT 329,371 (19.06%) / FF 437,671 (12.66%) / BRAM36 1,812.5 (67.43%) / URAM288 235 (18.36%) / DSP48E2 3,345 (27.22%); routed LUT 322,628 (18.67%) (source: docs/NETWORKS_DATA_ANATOMY.md:236-248,302). Board: Alveo U250.
- INT8 per-channel top-1 71.27% (float ceiling 72.73%) (source: docs/NETWORKS_DATA_ANATOMY.md §3 L210; §3.7 L384). Board: Alveo U250.

**MobileNetV2 U250 — PARTIAL/PENDING:**
- timing_met=false at the 8 ns and 7 ns constraints; quoted Fmax are signoff Fmax, not met constraints. A clean "met" needs a ~10 ns re-route or the TREE_STAGES=1 KPAR8 adder-tree lever (source: docs/NETWORKS_DATA_ANATOMY.md:388-398).
- Parked levers: FRAME-PIPE (~30% throughput, deadlock-adjacent) and TREE_STAGES=1, documented not implemented (source: NETWORKS_DATA_ANATOMY.md §3.5; docs/agent_tasks/PAIR812_ANALYSIS.md:26-27).

**ResNet-8 three-flow (ZCU104 = xczu7ev-ffvc1156-2-e):**
- **nn2rtl (Leg A): DONE — ROUTED.** Newest postroute (2026-06-16 14:21, speed −2 PRODUCTION, Physopt postRoute) closes @**7.000 ns**: setup WNS **+0.009 ns**, hold WHS **+0.011 ns** (MET), all constraints met → routed Fmax = 1000/(7.000−0.009) = **143.04 MHz** (source: output/resnet8/reports/synth/resnet8_postroute_timing.rpt, Jun 16 14:21). Byte-exact 8/8 at 14,774 cycles/frame (source: verilator_resnet8_top_value/result.json). Util LUT **154,188 (66.92%)** / FF 64,728 (14.05%) / DSP **1,717 (99.36%)** / BRAM tile 199 (63.78%) / URAM 75 (78.13%) (source: output/resnet8/reports/synth/resnet8_postroute_util.rpt). Throughput @143.04 MHz ≈ **9,682 fps** (M3). Accuracy 87.19% (source: memory project_rq2_resnet8_results.md). Board: ZCU104.
- **FINN (Leg B): DONE — bitfiles built (FINN analytical estimates; artifacts on the WSL build host).** Accuracy **86.68%** (W4A4). Three FIT, timing-MET operating points: baseline **~1,017 fps** @100 MHz (LUT 25,760 / 11%, DSP 74 / 4%, BRAM ~13%); matched-throughput **3,052 fps** @100 MHz (LUT 30,824 / 13.4%, DSP 111 / 6.4%); MAXFOLD ceiling **~36,169 fps** @333 MHz (WNS +0.047; LUT 63,739 / 28%, DSP 569 / 33%) (source: memory/project_rq2_resnet8_results.md, 2026-06-16; on-host /root/rq2_training/finn_resnet8/). FINN fps = analytical (clk/max_cycles), not M3. Board: ZCU104.
- **hls4ml (Leg C): DONE characterization — does NOT fit (csynth only, no route).** Accuracy **89.11%** (QKeras-8bit QAT, epoch 488) — highest of three (source: rq2_resnet8/hls4ml_final/convert_resnet8_final.py:6). Final fitted-attempt csynth over budget: **BRAM ~194% (1,216 BRAM18)**, LUT 200,938 (87%), FF 100,239, DSP 488 (28%), latency 175,714 cyc (source: memory/project_rq2_resnet8_results.md, 2026-06-16); an earlier in-repo config was heavier (BRAM 234% / LUT 339,196 / 147% / 90,146 cyc — rq2_resnet8/hls4ml/CSYNTH_SUMMARY.txt:80-97). No route legalizes the io_stream skip-FIFOs (cosim FIFO-opt hung ~9.5 h, killed). Board: ZCU104.

### RQ3 — verification / debugging methodology
**DONE (root-caused + fixed, byte-exact):**
- Engine weight-read-latency mismatch (2-cycle URAM): WLAT=1 byte-exact, WLAT=2 → 15058/50176 wrong; fixed (source: memory/project_e2e_value_verification.md:22-32). Board: Alveo U250.
- 22-of-48 ReLU activation-rescale bug, all 22 → 0 mismatch (source: memory/project_relu_rescale_bug.md:12,20,28).
- Stale engine bias map (~0.43×), rebuilt via build_bias_memory_map.py (source: memory/project_phase2_e2e_localization.md:74-83).
- FINAL ResNet correctness root cause: stale engine scale.mem (old [21:16]/[15:0] vs FIT-FIX requant) → mismatch 2953/100352; fixed via build_scale_memory_map.py (source: memory/project_resnet_2953_stale_scalemem.md:10-16). Board: Alveo U250.
- conv_200 "94-96% wrong" = X-init simulation artifact (only Verilator --x-initial 0 is hardware-faithful) (source: memory/project_xinit_artifact_conv200.md:10-23).
- MobileNetV2 synth-OOM conclusively fixed (lane-serialize node_mean/node_linear + banked weight ROM), byte-exact (source: memory/project_mbv2_synth_oom.md). Board: Alveo U250.

**PARTIAL / PENDING:**
- spatial_run handshake asymmetry (83 nodes): cycle fix kept, root-cause claim SUPERSEDED by relu-rescale bug (source: memory/project_spatialrun_handshake_bug.md:3,10-24).
- 2026-05-27 E2E_SIM_DEBUG_HANDOFF: integrated nn2rtl_top.v had NEVER produced a full output frame at that date — SUPERSEDED by the June 2026 routed-netlist results (ResNet 5,299,588-cyc byte-exact; MBV2 8/8) (source: docs/E2E_SIM_DEBUG_HANDOFF.md). Recorded as the documented mid-debug state. Board: Alveo U250.

**Pending verification items:**
- ResNet-50 FINAL-netlist routed Fmax (key known-unknown): only _final_c14 completed and FAILED organically; weight_bus/SLR1 epicenter is the fix target (source: docs/NETWORKS_DATA_ANATOMY.md:388-398).
- Config-B routing tension — BRAM binding (98.81%) while URAM cannot be bitstream-init non-zero on U250 (forces weights into BRAM); ResNet route is CLB-slice / congestion bound, not BRAM-density-bound (source: docs/NETWORKS_DATA_ANATOMY.md:58-69; memory/project_resnet_route_logic_bound.md:12-16,32).
- The c14 congestion failure (22,199 then 24,567 overlaps on retry) is unresolved; a relaxed (14-16 ns) flow needs re-place, not just re-route, after fixing the clock-flag bug (source: docs/NETWORKS_DATA_ANATOMY.md:39-49).
- RQ2 ResNet-8 14,774-cyc is from a 4-day-old memory diary; confirm against a fresh run_resnet8_top_value.ts (source: memory/project_rq2_resnet8_results.md; result.json).

**Likely to change before ~June 21 deadline:**
- A ResNet-50 FINAL-netlist routed Fmax (a third synth branch targeting the weight_bus/SLR1 epicenter, e.g. conv_288→engine −285 BRAM, would unblock the c14 route).
- [RESOLVED 2026-06-17] nn2rtl ResNet-8 citable clock = routed postroute **143.04 MHz @ 7 ns** (WNS +0.009 MET), 14,774 cyc → ~9,682 fps; supersedes the earlier 105.49 MHz (10 ns) and 122.26 MHz synth-stage readings.
- FINN/hls4ml ResNet-8 numbers may be re-measured (FINN bitfile rpts live on the remote build host).

---

## 6. Reproducibility

Repo: https://github.com/bote05/RTL_LLM_CLAUDE.git (TScIT 2026, deadline ~June 21, 2026). The autonomous pipeline (orchestrator sdk/orchestrate.ts + Cartographer/Foundry/Surgeon + Failure-Classifier/Retrospector + deterministic Assayer) and the generated RTL are open-sourceable; every optimization ships an anchor-asserted idempotent applier with timestamped backups and a byte-exact cmp reproduction proof; generators are deterministic and byte-reproducible (E2E cycle/FIFO numbers are valid from a zero-input run because the dataflow is statically scheduled — no control path branches on data value). Gated / not bitstream-portable: **Vivado** (serialize all synth — one MBV2 depthwise OOC ballooned to ~75 GB; RAM kill 90-95%; never run until proven bit-exact + accurate + fit-confirmed; one synth-only checkpoint + many resume-routes); **weights** (never passed to LLMs — Cartographer writes .hex to disk, RTL loads via $readmemh; repacked KPAR8 banks _kp8.mem are gitignored and regenerated on promotion); **ImageNet** (the accuracy-measurement calibration differs from the deployment weights — default generate_golden does NOT reproduce deployed weights); **URAM bitstream init** (UltraRAM cannot be non-zero-initialized on U250 / Vivado 2025.2, UG573; all weight ROMs in BRAM/LUT, all URAM runtime-written). Note: nn2rtl_top.v is generated then PATCHED IN PLACE by the integration agent — do not blindly regenerate (destroys patches, deadlocks e2e). Full regen-step + data-contract detail: archive §2.10/§6 and feedback_regen_must_rebuild_engine_maps.md.

---

## 7. Citations

Note: verified canonical entries supplied here; the author's full proposal .bib (which already contains FINN, hls4ml, He/ResNet and the LLM-for-RTL set) was not on disk — paste it in and merge.

**Verified canonical references (supply verbatim; arXiv ids checked):**
- K. He, X. Zhang, S. Ren, J. Sun. "Deep Residual Learning for Image Recognition." CVPR 2016. arXiv:1512.03385.  [ResNet]
- M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, L.-C. Chen. "MobileNetV2: Inverted Residuals and Linear Bottlenecks." CVPR 2018. arXiv:1801.04381.  [MobileNetV2 — was missing]
- E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." ICLR 2023. arXiv:2210.17323.  [GPTQ — was missing]
- C. Banbury et al. "MLPerf Tiny Benchmark." NeurIPS 2021 Datasets & Benchmarks. arXiv:2106.07597.  [MLPerf-Tiny — was missing]
- Y. Umuroglu et al. "FINN: A Framework for Fast, Scalable Binarized Neural Network Inference." FPGA 2017. arXiv:1612.07119.  [FINN]
- M. Blott et al. "FINN-R: An End-to-End Deep-Learning Framework for Fast Exploration of Quantized Neural Networks." ACM TRETS 2018. arXiv:1809.04570.  [FINN-R]
- J. Duarte et al. "Fast inference of deep neural networks in FPGAs for particle physics" (hls4ml). JINST 13 P07027, 2018. arXiv:1804.06913.  [hls4ml]

**RQ2 prior-art (from repo, confirm exact citation):**
- Minnella et al. arXiv:2309.15631 — custom HLS ResNet-8 (W8A8, 88.7%, 12,971 FPS Ultra96-V2 / 30,153 FPS KV260) (source: memory/project_resnet8_3way_plan.md L13).
- Tailor et al. arXiv:2301.07247 — full-skip ResNet-8 in hls4ml (U200, ap_fixed<16,6> RF=72, 87.4%, 304,697 cyc @100 MHz, 158K LUT) (source: memory/project_resnet8_3way_plan.md L13).
- Reference benchmark arXiv:2206.11791 (hls4ml-vs-FINN same-board, Pynq-Z2: IC 27.3 ms FINN vs 1.5 ms hls4ml; MLPerf-Tiny reference) (source: memory/project_resnet8_3way_plan.md L13,L17; project_hls4ml_finn_comparison.md L26).
- fpgaConvNet MLPerf-Tiny v1.1 closed (Venieris & Bouganis) — ran the exact reference ResNet-8 (INT8: ZC706 6,790 inf/s, ZedBoard 2,431, Zybo 318, 86.0% top-1) (source: memory/project_resnet8_3way_plan.md L13).

**FINN external-literature (U250 comparison, from repo):** FINN ResNet-50 W1A2 = 67.27% (binary) / 69.85% (ternary), 2703 FPS (paper Table II), 195 MHz, 1027 kLUT / 3870 BRAM18 / 1611 DSP; primary source arXiv:2011.07317 Table II (source: docs/agent_tasks/THESIS_FINN_HLS4ML_COMPARISON.md:11-14,28-44). No official ImageNet-scale hls4ml ResNet-50/MobileNetV2 numbers exist (itself a citable finding).

**LLM-for-RTL:** README cites "Tomlinson et al. (2024)" only; the proposal's full LLM-for-RTL set is NOT on disk — **[author: paste remaining LLM-for-RTL citations from the proposal .bib]**.

**Tier-A comparison (Artix-7, prior milestone):** nn2rtl vs hls4ml vs FINN on 8 ResNet-50 stem+layer1 layers: nn2rtl 3,992 LUT vs hls4ml 32,174 vs FINN 49,079 (~8×/12× fewer); Fmax 339 / 71 / 109 MHz; fps 9.50 / 1,407 / 183. hls4ml 3×3 convs UNSYNTHESIZABLE (QONNX scalar-Quant hangs the Vitis HLS clang frontend) (source: docs/THESIS_SOURCE_MAP.md:54,442,613; memory/project_tier_a_complete.md). Board: Artix-7.

---

## Open questions for the author

a. Paste the proposal .bib + the remaining LLM-for-RTL citations (only "Tomlinson et al. (2024)" is on disk).
b. Retrieve the FINN/hls4ml ResNet-8 on-host rpts (currently memory-only; FINN bitfile rpts live on the remote build host).
c. Decide the ResNet-50 final-netlist framing — report previous-netlist 83.33 MHz / 14.71 fps OR attempt a third synth branch (conv_288→engine, −285 BRAM toward the weight_bus/SLR1 congestion epicenter) before the deadline.
d. Confirm the "largely autonomous" wording for system (2) (the separate Claude Opus 4.8 Code-agent instance that built the engine, integrated the network, and ran the whole Vivado campaign under standing human direction).
