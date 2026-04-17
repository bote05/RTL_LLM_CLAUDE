# nn2rtl

Neural Network to RTL compiler

Last updated: April 14, 2026  
Author: Daniel — University of Twente, Bachelor Thesis

## Overview

`nn2rtl` is an autonomous multi-agent AI system that takes a trained PyTorch neural network and produces synthesizable Verilog RTL suitable for FPGA implementation. Instead of a human hardware engineer manually writing tens of thousands of lines of RTL, the system coordinates four LLM agents (Cartographer, Foundry, Assayer, Surgeon) around a deterministic TypeScript orchestrator that:

- extract the network structure from a quantized PyTorch checkpoint,
- generate synthesizable Verilog modules,
- verify those modules against golden numerical outputs,
- synthesize passing modules for PPA proxy metrics,
- and repair failures automatically through a structured retry loop.

The central research claim is intentionally strong: this system is designed to demonstrate that LLMs can automate the NN-to-RTL workflow end to end, not merely assist a human designer with isolated code snippets.

Compared with prior NN-to-hardware LLM work such as Tomlinson et al. (2024), which used manual copy-paste prompting to generate a small spiking network with no integrated toolchain verification, `nn2rtl` targets production-relevant scale: 50,000+ lines of RTL, real simulator and synthesis integration, and a closed-loop autonomous repair mechanism.

## Reference Documentation

All implementation decisions in this project are grounded in the following official documentation. When in doubt, these sources are authoritative.

- Plugin structure: <https://code.claude.com/docs/en/plugins>
- Plugin manifest reference: <https://code.claude.com/docs/en/plugins-reference>
- Subagent frontmatter fields: <https://code.claude.com/docs/en/sub-agents>
- Agent SDK TypeScript reference: <https://platform.claude.com/docs/en/agent-sdk/typescript>
- Agent SDK subagents: <https://platform.claude.com/docs/en/agent-sdk/subagents>
- Agent SDK custom tools: <https://platform.claude.com/docs/en/agent-sdk/custom-tools>
- Agent SDK structured outputs: <https://platform.claude.com/docs/en/agent-sdk/structured-outputs>
- Agent SDK plugins in SDK: <https://code.claude.com/docs/en/agent-sdk/plugins>
- MCP in Agent SDK: <https://platform.claude.com/docs/en/agent-sdk/mcp>
- Tools reference: <https://code.claude.com/docs/en/tools-reference>
- Skills: <https://code.claude.com/docs/en/skills>
- Hooks reference: <https://code.claude.com/docs/en/hooks>

## What This Project Is

This repository implements a three-layer system:

1. A Claude Code plugin that defines agent roles and reusable skills.
2. A TypeScript orchestration layer built on the Claude Agent SDK.
3. A local MCP server that exposes synthesis and verification tools to the agents.

The system consumes a quantized PyTorch ResNet-50 residual block stack and is designed to emit synthesizable, verifiable Verilog RTL for those blocks without human intervention during a pipeline run.

## Scope Decisions

### Target Network

The target is the ResNet-50 stem plus the residual block stack (the 16 residual blocks that dominate compute). The current runtime source of truth is the generated `LayerIR` plus golden vectors, not stale prose in this README. On the default legacy `.pth` path, `layer0_0_conv1` is not a fused MaxPool stage; the ONNX frontend carries explicit conv / maxpool geometry instead of asking Foundry to infer it. Still out of scope:

- global average pooling,
- the fully connected classifier layer,
- full-chip SoC integration,
- ASIC backend and tapeout flows.

### Why Residual Blocks

The residual blocks have a repeating and well-understood structure:

- `1x1` convolution,
- `3x3` convolution,
- `1x1` convolution,
- skip connection addition,
- batch normalization.

This makes them well suited for automated RTL generation at scale, while still being large enough to represent production-relevant hardware design complexity.

### Target Hardware

The validation target is FPGA, not ASIC. A successful synthesis and place-and-route flow for FPGA-class RTL is sufficient proof that the output is real hardware, not merely simulation-only code.

### Number Format

The entire flow is built around INT8 symmetric per-tensor post-training quantization. PyTorch float32 weights are quantized before the pipeline starts. This keeps multipliers manageable, aligns with practical accelerator design, and allows direct verification against quantized golden vectors.

### Batch Normalization

Batch normalization is folded into the preceding convolution during extraction. Generated hardware never implements batch normalization as a standalone runtime operation.

## Core Technical Problem

PyTorch operates in floating point. Hardware operates in fixed-point integer arithmetic. Quantization is the bridge between these worlds, and it affects every operator in the generated RTL.

Before the pipeline runs, preprocessing scripts:

- quantize the model into INT8,
- compute layer-specific scale factors,
- run a fixed test image through the quantized model,
- capture golden activation tensors at layer boundaries,
- and write those golden vectors to disk.

These golden vectors are the ground truth. The generated Verilog must reproduce them within an empirically calibrated fixed-point tolerance.

The hardest part of the system is not writing Verilog. It is proving that the generated Verilog numerically matches the quantized PyTorch model under a real simulation toolchain.

## Critical Architectural Decisions

### Weights Are Never Passed to LLMs

ResNet-50 contains millions of parameters. Passing raw weight tensors through agent prompts would exhaust context windows, inflate cost, and introduce truncation risk.

Instead:

- Cartographer writes weights to binary-friendly `.hex` files on disk,
- the Layer IR contains only weight metadata and file paths,
- and Foundry generates RTL that loads weights using `$readmemh`.

Files are stored in `output/weights/` using names such as:

- `<module_id>_weights.hex`
- `<module_id>_bias.hex`

The intended Verilog pattern is:

```verilog
reg signed [7:0] weights [0:NUM_WEIGHTS-1];
initial $readmemh("/absolute/path/to/output/weights/block_1_conv1_weights.hex", weights);
```

Golden activation vectors are written to per-module binary sidecar files (`.goldin` / `.goldout`) under `output/goldens/` using the `NN2V` format (16-byte header + int32 LE samples). The LayerIR carries only the paths, not the raw tensors, so even a full ResNet-50 LayerIR stays well under Node's string size limits.

### All Generated Modules Must Be Fully Pipelined

Behavioral simulation alone is not enough. A giant combinational implementation may pass numerical checks while being physically useless.

For that reason, the Layer IR includes an exact timing contract. Every module must implement the requested pipeline latency and valid/ready behavior.

Latency targets by operation type:

| Operation | Minimum latency |
| --- | --- |
| `1x1` convolution | 3 cycles |
| `3x3` convolution | 5 cycles |
| folded batchnorm stage | 2 cycles |
| `ReLU` | 1 cycle |
| residual add | 1 cycle |

Each generated module must implement:

- `valid_in`,
- `ready_in` or equivalent backpressure semantics,
- `valid_out`,
- exact cycle-accurate latency from input valid to output valid.

The baseline target is `50 MHz` on an iCE40 FPGA. A functionally correct module that fails timing is treated as a real pipeline failure.

### The C++ Testbench Is Static Infrastructure

The Verilator C++ testbench is handwritten once, committed to the repository, and never generated by an agent.

This avoids a dangerous two-bug problem:

- the Verilog under test could be wrong,
- and an agent-generated testbench could also be wrong.

Instead, Assayer only generates a JSON sidecar describing:

- module name,
- port names and widths,
- clock and reset signal names,
- pipeline latency,
- input and output golden vector file paths.

The static testbench reads that sidecar, drives the module, captures outputs, checks timing, and writes a structured results JSON file.

## Platform Decision

### Claude Code Agent SDK

The orchestrator is implemented as real TypeScript code using `@anthropic-ai/claude-agent-sdk`. The control flow is not prompt-driven. The `PipelineStateManager` is a deterministic state machine that:

- reads pipeline state from disk,
- decides what to do next,
- dispatches agents using `query()`,
- parses their results,
- updates state,
- and resumes cleanly after interruption.

This is important for a thesis because it makes the system:

- auditable,
- reproducible,
- measurable,
- and amenable to cost tracking.

### Why Not Codex

Codex-style prompt-mediated delegation was explicitly rejected for this orchestration layer because it would make the control logic itself dependent on LLM interpretation. For a reproducible research system with retries, resumption, and measurable outcomes, orchestration belongs in code.

### SDK Package

Use:

- `@anthropic-ai/claude-agent-sdk`

Do not use:

- `@anthropic-ai/claude-code`

### Version Pinning

The SDK version must be pinned before the experiment phase. The exact version used must be documented in the thesis and must not be upgraded mid-experiment.

## System Architecture

The repository is structured into three layers:

### Layer 1: Claude Code Plugin

Located in `nn2rtl-plugin/`

Contains:

- `.claude-plugin/plugin.json`
- `agents/`
- `skills/`
- `.mcp.json`

This layer defines the agent roles, system prompts, and reusable domain skills.

### Layer 2: Agent SDK Orchestrator

Located in `sdk/`

Contains the real orchestration logic:

- pipeline state machine,
- typed data contracts,
- agent dispatch loop,
- resume support,
- cost tracking,
- run logging.

### Layer 3: MCP Server

Located in `mcp/`

Provides the tool boundary between agents and the external toolchain:

- syntax checking,
- simulation,
- synthesis,
- weight extraction,
- Verilog persistence.

## The Orchestrator and Four Agents

### Orchestrator (formerly an LLM "Conductor")

The pipeline-coordinator role lives in the deterministic TypeScript orchestrator
in `sdk/orchestrate.ts`, not in an LLM agent. It is the only component that
sees the whole pipeline simultaneously, and its job is to:

- maintain `output/pipeline_state.json`,
- select the next action,
- dispatch the four LLM agents below,
- enforce retry limits,
- run Yosys directly on every Assayer-passed module, and
- write final summary reports.

Keeping the coordinator deterministic makes the run reproducible and avoids
burning model calls on state-machine bookkeeping.

### Cartographer

- Model: Sonnet
- Effort: Low

Cartographer runs once at startup. It:

- loads the quantized ResNet-50 checkpoint,
- traces the model using `torch.fx`,
- folds batchnorm into convolution parameters,
- writes weight and bias `.hex` files,
- and emits the Layer IR consumed by the rest of the system.

Cartographer knows PyTorch, not Verilog.

### Foundry

- Model: Sonnet
- Effort: Medium

Foundry is the main RTL generator. It receives exactly one module specification and produces one synthesizable Verilog module.

Hard constraints include:

- INT8 fixed-point arithmetic,
- 32-bit widened accumulators where needed,
- weights loaded through `$readmemh`,
- valid/ready handshake,
- exact pipeline latency,
- saturating arithmetic for residual add,
- signed declarations for weights and activations,
- no simulation-only constructs in synthesizable modules.

### Assayer

- Model: Haiku
- Effort: Minimal

Assayer is a tool execution agent. It does not reason deeply or edit files. It:

- generates the JSON sidecar for the static testbench,
- runs `iverilog`,
- runs `verilator`,
- parses structured simulator output,
- and returns a `VerifResult`.

It is the most frequently invoked agent in the system.

### Surgeon

- Model: Opus
- Effort: Maximum

Surgeon activates only on failures. It receives:

- the broken Verilog module,
- the verification result,
- and the original Layer IR.

It must:

1. classify the failure,
2. identify the exact faulty section,
3. rewrite only that section,
4. preserve the interface,
5. and return a minimally repaired module.

Surgeon is the most expensive per-call component and is capped at three retries per module.

## Pipeline Flow

### Pre-pipeline

Human-run one-time steps before the autonomous loop:

1. Run `quantize_model.py`
2. Run `generate_golden.py`
3. Validate the static testbench with a handwritten reference module

### Autonomous Pipeline Run

1. The orchestrator starts.
2. If `output/layer_ir.json` does not exist, it invokes Cartographer.
3. Cartographer emits Layer IR and writes weight files.
4. The orchestrator initializes pipeline state with all modules set to `pending`.
5. The orchestrator selects the first pending module and invokes Foundry.
6. Foundry generates Verilog and persists it via `write_verilog`.
7. The orchestrator marks the module `verifying` and invokes Assayer.
8. Assayer runs syntax and simulation checks and returns a `VerifResult`.
9. If the module passes, the orchestrator records success and runs Yosys for synthesis metrics.
10. If the module fails and retries remain, the orchestrator invokes Surgeon.
11. Surgeon repairs the module and returns a replacement `VerilogModule`.
12. Assayer verifies the repaired module.
13. If retries are exhausted, the module becomes `fail_abort` and the pipeline continues.
14. When all modules are terminal, the orchestrator writes `output/reports/pipeline_summary.json`.

### Resume Behavior

Pipeline state is written after every transition. A crashed or interrupted run can be resumed using `--resume`.

## Module State Machine

Each module moves through the following states:

```text
pending
  -> generating
    -> verifying
      -> pass
      -> fail_retry
        -> generating
      -> fail_abort
```

The authoritative record is `output/pipeline_state.json`.

## Data Contracts Between Agents

Agents communicate only via JSON strings embedded in prompts and returned as final outputs.

### LayerIR

The master per-module specification. It is written by Cartographer and consumed by Foundry, Assayer, and Surgeon.

Required semantics:

- `module_id`
- `op_type`
- `input_shape`
- `output_shape`
- `weights_path`
- `bias_path`
- `weight_shape`
- `num_weights`
- `scale_factor`
- `zero_point`
- `pipeline_latency_cycles`
- `clock_period_ns`
- `input_width_bits`
- `output_width_bits`
- `valid_in_signal`
- `valid_out_signal`
- `clock_signal`
- `reset_signal`
- `golden_inputs`
- `golden_outputs`

### VerilogModule

Produced by Foundry and Surgeon:

- `module_id`
- `spec_hash`
- `verilog_source`
- `generated_by`
- `attempt`

### VerifResult

Produced by Assayer:

- `module_id`
- `status`
- `timing_pass`
- `timing_actual_cycles`
- `timing_expected_cycles`
- `mismatch_layer`
- `expected`
- `got`
- `max_error`
- `mean_error`
- `failure_class`
- `fix_hint`
- `iverilog_stderr`
- `verilator_stderr`

### PipelineState

Maintained by the orchestrator:

- `run_id`
- `started_at`
- `modules`
- `attempts`
- `results`
- `max_retries`
- `total_cost_usd`
- `model_usage`

## Tool Infrastructure

The MCP server exposes exactly five tools.

### `run_iverilog`

- Inputs: Verilog source, module name
- Action: writes temporary source and runs `iverilog -o /dev/null -g2012`
- Output: `{ success, stderr }`

### `run_verilator`

- Inputs: Verilog source, module name, sidecar path
- Action: compiles with Verilator, runs the static C++ testbench, parses results JSON
- Output: full `VerifResult`

### `run_yosys`

- Inputs: Verilog source, module name
- Action: runs synthesis and extracts LUT and Fmax proxy metrics
- Output: `{ success, lut_count, fmax_mhz, report }`

### `read_weights`

- Inputs: checkpoint path, quantization config
- Action: spawns the Python golden-vector generation flow and returns parsed IR
- Output: `PipelineIR`

### `write_verilog`

- Inputs: `VerilogModule`, output directory
- Action: writes `.v` and metadata JSON
- Output: absolute file path

This is the only permitted way to persist generated Verilog.

## Verification Strategy

### Phase 1: Syntax

`iverilog` performs a fast syntax pass. If this fails, the module does not proceed further.

### Phase 2: Functional and Timing Verification

`verilator` plus the static testbench verifies:

- output values,
- exact pipeline latency,
- correct valid/ready timing behavior.

The acceptance threshold is based on fixed-point tolerance calibrated empirically on a handwritten reference module. The working expectation is:

- well-implemented modules: `max_error <= 1`
- pass threshold: `max_error <= 3`

### Phase 3: Synthesis

`yosys` runs only after functional and timing success. This phase confirms that the RTL is actually synthesizable and provides proxy metrics for:

- area via LUT count,
- performance via Fmax estimate.

## Failure Mode Taxonomy

Surgeon must classify every repair into exactly one of the following categories.

### Arithmetic Failures

1. Integer overflow
2. Sign extension error
3. Bit shift wrong
4. Rounding mode wrong
5. Saturation missing

### Structural Failures

6. Loop bounds incorrect
7. Array indexing error
8. Port width mismatch
9. Residual addition overflow

### Control Failures

10. Missing pipeline register
11. Pipeline latency wrong
12. Reset logic broken
13. Enable signal ignored

### Numerical Precision Failures

14. Scale factor misapplied
15. Bias term missing
16. Batch norm not folded

The distribution of these failure classes across the full experiment is itself a thesis result.

## Simulation and Synthesis Stack

### Icarus Verilog

Used for:

- syntax checking

### Verilator

Used for:

- functional verification
- timing verification
- high-performance simulation via the static C++ testbench

### Yosys

Used for:

- synthesis validation
- LUT count extraction
- Fmax proxy measurement for iCE40 FPGA targets

The agent loop is deliberately built on open-source tooling only so that the experiment remains reproducible and cluster-friendly.

## Research Questions

Primary question:

- Can LLMs fully automate the NN-to-RTL workflow for a production-scale neural network without human intervention during a run?

Secondary questions:

- Which stages benefit most from LLM assistance?
- What is the per-module failure rate?
- What are the dominant failure classes?
- What PPA is achieved relative to published baselines?
- What is the token cost per module and per full run?
- How many Surgeon interventions are required on average?

## What Makes This Novel

### Scale

This system targets 50,000+ lines of autonomously generated RTL, far beyond prior small-scale prompt-only studies.

### Infrastructure

It combines:

- multi-agent orchestration,
- real MCP tool integration,
- simulation and synthesis feedback,
- autonomous repair,
- and structured state logging.

### Evaluation Methodology

This project asks not whether an LLM can write some Verilog, but whether a complete autonomous NN-to-RTL pipeline can be measured in terms of:

- pass rate,
- failure distribution,
- synthesis quality,
- and economic cost.

### Autonomous Repair Loop

The structured Surgeon repair loop and failure taxonomy are contributions in their own right.

## Known Risks and Mitigations

### Numerical Precision Failures

Risk: fixed-point errors dominate and block progress.  
Mitigation: use the failure taxonomy, calibrate tolerance empirically, and start with simpler reference modules.

### Context Window Overload

Risk: model inputs become too large.  
Mitigation: never pass weight tensors through prompts; only pass per-module IR slices.

### Static Testbench Portability

Risk: the testbench behaves differently on the target cluster.  
Mitigation: validate it on the deployment environment before the experiment phase.

### SDK Instability

Risk: Agent SDK changes break the system mid-project.  
Mitigation: pin the SDK version and do not upgrade during experiments.

### API Cost

Risk: total model cost becomes too high.  
Mitigation: use Haiku for Assayer, monitor `total_cost_usd`, secure research credits, and budget for multiple full runs.

### Yosys Compatibility

Risk: generated RTL passes simulation but uses synthesis-hostile constructs.  
Mitigation: validate a reference module through the full synthesis flow before building the pipeline.

### `$readmemh` Path Breakage

Risk: relative paths fail on cluster or under different working directories.  
Mitigation: derive absolute paths from runtime environment and never hardcode fragile relative paths in generated RTL.

## Open Questions

The following must be resolved before the full experiment phase:

- Empirical tolerance threshold for INT8 verification
- Exact 16 residual-block module IDs from the actual `torch.fx` trace
- Confirmation that SURF Snellius access is available for Verilator workloads
- Anthropic research API credit approval
- Justification for the `50 MHz` baseline in the thesis
- Validation that folded batch normalization is numerically equivalent to the original quantized reference

## Repository Intent

This repository is not just a software project. It is an experimental platform for answering whether autonomous LLM systems can generate real hardware at meaningful scale.

The success criterion is not merely producing Verilog text. The success criterion is producing RTL that:

- compiles,
- simulates,
- matches quantized PyTorch numerically,
- synthesizes,
- and can be measured rigorously.
