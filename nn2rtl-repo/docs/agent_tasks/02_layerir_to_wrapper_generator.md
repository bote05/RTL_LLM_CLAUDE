---
task_id: 02
title: LayerIR → top-level wrapper generator
type: TypeScript tooling
status: review
depends_on: []
unblocks: [Phase 2 integration]
---

# Task 02 — LayerIR → top-level wrapper generator

## Goal

Write a TypeScript script that reads ResNet-50's `output/layer_ir.json` and emits the top-level Verilog wrapper that wires the per-layer modules together into a single dataflow design.

The wrapper has three jobs:
1. Instantiate each spatial module in the LayerIR graph order.
2. Wire each module's `data_out` to the next module's `data_in` via a small streaming FIFO.
3. Carve out a "hole" for the shared engine: the heavy modules in the LayerIR are not instantiated; instead, their input/output BRAM banks are exposed as ports that the engine block (instantiated separately) will drive.

This is pure deterministic tooling. No LLM involved. The output is one Verilog file, deterministically generated, that compiles cleanly under `iverilog`.

## Deliverable

A new script at `scripts/build_top_wrapper.ts` plus the generated artefact.

### Script behaviour

- Reads `output/layer_ir.json`.
- Classifies each layer as either *spatial* (instantiated directly in the wrapper) or *engine-dispatched* (placeholder — the engine block runs it). The classification is driven by a parameter file (see "Heavy module list" below) so retargeting to a different network only requires changing that list.
- Emits one top-level module `module nn2rtl_top (...)` containing:
  - `clk`, `rst_n` ports
  - AXI4-Stream input (network image input)
  - AXI4-Stream output (network logits output)
  - AXI4-Lite control slave
  - URAM weight memory instantiation (sized from `weight_memory_map.json` produced by task 01)
  - Each spatial module instantiated with `wire` connections to neighbouring modules' ports
  - One instantiation of `shared_engine` (the engine block from task 00) wired to the activation BRAM banks
  - Skip-FIFO instantiations for the 16 residual adds (sized via task 04's output)

### CLI

```
npx tsx scripts/build_top_wrapper.ts \
    --network=resnet-50 \
    [--layer-ir=output/layer_ir.json] \
    [--engine-modules=docs/agent_tasks/06_phase1_compression_candidates_HEAVY.txt] \
    [--fifo-sizes=output/wrapper/skip_fifo_sizes.json] \
    [--weight-map=output/weights/weight_memory_map.json] \
    [--out=output/rtl/nn2rtl_top.v]
```

## Context (read this before starting)

- Deployment plan §6.6 describes what the wrapper does at the architectural level.
- The wrapper is "mechanically generated from LayerIR" (plan §6.6, last paragraph). This is the tool that does the mechanical generation.
- Each per-layer module has a canonical port set (from the protected patterns): `clk`, `rst_n`, `valid_in`, `ready_in`, `data_in[N:0]`, `valid_out`, `data_out[M:0]`. Bus widths come from each layer's `input_width_bits` and `output_width_bits` in LayerIR.
- The residual-add layers (`node_add_*` modules) have *two* input streams (main path + skip path), each with its own `valid/ready/data` triplet. The wrapper must wire both.
- Task 01 produces `weight_memory_map.json` which the wrapper reads to size the URAM region.
- Task 04 produces `skip_fifo_sizes.json` which the wrapper reads to size each residual FIFO.
- Task 00 produces the engine skeleton; the wrapper instantiates one copy.

## Heavy module list

The wrapper needs an explicit list of which modules are dispatched through the engine vs which are spatial. This list is produced by task 06 (Phase 1 compression candidates).

**This task is Wave 1 dispatchable, so it runs before task 06 lands.** Use the fallback list below in Wave 1. After task 06 produces the real list (post Phase 0 U250 baseline), the orchestrator **re-runs this task** with the real list. The script is deterministic — re-running on the same inputs produces byte-identical output, so the second run is cheap and the wrapper just gets refreshed with the actual heavy modules.

The agent's first-pass wrapper using the fallback is **not wasted work** — it lets task 13 (integration) start preparing the assembly process. The wrapper is regenerated when the real list arrives; downstream tasks (scheduler, etc.) consume the regenerated file.

```
node_conv_284
node_conv_286
node_conv_290
node_conv_292
node_conv_296
node_conv_298
node_conv_282
node_conv_288
node_conv_294
node_conv_220
```

(Final list comes from task 06's analysis of the U250 post-Phase-0 baseline.)

## How to verify

1. Run the script against the current LayerIR. Confirm it produces `output/rtl/nn2rtl_top.v`.
2. `iverilog -t null output/rtl/nn2rtl_top.v output/rtl/*.v` should compile cleanly (engine sub-blocks may be empty stubs at this point — that is OK, iverilog should at least parse the wrapper).
3. The generated wrapper must contain exactly 119 minus heavy_count instantiations of spatial modules.
4. Every spatial module's `data_out` must be wired to exactly one downstream consumer (no orphan outputs, no double-driven inputs).
5. The residual adds must have *both* their input streams wired (one from the main path, one from a skip FIFO).
6. Determinism: running the script twice on the same inputs produces byte-identical output.

## Out of scope

- Do NOT implement the engine. Task 00 owns the skeleton, tasks 07-11 own the sub-blocks. The wrapper just instantiates `shared_engine` as a black box and wires its ports.
- Do NOT generate the scheduler. That is task 03.
- Do NOT modify LayerIR, weights, or per-layer RTL.
- Do NOT generate the AXI4-Lite control register block. That is part of the engine (task 10).
- Do NOT call any LLM agents.

## Success criteria

- Script runs in under 10 seconds.
- Output file is exactly one Verilog file, deterministically generated.
- The wrapper compiles under iverilog parse stage even if engine sub-blocks are still empty (task 00 must have committed at least the empty skeleton).
- The generated wrapper passes a simple visual sanity check: 16 skip FIFOs (one per residual add), one engine instantiation, one URAM instantiation, ~95 spatial-module instantiations.
