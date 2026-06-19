---
task_id: 05
title: On-chip-weights contract
type: New contract artefact + pattern doc
status: review
depends_on: []
unblocks: [Phase 2 integration, future MobileNetV2 retarget]
---

# Task 05 — On-chip-weights contract

## Goal

Create a new nn2rtl contract called `on-chip-weights` that captures the architectural pattern used by the U250 deployment: weights live in a pre-loaded UltraRAM region, addressed by a base offset per dispatched layer, with no external memory interface. This contract is the on-chip analogue of the existing `dram-backed-weights` contract.

The deployment plan §16.5 calls this out explicitly: "A separate `on-chip-weights` contract variant may be promoted to a first-class artefact if it stabilises during Phase 2." This task does that promotion proactively.

## Deliverable

Five files, mirroring the structure of the existing contracts:

1. `contracts/on-chip-weights/metadata.json` — contract definition (interface signals, supported ops, fit constraints, protocol rules).
2. `contracts/on-chip-weights/testbench.cpp` — Verilator testbench wrapper specific to this contract.
3. `contracts/on-chip-weights/latency.ts` — TypeScript pipeline-latency calculator for layers under this contract.
4. `contracts/on-chip-weights/golden.py` — Python golden-vector generator for this contract.
5. `knowledge/patterns/protected/13_on_chip_weights.md` — pattern doc explaining the contract's discipline, common bugs, and the canonical structure of a layer body under this contract.

## Context (read this before starting)

- Read `contracts/dram-backed-weights/metadata.json` and `knowledge/patterns/protected/09_dram_backed_weights.md` first — those are the closest existing artefacts. The new on-chip-weights contract is structurally similar except the weight port is an on-chip URAM read interface instead of an AXI4-MM master.
- Read deployment plan §3, §6.1, §6.7 for the architectural rationale.
- The seed reference for this contract is the engine itself (task 00's skeleton, when implemented). For now, the contract metadata can list `node_conv_288` as the closest seed (knowing it currently lives under `dram-backed-weights`).
- The contract supports `conv2d` only (the engine runs heavy convs; ReLU / Add / MaxPool stay on flat-bus).

## `metadata.json` shape (mirror existing contracts)

```json
{
  "name": "on-chip-weights",
  "display_name": "On-Chip Weights (UltraRAM)",
  "complexity_rank": 1,
  "interface_signals": [
    { "name": "clk", ... },
    { "name": "rst_n", ... },
    { "name": "valid_in", "direction": "input", ... },
    { "name": "ready_in", "direction": "output", ... },
    { "name": "data_in", "direction": "input", "width_expr": "input_width_bits", ... },
    { "name": "valid_out", ... },
    { "name": "data_out", ... },
    {
      "name": "weight_rd_addr",
      "direction": "output",
      "width_expr": "weight_addr_bits",
      "role": "URAM read address; layer's base address is a parameter"
    },
    {
      "name": "weight_rd_data",
      "direction": "input",
      "width_expr": "uram_word_bits",
      "role": "URAM-word-wide read data (288 bits on UltraScale+)"
    },
    {
      "name": "weight_rd_en",
      "direction": "output",
      "width_bits": 1,
      "role": "URAM read enable"
    },
    {
      "name": "weight_base_word",
      "direction": "input",
      "width_expr": "weight_addr_bits",
      "role": "Layer's URAM base address (in 288-bit URAM words), loaded by scheduler"
    }
  ],
  "fit_constraints": {
    "max_bus_width_bits": 8192,
    "weight_memory": "on-chip UltraRAM (URAM) only; no external memory",
    "buffer_sizing_rules": [
      "Public streaming interface is identical to flat-bus.",
      "Weight reads come from a URAM region whose base address is parameter-loaded per layer.",
      "Activations are still streamed; the activation BRAM ping-pong is managed by the scheduler outside this module."
    ]
  },
  "supported_ops": ["conv2d"],
  "dependencies": [],
  "docs": [
    "knowledge/patterns/01_context.md",
    "knowledge/patterns/13_on_chip_weights.md",
    "knowledge/patterns/08_common_bugs.md"
  ],
  "protocol_rules": [
    "All weight bytes for this layer must be addressable as a contiguous URAM region starting at weight_base_word.",
    "The module must not have any external memory interface (no AXI4-MM, no DDR).",
    "Weight reads must use weight_rd_en + weight_rd_addr + weight_rd_data; no $readmemh inside the module (the URAM is initialised at bitfile load via the top-level wrapper)."
  ]
}
```

## Pattern doc structure (mirror `09_dram_backed_weights.md`)

The pattern doc must cover:
- When to choose this contract over flat-bus or dram-backed-weights.
- Canonical layer body structure (weight read state machine, MAC + accumulator + bias + scale + saturate pipeline).
- URAM read latency (typically 2 cycles for UltraRAM on UltraScale+) and how to mask it with the MAC pipeline.
- Common bugs:
  - Reading past `weight_base_word + WSIZE_words` — fence the address generator.
  - Forgetting to load `weight_base_word` before deasserting `engine_start` — the scheduler enforces this externally but the module should not crash if it sees an early `valid_in`.
  - Mixing up URAM-word width (288 bits) with byte-aligned weight access (1 byte per INT8 weight = 36 bytes per URAM word = 36 weights per word).
- An empty "Successful references" section that gets populated by the auto-promote machinery as Phase 2's sub-blocks land.

## `latency.ts` content

Mirror the existing `contracts/dram-backed-weights/latency.ts`. The on-chip variant's latency formula is essentially the same as flat-bus + a URAM read pipeline stage (2 cycles). Write the formula as:

```
pipeline_latency = flat_bus_conv_latency + URAM_READ_LATENCY_CYCLES
```

where `URAM_READ_LATENCY_CYCLES = 2`. The full conv latency calculator can reuse the existing `compute_conv2d_latency_cycles` helper.

## `golden.py` and `testbench.cpp`

These mirror the existing dram-backed-weights variants. The key difference: instead of an AXI4-MM weight slave, the testbench provides a behavioural URAM model that responds to `weight_rd_addr / weight_rd_en` with `weight_rd_data` from a pre-loaded `.mem` file.

## How to verify

1. The new `contracts/on-chip-weights/` directory has all four files.
2. `metadata.json` validates against the same Zod schema as the other contracts (look at `sdk/contracts.ts` for the schema). The orchestrator should be able to load it without modification.
3. `latency.ts` exports a `computeLatency(layer)` function with the same signature as the existing contracts'.
4. The pattern doc `13_on_chip_weights.md` follows the structure of `09_dram_backed_weights.md` (when-to-use, semantics, architecture hints, required interface, common bugs, anticipated failure modes).
5. `sdk/contracts.ts` (or wherever contracts are registered) lists the new contract.

## Out of scope

- Do NOT yet implement a passing reference layer body under this contract. That comes from Phase 2's engine sub-blocks; the contract artefact is the *spec*, not the implementation.
- Do NOT modify other contracts.
- Do NOT touch the pipeline state or layer IR.
- Do NOT call any LLM agents.

## Success criteria

- All five files exist.
- The new contract is registered (the orchestrator can list it).
- The pattern doc compiles into the existing `knowledge/patterns/protected/` numbering (next after `12_depthwise_conv.md`).
- A trial run of `npm run typecheck` in both `sdk/` and `mcp/` passes after the new contract is added.
