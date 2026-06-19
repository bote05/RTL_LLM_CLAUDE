---
task_id: 03
title: Scheduler generator
type: Python tooling
status: review
depends_on: []
unblocks: [Phase 2 integration]
---

# Task 03 — Scheduler generator

## Goal

Write a Python script that reads ResNet-50's `output/layer_ir.json` plus the heavy-module list and emits the deterministic scheduler FSM in Verilog. The scheduler:

1. Tracks which activation BRAM bank is "current" for the dataflow chain.
2. When the dataflow stalls because a heavy layer comes up, halts the spatial chain, configures the shared engine for that layer (writes its config registers over AXI4-Lite), waits for the engine to finish, then resumes the chain.
3. Manages ping-pong of activation BRAM banks across heavy-layer dispatches (and handles residual-path activation retention — see "Activation memory ownership" in plan §6.4).

This is pure deterministic Python tooling. No LLM. The output is one Verilog module + one JSON sidecar describing the dispatch schedule for documentation / debugging.

## Deliverable

A new script at `scripts/build_scheduler.py` plus the generated artefacts.

### Script behaviour

- Reads `output/layer_ir.json` and a heavy-module list (from task 06 output).
- Walks the LayerIR graph in topological order. For each layer:
  - If spatial: no scheduler action; that layer runs continuously inside the dataflow chain.
  - If engine-dispatched: emit one dispatch entry in the scheduler FSM. The entry records: input BRAM bank ID, output BRAM bank ID, engine config (channel counts, kernel, stride, scale, zero-point, URAM weight base address from task 01).
- Emits `output/rtl/nn2rtl_scheduler.v` — a Verilog module containing the scheduler state machine. Each dispatch is a state in the FSM. The FSM walks through the dispatches in order.
- Emits `output/rtl/nn2rtl_scheduler_schedule.json` — the same schedule in machine-readable form, for debugging and for any downstream verification tooling that needs to know what the scheduler is doing.

### CLI

```
python scripts/build_scheduler.py \
    --network=resnet-50 \
    [--layer-ir=output/layer_ir.json] \
    [--engine-modules=docs/agent_tasks/06_phase1_compression_candidates_HEAVY.txt] \
    [--weight-map=output/weights/weight_memory_map.json] \
    [--out-verilog=output/rtl/nn2rtl_scheduler.v] \
    [--out-schedule=output/rtl/nn2rtl_scheduler_schedule.json]
```

## Scheduler interface (must match)

The scheduler must drive these signals (defined in task 00's skeleton spec, see also that task's PORTS doc):

- `engine_start` (output to engine) — pulsed high for one cycle when the engine should begin a dispatched layer.
- `engine_busy` (input from engine) — held high while the engine is running.
- `engine_done` (input from engine) — pulsed by the engine when its current dispatch is complete.
- `s_axil_*` (master output to engine's AXI4-Lite slave) — the scheduler is an AXI4-Lite master that writes the engine's config registers before each dispatch.
- BRAM bank selection signals for the activation memory (which bank is the input, which is the output, for the current dispatch).

The scheduler is also the source of `spatial_stall` — a signal that backpressures the spatial chain upstream of the engine while the engine runs. Downstream of the engine, a separate `engine_output_ready` signal lets the spatial chain consume the engine's BRAM output.

## Fallback heavy list (Wave 1 dispatch)

This task is Wave 1 dispatchable, so it runs before task 06 lands. In Wave 1, use the fallback heavy list documented in task 02. After task 06 produces the real list, the orchestrator re-runs this task to regenerate the scheduler against the real list. The script is deterministic; re-running is cheap.

## Register map dependency

The scheduler writes per-layer config registers via AXI4-Lite. The byte offsets must match the engine config register block in task 10. **Source of truth for the register map: task 10's file** (`docs/agent_tasks/10_engine_config_register_block.md` → "Register map" section). Read that section before writing the scheduler's AXI4-Lite write sequence. If the agent finds the two specs disagree, stop and ask the orchestrator to reconcile — do not invent a third addressing scheme.

## Context (read this before starting)

- Plan §6.4 describes the scheduler in detail. The scheduler is "mechanically generated from the LayerIR graph (which layers go to the engine, what order, what activations they consume and produce). It is not LLM-generated."
- Plan §6.4 also describes the four moving parts the scheduler coordinates: activation memory ownership, dispatch interface, backpressure, per-layer configuration loading. The script must handle all four.
- Plan §6.7 specifies that per-layer URAM base addresses come from the weight memory map (task 01). The scheduler writes these addresses into the engine's config registers before each dispatch.
- ResNet-50 has 16 residual blocks. The scheduler must keep the skip activation alive (in a separate BRAM bank) for the duration of all the main-path layers in that residual block. Use the LayerIR graph topology to figure out which residual blocks span engine-dispatched layers.

## Activation BRAM bank model

Use a small fixed number of activation BRAM banks (e.g. 4 or 6) and statically assign banks to the live activation tensors at each point in the dispatch sequence. The script does this allocation deterministically — it walks the LayerIR graph, tracks which activation is live, and assigns it to a bank using a simple longest-lifetime-first packing.

Output the bank assignment in the `_schedule.json` so it can be visualised:

```json
{
  "banks": [
    {"bank_id": 0, "max_bytes_used": 802816, "module_owners": ["node_conv_196_out", "node_relu_post_conv_196_out", ...]},
    ...
  ],
  "dispatches": [
    {
      "dispatch_index": 0,
      "module_id": "node_conv_284",
      "input_bank": 0,
      "output_bank": 1,
      "weight_base_word": NNNN,
      "channel_in": 512, "channel_out": 512,
      "kernel": [3, 3], "stride": [1, 1],
      "scale_mult": ..., "scale_shift": ..., "zero_point": ...
    },
    ...
  ]
}
```

## How to verify

1. Run the script. It should print the number of dispatches and the total BRAM banks allocated.
2. Open the generated `nn2rtl_scheduler.v` and confirm it parses under `iverilog -t null`.
3. Open the `_schedule.json` and confirm:
   - Every engine-dispatched layer in LayerIR has exactly one dispatch entry.
   - Input/output banks alternate sensibly (no dispatch reads and writes the same bank).
   - For each residual block that spans an engine dispatch, the skip activation's bank is reserved for the duration.
4. The FSM has exactly one transition arc out of every dispatch state (no deadlocks).
5. The scheduler's `engine_start` is sequenced strictly after all four config registers have been written.

## Out of scope

- Do NOT generate the engine. Task 00 owns the skeleton, tasks 07-11 own the sub-blocks.
- Do NOT generate the top wrapper. Task 02 owns that.
- Do NOT modify LayerIR or weights.
- Do NOT call any LLM agents.

## Success criteria

- Script runs in under 5 seconds.
- Output Verilog compiles under iverilog parse stage.
- Output JSON is well-formed and matches the schema above.
- The number of dispatch states matches the number of engine-dispatched layers in the heavy-module list.
- Activation BRAM bank allocation never double-assigns a bank to two live tensors at the same dispatch index.
