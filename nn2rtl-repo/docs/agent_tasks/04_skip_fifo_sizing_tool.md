---
task_id: 04
title: Skip-FIFO sizing tool (Phase A analytical + Phase B Verilator verify)
type: Python tooling + Verilator simulation harness
status: review
depends_on_phase_a: []
depends_on_phase_b: [00, 03, 07, 08, 09, 10, 11]
unblocks: [Phase 2 integration]
phase_b_status: review
---

# Task 04 — Skip-FIFO sizing tool

This task has two phases. **Phase A is Wave 1 work (no dependencies). Phase B is Wave 2 work (needs the engine to exist).** Treat them as two separate dispatchable units in the README — 04a and 04b.

## Goal

For each of ResNet-50's 16 residual adds, compute the depth required for the synchronisation FIFO on the skip path. Output one JSON file mapping each `node_add_*` module to its required FIFO depth, ready for task 02's top-level wrapper to consume.

This is a two-phase computation:

1. <a id="phase-a"></a>**Phase A — analytical** (cheap, deterministic, Wave 1 dispatchable now): use per-layer latency from LayerIR plus the dispatch order from task 03's schedule to compute the worst-case main-path-minus-skip-path delay, plus a backpressure margin.
2. <a id="phase-b"></a>**Phase B — Verilator verification** (cycle-accurate, Wave 2 dispatchable after the engine exists): build a small Verilator harness that simulates each residual block under representative workload (including the scheduler's stalls when an engine-dispatched layer runs inside the block), and confirms no FIFO under/overflow.

If Phase B flags a deadlock or overflow on any block, increase that block's depth and rerun.

## Deliverable

Two outputs:

1. `scripts/size_skip_fifos.py` — the sizing tool (CLI script).
2. `output/wrapper/skip_fifo_sizes.json` — the per-add FIFO depth map, consumed by task 02.

### `skip_fifo_sizes.json` schema

```json
{
  "method": "analytical + verilator-verified",
  "backpressure_margin_factor": 1.5,
  "fifos": [
    {
      "add_module_id": "node_add_198",
      "main_path_modules": ["node_conv_198", "node_relu_198", "node_conv_200", ...],
      "main_path_latency_cycles": NNN,
      "skip_path_latency_cycles": MMM,
      "engine_dispatches_in_main_path": K,
      "engine_worst_case_occupancy_cycles": LLL,
      "analytical_depth": ZZZ,
      "verified_depth": ZZZ,
      "verilator_status": "no_deadlock_no_overflow"
    },
    ...
  ]
}
```

### CLI

```
python scripts/size_skip_fifos.py \
    --network=resnet-50 \
    [--layer-ir=output/layer_ir.json] \
    [--schedule=output/rtl/nn2rtl_scheduler_schedule.json] \
    [--engine-modules=docs/agent_tasks/06_phase1_compression_candidates_HEAVY.txt] \
    [--margin=1.5] \
    [--out=output/wrapper/skip_fifo_sizes.json] \
    [--skip-verilator]   # for first analytical pass before the engine exists
```

## Context (read this before starting)

- Plan §6.5 is the authoritative spec for this task. Read it carefully.
- The sizing formula from §6.5: `FIFO_depth_initial = (main_path_latency_cycles − skip_path_latency_cycles) + 1.5 × backpressure_margin`. The backpressure margin accounts for the engine sequentialisation — when the engine runs a heavy layer in the main path, the spatial chain stalls and the skip FIFO must absorb the buildup.
- Latencies come from each module's `pipeline_latency_cycles` field in LayerIR.
- The "engine worst-case occupancy" is the cycle count the engine needs to run its largest dispatched layer. Read this from task 03's schedule JSON.
- Plan §6.5 step 3 is non-negotiable: "Verify in Verilator under representative workload. Run the full residual stage in cycle-accurate simulation with the engine actually dispatched in sequence. Confirm no deadlock, no FIFO underflow / overflow."
- This is what FINN's `auto_fifosize` does. We are doing it deterministically and naming the rule explicitly.

## Two-phase implementation

### Phase A — analytical

Build a graph of the LayerIR. For each `node_add_*`:
- Identify the main-path layers (the convolution chain from the residual-block entry to the add).
- Identify the skip-path layers (typically just a 1×1 conv or identity, depending on the block).
- Sum `pipeline_latency_cycles` along each path.
- Count how many engine-dispatched layers are in the main path; add their worst-case engine occupancy.
- `analytical_depth = (main_latency + engine_overhead) - skip_latency`, rounded up to the next power of 2 plus 50% margin.

Run this phase first. Emit `skip_fifo_sizes.json` with `verified_depth == analytical_depth` and `verilator_status: "not_yet_verified"`. Task 02 can use this as a placeholder until the engine exists and phase B can run.

### Phase B — Verilator verification

For each residual block:
- Generate a small testbench that instantiates the full main path (including the engine block for any dispatched layers) and the skip path, plus the FIFO at the analytical depth.
- Drive a representative input stream (use real activation values from goldens for the residual block's first layer).
- Watch for FIFO `full` (which would cause main-path backpressure) and FIFO `empty` while the add module is waiting (which would cause deadlock).
- If overflow detected: depth too small. Double and rerun.
- If underflow + deadlock: a different bug (FIFO sized OK but something else is wrong); flag and stop.
- If clean: lock the depth, write `verified_depth` and `verilator_status: "no_deadlock_no_overflow"`.

## How to verify the tool itself

1. Run phase A only (`--skip-verilator`). Confirm it produces a JSON file with 16 entries and `verilator_status: "not_yet_verified"` for all.
2. Open the JSON. For each add, confirm:
   - `main_path_latency_cycles` ≥ `skip_path_latency_cycles` (main is always longer)
   - `analytical_depth` is a power of 2 (so Verilog FIFO sizing is clean)
3. Once the engine exists, rerun without `--skip-verilator`. Confirm Verilator runs cleanly and `verilator_status: "no_deadlock_no_overflow"` for all 16.
4. Determinism: phase A output is byte-identical across runs.

## Out of scope

- Do NOT modify LayerIR or weights.
- Do NOT generate the engine itself.
- Do NOT generate the top wrapper.
- Do NOT call any LLM agents.

## Success criteria

- Phase A runs in under 10 seconds, produces a JSON file with 16 entries (one per residual add in ResNet-50).
- Phase B (when the engine exists) runs each block in cycle-accurate Verilator and reports `no_deadlock_no_overflow` for all 16.
- If any block fails verification, the failure is written into the JSON (`verilator_status: "deadlock_at_cycle_NNN"`) so the orchestrating Claude can route a Surgeon repair at the offending module.
