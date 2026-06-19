---
task_id: 12
title: Phase 1 improve sweep — one agent per spatial-compression module
type: Improve loop (LLM-driven, multi-agent parallel)
status: pending
depends_on: [Phase 0 baseline, task 06]
unblocks: [Phase 2 wrapper assembly]
---

# Task 12 — Phase 1 improve sweep

## Goal

Run nn2rtl's existing improve loop on each module in the spatial-compression candidate list (from task 06). Each module is an independent sub-task; the orchestrating Claude can dispatch one agent per module in parallel.

This task is the actual Phase 1 of the deployment plan ([§5](../nn2rtl_u250_deployment_plan.md)). The goal is to drop the spatial side's LUT sum from its current U250 baseline (~1.0 M LUTs raw across the spatial modules) to ~0.75 M after compression. That target uses the planning-assumption rates: −30% on heavy convs, −15% on mediums.

## Per-module sub-task

For each `module_id` in `docs/agent_tasks/06_phase1_compression_candidates_SPATIAL.txt`, an agent runs this sequence:

1. Read the module's current U250 baseline LUT, FF, DSP, BRAM18 from `output/reports_u250/<module_id>.vivado.json`.

2. Build the target list using these **flat, evaluable-in-order rules** (each rule is independent; multiple may fire for the same module):

   | Rule | Condition | Target to run |
   | --- | --- | --- |
   | R1 | `lut > 30000` | `reduce-lut` |
   | R2 | `bram18_count == 0 AND ff > 5000` | `use-bram` |
   | R3 | `ff > 50000` | `reduce-ff` |
   | R4 | `lut > 30000 AND no rule above fired (i.e. R1 was the only hit but it failed)` | `use-bram` as a fallback |

   The agent runs **every target whose rule fires**, in the order R1, R2, R3, R4. Skipping a rule because an earlier one fired is **not allowed** — `use-bram` and `reduce-lut` target different mechanisms (storage vs combinatorial); both can apply.

3. For each fired target, invoke the nn2rtl improve flow:
   ```
   npx tsx sdk/main.ts improve <module_id> <target> --network=resnet-50 --part=xcu250-figd2104-2L-e
   ```

4. **Per-target budget cap = $20.** If a single target run exceeds $20 of LLM cost, the worker aborts that target, marks it `aborted_budget_exceeded`, and continues to the next target. **Per-module total cap = $40.** If the module's running total reaches $40, the worker stops processing further targets for that module.

5. After all fired targets complete (or are aborted), write a per-module summary to `docs/agent_tasks/12_phase1_improve_sweep_<module_id>.md`:
   - Baseline metrics
   - Targets attempted (with which rule fired each)
   - Success / failure / aborted-budget status of each target
   - Cost spent per target and total
   - Final post-compression metrics
   - LUT and FF deltas in absolute and percentage

## Dispatch model

The orchestrating Claude maintains a worker pool. Each worker handles one module's sweep end-to-end. With ~15-20 modules in the spatial-compression list and an average of ~$5-15 of LLM cost per module, the full sweep is bounded at ~$100-300 if all run in parallel.

A worker's lifecycle:
1. Accept a `module_id`.
2. Pull the latest `output/reports_u250/<module_id>.vivado.json`.
3. Apply the decision rules from step 2 above.
4. Run the improve sequence.
5. Write the per-module summary.
6. Report back to the orchestrator.

Workers do **not** coordinate, but they **do not write to shared mutable state directly**. Mutable shared state is `knowledge/doc_lifecycle.json` (and the `knowledge/patterns/improved/` + `knowledge/references/improved/` directories under it).

**Worker isolation contract (mandatory):**

1. Each worker writes its outputs to a per-worker staging directory:
   ```
   output/improve_staging/<module_id>/
     ├── per-target subdirs from the improve flow
     ├── lifecycle_delta.json   (a small JSON describing the new improved/ entry to be added)
     ├── pattern.md              (the proposed pattern doc for the variant)
     └── reference.v             (the proposed RTL for the variant)
   ```
2. Workers **never** modify `knowledge/doc_lifecycle.json`, `knowledge/patterns/improved/`, or `knowledge/references/improved/` directly.
3. The orchestrator merges the staging directories into the canonical lifecycle registry **serially**, one worker at a time. The improve flow's existing auto-promote machinery must be modified to honour `NN2RTL_IMPROVE_STAGING_DIR=...` so it writes to the staging dir instead of the canonical location. (Tracked as a small SDK change; the worker itself just exports the env var.)
4. If two workers somehow target the same module, the orchestrator rejects the second worker at dispatch time using a simple in-memory worker registry.

**No file lock is needed** because workers never touch the canonical state. The orchestrator's serial merge is the synchronisation point.

## Context to read

- `docs/nn2rtl_u250_deployment_plan.md §5` — the Phase 1 spec.
- `sdk/improve.ts` — the existing improve flow implementation.
- `knowledge/patterns/improved/` — examples of successful improve outputs (e.g. `node_conv_248__use-bram.md`, `node_conv_298__reduce-ff.md`) so the agent knows what successful outputs look like.

## How to verify each per-module sweep

1. After the sweep, the module's `output/rtl/<module_id>.v` is either:
   - Unchanged (if all targets failed) → `12_phase1_improve_sweep_<module_id>.md` reports "no improvement".
   - Updated (if at least one target succeeded) → the new `.v` file passes Verilator bit-exact against the same goldens AND post-route Vivado synth on U250 reports a measurable LUT or FF reduction.
2. The lifecycle registry (`knowledge/doc_lifecycle.json`) has one new `improved/` entry per successful target.
3. The per-module summary markdown matches the schema and includes the before/after metrics table.

## Aggregate verification (orchestrator-side)

Once all per-module sweeps are done, the orchestrating Claude runs a final aggregation:

1. Sum the post-sweep U250 LUT counts across the spatial-compression list.
2. Compare against the pre-sweep sum.
3. Expected reduction: ~25-30% in total, weighted by module size.
4. Write a Phase 1 aggregate report to `docs/agent_tasks/12_phase1_improve_sweep_AGGREGATE.md` with:
   - Per-module before/after table
   - Network-wide spatial-side LUT total before / after / delta
   - Whether the deployment plan's −30%/−15% assumption was achieved, exceeded, or undershot

## Out of scope per worker

- Workers do NOT touch other modules' RTL.
- Workers do NOT modify contracts, pattern docs (other than the auto-promote entries from the improve flow), or pipeline state.
- Workers do NOT run the heavy-module engine generation (those modules are not in the spatial-compression list).

## Success criteria — per worker

- The sweep for the assigned module completes (whether or not improvements were found).
- A per-module summary markdown is produced.
- If the canonical RTL changed, it still passes Verilator bit-exact against the existing goldens.

## Success criteria — orchestrator aggregate

- All modules in the spatial-compression list have a summary file.
- Total spatial-side LUT reduction is ≥ 20% (with target 25-30%, planning assumption 30%).
- No module's RTL was corrupted or de-promoted.
