---
task_id: 06
title: Phase 1 compression candidates list
type: Analysis
status: pending
depends_on: [Phase 0 — full U250 baseline]
unblocks: [02, 03, 12]
---

# Task 06 — Phase 1 compression candidates list

## Goal

Analyse the U250-baselined area numbers (from Phase 0 of the deployment plan) and produce two ranked lists:

1. **Heavy modules to dispatch through the shared engine.** Approximately the top 10 modules by LUT count on U250. These do not need Phase 1 compression — they are replaced by the engine.
2. **Spatial modules to send through Phase 1's improve sweep.** Modules that remain spatial but are large enough to be worth compressing. Ranked by priority (largest first).

This task is pure analysis. No LLM. No RTL. Just reading the U250 baseline JSON files and producing two text files plus a short report.

## Deliverable

Three files in `docs/agent_tasks/`:

1. `06_phase1_compression_candidates_HEAVY.txt` — newline-separated list of module IDs that go to the shared engine. ~10 entries.
2. `06_phase1_compression_candidates_SPATIAL.txt` — newline-separated list of module IDs to send through the improve sweep, in priority order (largest first). ~15-20 entries.
3. `06_phase1_compression_candidates_REPORT.md` — short markdown summary explaining the cutoffs, the rationale, and the expected total LUT impact.

## Inputs

- `output/reports_u250/*.vivado.json` — per-module Vivado reports for all 119 ResNet-50 passing modules, freshly baselined against the U250 part. Produced by Phase 0.
- `output/reports_u250/_aggregate.json` — the aggregate summary from the same baseline run.
- `output/pipeline_state.json` — the ResNet-50 pipeline state (used to confirm which modules are still passing).

## Method

Sort all 119 modules by U250 post-synth LUT count descending. Then apply these cutoffs:

- **Heavy threshold**: take the top N modules where N is chosen such that (a) N is between 8 and 12, and (b) their combined U250 LUT count covers at least 60% of the total network LUT sum. The engine will replace these.
- **Spatial-compression threshold**: from the remaining modules, take all those with U250 LUT count above some cutoff (suggested: ≥ 20,000 LUTs). These go through the improve sweep.
- **Trivially small**: the rest are below the compression-worth threshold and are left alone.

Tune the thresholds slightly if the cutoffs land in awkward places (e.g. a cluster of 5 modules all around the boundary — push the boundary so the cluster falls cleanly on one side).

## Report content

The `_REPORT.md` should contain:

1. **Top of network LUT distribution.** A table of the top 25 modules by U250 LUT with their per-module LUT, FF, DSP, BRAM18-equiv, Fmax.
2. **Cutoffs chosen.** The exact LUT thresholds used to split heavy / spatial-compression / leave-alone, plus the count of modules in each bucket.
3. **Engine-targeted sum.** Total LUT covered by the heavy list, as % of network sum.
4. **Phase 1 compression target.** Expected post-compression LUT reduction on the spatial-compression list, using planning-assumption rates (heavy convs −30%, mediums −15%). The plan §5 says this drops the spatial side from ~1.0M raw → ~0.75M after compression; this report should confirm or refute that estimate with the actual U250 data.
5. **Recommended dispatch order for task 12.** A rough priority order — biggest spatial-compression candidates first, since they have the most LUT to give up.

## How to verify

1. The HEAVY list has between 8 and 12 entries.
2. The SPATIAL list has between 10 and 25 entries, sorted by descending U250 LUT.
3. The REPORT's "engine-targeted sum" matches the sum of U250 LUTs across the HEAVY list.
4. The HEAVY + SPATIAL lists are disjoint (no module in both).
5. Re-running the analysis on the same inputs produces byte-identical output (deterministic).

## Out of scope

- Do NOT touch any RTL.
- Do NOT modify LayerIR.
- Do NOT call any LLM agents.
- Do NOT actually run the improve sweep — that is task 12.

## Success criteria

- All three deliverables exist.
- The HEAVY list is what tasks 02 and 03 will consume as their engine-modules input.
- The SPATIAL list is what task 12 will iterate over, one improve sweep per module.
- The REPORT gives the orchestrating Claude (and the supervisor) a clear picture of where the area is concentrated post-U250-baseline.
