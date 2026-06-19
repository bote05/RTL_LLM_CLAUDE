---
task_id: 13
title: Integration & first-light
type: Assembly + first synthesis run
status: pending
depends_on: [00, 02, 03, 04b, 05, 07, 08, 09, 10, 11, 12]
unblocks: [Phase 3 end-to-end verification, Phase 4a timing closure]
---

# Task 13 — Integration & first-light

## Goal

Take everything Wave 1, Wave 2, and Wave 3 produced — the engine skeleton with all 5 sub-blocks instantiated, the top-level wrapper, the scheduler, the sized skip FIFOs, the on-chip-weights contract, the URAM memory map, and the compressed spatial modules — and produce **the first integrated top-level design that synthesises without errors**.

This is where implicit interface mismatches surface. Treating integration as "happens automatically once all the pieces exist" is exactly the mistake the README explicitly warns about. This task does the assembly, runs the port-consistency checks, lights up iverilog on the full design, runs Verilator on a small fixed test input, and runs Vivado synth (not P&R yet) on the integrated top to produce the first-light area/Fmax numbers.

Goal of "first light": not bit-exact end-to-end (that is Phase 3 / task 12's job), and not timing-closed (that is Phase 4a). Goal is **the design synthesises, the data flows through, and we have numbers**.

## Deliverables

1. **`output/rtl/nn2rtl_top_integrated.v`** — the canonical integrated top, produced by task 02's wrapper generator and amended by this task with the engine + scheduler + skip FIFOs wired in.
2. **`output/reports_integrated/first_light_synth.json`** — Vivado synth report for the integrated top, in the same shape as the per-module `.vivado.json` files. LUT/FF/DSP/BRAM/URAM/Fmax of the whole design.
3. **`output/reports_integrated/first_light_verilator.json`** — Verilator simulation result on a small fixed test input (one ImageNet image, not 50k). Records: did the simulation complete, max_error vs golden, mismatch_count, end-to-end cycle count.
4. **`docs/agent_tasks/13_integration_first_light_REPORT.md`** — printable markdown summary: what synth produced, what verilator produced, what failed and how it was repaired (if applicable), what the next-phase risks look like.

## Required pre-task checks (do these first; abort if any fails)

Before assembly, the agent runs these checks. Each is a hard gate. If any check fails, **stop, write what failed to the report, and ask the orchestrator for a Surgeon repair on the offending artefact**.

1. **Wave 1 review gate already cleared** (task 00 sign-off): `output/rtl/shared_engine_skeleton.v` exists and parses; `00_engine_skeleton_spec_PORTS.md` and `00_engine_skeleton_spec_FSM.md` exist.
2. **Port consistency on every Wave 2 sub-block**: run `python scripts/check_subblock_ports.py` for each of `mac_array, requant_pipeline, address_generator, config_register_block, bram_to_stream_bridge`. All must exit 0. If any fail, the sub-block's RTL must be fixed before integration starts.
3. **Scheduler-engine register map agreement**: the scheduler's AXI4-Lite write addresses (in `output/rtl/nn2rtl_scheduler.v`) must match the engine's register map (in `output/rtl/engine/config_register_block.v`). The check is mechanical: grep both files for the byte offsets used and confirm they agree. Mismatch is a hard gate; one side must update.
4. **URAM weight map matches LayerIR**: `output/weights/weight_memory_map.json` covers all 53 conv2d layers in `output/layer_ir.json`; each layer's `weight_base_word` is non-negative and the total URAM word count is ≤ 1,280 × 4,096.
5. **Skip-FIFO sizes are present**: `output/wrapper/skip_fifo_sizes.json` has 16 entries, all with `verified_depth` set (i.e. task 04b ran cleanly, not just 04a).

## Assembly steps

1. **Run task 02's wrapper generator** with the final (post-task-06) heavy list. This produces the top-level wrapper with the right engine instantiation, the right number of spatial modules, and the right skip FIFOs.
2. **Run task 01's weight memory map generator**. Confirm the `.mem` file matches the layout the wrapper expects.
3. **Run task 03's scheduler generator**. Confirm the dispatch sequence covers every heavy module.
4. **Pull task 12's compressed spatial RTL** from the canonical `output/rtl/` directory (already auto-promoted by the improve loop).
5. **Pull each Wave 2 sub-block** from `output/rtl/engine/` and confirm they instantiate inside the skeleton stubs from task 00.
6. **Pull the engine skeleton** and confirm all 5 `// SUBBLOCK: <name>` stubs have been filled in.
7. Glue together. Filename: `output/rtl/nn2rtl_top_integrated.v` (the wrapper plus all `\`include` references to the sub-modules, OR the wrapper as a single self-contained file — both are acceptable; agent picks what is cleanest).

## Verification

1. **iverilog parse**: `iverilog -t null output/rtl/nn2rtl_top_integrated.v output/rtl/**/*.v` compiles without errors. Warnings are OK (will be addressed in timing closure).
2. **Verilator simulation on one image**: take any one of the existing ImageNet test images used for per-layer goldens (e.g. the first vector of `output/goldens/node_conv_196.goldin`). Feed it through the integrated design's input port. Sample the output (the final layer's logits). Record:
   - Did the simulation complete (didn't deadlock)?
   - What was the max_error vs the expected logits?
   - How many cycles did the design take end-to-end?
3. **Vivado synth (not P&R)**: run `synth_design -part xcu250-figd2104-2L-e -top nn2rtl_top_integrated` and capture the resource report. Write to `output/reports_integrated/first_light_synth.json`.
4. **Report**: write `13_integration_first_light_REPORT.md` containing:
   - Which checks passed.
   - Which sub-blocks needed Surgeon repair before integration succeeded (if any).
   - Synth resource utilisation as a percentage of U250 budget.
   - First-light end-to-end max_error (likely > 3 at this stage; real bit-exact verification is Phase 3).
   - List of suspected timing-critical paths for Phase 4a to tackle.

## What "first light passes" means

- iverilog parse succeeds: ✓
- Verilator simulation completes without deadlock: ✓ (max_error can be non-zero at this stage; deadlock is a hard fail)
- Vivado synth completes and reports finite area: ✓ (timing failure at this stage is expected; Phase 4a fixes it)
- LUT post-synth ≤ 95% of U250 budget: ✓ (or escalate to orchestrator with a compression follow-up)

If first light produces a max_error of, say, 50 across the output logits — that is **fine for this task**. Phase 3 chases bit-exactness. Task 13's job is to prove the structural integration is sound: the data flows through, the engine dispatches happen in the right order, the skip FIFOs absorb backpressure, no module deadlocks.

## Out of scope

- Do NOT attempt to fix bit-exact correctness here. That is Phase 3 / task 12 / Surgeon repair of individual modules.
- Do NOT attempt timing closure. That is Phase 4a.
- Do NOT regenerate sub-blocks. If a sub-block fails port consistency, request a re-run of that Wave 2 task — do not patch sub-block RTL in this task.
- Do NOT call any LLM agents directly. This task is mechanical assembly + verification; if a Surgeon repair is needed, the orchestrator routes it.

## Success criteria

- All five pre-task checks pass.
- `nn2rtl_top_integrated.v` exists and parses under iverilog.
- Verilator simulation completes without deadlock.
- Vivado synth produces a finite resource report.
- The report markdown is written and lists at minimum: synth LUT/FF/DSP/BRAM/URAM, Verilator end-to-end max_error and mismatch_count, end-to-end cycle count.
- The report names the top 5 timing-critical paths (preview for Phase 4a) and the top 5 highest-error output channels (preview for Phase 3).

This is the first integrated number we have for the whole design. Phase 3 closes the bit-exact gap; Phase 4a closes the timing gap. After this task lands, both phases run in parallel.
