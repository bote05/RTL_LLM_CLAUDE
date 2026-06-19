# Parallel agent tasks — U250 deployment

Each `NN_*.md` in this directory is a self-contained brief for an agent instance. Tasks are written so that one agent can pick up one task and produce its deliverable without coordinating with other running agents.

Read the parent deployment plan first: [../nn2rtl_u250_deployment_plan.md](../nn2rtl_u250_deployment_plan.md). That is the master document; these task files are slices of it.

## Dispatch order

Tasks are grouped by wave. A wave starts when all its prerequisites are complete. **Wave 1 → Wave 1 review gate → Wave 2 → Phase 0 baseline → Wave 3 → integration.**

### Wave 1 — dispatchable now (no dependencies beyond the plan doc)

| Task | Brief | Type | Approx work | Notes |
| --- | --- | --- | ---: | --- |
| 00 | [Engine skeleton spec](./00_engine_skeleton_spec.md) | Design document | 1-2 days | **Linchpin** — Wave 2 cannot start until this is reviewed |
| 01 | [Weight memory map generator](./01_weight_memory_map_generator.md) | Python tooling | 1 day | |
| 02 | [LayerIR → top-level wrapper generator](./02_layerir_to_wrapper_generator.md) | TypeScript tooling | 2-3 days | Uses fallback heavy list until task 06 lands |
| 03 | [Scheduler generator](./03_scheduler_generator.md) | Python tooling | 2 days | Uses fallback heavy list until task 06 lands |
| 04a | [Skip-FIFO sizing — analytical phase](./04_skip_fifo_sizing_tool.md#phase-a) | Python tooling | 1 day | Phase A only in Wave 1 (cycle-accurate verification is Wave 2's task 04b) |
| 05 | [On-chip-weights contract](./05_on_chip_weights_contract.md) | New contract artefact | 1 day | |

These six tasks have no inter-task dependencies. They can all run simultaneously.

**Tasks 02 and 03 use a fallback heavy-module list** (hardcoded in those task files) while task 06 is blocked on Phase 0 baseline. After task 06 lands, the orchestrator re-runs tasks 02 and 03 against the real list. The wrapper and scheduler are deterministically regenerated; this is cheap.

### Wave 1 review gate — task 00 sign-off (CRITICAL)

**Wave 2 cannot dispatch until the orchestrator reviews and signs off on task 00's outputs.** Specifically:

1. `output/rtl/shared_engine_skeleton.v` compiles under `iverilog -t null`.
2. `docs/agent_tasks/00_engine_skeleton_spec_PORTS.md` covers every signal in the skeleton.
3. `docs/agent_tasks/00_engine_skeleton_spec_FSM.md` has one transition arc out of every state.
4. The orchestrator (or a separate review agent) confirms the port spec resolves every interface decision named in tasks 07-11. Open questions in PORTS.md = block Wave 2.

The PORTS document is the **single source of truth** for Wave 2's port lists. Wave 2 agents must copy port declarations verbatim. See "port consistency check" below.

### Wave 2 — dispatchable after Wave 1 review gate passes

| Task | Brief | Type | Depends on |
| --- | --- | --- | --- |
| 07 | [Engine MAC array sub-block](./07_engine_mac_array.md) | Foundry RTL gen | 00 |
| 08 | [Engine requantisation pipeline](./08_engine_requant_pipeline.md) | Foundry RTL gen | 00 |
| 09 | [Engine address generator](./09_engine_address_generator.md) | Foundry RTL gen | 00, 01 |
| 10 | [Engine config register block](./10_engine_config_register_block.md) | Foundry RTL gen | 00 |
| 11 | [BRAM-to-stream bridge](./11_bram_to_stream_bridge.md) | Foundry RTL gen | 00 |
| 04b | [Skip-FIFO sizing — Verilator verification](./04_skip_fifo_sizing_tool.md#phase-b) | Python + Verilator | 00, 07-11, 03 |

Tasks 07-11 can run simultaneously after the review gate. Task 04b runs after they integrate (it needs a functional engine instantiation to verify deadlock-freeness under backpressure).

### Phase 0 gate — full U250 baseline must be complete before Wave 3

The Vivado re-baseline of all 119 ResNet-50 modules against U250 (running in background as task `vivado_baseline.ts`) must produce `output/reports_u250/_aggregate.json` before task 06 can run.

### Wave 3 — dispatchable after Phase 0 baseline + task 06

| Task | Brief | Type | Depends on |
| --- | --- | --- | --- |
| 06 | [Phase 1 compression candidates](./06_phase1_compression_candidates.md) | Analysis | Phase 0 baseline |
| 12 | [Phase 1 improve sweep per spatial module](./12_phase1_improve_sweep.md) | Improve loop (multi-agent) | Phase 0, 06 |

Task 12 is itself parallel across the modules in the candidate list. One agent per module, all running simultaneously, with per-worker budget cap (see task 12 file).

### Pre-integration gate — task 04c (skip FIFO resize)

| Task | Brief | Type | Depends on |
| --- | --- | --- | --- |
| 04c | [Skip-FIFO Phase A revisit — throttled sizing](./04c_skip_fifo_resize_throttled.md) | Python + Verilog tooling fix | 04a, 04b |

**Why this is a gate.** Phase A's analytical depths verified clean in 04b, but at sizes that would consume ~50× the U250's total on-chip memory (~2 GB FIFO sum vs ~57 MB BRAM+URAM budget). 04c rewrites the analytical formula under the deployment's actual throttled-producer assumption (engine_busy gates the spatial chain at residual forks), re-runs Phase B against the smaller depths, and wires the throttle signal into the top wrapper. The integration phase cannot start until the FIFO sum fits the chip.

### Integration phase — Wave 4

| Task | Brief | Type | Depends on |
| --- | --- | --- | --- |
| 13 | [Integration & first-light](./13_integration_first_light.md) | Assembly + first synth | 00, 02, 03, 04b, 04c, 07-11, 12 |

Take the skeleton, the 5 sub-blocks, the wrapper, the scheduler, and the sized FIFOs. Produce the first integrated top-level that synthesises without errors. This is where implicit interface mismatches surface; it gets its own task because integration is not free.

## Port consistency check (Wave 2 contract)

Wave 2 agents must produce sub-blocks whose port lists match the canonical declarations in `docs/agent_tasks/00_engine_skeleton_spec_PORTS.md` exactly. Mechanical check:

- Each Wave 2 task file (07-11) requires the agent to include the port declarations *verbatim* from the PORTS document — no renaming, no widening, no helpful refactoring.
- The orchestrator runs a diff after each Wave 2 sub-block lands: extract the sub-block's port list from its `.v` file, compare against the entry in PORTS.md. Any mismatch is a hard fail.
- A small helper script `scripts/check_subblock_ports.py` does the diff. See [13_integration_first_light.md](./13_integration_first_light.md) for the check's contract.

## What changed if you ran Wave 1 already

- If task 04 was dispatched as one-shot: relabel it 04a (its Phase A is what was needed). 04b is a separate Wave 2 task.
- If tasks 02 / 03 already used the fallback heavy list: keep going. After task 06 lands, re-run them with the real list. Both scripts are deterministic; output is byte-identical given the same input.

## How to dispatch

Each agent task file is self-contained. Hand the agent the task file, give it read access to the repo, and let it produce its deliverable. The task file specifies:

- **What** the deliverable is (specific file paths, specific function signatures)
- **Why** — context the agent needs to make sensible decisions
- **Where** to read source data from (LayerIR paths, contract paths, existing scripts to mimic)
- **Verification** — how the user (or the orchestrating Claude) can check the work
- **Out of scope** — what the agent must not touch

When an agent finishes, the deliverables are committed and the parent plan can advance.

## Status tracking

Each task is independent. Track completion in the `Status` row at the top of each task file (`status: pending | in-progress | review | done`). The orchestrating Claude can sweep this directory to see what's left.
