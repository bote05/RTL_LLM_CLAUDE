---
task_id: 04c
title: Skip-FIFO Phase A revisit — backpressure-throttled sizing
type: Python tooling refinement + re-run of Phase A and Phase B
status: review
depends_on: [04a, 04b]
unblocks: [13]
---

# Task 04c — Skip-FIFO Phase A revisit (backpressure-throttled sizing)

## Why this task exists

Task 04b verified that the Phase-A analytical FIFO depths do not deadlock and do not overflow in cycle-accurate Verilator simulation. The depths are correct *for that model*. They are also **physically infeasible** for the U250 target:

| Residual add | Analytical depth (entries) | Per-entry bus (estimated) | Total bytes | Fits on U250? |
| --- | ---: | ---: | ---: | :---: |
| `node_add_3` | 4,194,304 | 512 | ~2 GB | No |
| `node_add_4` | 524,288 | 512 | ~256 MB | No |
| `node_add_1`, `node_add_2` | 131,072 each | 512 | ~64 MB each | No |
| `node_add` (first) | 65,536 | smaller | ~16 MB | No |

U250 has ~45 MB UltraRAM + ~11.8 MB BRAM total. The current FIFO sum exceeds the entire chip's storage by ~50×.

The root cause is a modelling choice in Phase A: the input feed produces one sample per cycle continuously, and the FIFO must absorb the entire main-path latency, including the worst-case engine occupancy for every engine-dispatched layer in the residual block. With no backpressure, that turns 4 M cycles of engine occupancy into 4 M FIFO entries.

In real deployment we throttle the producer when the engine is busy. The FIFO then only needs to absorb the *pipeline fill delta*, not the entire engine occupancy.

## Goal

Revise Phase A to model the throttled case, re-run Phase A, re-run Phase B (task 04b), and produce FIFO depths that fit U250's on-chip memory budget without deadlock.

## Two equivalent ways to model "throttled"

Pick **option B** below — it matches what the top wrapper will actually do.

**Option A: Drop engine occupancy from the analytical formula.**

```
analytical_depth = (main_path_spatial_latency_cycles - skip_path_latency_cycles)
                 + pipeline_fill_margin
```

That is, do not add `engine_dispatches_in_main_path × engine_worst_case_occupancy_cycles`. The throttle gates the producer when the engine is busy, so those cycles do not consume FIFO entries.

This drops the analytical depths by 10–100×.

**Option B: Add throttle-aware backpressure to the Verilog discrete-event model in Phase B's `skip_fifo_block_dut.v`.**

Add a `throttle` input port that the producer reads. When the main path indicates "engine busy", the producer stalls and stops pushing to the FIFO. Re-run Phase B against the *new* (smaller) analytical depths. If overflow appears, grow the depth conservatively; that is the empirical answer.

This is more rigorous than option A alone because it lets Phase B detect cases where the throttle does not fully fix the imbalance (which would show up as overflow in the Verilog model).

**Recommendation: do both.** Option A produces the new analytical seed; option B verifies it under cycle-accurate timing.

## Required throttle handshake in the integrated design

The top-level wrapper (task 02) must wire a `spatial_throttle` signal from the engine's `engine_busy` output back to the spatial chain's *fork point* (where the main path and skip path diverge). When `engine_busy` is high, the spatial chain upstream of the fork stalls. This is the deployment-level backpressure the throttled Phase A assumes.

This wiring is currently *missing* from `output/rtl/nn2rtl_top.v`. Task 04c includes the wrapper-update step as part of its scope; without the throttle wire, the throttle assumption is theoretical.

## Deliverables

1. Updated `scripts/size_skip_fifos.py`:
   - Phase A: new analytical formula without the engine-occupancy term (option A).
   - Phase B: extend `skip_fifo_block_dut.v` with a `throttle` input and model the producer stalling when throttle is asserted.
2. Updated `output/wrapper/skip_fifo_sizes.json` with the new (smaller) `analytical_depth` and `verified_depth` values for all 16 entries.
3. Update `scripts/build_top_wrapper.ts` (task 02) so that the generated `nn2rtl_top.v` wires `engine_busy → spatial_throttle` between the engine and the spatial chain upstream of each residual fork point.
4. Regenerate `output/rtl/nn2rtl_top.v` (running the updated task 02 script).
5. Confirm:
   - All 16 FIFOs report `no_deadlock_no_overflow`.
   - Sum of FIFO bytes ≤ ~6 MB (target: ≤ 50% of U250's BRAM budget, leaving room for the activation ping-pong and the rest of the design).
   - The full integration parse (`top + scheduler + engine + sub-blocks`) still exits 0 under iverilog.

## Verification gate

This task is the precondition for **task 13 (integration first-light)**. The main agent runs the following before dispatching task 13:

```bash
node -e "
const fs=require('fs');
const s=JSON.parse(fs.readFileSync('output/wrapper/skip_fifo_sizes.json','utf8'));
const total=s.fifos.reduce((a,f)=>a+f.verified_depth*512,0);
const allClean=s.fifos.every(f=>f.verilator_status==='no_deadlock_no_overflow');
console.log('total fifo bytes:', total, '  all clean:', allClean);
if (total > 12*1024*1024) { console.log('FAIL: > 12 MB FIFO total; revisit task 04c'); process.exit(1); }
if (!allClean) { console.log('FAIL: not all FIFOs verified clean'); process.exit(1); }
console.log('PASS: FIFOs fit U250 budget and verify cleanly');
"
```

If this exits non-zero, task 13 must not dispatch.

## Out of scope

- Do NOT modify the per-layer .v files, the LayerIR, the goldens, or any contracts.
- Do NOT touch the engine sub-blocks (Wave 2 deliverables 07–11).
- Do NOT touch the scheduler (task 03).
- Do NOT call any LLM agents — this is mechanical tooling work.

## Notes

- The throttle handshake is one wire (the engine's existing `engine_busy` output). No new ports.
- The wrapper change is small: at each residual fork, instead of "tee" the data straight to both paths, the tee gates the main and skip path producers' `valid` outputs against `~engine_busy`.
- The wrapper change does not affect cycle-accurate latency for layers that do not include an engine dispatch — those residual blocks see `engine_busy = 0` continuously and behave as before.
- After this task, the **deployment plan §6.5** can be updated to name the throttled-Phase-A formula as the authoritative one. Phase A as-written (engine-occupancy in the formula) was correct only under the unthrottled assumption; the throttled formula is the real one.

## Success criteria

- Sum of all 16 verified FIFO byte sizes ≤ 12 MB (target ≤ 6 MB, hard ceiling 12 MB).
- All 16 entries `verilator_status: "no_deadlock_no_overflow"`.
- Full integration parse remains clean.
- Verification gate script (above) exits 0.
- Task 13 is then unblocked.
