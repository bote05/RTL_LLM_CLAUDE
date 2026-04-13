---
name: conductor
description: Full pipeline state machine reference for nn2rtl, including transition rules, retry limits, cost tracking, Yosys-on-pass behavior, and updated JSON schemas.
---
# Conductor Skill

Use this skill when coordinating the full NN-to-RTL pipeline.

## State Table

| Current state | Trigger | Next state | Action |
| --- | --- | --- | --- |
| `pending` | scheduler tick | `generating` | invoke Foundry |
| `generating` | generation completed | `verifying` | invoke Assayer |
| `verifying` | verification passed | `pass` | record result and run Yosys |
| `verifying` | verification failed and retries remain | `fail_retry` | prepare Surgeon retry |
| `verifying` | verification failed and retries exhausted | `fail_abort` | stop retrying |
| `fail_retry` | scheduler tick | `generating` | invoke Surgeon |

## Retry Policy

- Maximum Surgeon retries per module: `3`
- `pass` and `fail_abort` are terminal
- Track aggregate `total_cost_usd` and `model_usage` in `PipelineState`

## Updated Contracts

- `LayerIR` references `weights_path` and `bias_path` instead of inline weights
- `VerifResult` includes timing and `failure_class`
- `PipelineState` includes cost tracking
