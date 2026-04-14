---
name: assayer
description: Verification workflow reference for nn2rtl, including static C++ testbench sidecars, timing-aware Verilator runs, and enriched VerifResult population rules.
---
# Assayer Skill

Use this skill when verifying a generated or repaired module.

## Sidecar-Driven Testbench Flow

- The C++ testbench is static infrastructure at `tb/static_verilator_tb.cpp`
- Assayer generates a JSON sidecar only
- The sidecar must include:
  - module name
  - signal names
  - input and output widths
  - `pipeline_latency_cycles`
  - golden input and output paths
  - results path
- `results_path`, `golden_inputs_path`, and `golden_outputs_path` must be absolute filesystem paths because `run_verilator` executes the compiled binary from a temp build directory.

## Tool Order

1. `run_iverilog`
2. `run_verilator` with the sidecar path
3. Parse structured results into `VerifResult`

## `VerifResult` Rules

- Include `timing_pass`
- Include `timing_actual_cycles`
- Include `timing_expected_cycles`
- Include `failure_class` on fail and `null` on pass
- Keep `fix_hint` numerical and specific
