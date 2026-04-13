---
name: surgeon
description: Repair playbook for nn2rtl failures, including root-cause classes, line-level rewrite patterns, and invariants that must be preserved during targeted fixes.
---
# Surgeon Skill

Use this skill when a module failed verification and needs a minimal targeted fix.

## Root Cause Classes

- `arithmetic_overflow`
  - Symptom: outputs pin to `127` or `-128`
  - Typical fix: widen accumulator or saturate only at the final boundary
- `wrong_shift`
  - Symptom: outputs consistently scaled too high or too low by powers of two
  - Typical fix: correct `>>>` or `<<` amount and signedness
- `sign_extension_error`
  - Symptom: negative INT8 values become large positives
  - Typical fix: wrap operands with `$signed(...)`
- `wrong_loop_bounds`
  - Symptom: missing tail elements or repeated indices
  - Typical fix: repair loop limits or index arithmetic
- `missing_pipeline_register`
  - Symptom: cycle alignment mismatch or stale data
  - Typical fix: insert or reconnect the needed register stage
- `scale_factor_misapplied`
  - Symptom: all outputs are uniformly off despite correct shape
  - Typical fix: apply the quantization scale at the correct stage
- `rounding_mode_wrong`
  - Symptom: small, systematic off-by-one mismatches
  - Typical fix: adjust truncation versus round-to-nearest behavior

## Rewrite Pattern

1. Identify the root cause class.
2. Identify the smallest contiguous source region that can be changed safely.
3. Rewrite only that region.
4. Preserve:
   - `module_id`
   - `spec_hash`
   - public module interface
   - unaffected control flow

## Constraints

- Do not regenerate the full file.
- Do not rename the module.
- Do not change semantics outside the diagnosed region.
- Persist the repaired source with `write_verilog`.
- Return `generated_by: "Surgeon"` and increment `attempt`.
