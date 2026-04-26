---
name: surgeon
description: Repair playbook for nn2rtl failures, including the full 17-class taxonomy, line-level rewrite patterns, and timing/handshake preservation constraints.
---
# Surgeon Skill

Use this skill when a module failed verification and needs a minimal targeted fix.

## Failure Classes

- `integer_overflow`
- `sign_extension_error`
- `bit_shift_wrong`
- `rounding_mode_wrong`
- `saturation_missing`
- `loop_bounds_incorrect`
- `array_indexing_error`
- `port_width_mismatch`
- `residual_addition_overflow`
- `missing_pipeline_register`
- `pipeline_latency_wrong`
- `reset_logic_broken`
- `enable_signal_ignored`
- `scale_factor_misapplied`
- `bias_term_missing`
- `batch_norm_not_folded`
- `synthesis_failed`

## Rewrite Constraints

- Preserve module name and interface
- Preserve the valid/ready contract
- Preserve the intended `pipeline_latency_cycles`
- Read compiler stderr first on `status="syntax_error"` before editing datapath logic
- Treat `[INVARIANT:*]` lines as protected unless raw evidence directly implicates them
- Rewrite only the smallest faulty source region
- Return `generated_by: "Surgeon"` and increment `attempt`
