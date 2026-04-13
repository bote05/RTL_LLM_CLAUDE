---
name: assayer
description: Verification workflow reference for nn2rtl, including iverilog and Verilator command patterns, testbench shape, golden vector comparison, and VerifResult rules.
---
# Assayer Skill

Use this skill when verifying a generated or repaired module.

## Command Patterns

- Syntax pass:

```bash
iverilog -g2012 -o /dev/null output/rtl/<module_id>.v
```

- Lint and simulation pass:

```bash
verilator --lint-only output/rtl/<module_id>.v
```

- Full execution flow may additionally compile a generated testbench and run the resulting executable.

## Testbench Format

- One deterministic vector stream per layer.
- Golden inputs and golden outputs come from `LayerIR`.
- Emit machine-readable output so per-index actual values can be parsed reliably.

## Comparison Procedure

1. Parse simulated outputs into numeric arrays.
2. Compare against `golden_outputs`.
3. Compute:
   - `max_error`
   - `mean_error`
4. If every compared value matches within tolerance, return `status: "pass"`.
5. Otherwise return `status: "fail"` and populate `fix_hint`.

## Error Classification Patterns

- Syntax failure
  - `status: "syntax_error"`
  - Populate `iverilog_stderr`
- Functional mismatch
  - `status: "fail"`
  - Populate `expected`, `got`, `max_error`, `mean_error`, and `fix_hint`

## VerifResult Population Rules

- Always include `module_id`.
- Always include `status`.
- Use `fix_hint` for a concrete human-readable diagnosis.
- Keep `expected` and `got` aligned by index whenever they are present.
