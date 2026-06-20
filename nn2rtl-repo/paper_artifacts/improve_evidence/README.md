# Objective-directed RTL improvement — curated evidence

This is a small, self-contained evidence package for nn2rtl's **`improve`** command: you hand the
LLM an already-generated, already-passing RTL module plus a **fixed-objective prompt** (e.g.
"use more DSP", "reduce LUT"), and it produces an optimized variant that is then **deterministically
verified** (bit-exact in Verilator + re-synthesized in Vivado) before being accepted.

It is intentionally curated — two worked examples with the full prompt→result→verification trail —
rather than the entire (multi-GB, gitignored) `output/improve/` run directory.

## The command

```
npx tsx sdk/improve.ts <module_id> --targets=<t1>,<t2>,...   # e.g. --targets=use-dsp
```
(dispatched from `sdk/main.ts`; batch sweeps via `scripts/improve_sweep.ts`).

## Fixed objective taxonomy (`sdk/improve.ts` → `IMPROVEMENT_TARGETS`)

`use-dsp · use-bram · reduce-lut · reduce-ff · improve-fmax · reduce-latency · increase-throughput`

Each target has (a) a prose **guidance prompt** (`TARGET_GUIDANCE`, GOAL/HOW/PITFALLS) given to the
agent, and (b) a **deterministic acceptance rule** (`evaluateImprovementTargets`) that decides
success from measured PPA — the LLM cannot "declare victory"; the rule does.

## What the agent receives (the prompt side)

1. System prompt: `nn2rtl-plugin/agents/improve_foundry.md` (the Improve Foundry agent, `claude-opus-4-7`).
2. Per-target guidance: `objective_prompt.md` in each example below (verbatim from `TARGET_GUIDANCE`).
3. The baseline RTL: `baseline.v`.

## The two worked examples

| Folder | Objective | Measured (Vivado, xczu9eg) | Acceptance rule | Verified |
|--------|-----------|----------------------------|-----------------|----------|
| `use-dsp_node_conv_248/` | `use-dsp` | **DSP 1 → 25** | `new.dsp ≥ max(baseline.dsp+1, 8)` | bit-exact + synth |
| `reduce-lut_node_conv_284/` | `reduce-lut` | **LUT 227,703 → 64,530 (−71.6 %)** | `new.lut < baseline.lut·(1−0.05)` | bit-exact + synth |

## Files in each example folder

- `objective_prompt.md` — the exact per-target guidance + acceptance rule + measured result.
- `baseline.v` — the RTL the agent was asked to improve (the run's input).
- `improved.v` — the accepted optimized variant (`knowledge/references/improved/<module>__<target>.v`).
- `transcript.json` — the model's generation stream for the successful run (assistant + result
  messages: its reasoning and the `write_verilog` tool calls that produced `improved.v`). The input
  prompt itself is the system prompt + `objective_prompt.md` + `baseline.v` above.
- `report_summary.json` — the deterministic verdict + per-attempt PPA metrics (trimmed from the full
  `output/reports/improve_<module>__<target>.json`, which inlines the transcript and raw Vivado dump).

## Notes / provenance

- These are **per-module optimization demonstrations**: each module is improved and verified in
  isolation. Whether a given variant was carried into the final routed network is a separate question
  — this package documents that the objective-directed improve command works on demand, with numbers.
- `node_conv_248`'s `use-dsp` run was chained on a prior `use-bram` pass, so its `baseline.v` is the
  BRAM-optimized variant (1 DSP); `use-dsp` then added the DSP-banking (→ 25 DSP).
- Absolute machine paths in the transcripts/reports have been **scrubbed** (`<repo>`, `<home>`, etc.);
  no values or RTL were altered.
- Full source (not duplicated here, already in the repo): `sdk/improve.ts`,
  `nn2rtl-plugin/agents/improve_foundry.md`, and the complete reports under `output/reports/improve_*.json`.
