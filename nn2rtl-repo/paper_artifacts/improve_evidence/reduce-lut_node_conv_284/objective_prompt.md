# Objective prompt - `reduce-lut`

The Improve Foundry agent (claude-opus-4-7) receives: the system prompt `nn2rtl-plugin/agents/improve_foundry.md`, the baseline RTL (`baseline.v`), and the per-target guidance (verbatim from `sdk/improve.ts` TARGET_GUIDANCE["reduce-lut"]) below.

**Deterministic acceptance rule** (evaluateImprovementTargets): `new.lut < baseline.lut * (1 - 0.05)`

**Measured result here:** LUT 227,703 -> 64,530  (-71.6%); verified bit-exact (Verilator) + Vivado-synthesized (`report_summary.json`); model generation in `transcript.json`.

---

```text
GOAL: reduce CLB LUT count by at least the configured delta (`new.lut < baseline.lut * (1 - reduceLutMinDelta)`).
HOW:
  - Move large constant tables into BRAM-backed ROMs (overlaps with `use-bram`).
  - Replace wide `case` / nested `if` chains over a discrete encoder with arithmetic / table lookups.
  - Combine repeated comparators against a counter (e.g. `cnt == 0 || cnt == 1 || ...`) into a single bound check.
PITFALLS:
  - Do not eliminate logic that's required by the contract. The Verilator gate runs first; functional regressions are caught before the LUT count is even read.
  - LUT count includes `LUT as Logic` AND `LUT as Memory` rows â moving distributed RAM to BRAM moves the count from `LUT as Memory` to BRAM18 only if the read is registered (see `use-bram`).
```
