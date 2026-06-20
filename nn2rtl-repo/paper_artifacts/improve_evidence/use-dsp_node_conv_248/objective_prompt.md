# Objective prompt - `use-dsp`

The Improve Foundry agent (claude-opus-4-7) receives: the system prompt `nn2rtl-plugin/agents/improve_foundry.md`, the baseline RTL (`baseline.v`), and the per-target guidance (verbatim from `sdk/improve.ts` TARGET_GUIDANCE["use-dsp"]) below.

**Deterministic acceptance rule** (evaluateImprovementTargets): `new.dsp >= max(baseline.dsp + 1, 8)`

**Measured result here:** DSP 1 -> 25 (Vivado-synthesized, xczu9eg); verified bit-exact (Verilator) + Vivado-synthesized (`report_summary.json`); model generation in `transcript.json`.

---

```text
GOAL: map multipliers / MAC operations onto DSP48E2 slices instead of LUT-based ripple multipliers.
HOW (correctness-preserving levers, in order of preference):
  - Annotate the multiply with `(* use_dsp = "yes" *)` on the line before the assignment, OR factor the multiply into a registered intermediate `reg signed [W-1:0] mul_q; always @(posedge clk) mul_q <= a * b;` so Vivado can pattern-match a DSP cell. This alone moves the existing scalar multiply into a DSP without changing throughput.
  - Make both operands `signed [N-1:0]` of the same width before the multiply. Mixed signed/unsigned or width-mismatched operands routinely keep multipliers in LUT.
  - Register the multiplier output BEFORE feeding into shifts, saturation, or accumulation. A direct `(a*b) >>> N` combinational chain is one of the most common DSP-inference rejections.
  - Controlled banking / parallel MACs are ALLOWED if you keep correctness and timing. If a single annotated multiply does not push DSP count to the required threshold, you may bank the MAC across N parallel lanes (each lane its own `(* use_dsp = "yes" *)` registered multiply) and a balanced adder tree â provided the public interface, latency contract, and bit-exact output are unchanged. This is often necessary when the threshold is much higher than the baseline.
PITFALLS:
  - Do NOT blindly unroll a serial loop into a structurally different datapath that changes when `valid_out` fires or which weights pair with which inputs â that breaks Verilator. Banking is correctness-preserving only when each lane consumes exactly the same operands the original sequential MAC would have, just in parallel.
  - Multiplications inside `for ... if (...)` conditional generate-for blocks may map differently across Vivado versions; prefer unconditional registered multiplies inside an `always_ff`.
  - Adding lanes increases LUT fan-out around the adder tree. If you also have `reduce-lut` in the targets, balance carefully â banking too aggressively trades DSPs for LUTs.
```
