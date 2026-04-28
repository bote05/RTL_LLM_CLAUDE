# nn2rtl milestones

A running log of end-to-end pipeline runs that have passed. The verified
RTL itself lives outside git (gitignored under `output/rtl/`); this file
captures the metrics and the run conditions so the result is reproducible
from the same commit hash.

---

## 2026-04-28: Layer 1 (full ResNet-50 first stage, 17 modules)

Commits in scope:
- `fb7e0d4` BRAM-backed line_buf_window + 7x7 stem reference + threading default fix
- `4f7183b` Add op pipelining + Vivado timing-met-without-WNS gate fixes

Pipeline run: `npm run pipeline -- ../checkpoints/resnet50_int8.pth`
(no `--only`, all 17 modules from `output/layer_ir.json`).

**All 17 modules pass.** Verilator: bit-exact (`max_error <= 1` rounding-tie
LSB across the entire layer). Vivado synth: `success=true, timing_met=true`
on `xc7a100tcsg324-1` at 50 MHz target. 15 modules first-shot, 2 modules
needed 1 Surgeon retry (both adds — Foundry's first attempt missed the
fused-scale fixed-point constants and Surgeon corrected them).

| Module | Op | Retries | LUT | FF | DSP | BRAM18 | Fmax (MHz) |
|---|---|---:|---:|---:|---:|---:|---:|
| layer0_0_conv1        | conv 7×7 | 0 |   3,805 |  1,920 |   1 |  7 |  63.7 |
| layer1_0_conv1        | conv 1×1 | 0 |   1,729 |  1,428 |   1 |  0 | 130.9 |
| layer1_0_conv2        | conv 3×3 | 0 |  10,756 |  4,553 |   2 | 45 |  78.9 |
| layer1_0_conv3        | conv 1×1 | 0 |   3,816 |  3,010 |   1 |  0 | 124.1 |
| layer1_0_downsample   | conv 1×1 | 0 |   3,979 |  3,007 |   1 |  0 | 124.8 |
| layer1_0_add          | add      | 0 |  22,601 | 10,900 | 240 |  0 | 314.7 |
| layer1_0_post_add_relu| relu     | 0 |   1,025 |  1,793 |   0 |  0 |  N/A* |
| layer1_1_conv1        | conv 1×1 | 0 |   3,179 |  3,026 |   1 |  0 | 122.3 |
| layer1_1_conv2        | conv 3×3 | 0 |  13,558 |  4,538 |   2 | 45 |  76.3 |
| layer1_1_conv3        | conv 1×1 | 0 |   6,051 |  3,012 |   1 |  0 | 129.9 |
| layer1_1_add          | add      | 1 |  32,161 | 11,663 | 240 |  0 | 278.4 |
| layer1_1_post_add_relu| relu     | 0 |   1,025 |  1,793 |   0 |  0 |  N/A* |
| layer1_2_conv1        | conv 1×1 | 0 |   3,259 |  3,041 |   1 |  0 | 122.3 |
| layer1_2_conv2        | conv 3×3 | 0 |  10,003 |  4,559 |   2 | 45 |  86.4 |
| layer1_2_conv3        | conv 1×1 | 0 |   5,444 |  3,014 |   1 |  0 | 128.0 |
| layer1_2_add          | add      | 1 |  32,097 | 12,244 | 240 |  0 | 278.4 |
| layer1_2_post_add_relu| relu     | 0 |   1,025 |  1,793 |   0 |  0 |  N/A* |
| **Sum**               |          |   |**155,513**|**75,294**|**734**|**142**|       |

\* The post-residual ReLU has no inter-FF setup paths -- every output
register is driven directly from primary inputs through one ReLU
combinational gate. Vivado prints `WNS = NA` and asserts "All user
specified timing constraints are met"; there is no critical path to
turn into an Fmax number. The orchestrator's gate (patched in `4f7183b`)
treats the explicit constraints-met line as authoritative for these
designs.

**Run characteristics:**
- Wall-clock: ~3 hours (~10:44 → 13:44 local time).
- Total LLM cost: **$8.02** (Foundry + Surgeon, claude-opus-4-7).
- Surgeon round-trips: 2 of 17 modules (= 12% retry rate). Both were
  layer1_*_add layers where Foundry got the architecture correct
  (timing 3/3 cycles, the new pipelined contract from
  `05_add_quantized.md`) but produced fused-scale constants that
  drifted by a few LSBs. Surgeon read the verifier evidence and
  corrected them in one shot. layer1_0_add passed first-shot in this
  run (it had needed Surgeon in the smaller solo run earlier in the
  same session) -- the doc updates absorbed in `4f7183b` improved
  Foundry's first-shot rate but the rate is not 100% on adds yet.
- Spec-hash cloning kicked in for all three `*_post_add_relu` layers
  in blocks 1 and 2 (identical to block 0's relu by shape) and
  shaved ~$0.50 each.

**Artix-7 100T capacity vs. fully-unrolled sum:**

| Resource | 100T capacity | Sum across 17 | Util if all instantiated |
|---|---:|---:|---:|
| LUT (Slice) | 63,400 | 155,513 | 245% |
| Register | 126,800 | 75,294 |  59% |
| DSP48E1 | 240 | 734 | 306% |
| BRAM18 | 270 | 142 | 53% |

The 245%-LUT and 306%-DSP totals are the **fully-unrolled worst case**;
inference of a single image streams the layers sequentially through a
shared compute engine, so the per-active-layer numbers are what
actually has to fit. Largest per-module footprint: `layer1_1_add` at
32K LUT (50% of 100T) and 240 DSPs (full DSP utilisation per active
add). Largest per-module BRAM: 45 BRAM18 (17%) on each of the three
3x3 spatial convs (BRAM-backed line buffer + weight ROM).

**What this milestone validates:**
1. The split-architecture spatial-conv path
   (`coord_scheduler` + `line_buf_window` + `conv_datapath`)
   handles every kernel/stride/padding combination in layer 1
   (1x1 stride 1, 3x3 stride 1 padding 1, 7x7 stride 2 padding 3).
2. The BRAM-backed `line_buf_window` rewrite (rotating-pointer +
   per-slot BRAM + `row_valid`) closes timing on Artix-7 100T for
   spatial convs that previously didn't fit -- 162K LUT / 178K FF
   on a 64K-LUT device became 10.7K LUT / 4.5K FF for the same
   `layer1_0_conv2`.
3. The 3-stage pipelined `add` pattern documented in
   `05_add_quantized.md` closes timing at OC=256 on Artix-7 100T
   (the legacy single-cycle combinational implementation maxed out
   the 240 DSPs and missed 50 MHz).
4. The full Foundry -> Verilator -> Vivado -> orchestrator gate ->
   Surgeon repair loop converges on 17/17 modules with a ~12%
   Surgeon-retry rate at $8.02 total cost.
