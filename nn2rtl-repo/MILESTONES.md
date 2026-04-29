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
- `d4fbdc8` Serialized add template + LLM-as-default for add layers
  *(add rows below re-measured against this commit on 2026-04-29; see
  "Post-fix refresh" note below)*

Pipeline run: `npm run pipeline -- ../checkpoints/resnet50_int8.pth`
(no `--only`, all 17 modules from `output/layer_ir.json`).

**All 17 modules pass.** Verilator: bit-exact (`max_error <= 2` rounding-tie
LSB across the entire layer). Vivado synth: `success=true, timing_met=true`
on `xc7a100tcsg324-1` at 50 MHz target. **All 17 modules first-shot** after
the post-fix refresh — the two Surgeon retries on adds in the original run
disappeared once the serialized add architecture replaced the buggy
fully-parallel one.

| Module | Op | Retries | LUT | FF | DSP | BRAM18 | Fmax (MHz) |
|---|---|---:|---:|---:|---:|---:|---:|
| layer0_0_conv1        | conv 7×7 | 0 |   3,805 |  1,920 |   1 |  7 |  63.7 |
| layer1_0_conv1        | conv 1×1 | 0 |   1,729 |  1,428 |   1 |  0 | 130.9 |
| layer1_0_conv2        | conv 3×3 | 0 |  10,756 |  4,553 |   2 | 45 |  78.9 |
| layer1_0_conv3        | conv 1×1 | 0 |   3,816 |  3,010 |   1 |  0 | 124.1 |
| layer1_0_downsample   | conv 1×1 | 0 |   3,979 |  3,007 |   1 |  0 | 124.8 |
| layer1_0_add †        | add      | 0 |   1,551 |  5,930 |   2 |  0 | 106.6 |
| layer1_0_post_add_relu| relu     | 0 |   1,025 |  1,793 |   0 |  0 |  N/A* |
| layer1_1_conv1        | conv 1×1 | 0 |   3,179 |  3,026 |   1 |  0 | 122.3 |
| layer1_1_conv2        | conv 3×3 | 0 |  13,558 |  4,538 |   2 | 45 |  76.3 |
| layer1_1_conv3        | conv 1×1 | 0 |   6,051 |  3,012 |   1 |  0 | 129.9 |
| layer1_1_add †        | add      | 0 |   4,831 |  6,222 |   2 |  0 | 106.6 |
| layer1_1_post_add_relu| relu     | 0 |   1,025 |  1,793 |   0 |  0 |  N/A* |
| layer1_2_conv1        | conv 1×1 | 0 |   3,259 |  3,041 |   1 |  0 | 122.3 |
| layer1_2_conv2        | conv 3×3 | 0 |  10,003 |  4,559 |   2 | 45 |  86.4 |
| layer1_2_conv3        | conv 1×1 | 0 |   5,444 |  3,014 |   1 |  0 | 128.0 |
| layer1_2_add †        | add      | 0 |   4,842 |  6,225 |   2 |  0 | 106.6 |
| layer1_2_post_add_relu| relu     | 0 |   1,025 |  1,793 |   0 |  0 |  N/A* |
| **Sum**               |          |   | **79,878** | **58,864** | **20** | **142** |       |

† Add rows re-measured 2026-04-29 against commit `d4fbdc8`. The original
2026-04-28 run reported 22,601 / 32,161 / 32,097 LUT and 240 DSPs each
(buggy fully-parallel architecture; Surgeon retries 1 and 1 on `*_1_*` /
`*_2_*`). Replaced with full-pipeline (Foundry → Verilator → Vivado) re-runs
that produced first-shot passes with `max_error <= 2`. Combined cost of the
three add re-runs: $3.92.

\* The post-residual ReLU has no inter-FF setup paths -- every output
register is driven directly from primary inputs through one ReLU
combinational gate. Vivado prints `WNS = NA` and asserts "All user
specified timing constraints are met"; there is no critical path to
turn into an Fmax number. The orchestrator's gate (patched in `4f7183b`)
treats the explicit constraints-met line as authoritative for these
designs.

**Run characteristics:**
- Wall-clock: ~3 hours for the original 14 conv/relu modules (~10:44 →
  13:44 on 2026-04-28). Three add modules re-measured on 2026-04-29
  in ~25 min combined (~13:24 → 13:50).
- Total LLM cost: **$11.94** = $8.02 original 17-module run +
  $3.92 add re-measure run on the post-fix architecture
  ($1.30 / $1.36 / $1.26 for layer1_0/1/2_add). The original run's
  $8.02 included two Surgeon retries on the buggy adds; the
  re-measure was 0 retries because the serialized add architecture
  in `d4fbdc8` lets Foundry first-shot every add.
- Surgeon round-trips: **0 of 17 modules** after the post-fix refresh
  (was 2 of 17 in the original run, both adds). The one full-pipeline
  rerun would now project to roughly $5–6 with no retries.
- Spec-hash cloning kicked in for all three `*_post_add_relu` layers
  in blocks 1 and 2 (identical to block 0's relu by shape) and
  shaved ~$0.50 each.

**Post-fix refresh (2026-04-29):** The original 2026-04-28 add rows
were measured on the fully-parallel 256-channel datapath shipped in
`4f7183b`, which closed timing at 314 MHz but consumed all 240 DSPs
and 22-32K LUT per add. Commit `d4fbdc8` replaced that with a
serialized 3-stage pipeline (1 channel/cycle, 2 DSPs total, OC+3
latency) and made the LLM path the default for adds (deterministic
template gated behind `NN2RTL_DETERMINISTIC_ADD=1`, testing only).
The three add rows above were re-measured end-to-end through the
LLM pipeline on 2026-04-29 against `d4fbdc8` and show 1.5K-4.8K LUT
/ 2 DSP / 106.6 MHz Fmax with `max_error <= 2`. **DSP utilisation
across all 17 modules dropped from 306% to 8%; LUT from 245% to 126%.**

**Artix-7 100T capacity vs. fully-unrolled sum:**

| Resource | 100T capacity | Sum across 17 | Util if all instantiated |
|---|---:|---:|---:|
| LUT (Slice) | 63,400 | 79,878 | 126% |
| Register | 126,800 | 58,864 |  46% |
| DSP48E1 | 240 | 20 |   8% |
| BRAM18 | 270 | 142 |  53% |

The 126%-LUT total is the **fully-unrolled worst case**; inference
streams the layers sequentially through a shared compute engine, so
the per-active-layer numbers are what actually has to fit. Largest
per-module footprint after the refresh: `layer1_1_conv2` at 13.6K LUT
(21% of 100T). Largest per-module BRAM: 45 BRAM18 (17%) on each of
the three 3x3 spatial convs (BRAM-backed line buffer + weight ROM).
Largest per-module DSP: 2 (the add layers).

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
3. The original 3-stage fully parallel `add` pattern closes timing at
   OC=256 on Artix-7 100T but is the wrong area architecture: 512
   fused-scale multipliers consume all 240 DSPs and spill the rest
   into LUTs (22-32K LUT per add layer). The serialized 3-stage
   pipeline documented in `05_add_quantized.md` and shipped in
   `d4fbdc8` brings each add down to 1.5-4.8K LUT / 2 DSP at OC=256
   without changing the verifier-visible latency contract (still
   `pipeline_latency_cycles = OC + 3`).
4. The full Foundry -> Verilator -> Vivado -> orchestrator gate ->
   Surgeon repair loop converges on 17/17 modules. After the
   post-fix add refresh the retry rate is 0% (was 12% on the
   original 2026-04-28 run, both retries on the buggy adds).
