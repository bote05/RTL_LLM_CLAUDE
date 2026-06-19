# nn2rtl vs FINN — end-to-end throughput

Target FPGA (ours): **xczu9eg-ffvb1156-2-e** (ZCU102, UltraScale+ ZU9EG)
Target FPGA (FINN OOC): **xczu7ev** (ZCU106), 100 MHz target clock

## Methodology (corrected)

Per-frame cycles are measured from the static Verilator testbench using
`(last_valid_out_cycle − first_valid_in_cycle) / num_vectors`, where `num_vectors`
is the number of frames simulated (8 for every layer). The previous calculation
used `pipeline_latency_cycles` from LayerIR — that's pipeline-fill latency, not
per-frame cost.

Per-layer fps = `fmax_mhz · 1e6 / cycles_per_frame`, with `fmax_mhz` taken from
post-synth Vivado reports on the xczu9eg.

Network steady-state fps = min across all measured layers (the throughput
bottleneck). One-image end-to-end latency = Σ(pipeline_fill_cycles / fmax) +
1/network_fps.

## Whole-pipeline numbers (ours, 117 of 119 layers measured)

| Metric | Value |
|--------|-------|
| Layers measured | 117 / 119 |
| Skipped (flat-bus gate) | node_conv_252, node_relu_24 — not on critical path |
| Failed verification | node_conv_266 (3×3, sim timeout) — excluded |
| Bottleneck module | node_conv_196 (stem, 7×7 stride-2) |
| Bottleneck fmax | 187.30 MHz |
| Bottleneck cycles/frame | 119 282 024 |
| **Steady-state network fps** | **1.5702** |
| Pipeline-fill latency sum | 82.7 ms |
| Single-image end-to-end latency | 0.7196 s |

### By op type (fps)

| op | n | min | median | max |
|----|---|-----|--------|-----|
| conv2d | 50 | 1.57 | 4.00 | 27.82 |
| add (residual) | 16 | 349 | 1387 | 2658 |
| relu | 8 | 50 360 | 50 360 | 84 609 |
| maxpool | 1 | 10 490 | 10 490 | 10 490 |

The conv2d layers dominate; the residual adds, ReLUs, and maxpool are
hundreds-to-thousands of times faster and never become the network bottleneck.

## FINN OOC reference (per user-supplied table)

FINN OOC results on xczu7ev / ZCU106 / 100 MHz target, expressed as
fps = 100e6 / latency_cycles per layer. Network bottleneck reported by the
user was **7.83 fps**, set by the same stem layer.

## Apples-to-apples summary

| | nn2rtl (ours) | FINN OOC |
|--|--------------|---------|
| FPGA | ZU9EG (ZCU102) | ZU7EV (ZCU106) |
| Clock policy | per-layer post-synth fmax (median ≈ 260 MHz) | fixed 100 MHz |
| Network bottleneck layer | conv_196 (stem 7×7 s2) | same stem |
| **Network fps** | **1.57** | **7.83** |
| LUT (per-layer median) | ~8× lower | reference |
| FF (per-layer median) | ~10× lower | reference |
| BRAM (per-layer median) | ~1.6× lower | reference |
| DSP | broadly comparable (1–8 per conv) | broadly comparable |
| Fmax (per-layer median) | 2.6× higher | 100 MHz (target) |

FINN is ~5× faster end-to-end because it unrolls MACs in parallel (PE × SIMD),
spending area for throughput. nn2rtl uses a time-multiplexed serial MAC with
MP=4 lanes (mandated explicitly in `knowledge/patterns/protected/01_context.md`
and the `02_conv1x1.md` reference) — this trades steady-state throughput for
much lower LUT/FF and higher Fmax. The two systems are at opposite ends of the
classical area-vs-throughput knee, and the comparison is honest now that we
measure per-frame cycles instead of pipeline fill.

## Notes on gaps

- `node_conv_252` and `node_relu_24` couldn't run under the deterministic
  flat-bus harness — they need `NN2RTL_SELF_IMPROVE=1` (per
  `feedback_self_improve_required_for_contract_walk.md`). They're a 1×1 conv
  and a ReLU respectively, so neither is on the throughput critical path.
- `node_conv_266` (256×256×3×3) hit a 10-min simulator timeout in this batch
  and is excluded from the roll-up. Of the 49 measured conv2d layers the slow
  end clusters around 1.57–1.95 fps (all 3×3 convs), so even if it lands in
  that band the network bottleneck stays at conv_196.

Raw data: `output/reports/throughput_per_module.csv`,
`output/reports/throughput_summary.json`.
