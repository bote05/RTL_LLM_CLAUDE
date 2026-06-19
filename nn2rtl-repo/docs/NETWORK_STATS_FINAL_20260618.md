# Final Hardware Stats — All Networks (2026-06-18)

Companion artifact to the thesis *"Autonomous LLM-Driven RTL Generation for Deep Neural Networks
via Multi-Agent Orchestration."* All numbers are read from **post-route Vivado reports** in this
repo (`report_timing_summary` / `report_power` / `report_utilization`), and the accuracy figures are
**independently re-derived** (logs cited per-row). Numbers here match the paper's Tables 3–4.

Power is **vectorless** (no SAIF) — *Low* confidence on the nn2rtl runs, *Medium* on FINN; treat power
as a **relative** comparison, not absolute silicon draw.

**Fmax convention (matches the paper, §4.5):** for a route that **meets** its target we report the
**guaranteed clock Fmax = 1000/T**; for a route that does **not** meet timing we report the
**achievable Fmax = 1000/(T − WNS)**. ResNet-50 (12 ns), ResNet-8/nn2rtl (7 ns) and FINN (3.333 ns)
all met → guaranteed clock. MobileNetV2 did **not** meet 7 ns (WNS −2.017) → achievable 110.90 MHz.

| # | Network | Device | Quant | Fmax (routed) | Cyc/frame | Throughput | Power |
|---|---------|--------|-------|---------------|-----------|------------|-------|
| 1 | **ResNet-50** (nn2rtl) | U250 | INT3/INT4 GPTQ | **83.33 MHz** (12 ns met) | 5,664,715 | **14.71 fps** (latency) | **16.0 W** |
| 2 | **MobileNetV2** (nn2rtl) | U250 | INT8 (per-ch DW) | **110.90 MHz** (7 ns missed, WNS −2.017) | 1,184,731 | **93.61 fps** (latency) | **11.1 W** |
| 3 | **ResNet-8** (nn2rtl) | ZCU104 | INT8 PTQ | **142.86 MHz** (7 ns met) | 14,774 | **~9,670 fps** (latency) | **7.3 W** |
| 4 | **ResNet-8** (FINN, max-fold) | ZCU104 | W4A4 QAT | **300.03 MHz** (3.333 ns met) | II = 9,216 | **~32,555 fps** (pipelined) | **9.1 W** |

**Throughput classes (paper §4.5):** nn2rtl is **M3** — cycle-accurate latency-bound, fps = Fmax ÷ cycles/frame
(one frame at a time). FINN is a **pipelined dataflow** estimate, fps = Fmax ÷ bottleneck II (9,216), far
smaller than total latency — which is why FINN's ResNet-8 throughput is **3.37×** nn2rtl's on the same chip.
These two fps numbers are **not the same metric**; do not compare them as if they were.

---

## Table 1 — device budgets (Vivado totals)

| Resource | Alveo U250 | ZCU104 |
|----------|-----------|--------|
| LUT | 1,728,000 | 230,400 |
| FF | 3,456,000 | 460,800 |
| DSP | 12,288 | 1,728 |
| BRAM36 | 2,688 | 312 |
| URAM | 1,280 | 96 |

## 1. ResNet-50 — nn2rtl, U250, INT3/INT4 GPTQ  (paper Table 4)

- **Checkpoint:** `output/reports_integrated/checkpoints/first_light_routed_kp4mp32_c16.dcp` *(dcp not published — GB-scale; reports are)*
- **Timing:** 12.000 ns, **WNS +0.102** (setup MET), WHS +0.010 (hold MET), 3,150,220 endpoints, 0 overlaps → **83.33 MHz** (= 1000/12, guaranteed). *(The `_c16` tag is legacy — the routed netlist closes at 12 ns, not 16 ns; the old 67.15 MHz figure was a 16 ns re-read and is wrong.)*
- **Cycles/frame:** 5,664,715 → **14.71 fps** · **Speedup** vs the 35.35 MHz / 13.54 M-cycle baseline (2.61 fps) → **5.64×**
- **Power:** **16.014 W** (Dyn 12.698 / Static 3.316, Low conf)
- **Utilization:** LUT 1,196,343 (**69.23 %**) · FF 1,189,379 (34.41 %) · BRAM 2,656 (**98.81 %**) · URAM 662 (51.72 %) · DSP 6,983 (56.83 %)
- **Accuracy:** **77.07 %** top-1 ImageNet (deployed, byte-exact). Float baseline **80.07 %** (torchvision-V2, 1,500-img subset); all-INT4 config **79.47 %** → ordering 80.07 > 79.47 > 77.07 is consistent.
  - **Independently re-derived 2026-06-18**: float 80.07 %, all-INT4 79.47 %, deployed 77.07 % — exact; BN-fold harness self-validated (folded == stock 80.07 %, Δ0.00), 4 trust gates pass. Logs: `output/reports/rederive_acc_20260618.log` + `rederive_deployed_20260618.log`.
- **Note:** the faster KPAR8 netlist (5,299,588 cyc) *places* at 67.95 MHz but its routing is SLR-weight-bus congestion-bound; even if routed it would give only ~12.8 fps < 14.71 — so kp4mp32 is the best routable version.

## 2. MobileNetV2 — nn2rtl, U250, INT8  (paper Table 4)

- **Checkpoint:** `output/mobilenet-v2/reports/synth/checkpoints/mbv2_route_routed_physopt_aggr_c7.dcp` *(dcp not published)*
- **Timing:** 7 ns target, **WNS −2.017** (does NOT meet 7 ns), WHS +0.004 (hold MET), 0 overlaps → **achievable 110.90 MHz** (= 1000/(7+2.017)). Best of the MBV2 routes (the 8 ns route closes at ~98 MHz).
- **Cycles/frame:** 1,184,731 → **93.61 fps** · **Speedup** vs the 93.4 MHz / 7.59 M-cycle baseline (12.30 fps) → **7.61×**
- **Power:** **11.077 W** (Dyn 7.932 / Static 3.145, Low conf)
- **Utilization:** LUT 322,629 (**18.67 %**) · FF 438,330 (12.68 %) · BRAM 1,812.5 (**67.43 %**) · URAM 235 (18.36 %) · DSP 3,345 (27.22 %)
- **Accuracy:** **71.27 %** top-1 ImageNet (deployed INT8, per-channel scales on the 17 depthwise layers; float 72.67 % → 1.40 % gap). The 67.27 % in older notes was the earlier per-tensor-only state.
  - **Independently re-derived 2026-06-18**: stock 72.67 %, folded-float 72.73 % (Δ +0.07 % → harness validated), deployed+A8 71.27 % — exact; 3 trust gates pass. Log: `output/reports/rederive_mbv2_20260618.log`.

## 3. ResNet-8 — nn2rtl, ZCU104, INT8 PTQ  (paper Table 3)

- **Checkpoint:** `output/resnet8/reports/synth/checkpoints/resnet8_post_routed.dcp` *(dcp not published)*
- **Timing:** 7.000 ns, **WNS +0.009** (setup MET), WHS +0.011 (hold MET), 264,336 endpoints, 0 overlaps → **142.86 MHz** (= 1000/7, guaranteed).
- **Cycles/frame:** 14,774 → **~9,670 fps**
- **Power:** **7.252 W** (Dyn 6.612 / Static 0.640, Low conf)
- **Utilization:** LUT 154,188 (**66.92 %**) · FF 64,728 (14.05 %) · BRAM 199 (63.78 %) · URAM 75 (78.13 %) · DSP 1,717 (**99.36 %** — DSP-bound)
- **Accuracy:** **87.19 %** top-1 CIFAR-10 (8719/10000, full test set). **Independently re-scored 2026-06-18**: h5 + ref ONNX + folded ONNX all 0.8719 — exact. Log: `output/reports/rederive_resnet8_cifar_20260618.log`.
- **Note:** improved this session from the RQ2-reported 14.45 fps (serialized-to-fit) to ~9,670 fps via conv-FSM pixel-pipelining + stem-accumulator pipelining + max_fanout replication, all byte-exact (8/8).

## 4. ResNet-8 — FINN, ZCU104, W4A4 QAT  (paper Table 3)

- **Checkpoint:** `…/vivado_zynq_proj_cx99v82t/.../top_wrapper_routed.dcp` (WSL build-host, Vivado 2024.2 — *outside this repo*)
- **Timing:** 3.333 ns, **WNS +0.047** (MET) → **300.03 MHz** (= 1000/3.333, guaranteed).
- **Throughput:** **~32,555 fps** (pipelined dataflow) = 300.03 MHz ÷ **9,216-cycle bottleneck II** (MVAU_rtl_0). FINN's own report estimates 36,169 fps assuming 333.33 MHz.
- **Power:** **9.106 W** (Dyn 8.372 / Static 0.735, Medium conf)
- **Utilization:** LUT 63,739 (27.66 %) · FF 62,499 (13.56 %) · BRAM 64.5 (20.67 %) · URAM 0 · DSP 569 (32.93 %)
- **Accuracy:** **86.68 %** top-1 CIFAR-10 — test accuracy logged by the brevitas QAT run (epoch 300); not independently re-scored here (needs a retrain or a QONNX-runtime rescore script).
- **FINN base fold** (for reference, paper Table 3): ~1,017 fps @ 100 MHz, II ≈ 98,304, LUT ~25,760 (11 %), DSP 74 (4 %), BRAM ~40 (13 %), FF 30,824 (7 %).

---

## Table 3 — ResNet-8 on ZCU104, three flows (CIFAR-10, native quant each)

| Metric | nn2rtl (INT8 PTQ) | FINN base (W4A4) | FINN max-fold (W4A4) | hls4ml (W8A8 QAT) |
|--------|-------------------|------------------|----------------------|-------------------|
| Top-1 | 87.19 % | 86.68 % | 86.68 % | 89.10 % |
| Fmax | 142.86 MHz | 100 MHz | 300.03 MHz | 137.32 MHz (est.) |
| Throughput | ~9,670 fps | ~1,017 fps | ~32,555 fps | did not fit |
| Cycles/frame | 14,774 (latency) | ~98,304 (II) | 9,216 (II) | 175,714 (C-synth) |
| LUT | 154,188 (66.92 %) | 25,760 (11 %) | 63,739 (28 %) | 200,938 (87 %) |
| DSP | 1,717 (99.36 %) | 74 (4 %) | 569 (33 %) | 488 (28 %) |
| BRAM | 199 tiles (63.78 %) | ~40 (13 %) | 64.5 (21 %) | 1,216 BRAM18 (~194 %) |
| FF | 64,728 (14.05 %) | 30,824 (7 %) | 62,499 (14 %) | 100,239 (22 %) |
| Power | 7.3 W | — | 9.1 W | — |
| Fit / route | routed, timing met | routed | routed | C-synth only, no route |

> The FINN figures and the hls4ml fit-oriented figures (200,938 LUT, 1,216 BRAM18, 175,714 cycles) come from
> **build-host reports outside this repo**; the nn2rtl figures are from **in-repo post-route reports**.
> Throughput classes differ: nn2rtl is cycle-accurate from a routed design (M3); FINN is an analytical estimate
> from its routed Fmax and a modelled II; hls4ml is a C-synth latency estimate and never routed.

## Table 4 — U250 place-and-route summary (both fully placed-and-routed)

| Metric | ResNet-50 (INT3/INT4) | MobileNetV2 (INT8 per-ch) |
|--------|----------------------|---------------------------|
| Top-1 | 77.07 % | 71.27 % |
| Routed Fmax | 83.33 MHz | 110.90 MHz |
| Throughput | 14.71 fps | 93.61 fps |
| Cycles/frame | 5,664,715 | 1,184,731 |
| LUT | 1,196,343 (69.23 %) | 322,629 (18.67 %) |
| BRAM tile | 2,656 (98.81 %) | 1,812.5 (67.43 %) |
| DSP | 6,983 (56.83 %) | 3,345 (27.22 %) |
| Power (vectorless) | 16.0 W | 11.1 W |

---

## Efficiency — energy & area (derived from the above)

| Network | fps | Power (W) | **fps/W** (inf/J) | **mJ/inference** | **fps/kLUT** |
|---------|-----|-----------|-------------------|------------------|--------------|
| ResNet-50 INT4 (U250) | 14.71 | 16.014 | 0.92 | 1088.6 | 0.012 |
| MobileNetV2 INT8 (U250) | 93.61 | 11.077 | 8.45 | 118.3 | 0.290 |
| ResNet-8 nn2rtl (ZCU104) | 9,670 | 7.252 | 1,333 | 0.75 | 62.7 |
| ResNet-8 FINN (ZCU104) | 32,555 | 9.106 | 3,575 | 0.28 | 510.8 |

- ResNet-8 (same chip): **FINN is 2.68× more energy-efficient and 8.14× more area-efficient** than nn2rtl — the dataflow-vs-time-mux tradeoff in one number.
- U250 (same chip): **MobileNetV2 is 9.20× more energy-efficient** than the BRAM-bound INT4 ResNet-50.
- **Cross-device caveat:** U250 designs (~3.2 W static) and ZCU104 designs (~0.6–0.7 W static) are *not* directly comparable on energy — compare only **within** a platform.

## Accuracy re-derivation — 2026-06-18 (independent)

| Net / config | Top-1 | How |
|--------------|-------|-----|
| ResNet-50 float baseline | 80.07 % | torchvision V2, 1,500-img subset (full-50k official = 80.858 %) |
| ResNet-50 all-INT4 (W4 per-ch + A8) | 79.47 % | `gptq_int4.py 256 1500 channel` |
| ResNet-50 deployed (INT3/INT4 + A8) | 77.07 % | `measure_deployed_configb_acc.py 1500 256` |
| MobileNetV2 float | 72.67 % | torchvision V2, 1,500-img subset |
| MobileNetV2 deployed (INT8 per-ch DW) | 71.27 % | `measure_deployed_mbv2_acc.py 1500 256` |
| ResNet-8 nn2rtl | 87.19 % | `eval_cifar10.py` (h5 + 2 ONNX agree, 8719/10000) |
| ResNet-8 hls4ml (test) | 89.10 % | `score_10k.py` (8910/10000; reported 89.11 % was *validation*) |
| ResNet-8 FINN | 86.68 % | brevitas QAT logged test acc (not re-scored here) |

All re-derivations: full logs under `output/reports/rederive_*.log`. BN-fold-aware harness (the ImageNet ones)
self-validates folded-float == stock before trusting the deployed number — the gate against BN double-counting.

## Caveats / provenance
- Power is **vectorless** (Vivado default switching) → relative comparison only.
- Fmax convention is the paper's (§4.5): guaranteed `1000/T` for met routes, achievable `1000/(T−WNS)` for un-met (only MBV2).
- nn2rtl throughput is latency-bound (M3); FINN is pipelined (II-bound); hls4ml is C-synth-only and never routed.
- ResNet-50 / MobileNetV2 = ImageNet (1,500-img subset for accuracy); ResNet-8 = CIFAR-10 (full 10k test).
- All nn2rtl designs are byte-exact vs golden (ResNet-50 / MBV2 vec0+vec1 0/100352; ResNet-8 8/8 mismatch 0).
- FINN/hls4ml fit/throughput figures are build-host (WSL) reports outside this repo; nn2rtl figures are in-repo post-route reports.
