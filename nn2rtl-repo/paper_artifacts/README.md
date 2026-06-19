# Paper artifacts — TScIT 45 (nn2rtl)

Maps each headline table cell in the paper to the in-repo report that produced it.
Sealed commit: `50c3054`. nn2rtl tool versions: Vivado 2025.2, Verilator, Icarus Verilog
(`iverilog -g2012`); ONNX frontend onnx 1.21.0 / onnxruntime 1.23.0 / opset 18.
FINN v0.10.1 on Vivado 2024.2; hls4ml 1.3.0 on Vitis HLS 2024.2.

## Table 3 — ResNet-8, three flows (ZCU104)
| Cell | Source |
|---|---|
| nn2rtl Top-1 87.19% | deterministic quantised reference (CIFAR-10 10k) |
| nn2rtl cycles 14,774 (e2e latency) | `output/resnet8/reports/verilator_resnet8_top_value/result.json` + `run.log` (PASS, 0 mismatch, 8 vectors) |
| nn2rtl Fmax 142.86 MHz / LUT 154,188 / DSP 1,717 / BRAM 199 / FF 64,728 / 7.3 W | ResNet-8 routed checkpoint timing/utilisation/power reports under `output/resnet8/reports/` |
| FINN baseline (25,760 LUT / 74 DSP / ~40 BRAM / 30,824 FF / 100 MHz / ~98,304 II / ~1,017 fps) | **build host (external)** — FINN `out_resnet8_zcu104/` estimate + impl reports |
| FINN max-fold (63,739 LUT / 569 DSP / 64.5 BRAM / 62,499 FF / 300.03 MHz / 9,216 II / ~32,555 fps / 9.1 W) | **build host (external)** — FINN `out_resnet8_zcu104_MAXFOLD/` post-route reports |
| hls4ml (89.10% / 200,938 LUT / 1,216 BRAM18 / 175,714 cyc) | **build host (external)** — hls4ml C-synthesis report |

## Table 4 — full-network P&R (Alveo U250)
| Cell | Source |
|---|---|
| ResNet-50 77.07% deployed | `output/reports/rederive_deployed_20260618.log` (BN-fold-aware re-derivation, 1500-img subset) |
| ResNet-50 83.33 MHz / 5,664,715 cyc / LUT 1,196,343 / BRAM 2,656 / DSP 6,983 / 16.0 W | kp4mp32_c16 routed checkpoint: `first_light_postroute_timing_kp4mp32_c16.rpt` (+ util/power) under `output/reports_integrated/` |
| ResNet-50 byte-exact 0/100,352 | `output/mobilenet-v2/reports/dw_quartet/resnet_inertness_stage1.log` and `stage2.log` (both at 5,664,715 cyc) |
| MobileNetV2 71.27% deployed | `output/reports/rederive_mbv2_20260618.log` (1500-img subset) |
| MobileNetV2 110.90 MHz / 1,184,731 cyc / LUT 322,628 / BRAM 1,812.5 / DSP 3,345 / 11.1 W | `output/mobilenet-v2/reports/synth/checkpoints/mbv2_route_postroute_util_final_c8.rpt` + `mbv2_route_final_c8.json` |
| MobileNetV2 byte-exact 8/8 | `output/mobilenet-v2/reports/final_bundle/e2e_result.txt` |

## Table 8 — derived efficiency
Computed from the locked routed-Fmax, vectorless power, and post-route LUT figures above
(FINN at its max-fold operating point). No additional synthesis.

## External (build-host) reports
The FINN baseline/max-fold reports and the hls4ml C-synthesis report were produced on a
separate WSL build host (Vivado/Vitis 2024.2) and are not in the main repo tree. Archive the
following before camera-ready: FINN `out_resnet8_zcu104{,_MAXFOLD}/report/*.json` and impl
utilisation/timing/power reports; hls4ml final C-synthesis report.
