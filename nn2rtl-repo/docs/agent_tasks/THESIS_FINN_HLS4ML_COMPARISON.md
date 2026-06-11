# Thesis comparison: this work vs FINN / hls4ml (on-chip, no-DRAM constraint)
*Synthesis of a 103-agent deep-research run (2026-06-11); every claim below survived 3-0 adversarial verification. Raw claims + quotes: `AGENT_MEMORY_EXPORT/thesis_finn_hls4ml_research_VERIFIED_CLAIMS.json`.*

## 1. Executive answer
**No local FINN or hls4ml runs are required.** Official, citable, on-chip-verified numbers exist for both FINN comparison cells (ResNet-50 and MobileNet); the hls4ml cells are *structurally empty* (no official ImageNet-scale numbers exist anywhere — itself a citable finding), and the quantization-ingestion mismatch makes author-run FINN builds methodologically impossible rather than merely laborious (§5).

## 2. The comparison table (all on Alveo U250 unless noted, all weights on-chip)

| Design | Quantization | Top-1 | Throughput | Latency | Fmax | Resources | Source |
|---|---|---|---|---|---|---|---|
| **FINN ResNet-50** | **W1A2** (1b weights / 2b acts; 8b first/last, 4b residual) | **67.27%** (binary) / 69.85% (ternary) | 2703 FPS (paper) / 2000 FPS (launch blog) / ~3000 (v0.6 blog) | 1.9–2 ms | 195 MHz | 1027 kLUT, 3870 BRAM18, 1611 DSP | arXiv:2011.07317 Table II; xilinx.github.io/finn blogs |
| **This work ResNet-50** | **INT4/INT3 GPTQ per-OC** | **77.07%** | 11.9 fps¹ | ~79 ms¹ | 67.15 MHz routed¹ | 1196 kLUT, 2642 BRAM36, 6983 DSP | this thesis (routed dcp + timing rpts) |
| **FINN MobileNet-v1** (NOT v2) | **W4A4** (8b first layer) | **70.4%** | 1800 FPS (U250) / ~450 FPS (ZCU104) | — | — | ~2 MB weights | finn-examples README/blogs |
| **This work MobileNetV2** | **INT8 per-channel PTQ** | **71.27%** | 73.2 fps¹ | ~13.7 ms¹ | 86.67 MHz routed¹ | 326 kLUT (synth), 1809 BRAM36 | this thesis |

¹ Numbers from the first full routed implementations; final-bundle re-route pending (expected ResNet ~15–18 fps, MBV2 ~80–97 fps — update on completion).

## 3. The honest framing (what each side wins)
- **FINN wins raw throughput by ~2 orders of magnitude** (2703 vs ~12 FPS on ResNet-50). This is architectural: FINN's per-layer dataflow pipelines all layers concurrently at W1A2 weight sizes that make massive parallelism affordable on-chip. Cite it plainly.
- **This work wins accuracy at on-chip parity: +9.8 points on ResNet-50 (77.07 vs 67.27)** on the same device under the same no-DRAM constraint. The thesis claim is occupancy of an accuracy point that the deterministic flow cannot reach (no official FINN ResNet-50 exists above W1A2 — claims [5],[11],[17]), not throughput superiority.
- MobileNet cell: accuracy roughly at parity (71.27 vs 70.4) **but the architectures differ** (v2 with inverted residuals vs v1) — state the version difference in every mention (claims [7],[14],[16],[19]).

## 4. On-chip (no-DRAM) verification — per cell
- **FINN RN50**: ResBlock conv weights in BRAM, FC weights + activations/FIFOs in URAM; the paper itself calls it "the largest topology ever implemented with on-chip weights" in that architecture. *Caveat to quote*: the paper notes FC weights "could fall back to HBM/DDR on other platforms" — on-chip status is U250-specific (claim [2]).
- **FINN MNv1 (U250)**: the official folding config sets every MVAU to `mem_mode internal_decoupled` + `ram_style block`; **no layer uses `external`** → genuinely DRAM-free for weights; weights are runtime-loaded into on-chip memory by the driver, not streamed (claim [10]). ZCU104 config likewise all-internal (claim [21]); v0.6 introduced URAM weight storage and the MNv1 example showcases it (claim [15]).
- DRAM-freeness must be checked **per example** in FINN (DDR remains a supported option) — your designs are DRAM-free by construction; FINN's flagship examples happen to be too.

## 5. Empty cells & why author-run builds are NOT the answer
| Cell | Status | Blocker |
|---|---|---|
| FINN ResNet-50 @ INT4/INT8 | **does not exist officially** | FINN ingests Brevitas QAT networks; it cannot consume this work's PTQ (GPTQ post-training) quantization. A quantization-matched FINN build is impossible *by design* — cite as a structural difference between flows, not a missing experiment. |
| FINN MobileNet**V2** | **does not exist officially** | v2's inverted residuals unsupported in the official examples; only v1 W4A4 exists. |
| hls4ml ResNet-50 / MobileNetV2 | **no official numbers anywhere** | hls4ml's domain is small/tiny models; this work's own Tier-A study (14/17 layers compared on Artix-7, 2–10× LUT/FF advantage, 3×3 convs unsynthesizable at scale) is the correct and only available comparison methodology — cite the absence + the per-layer study. |

## 6. Reproducibility audit
- **finn-examples** ships prebuilt bitfiles + PYNQ drivers + Jupyter notebooks per example, and documents full rebuild-from-source via the FINN compiler (claim [12]) → the FINN numbers are reproducible-in-principle; rebuilds need the pinned FINN/Vitis toolchain in the repo docs.
- **Important nuance for the thesis**: the official notebooks publish *no* accuracy/throughput numbers — they ship measurement code (a 50k-image validation loop + `accel.throughput_test()`); the citable figures live in the paper/README/release blogs (claim [18]). If a physical U250 is available, re-measuring FINN's prebuilt RN50 bitstream is optional extra credit, not a requirement.
- **Number discrepancies to handle in writing**: RN50 throughput appears as 2000 FPS (2020 launch blog), 2703 FPS (the paper's Table II), ~3000 FPS (v0.6 release blog) — cite the peer-reviewed Table II as canonical and footnote the blogs. The build script's `target_fps=300` / 4.0 ns are **build targets, not measurements** (claim [6]) — do not cite them as results.

## 7. Primary sources
1. arXiv:2011.07317 — *"Evaluating and optimizing FPGA-based dataflow..."* (RN50-W1A2 paper; Table II) — claims [0–2]
2. xilinx.github.io/finn/2020/03/11/rn50-released.html — claims [3]
3. github.com/Xilinx/finn-examples — build/resnet50/build.py [4–6], build/mobilenet-v1 [7–10], README [11–12], notebooks/2_imagenet_with_cnns.ipynb [16–18], ZCU104 folding config [21]
4. xilinx.github.io/finn v0.6 (2021-06-15) [13–15] and v0.5b (2020-12-17) [19–20] release notes
