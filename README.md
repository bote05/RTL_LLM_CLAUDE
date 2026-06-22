# nn2rtl — Autonomous LLM-Driven RTL Generation for Deep Neural Networks

Companion code and evidence artifact for the BSc thesis / TScIT 2026 paper
*"Autonomous LLM-Driven RTL Generation for Deep Neural Networks via Multi-Agent
Orchestration"* by **Daniel Botezatu** (University of Twente).

nn2rtl (neural network to register-transfer level) is an autonomous multi-agent
large language model (LLM) pipeline that translates trained, quantised PyTorch
convolutional networks into independently verified, on-chip-only synthesisable
Verilog for FPGAs, under a hard constraint forbidding external DRAM. Two
cooperating AI systems produce all the RTL and run the entire tool flow; the
author directed the work and built the deterministic pipeline and verification
environment the agents run and self-correct inside.

> The thesis/paper itself is published separately and is **not** part of this
> code artifact.

## Repository layout

| Path | Contents |
|---|---|
| [`nn2rtl-repo/`](nn2rtl-repo/) | The pipeline (System 1 per-module generation + System 2 integration), all generated RTL, golden vectors, layer IR, and post-route Vivado timing/utilisation/power reports. See [`nn2rtl-repo/README.md`](nn2rtl-repo/README.md) for full detail. |
| [`rq2_resnet8/`](rq2_resnet8/) | The ResNet-8 cross-flow comparison (nn2rtl vs FINN vs hls4ml) on the ZCU104, including the MLCommons Tiny reference model and provenance. |

## Networks and headline results

| Network | Board | Result |
|---|---|---|
| ResNet-50 (INT3/INT4) | Alveo U250 | routed, 83.33 MHz, 77.07% top-1 (1500-image ImageNet subset) |
| MobileNetV2 (INT8) | Alveo U250 | routed, 110.90 MHz achievable |
| ResNet-8 (INT8) | ZCU104 | routed, 142.86 MHz, 87.19% top-1 (full CIFAR-10) |

All three networks were generated and verified byte-exact against their
quantised references.

## Where the evidence lives

- Generated RTL: `nn2rtl-repo/output/rtl/` (and `output/resnet8/rtl/`, `output/mobilenet-v2/rtl/`)
- Per-network golden vectors / layer IR: `nn2rtl-repo/output/**/golden_vectors.json`, `layer_ir.json`
- Post-route timing / utilisation / power: `nn2rtl-repo/output/**/reports/`, `nn2rtl-repo/output/power/`
- Paper table-to-artifact map: `nn2rtl-repo/paper_artifacts/README.md`

## Reproducibility notes

- **Weights are not committed.** Weight/bias memories (`output/**/weights/`) are
  pipeline-generated and gitignored. The published RTL loads them via `$readmemh`;
  regenerate them with the export/quantisation scripts in `nn2rtl-repo/scripts/`
  (e.g. `export_resnet50_full.py`, `gptq_int4.py`, `generate_golden.py`).
- **Absolute paths in generated RTL.** The generated RTL embeds the absolute
  `$readmemh` paths from the machine it was produced on (e.g. `D:/RTL_LLM_CLAUDE/...`).
  The RTL is provided as static thesis evidence; to re-simulate, rewrite those
  paths for your checkout and regenerate the weight memories first.
- **Input models.** ResNet-8 is provided under `rq2_resnet8/model/` with
  checksums in `rq2_resnet8/ACQUISITION.md`. ResNet-50 is regenerated from
  torchvision via `nn2rtl-repo/scripts/export_resnet50_full.py`. The MobileNetV2
  ONNX is imported from torchvision (see `nn2rtl-repo/scripts/onnx_frontend.py`).
- **Python deps:** `nn2rtl-repo/requirements.txt`. Node toolchain: Node >= 20.

## Datasets

- **ImageNet** (ResNet-50, MobileNetV2) — used under its non-commercial,
  research-only terms (image-net.org). The dataset is **not** redistributed
  here; accuracy is reported on a 1500-image validation subset.
- **CIFAR-10** (ResNet-8) — Krizhevsky (2009), https://www.cs.toronto.edu/~kriz/cifar.html.
  Not redistributed here.

## Licence

This repository's original work is released under the [MIT Licence](LICENSE).
Redistributed third-party components keep their own licences; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
