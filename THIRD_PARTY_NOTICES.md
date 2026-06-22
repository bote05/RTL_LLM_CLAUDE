# Third-party notices

This repository's original work is licensed under MIT (see [LICENSE](LICENSE)).
It also includes or depends on third-party material that retains its own
licence, listed below.

## Redistributed source and models

| Component | Path | Origin | Licence |
|---|---|---|---|
| nlohmann/json (`json.hpp`, v3.11.3) | `nn2rtl-repo/tb/third_party/json.hpp` | Niels Lohmann, https://github.com/nlohmann/json | MIT (SPDX header retained) |
| MLCommons Tiny ResNet-8 reference model | `rq2_resnet8/model/resnet8_ref.onnx`, `resnet8_folded.onnx` | MLCommons / TinyMLPerf, https://github.com/mlcommons/tiny | Apache-2.0 |
| MLPerf Tiny reference scripts | `rq2_resnet8/scripts/upstream_refs/*.py` | MLCommons / TinyMLPerf | Apache-2.0 (see `rq2_resnet8/scripts/upstream_refs/LICENSE`) |
| EEMBC evaluation functions | `rq2_resnet8/scripts/upstream_refs/eval_functions_eembc.py` | EEMBC / SiliconLabs `platform_ml_models` | per upstream (EEMBC); see file header |
| MLPerf Tiny sample-index arrays | `rq2_resnet8/scripts/upstream_refs/*.npy` | MLCommons / TinyMLPerf | Apache-2.0 |
| Brevitas quantisation patterns | `rq2_resnet8/training/brevitas/resnet8_w4a4.py` | AMD/Xilinx Brevitas | BSD-3-Clause (header retained) |
| torchvision model definitions (ResNet-50, MobileNetV2) | used at runtime, not vendored | PyTorch / Meta | BSD-3-Clause |

Provenance (source repository, pinned commit, and SHA-256 checksums) for the
ResNet-8 model and CIFAR-10 acquisition is recorded in
[`rq2_resnet8/ACQUISITION.md`](rq2_resnet8/ACQUISITION.md).

The SkyWater sky130 standard-cell library (`vendor/sky130/`, Apache-2.0, Google /
SkyWater) is referenced for download only and is **not** committed to this
repository; see `nn2rtl-repo/vendor/sky130/README.md`.

## Datasets (not redistributed)

- **ImageNet** — non-commercial research-only terms (image-net.org). Used for
  ResNet-50 / MobileNetV2 accuracy on a 1500-image subset.
- **CIFAR-10** — Krizhevsky (2009), https://www.cs.toronto.edu/~kriz/cifar.html.
  Used for ResNet-8.
