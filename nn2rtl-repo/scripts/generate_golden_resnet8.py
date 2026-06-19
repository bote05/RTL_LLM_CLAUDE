#!/usr/bin/env python3
"""Generate LayerIR + golden vectors for the MLPerf Tiny ResNet-8 (CIFAR-10).

NEW launcher: drives the EXACT code path scripts/generate_golden.py uses for
.onnx checkpoints (scripts.onnx_frontend.build_pipeline_ir_from_onnx +
generate_golden._write_layer_ir_json) with ONE injection and ZERO edits to
existing files:

    sys.modules['imagenet_util'] = scripts.cifar10_util

is seeded BEFORE scripts.onnx_frontend is imported, so the frontend's
NN2RTL_IMAGENET_CALIB branch (onnx_frontend.py step 4 does a plain
``import imagenet_util as _iu`` after inserting <repo>/scripts on sys.path;
Python resolves it from sys.modules first) feeds real CIFAR-10 images
instead of 224x224 ImageNet parquet shards.

Quantization config (applied via os.environ.setdefault, i.e. an explicit
environment still wins; module-level env reads in onnx_frontend happen at
import time, AFTER these defaults are set):

  NN2RTL_WEIGHT_BITS=8       INT8 weights (USE_GPTQ auto-off at 8 bits)
  NN2RTL_IMAGENET_CALIB=256  256 real CIFAR feeds = 8 golden-head TEST images
                             + 248 official MLPerf Tiny calibration samples
  NN2RTL_GOLDEN_VECTORS=8    8 golden vectors (= the 8 golden-head feeds)
  NN2RTL_STEM_PER_CHANNEL=1  per-OC weight scales for all groups==1 non-1x1
                             convs (repo default, kept)
  NN2RTL_PW_PER_CHANNEL=0    1x1 convs stay per-TENSOR (repo default, kept;
                             see onnx_frontend.py 'improvement E' note)

Also performs the registration step sdk/import_network.ts prepareNetwork
would do (without touching networks.json): writes the
<output>/layer_ir.json.checkpoint fingerprint sidecar containing the absolute
checkpoint path, so sdk ensureLayerIr accepts the IR as non-stale
(pathFingerprintKey normalizes separators/case).  Additionally writes
<output>/golden_labels.json documenting golden/calibration provenance.

Usage (from the repo root, mirrors generate_golden.py's ONNX branch):
  python scripts/generate_golden_resnet8.py [checkpoints/resnet8.onnx]
      [--name resnet8] [--output-dir output/resnet8] [--samples 256]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate <output-dir>/layer_ir.json (+ goldens/weights) "
                    "for ResNet-8 with real CIFAR-10 calibration.")
    parser.add_argument("model_path", nargs="?", default="checkpoints/resnet8.onnx")
    parser.add_argument("--name", default="resnet8")
    parser.add_argument("--output-dir", default="output/resnet8")
    parser.add_argument("--samples", type=int, default=256,
                        help="Real calibration feeds (default 256).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Env defaults BEFORE any scripts.* import (onnx_frontend reads
    # NN2RTL_WEIGHT_BITS / USE_GPTQ at module import time).
    os.environ.setdefault("NN2RTL_WEIGHT_BITS", "8")
    os.environ.setdefault("NN2RTL_IMAGENET_CALIB", str(args.samples))
    os.environ.setdefault("NN2RTL_GOLDEN_VECTORS", "8")
    os.environ.setdefault("NN2RTL_STEM_PER_CHANNEL", "1")
    os.environ.setdefault("NN2RTL_PW_PER_CHANNEL", "0")

    # The injection: must precede the scripts.onnx_frontend import.
    import scripts.cifar10_util as cifar10_util
    sys.modules["imagenet_util"] = cifar10_util

    import torch
    torch.manual_seed(0)                      # mirrors generate_golden.main()

    from scripts.paths import detect_repo_root
    rr = detect_repo_root(__file__)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = rr / output_dir
    os.environ["NN2RTL_OUTPUT_DIR"] = output_dir.as_posix()

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = rr / model_path

    from scripts.generate_golden import _write_layer_ir_json
    from scripts.onnx_frontend import build_pipeline_ir_from_onnx

    payload = build_pipeline_ir_from_onnx(
        onnx_path=model_path,
        repo_root=rr,
        model_name=args.name,
        num_calibration_samples=int(os.environ["NN2RTL_IMAGENET_CALIB"]),
        rtl_compat_conv=False,
    )
    output_path = _write_layer_ir_json(rr, payload)

    # Registration fingerprint (replicates sdk/import_network.ts
    # prepareNetwork's sidecar write; content = absolute checkpoint path).
    fingerprint_path = Path(f"{output_path}.checkpoint")
    fingerprint_path.write_text(str(model_path.resolve()), encoding="utf8")

    # Golden/calibration provenance for the thesis record.
    (output_dir / "golden_labels.json").write_text(json.dumps({
        "golden_vectors": int(os.environ["NN2RTL_GOLDEN_VECTORS"]),
        "golden_source": "CIFAR-10 test_batch images 0..G-1 (raw 0..255, NCHW)",
        "golden_labels": cifar10_util.golden_head_labels(),
        "calibration_samples": int(os.environ["NN2RTL_IMAGENET_CALIB"]),
        "calibration_source": (
            "feed rows G..N-1 = official MLPerf Tiny "
            "calibration_samples_idxs.npy order (indices into the TEST set, "
            "exactly as upstream model_converter.py uses them)"),
        "cifar_dir": str(cifar10_util.CIFAR_DIR),
        "calib_idx_path": str(cifar10_util.CALIB_IDX_PATH),
    }, indent=2) + "\n", encoding="utf8")

    summary = {
        "status": "ok",
        "model_name": payload["model_name"],
        "num_layers": len(payload["layers"]),
        "frontend": "onnx",
        "onnx_path": model_path.resolve().as_posix(),
        "pipeline_ir_path": output_path.resolve().as_posix(),
        "fingerprint_path": fingerprint_path.resolve().as_posix(),
        "weight_bits": os.environ["NN2RTL_WEIGHT_BITS"],
        "calib_samples": os.environ["NN2RTL_IMAGENET_CALIB"],
        "golden_vectors": os.environ["NN2RTL_GOLDEN_VECTORS"],
        "stem_per_channel": os.environ["NN2RTL_STEM_PER_CHANNEL"],
        "pw_per_channel": os.environ["NN2RTL_PW_PER_CHANNEL"],
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
