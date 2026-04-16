#!/usr/bin/env python3
"""Generate deterministic golden vectors for the automated nn2rtl test flow.

Accepts two input formats:
  - A .onnx file  → uses the universal ONNX frontend (onnx_frontend.py)
  - A .pth file   → uses the legacy PyTorch checkpoint path (golden_impl.py)

Usage:
  python scripts/generate_golden.py                              # default .pth checkpoint
  python scripts/generate_golden.py path/to/model.pth           # explicit .pth checkpoint
  python scripts/generate_golden.py path/to/model.onnx          # ONNX model
  python scripts/generate_golden.py path/to/model.onnx --name mymodel
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from scripts.golden_impl import summarize_pipeline_ir, write_pipeline_ir
from scripts.paths import detect_repo_root
from scripts.quantize_impl import resolve_checkpoint_path

LAYER_IR_FILE_NAME = "layer_ir.json"
LEGACY_GOLDEN_FILE_NAME = "golden_vectors.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate output/layer_ir.json from a quantised checkpoint (.pth) "
            "or a pre-exported ONNX model (.onnx)."
        ),
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        default=None,
        help=(
            "Path to the model file.  A .pth extension is routed through the "
            "legacy PyTorch checkpoint path; a .onnx extension uses the "
            "universal ONNX frontend.  Relative paths are resolved against the "
            "repository root.  Defaults to checkpoints/resnet50_int8.pth."
        ),
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Logical model name recorded in layer_ir.json (ONNX path only; "
             "ignored for .pth checkpoints).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=8,
        help="Number of synthetic calibration samples (ONNX path only; default 8).",
    )
    parser.add_argument(
        "--faithful-conv",
        dest="faithful_conv",
        action="store_true",
        help=(
            "Generate goldens that match a real 2D convolution instead of the "
            "current RTL's spatially-summed 1x1 approximation (ONNX path only). "
            "Only works against an RTL datapath that implements full KH x KW "
            "receptive fields — the default single-MAC RTL will fail verification "
            "on non-1x1 kernels."
        ),
    )
    return parser.parse_args()


def _write_layer_ir_json(repo_root: Path, payload: dict) -> Path:
    output_dir = repo_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_ir_path = output_dir / LAYER_IR_FILE_NAME
    legacy_path = output_dir / LEGACY_GOLDEN_FILE_NAME
    encoded = json.dumps(payload, indent=2) + "\n"
    layer_ir_path.write_text(encoded, encoding="utf8", newline="")
    legacy_path.write_text(encoded, encoding="utf8", newline="")
    return layer_ir_path


def main() -> None:
    torch.manual_seed(0)
    args = parse_args()
    repo_root = detect_repo_root(__file__)

    model_path_str: str | None = args.model_path
    if model_path_str is None:
        # Default: legacy .pth checkpoint
        checkpoint_path = resolve_checkpoint_path(repo_root, None)
        output_path = write_pipeline_ir(repo_root, checkpoint_path)
        payload = json.loads(output_path.read_text(encoding="utf8"))
        print(json.dumps(summarize_pipeline_ir(payload, checkpoint_path, output_path)))
        return

    model_path = Path(model_path_str)
    if not model_path.is_absolute():
        model_path = repo_root / model_path

    suffix = model_path.suffix.lower()

    if suffix == ".onnx":
        from scripts.onnx_frontend import build_pipeline_ir_from_onnx

        payload = build_pipeline_ir_from_onnx(
            onnx_path=model_path,
            repo_root=repo_root,
            model_name=args.name,
            num_calibration_samples=args.samples,
            rtl_compat_conv=not args.faithful_conv,
        )
        output_path = _write_layer_ir_json(repo_root, payload)
        summary = {
            "status": "ok",
            "model_name": payload["model_name"],
            "num_layers": len(payload["layers"]),
            "frontend": "onnx",
            "onnx_path": model_path.resolve().as_posix(),
            "pipeline_ir_path": output_path.resolve().as_posix(),
        }
        print(json.dumps(summary))
        return

    # Default: legacy .pth checkpoint path
    checkpoint_path = resolve_checkpoint_path(repo_root, model_path_str)
    output_path = write_pipeline_ir(repo_root, checkpoint_path)
    payload = json.loads(output_path.read_text(encoding="utf8"))
    print(json.dumps(summarize_pipeline_ir(payload, checkpoint_path, output_path)))


if __name__ == "__main__":
    main()
