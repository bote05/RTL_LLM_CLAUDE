#!/usr/bin/env python3
"""Generate golden vectors for the nn2rtl pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torchvision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate output/golden_vectors.json from a quantized ResNet-50 checkpoint.",
    )
    parser.add_argument(
        "checkpoint_path",
        nargs="?",
        default="checkpoints/resnet50_int8.pth",
        help="Path to the quantized checkpoint file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    output_path = repo_root / "output" / "golden_vectors.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # TODO: Load the quantized ResNet-50 checkpoint from args.checkpoint_path into a model definition that matches the PTQ flow used in quantize_model.py.
    # TODO: Run a fixed 224x224 all-zero test image through the network and capture INT8 activation tensors after each residual block operation with torch.fx.
    # TODO: Convert the captured tensors into the PipelineIR schema expected by Cartographer, preserving per-layer scale factors, flattened golden inputs, and flattened golden outputs.

    checkpoint_fingerprint = hashlib.sha256(str(args.checkpoint_path).encode("utf8")).hexdigest()

    placeholder = {
      "model_name": "resnet50",
      "quantization": "int8_symmetric_per_tensor",
      "generated_at": "TODO",
      "layers": [],
      "checkpoint_fingerprint": checkpoint_fingerprint,
      "note": "TODO: replace this placeholder with real torch.fx-derived PipelineIR output.",
    }

    output_path.write_text(json.dumps(placeholder, indent=2) + "\n", encoding="utf8")

    raise NotImplementedError(
        "generate_golden.py is scaffolded but not implemented; replace the TODO blocks with real torch.fx capture logic."
    )


if __name__ == "__main__":
    main()
