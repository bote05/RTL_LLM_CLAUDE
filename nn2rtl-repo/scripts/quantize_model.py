#!/usr/bin/env python3
"""Quantize a torchvision ResNet-50 checkpoint for the nn2rtl pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torchvision


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    checkpoints_dir = repo_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / "resnet50_int8.pth"

    # TODO: Load the canonical torchvision ResNet-50 weights and prepare the model for post-training static quantization with symmetric per-tensor INT8 settings.
    # TODO: Run the required calibration flow, convert the model with torch.quantization, and persist the quantized checkpoint to checkpoints/resnet50_int8.pth.
    # TODO: Collect the final per-layer scale factors and print them to stdout as machine-readable JSON so the rest of the pipeline can reuse the same calibration metadata.

    placeholder_scales = {
        "model_name": "resnet50",
        "quantization": "int8_symmetric_per_tensor",
        "checkpoint_path": str(checkpoint_path),
        "layers": {},
        "note": "TODO: replace this placeholder with real PTQ output.",
    }

    checkpoint_path.write_text(
        json.dumps({"note": "TODO: replace with quantized model state_dict."}, indent=2) + "\n",
        encoding="utf8",
    )
    print(json.dumps(placeholder_scales, indent=2))

    raise NotImplementedError(
        "quantize_model.py is scaffolded but not implemented; replace the TODO blocks with a real PTQ pipeline."
    )


if __name__ == "__main__":
    main()
