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
    weights_dir = repo_root / "output" / "weights"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Load the quantized ResNet-50 checkpoint from args.checkpoint_path into a model definition that matches the PTQ flow used in quantize_model.py.
    # TODO: Run a fixed 224x224 all-zero test image through the network and capture INT8 activation tensors after each residual block operation with torch.fx.
    # TODO: Fold batch normalization into the preceding convolution weights and bias before serialization so the emitted LayerIR uses only conv2d, relu, and add ops at runtime.
    # TODO: Convert the captured tensors into the PipelineIR schema expected by Cartographer, emitting weights_path and bias_path instead of inline weights, plus latency, clock, width, and signal-name metadata per module.
    # TODO: Serialize quantized weights and bias tensors to $readmemh-compatible hex files in output/weights/ using uppercase hex with one signed INT8 value per line.

    checkpoint_fingerprint = hashlib.sha256(str(args.checkpoint_path).encode("utf8")).hexdigest()
    placeholder_weights = weights_dir / "placeholder_weights.hex"
    placeholder_bias = weights_dir / "placeholder_bias.hex"
    placeholder_weights.write_text("00\n", encoding="utf8")
    placeholder_bias.write_text("00\n", encoding="utf8")

    placeholder = {
        "model_name": "resnet50",
        "quantization": "int8_symmetric_per_tensor",
        "generated_at": "TODO",
        "layers": [
            {
                "module_id": "stage2_block1_conv1",
                "op_type": "conv2d",
                "input_shape": [1, 64, 56, 56],
                "output_shape": [1, 64, 56, 56],
                "weights_path": str(placeholder_weights),
                "bias_path": str(placeholder_bias),
                "weight_shape": [64, 64, 1, 1],
                "num_weights": 4096,
                "scale_factor": 0.03125,
                "zero_point": 0,
                "pipeline_latency_cycles": 3,
                "clock_period_ns": 20.0,
                "input_width_bits": 8,
                "output_width_bits": 8,
                "valid_in_signal": "valid_in",
                "valid_out_signal": "valid_out",
                "clock_signal": "clk",
                "reset_signal": "rst_n",
                "golden_inputs": [[0, 0, 0, 0]],
                "golden_outputs": [[0, 0, 0, 0]],
            }
        ],
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "note": "TODO: replace this placeholder with real torch.fx-derived PipelineIR output and real output/weights/*.hex artifacts.",
    }

    output_path.write_text(json.dumps(placeholder, indent=2) + "\n", encoding="utf8")

    raise NotImplementedError(
        "generate_golden.py is scaffolded but not implemented; replace the TODO blocks with real torch.fx capture logic."
    )


if __name__ == "__main__":
    main()
