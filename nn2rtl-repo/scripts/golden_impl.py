"""Importable helpers for generate_golden.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch

from scripts.quantize_impl import create_toy_model, load_quantized_checkpoint, run_toy_model


def get_output_paths(repo_root: Path) -> tuple[Path, Path]:
    output_path = repo_root / "output" / "golden_vectors.json"
    weights_dir = repo_root / "output" / "weights"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    return output_path, weights_dir


def get_weight_artifact_paths(repo_root: Path, module_id: str) -> tuple[Path, Path]:
    _, weights_dir = get_output_paths(repo_root)
    return (
        weights_dir / f"{module_id}_weights.hex",
        weights_dir / f"{module_id}_bias.hex",
    )


def int8_to_hex(value: int) -> str:
    if value < -128 or value > 127:
        raise ValueError(f"INT8 value out of range: {value}")
    return f"{value & 0xFF:02X}"


def write_signed_int8_hex(values: Iterable[int], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    # $readmemh accepts LF everywhere, and the bench tests pin the on-disk
    # bytes. newline="" disables Python's platform-specific CRLF translation
    # on Windows.
    file_path.write_text(
        "".join(f"{int8_to_hex(int(value))}\n" for value in values),
        encoding="utf8",
        newline="",
    )


def fold_batch_norm_into_conv(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bias is None:
        bias = torch.zeros_like(running_mean)

    scale = bn_weight / torch.sqrt(running_var + eps)
    folded_weight = weight * scale.reshape(-1, 1, 1, 1)
    folded_bias = (bias - running_mean) * scale + bn_bias
    return folded_weight, folded_bias


def tensor_to_int8_list(tensor: torch.Tensor) -> list[int]:
    flattened = tensor.reshape(-1).to(torch.float32)
    return [int(value) for value in torch.clamp(flattened.round(), -128, 127).tolist()]


def build_pipeline_ir_payload(checkpoint_path: Path, repo_root: Path) -> dict[str, Any]:
    checkpoint = load_quantized_checkpoint(checkpoint_path)
    module_id = checkpoint["module_id"]

    weights_path, bias_path = get_weight_artifact_paths(repo_root, module_id)
    model = create_toy_model(checkpoint)

    input_stream = [int(value) for value in checkpoint["golden_input_stream"]]
    output_stream = run_toy_model(model, input_stream)

    conv_weight = torch.tensor(checkpoint["conv_weight"], dtype=torch.float32).reshape(
        checkpoint["weight_shape"]
    )
    conv_bias = torch.tensor(checkpoint["conv_bias"], dtype=torch.float32)
    batch_norm = checkpoint["batch_norm"]
    folded_weight, folded_bias = fold_batch_norm_into_conv(
        conv_weight,
        conv_bias,
        torch.tensor(batch_norm["weight"], dtype=torch.float32),
        torch.tensor(batch_norm["bias"], dtype=torch.float32),
        torch.tensor(batch_norm["running_mean"], dtype=torch.float32),
        torch.tensor(batch_norm["running_var"], dtype=torch.float32),
        float(batch_norm["eps"]),
    )

    write_signed_int8_hex(tensor_to_int8_list(folded_weight), weights_path)
    write_signed_int8_hex(tensor_to_int8_list(folded_bias), bias_path)

    stream_shape = [1, 1, 1, len(input_stream)]
    return {
        "model_name": checkpoint["model_name"],
        "quantization": checkpoint["quantization"],
        "generated_at": checkpoint["generated_at"],
        "layers": [
            {
                "module_id": module_id,
                "op_type": "conv2d",
                "input_shape": stream_shape,
                "output_shape": stream_shape,
                "weights_path": str(weights_path.resolve()),
                "bias_path": str(bias_path.resolve()),
                "weight_shape": checkpoint["weight_shape"],
                "num_weights": 1,
                "scale_factor": checkpoint["scale_factor"],
                "zero_point": checkpoint["zero_point"],
                "pipeline_latency_cycles": checkpoint["pipeline_latency_cycles"],
                "clock_period_ns": checkpoint["clock_period_ns"],
                "input_width_bits": checkpoint["input_width_bits"],
                "output_width_bits": checkpoint["output_width_bits"],
                "clock_signal": "clk",
                "reset_signal": "rst_n",
                "valid_in_signal": "valid_in",
                "valid_out_signal": "valid_out",
                "ready_in_signal": "ready_in",
                "data_in_signal": "data_in",
                "data_out_signal": "data_out",
                "golden_inputs": [input_stream],
                "golden_outputs": [output_stream],
            }
        ],
    }


def write_pipeline_ir(repo_root: Path, checkpoint_path: Path) -> Path:
    output_path, _ = get_output_paths(repo_root)
    payload = build_pipeline_ir_payload(checkpoint_path, repo_root)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf8")
    return output_path
