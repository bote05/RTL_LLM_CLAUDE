"""Importable helpers for quantize_model.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch


DEFAULT_CHECKPOINT_NAME = "resnet50_int8.pth"
DEFAULT_MODEL_NAME = "toy_stream_net"
DEFAULT_MODULE_ID = "toy_conv1x1"
DEFAULT_QUANTIZATION = "int8_symmetric_per_tensor"
DEFAULT_GENERATED_AT = "2026-04-14T00:00:00Z"
DEFAULT_INPUT_STREAM = [0, 1, 2, 7]


class CheckpointValidationError(ValueError):
    """Raised when a quantized checkpoint does not match the expected toy format."""


class ToyPointwiseModel(torch.nn.Module):
    """Small deterministic int8-friendly model used by the automated test flow."""

    def __init__(self, weight: int, bias: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.tensor(weight, dtype=torch.int32))
        self.register_buffer("bias", torch.tensor(bias, dtype=torch.int32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.to(torch.int32) * self.weight + self.bias
        return torch.clamp(y, -128, 127).to(torch.int32)


def resolve_checkpoint_path(repo_root: Path, checkpoint_path: str | Path | None = None) -> Path:
    if checkpoint_path is None:
        return repo_root / "checkpoints" / DEFAULT_CHECKPOINT_NAME

    candidate = Path(checkpoint_path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def get_quantized_checkpoint_path(repo_root: Path) -> Path:
    return resolve_checkpoint_path(repo_root)


def build_toy_quantized_checkpoint(
    checkpoint_path: Path,
    quantization_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model_name": DEFAULT_MODEL_NAME,
        "module_id": DEFAULT_MODULE_ID,
        "quantization": DEFAULT_QUANTIZATION,
        "generated_at": DEFAULT_GENERATED_AT,
        "checkpoint_path": str(checkpoint_path),
        "conv_weight": [2],
        "conv_bias": [1],
        "weight_shape": [1, 1, 1, 1],
        "input_width_bits": 8,
        "output_width_bits": 8,
        "pipeline_latency_cycles": 1,
        "clock_period_ns": 20.0,
        "scale_factor": 0.125,
        "zero_point": 0,
        "golden_input_stream": list(DEFAULT_INPUT_STREAM),
        "batch_norm": {
            "weight": [1.0],
            "bias": [0.0],
            "running_mean": [0.0],
            "running_var": [1.0],
            "eps": 1e-5,
        },
        "quantization_config": dict(quantization_config or {}),
    }


def write_quantized_checkpoint(checkpoint_path: Path, payload: Mapping[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), checkpoint_path)


def _require_int_list(payload: Mapping[str, Any], field: str, expected_len: int) -> list[int]:
    values = payload.get(field)
    if not isinstance(values, list) or len(values) != expected_len or not all(
        isinstance(value, int) for value in values
    ):
        raise CheckpointValidationError(
            f"Checkpoint field '{field}' must be a list of {expected_len} integers."
        )
    return [int(value) for value in values]


def load_quantized_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise CheckpointValidationError("Checkpoint payload must deserialize to a dict.")

    required_strings = {
        "model_name": DEFAULT_MODEL_NAME,
        "module_id": DEFAULT_MODULE_ID,
        "quantization": DEFAULT_QUANTIZATION,
        "generated_at": DEFAULT_GENERATED_AT,
    }
    for field, expected in required_strings.items():
        value = payload.get(field)
        if value != expected:
            raise CheckpointValidationError(
                f"Checkpoint field '{field}' must be '{expected}', got '{value}'."
            )

    required_numbers = {
        "format_version": 1,
        "input_width_bits": 8,
        "output_width_bits": 8,
        "pipeline_latency_cycles": 1,
        "zero_point": 0,
    }
    for field, expected in required_numbers.items():
        value = payload.get(field)
        if value != expected:
            raise CheckpointValidationError(
                f"Checkpoint field '{field}' must be {expected}, got {value}."
            )

    scale_factor = payload.get("scale_factor")
    if not isinstance(scale_factor, (int, float)):
        raise CheckpointValidationError("Checkpoint field 'scale_factor' must be numeric.")

    clock_period_ns = payload.get("clock_period_ns")
    if not isinstance(clock_period_ns, (int, float)):
        raise CheckpointValidationError("Checkpoint field 'clock_period_ns' must be numeric.")

    weight_shape = payload.get("weight_shape")
    if weight_shape != [1, 1, 1, 1]:
        raise CheckpointValidationError(
            f"Checkpoint field 'weight_shape' must be [1, 1, 1, 1], got {weight_shape}."
        )

    golden_input_stream = payload.get("golden_input_stream")
    if not isinstance(golden_input_stream, list) or not golden_input_stream:
        raise CheckpointValidationError(
            "Checkpoint field 'golden_input_stream' must be a non-empty list."
        )
    if not all(isinstance(value, int) for value in golden_input_stream):
        raise CheckpointValidationError(
            "Checkpoint field 'golden_input_stream' must contain integers only."
        )

    batch_norm = payload.get("batch_norm")
    if not isinstance(batch_norm, dict):
        raise CheckpointValidationError("Checkpoint field 'batch_norm' must be a dict.")
    for field in ("weight", "bias", "running_mean", "running_var"):
        values = batch_norm.get(field)
        if not isinstance(values, list) or len(values) != 1 or not all(
            isinstance(value, (int, float)) for value in values
        ):
            raise CheckpointValidationError(
                f"Checkpoint batch_norm field '{field}' must be a single-value list."
            )
    if not isinstance(batch_norm.get("eps"), (int, float)):
        raise CheckpointValidationError("Checkpoint batch_norm field 'eps' must be numeric.")

    _require_int_list(payload, "conv_weight", 1)
    _require_int_list(payload, "conv_bias", 1)

    quantization_config = payload.get("quantization_config")
    if quantization_config is None:
        payload["quantization_config"] = {}
    elif not isinstance(quantization_config, dict):
        raise CheckpointValidationError(
            "Checkpoint field 'quantization_config' must be a dict when present."
        )

    return dict(payload)


def create_toy_model(payload: Mapping[str, Any]) -> ToyPointwiseModel:
    weight = _require_int_list(payload, "conv_weight", 1)[0]
    bias = _require_int_list(payload, "conv_bias", 1)[0]
    return ToyPointwiseModel(weight=weight, bias=bias)


def run_toy_model(model: ToyPointwiseModel, input_stream: list[int]) -> list[int]:
    inputs = torch.tensor(input_stream, dtype=torch.int32)
    with torch.no_grad():
        outputs = model(inputs)
    return [int(value) for value in outputs.tolist()]


def build_quantization_summary(
    checkpoint_path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_name": payload["model_name"],
        "quantization": payload["quantization"],
        "checkpoint_path": str(checkpoint_path),
        "layers": {
            payload["module_id"]: {
                "scale_factor": payload["scale_factor"],
                "zero_point": payload["zero_point"],
                "weight_shape": payload["weight_shape"],
            }
        },
    }
