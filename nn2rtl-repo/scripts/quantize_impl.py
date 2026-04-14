"""Importable helpers for quantize_model.py."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import re
from pathlib import Path
from typing import Any, Callable, Iterable

import torch


DEFAULT_CHECKPOINT_NAME = "resnet50_int8.pth"
DEFAULT_MODEL_NAME = "resnet50"
DEFAULT_QUANTIZATION = "int8_symmetric_per_tensor"
DEFAULT_FORMAT_VERSION = 2
DEFAULT_GENERATED_AT = "2026-04-14T00:00:00Z"
DEFAULT_INPUT_WIDTH_BITS = 8
DEFAULT_OUTPUT_WIDTH_BITS = 8
CALIBRATION_SEED = 0
CALIBRATION_SAMPLE_COUNT = 32
CALIBRATION_INPUT_SHAPE = (1, 3, 224, 224)
LAYER1_EXPORT_NOTE = (
    "Checkpoint export is currently limited to the fused stem conv and layer1 bottlenecks. "
    "Calibration uses 32 synthetic tensors; swap in ImageNet samples for real PTQ."
)
MODULE_ID_PATTERN = re.compile(r"^layer\d+_\d+_(conv[123]|downsample|add|post_add_relu)$")

TOY_MODEL_NAME = "toy_stream_net"
TOY_MODULE_ID = "toy_conv1x1"
TOY_INPUT_STREAM = [0, 1, 2, 7]


class CheckpointValidationError(ValueError):
    """Raised when a quantized checkpoint does not match the expected format."""


@dataclass(frozen=True)
class LayerCalibrationStats:
    input_shape: list[int]
    output_shape: list[int]
    input_max_abs: float
    output_max_abs: float


class ToyPointwiseModel(torch.nn.Module):
    """Small deterministic int8-friendly model used by the legacy test flow."""

    def __init__(self, weight: int, bias: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.tensor(weight, dtype=torch.int32))
        self.register_buffer("bias", torch.tensor(bias, dtype=torch.int32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.to(torch.int32) * self.weight + self.bias
        return torch.clamp(y, -128, 127).to(torch.int32)


class FallbackBottleneck(torch.nn.Module):
    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int, *, use_downsample: bool) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = torch.nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = torch.nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = torch.nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=False)
        if use_downsample:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                torch.nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None


class FallbackResNet50(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(4)
        self.relu = torch.nn.ReLU(inplace=False)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = torch.nn.Sequential(
            FallbackBottleneck(4, 2, 4, use_downsample=True),
            FallbackBottleneck(4, 2, 4, use_downsample=False),
            FallbackBottleneck(4, 2, 4, use_downsample=False),
        )


def resolve_checkpoint_path(repo_root: Path, checkpoint_path: str | Path | None = None) -> Path:
    if checkpoint_path is None:
        return repo_root / "checkpoints" / DEFAULT_CHECKPOINT_NAME

    candidate = Path(checkpoint_path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def get_quantized_checkpoint_path(repo_root: Path) -> Path:
    return resolve_checkpoint_path(repo_root)


def write_quantized_checkpoint(checkpoint_path: Path, payload: Mapping[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), checkpoint_path)


def utc_now_iso8601() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def _load_torchvision_resnet50() -> torch.nn.Module:
    try:
        from torchvision.models import resnet50
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required to build the ResNet-50 PTQ checkpoint."
        ) from exc
    except Exception as exc:
        raise RuntimeError("Failed to import torchvision.models.resnet50.") from exc

    try:
        from torchvision.models import ResNet50_Weights
    except ImportError:
        ResNet50_Weights = None  # type: ignore[assignment]

    if ResNet50_Weights is not None:
        try:
            return resnet50(weights=ResNet50_Weights.DEFAULT)
        except Exception:
            pass

    try:
        return resnet50(weights=None)
    except (TypeError, ValueError):
        return resnet50(pretrained=False)


def _flatten_int_tensor(tensor: torch.Tensor, dtype: torch.dtype) -> list[int]:
    flattened = tensor.reshape(-1).to(dtype)
    return [int(value) for value in flattened.tolist()]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _as_shape_list(tensor: torch.Tensor) -> list[int]:
    return [int(dim) for dim in tensor.shape]


def _update_stats(
    stats: dict[str, LayerCalibrationStats],
    module_id: str,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
) -> None:
    input_shape = _as_shape_list(input_tensor)
    output_shape = _as_shape_list(output_tensor)
    input_max_abs = float(input_tensor.detach().abs().max().item())
    output_max_abs = float(output_tensor.detach().abs().max().item())

    existing = stats.get(module_id)
    if existing is None:
        stats[module_id] = LayerCalibrationStats(
            input_shape=input_shape,
            output_shape=output_shape,
            input_max_abs=input_max_abs,
            output_max_abs=output_max_abs,
        )
        return

    if existing.input_shape != input_shape:
        raise CheckpointValidationError(
            f"Layer '{module_id}' saw inconsistent input shapes: {existing.input_shape} vs {input_shape}."
        )
    if existing.output_shape != output_shape:
        raise CheckpointValidationError(
            f"Layer '{module_id}' saw inconsistent output shapes: {existing.output_shape} vs {output_shape}."
        )
    stats[module_id] = LayerCalibrationStats(
        input_shape=existing.input_shape,
        output_shape=existing.output_shape,
        input_max_abs=max(existing.input_max_abs, input_max_abs),
        output_max_abs=max(existing.output_max_abs, output_max_abs),
    )


def _make_calibration_inputs() -> list[torch.Tensor]:
    generator = torch.Generator()
    generator.manual_seed(CALIBRATION_SEED)
    return [
        torch.randn(CALIBRATION_INPUT_SHAPE, generator=generator, dtype=torch.float32)
        for _ in range(CALIBRATION_SAMPLE_COUNT)
    ]


def _collect_layer1_stats(
    model: torch.nn.Module,
    inputs: Iterable[torch.Tensor],
) -> dict[str, LayerCalibrationStats]:
    stats: dict[str, LayerCalibrationStats] = {}
    layer1_blocks = list(model.layer1)
    if len(layer1_blocks) != 3:
        raise CheckpointValidationError(
            f"Expected torchvision ResNet-50 layer1 to contain 3 bottlenecks, found {len(layer1_blocks)}."
        )

    with torch.no_grad():
        for sample in inputs:
            x = sample
            stem_input = x
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            _update_stats(stats, "layer0_0_conv1", stem_input, x)

            for block_index, block in enumerate(layer1_blocks):
                conv1_input = x
                identity = x
                if block.downsample is not None:
                    identity = block.downsample(x)
                    _update_stats(
                        stats,
                        f"layer1_{block_index}_downsample",
                        conv1_input,
                        identity,
                    )

                out = block.conv1(x)
                out = block.bn1(out)
                out = block.relu(out)
                _update_stats(stats, f"layer1_{block_index}_conv1", conv1_input, out)

                conv2_input = out
                out = block.conv2(out)
                out = block.bn2(out)
                out = block.relu(out)
                _update_stats(stats, f"layer1_{block_index}_conv2", conv2_input, out)

                conv3_input = out
                out = block.conv3(out)
                out = block.bn3(out)
                _update_stats(stats, f"layer1_{block_index}_conv3", conv3_input, out)

                add_output = out + identity
                _update_stats(stats, f"layer1_{block_index}_add", out, add_output)

                relu_input = add_output
                x = block.relu(add_output)
                _update_stats(stats, f"layer1_{block_index}_post_add_relu", relu_input, x)

    return stats


def _safe_scale(max_abs: float) -> float:
    return max_abs / 127.0 if max_abs > 0.0 else 1.0


def _quantize_weight_tensor(weight: torch.Tensor) -> tuple[list[int], float]:
    max_abs = float(weight.detach().abs().max().item())
    scale = _safe_scale(max_abs)
    quantized = torch.clamp(torch.round(weight / scale), -128, 127).to(torch.int8)
    return _flatten_int_tensor(quantized, torch.int8), scale


def _quantize_bias_tensor(
    bias: torch.Tensor | None,
    input_scale: float,
    weight_scale: float,
) -> list[int] | None:
    if bias is None:
        return None
    scale = input_scale * weight_scale
    if scale == 0.0:
        scale = 1.0
    quantized = torch.round(bias / scale).to(torch.int32)
    return _flatten_int_tensor(quantized, torch.int32)


def _serialize_conv_layer(
    stats: LayerCalibrationStats,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> dict[str, Any]:
    weight_int8, weight_scale = _quantize_weight_tensor(weight)
    input_scale = _safe_scale(stats.input_max_abs)

    return {
        "op_type": "conv2d",
        "input_shape": list(stats.input_shape),
        "output_shape": list(stats.output_shape),
        "weight_int8": weight_int8,
        "bias_int32": _quantize_bias_tensor(bias, input_scale, weight_scale),
        "weight_shape": [int(dim) for dim in weight.shape],
        "num_weights": int(weight.numel()),
        "scale_factor": float(weight_scale),
        "zero_point": 0,
        "input_width_bits": DEFAULT_INPUT_WIDTH_BITS,
        "output_width_bits": DEFAULT_OUTPUT_WIDTH_BITS,
    }


def _serialize_activation_layer(
    op_type: str,
    stats: LayerCalibrationStats,
    *,
    scale_factor: float,
    lhs_scale_factor: float | None = None,
    rhs_scale_factor: float | None = None,
) -> dict[str, Any]:
    if op_type not in {"add", "relu"}:
        raise ValueError(f"Unsupported activation op_type: {op_type}")

    payload = {
        "op_type": op_type,
        "input_shape": list(stats.input_shape),
        "output_shape": list(stats.output_shape),
        "weight_shape": [1],
        "num_weights": 0,
        "scale_factor": float(scale_factor),
        "zero_point": 0,
        "input_width_bits": (
            DEFAULT_INPUT_WIDTH_BITS * 2 if op_type == "add" else DEFAULT_INPUT_WIDTH_BITS
        ),
        "output_width_bits": DEFAULT_OUTPUT_WIDTH_BITS,
    }
    if op_type == "add":
        if lhs_scale_factor is None or rhs_scale_factor is None:
            raise ValueError("Add layers require lhs_scale_factor and rhs_scale_factor.")
        payload["lhs_scale_factor"] = float(lhs_scale_factor)
        payload["rhs_scale_factor"] = float(rhs_scale_factor)
    return payload


def _fold_stem_conv(model: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    return fold_batch_norm_into_conv(
        model.conv1.weight.detach().to(torch.float32),
        None if model.conv1.bias is None else model.conv1.bias.detach().to(torch.float32),
        model.bn1.weight.detach().to(torch.float32),
        model.bn1.bias.detach().to(torch.float32),
        model.bn1.running_mean.detach().to(torch.float32),
        model.bn1.running_var.detach().to(torch.float32),
        float(model.bn1.eps),
    )


def _fold_block_conv(block: torch.nn.Module, conv_name: str, bn_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    conv = getattr(block, conv_name)
    bn = getattr(block, bn_name)
    return fold_batch_norm_into_conv(
        conv.weight.detach().to(torch.float32),
        None if conv.bias is None else conv.bias.detach().to(torch.float32),
        bn.weight.detach().to(torch.float32),
        bn.bias.detach().to(torch.float32),
        bn.running_mean.detach().to(torch.float32),
        bn.running_var.detach().to(torch.float32),
        float(bn.eps),
    )


def _fold_downsample_conv(downsample: torch.nn.Sequential) -> tuple[torch.Tensor, torch.Tensor]:
    if len(downsample) < 2:
        raise CheckpointValidationError(
            "Expected downsample sequential to contain Conv2d + BatchNorm2d."
        )
    conv = downsample[0]
    bn = downsample[1]
    if not isinstance(conv, torch.nn.Conv2d) or not isinstance(bn, torch.nn.BatchNorm2d):
        raise CheckpointValidationError(
            "Expected downsample sequential to contain Conv2d + BatchNorm2d."
        )
    return fold_batch_norm_into_conv(
        conv.weight.detach().to(torch.float32),
        None if conv.bias is None else conv.bias.detach().to(torch.float32),
        bn.weight.detach().to(torch.float32),
        bn.bias.detach().to(torch.float32),
        bn.running_mean.detach().to(torch.float32),
        bn.running_var.detach().to(torch.float32),
        float(bn.eps),
    )


def _conv_operation(
    module_id: str,
    input_ref: str,
    conv: torch.nn.Conv2d,
) -> dict[str, Any]:
    return {
        "module_id": module_id,
        "op_type": "conv2d",
        "input": input_ref,
        "stride": [int(value) for value in conv.stride],
        "padding": [int(value) for value in conv.padding],
        "dilation": [int(value) for value in conv.dilation],
        "groups": int(conv.groups),
    }


def _build_residual_stack_spec(model: torch.nn.Module) -> dict[str, Any]:
    operations: list[dict[str, Any]] = [
        _conv_operation("layer0_0_conv1", "input", model.conv1),
    ]
    previous_output = "layer0_0_conv1"

    for block_index, block in enumerate(model.layer1):
        conv1_id = f"layer1_{block_index}_conv1"
        conv2_id = f"layer1_{block_index}_conv2"
        conv3_id = f"layer1_{block_index}_conv3"
        add_id = f"layer1_{block_index}_add"
        relu_id = f"layer1_{block_index}_post_add_relu"

        operations.append(_conv_operation(conv1_id, previous_output, block.conv1))
        operations.append(_conv_operation(conv2_id, conv1_id, block.conv2))
        operations.append(_conv_operation(conv3_id, conv2_id, block.conv3))

        if block_index == 0:
            if block.downsample is None:
                raise CheckpointValidationError("layer1[0] must expose a downsample path.")
            downsample = block.downsample[0]
            if not isinstance(downsample, torch.nn.Conv2d):
                raise CheckpointValidationError("layer1[0].downsample[0] must be a Conv2d.")
            downsample_id = "layer1_0_downsample"
            operations.append(_conv_operation(downsample_id, previous_output, downsample))
            rhs_ref = downsample_id
        else:
            rhs_ref = previous_output

        operations.append(
            {
                "module_id": add_id,
                "op_type": "add",
                "lhs": conv3_id,
                "rhs": rhs_ref,
            }
        )
        operations.append(
            {
                "module_id": relu_id,
                "op_type": "relu",
                "input": add_id,
            }
        )
        previous_output = relu_id

    return {
        "input_name": "input",
        "output_module_id": previous_output,
        "operations": operations,
    }


def build_resnet50_quantized_checkpoint(
    checkpoint_path: Path,
    *,
    model_loader: Callable[[], torch.nn.Module] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    torch.manual_seed(CALIBRATION_SEED)
    model = (model_loader or _load_torchvision_resnet50)()
    model.eval()

    calibration_inputs = _make_calibration_inputs()
    stats = _collect_layer1_stats(model, calibration_inputs)

    layers: dict[str, dict[str, Any]] = {}

    stem_weight, stem_bias = _fold_stem_conv(model)
    layers["layer0_0_conv1"] = _serialize_conv_layer(
        stats["layer0_0_conv1"],
        stem_weight,
        stem_bias,
    )

    for block_index, block in enumerate(model.layer1):
        for conv_name, bn_name in (("conv1", "bn1"), ("conv2", "bn2"), ("conv3", "bn3")):
            module_id = f"layer1_{block_index}_{conv_name}"
            weight, bias = _fold_block_conv(block, conv_name, bn_name)
            layers[module_id] = _serialize_conv_layer(stats[module_id], weight, bias)

        if block_index == 0:
            if block.downsample is None:
                raise CheckpointValidationError("layer1[0] must expose a downsample path.")
            downsample_weight, downsample_bias = _fold_downsample_conv(block.downsample)
            layers["layer1_0_downsample"] = _serialize_conv_layer(
                stats["layer1_0_downsample"],
                downsample_weight,
                downsample_bias,
            )

        add_id = f"layer1_{block_index}_add"
        add_scale = _safe_scale(stats[add_id].output_max_abs)
        rhs_scale_factor = (
            layers["layer1_0_downsample"]["scale_factor"]
            if block_index == 0
            else layers[f"layer1_{block_index - 1}_post_add_relu"]["scale_factor"]
        )
        layers[add_id] = _serialize_activation_layer(
            "add",
            stats[add_id],
            scale_factor=add_scale,
            lhs_scale_factor=layers[f"layer1_{block_index}_conv3"]["scale_factor"],
            rhs_scale_factor=rhs_scale_factor,
        )

        relu_id = f"layer1_{block_index}_post_add_relu"
        layers[relu_id] = _serialize_activation_layer(
            "relu",
            stats[relu_id],
            scale_factor=add_scale,
        )

    return {
        "format_version": DEFAULT_FORMAT_VERSION,
        "model_name": DEFAULT_MODEL_NAME,
        "quantization": DEFAULT_QUANTIZATION,
        "generated_at": generated_at or utc_now_iso8601(),
        "residual_stack_spec": _build_residual_stack_spec(model),
        "layers": layers,
    }


def build_toy_quantized_checkpoint(
    checkpoint_path: Path,
    quantization_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model_name": TOY_MODEL_NAME,
        "module_id": TOY_MODULE_ID,
        "quantization": DEFAULT_QUANTIZATION,
        "generated_at": DEFAULT_GENERATED_AT,
        "checkpoint_path": str(checkpoint_path),
        "conv_weight": [2],
        "conv_bias": [1],
        "weight_shape": [1, 1, 1, 1],
        "input_width_bits": DEFAULT_INPUT_WIDTH_BITS,
        "output_width_bits": DEFAULT_OUTPUT_WIDTH_BITS,
        "pipeline_latency_cycles": 1,
        "clock_period_ns": 20.0,
        "scale_factor": 0.125,
        "zero_point": 0,
        "golden_input_stream": list(TOY_INPUT_STREAM),
        "batch_norm": {
            "weight": [1.0],
            "bias": [0.0],
            "running_mean": [0.0],
            "running_var": [1.0],
            "eps": 1e-5,
        },
        "quantization_config": dict(quantization_config or {}),
    }


def _require_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise CheckpointValidationError(f"Checkpoint field '{field}' must be a non-empty string.")
    return value


def _require_int(payload: Mapping[str, Any], field: str, expected: int | None = None) -> int:
    value = payload.get(field)
    if not _is_int(value):
        raise CheckpointValidationError(f"Checkpoint field '{field}' must be an integer.")
    value = int(value)
    if expected is not None and value != expected:
        raise CheckpointValidationError(
            f"Checkpoint field '{field}' must be {expected}, got {value}."
        )
    return value


def _require_number(payload: Mapping[str, Any], field: str) -> float:
    value = payload.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CheckpointValidationError(f"Checkpoint field '{field}' must be numeric.")
    return float(value)


def _require_int_list(
    payload: Mapping[str, Any],
    field: str,
    *,
    expected_len: int | None = None,
    min_value: int | None = None,
    max_value: int | None = None,
) -> list[int]:
    values = payload.get(field)
    if not isinstance(values, list):
        raise CheckpointValidationError(f"Checkpoint field '{field}' must be a list of integers.")
    if expected_len is not None and len(values) != expected_len:
        raise CheckpointValidationError(
            f"Checkpoint field '{field}' must contain {expected_len} integers."
        )

    result: list[int] = []
    for value in values:
        if not _is_int(value):
            raise CheckpointValidationError(
                f"Checkpoint field '{field}' must contain integers only."
            )
        int_value = int(value)
        if min_value is not None and int_value < min_value:
            raise CheckpointValidationError(
                f"Checkpoint field '{field}' must be >= {min_value}, got {int_value}."
            )
        if max_value is not None and int_value > max_value:
            raise CheckpointValidationError(
                f"Checkpoint field '{field}' must be <= {max_value}, got {int_value}."
            )
        result.append(int_value)
    return result


def _require_shape(value: Any) -> list[int]:
    if not isinstance(value, list) or not value:
        raise CheckpointValidationError("Checkpoint field 'weight_shape' must be a non-empty list.")
    shape = []
    for dim in value:
        if not _is_int(dim) or int(dim) <= 0:
            raise CheckpointValidationError(
                "Checkpoint field 'weight_shape' must contain positive integers only."
            )
        shape.append(int(dim))
    return shape


def _validate_iso8601_utc(value: str) -> str:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CheckpointValidationError(
            "Checkpoint field 'generated_at' must be an ISO-8601 timestamp."
        ) from exc
    return value


def _validate_toy_checkpoint(payload: Mapping[str, Any]) -> dict[str, Any]:
    required_strings = {
        "model_name": TOY_MODEL_NAME,
        "module_id": TOY_MODULE_ID,
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
        "input_width_bits": DEFAULT_INPUT_WIDTH_BITS,
        "output_width_bits": DEFAULT_OUTPUT_WIDTH_BITS,
        "pipeline_latency_cycles": 1,
        "zero_point": 0,
    }
    for field, expected in required_numbers.items():
        value = payload.get(field)
        if value != expected:
            raise CheckpointValidationError(
                f"Checkpoint field '{field}' must be {expected}, got {value}."
            )

    _require_number(payload, "scale_factor")
    _require_number(payload, "clock_period_ns")

    if payload.get("weight_shape") != [1, 1, 1, 1]:
        raise CheckpointValidationError(
            f"Checkpoint field 'weight_shape' must be [1, 1, 1, 1], got {payload.get('weight_shape')}."
        )

    golden_input_stream = payload.get("golden_input_stream")
    if not isinstance(golden_input_stream, list) or not golden_input_stream:
        raise CheckpointValidationError(
            "Checkpoint field 'golden_input_stream' must be a non-empty list."
        )
    if not all(_is_int(value) for value in golden_input_stream):
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

    _require_int_list(payload, "conv_weight", expected_len=1)
    _require_int_list(payload, "conv_bias", expected_len=1)

    quantization_config = payload.get("quantization_config")
    if quantization_config is None:
        payload = dict(payload)
        payload["quantization_config"] = {}
    elif not isinstance(quantization_config, dict):
        raise CheckpointValidationError(
            "Checkpoint field 'quantization_config' must be a dict when present."
        )

    return dict(payload)


def _validate_layer_payload(module_id: str, layer: Any) -> dict[str, Any]:
    if not MODULE_ID_PATTERN.match(module_id):
        raise CheckpointValidationError(
            f"Checkpoint layer id '{module_id}' does not match the required layer{{stage}}_{{block}}_* scheme."
        )
    if not isinstance(layer, dict):
        raise CheckpointValidationError(f"Checkpoint layer '{module_id}' must be a dict.")

    op_type = layer.get("op_type")
    if op_type not in {"conv2d", "relu", "add"}:
        raise CheckpointValidationError(
            f"Checkpoint layer '{module_id}' has unsupported op_type '{op_type}'."
        )

    input_shape = layer.get("input_shape")
    output_shape = layer.get("output_shape")
    if not isinstance(input_shape, list) or not input_shape or not all(
        _is_int(dim) and int(dim) > 0 for dim in input_shape
    ):
        raise CheckpointValidationError(
            f"Checkpoint layer '{module_id}' field 'input_shape' must be a non-empty list of positive integers."
        )
    if not isinstance(output_shape, list) or not output_shape or not all(
        _is_int(dim) and int(dim) > 0 for dim in output_shape
    ):
        raise CheckpointValidationError(
            f"Checkpoint layer '{module_id}' field 'output_shape' must be a non-empty list of positive integers."
        )

    if _require_int(layer, "zero_point") != 0:
        raise CheckpointValidationError(
            f"Checkpoint layer '{module_id}' field 'zero_point' must be 0."
        )
    expected_input_width_bits = (
        DEFAULT_INPUT_WIDTH_BITS * 2 if op_type == "add" else DEFAULT_INPUT_WIDTH_BITS
    )
    _require_int(layer, "input_width_bits", expected=expected_input_width_bits)
    _require_int(layer, "output_width_bits", expected=DEFAULT_OUTPUT_WIDTH_BITS)
    scale_factor = _require_number(layer, "scale_factor")
    if scale_factor <= 0.0:
        raise CheckpointValidationError(
            f"Checkpoint layer '{module_id}' field 'scale_factor' must be positive."
        )
    weight_shape = _require_shape(layer.get("weight_shape"))
    num_weights = _require_int(layer, "num_weights")
    if num_weights < 0:
        raise CheckpointValidationError(
            f"Checkpoint layer '{module_id}' field 'num_weights' must be non-negative."
        )

    validated = {
        "op_type": op_type,
        "input_shape": [int(dim) for dim in input_shape],
        "output_shape": [int(dim) for dim in output_shape],
        "weight_shape": weight_shape,
        "num_weights": num_weights,
        "scale_factor": float(scale_factor),
        "zero_point": 0,
        "input_width_bits": expected_input_width_bits,
        "output_width_bits": DEFAULT_OUTPUT_WIDTH_BITS,
    }

    if op_type == "conv2d":
        weight_int8 = _require_int_list(
            layer,
            "weight_int8",
            min_value=-128,
            max_value=127,
        )
        expected_values = math.prod(weight_shape)
        if len(weight_int8) != expected_values:
            raise CheckpointValidationError(
                f"Checkpoint layer '{module_id}' has {len(weight_int8)} weights, expected {expected_values} from weight_shape."
            )
        if num_weights != expected_values:
            raise CheckpointValidationError(
                f"Checkpoint layer '{module_id}' num_weights must match weight_shape ({expected_values})."
            )
        bias_value = layer.get("bias_int32")
        if bias_value is None:
            bias_int32 = None
        else:
            bias_int32 = _require_int_list(layer, "bias_int32")
            if len(bias_int32) != weight_shape[0]:
                raise CheckpointValidationError(
                    f"Checkpoint layer '{module_id}' bias_int32 length must match output channels ({weight_shape[0]})."
                )
        validated["weight_int8"] = weight_int8
        validated["bias_int32"] = bias_int32
    else:
        if num_weights != 0:
            raise CheckpointValidationError(
                f"Checkpoint layer '{module_id}' field 'num_weights' must be 0 for non-conv2d layers."
            )
        for forbidden_field in ("weight_int8", "bias_int32"):
            if forbidden_field in layer:
                raise CheckpointValidationError(
                    f"Checkpoint layer '{module_id}' field '{forbidden_field}' is only valid for conv2d layers."
                )
        if op_type == "add":
            lhs_scale_factor = _require_number(layer, "lhs_scale_factor")
            rhs_scale_factor = _require_number(layer, "rhs_scale_factor")
            if lhs_scale_factor <= 0.0 or rhs_scale_factor <= 0.0:
                raise CheckpointValidationError(
                    f"Checkpoint layer '{module_id}' add scale factors must be positive."
                )
            validated["lhs_scale_factor"] = float(lhs_scale_factor)
            validated["rhs_scale_factor"] = float(rhs_scale_factor)

    return validated


def _expected_resnet50_module_ids() -> list[str]:
    module_ids = ["layer0_0_conv1"]
    for block_index in range(3):
        module_ids.extend(
            [
                f"layer1_{block_index}_conv1",
                f"layer1_{block_index}_conv2",
                f"layer1_{block_index}_conv3",
            ]
        )
        if block_index == 0:
            module_ids.append("layer1_0_downsample")
        module_ids.extend(
            [
                f"layer1_{block_index}_add",
                f"layer1_{block_index}_post_add_relu",
            ]
        )
    return module_ids


def _validate_residual_stack_spec(
    payload: Mapping[str, Any],
    valid_module_ids: set[str],
) -> dict[str, Any]:
    raw_spec = payload.get("residual_stack_spec")
    if not isinstance(raw_spec, Mapping):
        raise CheckpointValidationError(
            "Checkpoint field 'residual_stack_spec' must be a dict for format_version=2 checkpoints."
        )

    if _require_string(raw_spec, "input_name") != "input":
        raise CheckpointValidationError("Checkpoint residual_stack_spec.input_name must be 'input'.")

    output_module_id = _require_string(raw_spec, "output_module_id")
    if output_module_id != "layer1_2_post_add_relu":
        raise CheckpointValidationError(
            "Checkpoint residual_stack_spec.output_module_id must be 'layer1_2_post_add_relu'."
        )

    operations = raw_spec.get("operations")
    if not isinstance(operations, list) or not operations:
        raise CheckpointValidationError(
            "Checkpoint residual_stack_spec.operations must be a non-empty list."
        )

    expected_module_ids = _expected_resnet50_module_ids()
    validated_operations: list[dict[str, Any]] = []
    seen_module_ids: list[str] = []
    produced_refs = {"input"}

    for operation in operations:
        if not isinstance(operation, Mapping):
            raise CheckpointValidationError(
                "Checkpoint residual_stack_spec.operations entries must be dicts."
            )
        module_id = _require_string(operation, "module_id")
        op_type = _require_string(operation, "op_type")
        if module_id not in valid_module_ids:
            raise CheckpointValidationError(
                f"Checkpoint residual_stack_spec references unknown module_id '{module_id}'."
            )
        if op_type not in {"conv2d", "relu", "add"}:
            raise CheckpointValidationError(
                f"Checkpoint residual_stack_spec op_type must be one of conv2d/relu/add, got '{op_type}'."
            )

        validated_operation = {
            "module_id": module_id,
            "op_type": op_type,
        }
        if op_type == "add":
            lhs = _require_string(operation, "lhs")
            rhs = _require_string(operation, "rhs")
            if lhs not in produced_refs or rhs not in produced_refs:
                raise CheckpointValidationError(
                    f"Checkpoint residual_stack_spec add '{module_id}' must reference previously produced operands."
                )
            validated_operation["lhs"] = lhs
            validated_operation["rhs"] = rhs
        else:
            input_ref = _require_string(operation, "input")
            if input_ref not in produced_refs:
                raise CheckpointValidationError(
                    f"Checkpoint residual_stack_spec op '{module_id}' references unknown input '{input_ref}'."
                )
            validated_operation["input"] = input_ref
            if op_type == "conv2d":
                validated_operation["stride"] = _require_int_list(operation, "stride", expected_len=2)
                validated_operation["padding"] = _require_int_list(operation, "padding", expected_len=2)
                validated_operation["dilation"] = _require_int_list(operation, "dilation", expected_len=2)
                validated_operation["groups"] = _require_int(operation, "groups")
        produced_refs.add(module_id)
        seen_module_ids.append(module_id)
        validated_operations.append(validated_operation)

    if seen_module_ids != expected_module_ids:
        raise CheckpointValidationError(
            f"Checkpoint residual_stack_spec.operations must match the constrained stem+layer1 order. "
            f"Expected {expected_module_ids}, got {seen_module_ids}."
        )

    return {
        "input_name": "input",
        "output_module_id": output_module_id,
        "operations": validated_operations,
    }


def _validate_resnet50_checkpoint(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_int(payload, "format_version", expected=DEFAULT_FORMAT_VERSION)
    if _require_string(payload, "model_name") != DEFAULT_MODEL_NAME:
        raise CheckpointValidationError(
            f"Checkpoint field 'model_name' must be '{DEFAULT_MODEL_NAME}'."
        )
    if _require_string(payload, "quantization") != DEFAULT_QUANTIZATION:
        raise CheckpointValidationError(
            f"Checkpoint field 'quantization' must be '{DEFAULT_QUANTIZATION}'."
        )
    _validate_iso8601_utc(_require_string(payload, "generated_at"))

    layers = payload.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise CheckpointValidationError(
            "Checkpoint field 'layers' must be a non-empty dict keyed by module_id."
        )

    validated_layers: dict[str, dict[str, Any]] = {}
    for module_id, layer in layers.items():
        if not isinstance(module_id, str):
            raise CheckpointValidationError("Checkpoint layer ids must be strings.")
        validated_layers[module_id] = _validate_layer_payload(module_id, layer)

    expected_module_ids = set(_expected_resnet50_module_ids())

    missing = sorted(expected_module_ids - set(validated_layers))
    extra = sorted(set(validated_layers) - expected_module_ids)
    if missing or extra:
        raise CheckpointValidationError(
            f"Checkpoint layers must match the constrained stem+layer1 export set. Missing: {missing}; extra: {extra}."
        )

    residual_stack_spec = _validate_residual_stack_spec(payload, set(validated_layers))

    return {
        "format_version": DEFAULT_FORMAT_VERSION,
        "model_name": DEFAULT_MODEL_NAME,
        "quantization": DEFAULT_QUANTIZATION,
        "generated_at": payload["generated_at"],
        "residual_stack_spec": residual_stack_spec,
        "layers": validated_layers,
    }


def load_quantized_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise CheckpointValidationError("Checkpoint payload must deserialize to a dict.")

    format_version = payload.get("format_version")
    if format_version == DEFAULT_FORMAT_VERSION:
        return _validate_resnet50_checkpoint(payload)
    if format_version == 1:
        return _validate_toy_checkpoint(payload)
    raise CheckpointValidationError(
        f"Checkpoint field 'format_version' must be 1 or {DEFAULT_FORMAT_VERSION}, got {format_version}."
    )


def create_toy_model(payload: Mapping[str, Any]) -> ToyPointwiseModel:
    weight = _require_int_list(payload, "conv_weight", expected_len=1)[0]
    bias = _require_int_list(payload, "conv_bias", expected_len=1)[0]
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
    if payload.get("format_version") == 1:
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

    raw_layers = payload.get("layers", {})
    if not isinstance(raw_layers, Mapping):
        raise CheckpointValidationError("Checkpoint field 'layers' must be a dict.")

    layers: dict[str, dict[str, Any]] = {}
    for module_id, layer in raw_layers.items():
        if not isinstance(module_id, str) or not isinstance(layer, Mapping):
            raise CheckpointValidationError("Checkpoint summary expected string keyed layer dicts.")
        layer_summary = {
            "op_type": layer["op_type"],
            "scale_factor": layer["scale_factor"],
            "zero_point": layer["zero_point"],
        }
        if "weight_shape" in layer:
            layer_summary["weight_shape"] = layer["weight_shape"]
        layers[module_id] = layer_summary

    return {
        "model_name": payload["model_name"],
        "quantization": payload["quantization"],
        "checkpoint_path": str(checkpoint_path),
        "export_scope": "stem_plus_layer1",
        "notes": [LAYER1_EXPORT_NOTE],
        "layers": layers,
    }
