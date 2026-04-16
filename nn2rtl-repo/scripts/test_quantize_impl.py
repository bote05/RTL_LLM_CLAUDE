from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from scripts.quantize_impl import (
    CheckpointValidationError,
    build_quantization_summary,
    build_resnet50_quantized_checkpoint,
    build_toy_quantized_checkpoint,
    create_toy_model,
    get_quantized_checkpoint_path,
    load_quantized_checkpoint,
    resolve_checkpoint_path,
    run_toy_model,
    write_quantized_checkpoint,
)


class FakeBottleneck(torch.nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + identity
        out = self.relu(out)
        return out


class FakeResNet50(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(4)
        self.relu = torch.nn.ReLU(inplace=False)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = torch.nn.Sequential(
            FakeBottleneck(4, 2, 4, use_downsample=True),
            FakeBottleneck(4, 2, 4, use_downsample=False),
            FakeBottleneck(4, 2, 4, use_downsample=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return self.layer1(x)


def build_fake_resnet50() -> FakeResNet50:
    return FakeResNet50()


def expected_module_ids() -> list[str]:
    ids = ["layer0_0_conv1"]
    for block_index in range(3):
        ids.extend(
            [
                f"layer1_{block_index}_conv1",
                f"layer1_{block_index}_conv2",
                f"layer1_{block_index}_conv3",
            ]
        )
        if block_index == 0:
            ids.append("layer1_0_downsample")
        ids.extend(
            [
                f"layer1_{block_index}_add",
                f"layer1_{block_index}_post_add_relu",
            ]
        )
    return ids


def test_quantized_checkpoint_path_is_deterministic(tmp_path: Path) -> None:
    assert get_quantized_checkpoint_path(tmp_path) == tmp_path / "checkpoints" / "resnet50_int8.pth"


def test_resolve_checkpoint_path_supports_relative_and_absolute_paths(tmp_path: Path) -> None:
    absolute = tmp_path / "abs" / "resnet50.pth"
    assert resolve_checkpoint_path(tmp_path, "checkpoints/custom.pth") == tmp_path / "checkpoints" / "custom.pth"
    assert resolve_checkpoint_path(tmp_path, absolute) == absolute


def test_resnet50_checkpoint_round_trip_preserves_new_schema(tmp_path: Path) -> None:
    checkpoint_path = get_quantized_checkpoint_path(tmp_path)
    payload = build_resnet50_quantized_checkpoint(
        checkpoint_path,
        model_loader=build_fake_resnet50,
        generated_at="2026-04-14T12:00:00Z",
    )
    write_quantized_checkpoint(checkpoint_path, payload)

    loaded = load_quantized_checkpoint(checkpoint_path)
    summary = build_quantization_summary(checkpoint_path, loaded)

    assert loaded["format_version"] == 2
    assert loaded["model_name"] == "resnet50"
    assert loaded["quantization"] == "int8_symmetric_per_tensor"
    assert list(loaded["layers"]) == expected_module_ids()
    assert len(loaded["layers"]) == 17
    assert summary["export_scope"] == "stem_plus_layer1"
    assert "ImageNet samples" in summary["notes"][0]
    assert loaded["residual_stack_spec"]["input_name"] == "input"
    assert loaded["residual_stack_spec"]["output_module_id"] == "layer1_2_post_add_relu"
    assert [
        operation["module_id"] for operation in loaded["residual_stack_spec"]["operations"]
    ] == expected_module_ids()

    stem = loaded["layers"]["layer0_0_conv1"]
    assert stem["op_type"] == "conv2d"
    assert stem["input_shape"] == [1, 3, 224, 224]
    assert stem["output_shape"] == [1, 4, 112, 112]
    assert stem["zero_point"] == 0
    assert stem["input_width_bits"] == 24
    assert stem["output_width_bits"] == 32
    assert len(stem["weight_int8"]) == math.prod(stem["weight_shape"])
    assert len(stem["bias_int32"]) == stem["weight_shape"][0]
    assert stem["scale_factor"] > 0.0

    downsample = loaded["layers"]["layer1_0_downsample"]
    assert downsample["op_type"] == "conv2d"
    assert downsample["input_shape"] == [1, 4, 112, 112]
    assert downsample["output_shape"] == [1, 4, 112, 112]
    assert downsample["num_weights"] == math.prod(downsample["weight_shape"])

    add_layer = loaded["layers"]["layer1_0_add"]
    assert add_layer["op_type"] == "add"
    assert add_layer["input_shape"] == [1, 4, 112, 112]
    assert add_layer["output_shape"] == [1, 4, 112, 112]
    assert add_layer["weight_shape"] == [1]
    assert add_layer["num_weights"] == 0
    assert add_layer["scale_factor"] > 0.0
    assert add_layer["lhs_scale_factor"] > 0.0
    assert add_layer["rhs_scale_factor"] > 0.0
    assert add_layer["input_width_bits"] == 64
    assert add_layer["output_width_bits"] == 32
    assert summary["layers"]["layer1_0_conv1"]["weight_shape"] == [2, 4, 1, 1]
    assert summary["layers"]["layer1_0_post_add_relu"]["op_type"] == "relu"


def test_resnet50_checkpoint_generation_is_deterministic_for_fake_model(tmp_path: Path) -> None:
    checkpoint_path = get_quantized_checkpoint_path(tmp_path)
    first = build_resnet50_quantized_checkpoint(
        checkpoint_path,
        model_loader=build_fake_resnet50,
        generated_at="2026-04-14T12:00:00Z",
    )
    second = build_resnet50_quantized_checkpoint(
        checkpoint_path,
        model_loader=build_fake_resnet50,
        generated_at="2026-04-14T12:00:00Z",
    )

    assert first == second


def test_toy_model_outputs_are_still_available_for_legacy_golden_tests(tmp_path: Path) -> None:
    checkpoint_path = get_quantized_checkpoint_path(tmp_path)
    payload = build_toy_quantized_checkpoint(checkpoint_path)
    model = create_toy_model(payload)

    assert run_toy_model(model, [0, 1, 2, 7]) == [1, 3, 5, 15]


def test_load_quantized_checkpoint_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_quantized_checkpoint(tmp_path / "missing.pth")


def test_load_quantized_checkpoint_rejects_malformed_resnet50_metadata(tmp_path: Path) -> None:
    checkpoint_path = get_quantized_checkpoint_path(tmp_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "model_name": "resnet50",
            "quantization": "int8_symmetric_per_tensor",
            "generated_at": "2026-04-14T12:00:00Z",
            "layers": {
                "layer0_0_conv1": {
                    "op_type": "conv2d",
                    "input_shape": [1, 3, 224, 224],
                    "output_shape": [1, 4, 112, 112],
                    "weight_int8": [1, 2],
                    "bias_int32": [0, 0, 0, 0],
                    "weight_shape": [4, 3, 3, 3],
                    "num_weights": 108,
                    "scale_factor": 0.1,
                    "zero_point": 0,
                    "input_width_bits": 24,
                    "output_width_bits": 32,
                }
            },
        },
        checkpoint_path,
    )

    with pytest.raises(CheckpointValidationError, match="expected 108 from weight_shape"):
        load_quantized_checkpoint(checkpoint_path)
