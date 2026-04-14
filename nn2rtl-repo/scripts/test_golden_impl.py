from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.golden_impl import (
    build_pipeline_ir_payload,
    fold_batch_norm_into_conv,
    int8_to_hex,
    is_absolute_posix_path,
    write_pipeline_ir,
    write_signed_int8_hex,
)
from scripts.quantize_impl import (
    build_toy_quantized_checkpoint,
    get_quantized_checkpoint_path,
    write_quantized_checkpoint,
)


def build_fx_checkpoint_payload() -> dict[str, object]:
    scalar_shape = [1, 1, 1, 1]
    passthrough_shape = [1]
    return {
        "format_version": 2,
        "model_name": "resnet50",
        "quantization": "int8_symmetric_per_tensor",
        "generated_at": "2026-04-14T12:00:00Z",
        "residual_stack_spec": {
            "input_name": "input",
            "output_module_id": "relu2",
            "operations": [
                {"module_id": "conv1", "op_type": "conv2d", "input": "input"},
                {"module_id": "relu1", "op_type": "relu", "input": "conv1"},
                {"module_id": "conv2", "op_type": "conv2d", "input": "relu1"},
                {"module_id": "add0", "op_type": "add", "lhs": "conv2", "rhs": "input"},
                {"module_id": "relu2", "op_type": "relu", "input": "add0"},
            ],
        },
        "layers": {
            "conv1": {
                "op_type": "conv2d",
                "input_shape": list(scalar_shape),
                "output_shape": list(scalar_shape),
                "weight_shape": [1, 1, 1, 1],
                "num_weights": 1,
                "scale_factor": 0.125,
                "zero_point": 0,
                "weights": [2],
                "bias": [1],
            },
            "relu1": {
                "op_type": "relu",
                "input_shape": list(scalar_shape),
                "output_shape": list(scalar_shape),
                "weight_shape": list(passthrough_shape),
                "num_weights": 0,
                "scale_factor": 1.0,
                "zero_point": 0,
            },
            "conv2": {
                "op_type": "conv2d",
                "input_shape": list(scalar_shape),
                "output_shape": list(scalar_shape),
                "weight_shape": [1, 1, 1, 1],
                "num_weights": 1,
                "scale_factor": 0.25,
                "zero_point": 0,
                "weights": [-1],
                "bias": [2],
            },
            "add0": {
                "op_type": "add",
                "input_shape": list(scalar_shape),
                "output_shape": list(scalar_shape),
                "weight_shape": list(passthrough_shape),
                "num_weights": 0,
                "scale_factor": 0.5,
                "lhs_scale_factor": 0.25,
                "rhs_scale_factor": 0.5,
                "zero_point": 0,
                "input_width_bits": 16,
                "output_width_bits": 8,
            },
            "relu2": {
                "op_type": "relu",
                "input_shape": list(scalar_shape),
                "output_shape": list(scalar_shape),
                "weight_shape": list(passthrough_shape),
                "num_weights": 0,
                "scale_factor": 0.5,
                "zero_point": 0,
            },
        },
    }


def pack_int8_pair(lhs: int, rhs: int) -> int:
    packed = (int(lhs) & 0xFF) | ((int(rhs) & 0xFF) << 8)
    if packed >= 2**15:
        packed -= 2**16
    return packed


def test_int8_to_hex_serializes_signed_values() -> None:
    assert int8_to_hex(-128) == "80"
    assert int8_to_hex(-1) == "FF"
    assert int8_to_hex(0) == "00"
    assert int8_to_hex(127) == "7F"


def test_int8_to_hex_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError):
        int8_to_hex(128)


def test_write_signed_int8_hex_writes_one_uppercase_value_per_line(tmp_path: Path) -> None:
    hex_path = tmp_path / "weights.hex"
    write_signed_int8_hex([-1, 0, 127, -128], hex_path)
    assert hex_path.read_text(encoding="utf8") == "FF\n00\n7F\n80\n"


def test_fold_batch_norm_into_conv_matches_the_standard_formula() -> None:
    weight = torch.tensor([[[[2.0]]]])
    bias = torch.tensor([1.0])
    bn_weight = torch.tensor([0.5])
    bn_bias = torch.tensor([3.0])
    running_mean = torch.tensor([2.0])
    running_var = torch.tensor([4.0])

    folded_weight, folded_bias = fold_batch_norm_into_conv(
        weight,
        bias,
        bn_weight,
        bn_bias,
        running_mean,
        running_var,
        eps=0.0,
    )

    assert torch.allclose(folded_weight, torch.tensor([[[[0.5]]]]))
    assert torch.allclose(folded_bias, torch.tensor([2.75]))


@pytest.mark.full
def test_build_pipeline_ir_payload_keeps_legacy_toy_flow_working(tmp_path: Path) -> None:
    checkpoint_path = get_quantized_checkpoint_path(tmp_path)
    write_quantized_checkpoint(checkpoint_path, build_toy_quantized_checkpoint(checkpoint_path))

    pipeline_ir = build_pipeline_ir_payload(checkpoint_path, tmp_path)
    layer = pipeline_ir["layers"][0]

    assert pipeline_ir["generated_at"] == "2026-04-14T00:00:00Z"
    assert layer["clock_signal"] == "clk"
    assert layer["reset_signal"] == "rst_n"
    assert layer["valid_in_signal"] == "valid_in"
    assert layer["valid_out_signal"] == "valid_out"
    assert layer["ready_in_signal"] == "ready_in"
    assert layer["data_in_signal"] == "data_in"
    assert layer["data_out_signal"] == "data_out"
    assert layer["golden_inputs"] == [[0, 1, 2, 7]]
    assert layer["golden_outputs"] == [[1, 3, 5, 15]]
    assert Path(layer["weights_path"]).exists()
    assert Path(layer["bias_path"]).exists()


@pytest.mark.full
def test_build_pipeline_ir_payload_captures_fx_layers_in_topological_order(tmp_path: Path) -> None:
    checkpoint_path = get_quantized_checkpoint_path(tmp_path)
    write_quantized_checkpoint(checkpoint_path, build_fx_checkpoint_payload())

    pipeline_ir = build_pipeline_ir_payload(
        checkpoint_path,
        tmp_path,
        generated_at="2026-04-14T12:34:56Z",
    )

    assert pipeline_ir["model_name"] == "resnet50"
    assert pipeline_ir["generated_at"] == "2026-04-14T12:34:56Z"
    assert [layer["module_id"] for layer in pipeline_ir["layers"]] == [
        "conv1",
        "relu1",
        "conv2",
        "add0",
        "relu2",
    ]

    for index, layer in enumerate(pipeline_ir["layers"]):
        assert layer["ready_in_signal"] == "ready_in"
        assert is_absolute_posix_path(layer["weights_path"])
        assert Path(layer["weights_path"]).exists()
        assert len(layer["golden_outputs"]) == 8
        if index == 0:
            assert len(layer["golden_inputs"]) == 8
        elif layer["module_id"] != "add0":
            assert layer["golden_inputs"] == pipeline_ir["layers"][index - 1]["golden_outputs"]
        if layer["bias_path"] is not None:
            assert is_absolute_posix_path(layer["bias_path"])
            assert Path(layer["bias_path"]).exists()

    add_layer = next(layer for layer in pipeline_ir["layers"] if layer["module_id"] == "add0")
    conv2_layer = next(layer for layer in pipeline_ir["layers"] if layer["module_id"] == "conv2")
    input_vectors = pipeline_ir["layers"][0]["golden_inputs"]
    expected_packed_inputs = [
        [pack_int8_pair(lhs, rhs) for lhs, rhs in zip(lhs_vector, rhs_vector)]
        for lhs_vector, rhs_vector in zip(conv2_layer["golden_outputs"], input_vectors)
    ]
    expected_add_outputs = [
        [
            max(-128, min(127, round((lhs * 0.25 + rhs * 0.5) / 0.5)))
            for lhs, rhs in zip(lhs_vector, rhs_vector)
        ]
        for lhs_vector, rhs_vector in zip(conv2_layer["golden_outputs"], input_vectors)
    ]

    assert add_layer["input_width_bits"] == 16
    assert add_layer["lhs_scale_factor"] == pytest.approx(0.25)
    assert add_layer["rhs_scale_factor"] == pytest.approx(0.5)
    assert add_layer["golden_inputs"] == expected_packed_inputs
    assert add_layer["golden_outputs"] == expected_add_outputs
    assert pipeline_ir["layers"][-1]["golden_inputs"] == add_layer["golden_outputs"]

    weights_dir = tmp_path / "output" / "weights"
    assert (weights_dir / "conv1_weights.hex").read_text(encoding="utf8") == "02\n"
    assert (weights_dir / "conv1_bias.hex").read_text(encoding="utf8") == "01\n"
    assert (weights_dir / "conv2_weights.hex").read_text(encoding="utf8") == "FF\n"
    assert (weights_dir / "conv2_bias.hex").read_text(encoding="utf8") == "02\n"
    assert (weights_dir / "relu1_weights.hex").read_text(encoding="utf8") == ""
    assert (weights_dir / "add0_bias.hex").read_text(encoding="utf8") == ""


@pytest.mark.full
def test_write_pipeline_ir_writes_layer_ir_and_legacy_mirror(tmp_path: Path) -> None:
    checkpoint_path = get_quantized_checkpoint_path(tmp_path)
    write_quantized_checkpoint(checkpoint_path, build_fx_checkpoint_payload())

    output_path = write_pipeline_ir(
        tmp_path,
        checkpoint_path,
        generated_at="2026-04-14T12:34:56Z",
    )
    payload = json.loads(output_path.read_text(encoding="utf8"))
    mirrored = json.loads((tmp_path / "output" / "golden_vectors.json").read_text(encoding="utf8"))

    assert output_path == tmp_path / "output" / "layer_ir.json"
    assert mirrored == payload
    assert payload["layers"][0]["weights_path"].endswith("conv1_weights.hex")
    assert payload["layers"][0]["bias_path"].endswith("conv1_bias.hex")
