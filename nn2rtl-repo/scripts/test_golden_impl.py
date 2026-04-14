from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.golden_impl import (
    build_pipeline_ir_payload,
    fold_batch_norm_into_conv,
    int8_to_hex,
    write_pipeline_ir,
    write_signed_int8_hex,
)
from scripts.quantize_impl import build_toy_quantized_checkpoint, get_quantized_checkpoint_path, write_quantized_checkpoint


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
def test_build_pipeline_ir_payload_uses_canonical_signals_and_writes_artifacts(tmp_path: Path) -> None:
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
def test_write_pipeline_ir_creates_the_expected_json_output(tmp_path: Path) -> None:
    checkpoint_path = get_quantized_checkpoint_path(tmp_path)
    write_quantized_checkpoint(checkpoint_path, build_toy_quantized_checkpoint(checkpoint_path))

    output_path = write_pipeline_ir(tmp_path, checkpoint_path)
    payload = json.loads(output_path.read_text(encoding="utf8"))

    assert output_path == tmp_path / "output" / "golden_vectors.json"
    assert payload["layers"][0]["weights_path"].endswith("toy_conv1x1_weights.hex")
    assert payload["layers"][0]["bias_path"].endswith("toy_conv1x1_bias.hex")
