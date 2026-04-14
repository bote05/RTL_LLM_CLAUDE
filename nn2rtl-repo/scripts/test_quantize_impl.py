from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.quantize_impl import (
    CheckpointValidationError,
    build_quantization_summary,
    build_toy_quantized_checkpoint,
    create_toy_model,
    get_quantized_checkpoint_path,
    load_quantized_checkpoint,
    resolve_checkpoint_path,
    run_toy_model,
    write_quantized_checkpoint,
)


def test_quantized_checkpoint_path_is_deterministic(tmp_path: Path) -> None:
    assert get_quantized_checkpoint_path(tmp_path) == tmp_path / "checkpoints" / "resnet50_int8.pth"


def test_resolve_checkpoint_path_supports_relative_and_absolute_paths(tmp_path: Path) -> None:
    absolute = tmp_path / "abs" / "toy.pth"
    assert resolve_checkpoint_path(tmp_path, "checkpoints/custom.pth") == tmp_path / "checkpoints" / "custom.pth"
    assert resolve_checkpoint_path(tmp_path, absolute) == absolute


def test_checkpoint_round_trip_preserves_quantization_metadata(tmp_path: Path) -> None:
    checkpoint_path = resolve_checkpoint_path(tmp_path, "checkpoints/toy_quantized.pth")
    payload = build_toy_quantized_checkpoint(
        checkpoint_path,
        quantization_config={"calibration": "toy"},
    )
    write_quantized_checkpoint(checkpoint_path, payload)

    loaded = load_quantized_checkpoint(checkpoint_path)

    assert loaded["checkpoint_path"] == str(checkpoint_path)
    assert loaded["quantization_config"] == {"calibration": "toy"}
    assert build_quantization_summary(checkpoint_path, loaded)["layers"]["toy_conv1x1"] == {
        "scale_factor": 0.125,
        "zero_point": 0,
        "weight_shape": [1, 1, 1, 1],
    }


def test_toy_model_outputs_are_deterministic(tmp_path: Path) -> None:
    checkpoint_path = get_quantized_checkpoint_path(tmp_path)
    payload = build_toy_quantized_checkpoint(checkpoint_path)
    model = create_toy_model(payload)

    assert run_toy_model(model, [0, 1, 2, 7]) == [1, 3, 5, 15]


def test_load_quantized_checkpoint_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_quantized_checkpoint(tmp_path / "missing.pth")


def test_load_quantized_checkpoint_rejects_malformed_metadata(tmp_path: Path) -> None:
    checkpoint_path = get_quantized_checkpoint_path(tmp_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_name": "toy_stream_net",
            "module_id": "toy_conv1x1",
            "quantization": "broken",
            "generated_at": "2026-04-14T00:00:00Z",
        },
        checkpoint_path,
    )

    with pytest.raises(CheckpointValidationError, match="quantization"):
        load_quantized_checkpoint(checkpoint_path)
