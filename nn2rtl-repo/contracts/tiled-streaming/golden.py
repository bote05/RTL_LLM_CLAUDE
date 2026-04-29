"""Tiled-streaming golden vector adapter."""

from __future__ import annotations

from typing import Iterable


def _tile_channels(sample: list[int], channel_tile: int) -> list[list[int]]:
    if channel_tile <= 0:
        raise ValueError("channel_tile must be positive")
    return [sample[i : i + channel_tile] for i in range(0, len(sample), channel_tile)]


def generate_contract_vectors(
    input_samples: Iterable[list[int]],
    expected_samples: Iterable[list[int]],
    *,
    channel_tile: int = 32,
) -> tuple[list[list[int]], list[list[int]]]:
    tiled_inputs: list[list[int]] = []
    tiled_outputs: list[list[int]] = []
    for sample in input_samples:
        tiled_inputs.extend(_tile_channels(sample, channel_tile))
    for sample in expected_samples:
        tiled_outputs.extend(_tile_channels(sample, channel_tile))
    return tiled_inputs, tiled_outputs
