"""Weight-tiling golden vector adapter."""

from __future__ import annotations

from typing import Iterable


def generate_contract_vectors(
    input_samples: Iterable[list[int]],
    expected_samples: Iterable[list[int]],
    *,
    weight_tile_count: int,
) -> tuple[list[list[int]], list[list[int]]]:
    if weight_tile_count <= 1:
        raise ValueError("weight-tiling requires more than one weight tile")
    # Final expected outputs are emitted only after all partial tiles complete.
    return list(input_samples), list(expected_samples)


def split_weight_indices(num_weights: int, weight_tile_count: int) -> list[tuple[int, int]]:
    if num_weights < 0:
        raise ValueError("num_weights must be non-negative")
    if weight_tile_count <= 0:
        raise ValueError("weight_tile_count must be positive")
    if num_weights == 0:
        return []
    tile_size = (num_weights + weight_tile_count - 1) // weight_tile_count
    return [
        (start, min(start + tile_size, num_weights))
        for start in range(0, num_weights, tile_size)
    ]
