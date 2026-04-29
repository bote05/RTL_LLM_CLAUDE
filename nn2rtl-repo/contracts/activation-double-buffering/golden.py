"""Activation-double-buffering golden vector adapter."""

from __future__ import annotations

from typing import Iterable


def generate_contract_vectors(
    input_samples: Iterable[list[int]],
    expected_samples: Iterable[list[int]],
    *,
    buffer_count: int = 2,
) -> tuple[list[list[int]], list[list[int]]]:
    if buffer_count != 2:
        raise ValueError("activation-double-buffering requires exactly two ping-pong buffers")
    return list(input_samples), list(expected_samples)
