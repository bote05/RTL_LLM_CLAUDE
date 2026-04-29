"""Flat-bus golden vector adapter.

The flat-bus contract uses the repository's native .goldin/.goldout format:
each sample is one full packed pixel.
"""

from __future__ import annotations

from typing import Iterable


def generate_contract_vectors(
    input_samples: Iterable[list[int]],
    expected_samples: Iterable[list[int]],
) -> tuple[list[list[int]], list[list[int]]]:
    return list(input_samples), list(expected_samples)
