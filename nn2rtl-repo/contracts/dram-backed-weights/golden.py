"""DRAM-backed-weight golden vector adapter."""

from __future__ import annotations

from typing import Iterable


def generate_contract_vectors(
    input_samples: Iterable[list[int]],
    expected_samples: Iterable[list[int]],
) -> tuple[list[list[int]], list[list[int]]]:
    # Activation vectors stay in stream order; weights are supplied by the
    # contract's AXI memory model from the LayerIR weights_path.
    return list(input_samples), list(expected_samples)


def pack_weight_memory(weights: Iterable[int], *, data_width_bits: int = 64) -> list[int]:
    if data_width_bits % 8 != 0:
        raise ValueError("data_width_bits must be a byte multiple")
    bytes_per_word = data_width_bits // 8
    packed: list[int] = []
    word = 0
    shift = 0
    for value in weights:
        word |= (int(value) & 0xFF) << shift
        shift += 8
        if shift == bytes_per_word * 8:
            packed.append(word)
            word = 0
            shift = 0
    if shift:
        packed.append(word)
    return packed
