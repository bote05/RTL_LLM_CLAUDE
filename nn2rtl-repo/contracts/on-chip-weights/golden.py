"""On-chip-weights golden vector adapter.

The public activation bus is identical to flat-bus: one packed pixel per
beat. Weights are not part of the goldins/goldouts stream; they are
pre-loaded into the on-chip UltraRAM region by the top-level wrapper at
bitfile load and addressed per layer by ``weight_base_word``. The
testbench's behavioural URAM model reads its contents from the sidecar's
``weights_path`` .mem image.
"""

from __future__ import annotations

from typing import Iterable


def generate_contract_vectors(
    input_samples: Iterable[list[int]],
    expected_samples: Iterable[list[int]],
) -> tuple[list[list[int]], list[list[int]]]:
    # Activation vectors stay in stream order; weights are supplied by the
    # contract's URAM model from the LayerIR weights_path.
    return list(input_samples), list(expected_samples)


def pack_uram_memory(weights: Iterable[int], *, uram_word_bits: int = 288) -> list[int]:
    """Pack a stream of INT8 weights into URAM-word integers (little-endian).

    UltraScale+ URAM288 stores 288-bit words = 36 INT8 weights per word. The
    behavioural URAM model in the testbench reads packed words directly; the
    last word is zero-padded if the weight count is not a multiple of 36.
    """
    if uram_word_bits % 8 != 0:
        raise ValueError("uram_word_bits must be a byte multiple")
    bytes_per_word = uram_word_bits // 8
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
