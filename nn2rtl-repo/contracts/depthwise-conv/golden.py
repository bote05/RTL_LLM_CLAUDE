"""Depthwise-conv golden vector adapter.

The public bus shape is identical to flat-bus: one packed activation pixel
per beat ([N, C, h, w] flattened to a `C * 8`-bit word). Only the internal
compute pattern differs (per-channel 2D conv, no cross-channel reduction),
which is irrelevant for the goldens layout — Verilator compares the
ports-as-bus byte stream emitted by the RTL against the goldens written
by `onnx_frontend.py`. The conv2d golden generator already produces the
right INT8 stream when `groups == in_channels == out_channels`, so this
adapter is a passthrough.
"""

from __future__ import annotations

from typing import Iterable


def generate_contract_vectors(
    input_samples: Iterable[list[int]],
    expected_samples: Iterable[list[int]],
) -> tuple[list[list[int]], list[list[int]]]:
    return list(input_samples), list(expected_samples)
