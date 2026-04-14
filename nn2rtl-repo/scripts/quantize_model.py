#!/usr/bin/env python3
"""Create a deterministic toy quantized checkpoint for the automated nn2rtl flow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.paths import detect_repo_root
from scripts.quantize_impl import (
    build_quantization_summary,
    build_toy_quantized_checkpoint,
    resolve_checkpoint_path,
    write_quantized_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a deterministic toy quantized checkpoint used by the local nn2rtl tests.",
    )
    parser.add_argument(
        "checkpoint_path",
        nargs="?",
        default=None,
        help="Optional checkpoint path. Relative paths are resolved against the repository root.",
    )
    return parser.parse_args()


def load_quantization_config() -> dict[str, object]:
    raw = os.environ.get("NN2RTL_QUANTIZATION_CONFIG")
    if not raw:
        return {}

    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("NN2RTL_QUANTIZATION_CONFIG must decode to a JSON object.")
    return parsed


def main() -> None:
    args = parse_args()
    repo_root = detect_repo_root(__file__)
    checkpoint_path = resolve_checkpoint_path(repo_root, args.checkpoint_path)

    payload = build_toy_quantized_checkpoint(
        checkpoint_path,
        quantization_config=load_quantization_config(),
    )
    write_quantized_checkpoint(checkpoint_path, payload)
    print(json.dumps(build_quantization_summary(checkpoint_path, payload), indent=2))


if __name__ == "__main__":
    main()
