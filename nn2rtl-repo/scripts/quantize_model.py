#!/usr/bin/env python3
"""Create a deterministic ResNet-50 int8 checkpoint for the automated nn2rtl flow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from scripts.paths import detect_repo_root
from scripts.quantize_impl import (
    build_quantization_summary,
    build_resnet50_quantized_checkpoint,
    resolve_checkpoint_path,
    write_quantized_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write a deterministic ResNet-50 int8 checkpoint. "
            "Calibration uses 32 synthetic tensors; swap in ImageNet samples for real PTQ."
        ),
    )
    parser.add_argument(
        "checkpoint_path",
        nargs="?",
        default=None,
        help="Optional checkpoint path. Relative paths are resolved against the repository root.",
    )
    parser.add_argument(
        "--network",
        default=None,
        help="Network id from networks.json. Uses its defaultCheckpointPath when checkpoint_path is omitted.",
    )
    return parser.parse_args()


def default_checkpoint_for_network(repo_root: Path, network_id: str | None) -> str | None:
    if not network_id:
        return None
    registry = json.loads((repo_root / "networks.json").read_text(encoding="utf8"))
    for network in registry.get("networks", []):
        if network.get("id") == network_id:
            return network.get("defaultCheckpointPath")
    known = ", ".join(str(n.get("id")) for n in registry.get("networks", []))
    raise ValueError(f"Unknown network '{network_id}'. Known: {known}")


def main() -> None:
    torch.manual_seed(0)
    args = parse_args()
    repo_root = detect_repo_root(__file__)
    checkpoint_path = resolve_checkpoint_path(
        repo_root,
        args.checkpoint_path or default_checkpoint_for_network(repo_root, args.network),
    )

    payload = build_resnet50_quantized_checkpoint(checkpoint_path)
    write_quantized_checkpoint(checkpoint_path, payload)
    print(json.dumps(build_quantization_summary(checkpoint_path, payload), indent=2))


if __name__ == "__main__":
    main()
