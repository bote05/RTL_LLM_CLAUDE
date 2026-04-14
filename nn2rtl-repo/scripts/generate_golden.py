#!/usr/bin/env python3
"""Generate deterministic golden vectors for the automated nn2rtl test flow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from scripts.golden_impl import summarize_pipeline_ir, write_pipeline_ir
from scripts.paths import detect_repo_root
from scripts.quantize_impl import resolve_checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate output/layer_ir.json from a quantized checkpoint.",
    )
    parser.add_argument(
        "checkpoint_path",
        nargs="?",
        default=None,
        help="Path to the quantized checkpoint file. Relative paths are resolved against the repository root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = detect_repo_root(__file__)
    checkpoint_path = resolve_checkpoint_path(repo_root, args.checkpoint_path)
    output_path = write_pipeline_ir(repo_root, checkpoint_path)
    payload = json.loads(output_path.read_text(encoding="utf8"))
    print(json.dumps(summarize_pipeline_ir(payload, checkpoint_path, output_path)))


if __name__ == "__main__":
    main()
