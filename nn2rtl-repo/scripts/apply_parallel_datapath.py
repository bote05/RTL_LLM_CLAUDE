#!/usr/bin/env python3
"""Switch the spatial conv modules from conv_datapath to conv_datapath_parallel.

For each target module:
  - Rewrite `conv_datapath #(` -> `conv_datapath_parallel #(`
  - Rewrite WEIGHTS_PATH from `_weights.hex` to `_weights_wide.hex`
  - (LATENCY_PAD handling is module-specific; node_conv_196 has been done
    manually; other modules don't have LATENCY_PAD.)

Run scripts/repack_weights_wide.py first to produce the *_weights_wide.hex
files.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def patch_one(path: Path, dry_run: bool) -> tuple[bool, str]:
    """Return (changed, message)."""
    txt = path.read_text()
    orig = txt
    # Skip if already converted.
    if "conv_datapath_parallel" in txt:
        return (False, "already-parallel")
    # Only patch files that actually instantiate conv_datapath. If the file
    # doesn't, bail BEFORE the weight-path rewrite — otherwise a global
    # `_weights.hex" -> _weights_wide.hex"` rewrite corrupts unrelated
    # modules (pointwise embedded-MAC modules use the flat _weights.hex).
    # This was the bug that broke 10 pointwise modules in the e2e build.
    if not re.search(r"\bconv_datapath\s*#\(", txt):
        return (False, "no-conv_datapath-instance")
    new = re.sub(r"\bconv_datapath\s*#\(", "conv_datapath_parallel #(", txt)
    new = re.sub(r"_weights\.hex\"", '_weights_wide.hex"', new)
    if new == orig:
        return (False, "no-change")
    if dry_run:
        return (True, "would-patch")
    path.write_text(new)
    return (True, "patched")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rtl-dir", type=Path, default=Path("output/rtl"))
    p.add_argument("--only", help="comma-separated module IDs (e.g. node_conv_200,node_conv_208)")
    p.add_argument("--skip", help="comma-separated module IDs to skip")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    only = set(args.only.split(",")) if args.only else None
    skip = set(args.skip.split(",")) if args.skip else set()

    rtl_dir = args.rtl_dir.resolve()
    patched = 0
    skipped = 0
    unchanged = 0
    for path in sorted(rtl_dir.glob("node_conv_*.v")):
        if path.name.endswith(".preimprove"):
            continue
        mid = path.stem
        if only is not None and mid not in only:
            continue
        if mid in skip:
            print(f"[skip-explicit] {mid}")
            skipped += 1
            continue
        changed, msg = patch_one(path, args.dry_run)
        if changed:
            print(f"[{'dry' if args.dry_run else 'patched'}] {mid}: {msg}")
            patched += 1
        else:
            print(f"[unchanged] {mid}: {msg}")
            unchanged += 1
    print(f"\n[summary] patched={patched} unchanged={unchanged} skipped={skipped}")


if __name__ == "__main__":
    main()
