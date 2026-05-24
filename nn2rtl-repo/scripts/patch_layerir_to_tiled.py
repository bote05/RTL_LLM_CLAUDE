#!/usr/bin/env python3
"""Rewrite LayerIR widths + io_mode to the canonical tiled-streaming ABI.

Canonical ABI (matches contracts/tiled-streaming/metadata.json and the
addenda in knowledge/patterns/protected/01_context.md):

- channel_tile = 32
- conv2d / relu / maxpool:  input_width_bits  = output_width_bits = 256
- add:                       input_width_bits = 512, output_width_bits = 256
- io_mode = "channel_tiled" on every layer that uses tiled-streaming
- contract_id = "tiled-streaming" (or "on-chip-weights" if weights > 1MB)

For layers whose existing .v module's spec_hash already matches the
canonical widths (the 47 currently-tiled late-network layers), the IR
just gets the metadata tags added — no re-dispatch required.

For layers whose existing .v module is flat-bus / wider, the IR widths
change to tile and Foundry will need to re-dispatch the module.

The script prints a summary at the end:
  * tagged-only (no re-dispatch needed)
  * needs-redispatch (existing .v widths don't match new IR widths)
  * heavy-skipped (engine-dispatched; .v file isn't used in chain anyway)

Usage:
    py scripts/patch_layerir_to_tiled.py [--network resnet-50] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

CHANNEL_TILE = 32
TILE_BITS = CHANNEL_TILE * 8        # = 256
ADD_INPUT_BITS = 2 * TILE_BITS      # = 512 (lhs|rhs)

# Per the updated tiled-streaming metadata, the weight cap was raised to
# 4 MiB so all ResNet-50 spatial convs fit under one uniform contract on
# AMD Alveo U250. Engine-dispatched heavy convs (in heavy_set) are
# excluded from contract assignment because their .v module is not used
# in the chain at all.
TILED_STREAMING_WEIGHT_CAP_BYTES = 4_194_304


def detect_repo_root(script_path: Path) -> Path:
    override = os.environ.get("NN2RTL_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return script_path.resolve().parent.parent


def load_network_config(repo_root: Path, network_id: str) -> dict:
    with (repo_root / "networks.json").open("r", encoding="utf-8") as fh:
        registry = json.load(fh)
    for net in registry["networks"]:
        if net["id"] == network_id:
            return net
    raise SystemExit(f"unknown network '{network_id}'")


def load_heavy_set(repo_root: Path) -> set[str]:
    p = repo_root / "docs/agent_tasks/06_phase1_compression_candidates_HEAVY.txt"
    if not p.exists():
        return set()
    out: set[str] = set()
    for line in p.open("r", encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(line)
    return out


def meta_widths(rtl_dir: Path, module_id: str) -> tuple[int, int] | None:
    meta_path = rtl_dir / f"{module_id}.meta.json"
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        sh = meta.get("spec_hash") or ""
        m = re.search(r"_i(\d+)_o(\d+)", sh)
        if m:
            return int(m.group(1)), int(m.group(2))
    except Exception:
        return None
    return None


def pick_contract(layer: dict) -> str:
    op = layer.get("op_type")
    if op == "conv2d":
        weight_bytes = layer.get("num_weights", 0)
        if weight_bytes > TILED_STREAMING_WEIGHT_CAP_BYTES:
            return "on-chip-weights"
    return "tiled-streaming"


def canonical_widths(op_type: str) -> tuple[int, int]:
    """Return (input_width_bits, output_width_bits) for the canonical
    tile-256 ABI, given the op_type."""
    if op_type == "add":
        return ADD_INPUT_BITS, TILE_BITS
    return TILE_BITS, TILE_BITS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", default="resnet-50")
    parser.add_argument("--ir", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="don't write back; just print the summary")
    args = parser.parse_args(argv)

    repo_root = detect_repo_root(Path(__file__))
    net = load_network_config(repo_root, args.network)
    output_dir = (repo_root / net["outputDir"]).resolve()
    ir_path = Path(args.ir) if args.ir else (output_dir / "layer_ir.json")
    rtl_dir = output_dir / "rtl"

    with ir_path.open("r", encoding="utf-8") as fh:
        ir = json.load(fh)

    heavy_set = load_heavy_set(repo_root)

    tagged_only: list[str] = []
    needs_redispatch: list[str] = []
    heavy_skipped: list[str] = []
    contract_counts: dict[str, int] = {}

    for L in ir["layers"]:
        mid = L["module_id"]
        op = L.get("op_type", "")
        # Compute target widths and contract per-layer.
        new_in, new_out = canonical_widths(op)
        new_contract = pick_contract(L)
        contract_counts[new_contract] = contract_counts.get(new_contract, 0) + 1
        old_in = L.get("input_width_bits", 0)
        old_out = L.get("output_width_bits", 0)

        # Decide bucket
        if mid in heavy_set:
            # Engine-dispatched: still tag the IR (the wrapper's
            # engine_output_bridge derives DATA_W from this), but the .v
            # module is engine-handled so re-dispatch isn't needed.
            heavy_skipped.append(mid)
        else:
            mw = meta_widths(rtl_dir, mid)
            if mw is None:
                # No meta file: treat as needs-redispatch (safer)
                needs_redispatch.append(mid)
            else:
                mi, mo = mw
                # The .v module's effective input on the chain:
                # for add, the existing .v module declares full input_width
                # (= 2 * lhs_tile). meta records its declared input width as
                # 2 * lhs_tile too (or 256 if the existing tiled relu had a
                # 256-wide input). Compare:
                expected_module_in = new_in if op != "add" else new_in
                if mi == expected_module_in and mo == new_out:
                    tagged_only.append(mid)
                else:
                    needs_redispatch.append(mid)

        # Apply the IR mutation.
        L["input_width_bits"] = new_in
        L["output_width_bits"] = new_out
        L["io_mode"] = "channel_tiled"
        L["channel_tile"] = CHANNEL_TILE
        L["contract_id"] = new_contract

    # Summary
    print(f"=== Canonical tile-256 ABI patch summary ===")
    print(f"  channel_tile        : {CHANNEL_TILE}")
    print(f"  tile_bits           : {TILE_BITS} (= {CHANNEL_TILE}*8)")
    print(f"  add_input_bits      : {ADD_INPUT_BITS} (= 2*tile)")
    print()
    print(f"  total layers        : {len(ir['layers'])}")
    print(f"  tagged-only (no redispatch): {len(tagged_only)}")
    print(f"  needs-redispatch    : {len(needs_redispatch)}")
    print(f"  heavy-skipped (engine-handled): {len(heavy_skipped)}")
    print()
    print(f"  contract assignment:")
    for c, n in sorted(contract_counts.items(), key=lambda x: -x[1]):
        print(f"    {c:<22} : {n}")
    print()
    if needs_redispatch:
        print(f"  modules that need RE-DISPATCH ({len(needs_redispatch)}):")
        for mid in needs_redispatch:
            print(f"    - {mid}")

    if args.dry_run:
        print()
        print("[--dry-run] not writing back.")
        return 0

    # Write back
    backup_path = ir_path.with_suffix(".json.pre_tile256_bak")
    if not backup_path.exists():
        with ir_path.open("r", encoding="utf-8") as fh:
            backup_path.write_text(fh.read(), encoding="utf-8")
        print(f"  wrote IR backup -> {backup_path}")
    with ir_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(ir, fh, indent=2)
        fh.write("\n")
    print(f"  wrote patched IR -> {ir_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
