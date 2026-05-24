#!/usr/bin/env python3
"""Boundary-width lint for the LayerIR chain.

Walks the LayerIR in topological order and checks, at each producer→consumer
edge, that the producer's `output_width_bits` matches the consumer's
EFFECTIVE input width:

    effective_input_width = layer.input_width_bits         (conv / relu / maxpool)
    effective_input_width = layer.input_width_bits / 2     (add: lhs half)

For `add` layers we additionally check that the skip-source's
`output_width_bits` matches the rhs half (`input_width_bits / 2`).

Exits non-zero on ANY mismatch. Prints the full mismatch list so the user
can decide which side to re-dispatch.

Usage:
    py scripts/lint_boundaries.py [--network resnet-50] [--ir output/layer_ir.json]
    py scripts/lint_boundaries.py --check-meta   # also cross-check against
                                                 # output/rtl/<id>.meta.json
                                                 # spec_hash widths.

Designed to be cheap and idempotent — runs in <1s, no LLM, no Vivado.
Run it BEFORE re-dispatch to know exactly which boundaries are wrong, and
AFTER re-dispatch to confirm the network is now coherent.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


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


def topology_chain(layers: list[dict]) -> list[tuple[str, str, str]]:
    """Return a list of (producer_id, consumer_id, edge_kind) tuples.

    edge_kind ∈ {"main", "skip"} — for add layers we emit both edges.

    Topology is inferred from IR order + the chain-tail / fork heuristics
    used by `scripts/build_top_wrapper.ts:computeTopology`. We replicate
    that logic in-line here so the lint runs without TS.
    """
    edges: list[tuple[str, str, str]] = []
    chain_tail = "PIXEL_IN"
    chain_width = layers[0]["input_width_bits"] if layers else 0
    last_fork: str | None = None
    pending_proj: str | None = None
    prev_op: str | None = None

    for i, L in enumerate(layers):
        mid = L["module_id"]
        if L["op_type"] == "add":
            edges.append((chain_tail, mid, "main"))
            skip_src = pending_proj or last_fork or "PIXEL_IN"
            edges.append((skip_src, mid, "skip"))
            pending_proj = None
            chain_tail = mid
            chain_width = L["output_width_bits"]
        elif L["op_type"] == "conv2d" and L["input_width_bits"] != chain_width:
            edges.append((last_fork or "PIXEL_IN", mid, "main"))
            pending_proj = mid
        else:
            edges.append((chain_tail, mid, "main"))
            chain_tail = mid
            chain_width = L["output_width_bits"]

        if L["op_type"] == "maxpool":
            last_fork = mid
        elif L["op_type"] == "relu" and prev_op == "add":
            last_fork = mid
        prev_op = L["op_type"]
    return edges


def effective_input_width(layer: dict, edge_kind: str) -> int:
    """Width the consumer expects on this edge.

    For `add`, lhs and rhs are each half of input_width_bits.
    """
    if layer["op_type"] == "add":
        return layer["input_width_bits"] // 2
    return layer["input_width_bits"]


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", default="resnet-50")
    parser.add_argument("--ir", default=None,
                        help="path to layer_ir.json; defaults to "
                             "<network outputDir>/layer_ir.json")
    parser.add_argument("--check-meta", action="store_true",
                        help="also cross-check against the per-module "
                             "output/rtl/<id>.meta.json spec_hash widths")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    repo_root = detect_repo_root(Path(__file__))
    net = load_network_config(repo_root, args.network)
    output_dir = (repo_root / net["outputDir"]).resolve()
    ir_path = Path(args.ir) if args.ir else (output_dir / "layer_ir.json")
    rtl_dir = output_dir / "rtl"

    with ir_path.open("r", encoding="utf-8") as fh:
        ir = json.load(fh)
    layers = ir["layers"]
    layers_by_id = {L["module_id"]: L for L in layers}

    edges = topology_chain(layers)
    print(f"[lint_boundaries] scanning {len(edges)} chain edges across "
          f"{len(layers)} layers")
    if args.verbose:
        for p, c, k in edges:
            print(f"  edge: {p:<22} -> {c:<22} ({k})")

    ir_mismatches: list[tuple[str, str, str, int, int]] = []
    meta_mismatches: list[tuple[str, int, int, int, int]] = []
    network_input_width = layers[0]["input_width_bits"] if layers else 0

    for prod_id, cons_id, kind in edges:
        cons = layers_by_id[cons_id]
        eff_in = effective_input_width(cons, kind)
        if prod_id == "PIXEL_IN":
            prod_out = network_input_width
        else:
            prod = layers_by_id.get(prod_id)
            if prod is None:
                print(f"[lint_boundaries] WARN: producer {prod_id} not in IR", file=sys.stderr)
                continue
            prod_out = prod["output_width_bits"]
        if prod_out != eff_in:
            ir_mismatches.append((prod_id, cons_id, kind, prod_out, eff_in))

    # Optional meta-cross-check. The spec_hash encodes the PHYSICAL bus
    # widths (lhs|rhs packed full bus for adds, single channel-tile beat
    # for everything else under tiled-streaming), so compare against
    # IR.input_width_bits / output_width_bits directly — do NOT halve
    # for adds, the spec_hash's `i` already includes both halves.
    if args.check_meta:
        for L in layers:
            mid = L["module_id"]
            mw = meta_widths(rtl_dir, mid)
            if mw is None:
                continue
            mi, mo = mw
            iri = L["input_width_bits"]
            iro = L["output_width_bits"]
            if mi != iri or mo != iro:
                meta_mismatches.append((mid, iri, iro, mi, mo))

    if ir_mismatches:
        print()
        print(f"[lint_boundaries] FAIL: {len(ir_mismatches)} IR boundary "
              f"mismatch(es):")
        print(f"  {'producer':<22} -> {'consumer':<22} {'edge':<6}  "
              f"{'prod.out':>8}  {'cons.in_eff':>11}")
        for prod_id, cons_id, kind, prod_out, eff_in in ir_mismatches:
            print(f"  {prod_id:<22} -> {cons_id:<22} {kind:<6}  "
                  f"{prod_out:>8}  {eff_in:>11}")

    if meta_mismatches:
        print()
        print(f"[lint_boundaries] FAIL: {len(meta_mismatches)} IR-vs-meta "
              f"width mismatch(es):")
        for mid, iri, iro, mi, mo in meta_mismatches:
            print(f"  {mid:<22}  IR.in_eff/out={iri}/{iro:<8}  "
                  f"meta.in/out={mi}/{mo}")

    if ir_mismatches or meta_mismatches:
        # Summarize unique modules involved
        prod_set = {e[0] for e in ir_mismatches if e[0] != "PIXEL_IN"}
        cons_set = {e[1] for e in ir_mismatches}
        meta_set = {m[0] for m in meta_mismatches}
        unique = prod_set | cons_set | meta_set
        print()
        print(f"[lint_boundaries] unique modules touched by mismatches: "
              f"{len(unique)}")
        print(f"  IR-only side: {len(prod_set | cons_set)} "
              f"({len(prod_set)} producers, {len(cons_set)} consumers)")
        print(f"  meta-only side: {len(meta_set)}")
        return 1

    print("[lint_boundaries] OK: all chain edges agree on widths.")
    if args.check_meta:
        print("[lint_boundaries] OK: per-module meta.json widths agree "
              "with LayerIR.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
