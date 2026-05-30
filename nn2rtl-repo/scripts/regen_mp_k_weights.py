#!/usr/bin/env python3
"""Regenerate ALL stale wide mp_k weight packings from the current flat weight
hex files.

ROOT CAUSE (2026-05-29): the INT4-GPTQ regen rewrote the flat per-conv weight
hex (node_conv_<id>_weights.hex) but did NOT regenerate the wide mp_k packings
(node_conv_<id>_weights_mp_k_<MPK>.hex) that conv_datapath_mp_k actually $readmemh's.
So every windowed spatial conv read STALE (pre-INT4) weights -> garbage MAC ->
saturation -> fed the engine garbage -> e2e relu_48 ~7% wrong. (Same class as the
stale residual-add constants bug.)

For each existing node_conv_<id>_weights_mp_k_<MPK>.hex this re-derives:
  OC, K_TOTAL from output/layer_ir.json (weight_shape = [OC, IC, KH, KW])
  MP from the wrapper output/rtl/node_conv_<id>.v (localparam integer MP=...)
  MP_K from the filename
and rewrites the wide packing from the CURRENT flat node_conv_<id>_weights.hex.

Usage: python scripts/regen_mp_k_weights.py [--dry-run]
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from repack_weights_wide import read_flat_weights, write_wide_weights  # noqa: E402

DRY = "--dry-run" in sys.argv
WDIR = ROOT / "output/weights"
RDIR = ROOT / "output/rtl"


def wrapper_mp(mid: str) -> int:
    txt = (RDIR / f"{mid}.v").read_text()
    # localparam integer MP=16, MP_K=9;  OR  localparam integer MP = 8;
    m = re.search(r"localparam\s+integer\s+MP\s*=\s*(\d+)", txt)
    if not m:
        raise RuntimeError(f"{mid}: MP localparam not found")
    return int(m.group(1))


def main() -> int:
    L = {l["module_id"]: l for l in json.load(open(ROOT / "output/layer_ir.json"))["layers"]}
    files = sorted(WDIR.glob("node_conv_*_weights_mp_k_*.hex"))
    print(f"found {len(files)} mp_k packings to regenerate")
    n_ok = n_skip = 0
    for f in files:
        m = re.match(r"(node_conv_\d+)_weights_mp_k_(\d+)\.hex", f.name)
        if not m:
            print(f"  SKIP (name): {f.name}"); n_skip += 1; continue
        mid, mpk = m.group(1), int(m.group(2))
        if mid not in L:
            print(f"  SKIP (no layer_ir): {mid}"); n_skip += 1; continue
        ws = L[mid]["weight_shape"]
        OC, IC, KH, KW = ws[0], ws[1], ws[2], ws[3]
        KT = IC * KH * KW
        MP = wrapper_mp(mid)
        flat = WDIR / f"{mid}_weights.hex"
        if not flat.exists():
            print(f"  SKIP (no flat): {mid}"); n_skip += 1; continue
        if KT % mpk != 0:
            print(f"  SKIP ({mid}: K_TOTAL {KT} %% MP_K {mpk} != 0)"); n_skip += 1; continue
        weights = read_flat_weights(flat)
        exp_len = OC * KT
        if len(weights) != exp_len:
            print(f"  WARN {mid}: flat has {len(weights)} weights, expected {exp_len}")
        if DRY:
            print(f"  [dry] {mid}: OC={OC} K_TOTAL={KT} MP={MP} MP_K={mpk} -> {f.name}")
            n_ok += 1; continue
        entries, padded = write_wide_weights(f, weights, OC, KT, MP, mpk)
        print(f"  [ok] {mid}: OC={OC} K_TOTAL={KT} MP={MP} MP_K={mpk} -> {entries} entries (pad={padded})")
        n_ok += 1
    print(f"\n{n_ok} regenerated, {n_skip} skipped" + ("  (dry-run)" if DRY else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
