#!/usr/bin/env python3
"""Localize the first point of divergence in the conv_196->relu_48 in-chain e2e.

The probe TB (run_nn2rtl_top_probe.ts) dumps probe_<id>.bin for each checkpoint
(raw int8, contract bus ABI: 32 bytes/beat). For each checkpoint, compare it
beat-for-beat to that module's contract .goldout (vec0). Print in CHAIN ORDER so
the first row with mismatch>0 localizes where the integration diverges.

Each add can have multiple contract dirs (different downstream tilings). We try
each variant and report the one with the LOWEST mismatch (the matching tiling),
flagging when variants disagree.

Usage: python scripts/localize_e2e_divergence.py [dumpdir]
  dumpdir default: output/reports_integrated/verilator_nn2rtl_top_probe/probe_dump
"""
from __future__ import annotations
import struct, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONTRACTS = ROOT / "output/goldens/contracts"
DUMP = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    ROOT / "output/reports_integrated/verilator_nn2rtl_top_probe/probe_dump"

# Chain order of probe checkpoints (subset that have contract goldouts).
CHAIN = ["node_conv_196", "node_max_pool2d", "node_add"] + \
        [f"node_add_{i}" for i in range(1, 16)] + ["node_relu_48"]


def load_goldout_vec0(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    magic, ver, nv, spv, bps = struct.unpack("<4sIIII", raw[:20])
    assert magic == b"NN2V", (magic, path)
    body = np.frombuffer(raw[20:20 + spv * bps], dtype=np.int8)
    return body.reshape(spv, bps)  # [beats, 32]


def variants(mid: str) -> list[Path]:
    if not CONTRACTS.exists():
        return []
    return sorted(p / f"{mid}.goldout" for p in CONTRACTS.iterdir()
                  if p.name.startswith(mid + "_") and (p / f"{mid}.goldout").exists())


def gold_bps(path: Path) -> int:
    raw = path.read_bytes()
    _, _, _, _, bps = struct.unpack("<4sIIII", raw[:20])
    return bps


def compare(probe: np.ndarray, gold: np.ndarray) -> tuple[int, int, int, int]:
    n = min(probe.shape[0], gold.shape[0])
    d = np.abs(probe[:n].astype(np.int16) - gold[:n].astype(np.int16))
    nz = int((d != 0).sum())
    first = int(np.argwhere(d.reshape(-1) != 0)[0][0]) if nz else -1
    return nz, int(d.max()) if nz else 0, first, n


def main() -> int:
    print(f"dumpdir = {DUMP}")
    print(f"{'checkpoint':<18} {'beats':>6} {'gbeats':>6} {'mismatch':>9} {'maxerr':>6} {'first':>7}  note")
    print("-" * 84)
    first_diverge = None
    for mid in CHAIN:
        pb = DUMP / f"probe_{mid}.bin"
        if not pb.exists():
            print(f"{mid:<18} {'--':>6} {'--':>6} {'(no probe bin)':>9}")
            continue
        probe = np.frombuffer(pb.read_bytes(), dtype=np.int8)
        if probe.size % 32:
            print(f"{mid:<18}  bad probe size {probe.size}")
            continue
        probe = probe.reshape(-1, 32)
        vs = [v for v in variants(mid) if gold_bps(v) == 32]  # only width-matching
        if not vs:
            print(f"{mid:<18} {probe.shape[0]:>6} {'--':>6} {'(no bps=32 golden)':>9}")
            continue
        best = None
        for v in vs:
            g = load_goldout_vec0(v)
            res = compare(probe, g)
            if best is None or res[0] < best[0][0]:
                best = (res, v, g.shape[0])
        (nz, mx, first, n), vpath, gbeats = best
        note = ""
        if len(vs) > 1:
            note = f"{len(vs)} variants; best={vpath.parent.name[:40]}"
        if nz and first_diverge is None:
            first_diverge = mid
            note = (note + "  <== FIRST DIVERGENCE").strip()
        print(f"{mid:<18} {probe.shape[0]:>6} {gbeats:>6} {nz:>9} {mx:>6} {first:>7}  {note}")
    print("-" * 84)
    print(f"FIRST DIVERGENCE: {first_diverge or 'NONE (all byte-exact!)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
