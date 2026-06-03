#!/usr/bin/env python3
"""Refresh the FINAL-OUTPUT contract golden after a weight regen.

The e2e harness (run_nn2rtl_top_value.ts) compares the RTL top m_axis to the
256-bit TILED contract golden at:
    output/goldens/contracts/<final>_*/<final>.goldout

That tiled golden is produced by the contract BUILD (build_top_wrapper), NOT by
generate_golden (which writes the UNTILED logical golden) nor by
rebuild_contract_goldens (which sees the final layer's IR output_width_bits =
logical 2048ch width == its own targetBytes -> no retile -> returns the logical
path and leaves the tiled contract file STALE). So after every weight regen the
final contract golden is left at its prior quantization -> the e2e silently
compares against a stale reference.

This script retiles the FRESH logical final golden -> 256-bit (32-byte) tiled
form and overwrites the contract-dir file in place. The NN2V data byte-order is
preserved by retiling (simple rechunk), so only the 20-byte header changes.

NN2V header (20 bytes, little-endian): magic 'NN2V' | u32 version | u32
num_vectors | u32 samples_per_vector | u32 bytes_per_sample.

Usage: python scripts/refresh_final_golden.py [final_module_id] [tile_bytes]
  defaults: final_module_id=node_relu_48  tile_bytes=32 (256-bit m_axis)
"""
from __future__ import annotations
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAGIC = b"NN2V"


def read_nn2v(p: Path):
    raw = p.read_bytes()
    magic, ver, nvec, samples, bps = struct.unpack_from("<4sIIII", raw, 0)
    assert magic == MAGIC, f"{p}: bad magic {magic!r}"
    data = raw[20:]
    assert len(data) == nvec * samples * bps, (
        f"{p}: data {len(data)} != {nvec}*{samples}*{bps}")
    return ver, nvec, samples, bps, data


def write_nn2v(p: Path, ver: int, nvec: int, samples: int, bps: int, data: bytes):
    assert len(data) == nvec * samples * bps
    p.write_bytes(struct.pack("<4sIIII", MAGIC, ver, nvec, samples, bps) + data)


def main() -> int:
    mid = sys.argv[1] if len(sys.argv) > 1 else "node_relu_48"
    tile_bytes = int(sys.argv[2]) if len(sys.argv) > 2 else 32

    logical = ROOT / "output" / "goldens" / f"{mid}.goldout"
    if not logical.exists():
        raise SystemExit(f"logical golden not found: {logical}")
    cands = sorted((ROOT / "output" / "goldens" / "contracts").glob(f"{mid}_*/{mid}.goldout"))
    if not cands:
        raise SystemExit(f"no contract-dir golden for {mid}")

    ver, nvec, lsamp, lbps, data = read_nn2v(logical)
    per_vec = lsamp * lbps                      # bytes per vector (preserved)
    if per_vec % tile_bytes != 0:
        raise SystemExit(f"{mid}: per-vector bytes {per_vec} not divisible by tile {tile_bytes}")
    tsamp = per_vec // tile_bytes               # tiled samples per vector

    for con in cands:
        # sanity: existing contract golden must have the same total payload
        _, cnvec, _, _, cdata = read_nn2v(con)
        if cnvec != nvec or len(cdata) != len(data):
            print(f"[refresh] WARN {con.name}: payload shape differs "
                  f"(con nvec={cnvec} bytes={len(cdata)} vs log nvec={nvec} bytes={len(data)}); skipping")
            continue
        write_nn2v(con, ver, nvec, tsamp, tile_bytes, data)
        print(f"[refresh] {mid}: wrote tiled contract golden ({nvec}x{tsamp}x{tile_bytes}B) -> "
              f"{con.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
