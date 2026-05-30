#!/usr/bin/env python3
"""Localize the first in-chain divergence by de-tiling each probe capture to
[pixels, channels] and comparing to that module's LOGICAL golden (channel-order,
ABI-unambiguous) — unlike the contract goldouts whose intermediate tiling does
NOT match the chain streaming order.

Probe bin: contract bus ABI, 32 bytes/beat, pixel-major:
    beat = pixel*tiles + tile ; channel = tile*32 + byte ; tiles = beats/pixels.
Logical golden (output/goldens/<id>.goldout): NN2V header + nv vectors, vec0 is
[spv=pixels, bps=channels] int8 in channel order.

Usage: python scripts/localize_logical.py [dumpdir]
"""
from __future__ import annotations
import struct, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DUMP = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    ROOT / "output/reports_integrated/verilator_nn2rtl_top_probe"
GOLD = ROOT / "output/goldens"

CHAIN = ["node_conv_196", "node_max_pool2d", "node_add"] + \
        [f"node_add_{i}" for i in range(1, 16)] + ["node_relu_48"]


def logical_vec0(mid: str):
    p = GOLD / f"{mid}.goldout"
    if not p.exists():
        return None
    raw = p.read_bytes()
    _, _, _, spv, bps = struct.unpack("<4sIIII", raw[:20])
    body = np.frombuffer(raw[20:20 + spv * bps], dtype=np.int8).reshape(spv, bps)
    return body  # [pixels, channels]


def detile(probe: np.ndarray, pixels: int) -> np.ndarray:
    beats = probe.shape[0]
    assert beats % pixels == 0, (beats, pixels)
    tiles = beats // pixels
    out = np.zeros((pixels, tiles * 32), dtype=np.int8)
    for t in range(tiles):
        out[:, t * 32:(t + 1) * 32] = probe[t::tiles] if False else \
            probe[np.arange(pixels) * tiles + t]
    return out


def main() -> int:
    print(f"dumpdir = {DUMP}")
    print(f"{'checkpoint':<18}{'pix':>6}{'ch':>6}{'mismatch':>11}{'pct':>7}{'maxerr':>7}  note")
    print("-" * 78)
    first = None
    for mid in CHAIN:
        pb = DUMP / f"probe_{mid}.bin"
        g = logical_vec0(mid)
        if not pb.exists() or g is None:
            print(f"{mid:<18}{'--':>6}{'--':>6}  {'(no probe or logical golden)'}")
            continue
        probe = np.frombuffer(pb.read_bytes(), dtype=np.int8).reshape(-1, 32)
        pixels, chans = g.shape
        dt = detile(probe, pixels)
        if dt.shape[1] != chans:
            print(f"{mid:<18}{pixels:>6}{chans:>6}  channel-count mismatch dt={dt.shape[1]}")
            continue
        d = np.abs(dt.astype(np.int16) - g.astype(np.int16))
        nz = int((d != 0).sum())
        pct = nz / d.size * 100
        note = ""
        if nz and first is None:
            first = mid
            note = "<== FIRST DIVERGENCE"
        print(f"{mid:<18}{pixels:>6}{chans:>6}{nz:>11}{pct:>6.1f}%{int(d.max()) if nz else 0:>7}  {note}")
    print("-" * 78)
    print(f"FIRST DIVERGENCE: {first or 'NONE (all byte-exact!)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
