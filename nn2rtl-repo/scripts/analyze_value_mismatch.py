#!/usr/bin/env python3
"""Analyze the e2e value-verification mismatch: diff the captured m_axis dump
against relu_48's contract goldout (vector 0) and report the ERROR STRUCTURE so
we can localize the integration bug.

Beat layout (contract bus ABI, pixel-major):
  3136 beats = 49 pixels x 64 tiles; beat = pixel*64 + tile; channel = tile*32 + byte.
  So output[pixel, channel] = int8 byte (channel%32) of beat (pixel*64 + channel//32).

Usage:
  python scripts/analyze_value_mismatch.py [captured.bin] [relu_48.goldout]
"""
from __future__ import annotations
import struct, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEF_CAP = ROOT / "output/reports_integrated/verilator_nn2rtl_top_value/captured_vec0.bin"
GOLDOUT_GLOB = "output/goldens/contracts/node_relu_48_*/node_relu_48.goldout"

N_BEATS, BYTES = 3136, 32
N_PIX, N_TILE, N_CH = 49, 64, 2048


def load_goldout_vec0(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    magic, ver, nv, spv, bps = struct.unpack("<4sIIII", raw[:20])
    assert magic == b"NN2V" and bps == BYTES and spv == N_BEATS, (magic, ver, nv, spv, bps)
    body = np.frombuffer(raw[20:], dtype=np.int8)  # int8 view of all bytes
    per_vec = spv * bps
    return body[:per_vec].reshape(N_BEATS, BYTES)  # vector 0


def load_capture(path: Path) -> np.ndarray:
    b = np.frombuffer(path.read_bytes(), dtype=np.int8)
    assert b.size == N_BEATS * BYTES, f"capture has {b.size} bytes, expected {N_BEATS*BYTES}"
    return b.reshape(N_BEATS, BYTES)


def beats_to_pix_ch(arr: np.ndarray) -> np.ndarray:
    # arr [N_BEATS, 32] -> [N_PIX, N_CH]
    out = np.zeros((N_PIX, N_CH), dtype=np.int16)
    for p in range(N_PIX):
        for t in range(N_TILE):
            out[p, t * 32:(t + 1) * 32] = arr[p * 64 + t]
    return out


def main() -> None:
    cap_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEF_CAP
    if len(sys.argv) > 2:
        gold_path = Path(sys.argv[2])
    else:
        gold_path = next(ROOT.glob(GOLDOUT_GLOB))
    print(f"[cap ] {cap_path}")
    print(f"[gold] {gold_path}")

    cap = beats_to_pix_ch(load_capture(cap_path)).astype(np.int16)
    gold = beats_to_pix_ch(load_goldout_vec0(gold_path)).astype(np.int16)

    diff = cap - gold
    mism = diff != 0
    total = mism.sum()
    print(f"\n=== OVERALL ===  mismatching elements {total}/{cap.size} ({100*total/cap.size:.2f}%)")
    print(f"max|err|={np.abs(diff).max()}  mean|err|(mismatches)="
          f"{np.abs(diff[mism]).mean():.3f}" if total else "no mismatches")

    # error-magnitude histogram
    print("\n=== ERROR MAGNITUDE (got-gold) histogram ===")
    vals, cnts = np.unique(diff[mism], return_counts=True)
    for v, c in sorted(zip(vals.tolist(), cnts.tolist()), key=lambda x: -x[1])[:20]:
        print(f"   err={v:+4d} : {c}")

    # relu-threshold flips
    relu_under = ((gold > 0) & (cap == 0)).sum()
    relu_over = ((gold == 0) & (cap != 0)).sum()
    print(f"\n=== RELU-THRESHOLD ===  gold>0 & cap==0 (underflow): {relu_under}   "
          f"gold==0 & cap!=0: {relu_over}")

    # per-pixel
    pp = mism.sum(axis=1)
    print(f"\n=== PER-PIXEL (49) mismatch counts ===\n{pp.tolist()}")
    print(f"   pixels with 0 mismatch: {(pp==0).sum()} / 49")

    # per-channel
    pc = mism.sum(axis=0)
    nz = np.nonzero(pc)[0]
    print(f"\n=== PER-CHANNEL (2048) ===  channels with >=1 mismatch: {nz.size}/2048")
    if nz.size:
        print(f"   channel range with mismatches: [{nz.min()}, {nz.max()}]")
        # cluster by 32-channel tile
        tile_counts = pc.reshape(N_TILE, 32).sum(axis=1)
        hot = [(t, int(tile_counts[t])) for t in range(N_TILE) if tile_counts[t]]
        print(f"   per-tile (32ch) mismatch counts (tile:count): {hot}")
        # top channels
        order = np.argsort(pc)[::-1]
        top = [(int(c), int(pc[c])) for c in order[:20] if pc[c]]
        print(f"   top-20 channels (ch:count): {top}")

    # sample first mismatches
    print("\n=== FIRST 20 MISMATCHES (pixel, channel, gold, got) ===")
    ps, cs = np.nonzero(mism)
    for i in range(min(20, ps.size)):
        p, c = int(ps[i]), int(cs[i])
        print(f"   pix={p:2d} ch={c:4d}  gold={gold[p,c]:4d} got={cap[p,c]:4d}")


if __name__ == "__main__":
    main()
