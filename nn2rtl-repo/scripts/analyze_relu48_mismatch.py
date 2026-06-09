#!/usr/bin/env python3
"""[RESNET 2953 DIAGNOSIS] Structural analysis of the e2e relu_48 mismatch.

Loads the raw RTL m_axis dump (NN2RTL_DUMP_PATH from nn2rtl_top_value_tb.cpp,
3136 beats x 32 bytes = 100352 bytes, signed int8) and the relu_48 contract
golden, then emits RAW EVIDENCE (no hypotheses): mismatch map by pixel /
channel / tile, error sign + magnitude, and the read-before-write (RTL==0)
fraction. Layout: beat -> pixel=beat//64, tile=beat%64, channel=tile*32+byte.

Usage: python scripts/analyze_relu48_mismatch.py [dump.bin]
"""
from __future__ import annotations
import sys, struct
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DUMP = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    ROOT / "output/reports_integrated/verilator_nn2rtl_top_value/rtl_dump_vec0.bin"


def find_golden() -> Path:
    cdir = ROOT / "output/goldens/contracts"
    for e in sorted(cdir.iterdir()):
        if e.name.startswith("node_relu_48_"):
            g = e / "node_relu_48.goldout"
            if g.exists():
                return g
    raise SystemExit("relu_48 contract goldout not found")


def load_golden_vec0() -> np.ndarray:
    raw = find_golden().read_bytes()
    magic, ver, nv, spv, bps = struct.unpack_from("<4sIIII", raw, 0)
    assert magic == b"NN2V" and bps == 32, (magic, bps)
    wps = (bps + 3) // 4
    nwords = spv * wps  # vec0 only
    words = np.frombuffer(raw[20:20 + nwords * 4], dtype="<u4").reshape(spv, wps)
    return words.view(np.uint8).reshape(spv, bps)  # (3136, 32)


def main() -> None:
    if not DUMP.exists():
        raise SystemExit(f"dump not found: {DUMP}")
    rtl = np.frombuffer(DUMP.read_bytes(), dtype=np.uint8)
    g = load_golden_vec0()
    nbeats = g.shape[0]
    rtl = rtl[:nbeats * 32].reshape(nbeats, 32)
    print(f"[analyze] beats={nbeats} rtl_bytes={rtl.size} golden_bytes={g.size}")

    rs = rtl.astype(np.int16); rs[rs > 127] -= 256
    gs = g.astype(np.int16);   gs[gs > 127] -= 256
    diff = rs - gs
    bad = diff != 0
    nbad = int(bad.sum())
    print(f"\n=== TOTALS ===")
    print(f"mismatch bytes = {nbad}/{rtl.size} ({100*nbad/rtl.size:.3f}%)")
    if nbad == 0:
        print("BYTE-EXACT — no mismatches"); return
    print(f"  RTL>golden (+): {int((diff>0).sum())}    RTL<golden (-): {int((diff<0).sum())}")
    print(f"  max |err| = {int(np.abs(diff).max())}   mean |err| (on bad) = {np.abs(diff[bad]).mean():.2f}")
    rtl0 = int(((rs == 0) & bad).sum())
    g0   = int(((gs == 0) & bad).sum())
    print(f"  bad bytes where RTL==0 : {rtl0} ({100*rtl0/nbad:.1f}%)   where golden==0: {g0}")

    print(f"\n=== ERROR MAGNITUDE HISTOGRAM (|err|) ===")
    mags = np.abs(diff[bad])
    for lo, hi in [(1,1),(2,2),(3,4),(5,8),(9,16),(17,32),(33,64),(65,127),(128,255)]:
        c = int(((mags >= lo) & (mags <= hi)).sum())
        if c: print(f"  |err| {lo:>3}-{hi:<3}: {c}")

    beat_bad = bad.sum(axis=1)
    nbeats_bad = int((beat_bad > 0).sum())
    print(f"\n=== PER-BEAT ===  beats with >=1 bad byte: {nbeats_bad}/{nbeats}")
    fb = int(np.where(beat_bad > 0)[0][0])
    print(f"  first bad beat = {fb} (pixel={fb//64}, tile={fb%64})")
    vals, cnts = np.unique(beat_bad[beat_bad > 0], return_counts=True)
    print("  bad-bytes-per-bad-beat:", {int(v): int(c) for v, c in zip(vals, cnts)})

    px = beat_bad.reshape(49, 64).sum(axis=1)
    print(f"\n=== PER-PIXEL (49) bad-byte counts ===")
    print("  ", {i: int(px[i]) for i in range(49) if px[i] > 0})

    tile_bad = bad.reshape(49, 64, 32).sum(axis=(0, 2))
    print(f"\n=== PER-TILE (0..63) bad-byte counts (summed over pixels) ===")
    print("  ", {i: int(tile_bad[i]) for i in range(64) if tile_bad[i] > 0})

    ch = bad.reshape(49, 64, 32).sum(axis=0).reshape(2048)
    bad_ch = np.where(ch > 0)[0]
    print(f"\n=== PER-CHANNEL (2048) ===  channels with >=1 bad byte: {len(bad_ch)}")
    print("  bad channels:", bad_ch.tolist()[:80], "..." if len(bad_ch) > 80 else "")
    order = np.argsort(ch)[::-1]
    top = [(int(c), int(ch[c])) for c in order[:15] if ch[c] > 0]
    print("  top channels (ch,count):", top)

    byte_pos = bad.reshape(-1, 32).sum(axis=0)
    print(f"\n=== WITHIN-TILE BYTE POSITION (0..31) bad counts ===")
    print("  ", {i: int(byte_pos[i]) for i in range(32) if byte_pos[i] > 0})


if __name__ == "__main__":
    main()
