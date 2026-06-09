#!/usr/bin/env python3
"""[RESNET 2953 LOCALIZER] Compare every output/taps/node_relu_*.bin (captured by
the instrumented top during ONE e2e run) to its contract golden, and print the
FIRST-DIVERGING node in chain order. The first relu whose in-chain output
diverges from its isolation-verified golden localizes the integration bug;
everything after it is collateral (1x1 engine smear).

Usage: python scripts/compare_all_taps.py
"""
from __future__ import annotations
import struct, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TAPS = ROOT / "output" / "taps"
CONTRACTS = ROOT / "output" / "goldens" / "contracts"


def relu_sort_key(name: str) -> int:
    m = re.match(r"node_relu(?:_(\d+))?$", name)
    return int(m.group(1)) if (m and m.group(1)) else 0


def load_golden_vec0(node: str):
    cdir = None
    for e in sorted(CONTRACTS.iterdir()):
        if e.name.startswith(node + "_"):
            cdir = e; break
    if cdir is None:
        return None
    g = cdir / f"{node}.goldout"
    if not g.exists():
        return None
    raw = g.read_bytes()
    magic, ver, nv, spv, bps = struct.unpack_from("<4sIIII", raw, 0)
    if magic != b"NN2V" or bps != 32:
        return None
    wps = (bps + 3) // 4
    words = np.frombuffer(raw[20:20 + spv * wps * 4], dtype="<u4").reshape(spv, wps)
    return words.view(np.uint8).reshape(spv, 32)


def main() -> None:
    bins = sorted(TAPS.glob("node_relu*.bin"), key=lambda p: relu_sort_key(p.stem))
    if not bins:
        raise SystemExit(f"no taps found under {TAPS}")
    print(f"{'node':<16} {'tap_beats':>9} {'gold_beats':>10} {'cmp':>7} {'mismatch':>9} {'pct':>7} {'firstbad':>9}")
    print("-" * 78)
    rows = []
    for b in bins:
        node = b.stem
        g = load_golden_vec0(node)
        rtl = np.frombuffer(b.read_bytes(), dtype=np.uint8)
        tap_beats = rtl.size // 32
        if g is None:
            print(f"{node:<16} {tap_beats:>9} {'NO-GOLD':>10}")
            continue
        gold_beats = g.shape[0]
        n = min(tap_beats, gold_beats)
        if n == 0:
            print(f"{node:<16} {tap_beats:>9} {gold_beats:>10} {0:>7} {'--':>9} {'--':>7} {'--':>9}")
            continue
        r = rtl[:n * 32].reshape(n, 32)
        gg = g[:n]
        bad = (r != gg)
        nbad = int(bad.sum())
        beat_bad = bad.any(axis=1)
        fb = int(np.where(beat_bad)[0][0]) if nbad else -1
        pct = 100.0 * nbad / (n * 32)
        rows.append((relu_sort_key(node), node, tap_beats, gold_beats, n, nbad, pct, fb))
        flag = "  <== beat-count MISMATCH" if tap_beats != gold_beats else ""
        print(f"{node:<16} {tap_beats:>9} {gold_beats:>10} {n:>7} {nbad:>9} {pct:>6.2f}% {fb:>9}{flag}")

    print("-" * 78)
    diverging = [row for row in rows if row[5] > 0]
    diverging.sort(key=lambda x: x[0])
    if not diverging:
        print("ALL TAPPED RELUS BYTE-EXACT vs golden -- bug is between last clean relu and m_axis, "
              "or in a non-relu node (add/conv/gap/fc). Tap those next.")
        return
    first = diverging[0]
    print(f"FIRST-DIVERGING (chain order): {first[1]}  mismatch={first[5]} ({first[6]:.2f}%) "
          f"first_bad_beat={first[7]}  tap_beats={first[2]} gold_beats={first[3]}")
    print("Diverging nodes (chain order):",
          ", ".join(f"{r[1]}({r[6]:.1f}%)" for r in diverging[:20]))
    # beat-count anomalies are the strongest signal of a dropped/duplicated beat
    bc = [r for r in rows if r[2] != r[3]]
    if bc:
        print("BEAT-COUNT anomalies (tap!=gold) -- prime suspects for a dropped/dup beat:",
              ", ".join(f"{r[1]}(tap={r[2]} gold={r[3]})" for r in bc))


if __name__ == "__main__":
    main()
