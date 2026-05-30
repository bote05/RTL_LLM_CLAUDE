#!/usr/bin/env python3
"""Compare each captured chain-probe stream to its module's contract goldout
(vec0) and show where the e2e error first appears.

Robustness: add modules' goldouts are 128-bit (16 ch/sample) while the e2e
captures 256-bit beats (32 ch). One 256-bit beat = two 128-bit samples; the
half-ordering within the beat is unknown, so for each checkpoint we compare both
in-order and half-swapped packings and report the better (lower-mismatch) one.
Stem (conv_196, max_pool2d) and relu_48 are 256-bit -> directly comparable.

Reports per checkpoint: byte-mismatch %, max|err|, mean|err|, nonzero cap/gold.
The first checkpoint whose mismatch% jumps from ~0 to large localizes the bug.
"""
from __future__ import annotations
import struct, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DUMP = ROOT / "output/reports_integrated/verilator_nn2rtl_top_probe"
ORDER = (["node_conv_196", "node_max_pool2d", "node_add"] + [f"node_add_{k}" for k in range(1, 7)]
         + ["node_conv_244", "node_conv_246", "node_conv_248", "node_conv_250"]  # engine-region (stage3 blk0)
         + [f"node_add_{k}" for k in range(7, 16)] + ["node_relu_48"])


def goldout_vec0(mod: str):
    hits = list(ROOT.glob(f"output/goldens/contracts/{mod}_*/{mod}.goldout"))
    if not hits:
        return None, None
    raw = hits[0].read_bytes()
    magic, ver, nv, spv, bps = struct.unpack("<4sIIII", raw[:20])
    return np.frombuffer(raw[20:20 + spv * bps], dtype=np.int8), bps


def half_swapped(cap: np.ndarray) -> np.ndarray:
    # reshape into 32-byte beats, swap the two 16-byte halves
    n = (cap.size // 32) * 32
    b = cap[:n].reshape(-1, 32).copy()
    b[:, :16], b[:, 16:] = cap[:n].reshape(-1, 32)[:, 16:], cap[:n].reshape(-1, 32)[:, :16]
    return b.reshape(-1)


def main() -> None:
    dump = Path(sys.argv[1]) if len(sys.argv) > 1 else DUMP
    print(f"{'checkpoint':16s} {'capB':>7s} {'goldB':>7s} {'mism%':>7s} {'maxE':>5s} {'meanE':>6s} {'cap!=0':>8s} {'gold!=0':>8s} {'pack':>5s}")
    print("-" * 92)
    for mod in ORDER:
        binp = dump / f"probe_{mod}.bin"
        gold, bps = goldout_vec0(mod)
        if not binp.exists() or gold is None:
            print(f"{mod:16s}  (missing probe or goldout)")
            continue
        cap = np.frombuffer(binp.read_bytes(), dtype=np.int8)
        n = min(cap.size, gold.size)
        # try in-order and half-swapped (for 128-bit add goldens on 256-bit beats)
        cands = {"ord": cap[:n]}
        if bps == 16:
            cands["swap"] = half_swapped(cap)[:n]
        best = None
        for tag, c in cands.items():
            err = c[:n].astype(np.int16) - gold[:n].astype(np.int16)
            mm = int((err != 0).sum())
            if best is None or mm < best[1]:
                best = (tag, mm, err)
        tag, mm, err = best
        size_pen = abs(cap.size - gold.size)
        pct = 100.0 * (mm + size_pen) / max(gold.size, 1)
        mxe = int(np.abs(err).max()) if err.size else 0
        mne = float(np.abs(err[err != 0]).mean()) if mm else 0.0
        print(f"{mod:16s} {cap.size:>7d} {gold.size:>7d} {pct:>6.2f}% {mxe:>5d} {mne:>6.2f} "
              f"{int((cap!=0).sum()):>8d} {int((gold!=0).sum()):>8d} {tag:>5s}")
    print("-" * 92)
    print("Look for the first checkpoint where mism% jumps from ~0 to large -> error onset.")


if __name__ == "__main__":
    main()
