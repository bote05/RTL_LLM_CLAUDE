#!/usr/bin/env python3
"""ORDER-INVARIANT, SENSITIVE localization of the in-chain ±1/±2 via MULTISET (sorted-value) compare.

The streaming probe taps capture each layer's output in EMISSION order, which does NOT match the
contract/logical golden order — so a direct per-position value compare is meaningless (shows ~97% even
when byte-correct). BUT the MULTISET of values is order-invariant: if a layer's output is byte-correct
(merely reordered), sorted(capture) == sorted(golden) EXACTLY. If the layer has N values wrong by ±k,
the sorted arrays differ in ~2N positions. So walking the chain, the FIRST checkpoint whose sorted
values differ from its golden's localizes WHERE the ±1 first enters — reliably and sensitively
(detects even 1 wrong value), unlike distribution-L1 (insensitive to tiny counts) or value-compare
(order-broken).

Requires capture and golden to have the SAME element count (else flagged — a held/dropped-beat issue).

Usage: python scripts/localize_multiset.py [probe_dir]
"""
from __future__ import annotations
import struct, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROBE = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    ROOT / "output/reports_integrated/verilator_nn2rtl_top_probe"
GOLD = ROOT / "output/goldens"

# Single-stream conv checkpoints (RELIABLE multiset taps, unlike the held-beat add taps) in chain
# order, bisecting stages 1/2/3/4: conv_196(stem) maxpool conv_198(s1) conv_212(s2) conv_244(s3)
# conv_284(s4) relu_48(out). First DIFFER = where real ±k value errors enter.
CHAIN = ["node_conv_196", "node_max_pool2d", "node_conv_198", "node_conv_200",
         "node_relu_3", "node_conv_206", "node_conv_212",
         "node_conv_244", "node_conv_284", "node_relu_48"]


def golden_vals(mid: str):
    p = GOLD / f"{mid}.goldout"
    if not p.exists():
        return None
    raw = p.read_bytes()
    _, _, _, spv, bps = struct.unpack("<4sIIII", raw[:20])
    return np.frombuffer(raw[20:20 + spv * bps], dtype=np.int8)


def main() -> int:
    print(f"probe = {PROBE}")
    print(f"{'checkpoint':<18}{'cap_n':>9}{'gold_n':>9}{'multiset':>12}{'sorted_diff':>12}  note")
    print("-" * 76)
    first = None
    for mid in CHAIN:
        pb = PROBE / f"probe_{mid}.bin"
        g = golden_vals(mid)
        if not pb.exists() or g is None:
            print(f"{mid:<18}{'--':>9}{'--':>9}  (missing probe/golden)")
            continue
        cap = np.frombuffer(pb.read_bytes(), dtype=np.int8)
        note = ""
        if cap.size != g.size:
            # count mismatch -> can't multiset-compare cleanly (held/dropped beats or tiling size diff)
            print(f"{mid:<18}{cap.size:>9}{g.size:>9}{'COUNT-MISMATCH':>12}{'--':>12}  {note}")
            continue
        cs = np.sort(cap.astype(np.int16)); gs = np.sort(g.astype(np.int16))
        eq = np.array_equal(cs, gs)
        sdiff = int((cs != gs).sum())
        status = "EXACT" if eq else "DIFFER"
        if not eq and first is None:
            first = mid
            note = "<== FIRST VALUE DIVERGENCE (±k enters here)"
        print(f"{mid:<18}{cap.size:>9}{g.size:>9}{status:>12}{sdiff:>12}  {note}")
    print("-" * 76)
    print(f"FIRST MULTISET DIVERGENCE: {first or 'NONE (all checkpoints byte-correct multiset!)'}")
    print("(EXACT = layer output is byte-correct, only reordered. DIFFER = real ±k value errors enter here.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
