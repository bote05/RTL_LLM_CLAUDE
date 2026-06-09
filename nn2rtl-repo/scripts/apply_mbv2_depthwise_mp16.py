#!/usr/bin/env python3
"""[THROUGHPUT C 2026-06-09] Raise MAC parallelism MP 8->16 on the 6 final-stage depthwise convs
(878/884/890 C=576, 896/902/908 C=960). The other 11 inlined depthwise are ALREADY MP=16; this
makes all 17 uniform. #4 deleted all 12 retile bridges (big slice relief) so the area headroom that
made #3 stop at MP=8 is recovered. Both C values divide by 16 (576/16=36, 960/16=60 OC_PASSES) so
no partial pass. Output values are MP-INVARIANT (depthwise channels independent) => BYTE-EXACT; the
e2e gate also catches any cadence regression at the (now bridge-free) elastic handshake.

Change per conv: MP 8->16 ; lane_counter [2:0]->[3:0] ; mac_lane_q1/q2 [2:0]->[3:0] ;
3'd0->4'd0 (resets), 3'd7->4'd15 (==MP-1), 3'd1->4'd1 (incr) ON lane_counter/mac_lane LINES ONLY.
Count-validated per regex (aborts on unexpected count); leftover-3'd sanity check. Backs up.

Usage: python scripts/apply_mbv2_depthwise_mp16.py [--dry-run]
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "output" / "mobilenet-v2" / "rtl"
CONVS = [878, 884, 890, 896, 902, 908]

# (regex, replacement, min_count, max_count) — counts validated per file.
SUBS = [
    (re.compile(r"(localparam integer MP\s*=\s*)8;"), r"\g<1>16;", 1, 1),
    (re.compile(r"reg \[2:0\] lane_counter;"), "reg [3:0] lane_counter;", 1, 1),
    (re.compile(r"(reg )\[2:0\]([ ]+mac_lane_q[12];)"), r"\g<1>[3:0]\g<2>", 2, 2),
    (re.compile(r"(lane_counter\s*<=\s*)3'd0"), r"\g<1>4'd0", 1, 8),
    (re.compile(r"(mac_lane_q[12]\s*<=\s*)3'd0"), r"\g<1>4'd0", 2, 2),
    (re.compile(r"(lane_counter == )3'd7"), r"\g<1>4'd15", 1, 1),
    (re.compile(r"(lane_counter <= lane_counter \+ )3'd1"), r"\g<1>4'd1", 1, 1),
]


def main() -> int:
    dry = "--dry-run" in sys.argv
    bk = ROOT / "backups" / "mbv2_depthwise_mp16"
    if not dry:
        bk.mkdir(parents=True, exist_ok=True)
    done, skip = 0, 0
    for cid in CONVS:
        f = RTL / f"node_conv_{cid}.v"
        t = f.read_text()
        if re.search(r"localparam integer MP\s*=\s*16;", t):
            print(f"  node_conv_{cid}: already MP=16 -> SKIP"); skip += 1; continue
        counts = []
        new = t
        bad = None
        for rx, rep, lo, hi in SUBS:
            n = len(rx.findall(new))
            counts.append(n)
            if not (lo <= n <= hi):
                bad = f"regex {rx.pattern!r} matched {n} (expect {lo}..{hi})"
                break
            new = rx.sub(rep, new)
        if bad:
            print(f"  node_conv_{cid}: ABORT — {bad}"); skip += 1; continue
        leftover = len(re.findall(r"(lane_counter|mac_lane_q[12])\s*(<=|==|\+)\s*3'd", new))
        if leftover:
            print(f"  node_conv_{cid}: ABORT — {leftover} stray 3'd lane/mac literals remain"); skip += 1; continue
        if dry:
            print(f"  node_conv_{cid}: OK counts={counts}")
        else:
            (bk / f"node_conv_{cid}.v").write_text(t, newline="\n")
            f.write_text(new, newline="\n")
            print(f"  node_conv_{cid}: APPLIED counts={counts}")
        done += 1
    print(f"[mp16] {'validated' if dry else 'applied'}={done} skipped={skip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
