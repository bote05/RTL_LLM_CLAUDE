#!/usr/bin/env python3
"""[THROUGHPUT 2026-06-08] Raise MAC parallelism MP 4->8 on the 6 final-stage depthwise convs
(878/884/890/896/902/908). The other 11 DW are already MP=16. Lane-serial cost is
OC_PASSES*(MP+6)/pixel, so MP=4->8 ~halves the pass count on these wide convs (-296K e2e cyc,
agent-proven byte-exact at MP=8 in isolation). Output values are MP-INVARIANT (depthwise channels
independent), so this is BYTE-EXACT; the e2e gate also catches any retile-bridge cadence issue.

Change per conv: MP 4->8 ; widen lane_counter/mac_lane_q1/q2 [1:0]->[2:0] ; 2'd0->3'd0 (resets),
2'd3->3'd7 (==MP-1), 2'd1->3'd1 (incr) ON THE lane_counter/mac_lane LINES ONLY (other 2'd literals
untouched). Count-validated per regex (aborts if a substitution count is unexpected). Backs up.

Usage: python scripts/apply_mbv2_depthwise_mp8.py [--dry-run]
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
    (re.compile(r"(localparam integer MP\s*=\s*)4;"), r"\g<1>8;", 1, 1),
    (re.compile(r"reg \[1:0\] lane_counter;"), "reg [2:0] lane_counter;", 1, 1),
    (re.compile(r"(reg )\[1:0\]([ ]+mac_lane_q[12];)"), r"\g<1>[2:0]\g<2>", 2, 2),
    (re.compile(r"(lane_counter\s*<=\s*)2'd0"), r"\g<1>3'd0", 1, 8),
    (re.compile(r"(mac_lane_q[12]\s*<=\s*)2'd0"), r"\g<1>3'd0", 2, 2),
    (re.compile(r"(lane_counter == )2'd3"), r"\g<1>3'd7", 1, 1),
    (re.compile(r"(lane_counter <= lane_counter \+ )2'd1"), r"\g<1>3'd1", 1, 1),
]


def main() -> int:
    dry = "--dry-run" in sys.argv
    bk = ROOT / "backups" / "mbv2_depthwise_mp8"
    if not dry:
        bk.mkdir(parents=True, exist_ok=True)
    done, skip = 0, 0
    for cid in CONVS:
        f = RTL / f"node_conv_{cid}.v"
        t = f.read_text()
        if re.search(r"localparam integer MP\s*=\s*8;", t):
            print(f"  node_conv_{cid}: already MP=8 -> SKIP"); skip += 1; continue
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
        # sanity: no stray 2'd lane_counter/mac_lane literals remain
        leftover = len(re.findall(r"(lane_counter|mac_lane_q[12])\s*(<=|==|\+)\s*2'd", new))
        if leftover:
            print(f"  node_conv_{cid}: ABORT — {leftover} stray 2'd lane/mac literals remain"); skip += 1; continue
        if dry:
            print(f"  node_conv_{cid}: OK counts={counts}")
        else:
            (bk / f"node_conv_{cid}.v").write_text(t, newline="\n")
            f.write_text(new, newline="\n")
            print(f"  node_conv_{cid}: APPLIED counts={counts}")
        done += 1
    print(f"[mp8] {'validated' if dry else 'applied'}={done} skipped={skip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
