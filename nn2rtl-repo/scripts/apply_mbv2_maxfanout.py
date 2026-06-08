#!/usr/bin/env python3
"""FMAX/CONGESTION 2026-06-08 (REVISED): cap fanout on the GENERAL-ROUTING high-fanout nets
in the MobileNetV2 depthwise convs.

REVISION RATIONALE: the placed-checkpoint fanout report shows Vivado ALREADY promotes the worst
net -- start_pulse (47K) -- to a global clock buffer (BUFGCE), which broadcasts it off general
routing. Forcing max_fanout replication on start_pulse would pull it BACK into general routing
(~184 replicas) = neutral-to-harmful. So we DO NOT touch start_pulse (let Vivado's BUFG handle it).
We cap only the FDCE nets that are genuinely in GENERAL ROUTING (not BUFG-promoted):
  - current_global_oc (~25K, FDCE, multi-bit channel-select -> not BUFG-able)
  - lane_counter      (~15K, FDCE, MAC lane select)
(tcnt in line_buf_window.v and in_tile in node_mean.v are handled by separate edits.)

max_fanout is a SYNTHESIS ATTRIBUTE only (Verilator-invisible) -> byte-exact by construction,
fit-safe. Benefit is a placement HINT, confirmable only by an in-context Vivado route.

Idempotent + backs up each original.
"""
from __future__ import annotations
import re, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RTL = REPO / "output" / "mobilenet-v2" / "rtl"
BACKUP_DIR = REPO / "backups" / "mbv2_maxfanout_routing_20260608"
MAXFAN = 256

DEPTHWISE = [812, 818, 824, 830, 836, 842, 848, 854, 860, 866,
             872, 878, 884, 890, 896, 902, 908]

# current_global_oc: "wire [W:0]  current_global_oc = ..." (W + whitespace vary)
CGO_RE  = re.compile(r"^(\s*)wire (\[\d+:0\]\s+current_global_oc\s*=)", re.MULTILINE)
# lane_counter: "reg [W:0] lane_counter;"
LANE_RE = re.compile(r"^(\s*)reg (\[\d+:0\] lane_counter;)", re.MULTILINE)
ATTR = f"(* max_fanout = {MAXFAN} *) "


def main() -> int:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    patched = 0
    for n in DEPTHWISE:
        f = RTL / f"node_conv_{n}.v"
        if not f.exists():
            print(f"[maxfanout] FATAL: {f.relative_to(REPO)} missing", file=sys.stderr); return 2
        src = f.read_text(encoding="utf-8")
        if "max_fanout" in src:
            print(f"[maxfanout] node_conv_{n}: already has max_fanout; skip"); continue
        orig = src
        cgo_n = len(CGO_RE.findall(src)); lane_n = len(LANE_RE.findall(src))
        if cgo_n == 1:  src = CGO_RE.sub(rf"\1{ATTR}wire \2", src, count=1)
        else:           print(f"[maxfanout] WARN node_conv_{n}: cgo hits={cgo_n}", file=sys.stderr)
        if lane_n == 1: src = LANE_RE.sub(rf"\1{ATTR}reg \2", src, count=1)
        else:           print(f"[maxfanout] WARN node_conv_{n}: lane_counter hits={lane_n}", file=sys.stderr)
        if src == orig:
            print(f"[maxfanout] node_conv_{n}: no change", file=sys.stderr); continue
        shutil.copy2(f, BACKUP_DIR / f.name)
        f.write_text(src, encoding="utf-8", newline="\n")
        patched += 1
        print(f"[maxfanout] node_conv_{n}: cgo+lane_counter capped @{MAXFAN}")
    print(f"[maxfanout] done: {patched} convs patched (backups in {BACKUP_DIR.relative_to(REPO)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
