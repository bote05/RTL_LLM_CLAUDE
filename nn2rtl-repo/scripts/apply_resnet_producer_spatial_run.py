#!/usr/bin/env python3
"""[RESNET BYTE-EXACT FIX 2026-06-09] Gate the PRODUCER ready_out by spatial_run on the 6 residual-
block joins that fix_relu_combined_spatial_run.py missed (its repl bailed unless the expr contained
'ldr_node_conv', so the skid/add-skip-fed joins were never gated). ROOT CAUSE of the e2e 2953/100352
(2.94%) relu_48 mismatch: the consuming skid only writes when spatial_run=1, but these 6 producers
ADVANCE on ready_out alone -> when spatial_run drops (engine_busy) mid-send while ready_out stays
high, the producer marches past a beat the skid never captured -> dropped beat -> corrupt 32-ch
tile -> smeared to relu_48 (RTL=0 where golden=4). The fix mirrors the already-gated loader relus,
is VALUE-PRESERVING (makes RTL match the EXISTING golden -> no golden/latency rebuild), and is
DEADLOCK-SAFE (spatial_run=0 -> ready_out_combined=0 -> relu/maxpool HOLDS, identical to the fixed
loader relus; bounded skip-FIFO sizing unchanged). Patches the on-disk top surgically (nn2rtl_top.v
is patched-not-regenerated). Idempotent + backup + count-validated.

Usage: python scripts/apply_resnet_producer_spatial_run.py [--dry-run]
"""
from __future__ import annotations
import re, sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOP = ROOT / "output" / "rtl" / "nn2rtl_top.v"
WIRES = [
    "node_max_pool2d_ready_out_combined",
    "node_relu_3_ready_out_combined",
    "node_relu_6_ready_out_combined",
    "node_relu_12_ready_out_combined",
    "node_relu_15_ready_out_combined",
    "node_relu_18_ready_out_combined",
]


def main() -> int:
    dry = "--dry-run" in sys.argv
    t = TOP.read_text()
    if not dry:
        bk = ROOT / "backups" / "resnet_producer_spatial_run"
        bk.mkdir(parents=True, exist_ok=True)
        shutil.copy(TOP, bk / "nn2rtl_top.v")
    done, skip, miss = 0, 0, 0
    for w in WIRES:
        pat = re.compile(r"(wire\s+" + re.escape(w) + r"\s*=\s*)([^;]*?)\s*;")
        m = pat.search(t)
        if not m:
            print(f"  MISS {w}"); miss += 1; continue
        expr = m.group(2)
        if "spatial_run" in expr:
            print(f"  SKIP {w} (already gated)"); skip += 1; continue
        repl = f"{m.group(1)}({expr}) & spatial_run;"
        t = t[:m.start()] + repl + t[m.end():]
        print(f"  GATE {w}: ({expr}) & spatial_run")
        done += 1
    if miss:
        print(f"[producer-spatial-run] ABORT — {miss} wire(s) not found (anchor drift); no write")
        return 1
    if not dry:
        TOP.write_text(t, newline="\n")
    print(f"[producer-spatial-run] {'validated' if dry else 'applied'}={done} skipped={skip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
