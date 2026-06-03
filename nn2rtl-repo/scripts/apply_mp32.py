#!/usr/bin/env python3
"""CYCLE-OPT (byte-exact-preserving): bump MP -> 32 on all INSTANTIATED spatial
conv wrappers (the 39 mp_k convs; NOT the 14 engine-dispatched convs, which are
computed by the shared engine and whose node mp_k datapath is vestigial). Halves
each conv's OC_PASSES = ceil(OC/MP). Byte-exact: the conv_datapath_mp_k math is
MP-independent (per-OC tree-sum, integer add associative); only the weight packing
changes (regen_mp_k_weights.py reads MP from the wrapper). Run regen + rebuild +
byte-exact gate (run_nn2rtl_top_value.ts 0 AND 1, both mismatch_bytes=0) after.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET_MP = int(sys.argv[sys.argv.index("--mp")+1]) if "--mp" in sys.argv else 32
DRY = "--dry-run" in sys.argv

sched = json.loads((ROOT/"output/rtl/nn2rtl_scheduler_schedule.json").read_text())
engine = {d["module_id"] for d in sched["dispatches"]}
# conv_196 is a SPECIAL stem wrapper (fixed 48-cyc output shift-reg) — MP-increase
# DEADLOCKS it (verified 2026-05-30). Exclude from the bulk; needs a wrapper rework.
SKIP = {"node_conv_196"}
engine = engine | SKIP
top = (ROOT/"output/rtl/nn2rtl_top.v").read_text()

n = 0; bk = ROOT/"backups"/f"mp{TARGET_MP}_20260530"; bk.mkdir(parents=True, exist_ok=True)
for f in sorted((ROOT/"output/rtl").glob("node_conv_*.v"), key=lambda p:int(re.search(r'\d+',p.stem).group())):
    mid = f.stem
    if mid in engine: continue                      # engine-dispatched -> node mp_k is vestigial
    if not ((f"u_{mid} " in top) or (f"{mid} u_" in top)): continue   # must be instantiated
    txt = f.read_text()
    # match `MP = N` regardless of trailing delimiter (handles both `MP = 8;` and
    # multi-decl `MP=16, MP_K=9;`). \b avoids matching MP_K.
    m = re.search(r"(localparam integer MP\b\s*=\s*)(\d+)", txt)
    if not m: print(f"  SKIP {mid}: no MP localparam"); continue
    cur = int(m.group(2))
    if cur >= TARGET_MP: print(f"  skip {mid}: MP={cur} already >= {TARGET_MP}"); continue
    new = txt[:m.start()] + f"{m.group(1)}{TARGET_MP}" + txt[m.end():]
    if DRY:
        print(f"  [dry] {mid}: MP {cur} -> {TARGET_MP}")
    else:
        (bk/f"{mid}.v").write_text(txt, newline="\n")
        f.write_text(new, newline="\n")
        print(f"  [ok] {mid}: MP {cur} -> {TARGET_MP}")
    n += 1
print(f"\n{'(dry) ' if DRY else ''}set MP={TARGET_MP} on {n} instantiated spatial convs. backups -> {bk}")
print("NEXT: python scripts/regen_mp_k_weights.py ; rebuild ; run_nn2rtl_top_value.ts 0 AND 1 (mismatch_bytes=0)")
