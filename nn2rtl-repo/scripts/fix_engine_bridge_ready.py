#!/usr/bin/env python3
"""Fix engine_output_bridge ready_out for SKID-fed bridges.

Each engine_output_bridge feeds a skid_node_relu_N, but its ready_out was wired
to node_relu_N_ready_in (the relu AFTER the skid) instead of skid_node_relu_N_ready
(the skid it actually hands off to). When that skid momentarily fills, the bridge
(seeing relu_ready_in=1) emits beats the full skid drops -> the engine conv's
consumer ends up short -> never completes a frame -> chain deadlocks at the next
engine dispatch. Aligning ready_out to the skid's in_ready makes the handoff lossless.

Only the SKID-fed bridges are touched. Bridges feeding an add/conv directly
(250->add_7, 286->conv_288, 294->add_14, 300->add_15) keep their consumer's
ready_in (correct, no intervening skid). conv_246 was fixed by hand already;
this script is idempotent and skips it.

USAGE: python scripts/fix_engine_bridge_ready.py
"""
from __future__ import annotations
from pathlib import Path

TOP = Path("output/rtl/nn2rtl_top.v")

# engine conv -> the relu# of the skid it feeds (skid_node_relu_<R>)
SKID_FED = {
    246: 23, 254: 26, 260: 29, 264: 31, 266: 32,
    272: 35, 278: 38, 282: 40, 290: 43, 296: 46,
}

txt = TOP.read_text()
fixed = 0
for conv, r in SKID_FED.items():
    old = f".ready_out((node_relu_{r}_ready_in & spatial_run))"
    new = f".ready_out((skid_node_relu_{r}_ready & spatial_run))"
    if new in txt:
        print(f"[skip] conv_{conv}: already fixed (skid_node_relu_{r}_ready)")
        continue
    if old not in txt:
        print(f"[WARN] conv_{conv}: pattern not found ({old})")
        continue
    txt = txt.replace(old, new, 1)
    fixed += 1
    print(f"[ok] conv_{conv}: ready_out -> skid_node_relu_{r}_ready & spatial_run")

TOP.write_text(txt)
print(f"\n[written] fixed {fixed}")
