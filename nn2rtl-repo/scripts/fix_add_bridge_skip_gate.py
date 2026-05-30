#!/usr/bin/env python3
"""Gate add-fed engine_output_bridge ready_out on the add's skip_valid.

These bridges feed a residual add directly: add.valid_in =
node_conv_X_valid_out & node_add_N_skip_valid & spatial_run. But the bridge's
ready_out was only `node_add_N_ready_in & spatial_run` — so when the add's
SKIP operand isn't valid yet, the bridge still emits (add_ready_in=1) and the
add can't capture (skip_valid=0) -> beats DROPPED. This bit block-8: conv_250
(fast projection, add main) drains before conv_248 (slow main path, add skip)
fills the skip FIFO -> conv_250's output is lost -> add_7 never completes ->
block-9 reduce never feeds dispatch 2 -> engine stuck at disp_idx=2.

Fix: ready_out = node_add_N_ready_in & node_add_N_skip_valid & spatial_run, so
the bridge only emits when the add will actually capture. skip_valid is the skip
FIFO's out_valid (non-empty), independent of the bridge, so no deadlock; for
identity-skip blocks (15/16) the skip is buffered early so the gate is a no-op.

USAGE: python scripts/fix_add_bridge_skip_gate.py
"""
from __future__ import annotations
from pathlib import Path

TOP = Path("output/rtl/nn2rtl_top.v")
# add-fed engine output bridges: add number -> (already implied)
ADDS = [7, 14, 15]

txt = TOP.read_text()
fixed = 0
for a in ADDS:
    old = f".ready_out((node_add_{a}_ready_in & spatial_run))"
    new = f".ready_out((node_add_{a}_ready_in & node_add_{a}_skip_valid & spatial_run))"
    if new in txt:
        print(f"[skip] add_{a}: already gated")
        continue
    if old not in txt:
        print(f"[WARN] add_{a}: pattern not found")
        continue
    txt = txt.replace(old, new, 1)
    fixed += 1
    print(f"[ok] add_{a}: ready_out += node_add_{a}_skip_valid")

TOP.write_text(txt)
print(f"\n[written] fixed {fixed}")
