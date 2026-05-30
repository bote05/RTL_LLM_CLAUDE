#!/usr/bin/env python3
"""Add `& spatial_run` to every engine-loader-feeding relu's ready_out_combined.

These relus feed an engine loader (and sometimes an add skip): their
`node_relu_N_ready_out_combined = ldr_node_conv_X_in_ready [& node_add_M_skip_in_ready]`
but the loader/skip in_valid is `relu_valid & spatial_run & ready_out_combined`.
The relu's held-valid FSM advances when ready_out=1 (= ldr_inrdy, NO spatial_run),
but the consumer only captures when spatial_run=1 too. So when spatial_run toggles
LOW mid-send while the loader is ready, the relu advances and the consumer drops the
beat. Empirically this loses ~8 beats (one 256-ch pixel) per engine-conv frame,
leaving the next loader one word short -> engine deadlocks (seen at block-11:
conv_264 -> relu_31 -> ldr5 stuck at 1560/1568 beats = 195/196 words).

Fix: ready_out_combined &= spatial_run, matching the consumer's capture gate.
Safe: when spatial_run=0 both relu and consumer wait (no drop, no deadlock).
Idempotent.
"""
from __future__ import annotations
import re
from pathlib import Path

TOP = Path("output/rtl/nn2rtl_top.v")
txt = TOP.read_text()

pat = re.compile(r"(wire node_relu_\d+_ready_out_combined = )([^;]+);")
fixed = 0
def repl(m):
    global fixed
    expr = m.group(2).strip()
    if "spatial_run" in expr:
        return m.group(0)  # already gated
    if "ldr_node_conv" not in expr:
        return m.group(0)  # only fix engine-loader-feeding relus
    fixed += 1
    return f"{m.group(1)}({expr}) & spatial_run;"

txt = pat.sub(repl, txt)
TOP.write_text(txt)
print(f"[written] gated {fixed} relu ready_out_combined with spatial_run")
