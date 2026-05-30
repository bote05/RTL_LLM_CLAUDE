#!/usr/bin/env python3
"""Fix the relu last-beat-not-held bug at block-15/16 (relu_42/45 -> reduce-loader-skid
+ add-skip broadcast). Counters proved relu PRODUCES all 3136 beats (r42_produce=3136)
but only 3135 are ACCEPTED (r42_bcast=3135): the relu's LAST beat per pixel (beat63,
emitted as sending->0) is presented for ONE cycle and NOT held under backpressure -- if
the spatial_run-gated combined-ready drops that exact cycle (block-15/16 see heavy engine
activity), beat63 is lost. ldr10/ldr12 then stuck 1 beat short -> engine deadlock.

Fix (external, no relu-module edit): make the relu -> output-skid transfer NOT depend on
spatial_run. The output buffers (reduce-loader skid + add-skip FIFO) are always-ready
(~not-full), so they accept every relu beat incl. the last regardless of spatial_run. The
spatial-chain freeze is preserved at the skid OUTPUTS (still spatial_run-gated). The relu's
INPUT was already spatial_run-free, so this fully decouples the relu (buffered both sides;
frame 3136 < skid 8192, no overflow). Idempotent.
"""
from __future__ import annotations
from pathlib import Path

TOP = Path("output/rtl/nn2rtl_top.v")
txt = TOP.read_text()

for R in (42, 45):
    # 1. ready_out_combined: drop trailing ' & spatial_run'.
    old_ro_suffix = f"node_relu_{R}_ready_out_combined = (skid_ldr_"
    # find the exact line and strip ' & spatial_run' before the ';'
    import re
    m = re.search(rf"(wire node_relu_{R}_ready_out_combined = \([^;]+?\)) & spatial_run;", txt)
    if m:
        txt = txt.replace(m.group(0), m.group(1) + ";", 1)
        print(f"[ok] relu_{R} ready_out_combined: dropped spatial_run")
    elif f"node_relu_{R}_ready_out_combined = (skid_ldr_" in txt and "& spatial_run;" not in re.search(rf"node_relu_{R}_ready_out_combined =.*", txt).group(0):
        print(f"[skip] relu_{R} ready_out_combined: already fixed")
    else:
        raise SystemExit(f"[FAIL] relu_{R} ready_out_combined pattern not found")

    # 2. relu->skid in_valid (skid_ldr + add-skip, both instances): drop ' spatial_run &'.
    old_iv = f"node_relu_{R}_valid_out & spatial_run & node_relu_{R}_ready_out_combined"
    new_iv = f"node_relu_{R}_valid_out & node_relu_{R}_ready_out_combined"
    n = txt.count(old_iv)
    if n == 0:
        print(f"[skip] relu_{R} in_valid: already fixed")
    else:
        txt = txt.replace(old_iv, new_iv)
        print(f"[ok] relu_{R} in_valid: dropped spatial_run ({n} occurrences)")

TOP.write_text(txt)
print("[written]")
