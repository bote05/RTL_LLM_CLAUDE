#!/usr/bin/env python3
"""Block-14 re-route (analogous to block-8): the wiring tool laid the engine/DRAM
convs as a linear chain and swapped relu_39<->conv_288 for the projection branch.
Correct: add_13 = conv_286(expand,engine) + conv_288(projection,DRAM);
conv_288's input is the block-14 input relu_39; conv_288 (pulse, no ready_out)
is buffered in the skip FIFO; conv_286 (engine bridge) is the live add input
gated on skip_valid. relu_39 fans to ldr8(conv_282 reduce)+conv_288.
Idempotent (checks each replacement). conv_288.ready_in IS an output (drives
node_conv_288_ready_in), so it stays in relu_39's combined.
"""
from __future__ import annotations
from pathlib import Path

TOP = Path("output/rtl/nn2rtl_top.v")
txt = TOP.read_text()

EDITS = [
    # (label, old, new)
    ("E1 conv_288 valid_in <- relu_39",
     ".valid_in(node_conv_286_valid_out & spatial_run),",
     ".valid_in(node_relu_39_valid_out & node_relu_39_ready_out_combined & spatial_run),"),
    ("E2 conv_288 data_in <- relu_39",
     ".data_in(node_conv_286_data_out),",
     ".data_in(node_relu_39_data_out),"),
    ("E3 add_13 main <- conv_286",
     ".valid_in(node_conv_288_valid_out & node_add_13_skip_valid & spatial_run),",
     ".valid_in(node_conv_286_valid_out & node_add_13_skip_valid & spatial_run),"),
    ("E4 add_13 data low <- conv_286",
     ".data_in({node_add_13_skip_data, node_conv_288_data_out[255:0]}),",
     ".data_in({node_add_13_skip_data, node_conv_286_data_out[255:0]}),"),
    ("E5+E6 skip_add_13 src <- conv_288 (scoped via [255:0])",
     ".in_valid(node_relu_39_valid_out & spatial_run & node_relu_39_ready_out_combined),\n        .in_data(node_relu_39_data_out[255:0]),",
     ".in_valid(node_conv_288_valid_out & spatial_run & node_add_13_skip_in_ready),\n        .in_data(node_conv_288_data_out[255:0]),"),
    ("E7 skip_add_13 out_ready gate on conv_286 (the live main)",
     ".out_ready(node_add_13_ready_in & node_conv_288_valid_out)",
     ".out_ready(node_add_13_ready_in & node_conv_286_valid_out)"),
    ("E8 conv_286 bridge ready_out -> add_13 skip-gated",
     ".ready_out((node_conv_288_ready_in & spatial_run)),",
     ".ready_out((node_add_13_ready_in & node_add_13_skip_valid & spatial_run)),"),
    ("E9 relu_39 combined -> ldr8 & conv_288_ready_in",
     "wire node_relu_39_ready_out_combined = (ldr_node_conv_282_in_ready & node_add_13_skip_in_ready) & spatial_run;",
     "wire node_relu_39_ready_out_combined = (ldr_node_conv_282_in_ready & node_conv_288_ready_in) & spatial_run;"),
]

for label, old, new in EDITS:
    if new in txt and old not in txt:
        print(f"[skip] {label}: already applied")
        continue
    cnt = txt.count(old)
    if cnt != 1:
        print(f"[FAIL] {label}: found {cnt} matches (need exactly 1)")
        continue
    txt = txt.replace(old, new, 1)
    print(f"[ok] {label}")

TOP.write_text(txt)
print("[written]")
