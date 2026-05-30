#!/usr/bin/env python3
"""Phase A follow-on (anticipated by the plan): decouple relu_39 -> conv_288 with
conv_288's OWN input skid so conv_282's engine loader (ldr8) is not throttled.

Block-14 input relu_39 broadcasts to BOTH ldr8 (conv_282 reduce loader) and
conv_288 (projection) via a shared held-valid handshake
(node_relu_39_ready_out_combined = ldr8_in_ready & <conv_288 accept> & spatial_run).
The old DRAM conv_288 buffered its whole input first, so its ready_in stayed high
and ldr8 filled at line rate. The Phase A parallel conv_288 computes pixel-by-pixel
and drops ready_in during per-pixel output drains -> the shared handshake stalls ->
ldr8 never reaches `loaded` -> engine hard-deadlocks at disp_idx=8 (observed: stuck
10M..106M cyc, 0 output beats).

Fix: insert u_skid_node_conv_288 (skip_fifo, 8192 deep > the 6272-beat relu_39
frame) between relu_39 and conv_288. relu_39's shared accept now gates on the
SKID's in_ready (not conv_288's), so the skid absorbs the whole frame even if
conv_288 stalls, and ldr8 fills uninterrupted. conv_288 drains the skid at its own
pace. Idempotent.
"""
from __future__ import annotations
from pathlib import Path

TOP = Path("output/rtl/nn2rtl_top.v")
txt = TOP.read_text()

if "u_skid_node_conv_288" in txt:
    print("[skip] conv_288 input skid already present")
    raise SystemExit(0)

EDITS = [
    # 1. relu_39 shared accept gates on the skid's in_ready, not conv_288's ready_in.
    ("relu_39 combined -> skid in_ready",
     "wire node_relu_39_ready_out_combined = (ldr_node_conv_282_in_ready & node_conv_288_ready_in) & spatial_run;",
     "wire node_relu_39_ready_out_combined = (ldr_node_conv_282_in_ready & skid_node_conv_288_in_ready) & spatial_run;"),
    # 2. conv_288 input now from the skid (was relu_39 direct).
    ("conv_288 valid_in <- skid",
     ".valid_in(node_relu_39_valid_out & node_relu_39_ready_out_combined & spatial_run),\n        .ready_in(node_conv_288_ready_in),\n        .data_in(node_relu_39_data_out),",
     ".valid_in(skid_node_conv_288_valid & spatial_run),\n        .ready_in(node_conv_288_ready_in),\n        .data_in(skid_node_conv_288_data),"),
]

for label, old, new in EDITS:
    if new in txt and old not in txt:
        print(f"[skip] {label}: already applied")
        continue
    if txt.count(old) != 1:
        raise SystemExit(f"[FAIL] {label}: found {txt.count(old)} matches (need 1)")
    txt = txt.replace(old, new, 1)
    print(f"[ok] {label}")

# 3. Insert the skid instance + wire decls just before the conv_288 instance.
anchor = "    node_conv_288 u_node_conv_288 ("
skid = (
    "    // [Phase A decouple] conv_288's own input skid so conv_282's loader (ldr8)\n"
    "    // isn't throttled by conv_288's incremental-compute ready_in stalls.\n"
    "    wire skid_node_conv_288_in_ready, skid_node_conv_288_valid;\n"
    "    wire [255:0] skid_node_conv_288_data;\n"
    "    skip_fifo #(.WIDTH(256), .DEPTH(8192)) u_skid_node_conv_288 (\n"
    "        .clk(clk), .rst_n(rst_n),\n"
    "        .in_valid(node_relu_39_valid_out & spatial_run & node_relu_39_ready_out_combined),\n"
    "        .in_data(node_relu_39_data_out),\n"
    "        .in_ready(skid_node_conv_288_in_ready),\n"
    "        .out_valid(skid_node_conv_288_valid),\n"
    "        .out_data(skid_node_conv_288_data),\n"
    "        .out_ready(node_conv_288_ready_in & spatial_run)\n"
    "    );\n"
)
if txt.count(anchor) != 1:
    raise SystemExit(f"[FAIL] conv_288 instance anchor found {txt.count(anchor)} times (need 1)")
txt = txt.replace(anchor, skid + anchor, 1)
print("[ok] inserted u_skid_node_conv_288")

TOP.write_text(txt)
print("[written]")
