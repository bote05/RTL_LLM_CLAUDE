#!/usr/bin/env python3
"""Block-15/16 identity-skip reduce loaders (ldr10=conv_290, ldr12=conv_296) lose
1 beat off their feeding relu's broadcast (relu_42->{ldr10,add_14-skip},
relu_45->{ldr12,add_15-skip}). Beat counters showed relu_42 captured 3136 input
but only 3135 reached the broadcast -> ldr10 stuck 1 beat short -> disp_idx stuck.

Same 1-beat-loss class fixed for conv_284 with a skid buffer. The LOADER is the
asymmetric broadcast consumer (the add skip is already a FIFO). Give the loader its
OWN always-ready input skid so the broadcast's combined-ready uses skid_in_ready
(always ~1) instead of the loader's ready -> relu's transition beat can't be dropped.
Loader then reads the skid (held-valid) and loads fully. Idempotent.
"""
from __future__ import annotations
from pathlib import Path

TOP = Path("output/rtl/nn2rtl_top.v")
txt = TOP.read_text()

# (relu id, reduce-conv id, add id)
BLOCKS = [(42, 290, 14), (45, 296, 15)]

for R, C, A in BLOCKS:
    sk = f"skid_ldr_{C}"
    if f"u_{sk}" in txt:
        print(f"[skip] ldr {C}: skid already present")
        continue

    # 1. ready_out_combined now gates on the skid's in_ready (not the loader's).
    old_ro = f"wire node_relu_{R}_ready_out_combined = (ldr_node_conv_{C}_in_ready & node_add_{A}_skip_in_ready) & spatial_run;"
    new_ro = f"wire node_relu_{R}_ready_out_combined = ({sk}_in_ready & node_add_{A}_skip_in_ready) & spatial_run;"
    if txt.count(old_ro) != 1:
        raise SystemExit(f"[FAIL] relu_{R} ready_out_combined count={txt.count(old_ro)} (need 1)")
    txt = txt.replace(old_ro, new_ro, 1)

    # 2. Loader reads the skid (do BEFORE inserting the skid, to keep patterns unique).
    old_in = (f".in_valid(node_relu_{R}_valid_out & spatial_run & node_relu_{R}_ready_out_combined),\n"
              f"         .in_data(node_relu_{R}_data_out),")
    new_in = (f".in_valid({sk}_valid & spatial_run),\n"
              f"         .in_data({sk}_data),")
    if txt.count(old_in) != 1:
        # tolerate 8-space indent variant
        old_in = (f".in_valid(node_relu_{R}_valid_out & spatial_run & node_relu_{R}_ready_out_combined),\n"
                  f"        .in_data(node_relu_{R}_data_out),")
        new_in = (f".in_valid({sk}_valid & spatial_run),\n"
                  f"        .in_data({sk}_data),")
    if txt.count(old_in) != 1:
        raise SystemExit(f"[FAIL] ldr {C} in_valid/in_data block not found uniquely")
    txt = txt.replace(old_in, new_in, 1)

    # 3. Insert the skid right after the (new) ready_out_combined wire.
    skid = (
        f"\n    // [reduce-loader skid] always-ready buffer relu_{R} -> ldr {C} so the\n"
        f"    // broadcast can't drop relu_{R}'s transition beat (block-{A} identity skip).\n"
        f"    wire {sk}_in_ready, {sk}_valid; wire [255:0] {sk}_data;\n"
        f"    skip_fifo #(.WIDTH(256), .DEPTH(8192)) u_{sk} (\n"
        f"        .clk(clk), .rst_n(rst_n),\n"
        f"        .in_valid(node_relu_{R}_valid_out & spatial_run & node_relu_{R}_ready_out_combined),\n"
        f"        .in_data(node_relu_{R}_data_out),\n"
        f"        .in_ready({sk}_in_ready),\n"
        f"        .out_valid({sk}_valid),\n"
        f"        .out_data({sk}_data),\n"
        f"        .out_ready(ldr_node_conv_{C}_in_ready & spatial_run)\n"
        f"    );\n"
    )
    txt = txt.replace(new_ro, new_ro + skid, 1)
    print(f"[ok] ldr {C}: inserted skid u_{sk} (relu_{R} broadcast)")

TOP.write_text(txt)
print("[written]")
