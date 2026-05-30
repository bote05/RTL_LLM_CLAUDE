#!/usr/bin/env python3
"""Insert a standard skip_fifo input skid between relu_{40,43,46} and the
parallelized 3x3 convs {284,292,298}.

Root cause (localized via beat counters): these 3 convs are fed DIRECTLY by their
relu with no boundary buffer. The old DRAM convs buffered their whole input
internally (line_buf), so the direct hookup was fine. The new split-arch convs
(apply_dram_conv3x3.py) don't, and they miss relu's 1-cycle output beat under a
timing edge -> conv receives 3135 of 3136 input beats -> stuck 1 beat short on its
last pixel -> engine deadlocks (disp_idx=9 for conv_284). Evidence: relu_40 emitted
all 3136 (idle, sending=0) but conv_284 captured only 3135.

Fix = the SAME pattern used at every other producer->consumer boundary (and the
conv_288 decouple): a skip_fifo. An always-ready FIFO can't miss the producer's
beat, then presents it to the conv with proper held backpressure. Idempotent.
"""
from __future__ import annotations
from pathlib import Path

TOP = Path("output/rtl/nn2rtl_top.v")
txt = TOP.read_text()

# conv id -> feeding relu id
CONV_RELU = {284: 40, 292: 43, 298: 46}

for m, r in CONV_RELU.items():
    if f"u_skid_node_conv_{m}" in txt:
        print(f"[skip] conv_{m}: input skid already present")
        continue

    # 1. relu_r ready_out gates on the skid's in_ready (not the conv's ready_in).
    old_ro = f".ready_out(node_conv_{m}_ready_in & spatial_run),"
    new_ro = f".ready_out(skid_node_conv_{m}_in_ready & spatial_run),"
    if txt.count(old_ro) != 1:
        raise SystemExit(f"[FAIL] conv_{m}: relu_{r} ready_out pattern count={txt.count(old_ro)} (need 1)")
    txt = txt.replace(old_ro, new_ro, 1)

    # 2. conv_m input now from the skid (valid_in + data_in).
    old_in = (f".valid_in(node_relu_{r}_valid_out & spatial_run),\n"
              f"        .ready_in(node_conv_{m}_ready_in),\n"
              f"        .data_in(node_relu_{r}_data_out),")
    new_in = (f".valid_in(skid_node_conv_{m}_valid & spatial_run),\n"
              f"        .ready_in(node_conv_{m}_ready_in),\n"
              f"        .data_in(skid_node_conv_{m}_data),")
    if txt.count(old_in) != 1:
        raise SystemExit(f"[FAIL] conv_{m}: input block count={txt.count(old_in)} (need 1)")
    txt = txt.replace(old_in, new_in, 1)

    # 3. Insert the skid just before the conv instance.
    anchor = f"    node_conv_{m} u_node_conv_{m} ("
    skid = (
        f"    // [input skid] standard skip_fifo boundary buffer relu_{r} -> conv_{m}\n"
        f"    wire skid_node_conv_{m}_in_ready, skid_node_conv_{m}_valid;\n"
        f"    wire [255:0] skid_node_conv_{m}_data;\n"
        f"    skip_fifo #(.WIDTH(256), .DEPTH(8192)) u_skid_node_conv_{m} (\n"
        f"        .clk(clk), .rst_n(rst_n),\n"
        f"        .in_valid(node_relu_{r}_valid_out & spatial_run & skid_node_conv_{m}_in_ready),\n"
        f"        .in_data(node_relu_{r}_data_out),\n"
        f"        .in_ready(skid_node_conv_{m}_in_ready),\n"
        f"        .out_valid(skid_node_conv_{m}_valid),\n"
        f"        .out_data(skid_node_conv_{m}_data),\n"
        f"        .out_ready(node_conv_{m}_ready_in & spatial_run)\n"
        f"    );\n"
    )
    if txt.count(anchor) != 1:
        raise SystemExit(f"[FAIL] conv_{m}: instance anchor count={txt.count(anchor)} (need 1)")
    txt = txt.replace(anchor, skid + anchor, 1)
    print(f"[ok] conv_{m}: inserted input skid from relu_{r}")

TOP.write_text(txt)
print("[written]")
