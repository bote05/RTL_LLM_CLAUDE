#!/usr/bin/env python3
"""Wire ready_out for the 14 parallelized stride-1 1x1 convs (step 1).
Downstream type determines the expr:
  reduce conv -> downstream skid: skid_node_relu_N_ready & spatial_run
  expand conv -> add main input:  node_add_M_skip_valid & spatial_run & node_add_M_ready_in
  skip proj   -> skip FIFO / loader bridge: <ready> & spatial_run
conv_224 (1x1 stride-2) is intentionally EXCLUDED — it stays serial (slow skip
projection feeding block-4 add; never overflows its consumer, so lossless w/o
backpressure). Idempotent.
"""
from __future__ import annotations
import re
from pathlib import Path

TOP = Path("output/rtl/nn2rtl_top.v")

READY = {
    204: "node_add_skip_in_ready & spatial_run",                          # skip proj -> skip FIFO
    224: "node_add_3_skip_in_ready & spatial_run",                        # decimator skip proj -> skip FIFO
    206: "skid_node_relu_4_ready & spatial_run",                          # reduce -> skid
    210: "node_add_1_skip_valid & spatial_run & node_add_1_ready_in",     # expand -> add
    212: "skid_node_relu_7_ready & spatial_run",
    216: "node_add_2_skip_valid & spatial_run & node_add_2_ready_in",
    226: "skid_node_relu_13_ready & spatial_run",
    230: "node_add_4_skip_valid & spatial_run & node_add_4_ready_in",
    232: "skid_node_relu_16_ready & spatial_run",
    238: "skid_node_relu_19_ready & spatial_run",
    248: "ldr_node_conv_250_in_ready & spatial_run",                      # skip proj -> loader bridge
    256: "node_add_8_skip_valid & spatial_run & node_add_8_ready_in",
    268: "node_add_10_skip_valid & spatial_run & node_add_10_ready_in",
    274: "node_add_11_skip_valid & spatial_run & node_add_11_ready_in",
    280: "node_add_12_skip_valid & spatial_run & node_add_12_ready_in",
}

txt = TOP.read_text()
patched = 0
for n, expr in READY.items():
    mod = f"node_conv_{n}"
    inst_re = re.compile(
        rf"({mod}\s+u_{mod}\s*\(.*?\.valid_out\({mod}_valid_out\),)(\s*\n)",
        re.DOTALL,
    )
    m = inst_re.search(txt)
    if not m:
        print(f"[WARN] {mod}: instantiation/valid_out not found")
        continue
    inst_close = txt.find(");", m.start())
    if ".ready_out(" in txt[m.start():inst_close]:
        print(f"[skip] {mod}: ready_out already wired")
        continue
    repl = m.group(1) + m.group(2) + f"        .ready_out({expr}),\n"
    txt = txt[:m.start()] + repl + txt[m.end():]
    patched += 1
    print(f"[ok] {mod}: .ready_out({expr})")

TOP.write_text(txt)
print(f"\n[written] patched {patched}")
