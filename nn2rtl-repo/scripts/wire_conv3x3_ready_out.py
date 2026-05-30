#!/usr/bin/env python3
"""Wire ready_out for the 7 spatial 3x3 convs (step 2b).
Each conv feeds a downstream skid (u_skid_node_relu_N); ready_out =
that skid's in_ready & spatial_run. Inserts `.ready_out(<expr>),` after the
conv instantiation's `.valid_out(...)`. Idempotent.
"""
from __future__ import annotations
import re
from pathlib import Path

TOP = Path("output/rtl/nn2rtl_top.v")

# conv id -> downstream skid ready signal (confirmed from each conv's output
# skid in_valid: `node_conv_N_valid_out & spatial_run & <skid>_ready`)
READY = {
    200: "skid_node_relu_2_ready & spatial_run",
    208: "skid_node_relu_5_ready & spatial_run",
    214: "skid_node_relu_8_ready & spatial_run",
    220: "skid_node_relu_11_ready & spatial_run",
    228: "skid_node_relu_14_ready & spatial_run",
    234: "skid_node_relu_17_ready & spatial_run",
    240: "skid_node_relu_20_ready & spatial_run",
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
