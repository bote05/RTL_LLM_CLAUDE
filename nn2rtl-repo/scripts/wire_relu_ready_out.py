#!/usr/bin/env python3
"""Wire ready_out for the 25 comprehensive-backpressure relus (option 2).
Each relu is single-consumer; ready_out = downstream consumer's accept gate.
Inserts `.ready_out(<expr>),` after the relu instantiation's `.valid_out(...)`.
Idempotent.
"""
from __future__ import annotations
import re
from pathlib import Path

TOP = Path("output/rtl/nn2rtl_top.v")

# relu suffix -> ready_out expression
READY = {
    "":   "skid_node_max_pool2d_ready & spatial_run",
    "_1": "skid_node_conv_200_ready & spatial_run",
    "_2": "skid_node_conv_202_ready & spatial_run",
    "_4": "skid_node_conv_208_ready & spatial_run",
    "_5": "skid_node_conv_210_ready & spatial_run",
    "_7": "skid_node_conv_214_ready & spatial_run",
    "_8": "skid_node_conv_216_ready & spatial_run",
    "_10": "skid_node_conv_220_ready & spatial_run",
    "_11": "skid_node_conv_222_ready & spatial_run",
    "_13": "skid_node_conv_228_ready & spatial_run",
    "_14": "skid_node_conv_230_ready & spatial_run",
    "_16": "skid_node_conv_234_ready & spatial_run",
    "_17": "skid_node_conv_236_ready & spatial_run",
    "_19": "skid_node_conv_240_ready & spatial_run",
    "_20": "skid_node_conv_242_ready & spatial_run",
    "_23": "skid_node_conv_248_ready & spatial_run",
    "_26": "skid_node_conv_256_ready & spatial_run",
    "_29": "skid_node_conv_262_ready & spatial_run",
    "_32": "skid_node_conv_268_ready & spatial_run",
    "_35": "skid_node_conv_274_ready & spatial_run",
    "_38": "skid_node_conv_280_ready & spatial_run",
    "_40": "node_conv_284_ready_in & spatial_run",
    "_43": "node_conv_292_ready_in & spatial_run",
    "_46": "node_conv_298_ready_in & spatial_run",
    "_48": "m_axis_tready",
}

txt = TOP.read_text()
patched = 0
for suf, expr in READY.items():
    mod = f"node_relu{suf}"
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
