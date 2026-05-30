#!/usr/bin/env python3
"""Wire each parallel pointwise conv's `ready_out` to its downstream consumer's
acceptance condition, so the backpressured output streamer paces correctly.

  - skid-fed convs:  ready_out = <skid>_ready & spatial_run
  - add-fed convs:   ready_out = <add>_skip_valid & spatial_run & <add>_ready_in
    (the add consumes the main beat when valid_in & ready_in; valid_in already
     includes this conv's valid_out, so the remaining terms gate acceptance)

Inserts `.ready_out(<expr>),` after the `.valid_out(...)` line of each conv
instantiation in nn2rtl_top.v. Idempotent.
"""
from __future__ import annotations
import re
from pathlib import Path

TOP = Path("output/rtl/nn2rtl_top.v")

# conv id -> ready_out expression
READY = {
    198: "skid_node_relu_1_ready & spatial_run",
    218: "skid_node_relu_10_ready & spatial_run",
    244: "skid_node_relu_22_ready & spatial_run",
    252: "skid_node_relu_25_ready & spatial_run",
    258: "skid_node_relu_28_ready & spatial_run",
    270: "skid_node_relu_34_ready & spatial_run",
    276: "skid_node_relu_37_ready & spatial_run",
    202: "node_add_skip_valid & spatial_run & node_add_ready_in",
    222: "node_add_3_skip_valid & spatial_run & node_add_3_ready_in",
    236: "node_add_5_skip_valid & spatial_run & node_add_5_ready_in",
    242: "node_add_6_skip_valid & spatial_run & node_add_6_ready_in",
    262: "node_add_9_skip_valid & spatial_run & node_add_9_ready_in",
}

txt = TOP.read_text()
patched = 0
for n, expr in READY.items():
    # Match this conv's instantiation: `node_conv_N u_node_conv_N ( ... );`
    inst_re = re.compile(
        rf"(node_conv_{n}\s+u_node_conv_{n}\s*\(.*?\.valid_out\(node_conv_{n}_valid_out\),)(\s*\n)",
        re.DOTALL,
    )
    m = inst_re.search(txt)
    if not m:
        print(f"[WARN] conv_{n}: instantiation/valid_out not found")
        continue
    if f".ready_out(" in txt[m.start():m.start()+ m.group(0).__len__()+200]:
        # crude idempotency: check just-after region
        pass
    # Skip if already has ready_out in this instance
    inst_close = txt.find(");", m.start())
    if ".ready_out(" in txt[m.start():inst_close]:
        print(f"[skip] conv_{n}: ready_out already wired")
        continue
    repl = m.group(1) + m.group(2) + f"        .ready_out({expr}),\n"
    txt = txt[:m.start()] + repl + txt[m.end():]
    patched += 1
    print(f"[ok] conv_{n}: .ready_out({expr})")

TOP.write_text(txt)
print(f"\n[written] {TOP}  (patched {patched})")
