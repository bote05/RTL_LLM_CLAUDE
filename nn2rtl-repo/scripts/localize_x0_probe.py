#!/usr/bin/env python3
"""Localize the real 2.7% residual using the HARDWARE-FAITHFUL (--x-initial 0) probe dumps.
Multiset-compares each probe_node_*.bin to that module's contract golden (order-invariant).
The first layer (in chain order) with a non-trivial multiset diff = the onset of the bug.
conv_200 should now be ~0% (locks the X-init artifact conclusion)."""
import struct, glob, os, re
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
PD = ROOT/"output/reports_integrated/verilator_nn2rtl_top_probe"

def gold(mid):
    for p in glob.glob(str(ROOT/f"output/goldens/contracts/{mid}_*/{mid}.goldout")) + [str(ROOT/f"output/goldens/{mid}.goldout")]:
        if os.path.exists(p):
            raw=Path(p).read_bytes(); _,nv,_,spv,bps=struct.unpack("<4sIIII",raw[:20])
            return np.frombuffer(raw[20:20+spv*bps],dtype=np.int8).astype(np.int32)
    return None

# chain order for readability
order = ["node_conv_196","node_max_pool2d","node_conv_198","node_conv_200","node_relu_3",
         "node_conv_206","node_conv_212","node_conv_244","node_conv_246","node_conv_248",
         "node_conv_250","node_conv_284","node_relu_48"]
rows=[]
for f in glob.glob(str(PD/"probe_node_*.bin")):
    mid = "node_"+re.sub(r"^probe_node_","",Path(f).stem)
    g = gold(mid)
    if g is None: continue
    c = np.frombuffer(Path(f).read_bytes(),dtype=np.int8).astype(np.int32)
    n=min(c.size,g.size); cc=c[:n]; gg=g[:n]
    md=int((np.sort(cc)!=np.sort(gg)).sum())
    pos=int((cc!=gg).sum())
    rows.append((mid, n, c.size, g.size, 100*md/n, 100*pos/n, int(np.abs(cc-gg).max())))

def key(r):
    m=re.search(r"\d+",r[0]); return (order.index(r[0]) if r[0] in order else 999, int(m.group()) if m else 0)
rows.sort(key=key)
print(f"{'layer':18s} {'n':>8s} {'multiset%':>10s} {'pos%':>8s} {'maxerr':>7s}   (HW-faithful x0 probe)")
for mid,n,cs,gs,ms,po,mx in rows:
    flag = "  <-- CLEAN" if ms<0.5 else ("  <<< ONSET?" if ms<20 else "")
    print(f"{mid:18s} {n:8d} {ms:9.2f}% {po:7.2f}% {mx:7d}{flag}")
