#!/usr/bin/env python3
"""Per-node probe-vs-golden comparison that selects the CORRECT contract dir by
layer_ir contract_id (NOT glob[0], which can grab the stale dram-backed-weights
dir — the bug that inflated conv_284 to '95.5%'). Used to analyze the conv_252/266/282
bisection of the late-engine corruption (workflow wm9trddo2)."""
import struct, glob, json, os
from pathlib import Path
import numpy as np

ROOT = Path(".")
IR = {l["module_id"]: l for l in json.loads((ROOT/"output/layer_ir.json").read_text())["layers"]}
PDIR = "output/reports_integrated/verilator_nn2rtl_top_probe"

def correct_goldout(mod):
    """Select contract goldout dir matching layer_ir contract_id."""
    cid = IR.get(mod, {}).get("contract_id", "")
    dirs = sorted(glob.glob(f"output/goldens/contracts/{mod}_*/"))
    if not dirs: return None
    # prefer dir whose name contains the live contract_id
    pick = [d for d in dirs if cid and cid in Path(d).name]
    d = (pick or dirs)[0]
    gp = Path(d)/f"{mod}.goldout"
    if not gp.exists(): return None
    raw = gp.read_bytes(); _,nv,_,spv,bps = struct.unpack("<4sIIII", raw[:20])
    return np.frombuffer(raw[20:20+spv*bps], dtype=np.int8).astype(np.int32), cid, Path(d).name

def cmp(mod):
    p = f"{PDIR}/probe_{mod}.bin"
    if not os.path.exists(p): return f"{mod:16s}: NO PROBE BIN"
    cap = np.frombuffer(Path(p).read_bytes(), dtype=np.int8).astype(np.int32)
    g = correct_goldout(mod)
    if g is None: return f"{mod:16s}: NO GOLDEN"
    gold, cid, dname = g
    n = min(cap.size, gold.size)
    pos = 100*(cap[:n]!=gold[:n]).mean()
    ms = 100*(np.sort(cap[:n])!=np.sort(gold[:n])).mean()
    d = np.abs(cap[:n]-gold[:n])
    flag = " ok" if ms<2 else "  <== OFF"
    return f"{mod:16s}: POS {pos:5.1f}% MULTI {ms:5.1f}% max|d|={int(d.max()):3d} mean|d|={d.mean():.2f}{flag}  [{cid}]"

print("=== BISECTION: late-engine corruption onset (correct contract_id golden selection) ===")
# frontier (known byte-exact) -> bisection taps -> conv_284 -> final
for m in ["node_conv_248","node_conv_250","node_conv_252","node_conv_266","node_conv_282","node_conv_284","node_relu_48"]:
    print("  "+cmp(m))
print("\nDECISION: c252 clean + c266/c282 off => late-engine-dispatch compute onset")
print("          c252 off                    => engine->spatial bridge / frontier")
print("          c282 clean but conv_284 off => relu_40/skid handshake into conv_284")
