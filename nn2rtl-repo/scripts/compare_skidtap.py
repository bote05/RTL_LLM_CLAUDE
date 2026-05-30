#!/usr/bin/env python3
"""Compare the two conv_200 taps (MOD = old combinational module-output probe; SKID =
registered value actually delivered to relu_2) against each other and the golden.
Resolves whether the 94% was a TAP artifact (MOD skewed, SKID clean) or real."""
import struct
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
def s8(b): return b-256 if b>=128 else b
def loadcap(p):
    out=[]
    for ln in Path(p).read_text().split():
        ln=ln.strip()
        if len(ln)!=64: continue
        pairs=[int(ln[k:k+2],16) for k in range(0,64,2)]
        out.append([s8(b) for b in pairs[::-1]])
    return np.array(out,dtype=np.int32)
mod = loadcap(ROOT/"output/reports_integrated/conv200_cap_MOD.hex")
skid= loadcap(ROOT/"output/reports_integrated/conv200_cap_SKID.hex")
raw=(ROOT/"output/goldens/node_conv_200.goldout").read_bytes()
_,nv,_,spv,bps=struct.unpack("<4sIIII",raw[:20])
g=np.frombuffer(raw[20:20+spv*bps],dtype=np.int8).astype(np.int32)
print(f"cap_MOD beats={mod.shape[0]}  cap_SKID beats={skid.shape[0]}")
# 1) do the two taps AGREE (same logical conv_200 output, in order)?
n=min(mod.shape[0],skid.shape[0]); a=mod[:n].ravel(); b=skid[:n].ravel()
print(f"\n[1] MOD vs SKID (same beats, direct): seq-diff={int((a!=b).sum())}/{a.size} ({100*(a!=b).mean():.2f}%)  => {'TAPS DISAGREE (one is skewed)' if (a!=b).any() else 'taps identical'}")
# 2,3) each vs golden (multiset, order-invariant)
for nm,c in [("MOD",mod),("SKID",skid)]:
    v=c.ravel(); N=min(v.size,g.size); vv=v[:N]; gg=g[:N]
    md=int((np.sort(vv)!=np.sort(gg)).sum())
    print(f"[{nm}] vs golden(first {N}): multiset-diff={md}/{N} ({100*md/N:.2f}%)  mean={vv.mean():.2f} (gold {gg.mean():.2f})")
print("\nVERDICT: if SKID multiset-diff << MOD (e.g. ~0 vs ~94%) => conv_200's DELIVERED output is correct;")
print("         the old 94% was a module-output TAP artifact, and conv_200 is fine. Bisect downstream next.")
