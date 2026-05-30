#!/usr/bin/env python3
"""Quantify + VALIDATE the ImageNet-prediction impact of the 2.7% relu_48 residual.
Cross-checks two independent RTL dumps (probe + value-TB native), reports cosine/norms
(layout sanity: golden must be COHERENT = confident), and the top errored channels."""
import struct, glob
from pathlib import Path
import numpy as np, torch
from torchvision.models import resnet50, ResNet50_Weights
ROOT = Path(__file__).resolve().parent.parent
SCALE = 0.719477255513349

def gap(arr):  # [N] int8 bytes -> [2048] dequant GAP
    d = arr[:100352].astype(np.float32).reshape(49,64,32).reshape(49,2048)
    return torch.tensor(d.mean(0)*SCALE, dtype=torch.float32)
def from_bin(p): return gap(np.frombuffer(Path(p).read_bytes(),dtype=np.int8))
def from_gold(p):
    raw=Path(p).read_bytes(); _,nv,_,spv,bps=struct.unpack("<4sIIII",raw[:20])
    return gap(np.frombuffer(raw[20:20+spv*bps],dtype=np.int8))

gold = from_gold(glob.glob(str(ROOT/"output/goldens/contracts/node_relu_48_*/node_relu_48.goldout"))[0])
probe= from_bin(ROOT/"output/reports_integrated/verilator_nn2rtl_top_probe/probe_node_relu_48.bin")
val  = from_bin(ROOT/"output/reports_integrated/relu48_native_dump.bin")  # value-TB native dump (cross-check)

m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval(); fc=m.fc
def pred(f):
    with torch.no_grad(): l=fc(f)
    s=torch.sort(l,descending=True); return int(s.indices[0]), torch.topk(l,5).indices.tolist(), float(s.values[0]-s.values[1]), l
g1,g5,gm,gl = pred(gold)
print(f"probe==value-TB dump? {bool(torch.equal(probe,val))}  (both x0 vec0 -> should match)")
print(f"feat norms: ||gold||={torch.norm(gold):.2f}  ||probeRTL||={torch.norm(probe):.2f}  cos(gold,RTL)={torch.nn.functional.cosine_similarity(gold,probe,dim=0):.4f}")
print(f"GOLD top1={g1} margin={gm:.3f} top5={g5}   <- coherent/confident => layout OK")
for nm,f in [("probeRTL",probe),("valueRTL",val)]:
    p1,p5,pm,_=pred(f)
    print(f"{nm}: top1={p1} margin={pm:.3f} top5={p5}  TOP1{'==' if p1==g1 else '!='}gold")
# severity: how concentrated is the GAP error?
err=(probe-gold).abs(); top=torch.topk(err,8)
print(f"\nGAP per-channel |err| (dequant): max={err.max():.3f} mean={err.mean():.4f}  top8 channels={top.indices.tolist()} vals={[round(float(x),2) for x in top.values]}")
print(f"frac channels with |GAP err|>0.1: {float((err>0.1).float().mean())*100:.1f}%")
