#!/usr/bin/env python3
"""VALIDATE the accuracy pipeline: run the FULL float torchvision ResNet-50 on the
goldin image and check it agrees with (a) my golden-backbone->head prediction and
(b) reveals whether the RTL's prediction is a genuine flip. If FLOAT == GOLDEN, the
dequant/layout/head pipeline is correct and 91 is the real prediction."""
import struct, glob
from pathlib import Path
import numpy as np, torch
from torchvision.models import resnet50, ResNet50_Weights
ROOT = Path(__file__).resolve().parent.parent
IN_SCALE = 0.02079; OUT_SCALE = 0.719477255513349

# --- goldin (int8 real image) -> normalized CHW float ---
gp=glob.glob(str(ROOT/"output/goldens/contracts/node_conv_196_*/node_conv_196.goldin"))[0]
raw=Path(gp).read_bytes(); _,nv,_,spv,bps=struct.unpack("<4sIIII",raw[:20])
d=np.frombuffer(raw[20:20+spv*bps],dtype=np.int8).astype(np.float32).reshape(spv,bps)
img = d[:, :3].reshape(224,224,3) * IN_SCALE              # HWC normalized
chw = torch.tensor(img.transpose(2,0,1)[None], dtype=torch.float32)  # [1,3,224,224]

m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
with torch.no_grad():
    full = m(chw)[0]                                       # FULL float model on the image
ftop = torch.topk(full,5).indices.tolist(); fs=torch.sort(full,descending=True).values
print(f"FLOAT full-ResNet50 on goldin image: top1={ftop[0]} margin={float(fs[0]-fs[1]):.3f} top5={ftop}")

# --- golden + RTL backbone -> my GAP+head pipeline ---
def head_pred(arr):
    a=arr[:100352].astype(np.float32).reshape(49,64,32).reshape(49,2048)
    feat=torch.tensor(a.mean(0)*OUT_SCALE,dtype=torch.float32)
    with torch.no_grad(): l=m.fc(feat)
    s=torch.sort(l,descending=True); return int(s.indices[0]), float(s.values[0]-s.values[1])
def gold():
    p=glob.glob(str(ROOT/"output/goldens/contracts/node_relu_48_*/node_relu_48.goldout"))[0]
    r=Path(p).read_bytes(); _,nv,_,spv,bps=struct.unpack("<4sIIII",r[:20])
    return np.frombuffer(r[20:20+spv*bps],dtype=np.int8)
g1,gm = head_pred(gold())
r1,rm = head_pred(np.frombuffer(Path(ROOT/"output/reports_integrated/relu48_native_dump.bin").read_bytes(),dtype=np.int8))
print(f"GOLDEN backbone -> my head: top1={g1} margin={gm:.3f}")
print(f"RTL    backbone -> my head: top1={r1} margin={rm:.3f}")
print(f"\nPIPELINE VALID? float({ftop[0]}) == golden({g1}): {ftop[0]==g1}   (if True, my dequant/layout/head is correct)")
print(f"RTL flips vs golden: {r1!=g1}   vs float: {r1!=ftop[0]}")
print(f"\n=> {'PIPELINE VALIDATED; ' if ftop[0]==g1 else 'PIPELINE MISMATCH (layout/scale/head suspect) -> single-image result UNRELIABLE; '}"
      f"{'and the 2.7% RTL residual DOES change this real image prediction' if (ftop[0]==g1 and r1!=g1) else 'inconclusive on accuracy'}")
