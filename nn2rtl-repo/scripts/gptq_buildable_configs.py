import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(ROOT/"scripts"))
import torch, torch.nn as nn, json
import gptq_int4 as g, imagenet_util as iu
from torchvision.models import resnet50, ResNet50_Weights
sched=json.load(open(ROOT/"output/rtl/nn2rtl_scheduler_schedule.json"))
eng_ids=set(d["module_id"] for d in sched["dispatches"])
ir=json.load(open(ROOT/"output/layer_ir.json"))
conv_mids=[l["module_id"] for l in ir["layers"] if l["module_id"].startswith("node_conv")]
dev="cuda"
base=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
nc,nv=256,1500
xcal,_=iu.load_batch(nc); xval,yval=iu.load_batch(nv,skip=nc)
convs=g.convs(base)
eng_idx={i for i,mid in enumerate(conv_mids) if mid in eng_ids}
spatial4={i for i,mid in enumerate(conv_mids) if mid in ("node_conv_284","node_conv_292","node_conv_298","node_conv_288")}
def mixed(int3:set):
    g.QMAX,g.QMIN=7,-8; m4=g.gptq_quant(base,xcal,dev,scale_mode="channel")
    g.QMAX,g.QMIN=3,-4; m3=g.gptq_quant(base,xcal,dev,scale_mode="channel")
    c4,c3=g.convs(m4),g.convs(m3)
    with torch.no_grad():
        for i,c in enumerate(c4):
            if i in int3: c.weight.copy_(c3[i].weight.data)
    del m3; return m4
tot=sum(c.weight.numel() for c in convs)
def rep(lbl,ids):
    w=(sum(convs[i].weight.numel() for i in ids)*3 + (tot-sum(convs[i].weight.numel() for i in ids))*4)/1e6
    m=mixed(ids); a=g.eval_top1(m,xval,yval,dev,act_int8=True)*100; del m
    print(f"  {lbl:42s}: wt={w:5.1f}Mbit  top-1={a:.2f}%")
print(f"float={g.eval_top1(base,xval,yval,dev)*100:.2f}%  (engine={len(eng_idx)} convs, spatial4={sorted(spatial4)})")
rep("A: 4 spatial INT3 only", spatial4)
rep("B: 4 spatial + 14 engine INT3", spatial4|eng_idx)
rep("C: engine INT3 only (uniform)", eng_idx)
