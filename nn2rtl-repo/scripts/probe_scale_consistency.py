#!/usr/bin/env python3
"""DECISIVE probe: is the DEPLOYED requant scale (scale_factor_per_oc -> scale.mem,
what the RTL actually applies) self-consistent with the deployed integer weights?

The RTL requant multiplies acc_int32 by scale_factor_per_oc[oc] (= input_scale *
weight_scale_used[oc] / output_scale). For the design to be ACCURATE, the implied
weight_scale_used[oc] = scale_factor_per_oc[oc] * output_scale / input_scale must
equal the GENERATING scale s[oc] = max_abs(W_ref[oc]) / qmax (the scale the integers
were quantized with). We compare:
  - weight_scale_per_oc  (layer_ir field; the reviewer says this is STALE)
  - weight_scale_implied (from scale_factor_per_oc; what the RTL REQUANT really uses)
  - s_gen = max_abs(W_tv[oc]) / qmax  (the true generating scale)
If weight_scale_implied ~= s_gen -> deployment is CORRECT (the 0% was a script artifact
using the stale field). If weight_scale_implied ~= stale field != s_gen -> the RTL
requant is BROKEN (accurate-on-paper byte-exact but wrong on images).
"""
import json, sys
from pathlib import Path
import numpy as np, torch
from torchvision.models import resnet50, ResNet50_Weights

ROOT = Path(__file__).resolve().parent.parent
IR = json.loads((ROOT / "output/layer_ir.json").read_text())

# layer list may be under a key; normalize to a list of layer dicts
layers = IR["layers"] if isinstance(IR, dict) and "layers" in IR else IR
by_id = {}
for L in layers:
    mid = L.get("module_id") or L.get("name") or L.get("id")
    if mid:
        by_id[mid] = L

m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
convs = [mod for mod in m.modules() if isinstance(mod, torch.nn.Conv2d)]

# conv order in layer_ir vs torchvision: build the ordered list of conv module_ids
conv_ids = [ (L.get("module_id") or "") for L in layers
             if (L.get("op_type") or L.get("op") or "").lower() in ("conv","conv2d") ]
print(f"layer_ir conv count={len(conv_ids)}  torchvision conv count={len(convs)}")

def show(mid):
    L = by_id.get(mid)
    if L is None:
        print(f"\n{mid}: NOT FOUND in layer_ir"); return
    # find this conv's torchvision index by position among conv_ids
    try: idx = conv_ids.index(mid)
    except ValueError: idx = None
    keys = [k for k in L.keys() if "scale" in k.lower() or "bits" in k.lower() or k in ("input_scale","output_scale")]
    wb = L.get("weight_bits", 4)
    qmax = 3 if wb == 3 else 7
    print(f"\n=== {mid}  (tv idx={idx}, weight_bits={wb}, qmax={qmax}) ===")
    print(f"   scale-ish keys: {keys}")
    insc = L.get("input_scale"); outsc = L.get("output_scale")
    wspo = L.get("weight_scale_per_oc"); sfpo = L.get("scale_factor_per_oc")
    print(f"   input_scale={insc}  output_scale={outsc}")
    def head(x):
        if x is None: return None
        a = np.asarray(x, dtype=np.float64).ravel()
        return a[:4].tolist()
    print(f"   weight_scale_per_oc[:4] = {head(wspo)}")
    print(f"   scale_factor_per_oc[:4] = {head(sfpo)}")
    if sfpo is not None and insc and outsc:
        sf = np.asarray(sfpo, dtype=np.float64).ravel()
        w_implied = sf * float(outsc) / float(insc)
        print(f"   weight_scale_IMPLIED[:4] (=sfpo*out/in) = {w_implied[:4].tolist()}")
    if idx is not None and idx < len(convs):
        W = convs[idx].weight.detach().numpy().astype(np.float64)  # [OC,IC,KH,KW]
        oc = W.shape[0]
        s_gen = np.abs(W.reshape(oc, -1)).max(1) / qmax
        print(f"   s_gen=max_abs(W_tv)/qmax [:4] = {s_gen[:4].tolist()}")
        if wspo is not None:
            ws = np.asarray(wspo, dtype=np.float64).ravel()[:oc]
            r = ws / s_gen
            print(f"   ratio weight_scale_per_oc / s_gen [:4] = {r[:4].tolist()}  (median={np.median(r):.3f})")
        if sfpo is not None and insc and outsc:
            r2 = w_implied[:oc] / s_gen
            print(f"   ratio weight_scale_IMPLIED / s_gen [:4] = {r2[:4].tolist()}  (median={np.median(r2):.3f})  <<< ==1.0 means RTL requant is CORRECT")

for mid in ["node_conv_196", "node_conv_246", "node_conv_284", "node_conv_288", "node_conv_300"]:
    show(mid)
