#!/usr/bin/env python3
"""Config-B-anchored accuracy-vs-BRAM sweep (EFFICIENT: builds GPTQ once).

Question being answered: starting from the DEPLOYED Config B (18 INT3 + 35 INT4,
~77.6% +A8, ~94% BRAM), if we demote MORE convs to INT3 (freeing BRAM), how much
top-1 do we lose and how much BRAM do we free? In particular: where does 75% land,
and how much BRAM is freed there?

Method (exact, no re-deploy):
  - Build all-INT4 (m4) and all-INT3 (m3) GPTQ models ONCE (per-channel, the proven
    79.47%/77.6% pipeline). The splice is EXACT because GPTQ here is non-sequential
    (per-layer independent Hessian) -- see gptq_mixed_sweep.py.
  - For each config, set each conv's weight to its INT3 copy if that conv index is in
    the config's int3 set, else its INT4 copy, in ONE reused working model. eval +A8.
  - Start from the deployed 18 INT3 (conv indices below), then add the largest-by-weight
    INT4 (all spatial, private ROMs) cumulatively -> max BRAM freed per conv added.

BRAM model: demoting a conv INT4->INT3 frees 1 bit/weight = numel/36864 RAMB36
(verified vs recon anchors: 1024x256x1x1=7.1, 128x128x3x3=4.0). Reported relative to
Config B (the 94% baseline) and as absolute total weight-ROM RAMB36.

Usage: gptq_configb_extend_sweep.py [n_calib=256] [n_val=1500]
Writes JSON results to output/reports_integrated/configb_acc_bram_sweep.json
"""
from __future__ import annotations
import sys, json, copy
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import torch
import gptq_int4 as g
import imagenet_util as iu
from torchvision.models import resnet50, ResNet50_Weights

R3 = (3, -4)   # INT3
R4 = (7, -8)   # INT4
BITS_PER_RAMB36 = 36864

# Deployed Config B INT3 conv-layer indices (module_ids 246..300 -> idx=(id-196)/2).
# 14 engine 1x1 + 4 spatial 3x3 (512x512). Verified in recon vs layer_ir.
CONFIG_B_INT3 = {25, 27, 29, 32, 34, 35, 38, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52}


def main() -> int:
    n_calib = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    n_val   = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    xcal, _    = iu.load_batch(n_calib)
    xval, yval = iu.load_batch(n_val, skip=n_calib)

    cs_base = g.convs(base)
    nconv = len(cs_base)
    numel = [c.weight.numel() for c in cs_base]
    total_w = sum(numel)

    # BRAM (weight ROM only), per conv, at a given bit-width.
    def bram_at(int3_set):
        return sum(numel[i] * (3 if i in int3_set else 4) for i in range(nconv)) / BITS_PER_RAMB36

    bram_allint4 = bram_at(set())
    bram_configb = bram_at(CONFIG_B_INT3)

    # INT4 convs (complement of Config B), ordered biggest-first = most BRAM freed per conv.
    int4_ids = [i for i in range(nconv) if i not in CONFIG_B_INT3]
    int4_ids.sort(key=lambda i: -numel[i])

    print(f"[setup] {nconv} convs, {total_w/1e6:.2f}M weights, n_calib={n_calib} n_val={n_val} dev={dev}")
    print(f"[setup] weight-ROM RAMB36: all-INT4={bram_allint4:.0f}  ConfigB(18 INT3)={bram_configb:.0f}  "
          f"all-INT3={bram_at(set(range(nconv))):.0f}")
    print(f"[setup] biggest INT4 convs to demote next (idx:freed-RAMB36): "
          + ", ".join(f"{i}:{numel[i]/BITS_PER_RAMB36:.1f}" for i in int4_ids[:14]))

    print("[build] all-INT4 GPTQ ...", flush=True)
    g.QMAX, g.QMIN = R4
    m4 = g.gptq_quant(base, xcal, dev, scale_mode="channel")
    print("[build] all-INT3 GPTQ ...", flush=True)
    g.QMAX, g.QMIN = R3
    m3 = g.gptq_quant(base, xcal, dev, scale_mode="channel")
    cs4, cs3 = g.convs(m4), g.convs(m3)

    # float / anchor reports
    g.QMAX, g.QMIN = R4
    acc_float = g.eval_top1(base, xval, yval, dev) * 100
    print(f"[anchor] float top-1 = {acc_float:.2f}%", flush=True)

    # one reused working model
    work = copy.deepcopy(m4)
    csw = g.convs(work)

    def eval_config(int3_set):
        with torch.no_grad():
            for i in range(nconv):
                src = cs3[i] if i in int3_set else cs4[i]
                csw[i].weight.copy_(src.weight.data)
        return g.eval_top1(work, xval, yval, dev, act_int8=True) * 100

    # Config list: all-INT4 anchor, Config B, then B + K biggest INT4 cumulatively, then all-INT3.
    add_steps = [0, 4, 8, 12, 18, 25, len(int4_ids)]
    configs = [("all-INT4 (0 INT3)", set())]
    for k in add_steps:
        s = set(CONFIG_B_INT3) | set(int4_ids[:k])
        name = "Config B (18 INT3)" if k == 0 else f"B + {k} biggest INT4 ({18+k} INT3)"
        if k == len(int4_ids):
            name = f"all-INT3 ({nconv})"
        configs.append((name, s))

    print(f"\n  {'config':<30} {'#INT3':>6} {'wt-RAMB36':>10} {'vs ConfigB':>11} {'top-1':>8}")
    results = []
    for name, s in configs:
        bram = bram_at(s)
        freed = bram_configb - bram   # +ve = freed below ConfigB baseline
        acc = eval_config(s)
        print(f"  {name:<30} {len(s):>6} {bram:>10.0f} {freed:>+11.1f} {acc:>7.2f}%", flush=True)
        results.append({"config": name, "n_int3": len(s),
                        "int3_ids": sorted(s),
                        "weight_ramb36": round(bram, 1),
                        "freed_vs_configb": round(freed, 1),
                        "top1": round(acc, 2)})

    out = {"n_calib": n_calib, "n_val": n_val, "float_top1": round(acc_float, 2),
           "bram_allint4": round(bram_allint4, 1), "bram_configb": round(bram_configb, 1),
           "note": "weight_ramb36 = weight-ROM RAMB36 only (numel*bits/36864); "
                   "freed_vs_configb = ConfigB weight-ROM minus this config's. "
                   "Accuracy is algorithmic GPTQ +A8 (calibrated to 79.47% all-INT4 / 77.6% Config B).",
           "results": results}
    outp = ROOT / "output" / "reports_integrated" / "configb_acc_bram_sweep.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
