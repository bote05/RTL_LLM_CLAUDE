#!/usr/bin/env python3
"""H3 provenance: is conv_284's tiled-streaming golden coherent with on-disk W/scale?

For each oc:
  biased[:,oc] = conv_acc[:,oc] + bias[oc]   (recomputed from goldin, raster HWC, 3x3 s2 p1)
  golden_slope[oc] = least-squares slope of goldout[:,oc] vs biased[:,oc]   (zero-intercept)
  golden_corr[oc]  = corr(biased[:,oc], goldout[:,oc])
Compare golden_slope to:
  scale_eff[oc] from scale.mem (mult/2^shift)
  scale_factor_per_oc[oc] from layer_ir
Also recompute my[:,oc] via per-OC requant (mult,shift) and report byte match.

Cross-check identical pipeline for conv_200 (known byte-exact) for calibration.
"""
import struct, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("c:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from golden_impl import compute_scale_approx

L = {l["module_id"]: l for l in json.load(open(ROOT / "output/layer_ir.json"))["layers"]}

def i8frame0(p):
    raw = Path(p).read_bytes()
    _, nv, u, spv, bps = struct.unpack("<4sIIII", raw[:20])
    return np.frombuffer(raw[20:20 + spv * bps], dtype=np.int8).astype(np.int64)

def load_w(p):
    return np.array([int(x, 16) for x in Path(p).read_text().split()], dtype=np.int32).astype(np.int8).astype(np.int64)

def i32(x):
    v = int(x, 16)
    return v - (1 << 32) if v >= (1 << 31) else v

def load_bias(p):
    p = Path(p)
    return np.array([i32(x) for x in p.read_text().split()], dtype=np.int64) if p.exists() else None

def load_scalemem(p):
    """scale.mem: one hex word per line, (shift<<16)|mult, mult=bits[15:0], shift=bits[21:16]."""
    words = [int(x, 16) for x in Path(p).read_text().split()]
    mult = np.array([w & 0xFFFF for w in words], dtype=np.int64)
    shift = np.array([(w >> 16) & 0x3F for w in words], dtype=np.int64)
    return mult, shift

def conv_acc(inp_hwc, w_ocickhkw, IH, IW, OC, IC, KH, KW, SH, SW, PH, PW):
    OH = (IH + 2*PH - KH)//SH + 1
    OW = (IW + 2*PW - KW)//SW + 1
    pad = np.zeros((IH + 2*PH, IW + 2*PW, IC), np.int64)
    pad[PH:PH+IH, PW:PW+IW, :] = inp_hwc
    acc = np.zeros((OH*OW, OC), np.int64)
    for kh in range(KH):
        for kw in range(KW):
            patch = pad[kh:kh+SH*OH:SH, kw:kw+SW*OW:SW, :]   # [OH,OW,IC]
            acc += patch.reshape(OH*OW, IC) @ w_ocickhkw[:, :, kh, kw].T.astype(np.int64)
    return acc, OH, OW

def analyze(MID, IC, OC, IH, IW, KH, KW, SH, SW, PH, PW):
    print(f"\n{'='*70}\n{MID}  IC={IC} OC={OC} {IH}x{IW} k{KH}x{KW} s{SH} p{PH}\n{'='*70}")
    gin = i8frame0(ROOT / f"output/goldens/{MID}.goldin")
    gout = i8frame0(ROOT / f"output/goldens/{MID}.goldout")
    w = load_w(ROOT / f"output/weights/{MID}_weights.hex").reshape(OC, IC, KH, KW)
    bias = load_bias(ROOT / f"output/weights/{MID}_bias.hex")
    if bias is None: bias = np.zeros(OC, np.int64)
    # raster HWC input
    inp = gin.reshape(IH, IW, IC)
    acc, OH, OW = conv_acc(inp, w, IH, IW, OC, IC, KH, KW, SH, SW, PH, PW)
    biased = acc + bias[None, :]
    gold = gout.reshape(OH*OW, OC)
    print(f"  goldin range[{gin.min()},{gin.max()}] nonzero{(gin!=0).mean()*100:.1f}%  goldout range[{gout.min()},{gout.max()}]")
    print(f"  weights range[{w.min()},{w.max()}] (int4={w.min()>=-8 and w.max()<=7})  bias[:3]={bias[:3].tolist()}")
    print(f"  acc shape {acc.shape} OH={OH} OW={OW}  biased range[{biased.min()},{biased.max()}]")

    # scale.mem
    mult, shift = load_scalemem(ROOT / f"output/weights/{MID}_scale.mem")
    scale_eff_mem = mult.astype(np.float64) / (2.0**shift)
    spo = L[MID].get("scale_factor_per_oc")
    spo = np.array(spo, np.float64) if spo else None

    # recompute my via per-OC requant from scale.mem
    my = np.zeros_like(gold)
    for oc in range(OC):
        m, s = int(mult[oc]), int(shift[oc])
        my[:, oc] = np.clip((biased[:, oc]*m + (1 << (s-1))) >> s, -128, 127)
    pos_mm = (my != gold).mean()*100
    ms_mm = (np.sort(my.ravel()) != np.sort(gold.ravel())).mean()*100
    print(f"  my(scale.mem) vs goldout: pos_mismatch={pos_mm:.2f}% multiset_mismatch={ms_mm:.2f}% max|err|={int(np.abs(my-gold).max())}")

    # golden effective per-OC slope (zero-intercept LS) and corr
    slopes = np.zeros(OC); corrs = np.zeros(OC); ninter = np.zeros(OC)
    for oc in range(OC):
        x = biased[:, oc].astype(np.float64); y = gold[:, oc].astype(np.float64)
        denom = (x*x).sum()
        slopes[oc] = (x*y).sum()/denom if denom > 0 else 0.0
        if x.std() > 0 and y.std() > 0:
            corrs[oc] = np.corrcoef(x, y)[0, 1]
        else:
            corrs[oc] = np.nan
    good = ~np.isnan(corrs)
    print(f"  golden per-OC corr(biased,goldout): mean={np.nanmean(corrs):.4f} "
          f"median={np.nanmedian(corrs):.4f} min={np.nanmin(corrs):.4f} "
          f"frac<0.99={(corrs[good]<0.99).mean()*100:.1f}% frac<0.9={(corrs[good]<0.9).mean()*100:.1f}%")
    # compare slope to scale.mem eff and to layer_ir spo
    ratio_mem = slopes / np.where(scale_eff_mem==0, np.nan, scale_eff_mem)
    print(f"  golden_slope/scale_eff_mem: mean={np.nanmean(ratio_mem):.4f} median={np.nanmedian(ratio_mem):.4f} "
          f"(==1 if golden used scale.mem)  frac in[0.9,1.1]={np.nanmean((ratio_mem>0.9)&(ratio_mem<1.1))*100:.1f}%")
    if spo is not None:
        ratio_ir = slopes / np.where(spo==0, np.nan, spo)
        print(f"  golden_slope/scale_factor_per_oc(ir): mean={np.nanmean(ratio_ir):.4f} median={np.nanmedian(ratio_ir):.4f} "
              f"frac in[0.9,1.1]={np.nanmean((ratio_ir>0.9)&(ratio_ir<1.1))*100:.1f}%")
        # also: scale.mem eff vs ir spo
        rr = scale_eff_mem / np.where(spo==0, np.nan, spo)
        print(f"  scale_eff_mem/scale_factor_per_oc(ir): mean={np.nanmean(rr):.4f} median={np.nanmedian(rr):.4f}")
    # global slope (single tensor) as fallback diagnostic
    xg = biased.ravel().astype(np.float64); yg = gold.ravel().astype(np.float64)
    gslope = (xg*yg).sum()/(xg*xg).sum()
    print(f"  GLOBAL slope(goldout/biased)={gslope:.5f}  layer_ir scale_factor={L[MID].get('scale_factor'):.5f}")
    return dict(corrs=corrs, slopes=slopes, scale_eff_mem=scale_eff_mem, spo=spo, my=my, gold=gold, biased=biased)

if __name__ == "__main__":
    r200 = analyze("node_conv_200", 64, 64, 56, 56, 3, 3, 1, 1, 1, 1)
    r284 = analyze("node_conv_284", 512, 512, 14, 14, 3, 3, 2, 2, 1, 1)
