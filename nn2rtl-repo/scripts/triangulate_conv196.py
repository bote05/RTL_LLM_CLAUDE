#!/usr/bin/env python3
"""Faithful numpy recompute of conv_196 (the stem) to separate DATA correctness
from in-chain RTL timing bugs.

Replicates the RTL's exact integer arithmetic:
  acc[oc,y,x] = sum_{ic,kh,kw} img[ic, y*2-3+kh, x*2-3+kw] * w[oc,ic,kh,kw] + bias[oc]
  (mult,shift) = compute_scale_approx(scale_factor_per_oc[oc])
  out = clamp( (acc*mult + (1<<(shift-1))) >> shift , -128, 127)

Compares to the LOGICAL goldout (output/goldens/node_conv_196.goldout, [12544,64]).
- match + no saturation  => DATA (img/weights/bias/scale) is CORRECT; the in-chain
  7056 spurious 127s are an RTL TIMING/WINDOWING bug, not a data/scale bug.
- saturation / mismatch   => the data path (weights/scale/input) is wrong.
"""
from __future__ import annotations
import struct, sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from golden_impl import compute_scale_approx  # noqa: E402

MID = "node_conv_196"
IC, OC, KH, KW, SH, SW, PH, PW = 3, 64, 7, 7, 2, 2, 3, 3
IH = IW = 224
OH = OW = 112


def main() -> int:
    L = {l["module_id"]: l for l in json.load(open(ROOT / "output/layer_ir.json"))["layers"]}
    c = L[MID]
    per_oc = c["scale_factor_per_oc"]

    # --- input image from CONTRACT goldin (50176 beats x 32 bytes; first 3 = RGB int8) ---
    gi = (ROOT / f"output/goldens/contracts").glob(f"{MID}_*")
    gdir = sorted(gi)[0]
    raw = (gdir / f"{MID}.goldin").read_bytes()
    _, _, _, spv, bps = struct.unpack("<4sIIII", raw[:20])
    beats = np.frombuffer(raw[20:20 + spv * bps], dtype=np.int8).reshape(spv, bps)
    assert spv == IH * IW, (spv, IH * IW)
    img = beats[:, :IC].reshape(IH, IW, IC).transpose(2, 0, 1).astype(np.int64)  # [IC,H,W]
    print(f"img [{IC},{IH},{IW}] range [{img.min()},{img.max()}] #sat={(np.abs(img)==127).sum()}")

    # --- weights [OC,IC,KH,KW] from flat hex (int8 holding INT4 values) ---
    wv = [int(x, 16) for x in (ROOT / f"output/weights/{MID}_weights.hex").read_text().split()]
    w = np.array(wv, dtype=np.int32).astype(np.int8).astype(np.int64).reshape(OC, IC, KH, KW)
    print(f"weights {w.shape} range [{w.min()},{w.max()}] (INT4 expected [-8,7])")

    # --- bias [OC] int32 ---
    bp = ROOT / f"output/weights/{MID}_bias.hex"
    def _i32(x):
        v = int(x, 16); return v - (1 << 32) if v >= (1 << 31) else v
    bias = np.array([_i32(x) for x in bp.read_text().split()], dtype=np.int64) if bp.exists() else np.zeros(OC, np.int64)

    # --- conv (stride 2, pad 3) integer accumulate ---
    imgp = np.zeros((IC, IH + 2 * PH, IW + 2 * PW), dtype=np.int64)
    imgp[:, PH:PH + IH, PW:PW + IW] = img
    acc = np.zeros((OC, OH, OW), dtype=np.int64)
    for kh in range(KH):
        for kw in range(KW):
            patch = imgp[:, kh:kh + OH * SH:SH, kw:kw + OW * SW:SW]  # [IC,OH,OW]
            # sum over ic of w[:,:,kh,kw] (OC,IC) x patch (IC,OH,OW)
            acc += np.einsum("oi,ihw->ohw", w[:, :, kh, kw], patch)
    acc += bias[:, None, None]

    # --- per-OC integer requant (RTL-exact) ---
    out = np.zeros_like(acc, dtype=np.int64)
    for oc in range(OC):
        mult, shift = compute_scale_approx(float(per_oc[oc]))
        rb = (1 << (shift - 1)) if shift > 0 else 0
        out[oc] = np.clip((acc[oc] * mult + rb) >> shift, -128, 127)

    nsat = int((out == 127).sum() + (out == -128).sum())
    print(f"recompute out range [{out.min()},{out.max()}] #saturated={nsat}/{out.size}")

    # --- compare to LOGICAL goldout [12544,64] = [OH*OW, OC] ---
    go = (ROOT / f"output/goldens/{MID}.goldout").read_bytes()
    _, _, _, gspv, gbps = struct.unpack("<4sIIII", go[:20])
    gold = np.frombuffer(go[20:20 + gspv * gbps], dtype=np.int8).reshape(gspv, gbps)  # [12544,64]
    mine = out.transpose(1, 2, 0).reshape(OH * OW, OC)  # [12544,64]
    d = np.abs(mine.astype(np.int16) - gold.astype(np.int16))
    nz = int((d != 0).sum())
    print(f"vs logical goldout: mismatch {nz}/{mine.size} ({nz/mine.size*100:.2f}%) max|err|={int(d.max()) if nz else 0}")
    print(f"goldout range [{gold.min()},{gold.max()}] #sat={int((gold==127).sum()+(gold==-128).sum())}")
    if nz:
        bad = np.argwhere(d != 0)[:5]
        for px, oc in bad:
            print(f"   px={px} oc={oc}: gold={gold[px,oc]} mine={mine[px,oc]}")
    print("\n=> match+no-sat: DATA correct, in-chain 127s = RTL TIMING/WINDOWING bug.")
    print("   saturation/mismatch here: data path (weights/scale/input/order) wrong.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
