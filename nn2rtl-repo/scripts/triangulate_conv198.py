#!/usr/bin/env python3
"""Localize the spatial per-OC bug via conv_198 (known-good stage-1, 1x1 IC=OC=64).
Recompute it in Python from its LOGICAL goldin (channel-order [3136,64], no tiling)
+ INT4 weights + per-OC scale, compare to the LOGICAL goldout. Decides:
  my==goldout -> golden is correct per-OC; the RTL (conv_datapath_mp_k) per-OC is buggy.
  my!=goldout -> scale-gen / golden / my-model issue (compare scale values too).
Mirrors the golden's requantize_tensor_with_scale_per_oc exactly.
"""
from __future__ import annotations
import struct, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from golden_impl import compute_scale_approx  # noqa: E402
import json
L = {l["module_id"]: l for l in json.load(open(ROOT / "output/layer_ir.json"))["layers"]}
MID = "node_conv_198"


def read_vec0(p, bps):
    raw = Path(p).read_bytes()
    m, v, nv, spv, b = struct.unpack("<4sIIII", raw[:20])
    assert b == bps, (b, bps)
    return np.frombuffer(raw[20:20 + spv * b], dtype=np.int8).reshape(spv, b).astype(np.int32)


def main():
    c = L[MID]; OC, IC = c["weight_shape"][0], c["weight_shape"][1]
    per_oc = c["scale_factor_per_oc"]
    gi = read_vec0(ROOT / f"output/goldens/{MID}.goldin", IC)        # [px, IC]
    gout = read_vec0(ROOT / f"output/goldens/{MID}.goldout", OC)     # [px, OC]
    w = np.array([int(x, 16) for x in (ROOT / f"output/weights/{MID}_weights.hex").read_text().split()],
                 dtype=np.int32).astype(np.int8).astype(np.int32).reshape(OC, IC)
    bp = ROOT / f"output/weights/{MID}_bias.hex"
    def _i32(x): v = int(x, 16); return v - (1 << 32) if v >= (1 << 31) else v
    bias = np.array([_i32(x) for x in bp.read_text().split()], dtype=np.int64) if bp.exists() else np.zeros(OC, np.int64)
    print(f"{MID}: OC={OC} IC={IC} px={gi.shape[0]} per_oc[0]={per_oc[0]:.5f}->{compute_scale_approx(per_oc[0])}")

    acc = gi.astype(np.int64) @ w.T.astype(np.int64) + bias[None, :]    # [px, OC]
    out = np.zeros_like(acc, dtype=np.int32)
    for oc in range(OC):
        mult, shift = compute_scale_approx(float(per_oc[oc]))
        rb = (1 << (shift - 1)) if shift > 0 else 0
        out[:, oc] = np.clip((acc[:, oc] * mult + rb) >> shift, -128, 127)

    d = np.abs(out - gout)
    nz = int((d != 0).sum())
    print(f"  my-recompute vs LOGICAL goldout: mismatch={nz}/{out.size} ({nz/out.size*100:.2f}%) max|err|={int(d.max())}")
    if nz:
        bad = np.argwhere(d != 0)[:5]
        for px, oc in bad:
            print(f"    px={px} oc={oc}: gold={gout[px,oc]} my={out[px,oc]} acc={acc[px,oc]} (mult,shift)={compute_scale_approx(float(per_oc[oc]))}")
    print("\n=> my==goldout: golden correct per-OC -> RTL conv_datapath_mp_k bug.\n"
          "   my!=goldout: scale-gen/golden/model issue.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
