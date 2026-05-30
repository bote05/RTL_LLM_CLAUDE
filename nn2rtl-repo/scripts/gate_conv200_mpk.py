#!/usr/bin/env python3
"""DECISIVE GATE TEST for conv_200.

Recompute conv_200 output from its golden INPUT and the EXACT weight file the
RTL reads (node_conv_200.v -> .WEIGHTS_PATH = node_conv_200_weights_mp_k_9.hex),
then compare to node_conv_200.goldout.

Steps:
 (a) load node_conv_200.goldin frame0 -> [56,56,64] HWC int input.
 (b) load mp_k_9 file, UN-PERMUTE its packing back to [oc=64,ic=64,kh=3,kw=3].
     VALIDATE the un-permutation == plain node_conv_200_weights.hex (multiset
     AND elementwise).
 (c) conv2d 3x3 pad1 stride1 + bias + per-OC requant. Compare to goldout.
"""
from __future__ import annotations
import struct, sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from golden_impl import compute_scale_approx  # noqa: E402

MID = "node_conv_200"
IC, OC, IH, IW, KH, KW = 64, 64, 56, 56, 3, 3
K_TOTAL = IC * KH * KW
MP, MP_K = 16, 9                     # from node_conv_200.v localparams
OC_PASSES = (OC + MP - 1) // MP      # 4
K_GROUPS = K_TOTAL // MP_K           # 64

L = {l["module_id"]: l for l in json.load(open(ROOT / "output/layer_ir.json"))["layers"]}


def sbyte(b):
    return b - 256 if b >= 128 else b


def load_plain_i32(p):
    return np.array([sbyte(int(x, 16)) for x in Path(p).read_text().split()], dtype=np.int32)


def i32(x):
    v = int(x, 16)
    return v - (1 << 32) if v >= (1 << 31) else v


def frame0(tag):
    raw = (ROOT / f"output/goldens/{MID}.{tag}").read_bytes()
    _, nv, _, spv, bps = struct.unpack("<4sIIII", raw[:20])
    return np.frombuffer(raw[20:20 + spv * bps], dtype=np.int8).astype(np.int32)


def unpermute_mpk(path):
    """Decode the conv_datapath_mp_k wide packing back to flat [oc][ic][kh][kw].

    Packing (scripts/repack_weights_wide.write_wide_weights):
      line index i = g*K_GROUPS + kg  (g=oc_group, kg=k_group)
      word bit [(lane*MP_K + kpos)*8 +: 8] = weight at
        (global_oc = g*MP + lane, k_lin = kg*MP_K + kpos),
      flat index = global_oc * K_TOTAL + k_lin.
    Byte 0 (least-significant) of the hex string is the LAST 2 hex chars.
    """
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    assert len(lines) == OC_PASSES * K_GROUPS, f"{len(lines)} words != {OC_PASSES*K_GROUPS}"
    flat = np.zeros(OC * K_TOTAL, dtype=np.int32)
    nbytes = MP * MP_K
    for i, hexstr in enumerate(lines):
        assert len(hexstr) == nbytes * 2, f"word {i} len {len(hexstr)} != {nbytes*2}"
        g = i // K_GROUPS
        kg = i % K_GROUPS
        # byte position p (0=LSB) occupies hex chars [-(p+1)*2 : len-p*2]
        for p in range(nbytes):
            lo = len(hexstr) - (p + 1) * 2
            b = int(hexstr[lo:lo + 2], 16)
            lane = p // MP_K
            kpos = p % MP_K
            global_oc = g * MP + lane
            k_lin = kg * MP_K + kpos
            if global_oc < OC and k_lin < K_TOTAL:
                flat[global_oc * K_TOTAL + k_lin] = sbyte(b)
    return flat


def main():
    inp = frame0("goldin").reshape(IH, IW, IC)
    gold = frame0("goldout").reshape(IH * IW, OC)

    mpk_path = ROOT / "output/weights/node_conv_200_weights_mp_k_9.hex"
    plain_path = ROOT / "output/weights/node_conv_200_weights.hex"

    w_unperm = unpermute_mpk(mpk_path)                 # flat from mp_k file
    w_plain = load_plain_i32(plain_path)               # flat plain file
    assert w_unperm.size == OC * K_TOTAL, w_unperm.size
    assert w_plain.size == OC * K_TOTAL, w_plain.size

    ms_equal = bool(np.array_equal(np.sort(w_unperm), np.sort(w_plain)))
    elem_equal = bool(np.array_equal(w_unperm, w_plain))
    n_elem_mismatch = int((w_unperm != w_plain).sum())
    print(f"[UNPERMUTE VALIDATE] mp_k_9 un-permuted vs PLAIN:")
    print(f"  multiset_equal={ms_equal}  elementwise_equal={elem_equal}  elem_mismatch={n_elem_mismatch}")

    # value range / int4 check (use the file the RTL reads = mp_k_9)
    is_int4 = bool(w_unperm.min() >= -8 and w_unperm.max() <= 7)
    print(f"  mp_k_9 value range [{w_unperm.min()},{w_unperm.max()}]  int4(all in [-8,7])? {is_int4}")

    # recompute using the EXACT file the RTL reads (un-permuted mp_k_9)
    w = w_unperm.reshape(OC, IC, KH, KW)               # [oc,ic,kh,kw]

    bp = ROOT / "output/weights/node_conv_200_bias.hex"
    bias = np.array([i32(x) for x in bp.read_text().split()], dtype=np.int64) if bp.exists() else np.zeros(OC, np.int64)

    spo = L[MID].get("scale_factor_per_oc")
    print(f"  per_oc_scale={'YES' if spo else 'NO (tensor)'}  bias[:3]={bias[:3].tolist()}")

    pad = np.zeros((IH + 2, IW + 2, IC), np.int64); pad[1:-1, 1:-1, :] = inp
    acc = np.zeros((IH, IW, OC), np.int64)
    for kh in range(KH):
        for kw in range(KW):
            acc += pad[kh:kh + IH, kw:kw + IW, :] @ w[:, :, kh, kw].T.astype(np.int64)
    acc = acc.reshape(IH * IW, OC) + bias[None, :]

    my = np.zeros((IH * IW, OC), np.int32)
    if spo:
        for oc in range(OC):
            mult, shift = compute_scale_approx(float(spo[oc]))
            my[:, oc] = np.clip((acc[:, oc] * mult + (1 << (shift - 1))) >> shift, -128, 127)
    else:
        mult, shift = compute_scale_approx(float(L[MID]["scale_factor"]))
        my = np.clip((acc * mult + (1 << (shift - 1))) >> shift, -128, 127)

    d = np.abs(my - gold)
    mismatch = int((d != 0).sum())
    total = int(my.size)
    maxerr = int(d.max())
    out_ms_equal = bool(np.array_equal(np.sort(my.ravel()), np.sort(gold.ravel())))
    golden_consistent = (mismatch == 0)

    print(f"\n[GATE] recompute(mp_k_9 file) vs GOLDOUT:")
    print(f"  mismatch_count={mismatch}/{total}  max_abs_err={maxerr}")
    print(f"  multiset_equal(out)={out_ms_equal}")
    print(f"  golden_consistent={golden_consistent}")

    # JSON summary line for the harness
    print("RESULT_JSON " + json.dumps({
        "weights_file_used": str(mpk_path),
        "weights_file_is_int4": is_int4,
        "unperm_multiset_equal_plain": ms_equal,
        "unperm_elementwise_equal_plain": elem_equal,
        "mismatch_count": mismatch,
        "total": total,
        "max_abs_err": maxerr,
        "out_multiset_equal": out_ms_equal,
        "golden_consistent": golden_consistent,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
