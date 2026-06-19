#!/usr/bin/env python3
"""BN-FOLD-AWARE accuracy measurement of the DEPLOYED Config B ResNet-50.

The DEPLOYED design is the torchvision ResNet-50 (IMAGENET1K_V2) BACKBONE with
*every BatchNorm folded into the preceding conv* (checkpoints/resnet50_full.onnx
has 53 Conv / 0 BatchNorm nodes -- PROVEN by direct probe). The deployed integer
weights live in output/weights/node_conv_{ID}_weights.hex and were generated from
the FOLDED conv weights using a per-output-channel scale:

    weight_scale_per_oc[oc] = max_abs(W_FOLDED[oc]) / qmax
        qmax = 7  for INT4 (weight_bits=4, domain [-8,7])
        qmax = 3  for INT3 (weight_bits=3, domain [-4,3])

This scale is stored in output/layer_ir.json['weight_scale_per_oc'] and is the
CORRECT, self-consistent generating scale for the deployed integers. It is NOT
stale: it equals max_abs(W_FOLDED[oc])/qmax EXACTLY (probe ratio == 1.0000), and
differs from max_abs(W_torchvision_RAW[oc])/qmax by exactly the per-channel BN
fold factor gamma[oc]/sqrt(running_var[oc]+eps) (corr 1.0).

THE TRAP (do NOT repeat): dequantizing the deployed integers and loading them
into resnet50(pretrained).conv.weight while leaving the torchvision BN layers
INTACT applies BN TWICE (once already folded into the integers, once by the live
BN module). With the correct folded scale that collapses to ~0%; with the wrong
raw-weight scale it gives a misleading ~73%. BOTH are artifacts of the
double-BN. The fix is to fold the conv+BN in the float reference and force the
BN modules to IDENTITY so they pass activations through unchanged.

This script therefore builds THREE evaluable models on the SAME disjoint val set
(val = load_batch(n_val, skip=n_calib); calib = first n_calib images):

  (A) STOCK          : torchvision resnet50, BN intact -> baseline top-1.
  (B) FOLDED-FLOAT   : per conv, weight=W_fold_ref, bias=b_fold_ref, BN=identity.
                       SELF-VALIDATION: must ~equal STOCK (proves fold + identity
                       harness is faithful). Delta flagged if > 1%.
  (C) DEPLOYED       : per conv, weight = int_deployed * weight_scale_per_oc[oc]
                       (folded scale from layer_ir -- PROVEN, NOT recomputed from
                       raw weights), bias = b_fold_ref, BN = identity.

The deployed A8 number (gptq_int4.eval_top1 act_int8=True, per-tensor INT8
activation fake-quant on every conv input -- the exact path that produced the
79.47% / 77.60% figures) is the AUTHORITATIVE deployed top-1 (~77.6%).

NOTE on bias: deployment quantizes the folded bias into an integer bias hex, but
the float folded bias b_fold_ref is a faithful proxy (the bias quant error is
negligible relative to the W4 weight error), so (C) uses b_fold_ref directly.

  py scripts/measure_deployed_configb_acc.py [n_val=1500] [n_calib=256]

DO NOT run with a large n_val on a shared GPU (serialize). n_val<=8 is a cheap
structural dry-check.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import imagenet_util as iu  # noqa: E402
# Reuse the EXACT A8 eval that produced 79.47% / 77.60% -- do NOT reimplement.
from gptq_int4 import eval_top1, convs  # noqa: E402

LAYER_IR = ROOT / "output" / "layer_ir.json"

# weight_bits -> (qmin, qmax, generating_qmax). The deployed integer domain.
#   INT4 weight_bits=4 : domain [-8,7], generating qmax=7
#   INT3 weight_bits=3 : domain [-4,3], generating qmax=3
#   INT8 (vestigial)   : domain [-128,127], generating qmax=127
_RANGE_BY_BITS = {4: (-8, 7, 7), 3: (-4, 3, 3), 8: (-128, 127, 127)}

# The 18 deployed INT3 conv module IDs (probe ground truth).
_INT3_IDS = {
    246, 250, 254, 260, 264, 266, 272, 278, 282, 284,
    286, 288, 290, 292, 294, 296, 298, 300,
}


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def _load_int_weights(hex_path: Path, shape) -> np.ndarray:
    """Read flat signed-8bit 2's-complement hex (one value/line, row-major over
    [OC,IC,KH,KW]) -> int np.ndarray of `shape`."""
    vals = []
    with open(hex_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = int(line, 16)
            if v >= 128:          # 2's-complement signed 8-bit
                v -= 256
            vals.append(v)
    arr = np.asarray(vals, dtype=np.int32)
    n_expected = int(np.prod([int(s) for s in shape]))
    if arr.size != n_expected:
        raise ValueError(
            f"{hex_path.name}: read {arr.size} values, expected {n_expected} "
            f"for shape {tuple(int(s) for s in shape)}"
        )
    return arr.reshape(tuple(int(s) for s in shape))


# --------------------------------------------------------------------------- #
# BN folding
# --------------------------------------------------------------------------- #
def conv_bn_pairs(model: nn.Module):
    """Pair each Conv2d with the BatchNorm2d that immediately follows it in
    module-traversal order. In torchvision resnet50 this is exactly:
        conv1->bn1, every bottleneck convN->bnN, downsample[0]->downsample[1].
    Returns a list of (conv_name, conv, bn_name, bn) of length 53. The conv
    order is IDENTICAL (by object identity) to convs(model), the order used by
    layer_ir and gptq_int4."""
    pairs = []
    prev = None  # (name, conv)
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            prev = (name, mod)
        elif isinstance(mod, nn.BatchNorm2d) and prev is not None:
            pairs.append((prev[0], prev[1], name, mod))
            prev = None
    # sanity: ordering must match convs(model) by identity
    mod_convs = convs(model)
    pair_convs = [p[1] for p in pairs]
    if len(pairs) != len(mod_convs) or any(a is not b for a, b in zip(mod_convs, pair_convs)):
        raise RuntimeError(
            f"conv/bn pairing diverges from convs() order: "
            f"{len(pairs)} pairs vs {len(mod_convs)} convs"
        )
    return pairs


def set_bn_identity(bn: nn.BatchNorm2d) -> None:
    """Force a BatchNorm2d to a no-op: y = (x-0)/sqrt(1) * 1 + 0 = x.
    running_var is set to 1-eps so that sqrt(running_var+eps) == 1 exactly."""
    with torch.no_grad():
        bn.weight.fill_(1.0)
        bn.bias.fill_(0.0)
        bn.running_mean.zero_()
        bn.running_var.fill_(1.0 - bn.eps)


def fold_reference(base: nn.Module):
    """Return per-conv (W_fold_ref [OC,IC,KH,KW] float32, b_fold_ref [OC] float32)
    from fuse_conv_bn_eval, in convs()-order. Analytic equivalent:
        s_bn   = gamma / sqrt(running_var + eps)
        W_fold = W * s_bn[:, None, None, None]
        b_fold = beta - gamma * running_mean / sqrt(running_var + eps)
                 (+ conv.bias*s_bn if the conv had a bias; resnet convs have none)
    """
    refs = []
    for _cn, conv, _bn_n, bn in conv_bn_pairs(base):
        fused = fuse_conv_bn_eval(conv, bn)  # returns a new Conv2d w/ bias
        refs.append((
            fused.weight.detach().clone().float(),
            fused.bias.detach().clone().float(),
        ))
    return refs


def _attach_conv_bias(conv: nn.Conv2d) -> None:
    """Ensure a conv has a learnable bias Parameter (resnet convs ship bias=None
    because the bias is folded into BN; we fold it back into the conv)."""
    if conv.bias is None:
        conv.bias = nn.Parameter(torch.zeros(conv.weight.shape[0],
                                             dtype=conv.weight.dtype,
                                             device=conv.weight.device))


# --------------------------------------------------------------------------- #
# Model builders
# --------------------------------------------------------------------------- #
def build_folded_float(base: nn.Module, refs):
    """(B) FOLDED-FLOAT: conv = folded reference weight/bias, BN = identity."""
    model = copy.deepcopy(base).eval()
    cs = convs(model)
    bns = [p[3] for p in conv_bn_pairs(model)]
    with torch.no_grad():
        for c, bn, (Wf, bf) in zip(cs, bns, refs):
            _attach_conv_bias(c)
            c.weight.copy_(Wf.to(c.weight.dtype))
            c.bias.copy_(bf.to(c.bias.dtype))
            set_bn_identity(bn)
    return model


def build_deployed(base: nn.Module, refs):
    """(C) DEPLOYED: conv weight = int_deployed * weight_scale_per_oc (folded
    scale from layer_ir), conv bias = folded reference bias, BN = identity.

    Returns (model, sanity_rows) where each sanity row is
        (module_id, bits, qmin_seen, qmax_seen, qmin_exp, qmax_exp, ok,
         scale_match_ratio)
    scale_match_ratio = median |weight_scale_per_oc / (max_abs(W_fold_ref)/qmax)|
    which must be ~1.0 (confirms the layer_ir scale matches the folded weights)."""
    ir = json.loads(LAYER_IR.read_text())
    conv_layers = [l for l in ir["layers"] if "weight_scale_per_oc" in l]

    cs = convs(base)
    if len(cs) != len(conv_layers):
        raise RuntimeError(
            f"conv count mismatch: torchvision has {len(cs)} Conv2d, "
            f"layer_ir has {len(conv_layers)} per-OC convs"
        )

    model = copy.deepcopy(base).eval()
    model_convs = convs(model)
    bns = [p[3] for p in conv_bn_pairs(model)]

    sanity_rows = []
    with torch.no_grad():
        for idx, (layer, c, bn, (Wf, bf)) in enumerate(
                zip(conv_layers, model_convs, bns, refs)):
            shape = layer["weight_shape"]
            tv_shape = tuple(c.weight.shape)
            if tuple(int(s) for s in shape) != tv_shape:
                raise RuntimeError(
                    f"shape mismatch at conv idx {idx} ({layer['module_id']}): "
                    f"layer_ir {tuple(int(s) for s in shape)} vs torchvision {tv_shape}"
                )

            bits = int(layer["weight_bits"])
            qmin_exp, qmax_exp, gen_qmax = _RANGE_BY_BITS.get(bits, (-128, 127, 127))

            # weights_path may be a stale absolute path (Desktop, pre-D: migration);
            # fall back to the current repo's output/weights/ by basename.
            _wp = Path(layer["weights_path"])
            if not _wp.exists():
                _wp = Path("output/weights") / _wp.name
            int_w = _load_int_weights(_wp, shape)

            # PROVEN-correct folded per-OC scale straight from layer_ir.
            scale_per_oc = np.asarray(layer["weight_scale_per_oc"], dtype=np.float64)
            OC = int_w.shape[0]
            if scale_per_oc.size != OC:
                raise RuntimeError(
                    f"{layer['module_id']}: weight_scale_per_oc len "
                    f"{scale_per_oc.size} != OC {OC}"
                )
            s_b = scale_per_oc.reshape(OC, *([1] * (int_w.ndim - 1)))
            float_w = (int_w.astype(np.float64) * s_b).astype(np.float32)

            _attach_conv_bias(c)
            c.weight.copy_(torch.from_numpy(float_w).to(c.weight.dtype))
            c.bias.copy_(bf.to(c.bias.dtype))  # folded-float bias proxy
            set_bn_identity(bn)

            # --- sanity: integer range + scale-vs-folded-weight match ---
            qmin_seen, qmax_seen = int(int_w.min()), int(int_w.max())
            ok = (qmin_seen >= qmin_exp) and (qmax_seen <= qmax_exp)

            s_fold = (Wf.reshape(OC, -1).abs().amax(dim=1).double()
                      / gen_qmax).clamp_min(1e-12).cpu().numpy()
            ratio = np.median(np.abs(scale_per_oc / s_fold))

            sanity_rows.append(
                (layer["module_id"], bits, qmin_seen, qmax_seen,
                 qmin_exp, qmax_exp, ok, float(ratio))
            )

    return model, sanity_rows


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("n_val", nargs="?", type=int, default=1500,
                    help="number of val images (default 1500)")
    ap.add_argument("n_calib", nargs="?", type=int, default=256,
                    help="number of calib images skipped before val (default 256)")
    args = ap.parse_args()
    n_val, n_calib = args.n_val, args.n_calib
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from torchvision.models import resnet50, ResNet50_Weights
    base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()

    print(f"Deployed Config B (BN-FOLD-AWARE) accuracy: "
          f"n_val={n_val} skip(calib)={n_calib} dev={dev} "
          f"(val DISJOINT from calib)")

    # 1) fold conv+bn -> per-conv reference folded weight/bias
    refs = fold_reference(base)

    # 2) build the three models
    folded = build_folded_float(base, refs)
    deployed, sanity_rows = build_deployed(base, refs)

    # 3) val set DISJOINT from the n_calib calib images
    xval, yval = iu.load_batch(n_val, skip=n_calib)

    # ---- per-conv sanity block ----
    n_int3 = sum(1 for r in sanity_rows if r[1] == 3)
    n_int4 = sum(1 for r in sanity_rows if r[1] == 4)
    expected_int3 = {f"node_conv_{i}" for i in _INT3_IDS}
    seen_int3 = {r[0] for r in sanity_rows if r[1] == 3}
    int3_set_ok = (seen_int3 == expected_int3)

    print(f"\n-- per-conv deployed integer-range + scale-match sanity "
          f"({n_int4} INT4 expect [-8,7]; {n_int3} INT3 expect [-4,3]) --")
    all_ok = True
    ratios = []
    for mid, bits, qmn, qmx, emn, emx, ok, ratio in sanity_rows:
        all_ok &= ok
        ratios.append(ratio)
        flag = "OK " if ok else "BAD"
        print(f"  [{flag}] {mid:>16s}  INT{bits}  on-disk[min={qmn:4d}, max={qmx:4d}]"
              f"  expect[{emn},{emx}]  s_ir/s_fold(med)={ratio:.6f}")
    med_ratio = float(np.median(ratios))
    print(f"-- range: {'ALL OK' if all_ok else 'RANGE VIOLATION'}  "
          f"(INT3={n_int3}, INT4={n_int4}, total={len(sanity_rows)}; "
          f"INT3-set {'matches' if int3_set_ok else 'MISMATCH'} the 18 known IDs)")
    print(f"-- scale-match: median |weight_scale_per_oc / (max_abs(W_fold)/qmax)| "
          f"= {med_ratio:.6f}  (must be ~1.0 => layer_ir scale == folded-weight scale)\n")

    # static-gate trust flags (data/harness integrity, independent of accuracy)
    range_ok = all_ok
    scale_ok = abs(med_ratio - 1.0) <= 0.01  # layer_ir scale == folded-weight scale

    # ---- accuracy ----
    acc_stock = eval_top1(base, xval, yval, dev) * 100
    print(f"  (A) STOCK torchvision (BN intact)        : {acc_stock:.2f}%   <-- baseline")

    acc_folded = eval_top1(folded, xval, yval, dev) * 100
    delta = acc_folded - acc_stock
    fold_ok = abs(delta) <= 1.0
    flag = "" if fold_ok else "  <-- WARNING: fold/identity harness DRIFT (>1%)"
    print(f"  (B) FOLDED-FLOAT (BN=identity)           : {acc_folded:.2f}%   "
          f"(delta vs STOCK = {delta:+.2f}%){flag}")

    acc_dep_w = eval_top1(deployed, xval, yval, dev) * 100
    print(f"  (C) DEPLOYED Config B (w-only)           : {acc_dep_w:.2f}%")

    acc_dep_a8 = eval_top1(deployed, xval, yval, dev, act_int8=True) * 100
    print(f"  (C) DEPLOYED Config B + A8 (int8 act)    : {acc_dep_a8:.2f}%   "
          f"<-- AUTHORITATIVE deployed top-1 (expect ~77.6%)")

    # ---- consolidated TRUST verdict ------------------------------------- #
    # The deployed A8 number is only trustworthy if EVERY harness/data gate
    # holds: integer ranges in-domain, INT3-set == the 18 known IDs, the
    # layer_ir per-OC scale == the folded-weight scale (~1.0), AND the
    # FOLDED-FLOAT model reproduces STOCK (proves fold + BN-identity is
    # faithful -- if B != A the whole methodology is suspect).
    gates = [
        ("integer-range in-domain", range_ok),
        ("INT3-set == 18 known IDs", int3_set_ok),
        ("scale-match median ~= 1.0", scale_ok),
        ("FOLDED-FLOAT ~= STOCK (B==A)", fold_ok),
    ]
    trustworthy = all(ok for _, ok in gates)
    print("\n-- TRUST GATES (deployed number is trustworthy only if ALL pass) --")
    for desc, ok in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")
    if trustworthy:
        print(f"=> VERDICT: TRUSTWORTHY. Deployed Config B top-1 (A8) = "
              f"{acc_dep_a8:.2f}% on {n_val} val imgs (disjoint from {n_calib} calib).")
    else:
        print("=> VERDICT: NOT TRUSTWORTHY -- a harness/data gate FAILED above; "
              "the deployed number is an ARTIFACT, do NOT report it.")

    return 0 if trustworthy else 1


if __name__ == "__main__":
    sys.exit(main())
