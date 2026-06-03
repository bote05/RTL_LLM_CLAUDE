#!/usr/bin/env python3
"""BN-FOLD-AWARE accuracy measurement of the DEPLOYED MobileNetV2 INT8.

Mirrors the proven ResNet harness scripts/measure_deployed_configb_acc.py, but
adapted to the PROVEN MobileNetV2 recon facts:

  - checkpoints/mobilenet_v2.onnx has 52 Conv / 0 BatchNorm nodes => every BN is
    already FOLDED into the preceding conv weights (the ConvBNActivation blocks
    in torchvision become a single conv at export). This is the SAME trap as
    ResNet: dequantizing the deployed integers into a stock torchvision
    mobilenet_v2 (whose BN modules are still LIVE) double-applies BN -> garbage.
    The fix is to fold conv+BN in the float reference (fuse_conv_bn_eval) and
    force every BN module to IDENTITY so it passes activations through unchanged.

  - Deployed quant family = int8_symmetric_per_tensor (domain [-128,127],
    generating qmax=127). UNLIKE ResNet, the weight scale is PER-TENSOR (one
    float scale per layer, NOT per-output-channel):
        w_float = int8_weight * weight_scale_per_tensor
    where the PROVEN generating weight scale is
        weight_scale_per_tensor = max_abs(W_FOLDED) / 127.
    PROBE RESULT (do NOT regress): round(W_FOLDED / (max_abs(W_FOLDED)/127))
    clamped to [-128,127] reproduces the deployed integers EXACTLY in 51/52 conv
    layers (the 52nd differs in a single rounding-tie element, 6e-6 of the
    tensor), and corr(int_w, W_FOLDED)=1.000 in EVERY layer. The layer_ir
    "scale_factor" field is the ACTIVATION/output requant scale used by the
    fixed-point RTL datapath -- it is NOT the weight-dequant scale (for the
    convs it differs by ~20x), so this harness derives the weight scale from the
    folded reference, not from scale_factor.

  - Deployed int weights: output/mobilenet-v2/weights/node_conv_{ID}_weights.hex
    (flat signed-8bit 2's-complement, row-major over [OC,IC,KH,KW]).

  - The ONNX INCLUDES the classifier (Gemm [1000,1280]), so the deployed fc
    WEIGHTS are ALSO int8-quantized per-tensor; this harness dequants the deployed
    fc integers too (preferred, faithful) so model (C) has the FULL deployed
    weight set, not a hybrid with a float head. PROBE RESULT (do NOT regress): the
    fc weight scale is max_abs(W_fc)/127 (~0.003929), which round-trips the
    deployed fc integers ELEMENT-EXACT (clamp(round(W_fc/scale))==int_fc at
    0.999998, corr 1.000) -- the SAME max_abs/127 rule as the convs, so the fc
    weight scale is derived that way for uniformity. NOTE: the layer_ir
    scale_factor for the gemm (~0.003882) is the ACTIVATION/output requant scale,
    NOT this weight scale (they differ ~1.2%; using scale_factor reproduces only
    ~88.6% of the fc integers), so this harness does NOT use scale_factor for the
    fc weight -- exactly as it does not for the convs.
    CAVEAT on the fc ACTIVATION: the deployed fc input is int8-quantized in the
    RTL, but the A8 fake-quant path (gptq_int4.eval_top1, mirrored from the proven
    ResNet harness) hooks ONLY nn.Conv2d inputs -- the classifier INPUT activation
    is left float under +A8. So (C)+A8 is faithful in the fc WEIGHTS but treats
    the fc INPUT as float (a small, documented approximation consistent with the
    trusted ResNet 77.6% methodology).

torchvision MobileNetV2 has 52 Conv2d + 52 BatchNorm2d (perfect 1:1 conv->BN
pairing in module-traversal order) + 1 Linear[1000,1280] classifier. Conv order
(by object identity / shape) is IDENTICAL to the layer_ir conv order and to
convs(model) -- verified: 0 shape mismatches across all 52. The reference
checkpoint is IMAGENET1K_V2 (float top-1 72.154%): corr(int_w, W_FOLDED_V2)
=1.000 every layer, whereas V1 gives corr ~0 -- V2 is the deployed source (the
"~71.9%" hint was loose; V2's 72.15% is the actual match). PROVEN by probe, do
NOT switch back to V1.

This script builds THREE evaluable models on the SAME disjoint val set
(val = load_batch(n_val, skip=n_calib); calib = first n_calib images):

  (A) STOCK          : torchvision mobilenet_v2 (V2), BN intact -> baseline top-1
                       (~72.2%).
  (B) FOLDED-FLOAT   : per conv weight=W_fold_ref, bias=b_fold_ref, BN=identity.
                       SELF-VALIDATION: must ~equal STOCK (proves the fold +
                       BN-identity harness is faithful). Delta flagged if > 1%.
  (C) DEPLOYED       : per conv weight = int_deployed * (max_abs(W_fold)/127)
                       (PER-TENSOR), bias = b_fold_ref, BN = identity; AND the
                       classifier weight = int_fc_deployed * (max_abs(W_fc)/127).
                       Reported w-only and +A8 (per-tensor int8 activation
                       fake-quant on every conv input).

NOTE on bias: deployment quantizes the folded bias into an integer bias hex, but
the float folded bias b_fold_ref is a faithful proxy (bias quant error is
negligible relative to the W8 weight error), so (C) uses b_fold_ref directly --
exactly as the proven ResNet harness does.

  py scripts/measure_deployed_mbv2_acc.py [n_val=1500] [n_calib=256]

DO NOT run with a large n_val on a shared machine (serialize -- synth may be
running). n_val<=8 is a cheap structural dry-check.
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
# Reuse the EXACT eval helpers (top-1 loop + per-tensor int8 act fake-quant) the
# ResNet path used -- do NOT reimplement.
from gptq_int4 import eval_top1, convs  # noqa: E402

LAYER_IR = ROOT / "output" / "mobilenet-v2" / "layer_ir.json"

# Deployed integer domain for int8_symmetric_per_tensor.
QMIN_INT8, QMAX_INT8, GEN_QMAX_INT8 = -128, 127, 127


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def _load_int_weights(hex_path: Path, shape) -> np.ndarray:
    """Read flat signed-8bit 2's-complement hex (one value/line, row-major over
    the given shape) -> int np.ndarray of `shape`."""
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
    module-traversal order. In torchvision mobilenet_v2 this is exactly every
    ConvBNActivation/InvertedResidual conv->bn pair (depthwise convs included),
    giving 52 pairs. The conv order is IDENTICAL (by object identity) to
    convs(model), the order used by layer_ir. Returns a list of
    (conv_name, conv, bn_name, bn)."""
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
    if len(pairs) != len(mod_convs) or any(
            a is not b for a, b in zip(mod_convs, pair_convs)):
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
                 (+ conv.bias*s_bn if the conv had a bias; mbv2 convs have none)
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
    """Ensure a conv has a learnable bias Parameter (mbv2 convs ship bias=None
    because the bias is folded into BN; we fold it back into the conv)."""
    if conv.bias is None:
        conv.bias = nn.Parameter(torch.zeros(conv.weight.shape[0],
                                             dtype=conv.weight.dtype,
                                             device=conv.weight.device))


# --------------------------------------------------------------------------- #
# Model builders
# --------------------------------------------------------------------------- #
def build_folded_float(base: nn.Module, refs):
    """(B) FOLDED-FLOAT: conv = folded reference weight/bias, BN = identity.
    Classifier left as the stock float Linear (it carries no BN)."""
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


def _linear_layer(model: nn.Module) -> nn.Linear:
    lins = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if len(lins) != 1:
        raise RuntimeError(f"expected exactly 1 Linear classifier, found {len(lins)}")
    return lins[0]


def _pertensor_scale(W_ref: np.ndarray) -> float:
    """The PROVEN generating per-tensor int8 weight scale: max_abs(W)/127."""
    return max(float(np.abs(W_ref).max()) / GEN_QMAX_INT8, 1e-12)


def build_deployed(base: nn.Module, refs):
    """(C) DEPLOYED: conv weight = int_deployed * (max_abs(W_fold)/127)
    (PER-TENSOR generating scale, PROVEN), conv bias = folded reference bias,
    BN = identity; classifier weight = int_fc_deployed * (max_abs(W_fc)/127).
    The fc bias is kept as the stock float bias (the deployed bias hex is its
    quantized proxy; quant error negligible).

    The deployed integers are a faithful round-trip of the folded reference at
    this scale, so we VERIFY rather than assume: requant_match_frac = fraction of
    elements where clamp(round(W_ref/scale)) == int_deployed (must be ~1.0). This
    replaces the ResNet "scale ratio ~1" gate with a stronger element-exact gate.

    Returns (model, sanity_rows, fc_row) where each conv sanity row is
        (module_id, qmin_seen, qmax_seen, ok, requant_match_frac)
    and fc_row mirrors it for the classifier."""
    ir = json.loads(LAYER_IR.read_text())
    conv_layers = [l for l in ir["layers"] if l.get("op_type") == "conv2d"]
    gemm_layers = [l for l in ir["layers"] if l.get("op_type") == "gemm"]

    cs = convs(base)
    if len(cs) != len(conv_layers):
        raise RuntimeError(
            f"conv count mismatch: torchvision has {len(cs)} Conv2d, "
            f"layer_ir has {len(conv_layers)} conv layers"
        )
    if len(gemm_layers) != 1:
        raise RuntimeError(
            f"expected exactly 1 gemm (classifier) layer in layer_ir, "
            f"found {len(gemm_layers)}"
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

            int_w = _load_int_weights(Path(layer["weights_path"]), shape)
            Wf_np = Wf.detach().cpu().numpy().astype(np.float64)

            # PROVEN per-tensor generating scale from the FOLDED reference
            # (the layer_ir "scale_factor" is the activation scale, NOT this).
            scale = _pertensor_scale(Wf_np)
            float_w = (int_w.astype(np.float64) * scale).astype(np.float32)

            _attach_conv_bias(c)
            c.weight.copy_(torch.from_numpy(float_w).to(c.weight.dtype))
            c.bias.copy_(bf.to(c.bias.dtype))  # folded-float bias proxy
            set_bn_identity(bn)

            # --- sanity: integer range + requant element-exact round-trip ---
            qmin_seen, qmax_seen = int(int_w.min()), int(int_w.max())
            range_ok = (qmin_seen >= QMIN_INT8) and (qmax_seen <= QMAX_INT8)
            q = np.clip(np.round(Wf_np / scale), QMIN_INT8, QMAX_INT8).astype(np.int32)
            match_frac = float((q == int_w.astype(np.int32)).mean())
            sanity_rows.append((layer["module_id"], qmin_seen, qmax_seen,
                                range_ok, match_frac))

        # --- classifier: dequant the deployed int8 fc (faithful) ---
        g = gemm_layers[0]
        lin = _linear_layer(model)
        gshape = tuple(int(s) for s in g["weight_shape"])  # [out, in] = [1000,1280]
        if tuple(lin.weight.shape) != gshape:
            raise RuntimeError(
                f"classifier shape mismatch: layer_ir {gshape} vs "
                f"torchvision {tuple(lin.weight.shape)}"
            )
        int_fc = _load_int_weights(Path(g["weights_path"]), gshape)
        W_fc = lin.weight.detach().cpu().numpy().astype(np.float64)  # no BN fold on fc
        fc_scale = _pertensor_scale(W_fc)
        float_fc = (int_fc.astype(np.float64) * fc_scale).astype(np.float32)
        lin.weight.copy_(torch.from_numpy(float_fc).to(lin.weight.dtype))
        # fc bias: keep stock float bias (deployed bias hex is its quantized proxy).
        fc_qmin, fc_qmax = int(int_fc.min()), int(int_fc.max())
        fc_range_ok = (fc_qmin >= QMIN_INT8) and (fc_qmax <= QMAX_INT8)
        q_fc = np.clip(np.round(W_fc / fc_scale), QMIN_INT8, QMAX_INT8).astype(np.int32)
        fc_match = float((q_fc == int_fc.astype(np.int32)).mean())
        fc_row = (g["module_id"], fc_qmin, fc_qmax, fc_range_ok, fc_match)

    return model, sanity_rows, fc_row


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("n_val", nargs="?", type=int, default=1500,
                    help="number of val images (default 1500)")
    ap.add_argument("n_calib", nargs="?", type=int, default=256,
                    help="number of calib images skipped before val (default 256)")
    args = ap.parse_args()
    n_val, n_calib = args.n_val, args.n_calib
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    # IMAGENET1K_V2 (float top-1 72.154%) is the PROVEN deployed source:
    # corr(deployed_int_w, W_FOLDED_V2)=1.000 every layer (V1 gives corr ~0).
    base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2).eval()

    print(f"Deployed MobileNetV2 INT8 (BN-FOLD-AWARE) accuracy: "
          f"n_val={n_val} skip(calib)={n_calib} dev={dev} "
          f"(val DISJOINT from calib)")

    # 1) fold conv+bn -> per-conv reference folded weight/bias
    refs = fold_reference(base)

    # 2) build the three models
    folded = build_folded_float(base, refs)
    deployed, sanity_rows, fc_row = build_deployed(base, refs)

    # 3) val set DISJOINT from the n_calib calib images
    xval, yval = iu.load_batch(n_val, skip=n_calib)

    # ---- per-conv sanity block ----
    print(f"\n-- per-conv deployed integer-range + requant round-trip sanity "
          f"({len(sanity_rows)} convs, all expect int8 [-128,127]) --")
    all_ok = True
    matches = []
    for mid, qmn, qmx, ok, mf in sanity_rows:
        all_ok &= ok
        matches.append(mf)
        flag = "OK " if ok else "BAD"
        print(f"  [{flag}] {mid:>16s}  INT8  on-disk[min={qmn:4d}, max={qmx:4d}]"
              f"  expect[{QMIN_INT8},{QMAX_INT8}]  requant_match={mf*100:.4f}%")
    # classifier row
    fc_mid, fc_qmn, fc_qmx, fc_ok, fc_mf = fc_row
    all_ok &= fc_ok
    matches.append(fc_mf)
    print(f"  [{'OK ' if fc_ok else 'BAD'}] {fc_mid:>16s}  INT8  "
          f"on-disk[min={fc_qmn:4d}, max={fc_qmx:4d}]  expect[{QMIN_INT8},"
          f"{QMAX_INT8}]  requant_match={fc_mf*100:.4f}%  <-- classifier (Gemm)")

    min_match = float(np.min(matches))
    med_match = float(np.median(matches))
    print(f"-- range: {'ALL OK' if all_ok else 'RANGE VIOLATION'}  "
          f"(total={len(sanity_rows)} convs + 1 gemm)")
    print(f"-- requant round-trip: clamp(round(W_fold / (max_abs/127))) vs "
          f"deployed int -> median match={med_match*100:.4f}%, "
          f"min match={min_match*100:.4f}%  (must be ~100% => deployed ints are "
          f"the folded-weight quantization at the derived scale)\n")

    # static-gate trust flags (data/harness integrity, independent of accuracy).
    # min_match ~ 1.0 PROVES both the V2-folded reference AND the max_abs/127
    # per-tensor scale are correct (the deployed integers round-trip element-exact
    # bar rare rounding ties).
    range_ok = all_ok
    scale_ok = min_match >= 0.999  # >=99.9% of every layer's ints reproduced

    # ---- accuracy ----
    acc_stock = eval_top1(base, xval, yval, dev) * 100
    print(f"  (A) STOCK torchvision mbv2 V2 (BN intact): {acc_stock:.2f}%   <-- baseline (~72.2%)")

    acc_folded = eval_top1(folded, xval, yval, dev) * 100
    delta = acc_folded - acc_stock
    fold_ok = abs(delta) <= 1.0
    flag = "" if fold_ok else "  <-- WARNING: fold/identity harness DRIFT (>1%)"
    print(f"  (B) FOLDED-FLOAT (BN=identity)           : {acc_folded:.2f}%   "
          f"(delta vs STOCK = {delta:+.2f}%){flag}")

    acc_dep_w = eval_top1(deployed, xval, yval, dev) * 100
    print(f"  (C) DEPLOYED MobileNetV2 INT8 (w-only)   : {acc_dep_w:.2f}%")

    acc_dep_a8 = eval_top1(deployed, xval, yval, dev, act_int8=True) * 100
    print(f"  (C) DEPLOYED MobileNetV2 INT8 + A8 (act) : {acc_dep_a8:.2f}%   "
          f"<-- AUTHORITATIVE deployed top-1")

    # ---- consolidated TRUST verdict ------------------------------------- #
    # The deployed number is only trustworthy if EVERY harness/data gate holds:
    # integer ranges in-domain, the deployed integers round-trip element-exact
    # from the V2-folded reference at max_abs/127 (proves both the reference
    # checkpoint AND the per-tensor scale), AND the FOLDED-FLOAT model reproduces
    # STOCK (proves fold + BN-identity is faithful -- if B != A the methodology
    # is suspect).
    gates = [
        ("integer-range in-domain [-128,127]", range_ok),
        ("requant round-trip min-match >= 99.9%", scale_ok),
        ("FOLDED-FLOAT ~= STOCK (B==A within 1%)", fold_ok),
    ]
    trustworthy = all(ok for _, ok in gates)
    print("\n-- TRUST GATES (deployed number is trustworthy only if ALL pass) --")
    for desc, ok in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")
    if trustworthy:
        print(f"=> VERDICT: TRUSTWORTHY. Deployed MobileNetV2 INT8 top-1 (A8) = "
              f"{acc_dep_a8:.2f}% on {n_val} val imgs (disjoint from {n_calib} calib).")
    else:
        print("=> VERDICT: NOT TRUSTWORTHY -- a harness/data gate FAILED above; "
              "the deployed number is an ARTIFACT, do NOT report it.")

    return 0 if trustworthy else 1


if __name__ == "__main__":
    sys.exit(main())
