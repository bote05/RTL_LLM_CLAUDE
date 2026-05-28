#!/usr/bin/env python3
"""Phase 1.6 software validation: does the ImageNet-calibrated quantized
reference (the SAME functions generate_golden uses) classify ImageNet well?

Reuses onnx_frontend internals: extract_layer_specs + calibrate_onnx +
fill_calibration_stats + _build_int8_module + run_int8_network. The backbone
ONNX outputs [1,2048,7,7]; we append GAP + the torchvision ResNet-50 fc head to
get 1000-class logits. Validates on images DISJOINT from the calibration set.

Set NN2RTL_WEIGHT_BITS=8 or =4 before running to compare INT8 vs INT4 weights.
  py scripts/validate_quant_accuracy.py [n_calib=256] [n_val=512]

A near-chance (0.1%) top-1 means the calibration / input-scale handling is
wrong — fix before regenerating goldens.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import scripts.onnx_frontend as F           # noqa: E402
import imagenet_util as iu                  # noqa: E402

ONNX = ROOT / "checkpoints/resnet50_full.onnx"


def build_quant_modules(n_calib: int):
    model = F.load_onnx(ONNX)
    model = F.simplify_onnx(model)
    specs = F.extract_layer_specs(model)
    net_in = F._real_graph_inputs(model)[0].name
    imgs, _ = iu.load_batch(n_calib)
    feeds = [{net_in: imgs[i:i + 1].astype(np.float32)} for i in range(n_calib)]
    stats = F.calibrate_onnx(model, feeds)
    F.fill_calibration_stats(specs, stats, network_input_name=net_in,
                             network_input_max_abs=128.0)
    modules = {s.module_id: F._build_int8_module(s, rtl_compat_conv=False)
               for s in specs}
    in_scale = float(specs[0].input_scale) or 1.0
    final = specs[-1]
    return specs, modules, in_scale, net_in, final.output_tensor_name, \
        float(final.output_scale)


def pick_head(backbone_float_fn, xs, ys, dev):
    """Identify which torchvision weights match the ONNX backbone by trying
    each fc head on the FLOAT backbone features; return (name, avgpool, fc)."""
    from torchvision.models import resnet50, ResNet50_Weights
    best = None
    for nm, w in [("V1", ResNet50_Weights.IMAGENET1K_V1),
                  ("V2", ResNet50_Weights.IMAGENET1K_V2)]:
        m = resnet50(weights=w).eval().to(dev)
        feats = backbone_float_fn(xs)  # (N,2048,7,7) float
        with torch.no_grad():
            pooled = torch.nn.functional.adaptive_avg_pool2d(
                torch.from_numpy(feats).to(dev), 1).flatten(1)
            logits = m.fc(pooled)
            acc = float((logits.argmax(1).cpu().numpy() == ys).mean())
        print(f"   float-backbone + {nm} fc head: top1={acc*100:.1f}%")
        if best is None or acc > best[0]:
            best = (acc, nm, m.avgpool, m.fc)
    return best


def main() -> int:
    n_calib = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    n_val = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"WEIGHT_BITS={F.WEIGHT_BITS}  n_calib={n_calib}  n_val={n_val}  dev={dev}")

    specs, modules, in_scale, net_in, out_name, out_scale = build_quant_modules(n_calib)
    print(f"calibrated: input_scale={in_scale:.5f}  final={out_name} "
          f"output_scale={out_scale:.5f}  layers={len(specs)}")

    # validation images (disjoint from calibration)
    xv, yv = iu.load_batch(n_val, skip=n_calib)
    import onnxruntime as ort

    def backbone_float(xs):
        sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
        outs = []
        for i in range(xs.shape[0]):
            outs.append(sess.run(None, {net_in: xs[i:i+1].astype(np.float32)})[0])
        return np.concatenate(outs, 0)

    # pick the matching fc head on a small float subset
    print("identifying fc head (float backbone):")
    nb = min(128, n_val)
    _, head_name, avgpool, fc = pick_head(backbone_float, xv[:nb], yv[:nb], dev)
    print(f"=> using {head_name} head")

    # quantized backbone forward (the pipeline path) on all val images
    top1 = top5 = 0
    bs = 64
    with torch.no_grad():
        for s in range(0, n_val, bs):
            xb = xv[s:s+bs]
            feats = []
            for i in range(xb.shape[0]):
                inp = torch.tensor(xb[i:i+1]) / in_scale
                inp = F.quantize_tensor_to_int8_range(inp)
                tmap = F.run_int8_network(specs, modules, inp)
                feats.append(tmap[out_name].cpu().numpy())
            feats = np.concatenate(feats, 0).astype(np.float32) * out_scale  # dequant
            pooled = torch.nn.functional.adaptive_avg_pool2d(
                torch.from_numpy(feats).to(dev), 1).flatten(1)
            logits = fc(pooled)
            pred1 = logits.argmax(1).cpu().numpy()
            pred5 = torch.topk(logits, 5, 1).indices.cpu().numpy()
            yb = yv[s:s+bs]
            top1 += int((pred1 == yb).sum())
            top5 += int(sum(yb[i] in pred5[i] for i in range(len(yb))))
            print(f"   [{s+len(yb)}/{n_val}] running top1={top1/(s+len(yb))*100:.1f}%")
    print(f"\n=== QUANTIZED reference (WEIGHT_BITS={F.WEIGHT_BITS}) on {n_val} val images ===")
    print(f"top1={top1/n_val*100:.2f}%  top5={top5/n_val*100:.2f}%  (head={head_name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
