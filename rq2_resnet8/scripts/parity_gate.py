#!/usr/bin/env python3
"""Leg A parity gate: torch ResNet-8 port vs onnxruntime(resnet8_folded.onnx)
on the FULL 10k CIFAR-10 test set (raw 0..255, same values both sides).

PASS requires:
  1. torch ref top-1 == 8719/10000 (87.19%)
  2. argmax(torch ref) == argmax(ort folded) on 10000/10000
  3. (report) max |softmax(torch logits) - ort probs|  (expect < 1e-4)
  4. export-variant (symmetric-pad reformulation) argmax == ref argmax
     10000/10000 (reported with max |logit diff|)

Exit code 0 only if gates 1, 2 and 4 hold.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from torch_resnet8 import DEFAULT_ONNX, build_export, build_ref, load_cifar10_test_nchw

BATCH = 500


def main() -> int:
    x, labels = load_cifar10_test_nchw()              # (10000,3,32,32) raw 0..255
    n = len(x)
    print(f"test set: {x.shape} {x.dtype} pixel range [{x.min():g},{x.max():g}]")

    sess = ort.InferenceSession(str(DEFAULT_ONNX), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    x_nhwc = np.transpose(x, (0, 2, 3, 1)).copy()     # reference model is NHWC

    ref, exp = build_ref(), build_export()

    ort_probs, ref_logits, exp_logits = [], [], []
    with torch.no_grad():
        for i in range(0, n, BATCH):
            ort_probs.append(sess.run(None, {iname: x_nhwc[i:i + BATCH]})[0])
            xb = torch.from_numpy(x[i:i + BATCH])
            ref_logits.append(ref(xb).numpy())
            exp_logits.append(exp(xb).numpy())
    ort_probs = np.concatenate(ort_probs)             # softmax probs
    ref_logits = np.concatenate(ref_logits)
    exp_logits = np.concatenate(exp_logits)

    ref_pred = ref_logits.argmax(1)
    ort_pred = ort_probs.argmax(1)
    exp_pred = exp_logits.argmax(1)

    ref_probs = torch.softmax(torch.from_numpy(ref_logits), dim=1).numpy()

    res = {
        "n": int(n),
        "torch_ref_top1_count": int((ref_pred == labels).sum()),
        "ort_folded_top1_count": int((ort_pred == labels).sum()),
        "ref_vs_ort_argmax_agree": int((ref_pred == ort_pred).sum()),
        "max_abs_prob_diff_ref_vs_ort": float(np.abs(ref_probs - ort_probs).max()),
        "export_vs_ref_argmax_agree": int((exp_pred == ref_pred).sum()),
        "max_abs_logit_diff_export_vs_ref": float(np.abs(exp_logits - ref_logits).max()),
        "export_top1_count": int((exp_pred == labels).sum()),
    }
    ok = (res["torch_ref_top1_count"] == 8719
          and res["ref_vs_ort_argmax_agree"] == n
          and res["export_vs_ref_argmax_agree"] == n)
    res["gate"] = "PASS" if ok else "FAIL"
    print(json.dumps(res, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
