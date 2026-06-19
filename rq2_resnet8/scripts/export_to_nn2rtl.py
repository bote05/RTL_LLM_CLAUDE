#!/usr/bin/env python3
"""Leg A step 3: export the (symmetric-pad, frontend-friendly) torch ResNet-8
to D:/RTL_LLM_CLAUDE/nn2rtl-repo/checkpoints/resnet8.onnx using the repo's own
scripts.onnx_frontend.export_pytorch_to_onnx helper (NEW file; no existing
nn2rtl file is touched).

Then verifies the exported graph:
  - node ops within the frontend-supported/passthrough set; specifically NO
    Softmax / Transpose / BatchNormalization / Pad / MatMul, and asymmetric
    Conv pads absent (frontend hard-rejects them),
  - input tensor [batch,3,32,32] float32,
and re-scores the exported ONNX with onnxruntime on the first 1000 CIFAR-10
test images: argmax must equal the torch export model's argmax 1000/1000.

Exit code 0 only on full pass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

NN2RTL_ROOT = Path("D:/RTL_LLM_CLAUDE/nn2rtl-repo")
RQ2_SCRIPTS = Path("D:/RTL_LLM_CLAUDE/rq2_resnet8/scripts")
OUT_ONNX = NN2RTL_ROOT / "checkpoints" / "resnet8.onnx"

sys.path.insert(0, str(NN2RTL_ROOT))
sys.path.insert(0, str(RQ2_SCRIPTS))

# Ops the nn2rtl onnx_frontend either extracts or treats as passthrough/skip
# without breaking graph-completeness (Constant feeds only shape inputs).
# "Shape" appears in the dynamo export's dynamic-batch Reshape chain
# (Shape(input)->Concat->Reshape.shape); its output feeds ONLY the Reshape
# shape input, and Reshape is an alias-passthrough in the frontend, so the
# skipped Shape/Concat never become spec inputs (verified for resnet8.onnx).
ALLOWED_OPS = {"Conv", "Relu", "Add", "GlobalAveragePool", "Gemm",
               "Flatten", "Reshape", "Squeeze", "Unsqueeze", "Identity",
               "Constant", "ReduceMean", "Concat", "Shape"}
FORBIDDEN_OPS = {"Softmax", "Transpose", "BatchNormalization", "Pad", "MatMul",
                 "AveragePool"}


def main() -> int:
    import onnx
    import onnxruntime as ort
    import torch

    from scripts.onnx_frontend import DEFAULT_ONNX_OPSET, export_pytorch_to_onnx
    from torch_resnet8 import build_export, load_cifar10_test_nchw

    model = build_export()
    export_pytorch_to_onnx(model, OUT_ONNX, input_shape=(1, 3, 32, 32))

    m = onnx.load(str(OUT_ONNX))
    ops = sorted({n.op_type for n in m.graph.node})
    op_list = [n.op_type for n in m.graph.node]
    bad = (set(ops) & FORBIDDEN_OPS) | (set(ops) - ALLOWED_OPS)

    # input shape check (batch dim may be symbolic)
    inits = {i.name for i in m.graph.initializer}
    real_in = [i for i in m.graph.input if i.name not in inits]
    dims = [d.dim_value if d.dim_value > 0 else str(d.dim_param)
            for d in real_in[0].type.tensor_type.shape.dim]

    # all Conv pads must be SYMMETRIC (frontend hard-rejects asymmetric)
    asym = []
    for n in m.graph.node:
        if n.op_type != "Conv":
            continue
        pads = next((list(a.ints) for a in n.attribute if a.name == "pads"),
                    [0, 0, 0, 0])
        if len(pads) >= 4 and (pads[0] != pads[2] or pads[1] != pads[3]):
            asym.append((n.name, pads))

    # re-score: ort(exported onnx) vs torch export model, 1000 test images
    x, labels = load_cifar10_test_nchw()
    x = x[:1000]
    sess = ort.InferenceSession(str(OUT_ONNX), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    ort_logits, torch_logits = [], []
    with torch.no_grad():
        for i in range(0, len(x), 500):
            ort_logits.append(sess.run(None, {iname: x[i:i + 500]})[0])
            torch_logits.append(model(torch.from_numpy(x[i:i + 500])).numpy())
    ort_pred = np.concatenate(ort_logits).argmax(1)
    torch_pred = np.concatenate(torch_logits).argmax(1)

    res = {
        "onnx_path": str(OUT_ONNX),
        "opset_requested": int(DEFAULT_ONNX_OPSET),
        "opset_in_file": [int(oi.version) for oi in m.opset_import if oi.domain in ("", "ai.onnx")],
        "node_ops": op_list,
        "unexpected_ops": sorted(bad),
        "asymmetric_conv_pads": asym,
        "input_dims": dims,
        "n_conv": op_list.count("Conv"),
        "n_relu": op_list.count("Relu"),
        "n_add": op_list.count("Add"),
        "rescore_n": int(len(x)),
        "ort_vs_torch_argmax_agree": int((ort_pred == torch_pred).sum()),
        "ort_top1_count_1000": int((ort_pred == labels[:1000]).sum()),
    }
    ok = (not bad and not asym
          and res["ort_vs_torch_argmax_agree"] == len(x)
          and dims[1:] == [3, 32, 32])
    res["gate"] = "PASS" if ok else "FAIL"
    print(json.dumps(res, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
