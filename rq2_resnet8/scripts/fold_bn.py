#!/usr/bin/env python3
"""Produce a BN-folded variant of resnet8_ref.onnx -> resnet8_folded.onnx.

Strategy: try onnxoptimizer fuse_bn_into_conv first; if any BatchNormalization
nodes survive (e.g. BN not directly downstream of a Conv with constant params),
apply a manual numpy fold. Verifies: 0 BN nodes remain, and numerical parity
vs the reference model on random raw-pixel-scale inputs.

Usage: python fold_bn.py [--root /mnt/d/RTL_LLM_CLAUDE/rq2_resnet8]
"""
import argparse
import collections
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper


def node_counts(m):
    return collections.Counter(n.op_type for n in m.graph.node)


def manual_fold(m):
    """Fold each BatchNormalization whose input comes from a Conv (single
    consumer) into that Conv's weights/bias. ONNX Conv here is NCHW: W is
    (OC, IC, kH, kW); BN scales per OC."""
    g = m.graph
    inits = {i.name: i for i in g.initializer}

    def consumers(tensor_name):
        return [n for n in g.node if tensor_name in n.input]

    changed = True
    while changed:
        changed = False
        for bn in [n for n in g.node if n.op_type == "BatchNormalization"]:
            prod = [n for n in g.node if bn.input[0] in n.output]
            if len(prod) != 1 or prod[0].op_type != "Conv":
                continue
            conv = prod[0]
            if len(consumers(conv.output[0])) != 1:
                continue
            if not all(i in inits for i in bn.input[1:5]):
                continue
            gamma = numpy_helper.to_array(inits[bn.input[1]]).astype(np.float64)
            beta = numpy_helper.to_array(inits[bn.input[2]]).astype(np.float64)
            mean = numpy_helper.to_array(inits[bn.input[3]]).astype(np.float64)
            var = numpy_helper.to_array(inits[bn.input[4]]).astype(np.float64)
            eps = 1e-5
            for a in bn.attribute:
                if a.name == "epsilon":
                    eps = a.f
            W = numpy_helper.to_array(inits[conv.input[1]]).astype(np.float64)
            if len(conv.input) > 2 and conv.input[2] in inits:
                b = numpy_helper.to_array(inits[conv.input[2]]).astype(np.float64)
                bias_name = conv.input[2]
            else:
                b = np.zeros(W.shape[0], dtype=np.float64)
                bias_name = conv.name + "_folded_bias"

            scale = gamma / np.sqrt(var + eps)
            W2 = (W * scale[:, None, None, None]).astype(np.float32)
            b2 = (beta + (b - mean) * scale).astype(np.float32)

            new_w = numpy_helper.from_array(W2, conv.input[1])
            inits[conv.input[1]].CopyFrom(new_w)
            new_b = numpy_helper.from_array(b2, bias_name)
            if bias_name in inits:
                inits[bias_name].CopyFrom(new_b)
            else:
                g.initializer.append(new_b)
                inits[bias_name] = g.initializer[-1]
                conv.input.append(bias_name)

            # rewire: conv output takes over BN's output name
            old_out = conv.output[0]
            conv.output[0] = bn.output[0]
            g.node.remove(bn)
            changed = True
            print("FOLDED", bn.name or bn.output[0], "into", conv.name, "(was", old_out, ")")
            break
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/d/RTL_LLM_CLAUDE/rq2_resnet8")
    args = ap.parse_args()

    ref_path = os.path.join(args.root, "model", "resnet8_ref.onnx")
    out_path = os.path.join(args.root, "model", "resnet8_folded.onnx")

    m = onnx.load(ref_path)
    print("REF_NODE_COUNTS:", dict(node_counts(m)))

    folded = None
    try:
        import onnxoptimizer
        cand = onnxoptimizer.optimize(m, ["fuse_bn_into_conv"])
        nbn = node_counts(cand).get("BatchNormalization", 0)
        print("onnxoptimizer fuse_bn_into_conv -> BN remaining:", nbn)
        if nbn == 0:
            folded = cand
            print("USING onnxoptimizer result")
    except Exception as e:
        print("onnxoptimizer unavailable/failed:", e)

    if folded is None:
        print("falling back to manual fold")
        folded = manual_fold(onnx.load(ref_path))
        nbn = node_counts(folded).get("BatchNormalization", 0)
        assert nbn == 0, f"manual fold left {nbn} BN nodes"

    onnx.checker.check_model(folded)
    onnx.save(folded, out_path)
    print("FOLDED_NODE_COUNTS:", dict(node_counts(folded)))

    # numerical parity ref vs folded
    import onnxruntime as ort
    rng = np.random.RandomState(1)
    x = rng.randint(0, 256, size=(64, 32, 32, 3)).astype(np.float32)
    s1 = ort.InferenceSession(ref_path, providers=["CPUExecutionProvider"])
    s2 = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    y1 = s1.run(None, {s1.get_inputs()[0].name: x})[0]
    y2 = s2.run(None, {s2.get_inputs()[0].name: x})[0]
    print("REF_vs_FOLDED_max_abs_diff:", float(np.max(np.abs(y1 - y2))))
    print("REF_vs_FOLDED_argmax_agree:", int((y1.argmax(1) == y2.argmax(1)).sum()), "/", len(x))


if __name__ == "__main__":
    sys.exit(main())
