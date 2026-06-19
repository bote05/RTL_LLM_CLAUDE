#!/usr/bin/env python3
"""Convert MLPerf Tiny pretrainedResnet.h5 (ResNet-8 / resnet_v1_eembc) to ONNX.

Produces model/resnet8_ref.onnx (opset 13, as emitted by tf2onnx).
Run inside /root/rq2_venv (TF 2.15 + tf2onnx 1.16).

Usage: python convert_h5_to_onnx.py [--root /mnt/d/RTL_LLM_CLAUDE/rq2_resnet8]
"""
import argparse
import collections
import os
import sys

import numpy as np
import tensorflow as tf
import tf2onnx
import onnx


def count_nodes(model_path):
    m = onnx.load(model_path)
    c = collections.Counter(n.op_type for n in m.graph.node)
    return c, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/d/RTL_LLM_CLAUDE/rq2_resnet8")
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    h5_path = os.path.join(args.root, "model", "pretrainedResnet.h5")
    out_path = os.path.join(args.root, "model", "resnet8_ref.onnx")

    model = tf.keras.models.load_model(h5_path)
    model.summary()
    print("INPUT:", model.input_shape, "OUTPUT:", model.output_shape)
    # report conv filter widths actually embedded in the h5
    for lyr in model.layers:
        if isinstance(lyr, tf.keras.layers.Conv2D):
            print("CONV", lyr.name, "filters=", lyr.filters,
                  "k=", lyr.kernel_size, "s=", lyr.strides)

    spec = (tf.TensorSpec((None, 32, 32, 3), tf.float32, name="input"),)
    onnx_model, _ = tf2onnx.convert.from_keras(
        model, input_signature=spec, opset=args.opset, output_path=out_path)

    c, m = count_nodes(out_path)
    print("ONNX_NODE_COUNTS:", dict(c))
    print("ONNX_INPUTS:", [(i.name, [d.dim_value or d.dim_param for d in
          i.type.tensor_type.shape.dim]) for i in m.graph.input])
    print("ONNX_OUTPUTS:", [(o.name, [d.dim_value or d.dim_param for d in
          o.type.tensor_type.shape.dim]) for o in m.graph.output])

    # parity check: keras vs onnxruntime on random data (raw-pixel scale)
    import onnxruntime as ort
    rng = np.random.RandomState(0)
    x = rng.randint(0, 256, size=(16, 32, 32, 3)).astype(np.float32)
    y_tf = model.predict(x, verbose=0)
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    y_ox = sess.run(None, {iname: x})[0]
    print("TF_vs_ONNX_max_abs_diff:", float(np.max(np.abs(y_tf - y_ox))))
    print("TF_vs_ONNX_argmax_agree:", int((y_tf.argmax(1) == y_ox.argmax(1)).sum()), "/", len(x))


if __name__ == "__main__":
    sys.exit(main())
