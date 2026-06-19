#!/usr/bin/env python3
"""Score the MLPerf Tiny ResNet-8 (.h5 and/or .onnx) on the FULL CIFAR-10
test set (10,000 images) with the EXACT upstream MLPerf Tiny preprocessing.

Preprocessing (replicates mlcommons/tiny train.py load_cifar_10_data,
negatives=False, as consumed by test.py):
  1. read test_batch (CIFAR-10 python pickle), data: uint8 (10000, 3072)
  2. reshape to (N, 3, 32, 32) then np.rollaxis(axis 1 -> 4) => NHWC uint8
  3. NO /255, NO mean/std: raw 0..255 values, cast to float32 by the
     framework (Keras casts implicitly; we cast explicitly for ORT).

Also reports accuracy on the official 200-image perf subset
(perf_samples_idxs.npy) for cross-checking against published MLPerf numbers.

Usage:
  python eval_cifar10.py --root /mnt/d/RTL_LLM_CLAUDE/rq2_resnet8 \
      [--h5] [--onnx model/resnet8_ref.onnx] [--onnx model/resnet8_folded.onnx]
"""
import argparse
import os
import pickle
import sys

import numpy as np


def load_test_set(data_dir):
    with open(os.path.join(data_dir, "test_batch"), "rb") as fo:
        d = pickle.load(fo, encoding="bytes")
    data = d[b"data"]                      # (10000, 3072) uint8
    labels = np.array(d[b"labels"])        # (10000,)
    data = data.reshape((len(data), 3, 32, 32))
    data = np.rollaxis(data, 1, 4)         # NHWC, uint8, raw 0..255
    return data, labels


def top1(pred, labels):
    return float((pred.argmax(axis=1) == labels).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/d/RTL_LLM_CLAUDE/rq2_resnet8")
    ap.add_argument("--h5", action="store_true", help="score model/pretrainedResnet.h5 via TF")
    ap.add_argument("--onnx", action="append", default=[], help="relative onnx path(s)")
    args = ap.parse_args()

    data_dir = os.path.join(args.root, "data", "cifar-10-batches-py")
    x_u8, labels = load_test_set(data_dir)
    x = x_u8.astype(np.float32)            # raw 0..255
    print("test set:", x.shape, x.dtype, "labels:", labels.shape,
          "pixel range: [%g, %g]" % (x.min(), x.max()))

    idx_path = os.path.join(args.root, "scripts", "upstream_refs",
                            "perf_samples_idxs.npy")
    perf_idx = np.load(idx_path) if os.path.exists(idx_path) else None

    results = {}

    for rel in args.onnx:
        import onnxruntime as ort
        p = os.path.join(args.root, rel)
        sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        preds = []
        for i in range(0, len(x), 500):
            preds.append(sess.run(None, {iname: x[i:i + 500]})[0])
        preds = np.concatenate(preds)
        acc = top1(preds, labels)
        results[rel] = acc
        print("ONNX %s FULL-10000 top-1: %.4f (%.2f%%)" % (rel, acc, 100 * acc))
        if perf_idx is not None:
            pacc = top1(preds[perf_idx], labels[perf_idx])
            print("ONNX %s PERF-200 top-1: %.4f (%.2f%%)" % (rel, pacc, 100 * pacc))
        del sess

    if args.h5:
        import tensorflow as tf
        model = tf.keras.models.load_model(
            os.path.join(args.root, "model", "pretrainedResnet.h5"))
        preds = model.predict(x, batch_size=500, verbose=0)
        acc = top1(preds, labels)
        results["pretrainedResnet.h5"] = acc
        print("H5 FULL-10000 top-1: %.4f (%.2f%%)" % (acc, 100 * acc))
        if perf_idx is not None:
            pacc = top1(preds[perf_idx], labels[perf_idx])
            print("H5 PERF-200 top-1: %.4f (%.2f%%)" % (pacc, 100 * pacc))

    print("RESULTS:", results)


if __name__ == "__main__":
    sys.exit(main())
