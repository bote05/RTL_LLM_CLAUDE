#!/usr/bin/env python
"""Numerical diagnostic: per-layer activation ranges vs quantizer clip ranges.

Loads the plateaued best.h5, runs a forward pass on real CIFAR-10 images, and
for every QActivation / Add / conv output reports:
  - pre-quant float range (min/max/mean-abs/p99.9)
  - the quantizer's representable max (clip ceiling)
  - SATURATION %: fraction of elements at/above the clip ceiling
This pinpoints where the 8-bit grid is throttling signal.
"""
import os, sys, json
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU-only diag, leave GPU for training
import numpy as np

DATA = "/mnt/d/RTL_LLM_CLAUDE/rq2_resnet8/data/cifar-10-batches-py"
ORIG = "/root/rq2_training/qkeras"
H5 = sys.argv[1] if len(sys.argv) > 1 else "/root/rq2_training/qkeras_gpu/resnet8_qkeras8_best.h5"

sys.path.insert(0, ORIG)
import tensorflow as tf
from tensorflow.keras.models import Model
from resnet8_qkeras8 import load_qkeras_h5

def load_cifar_test(n=512):
    import pickle
    with open(os.path.join(DATA, "test_batch"), "rb") as fo:
        d = pickle.load(fo, encoding="bytes")
    x = d[b"data"].reshape(-1,3,32,32).transpose(0,2,3,1)[:n]
    y = np.array(d[b"labels"])[:n]
    return (x/256.0).astype("float32"), y

x, y = load_cifar_test(512)
model = load_qkeras_h5(H5)
print("[loaded]", H5, "layers:", len(model.layers))

# quantizer clip ceilings (from the build config)
# quantized_relu(8,2): max ~= 2^2 - 2^-(8-2) = 4 - 0.015625 = 3.984375
RELU_MAX = 2.0**2 - 2.0**-(8-2)
# quantized_bits(8,2,alpha=1) signed: max ~= 2^(2-1) ... actually integer=2 keep_negative -> range [-2, 2)
# qkeras quantized_bits integer counts bits left of point INCLUDING sign region differently;
# representable max for quantized_bits(bits=8,integer=2,keep_negative=1) = 2^2 - 2^-(8-1-2) = 4 - 2^-5
ADD_MAX = 2.0**2 - 2.0**-(8-1-2)

# Tap every layer output
tap_names = [l.name for l in model.layers]
tap_model = Model(model.input, [l.output for l in model.layers])
outs = tap_model.predict(x, batch_size=128, verbose=0)

print("\n%-16s %-12s %10s %10s %10s %10s %10s" % (
    "layer", "type", "min", "max", "p99.9", "clip_ceil", "SAT%"))
print("-"*92)
for name, l, o in zip(tap_names, model.layers, outs):
    a = np.asarray(o).ravel()
    amin, amax = float(a.min()), float(a.max())
    p999 = float(np.percentile(np.abs(a), 99.9))
    ceil = ""
    sat = ""
    t = type(l).__name__
    lname = name.lower()
    if "relu" in lname or "stem_relu" in lname:
        ceil = "%.4f" % RELU_MAX
        sat = "%.2f%%" % (100.0*np.mean(a >= RELU_MAX*0.999))
    elif "branch_q" in lname or "proj_q" in lname:
        ceil = "%.4f" % ADD_MAX
        sat = "%.2f%%" % (100.0*np.mean(np.abs(a) >= ADD_MAX*0.999))
    print("%-16s %-12s %10.4f %10.4f %10.4f %10s %10s" % (
        name, t, amin, amax, p999, ceil, sat))

# accuracy sanity
logits = model.predict(x, batch_size=128, verbose=0)
acc = float(np.mean(np.argmax(logits, axis=1) == y))
print("\n[sanity] top-1 on %d test imgs = %.4f" % (len(y), acc))
print("[ceilings] relu(8,2) max=%.5f  add quantized_bits(8,2) max=%.5f" % (RELU_MAX, ADD_MAX))
