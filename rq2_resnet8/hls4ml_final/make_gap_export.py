"""Build a hls4ml-export copy of the retrained ResNet-8 with the tail
AveragePooling2D(8x8) swapped for GlobalAveragePooling2D, PRESERVING every
learned quantizer config exactly (the retrained model uses WIDER activation
int-bits than the build_resnet8_qkeras defaults: quantized_relu(8,4) on the
relu_out paths and quantized_bits(8,5) on the add operands -- a fresh rebuild
with defaults silently changes saturation and diverges).

WHY swap: hls4ml maps AveragePooling2D(8) to pooling2d_cl whose io_stream firmware
builds kernel_data[pool_h*pool_w*n_filt] + a 64-wide pool_window per filter, fully
unrolled over n_filt -> 114,905 LUT + 65,272 FF (the #1 resource hog; ReuseFactor
does NOT throttle it). GlobalAveragePooling2D -> global_pooling2d_cl, a single
data_window[n_filt] streaming accumulator + one divide -> a few hundred LUT.

CORRECTNESS: pool input is exactly 8x8 (whole feature map), so AveragePooling2D(8)
== GlobalAveragePooling2D numerically (verified: avg_pool out == manual global mean,
maxdiff 0.0). Pooling has no weights. We do the swap by CONFIG SURGERY on the trained
model's own serialized config (keeps every quantizer), then transfer weights by name,
then verify argmax-identity on the full 10k CIFAR-10 test set before accepting.
"""
import os, pickle, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np

sys.path.insert(0, "/root/rq2_training/qkeras")
from resnet8_qkeras8 import load_qkeras_h5

import tensorflow as tf
from tensorflow.keras.models import Model

TRAINED = "/root/rq2_training/qkeras_gpu_fixed_full/resnet8_qkeras8_best_nosoftmax.h5"
OUT = "/root/rq2_training/hls4ml_resnet8_final/resnet8_qkeras8_gap_nosoftmax.h5"
DATA = "/mnt/d/RTL_LLM_CLAUDE/rq2_resnet8/data/cifar-10-batches-py"


def build_gap_model(trained):
    """Clone the EXACT trained model but replace the AveragePooling2D layer with
    GlobalAveragePooling2D (identical math on an 8x8 input). clone_model rebuilds
    from each layer's own config, so all learned quantizer settings are preserved.
    The Flatten after the pool becomes a no-op on (None,1,1,64)->(None,64); GAP
    already yields (None,64), so we drop the Flatten by mapping it to a passthrough.
    """
    from tensorflow.keras.layers import GlobalAveragePooling2D

    def clone_fn(layer):
        cls = layer.__class__.__name__
        if cls == "AveragePooling2D":
            # exact-equivalent global average; keep the same name for weight transfer.
            # GAP yields (None, n_filt); the following Flatten stays (identity on a
            # rank-2 tensor) and is supported by hls4ml (Reshape).
            return GlobalAveragePooling2D(name=layer.name)
        return layer.__class__.from_config(layer.get_config())

    gap_model = tf.keras.models.clone_model(trained, clone_function=clone_fn)
    return gap_model


def load_test():
    with open(os.path.join(DATA, "test_batch"), "rb") as fo:
        d = pickle.load(fo, encoding="bytes")
    x = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.array(d[b"labels"])
    x = (x / 256.0).astype("float32")
    return x, y


def main():
    trained = load_qkeras_h5(TRAINED)
    print("[gap] trained model loaded (%d params)" % trained.count_params(), flush=True)

    gap_model = build_gap_model(trained)
    n_copied = 0
    for L in gap_model.layers:
        if L.get_weights():
            try:
                L.set_weights(trained.get_layer(L.name).get_weights())
                n_copied += 1
            except Exception as e:
                print("[gap] WARN no source for %s: %s" % (L.name, e), flush=True)
    print("[gap] transferred weights for %d layers; params=%d"
          % (n_copied, gap_model.count_params()), flush=True)

    x, y = load_test()
    lo_orig = trained.predict(x, batch_size=256, verbose=0)
    lo_gap = gap_model.predict(x, batch_size=256, verbose=0)
    p_orig = lo_orig.argmax(axis=1)
    p_gap = lo_gap.argmax(axis=1)
    match = int((p_orig == p_gap).sum())
    acc_orig = float((p_orig == y).mean())
    acc_gap = float((p_gap == y).mean())
    maxdiff = float(np.max(np.abs(lo_orig - lo_gap)))
    print("[gap] argmax match orig-vs-GAP = %d/%d  max|logit diff|=%.6f" % (match, len(y), maxdiff), flush=True)
    print("[gap] orig top1=%.4f  GAP top1=%.4f" % (acc_orig, acc_gap), flush=True)

    if match != len(y):
        print("GAP_SWAP_MISMATCH match=%d/%d" % (match, len(y)), flush=True)
        sys.exit(2)

    gap_model.save(OUT)
    print("[gap] saved %s" % OUT, flush=True)
    print("GAP_SWAP_OK match=%d/%d GAP_TOP1=%.4f MAXDIFF=%.6f" % (match, len(y), acc_gap, maxdiff), flush=True)


if __name__ == "__main__":
    main()
