"""Layer-by-layer QKeras vs hls4ml profiling to localize numerical divergence."""
import os, sys, pickle
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np
sys.path.insert(0, "/root/rq2_training/qkeras")
from resnet8_qkeras8 import load_qkeras_h5
import hls4ml
from tensorflow.keras.models import Model as KModel

M = "/root/rq2_training/qkeras/resnet8_qkeras8_best_nosoftmax.h5"
model = load_qkeras_h5(M)

d = pickle.load(open("/mnt/d/RTL_LLM_CLAUDE/rq2_resnet8/data/cifar-10-batches-py/test_batch", "rb"), encoding="bytes")
x = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)[:4] / 256.0
x = np.ascontiguousarray(x.astype("float32"))

cfg = hls4ml.utils.config_from_keras_model(
    model, granularity="name", backend="Vitis",
    default_precision="ap_fixed<16,6>", default_reuse_factor=72)
cfg["Model"]["Strategy"] = "Resource"
cfg["Model"]["ReuseFactor"] = 72
# trace ON for every layer
for ln, lc in cfg["LayerName"].items():
    lc["ReuseFactor"] = 72
    lc["Strategy"] = "Resource"
    lc["Trace"] = True
for nm, n_in in {"s2_proj": 16, "s3_proj": 32}.items():
    if nm in cfg["LayerName"]:
        cfg["LayerName"][nm]["ReuseFactor"] = n_in
# adder-alignment quantizer fix
for nm in ["s1_bn2", "s2_bn2", "s2_proj", "s3_bn2", "s3_proj"]:
    if nm in cfg["LayerName"]:
        p = cfg["LayerName"][nm].setdefault("Precision", {})
        if not isinstance(p, dict):
            p = {"result": p}
            cfg["LayerName"][nm]["Precision"] = p
        p["result"] = "fixed<8,3,RND_CONV,SAT,0>"

hls_model = hls4ml.converters.convert_from_keras_model(
    model, hls_config=cfg, output_dir="/root/rq2_training/hls4ml_resnet8/prj_trace",
    backend="Vitis", io_type="io_stream", part="xczu7ev-ffvc1156-2-e", clock_period=10.0)
hls_model.compile()

# hls4ml trace
hls_pred, hls_trace = hls_model.trace(x)
# QKeras per-layer activations via a multi-output sub-model
out_layers = [L.name for L in model.layers if L.name in hls_trace]
submodel = KModel(inputs=model.input, outputs=[model.get_layer(n).output for n in out_layers])
outs = submodel.predict(x, verbose=0)
keras_trace = {n: o for n, o in zip(out_layers, outs)}

print("=== per-layer max|QKeras - hls4ml| ===")
for k in hls_trace:
    if k in keras_trace:
        a = np.asarray(keras_trace[k]).reshape(-1)
        b = np.asarray(hls_trace[k]).reshape(-1)
        n = min(a.size, b.size)
        md = float(np.max(np.abs(a[:n] - b[:n]))) if n else float("nan")
        print("%-16s qk[%.3f..%.3f] hls[%.3f..%.3f] maxdiff=%.4f" % (
            k, float(a.min()), float(a.max()), float(b.min()), float(b.max()), md))
