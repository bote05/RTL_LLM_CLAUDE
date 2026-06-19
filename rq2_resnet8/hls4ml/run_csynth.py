"""Run C synthesis on the already-converted hls4ml project and print the report.

Re-converts with the SAME corrected config (RF clamp + adder-align precision pin)
so the project on disk is the validated one, then runs hls_model.build(synth=True).
"""
import os, sys, pprint
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np
sys.path.insert(0, "/root/rq2_training/qkeras")
from resnet8_qkeras8 import load_qkeras_h5
import hls4ml

M = "/root/rq2_training/qkeras/resnet8_qkeras8_best_nosoftmax.h5"
OUT = "/root/rq2_training/hls4ml_resnet8/prj"
REUSE = 72

model = load_qkeras_h5(M)
cfg = hls4ml.utils.config_from_keras_model(
    model, granularity="name", backend="Vitis",
    default_precision="ap_fixed<16,6>", default_reuse_factor=REUSE)
cfg["Model"]["Strategy"] = "Resource"
cfg["Model"]["ReuseFactor"] = REUSE
for ln, lc in cfg["LayerName"].items():
    lc["ReuseFactor"] = REUSE
    lc["Strategy"] = "Resource"
for nm, n_in in {"s2_proj": 16, "s3_proj": 32}.items():
    if nm in cfg["LayerName"]:
        cfg["LayerName"][nm]["ReuseFactor"] = n_in
for nm in ["s1_bn2", "s2_bn2", "s2_proj", "s3_bn2", "s3_proj"]:
    if nm in cfg["LayerName"]:
        p = cfg["LayerName"][nm].setdefault("Precision", {})
        if not isinstance(p, dict):
            p = {"result": p}
            cfg["LayerName"][nm]["Precision"] = p
        p["result"] = "fixed<8,3,RND_CONV,SAT,0>"

hls_model = hls4ml.converters.convert_from_keras_model(
    model, hls_config=cfg, output_dir=OUT, backend="Vitis",
    io_type="io_stream", part="xczu7ev-ffvc1156-2-e", clock_period=10.0)
hls_model.write()
print("[csynth] project written; launching C synthesis...", flush=True)

report = hls_model.build(reset=True, csim=False, synth=True, cosim=False,
                         export=False, vsynth=False, fifo_opt=False,
                         log_to_stdout=True)
print("[csynth] DONE", flush=True)
pprint.pprint(report)
print("CSYNTH_OK", flush=True)
