"""RQ2 Leg C -- full Vitis HLS export + Vivado P&R for the QKeras ResNet-8.

Runs AFTER csynth has produced its report. Reuses the SAME validated hls4ml
project (RF clamp + adder-align precision pin) and drives:
  1. fifo_opt cosim pass (profiles skip-FIFO occupancy -> minimal depths; avoids
     the residual-skip deadlock + BRAM blowup the recipe warns about), then
  2. Vitis HLS C/RTL synth + IP export (Verilog), then
  3. Vivado logic synth -> place -> route on xczu7ev-ffvc1156-2-e (vsynth=True).

This is the long (multi-hour) leg; launched detached by launch_export_pnr.sh.
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
print("[pnr] project re-written; starting FIFO-opt + synth + IP export + Vivado P&R",
      flush=True)

# fifo_opt=True : cosim-profiled skip-FIFO depth optimization (recipe requirement)
# export=True   : emit the Verilog IP
# vsynth=True   : Vivado logic synth + place + route on the target part
report = hls_model.build(
    reset=True, csim=False, synth=True, cosim=False,
    fifo_opt=True, export=True, vsynth=True, log_to_stdout=True)
print("[pnr] DONE", flush=True)
pprint.pprint(report)
print("PNR_OK", flush=True)
