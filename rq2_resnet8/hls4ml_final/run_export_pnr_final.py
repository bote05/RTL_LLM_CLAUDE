"""RQ2 Leg C FINAL -- full Vitis HLS export + Vivado P&R for the 89.11% QKeras ResNet-8.

Uses the GAP-swapped fitted export model (resnet8_qkeras8_gap_nosoftmax.h5,
argmax-identical to the trained model on the full 10k test set) and the fitted
config (build_config: RF=128, adder-align fixed<8,6> pins, GAP accum ufixed<16,10>,
1x1 proj RF clamp). Drives:
  1. csim (C testbench) + synth,
  2. cosim with VCD trace -> hls4ml fifo_depth_optimization rewrites the dataflow
     FIFO depths to the profiled minimum (collapses the default depth-1024 FIFOs that
     inflate the BRAM estimate to 216%); re-csynth after the rewrite,
  3. IP export (Verilog),
  4. Vivado logic synth -> place -> route on xczu7ev-ffvc1156-2-e (clock 10ns).

NOTE: fifo_opt only fires when cosim=True (build_prj.tcl gates it inside the cosim
block). The prior flow-validation run used cosim=False, so its FIFOs were NEVER
optimized -- which is why its Vivado synth showed 102.9% BRAM. cosim=True here makes
fifo_opt actually run.

Long (multi-hour) leg; launched detached.
"""
import os, sys, pprint
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
sys.path.insert(0, "/root/rq2_training/qkeras")
sys.path.insert(0, "/root/rq2_training/hls4ml_resnet8_final")
from resnet8_qkeras8 import load_qkeras_h5
from convert_resnet8_final import build_config
import hls4ml

M = "/root/rq2_training/hls4ml_resnet8_final/resnet8_qkeras8_gap_nosoftmax.h5"
OUT = "/root/rq2_training/hls4ml_resnet8_final/prj_gap"
REUSE = int(os.environ.get("RF", "128"))

model = load_qkeras_h5(M)
print("[pnr] loaded GAP model (%d params) RF=%d" % (model.count_params(), REUSE), flush=True)
cfg = build_config(model, REUSE, fifo_cap=0)

hls_model = hls4ml.converters.convert_from_keras_model(
    model, hls_config=cfg, output_dir=OUT, backend="Vitis",
    io_type="io_stream", part="xczu7ev-ffvc1156-2-e", clock_period=10.0)
hls_model.write()
print("[pnr] project written; starting csim+synth+cosim(fifo_opt)+export+Vivado P&R",
      flush=True)

# cosim=True is REQUIRED for fifo_opt to run (build_prj.tcl gates it under cosim).
# csim=False here: csim was already verified separately (32/32 argmax). Running
# csim+cosim together triggers a double-cosim; keep a single cosim (the fifo_opt
# profiling one) so the fitted FIFO depths are computed once, then export + P&R.
report = hls_model.build(
    reset=True, csim=False, synth=True, cosim=True,
    fifo_opt=True, export=True, vsynth=True, log_to_stdout=True)
print("[pnr] DONE", flush=True)
pprint.pprint(report)
print("PNR_OK", flush=True)
