"""Clean FIFO-depth optimization for the GAP-swapped fitted project via the
hls4ml vitis:fifo_depth_optimization FLOW (single cosim, no double-cosim).

Mechanism (hls4ml/backends/vitis/passes/fifo_depth_optimization.py):
  initialize_large_fifos -> execute_cosim_to_profile_fifos (ONE cosim) ->
  get_vitis_optimized_fifo_depths (reads per-FIFO CSVs) -> rewrite pragmas ->
  set_optimized_fifo_depths. Writes fifo_depths.json. Then we re-csynth to capture
  the FITTED resource estimate (BRAM should collapse from 216% as the default
  depth-1024/1156 dataflow FIFOs shrink to their profiled minima -- cosim already
  showed the residual skips need only depth 9/7/7/2).

Run via the optimizer flow so it does exactly one clean cosim (avoids the
double-cosim crash from build(csim=True, cosim=True, fifo_opt=True)).
"""
import os, sys, json, pprint
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
sys.path.insert(0, "/root/rq2_training/qkeras")
sys.path.insert(0, "/root/rq2_training/hls4ml_resnet8_final")
from resnet8_qkeras8 import load_qkeras_h5
from convert_resnet8_final import build_config
import hls4ml

M = "/root/rq2_training/hls4ml_resnet8_final/resnet8_qkeras8_gap_nosoftmax.h5"
OUT = "/root/rq2_training/hls4ml_resnet8_final/prj_gap_fifo"
REUSE = int(os.environ.get("RF", "128"))
PROFILE_DEPTH = 4096  # large profiling depth; must exceed peak occupancy (1156 seen)

model = load_qkeras_h5(M)
print("[fifoopt] GAP model (%d params) RF=%d profile_depth=%d" % (model.count_params(), REUSE, PROFILE_DEPTH), flush=True)
cfg = build_config(model, REUSE, fifo_cap=0)
# Enable the fifo-depth optimization flow at convert time.
cfg["Flows"] = ["vitis:fifo_depth_optimization"]
cfg.setdefault("Model", {})

hls_model = hls4ml.converters.convert_from_keras_model(
    model, hls_config=cfg, output_dir=OUT, backend="Vitis",
    io_type="io_stream", part="xczu7ev-ffvc1156-2-e", clock_period=10.0)
hls_model.write()

# Set the profiling depth on the registered optimizer instance (apply_flow uses
# the registry instance), then run the flow. Peak occupancy seen = 1156, so a
# 4096 profiling depth comfortably exceeds it while keeping cosim FIFOs modest.
from hls4ml.model.optimizer import get_optimizer
opt = get_optimizer("vitis:fifo_depth_optimization")
opt.profiling_fifo_depth = PROFILE_DEPTH
print("[fifoopt] profiling_fifo_depth=%d; running vitis:fifo_depth_optimization (single cosim) ..."
      % opt.profiling_fifo_depth, flush=True)
hls_model.apply_flow("vitis:fifo_depth_optimization")
print("[fifoopt] flow applied", flush=True)

# Show the optimized depths
fjson = os.path.join(OUT, "fifo_depths.json")
if os.path.exists(fjson):
    with open(fjson) as f:
        depths = json.load(f)
    n = len(depths)
    tot_init = sum(d["initial"] for d in depths.values())
    tot_opt = sum(d["optimized"] for d in depths.values())
    print("[fifoopt] %d FIFOs: total depth %d -> %d" % (n, tot_init, tot_opt), flush=True)

# Re-csynth to capture the fitted resource estimate.
print("[fifoopt] re-csynth with optimized FIFO depths ...", flush=True)
report = hls_model.build(reset=True, csim=False, synth=True, cosim=False,
                         export=False, vsynth=False, fifo_opt=False, log_to_stdout=True)
pprint.pprint(report)
print("FIFOOPT_CLEAN_OK", flush=True)
