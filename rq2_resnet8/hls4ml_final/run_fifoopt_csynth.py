"""Run hls4ml fifo_opt (cosim-profiled FIFO depth optimization) + csynth on the
GAP-swapped fitted project, and print the post-fifo_opt resource estimate.

The csynth BRAM estimate is dominated by default depth-1024 dataflow FIFOs
(1232 of 1352 BRAM_18K). fifo_opt profiles real stream occupancy via cosim and
rewrites the depths to the minimum, collapsing that BRAM. This is the legitimate
"fitted estimate" -- the same fifo_opt the full P&R build runs. We do it as a
standalone pass first to CONFIRM BRAM < 100% before committing to multi-hour P&R.

build(fifo_opt=True) implies a cosim run (needs csim=True for the C testbench).
"""
import os, sys, pprint
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
sys.path.insert(0, "/root/rq2_training/qkeras")
sys.path.insert(0, "/root/rq2_training/hls4ml_resnet8_final")
from resnet8_qkeras8 import load_qkeras_h5
from convert_resnet8_final import build_config, load_cifar10_test
import hls4ml

M = "/root/rq2_training/hls4ml_resnet8_final/resnet8_qkeras8_gap_nosoftmax.h5"
OUT = "/root/rq2_training/hls4ml_resnet8_final/prj_gap"
REUSE = int(os.environ.get("RF", "128"))

model = load_qkeras_h5(M)
print("[fifoopt] loaded GAP model (%d params) RF=%d" % (model.count_params(), REUSE), flush=True)
cfg = build_config(model, REUSE, fifo_cap=0)

hls_model = hls4ml.converters.convert_from_keras_model(
    model, hls_config=cfg, output_dir=OUT, backend="Vitis",
    io_type="io_stream", part="xczu7ev-ffvc1156-2-e", clock_period=10.0)
hls_model.compile()
print("[fifoopt] converted+compiled; running fifo_opt cosim + csynth ...", flush=True)

# fifo_opt requires a cosim; cosim needs csim. Use a tiny stimulus set.
report = hls_model.build(
    reset=True, csim=True, synth=True, cosim=True,
    fifo_opt=True, export=False, vsynth=False, log_to_stdout=True)
print("[fifoopt] DONE", flush=True)
pprint.pprint(report)
print("FIFOOPT_CSYNTH_OK", flush=True)
