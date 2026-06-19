"""One-frame UNCAPPED FIFO-depth-optimization profiling for the 89.11% QKeras
ResNet-8 (GAP-swapped) via the hls4ml vitis:fifo_depth_optimization FLOW.

=============================================================================
WHY THIS SCRIPT EXISTS  (the 5h-hang fix)
=============================================================================
The prior cosim hung at "0/5 frames after 5h". ROOT CAUSE (confirmed, NOT
deadlock): run_fifoopt_clean.py never set InputData, so write_test_bench wrote
NO tb_input_features.dat. The generated myproject_test.cpp then fell into its
else branch -> "const unsigned NUM_TEST_SAMPLES = 5;" -> the top function ran
FIVE times in xsim. At ~175,714 cycles/frame and xsim ~10 cyc/s that is ~5 h
PER FRAME, so frame 0 of 5 never finished. (The prior fifoopt_clean.log even
prints "INFO: Unable to open input/predictions file, using default input."
right before the 5 default rows -> proof it ran the 5-sample default.)

THE FIX implemented here:
  1. Provide REAL testbench data via input_data_tb / output_data_tb so the
     testbench loops over EXACTLY the samples we give it (one line per sample),
     NOT the hard-coded 5.
  2. Use N_PROFILE = 2 samples. hls4ml fifo_depth_optimization REQUIRES the top
     function to execute "at least twice" (execute_cosim_to_profile_fifos
     docstring). A single forward pass through a feed-forward CNN already drives
     every inter-layer FIFO to its peak occupancy; 2 is the minimum that both
     satisfies the API and observes the true peak. 2 frames ~= 2/5 of the prior
     work ~= ~2 h (vs 5 frames). One frame would TRIP the "at least twice"
     requirement, so 2 is the correct minimum, not 1.
  3. UNCAPPED FIFOs during profiling: profiling_fifo_depth = 4096 (>> the 1156
     peak seen). NO --fifo-cap. Capping during profiling is what causes the
     skip-path deadlock; depths are set ONLY AFTER profiling, from the measured
     values the cosim writes.

OUTPUT: hls4ml writes <OUT>/fifo_depths.json  {fifo: {initial, optimized}} and
rewrites the depth= pragmas in firmware/myproject.cpp in place. The cosim channel
CSVs live under
  <OUT>/myproject_prj/solution1/.autopilot/db/channel_depth_info/channel.zip
=============================================================================
"""
import os, sys, json, pprint
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np

sys.path.insert(0, "/root/rq2_training/qkeras")
sys.path.insert(0, "/root/rq2_training/hls4ml_resnet8_final")
from resnet8_qkeras8 import load_qkeras_h5
from convert_resnet8_final import build_config, load_cifar10_test
import hls4ml

M = "/root/rq2_training/hls4ml_resnet8_final/resnet8_qkeras8_gap_nosoftmax.h5"
OUT = os.environ.get("FIFO_OUT", "/root/rq2_training/hls4ml_resnet8_final/prj_gap_fifo1")
REUSE = int(os.environ.get("RF", "128"))
DATA = "/mnt/d/RTL_LLM_CLAUDE/rq2_resnet8/data/cifar-10-batches-py"
N_PROFILE = int(os.environ.get("N_PROFILE", "2"))            # API minimum: >= 2
PROFILE_DEPTH = int(os.environ.get("PROFILE_DEPTH", "4096"))  # >> 1156 peak; UNCAPPED

# --- testbench stimulus: N_PROFILE real CIFAR-10 frames (one line == one frame) -
x, y = load_cifar10_test(DATA, N_PROFILE)
x = np.ascontiguousarray(x.astype("float32"))
model = load_qkeras_h5(M)
yhat = model.predict(x, verbose=0).astype("float32")        # reference logits
os.makedirs(OUT, exist_ok=True)
TB_X = os.path.join(OUT, "tb_input_features.npy")
TB_Y = os.path.join(OUT, "tb_output_predictions.npy")
np.save(TB_X, x)
np.save(TB_Y, yhat)
print("[fifoopt1] wrote %d-sample testbench: x%s -> y%s  (UNCAPPED depth=%d)"
      % (N_PROFILE, x.shape, yhat.shape, PROFILE_DEPTH), flush=True)
print("[fifoopt1] GAP model %d params RF=%d" % (model.count_params(), REUSE), flush=True)

cfg = build_config(model, REUSE, fifo_cap=0)                # fifo_cap=0 => NO static cap
cfg.setdefault("Model", {})
# IMPORTANT: do NOT set cfg["Flows"] = ["vitis:fifo_depth_optimization"].
# That REPLACES the default build flow, so conversion would (a) launch the cosim
# at convert-time before write(), and (b) skip the normal firmware-generation flow.
# Instead convert with the DEFAULT flow (builds firmware), write(), then call
# apply_flow("vitis:fifo_depth_optimization") explicitly (it requires vitis:ip,
# which is already applied, so only the cosim profiling transform runs).

# input_data_tb / output_data_tb make write_test_bench emit tb_data/*.dat so the
# cosim loops over EXACTLY these N_PROFILE samples (NOT the 5-sample default).
hls_model = hls4ml.converters.convert_from_keras_model(
    model, hls_config=cfg, output_dir=OUT, backend="Vitis",
    io_type="io_stream", part="xczu7ev-ffvc1156-2-e", clock_period=10.0,
    input_data_tb=TB_X, output_data_tb=TB_Y)
hls_model.write()

# UNCAPPED profiling depth on the registry optimizer instance (apply_flow uses it).
from hls4ml.model.optimizer import get_optimizer
opt = get_optimizer("vitis:fifo_depth_optimization")
opt.profiling_fifo_depth = PROFILE_DEPTH
print("[fifoopt1] profiling_fifo_depth=%d (UNCAPPED); single cosim over %d frames ..."
      % (opt.profiling_fifo_depth, N_PROFILE), flush=True)

hls_model.apply_flow("vitis:fifo_depth_optimization")
print("[fifoopt1] flow applied", flush=True)

fjson = os.path.join(OUT, "fifo_depths.json")
if os.path.exists(fjson):
    with open(fjson) as f:
        depths = json.load(f)
    tot_init = sum(d["initial"] for d in depths.values())
    tot_opt = sum(d["optimized"] for d in depths.values())
    peak = max((d["optimized"] for d in depths.values()), default=0)
    print("[fifoopt1] %d FIFOs: total depth %d -> %d ; peak optimized=%d"
          % (len(depths), tot_init, tot_opt, peak), flush=True)

# Re-csynth with the MEASURED optimized depths to capture the fitted estimate.
print("[fifoopt1] re-csynth with optimized FIFO depths ...", flush=True)
report = hls_model.build(reset=True, csim=False, synth=True, cosim=False,
                         export=False, vsynth=False, fifo_opt=False, log_to_stdout=True)
pprint.pprint(report)
print("FIFOOPT_ONEFRAME_OK", flush=True)
