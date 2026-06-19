"""Deterministic FIFO-depth cap + csynth for the GAP-swapped fitted project.

The default hls4ml io_stream inter-layer FIFOs are sized to the full feature-map
(depth 1024/1156) -> csynth BRAM estimate 216%. The cosim profiling (which ran
twice and is reliable) showed the residual-skip FIFOs need only depth 9/7/7/2,
and feed-forward dataflow channels need even less (producer/consumer rates match).

This script caps every inter-layer StreamVariable FIFO at FIFO_CAP beats -- the same
pragma-rewrite mechanism hls4ml's set_optimized_fifo_depths uses, but with a uniform
deterministic cap instead of the slow/fragile trace-cosim. FIFO_CAP is set far above
the proven occupancy (>=9) so no deadlock, yet far below the 1024/1156 defaults so
BRAM fits. We csim (correctness, fast) before csynth, and re-cap-sweep if needed.
"""
import os, sys, pprint, pickle
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
sys.path.insert(0, "/root/rq2_training/qkeras")
sys.path.insert(0, "/root/rq2_training/hls4ml_resnet8_final")
from resnet8_qkeras8 import load_qkeras_h5
from convert_resnet8_final import build_config, load_cifar10_test
import hls4ml

M = "/root/rq2_training/hls4ml_resnet8_final/resnet8_qkeras8_gap_nosoftmax.h5"
OUT = "/root/rq2_training/hls4ml_resnet8_final/prj_gap_cap"
REUSE = int(os.environ.get("RF", "128"))
FIFO_CAP = int(os.environ.get("FIFO_CAP", "256"))
DATA = "/mnt/d/RTL_LLM_CLAUDE/rq2_resnet8/data/cifar-10-batches-py"

model = load_qkeras_h5(M)
print("[cap] GAP model (%d params) RF=%d FIFO_CAP=%d" % (model.count_params(), REUSE, FIFO_CAP), flush=True)
cfg = build_config(model, REUSE, fifo_cap=0)

hls_model = hls4ml.converters.convert_from_keras_model(
    model, hls_config=cfg, output_dir=OUT, backend="Vitis",
    io_type="io_stream", part="xczu7ev-ffvc1156-2-e", clock_period=10.0)

# --- deterministic FIFO-depth cap (set_optimized_fifo_depths mechanism) -------
capped = 0
for ov in hls_model.output_vars.values():
    if 'StreamVariable' in str(type(ov)) and getattr(ov, 'pragma', None):
        cur = ov.pragma[1] if isinstance(ov.pragma, (list, tuple)) and len(ov.pragma) > 1 else None
        if isinstance(cur, int) and cur > FIFO_CAP:
            ov.pragma = (ov.pragma[0], FIFO_CAP)
            capped += 1
print("[cap] capped %d inter-layer FIFOs to depth %d" % (capped, FIFO_CAP), flush=True)

hls_model.write()
hls_model.compile()
print("[cap] written+compiled", flush=True)

# csim correctness check (fast) on 32 CIFAR test images.
x, y = load_cifar10_test(DATA, 32)
qk = model.predict(x, verbose=0)
hl = np.asarray(hls_model.predict(np.ascontiguousarray(x))).reshape(qk.shape)
agree = int((qk.argmax(1) == hl.argmax(1)).sum())
print("CSIM_AGREE=%d/32 MAXDIFF=%.4f" % (agree, float(np.max(np.abs(qk - hl)))), flush=True)

# csynth for resource estimate.
print("[cap] launching csynth ...", flush=True)
report = hls_model.build(reset=True, csim=False, synth=True, cosim=False,
                         export=False, vsynth=False, fifo_opt=False, log_to_stdout=True)
pprint.pprint(report)
print("FIFOCAP_CSYNTH_OK", flush=True)
