"""LIGHT validation of the one-frame FIFO-opt wiring -- NO synth, NO cosim.

Confirms statically + via fast C-sim (compiled bridge, no Vitis) that:
  (1) the project converts with input_data_tb/output_data_tb set,
  (2) write() emits tb_data/tb_input_features.dat with EXACTLY N_PROFILE lines
      (so cosim loops N_PROFILE times, NOT the 5-sample default),
  (3) the generated myproject_test.cpp will OPEN that .dat (so it takes the real
      data branch, not the "Unable to open ... using default input" 5-loop),
  (4) the io_stream config is in place and FIFOs are NOT statically capped
      (fifo_cap=0 -> the default frame-depth pragmas, which the cosim profiles),
  (5) Python C-sim argmax matches QKeras on the 2 profiling frames.
"""
import os, sys, json, glob
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

sys.path.insert(0, "/root/rq2_training/qkeras")
sys.path.insert(0, "/root/rq2_training/hls4ml_resnet8_final")
from resnet8_qkeras8 import load_qkeras_h5
from convert_resnet8_final import build_config, load_cifar10_test
import hls4ml

M = "/root/rq2_training/hls4ml_resnet8_final/resnet8_qkeras8_gap_nosoftmax.h5"
OUT = "/root/rq2_training/hls4ml_resnet8_final/prj_gap_fifo1_val"
REUSE = int(os.environ.get("RF", "128"))
DATA = "/mnt/d/RTL_LLM_CLAUDE/rq2_resnet8/data/cifar-10-batches-py"
N_PROFILE = int(os.environ.get("N_PROFILE", "2"))

x, y = load_cifar10_test(DATA, N_PROFILE)
x = np.ascontiguousarray(x.astype("float32"))
model = load_qkeras_h5(M)
yhat = model.predict(x, verbose=0).astype("float32")
os.makedirs(OUT, exist_ok=True)
TB_X = os.path.join(OUT, "tb_input_features.npy")
TB_Y = os.path.join(OUT, "tb_output_predictions.npy")
np.save(TB_X, x); np.save(TB_Y, yhat)

cfg = build_config(model, REUSE, fifo_cap=0)
# Do NOT set cfg["Flows"] -- that would launch the cosim at convert-time. Convert
# with the default flow; the real run calls apply_flow() explicitly after write().

hls_model = hls4ml.converters.convert_from_keras_model(
    model, hls_config=cfg, output_dir=OUT, backend="Vitis",
    io_type="io_stream", part="xczu7ev-ffvc1156-2-e", clock_period=10.0,
    input_data_tb=TB_X, output_data_tb=TB_Y)
hls_model.write()
print("[val] convert+write OK", flush=True)

# (1)/(4) config checks
io = hls_model.config.get_config_value("IOType")
in_data = hls_model.config.get_config_value("InputData")
print("[val] IOType=%s  InputData=%s" % (io, in_data), flush=True)
assert io == "io_stream", "IOType must be io_stream for FIFO-opt"
assert in_data and in_data.endswith(".npy"), "InputData must be set"

# (2) the written .dat must have exactly N_PROFILE lines
datf = os.path.join(OUT, "tb_data", "tb_input_features.dat")
with open(datf) as f:
    lines = [l for l in f.read().splitlines() if l.strip()]
print("[val] tb_input_features.dat lines=%d (expect %d)" % (len(lines), N_PROFILE), flush=True)
assert len(lines) == N_PROFILE, "tb .dat must have exactly N_PROFILE samples"

# (3) the generated testbench opens that .dat (real-data branch, not 5-default)
tbcpp = os.path.join(OUT, "myproject_test.cpp")
tb = open(tbcpp).read()
assert 'tb_data/tb_input_features.dat' in tb, "testbench must read the input .dat"
print("[val] testbench reads tb_data/tb_input_features.dat -> real-data branch (NOT 5-default)", flush=True)

# (4) FIFOs uncapped: the firmware STREAM pragmas should still be at frame depths
fw = glob.glob(os.path.join(OUT, "firmware", "myproject.cpp"))
if fw:
    txt = open(fw[0]).read()
    import re
    depths = sorted({int(m) for m in re.findall(r"#pragma HLS STREAM variable=\S+ depth=(\d+)", txt)})
    print("[val] firmware STREAM depths present (uncapped frame depths): %s" % depths[-8:], flush=True)
    assert max(depths) >= 1024, "FIFOs should be at frame depth (uncapped) pre-profiling"

# (5) Python C-sim parity on the 2 profiling frames (fast; uses compiled bridge)
hls_model.compile()
hls_logits = np.asarray(hls_model.predict(x)).reshape(yhat.shape)
agree = int((hls_logits.argmax(1) == yhat.argmax(1)).sum())
maxd = float(np.max(np.abs(hls_logits - yhat)))
print("[val] CSIM argmax agree=%d/%d  max|logit diff|=%.4f" % (agree, N_PROFILE, maxd), flush=True)
assert agree == N_PROFILE, "C-sim argmax must match QKeras on the profiling frames"

print("VALIDATE_ONEFRAME_OK", flush=True)
