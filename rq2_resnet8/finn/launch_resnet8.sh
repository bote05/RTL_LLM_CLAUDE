#!/usr/bin/env bash
# FINN ZCU104 ResNet-8 launcher (bare-metal FINN v0.10.1, no docker).
# Cloned from /root/finn_canary/launch_canary.sh (the PROVEN canary harness).

cd /root/rq2_training/finn_resnet8 || exit 1

# --- FINN env ---
export FINN_XILINX_PATH=/tools/Xilinx
export FINN_XILINX_VERSION=2024.2
export FINN_ROOT=/root/tools/finn
export FINN_BUILD_DIR=/root/rq2_training/finn_resnet8/build_tmp
mkdir -p "$FINN_BUILD_DIR"
# Share the box: cap workers (FINN Vivado synth + hls4ml + GPU train run in parallel)
export NUM_DEFAULT_WORKERS=6

# --- Xilinx 2024.2 tools ---
source /tools/Xilinx/Vitis/2024.2/settings64.sh
source /tools/Xilinx/Vivado/2024.2/settings64.sh
source /tools/Xilinx/Vitis_HLS/2024.2/settings64.sh
export VIVADO_PATH=/tools/Xilinx/Vivado/2024.2
export VITIS_PATH=/tools/Xilinx/Vitis/2024.2
export HLS_PATH=/tools/Xilinx/Vitis_HLS/2024.2

# --- Python env (verified by import sanity check) ---
export PYTHONPATH=/root/finn_canary/pydeps:/root/tools/finn/src:/root/tools/finn/deps/qonnx/src:/root/tools/finn/deps/finn-experimental/src:/root/tools/finn/deps/brevitas/src:/root/tools/finn/deps/pyverilator
export PYTHONUNBUFFERED=1
PYBIN=/root/.venv-hls4ml/bin/python

# RESNET8_STOP_STEP / RESNET8_VERIFY passed through from caller's env.
set -o pipefail
"$PYBIN" /root/rq2_training/finn_resnet8/build_resnet8_zcu104.py 2>&1 | tee resnet8_build.log
exit "${PIPESTATUS[0]}"
