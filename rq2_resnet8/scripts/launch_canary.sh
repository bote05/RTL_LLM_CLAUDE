#!/usr/bin/env bash
# FINN ZCU104 canary launcher (bare-metal FINN v0.10.1, no docker).
# Env: /root/.venv-hls4ml python 3.10.12 + PYTHONPATH onto the finn clone,
# plus /root/finn_canary/pydeps (numpy 1.24.1 pin + finn-only deps).

cd /root/finn_canary || exit 1

# --- FINN env ---
export FINN_XILINX_PATH=/tools/Xilinx
export FINN_XILINX_VERSION=2024.2
export FINN_ROOT=/root/tools/finn
export FINN_BUILD_DIR=/root/finn_canary/build_tmp
mkdir -p "$FINN_BUILD_DIR"
export NUM_DEFAULT_WORKERS=4

# --- Xilinx 2024.2 tools ---
source /tools/Xilinx/Vitis/2024.2/settings64.sh
source /tools/Xilinx/Vivado/2024.2/settings64.sh
source /tools/Xilinx/Vitis_HLS/2024.2/settings64.sh
# FINN bare-metal expects the docker-entrypoint vars too:
export VIVADO_PATH=/tools/Xilinx/Vivado/2024.2
export VITIS_PATH=/tools/Xilinx/Vitis/2024.2
export HLS_PATH=/tools/Xilinx/Vitis_HLS/2024.2

# --- Python env (verified by import sanity check) ---
export PYTHONPATH=/root/finn_canary/pydeps:/root/tools/finn/src:/root/tools/finn/deps/qonnx/src:/root/tools/finn/deps/finn-experimental/src:/root/tools/finn/deps/brevitas/src:/root/tools/finn/deps/pyverilator
export PYTHONUNBUFFERED=1
PYBIN=/root/.venv-hls4ml/bin/python

set -o pipefail
"$PYBIN" build_tfc_zcu104.py 2>&1 | tee canary.log
exit "${PIPESTATUS[0]}"
