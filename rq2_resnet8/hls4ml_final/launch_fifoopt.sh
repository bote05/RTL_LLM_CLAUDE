#!/bin/bash
set -u
source /root/rq2_venv/bin/activate
source /tools/Xilinx/Vitis_HLS/2024.2/settings64.sh
export OMP_NUM_THREADS=8
export RF="${RF:-128}"
cd /root/rq2_training/hls4ml_resnet8_final
echo "[fifoopt] RF=${RF} start $(date)"
python run_fifoopt_csynth.py 2>&1
echo "[fifoopt] EXIT=$? $(date)"
