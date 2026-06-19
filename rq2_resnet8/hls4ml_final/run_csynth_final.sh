#!/bin/bash
set -u
source /root/rq2_venv/bin/activate
source /tools/Xilinx/Vitis_HLS/2024.2/settings64.sh
export OMP_NUM_THREADS=8
cd /root/rq2_training/hls4ml_resnet8_final
echo "[csynth_final] RF=128 start Sat Jun 13 04:54:40 PM CEST 2026"
python convert_resnet8_final.py --reuse 128 --do-csynth 2>&1
echo "[csynth_final] EXIT=0 Sat Jun 13 04:54:40 PM CEST 2026"
