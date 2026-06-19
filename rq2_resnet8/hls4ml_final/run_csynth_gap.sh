#!/bin/bash
set -u
source /root/rq2_venv/bin/activate
source /tools/Xilinx/Vitis_HLS/2024.2/settings64.sh
export OMP_NUM_THREADS=8
cd /root/rq2_training/hls4ml_resnet8_final
RF="${RF:-128}"
FIFOCAP="${FIFOCAP:-0}"
echo "[csynth_gap] RF=${RF} FIFOCAP=${FIFOCAP} start $(date)"
python convert_resnet8_final.py \
  --model /root/rq2_training/hls4ml_resnet8_final/resnet8_qkeras8_gap_nosoftmax.h5 \
  --out-dir /root/rq2_training/hls4ml_resnet8_final/prj_gap \
  --reuse "${RF}" --fifo-cap "${FIFOCAP}" --do-csynth 2>&1
echo "[csynth_gap] EXIT=$? $(date)"
