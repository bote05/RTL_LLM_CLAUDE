#!/bin/bash
# 1-epoch GPU smoke for the improved QKeras ResNet-8 retrain.
set -u
OUT=/root/rq2_training/qkeras_gpu
mkdir -p "$OUT/smoke"

# background GPU util sampler
(
  for i in $(seq 1 40); do
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    sleep 2
  done > "$OUT/smoke_gpu_util.log" 2>&1
) &
SAMP=$!

cd "$OUT"
/root/rq2_gpu_venv/bin/python train_resnet8_qkeras_gpu.py \
  --epochs 1 --out-dir "$OUT/smoke" --threads 6 \
  > "$OUT/smoke.log" 2>&1

kill "$SAMP" 2>/dev/null
echo "=== SMOKE TAIL ==="
grep -vE "Unable to register|oneDNN|cpu_feature|AVX2|rebuild TensorFlow|UserWarning|saving_api|warnings.warn|compile_metrics|keras.constraints" "$OUT/smoke.log" | tail -25
echo "=== GPU UTIL SAMPLES (top, during smoke) ==="
sort -t, -k1 -rn "$OUT/smoke_gpu_util.log" | head -8
