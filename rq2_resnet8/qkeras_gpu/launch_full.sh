#!/bin/bash
# Launch the FULL improved QKeras ResNet-8 GPU retrain, detached + resilient to
# wsl.exe session teardown (setsid nohup ... & disown; then settle + pgrep proof).
set -u
OUT=/root/rq2_training/qkeras_gpu
mkdir -p "$OUT"
# fresh history (append=True in CSVLogger; start clean for the full run)
rm -f "$OUT/train_history.csv" "$OUT/full.log"

cd "$OUT"
setsid nohup /root/rq2_gpu_venv/bin/python train_resnet8_qkeras_gpu.py \
  --epochs 500 --batch-size 128 --lr 0.001 --warmup-epochs 5 \
  --out-dir "$OUT" --threads 6 --ckpt-every 50 --seed 42 \
  > "$OUT/full.log" 2>&1 &
disown
PID=$!
sleep 6
echo "LAUNCHED_PID=$PID"
echo "--- pgrep proof ---"
pgrep -af "train_resnet8_qkeras_gpu" || echo "NOT_FOUND"
echo "--- log head ---"
head -8 "$OUT/full.log" 2>/dev/null | grep -vE "Unable to register|oneDNN|cpu_feature|AVX2|rebuild"
