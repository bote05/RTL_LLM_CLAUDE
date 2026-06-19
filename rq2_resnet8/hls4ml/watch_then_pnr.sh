#!/bin/bash
# Wait for the detached csynth to finish, capture its report numbers, then
# auto-launch the full export + Vivado P&R. Keeps Vitis serialized (one at a time).
set -u
DIR=/root/rq2_training/hls4ml_resnet8
RPT=$DIR/prj/myproject_prj/solution1/syn/report/myproject_csynth.rpt
SUMMARY=$DIR/CSYNTH_SUMMARY.txt

echo "[watch] waiting for csynth report..."
# Wait until csynth python exits AND the report exists (or csynth errored out).
while pgrep -f "run_csynth.py" >/dev/null 2>&1; do
  sleep 30
done
echo "[watch] run_csynth.py exited at $(date)"

if [ ! -f "$RPT" ]; then
  echo "[watch] NO csynth report produced -- csynth FAILED. Tail of log:" | tee "$SUMMARY"
  tail -40 "$DIR/csynth_run.log" | tee -a "$SUMMARY"
  echo "CSYNTH_FAILED" | tee -a "$SUMMARY"
  exit 1
fi

echo "[watch] csynth report found -- extracting numbers" | tee "$SUMMARY"
{
  echo "==== myproject_csynth.rpt key numbers ===="
  # Timing / latency / II / resources blocks
  sed -n '/== Performance Estimates/,/== Utilization Estimates/p' "$RPT"
  echo "---- Utilization ----"
  sed -n '/== Utilization Estimates/,/^$/p' "$RPT" | head -60
} | tee -a "$SUMMARY"
cp "$RPT" "$DIR/myproject_csynth.rpt.saved" 2>/dev/null
cp "$SUMMARY" /mnt/d/RTL_LLM_CLAUDE/rq2_resnet8/hls4ml/CSYNTH_SUMMARY.txt 2>/dev/null

echo "[watch] launching full export + Vivado P&R (detached child)..."
source /root/rq2_venv/bin/activate
source /tools/Xilinx/Vitis_HLS/2024.2/settings64.sh
export OMP_NUM_THREADS=6
cd "$DIR"
setsid nohup python run_export_pnr.py > pnr_run.log 2>&1 < /dev/null &
disown
sleep 5
pgrep -af "run_export_pnr.py" && echo "PNR_LAUNCHED" || echo "PNR_LAUNCH_FAILED"
