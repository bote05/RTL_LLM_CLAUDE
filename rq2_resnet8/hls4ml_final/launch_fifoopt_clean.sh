#!/bin/bash
# Fully daemonized launch (setsid + disown) so the job survives the parent shell
# being reaped. Writes its own log; we poll the log, never the process tree.
set -u
LOG=/root/rq2_training/hls4ml_resnet8_final/fifoopt_clean.log
cat > /root/rq2_training/hls4ml_resnet8_final/_fifoopt_inner.sh <<'INNER'
#!/bin/bash
source /root/rq2_venv/bin/activate
source /tools/Xilinx/Vitis_HLS/2024.2/settings64.sh
export OMP_NUM_THREADS=8
export RF=128
cd /root/rq2_training/hls4ml_resnet8_final
echo "[fifoopt_clean] START $(date)"
python run_fifoopt_clean.py
echo "[fifoopt_clean] EXIT=$? $(date)"
INNER
chmod +x /root/rq2_training/hls4ml_resnet8_final/_fifoopt_inner.sh
setsid bash /root/rq2_training/hls4ml_resnet8_final/_fifoopt_inner.sh > "$LOG" 2>&1 < /dev/null &
disown
sleep 3
echo "launched pid-group; log=$LOG"
pgrep -af "run_fifoopt_clean|_fifoopt_inner" | grep -v pgrep | head
