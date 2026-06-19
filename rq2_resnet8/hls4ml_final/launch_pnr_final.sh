#!/bin/bash
# Fully daemonized full P&R launch (setsid + disown) so the multi-hour job
# survives the parent shell being reaped. Writes its own log.
set -u
LOG=/root/rq2_training/hls4ml_resnet8_final/pnr_final.log
cat > /root/rq2_training/hls4ml_resnet8_final/_pnr_inner.sh <<'INNER'
#!/bin/bash
source /root/rq2_venv/bin/activate
source /tools/Xilinx/Vitis_HLS/2024.2/settings64.sh
export OMP_NUM_THREADS=8
export RF=128
cd /root/rq2_training/hls4ml_resnet8_final
echo "[pnr_final] START $(date)"
python run_export_pnr_final.py
echo "[pnr_final] EXIT=$? $(date)"
INNER
chmod +x /root/rq2_training/hls4ml_resnet8_final/_pnr_inner.sh
setsid bash /root/rq2_training/hls4ml_resnet8_final/_pnr_inner.sh > "$LOG" 2>&1 < /dev/null &
disown
sleep 3
echo "launched; log=$LOG"
pgrep -af "run_export_pnr_final|_pnr_inner" | grep -v pgrep | head
