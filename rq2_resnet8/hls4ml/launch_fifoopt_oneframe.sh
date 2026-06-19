#!/bin/bash
# =============================================================================
# ONE-FRAME (2-sample, API-minimum) UNCAPPED FIFO-depth-optimization cosim.
# This is the OVERNIGHT job the orchestrator launches as a tracked background
# process. It survives the parent shell (setsid + disown) and writes its own log.
#
# THE FIX vs the 5h hang: run_fifoopt_oneframe.py provides real testbench data
# (input_data_tb/output_data_tb) so the cosim loops over EXACTLY N_PROFILE=2
# frames instead of the hard-coded 5-sample default that made the prior run 5x
# too slow. FIFOs are UNCAPPED (profiling_fifo_depth=4096) during profiling.
#
# Simulator: XSIM (the only RTL simulator wired into this Vitis HLS 2024.2
# install; no Verilator/Questa/vsim cosim path is configured). 2 frames at
# ~175,714 cyc/frame, xsim ~10 cyc/s  =>  ~2 h wall.
#
# DONE looks like:  the log ends with "FIFOOPT_ONEFRAME_OK" and writes
#   <FIFO_OUT>/fifo_depths.json   (per-FIFO initial vs optimized depths)
#   <FIFO_OUT>/myproject_prj/solution1/.autopilot/db/channel_depth_info/channel.zip
#   + the in-place re-csynth resource report (BRAM should fall from ~216% toward ~45%).
# =============================================================================
set -u
export FIFO_OUT="${FIFO_OUT:-/root/rq2_training/hls4ml_resnet8_final/prj_gap_fifo1}"
export RF="${RF:-128}"
export N_PROFILE="${N_PROFILE:-2}"
export PROFILE_DEPTH="${PROFILE_DEPTH:-4096}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
LOG="${LOG:-/root/rq2_training/hls4ml_resnet8_final/fifoopt_oneframe.log}"

cat > /root/rq2_training/hls4ml_resnet8_final/_fifoopt_oneframe_inner.sh <<INNER
#!/bin/bash
source /root/rq2_venv/bin/activate
source /tools/Xilinx/Vitis_HLS/2024.2/settings64.sh
export OMP_NUM_THREADS=${OMP_NUM_THREADS}
export RF=${RF}
export N_PROFILE=${N_PROFILE}
export PROFILE_DEPTH=${PROFILE_DEPTH}
export FIFO_OUT=${FIFO_OUT}
cd /root/rq2_training/hls4ml_resnet8_final
echo "[fifoopt1] START \$(date) FIFO_OUT=${FIFO_OUT} N_PROFILE=${N_PROFILE} DEPTH=${PROFILE_DEPTH}"
python run_fifoopt_oneframe.py
echo "[fifoopt1] EXIT=\$? \$(date)"
INNER
chmod +x /root/rq2_training/hls4ml_resnet8_final/_fifoopt_oneframe_inner.sh

setsid bash /root/rq2_training/hls4ml_resnet8_final/_fifoopt_oneframe_inner.sh > "$LOG" 2>&1 < /dev/null &
disown
sleep 3
echo "launched one-frame FIFO-opt cosim; log=$LOG"
pgrep -af "run_fifoopt_oneframe|_fifoopt_oneframe_inner" | grep -v pgrep | head
