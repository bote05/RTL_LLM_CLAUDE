#!/bin/bash
RPT=/root/rq2_training/hls4ml_resnet8/prj/myproject_prj/solution1/syn/report/myproject_csynth.rpt
LOG=/root/rq2_training/hls4ml_resnet8/csynth_run.log
i=0
while [ "$i" -lt 16 ]; do
  if [ -f "$RPT" ]; then echo "REPORT EXISTS"; break; fi
  if ! pgrep -f run_csynth.py >/dev/null; then echo "CSYNTH EXITED"; break; fi
  sleep 30
  i=$((i + 1))
done
echo "now: $(date +%T)"
if [ -f "$RPT" ]; then
  echo "=== REPORT READY ==="
  ls -l "$RPT"
else
  echo "not yet; modules=$(grep -c 'Implementing module' "$LOG" 2>/dev/null)"
fi
echo "--- tail ---"
grep -E "Generating .*RTL|HLS 200-789|Finished Generating all RTL|csynth_design.* Elapsed|CSYNTH_OK|EXIT_CODE" "$LOG" 2>/dev/null | tail -4
