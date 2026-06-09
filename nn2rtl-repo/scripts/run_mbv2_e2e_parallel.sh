#!/usr/bin/env bash
# [THREADS 2026-06-09] Fast MBV2 e2e for the i9-13980HX (24c/32t):
#   1. build ONCE with parallel make (-j, --no-pch) via the tsx harness (build-only)
#   2. run all 8 vectors as PARALLEL single-thread Verilator processes
# Verilator INTERNAL MT is broken (#5: wrong values) -> each process is --threads 1 (byte-exact);
# OS-level process parallelism is the safe way to "use more threads" for the gate.
# Env: SKIP_BUILD=1 reuse exe | VEC_JOBS=N concurrency (default 8) | MBV2_MAKE_JOBS=N (default 24)
#      MBV2_MAX_CYCLES (default 12000000)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
EXE="output/mobilenet-v2/reports/verilator_mbv2_top_engine_value/obj_dir_engine_value/Vnn2rtl_top.exe"
GIN="output/mobilenet-v2/goldens/node_conv_810.goldin"
GOUT="output/mobilenet-v2/goldens/node_linear.goldout"
MAXC="${MBV2_MAX_CYCLES:-12000000}"
JOBS="${MBV2_MAKE_JOBS:-24}"
VJ="${VEC_JOBS:-8}"
export PATH="C:/Users/User/oss-cad-suite/bin:C:/Users/User/w64devkit/bin:$PATH"
LOGD="output/mobilenet-v2/reports/e2e_par"
mkdir -p "$LOGD"
t_start=$(date +%s)

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  echo "[par-e2e] building (-j $JOBS, --no-pch) ..."
  MBV2_MAKE_JOBS="$JOBS" MBV2_THREADS=1 npx tsx scripts/run_mbv2_top_engine_value.ts 0 > "$LOGD/build.log" 2>&1
  if [ ! -f "$EXE" ]; then echo "[par-e2e] BUILD FAILED (no exe). tail build.log:"; tail -25 "$LOGD/build.log"; exit 2; fi
  echo "[par-e2e] build OK ($(( $(date +%s) - t_start ))s)"
fi
t_sim=$(date +%s)

echo "[par-e2e] running 8 vectors, $VJ at a time (--threads 1 each) ..."
run_vec() { MBV2_MAX_CYCLES="$MAXC" "$EXE" "$GIN" "$GOUT" "$1" > "$LOGD/vec$1.log" 2>&1; }
running=0
for i in 0 1 2 3 4 5 6 7; do
  run_vec "$i" &
  running=$((running+1))
  if [ "$running" -ge "$VJ" ]; then wait -n 2>/dev/null || wait; running=$((running-1)); fi
done
wait

echo "[par-e2e] === per-vector results (sim $(( $(date +%s) - t_sim ))s) ==="
fail=0; total=0
for i in 0 1 2 3 4 5 6 7; do
  line=$(grep -hE "\[tb\]\[mbv2\]\[summary\]" "$LOGD/vec$i.log" | tail -1)
  [ -z "$line" ] && line="(no summary -- crash/timeout)"
  mm=$(echo "$line" | grep -oE "mismatch_bytes=-?[0-9]+" | grep -oE "\-?[0-9]+" | head -1)
  echo "  vec$i: $line"
  if [ "${mm:-X}" != "0" ]; then fail=1; fi
  if [ -n "${mm:-}" ] && [ "${mm}" -ge 0 ] 2>/dev/null; then total=$((total + mm)); fi
done
echo "[par-e2e] TOTAL mismatch (8 vecs) = $total ; wall=$(( $(date +%s) - t_start ))s"
if [ "$fail" = "0" ]; then echo "[par-e2e] RESULT: PASS (8/8 byte-exact)"; else echo "[par-e2e] RESULT: FAIL"; fi
exit $fail
