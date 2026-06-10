#!/usr/bin/env bash
# run_mbv2_engine_iso_quartet.sh — [DW-QUARTET] engine-ISO gate (WLAT=2
# deployment URAM) for the STRIDE-2 depthwise dispatches.
# Runs the REAL engine in the DEPLOYMENT config (-DKPAR8 -DENG_PIPE, kp8
# banks + quartet-extended bias/scale) against each conv's per-module golden
# (the same .goldin/.goldout the 8/8 e2e gate derives from) — this is the
# per-conv A/B proof that the engine's stride-2 DW window arithmetic
# (base_r = 2*oh_r + kh - 1, bounds vs IH) matches the spatial module.
# Usage: bash scripts/run_mbv2_engine_iso_quartet.sh [convs...]   (default: 830 848 890 818)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="/c/Users/User/w64devkit/bin:/c/Users/User/oss-cad-suite/bin:$PATH"
export VERILATOR_ROOT="C:/Users/User/oss-cad-suite/share/verilator"
LOGD="output/mobilenet-v2/reports/dw_quartet"
mkdir -p "$LOGD"
ENG="output/rtl/shared_engine_skeleton.v output/rtl/engine/address_generator.v \
output/rtl/engine/config_register_block.v output/rtl/engine/mac_array.v \
output/rtl/engine/requant_pipeline.v output/rtl/engine/bram_to_stream_bridge.v"
WNO="-Wno-fatal -Wno-UNOPTFLAT -Wno-WIDTH -Wno-CASEINCOMPLETE -Wno-UNUSED -Wno-BLKANDNBLK -Wno-PINMISSING -Wno-DECLFILENAME"

CONVS=("${@:-830 848 890 818}")
[ $# -eq 0 ] && CONVS=(830 848 890 818)

fail=0
run_case() { # <conv> <vec>
  local conv="$1" vec="$2"
  python scripts/gen_dw_engine_iso_cfg.py "$conv" "$vec" > /dev/null || { echo "  $conv vec$vec: CFG-GEN FAIL"; fail=1; return; }
  local mdir="$LOGD/obj_iso_q_${conv}_v${vec}"
  verilator_bin.exe --cc --exe -j 8 $WNO -CFLAGS "-O1" \
    --top-module engine_iso_wrap_mbv2 -DNN2RTL_ENGINE_SUBBLOCKS_PROVIDED \
    -DKPAR8 -DENG_PIPE \
    --Mdir "$mdir" -o iso.exe \
    tb/engine_iso_wrap_mbv2.v $ENG tb/engine_iso_wrap_mbv2_tb.cpp \
    > "$LOGD/build_iso_${conv}_v${vec}.log" 2>&1 \
  && C:/Users/User/w64devkit/bin/make.exe -j 8 "CXX=C:/Users/User/w64devkit/bin/g++" \
    "VERILATOR_ROOT=C:/Users/User/oss-cad-suite/share/verilator" \
    -C "$mdir" -f Vengine_iso_wrap_mbv2.mk \
    >> "$LOGD/build_iso_${conv}_v${vec}.log" 2>&1
  if [ ! -f "$mdir/iso.exe" ]; then echo "  $conv vec$vec: BUILD FAIL"; fail=1; return; fi
  "$mdir/iso.exe" > "$LOGD/iso_${conv}_v${vec}.log" 2>&1
  local rc=$?
  local line
  line=$(grep -E "took=|bytes=|PASS|MISMATCH" "$LOGD/iso_${conv}_v${vec}.log" | tr '\n' ' ')
  echo "  $conv vec$vec: rc=$rc  $line"
  if [ $rc -ne 0 ] || ! grep -q "PASS" "$LOGD/iso_${conv}_v${vec}.log"; then fail=1; fi
}

echo "[quartet-iso] === DEPLOYMENT config (K_PAR=8 + ENG_PIPE, WLAT=2) ==="
for c in ${CONVS[@]}; do
  for v in 0 1; do
    run_case "$c" "$v"
  done
done

if [ "$fail" = "0" ]; then echo "[quartet-iso] RESULT: PASS"; else echo "[quartet-iso] RESULT: FAIL"; fi
exit $fail
