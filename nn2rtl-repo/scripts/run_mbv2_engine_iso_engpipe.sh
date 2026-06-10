#!/usr/bin/env bash
# run_mbv2_engine_iso_engpipe.sh — [ENG_PIPE] engine-ISO gate (WLAT=2 deployment URAM).
# Builds tb/engine_iso_wrap_mbv2.v in four configs (all K_PAR=8 + _kp8 banks):
#   * ep0      (-DKPAR8): legacy stop-and-wait — cycle REFERENCE.
#   * ep1      (-DKPAR8 -DENG_PIPE): pipelined issue — must be byte-exact
#     vs the SAME goldens, with FEWER cycles.
#   * ep0_thr  (-DKPAR8 -DTHROTTLE): legacy + backpressure + LFSR-throttled
#     out_ready — byte-exact reference for the stall path.
#   * ep1_thr  (-DKPAR8 -DENG_PIPE -DTHROTTLE): pipelined issue under a
#     throttled output — exercises bridge hold + gap-hold (pend==2) +
#     fire_recap; must be byte-exact.
# Cases: dense small (816, IC=16, N=2 walks), dense big (898, IC=960),
# FC (linear), depthwise (896, serial walk) — vec0+vec1 unthrottled,
# vec0 throttled.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="/c/Users/User/w64devkit/bin:/c/Users/User/oss-cad-suite/bin:$PATH"
export VERILATOR_ROOT="C:/Users/User/oss-cad-suite/share/verilator"
LOGD="output/mobilenet-v2/reports/engpipe"
mkdir -p "$LOGD"
ENG="output/rtl/shared_engine_skeleton.v output/rtl/engine/address_generator.v \
output/rtl/engine/config_register_block.v output/rtl/engine/mac_array.v \
output/rtl/engine/requant_pipeline.v output/rtl/engine/bram_to_stream_bridge.v"
WNO="-Wno-fatal -Wno-UNOPTFLAT -Wno-WIDTH -Wno-CASEINCOMPLETE -Wno-UNUSED -Wno-BLKANDNBLK -Wno-PINMISSING -Wno-DECLFILENAME"

gen_cfg() { # <conv|linear> <vec>
  case "$1" in
    linear|896|902|908|824|836|842|854|860|866|872|878|884)
      python scripts/gen_dw_engine_iso_cfg.py "$1" "$2" ;;
    *)
      python scripts/gen_mbv2_dense_engine_iso_cfg.py "$1" "$2" ;;
  esac
}

build() { # <mdir> <extra-defines...>
  local mdir="$1"; shift
  verilator_bin.exe --cc --exe -j 8 $WNO -CFLAGS "-O1" \
    --top-module engine_iso_wrap_mbv2 -DNN2RTL_ENGINE_SUBBLOCKS_PROVIDED "$@" \
    --Mdir "$mdir" -o iso.exe \
    tb/engine_iso_wrap_mbv2.v $ENG tb/engine_iso_wrap_mbv2_tb.cpp \
    > "$LOGD/build_$(basename "$mdir").log" 2>&1 \
  && C:/Users/User/w64devkit/bin/make.exe -j 8 "CXX=C:/Users/User/w64devkit/bin/g++" \
    "VERILATOR_ROOT=C:/Users/User/oss-cad-suite/share/verilator" \
    -C "$mdir" -f Vengine_iso_wrap_mbv2.mk \
    >> "$LOGD/build_$(basename "$mdir").log" 2>&1
}

fail=0
run_case() { # <build-dir-tag> <conv> <vec> <defines...>
  local tag="$1" conv="$2" vec="$3"; shift 3
  gen_cfg "$conv" "$vec" > /dev/null || { echo "  $tag/$conv vec$vec: CFG-GEN FAIL"; fail=1; return; }
  local mdir="$LOGD/obj_iso_${tag}_${conv}_v${vec}"
  build "$mdir" "$@" || true
  if [ ! -f "$mdir/iso.exe" ]; then echo "  $tag/$conv vec$vec: BUILD FAIL (see build log)"; fail=1; return; fi
  "$mdir/iso.exe" > "$LOGD/iso_${tag}_${conv}_v${vec}.log" 2>&1
  local rc=$?
  local line
  line=$(grep -E "took=|bytes=|PASS|MISMATCH" "$LOGD/iso_${tag}_${conv}_v${vec}.log" | tr '\n' ' ')
  echo "  $tag/$conv vec$vec: rc=$rc  $line"
  if [ $rc -ne 0 ] || ! grep -q "PASS" "$LOGD/iso_${tag}_${conv}_v${vec}.log"; then fail=1; fi
}

echo "[engpipe-iso] === ep1 (K_PAR=8 + ENG_PIPE) ==="
for v in 0 1; do
  run_case ep1 816 $v -DKPAR8 -DENG_PIPE
  run_case ep1 898 $v -DKPAR8 -DENG_PIPE
  run_case ep1 linear $v -DKPAR8 -DENG_PIPE
  run_case ep1 896 $v -DKPAR8 -DENG_PIPE
done

echo "[engpipe-iso] === ep0 reference (K_PAR=8 legacy, cycle baseline) ==="
run_case ep0 816 0 -DKPAR8
run_case ep0 898 0 -DKPAR8
run_case ep0 linear 0 -DKPAR8
run_case ep0 896 0 -DKPAR8

echo "[engpipe-iso] === throttled out_ready (LFSR ~50%): legacy vs ENG_PIPE ==="
run_case ep0_thr 816 0 -DKPAR8 -DTHROTTLE
run_case ep0_thr 898 0 -DKPAR8 -DTHROTTLE
run_case ep0_thr linear 0 -DKPAR8 -DTHROTTLE
run_case ep0_thr 896 0 -DKPAR8 -DTHROTTLE
run_case ep1_thr 816 0 -DKPAR8 -DENG_PIPE -DTHROTTLE
run_case ep1_thr 898 0 -DKPAR8 -DENG_PIPE -DTHROTTLE
run_case ep1_thr linear 0 -DKPAR8 -DENG_PIPE -DTHROTTLE
run_case ep1_thr 896 0 -DKPAR8 -DENG_PIPE -DTHROTTLE

echo "[engpipe-iso] === cycle deltas (ep0 -> ep1, vec0) ==="
for c in 816 898 linear 896; do
  c0=$(grep -oE "took=[0-9]+" "$LOGD/iso_ep0_${c}_v0.log" 2>/dev/null | grep -oE "[0-9]+")
  c1=$(grep -oE "took=[0-9]+" "$LOGD/iso_ep1_${c}_v0.log" 2>/dev/null | grep -oE "[0-9]+")
  if [ -n "${c0:-}" ] && [ -n "${c1:-}" ]; then
    echo "  $c: ep0=$c0 ep1=$c1 delta=$((c1 - c0))"
    if [ "$c1" -ge "$c0" ]; then echo "  $c: WARNING ep1 not faster"; fi
  fi
done

if [ "$fail" = "0" ]; then echo "[engpipe-iso] RESULT: PASS"; else echo "[engpipe-iso] RESULT: FAIL"; fi
exit $fail
