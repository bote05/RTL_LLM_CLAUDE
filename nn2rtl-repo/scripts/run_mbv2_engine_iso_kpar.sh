#!/usr/bin/env bash
# run_mbv2_engine_iso_kpar.sh — [KPAR4] engine-ISO gate (WLAT=2 deployment URAM).
# Builds tb/engine_iso_wrap_mbv2.v twice:
#   * LEGACY (no -DKPAR4): K_PAR=1 elaboration + ORIGINAL banks — functional
#     inertness regression of the shared-file changes.
#   * KPAR4 (-DKPAR4): K_PAR=4 engine + repacked _kp4 banks — the new path.
# Runs: dense small (816, fast walk, IC=16), dense big (898, fast walk,
# IC=960/4 chunks), depthwise (896, serial fallback + per-lane act), FC
# (linear, serial fallback + base%4==1 subword select), each on 2 vectors.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# w64devkit g++/as FIRST (oss-cad's gcc lacks -fcf-protection=none and the
# ancient C:\MinGW as is 32-bit). POSIX-style entries: git-bash does not
# honor C:/-style PATH prepends for `which` resolution.
export PATH="/c/Users/User/w64devkit/bin:/c/Users/User/oss-cad-suite/bin:$PATH"
export VERILATOR_ROOT="C:/Users/User/oss-cad-suite/share/verilator"
LOGD="output/mobilenet-v2/reports/kpar4"
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
  # verilate, then make with CXX PINNED to w64devkit g++15 (the ancient
  # C:\MinGW g++ 6.3.0 on PATH rejects -faligned-new/-fcf-protection).
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
  line=$(grep -E "bytes=|PASS|MISMATCH" "$LOGD/iso_${tag}_${conv}_v${vec}.log" | tr '\n' ' ')
  echo "  $tag/$conv vec$vec: rc=$rc  $line"
  [ "$rc" != "0" ] && fail=1
}

echo "[kpar-iso] === KPAR4 build (K_PAR=4, repacked banks) ==="
for v in 0 1; do
  run_case kp4 816 "$v" -DKPAR4      # dense FAST, IC=16  (4 groups)
  run_case kp4 898 "$v" -DKPAR4      # dense FAST, IC=960 (chunk rotation)
  run_case kp4 896 "$v" -DKPAR4      # depthwise SERIAL fallback (per-lane act)
  run_case kp4 linear "$v" -DKPAR4   # FC SERIAL fallback (base 13413%4==1 subsel)
done

echo "[kpar-iso] === LEGACY build (K_PAR=1, original banks) — inertness regression ==="
for c in 816 896 linear; do
  run_case leg "$c" 0
done

if [ "$fail" = "0" ]; then echo "[kpar-iso] RESULT: PASS (all cases byte-exact)"; else echo "[kpar-iso] RESULT: FAIL"; fi
exit $fail
