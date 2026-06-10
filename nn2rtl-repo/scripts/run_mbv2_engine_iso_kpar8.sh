#!/usr/bin/env bash
# run_mbv2_engine_iso_kpar8.sh — [KPAR8] engine-ISO gate (WLAT=2 deployment URAM).
# Builds tb/engine_iso_wrap_mbv2.v twice:
#   * KPAR8 (-DKPAR8): K_PAR=8 engine + repacked _kp8 banks (FC-PAD relocated
#     image; cfg gen parses the patched scheduler base 13416) — the new path.
#   * KPAR4 (-DKPAR4): K_PAR=4 engine + _kp4 banks — DW serial-walk CYCLE
#     REFERENCE (the 896 dispatch must be cycle-IDENTICAL across builds).
#     NOTE: 'linear' must NOT run on the KPAR4 build post-FC-PAD (the kp4
#     banks still hold FC at 13413 while the cfg now says 13416).
# Runs (KPAR8): dense small (816, fast walk, IC=16), dense big (898, fast
# walk, IC=960/4-chunk rotation), FC (linear, post-pad FAST walk), depthwise
# (896, serial fallback) — each on vec0+vec1.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="/c/Users/User/w64devkit/bin:/c/Users/User/oss-cad-suite/bin:$PATH"
export VERILATOR_ROOT="C:/Users/User/oss-cad-suite/share/verilator"
LOGD="output/mobilenet-v2/reports/kpar8"
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

echo "[kpar8-iso] === KPAR8 build (K_PAR=8 + _kp8 banks) ==="
for v in 0 1; do
  run_case kp8 816 $v -DKPAR8
  run_case kp8 898 $v -DKPAR8
  run_case kp8 linear $v -DKPAR8
  run_case kp8 896 $v -DKPAR8
done

echo "[kpar8-iso] === KPAR4 reference build (DW cycle-identity) ==="
run_case kp4 896 0 -DKPAR4

c8=$(grep -oE "took=[0-9]+" "$LOGD/iso_kp8_896_v0.log" | grep -oE "[0-9]+")
c4=$(grep -oE "took=[0-9]+" "$LOGD/iso_kp4_896_v0.log" | grep -oE "[0-9]+")
echo "[kpar8-iso] DW 896 cycles: KPAR8=$c8 KPAR4=$c4"
if [ "${c8:-A}" != "${c4:-B}" ]; then echo "[kpar8-iso] DW CYCLE-IDENTITY FAIL"; fail=1; else echo "[kpar8-iso] DW cycle-identical OK"; fi

if [ "$fail" = "0" ]; then echo "[kpar8-iso] RESULT: PASS"; else echo "[kpar8-iso] RESULT: FAIL"; fi
exit $fail
