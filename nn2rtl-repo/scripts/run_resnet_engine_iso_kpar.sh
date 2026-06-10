#!/usr/bin/env bash
# run_resnet_engine_iso_kpar.sh — [KPAR4-RN] ResNet engine-ISO gate (WLAT=2
# deployment read latency). Builds tb/engine_iso_wrap_resnet.v twice:
#   * LEGACY (no -DKPAR4): K_PAR=1 elaboration + ORIGINAL 96b banks.
#   * KPAR4 (-DKPAR4): K_PAR=4 engine + repacked/transposed _kp4 banks.
#
# GATE = A/B EQUIVALENCE: for each (conv, vec) the engine output bytes of
# the KPAR4 build must be IDENTICAL to the LEGACY build's (cmp on raw dumps
# via NN2RTL_ISO_DUMP). The CONTRACT goldens canNOT serve as the reference
# here: the intermediate contract goldens are STALE (2026-05-30, pre the
# FIT-FIX scale.mem requant change of 06-07) — every intermediate
# activation shifted by a few LSB while only the FINAL relu_48 golden was
# refreshed (06-09). The deployed e2e gate (run_nn2rtl_top_value.ts vs the
# FRESH relu_48 golden, PASS 0/100352) is the authoritative byte-exact
# check; this ISO gate proves KPAR4==LEGACY at the engine level on REAL
# weights/inputs (incl. the transposed-bank 3x3 fetch path) and shows the
# ~4x fast-walk cycle reduction. The per-case golden mismatch counts are
# still printed as INFO (expect equal counts in both builds).
#
# Cases: dense 3x3 IC=256 (246, dispatch 0 — pos-major transposed walk),
# dense 1x1 IC=512/OC=1024 (250, dispatch 1 — chunk rotation + 4 oc passes),
# dense 3x3 IC=512 stride2 (284, dispatch 9 — 2-chunk 3x3), each on 2 vecs.
# The shared C++ driver tb/engine_iso_wrap_mbv2_tb.cpp is reused via
# --prefix Vengine_iso_wrap_mbv2 (identical port list).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="/c/Users/User/w64devkit/bin:/c/Users/User/oss-cad-suite/bin:$PATH"
export VERILATOR_ROOT="C:/Users/User/oss-cad-suite/share/verilator"
LOGD="output/reports_integrated/kpar4rn"
mkdir -p "$LOGD"
ENG="output/rtl/shared_engine_skeleton.v output/rtl/engine/address_generator.v \
output/rtl/engine/config_register_block.v output/rtl/engine/mac_array.v \
output/rtl/engine/requant_pipeline.v output/rtl/engine/bram_to_stream_bridge.v"
WNO="-Wno-fatal -Wno-UNOPTFLAT -Wno-WIDTH -Wno-CASEINCOMPLETE -Wno-UNUSED -Wno-BLKANDNBLK -Wno-PINMISSING -Wno-DECLFILENAME"

build() { # <mdir> <extra-defines...>
  local mdir="$1"; shift
  verilator_bin.exe --cc --exe -j 8 $WNO -CFLAGS "-O1" \
    --top-module engine_iso_wrap_resnet --prefix Vengine_iso_wrap_mbv2 \
    -DNN2RTL_ENGINE_SUBBLOCKS_PROVIDED "$@" \
    --Mdir "$mdir" -o iso.exe \
    tb/engine_iso_wrap_resnet.v $ENG tb/engine_iso_wrap_mbv2_tb.cpp \
    > "$LOGD/build_$(basename "$mdir").log" 2>&1 \
  && C:/Users/User/w64devkit/bin/make.exe -j 8 "CXX=C:/Users/User/w64devkit/bin/g++" \
    "VERILATOR_ROOT=C:/Users/User/oss-cad-suite/share/verilator" \
    -C "$mdir" -f Vengine_iso_wrap_mbv2.mk \
    >> "$LOGD/build_$(basename "$mdir").log" 2>&1
}

fail=0
run_case() { # <build-dir-tag> <conv> <vec>  (cfg regenerated per run; exe per tag)
  local tag="$1" conv="$2" vec="$3"
  python scripts/gen_resnet_engine_iso_cfg.py "$conv" "$vec" > /dev/null \
    || { echo "  $tag/$conv vec$vec: CFG-GEN FAIL"; fail=1; return; }
  local mdir="$LOGD/obj_iso_${tag}_${conv}_v${vec}"
  if [ "$tag" = "kp4" ]; then build "$mdir" -DKPAR4 || true; else build "$mdir" || true; fi
  if [ ! -f "$mdir/iso.exe" ]; then echo "  $tag/$conv vec$vec: BUILD FAIL (see build log)"; fail=1; return; fi
  NN2RTL_ISO_DUMP="$LOGD/dump_${tag}_${conv}_v${vec}.bin" \
    "$mdir/iso.exe" > "$LOGD/iso_${tag}_${conv}_v${vec}.log" 2>&1
  local line
  line=$(grep -E "took=|bytes=" "$LOGD/iso_${tag}_${conv}_v${vec}.log" | tr '\n' ' ')
  echo "  $tag/$conv vec$vec: $line"
}

echo "[kpar-rn-iso] === building + running both builds (3 convs x 2 vecs) ==="
for v in 0 1; do
  for c in 246 250 284; do
    run_case kp4 "$c" "$v"
    run_case leg "$c" "$v"
  done
done

echo "[kpar-rn-iso] === A/B equivalence (KPAR4 output bytes == LEGACY output bytes) ==="
for v in 0 1; do
  for c in 246 250 284; do
    a="$LOGD/dump_kp4_${c}_v${v}.bin"; b="$LOGD/dump_leg_${c}_v${v}.bin"
    if [ ! -s "$a" ] || [ ! -s "$b" ]; then echo "  $c vec$v: MISSING DUMP"; fail=1; continue; fi
    if cmp -s "$a" "$b"; then echo "  $c vec$v: IDENTICAL ($(stat -c%s "$a") bytes)"; else echo "  $c vec$v: DIFFER"; fail=1; fi
  done
done

if [ "$fail" = "0" ]; then echo "[kpar-rn-iso] RESULT: PASS (KPAR4 == LEGACY byte-identical on all cases)"; else echo "[kpar-rn-iso] RESULT: FAIL"; fi
exit $fail
