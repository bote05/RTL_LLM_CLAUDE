#!/usr/bin/env bash
# run_resnet_engine_iso_kpar8.sh — [KPAR8-RN 2026-06-11] ResNet engine-ISO
# gate (WLAT=2 deployment read latency). Builds tb/engine_iso_wrap_resnet.v
# three ways:
#   * LEGACY (no defines): K_PAR=1 serial walk + ORIGINAL 96b banks.
#   * KP8    (-DKPAR8): K_PAR=8 engine + repacked/transposed _kp8 banks.
#   * KP8EP  (-DKPAR8 -DENG_PIPE): the DEPLOYED ResNet config (K_PAR=8 +
#     pipelined issue) — value-identical, fewer cycles.
#
# GATE = A/B EQUIVALENCE: for each (conv, vec) the engine output bytes of
# the KP8 AND KP8EP builds must be IDENTICAL to the LEGACY build's (cmp on
# raw dumps via NN2RTL_ISO_DUMP). The intermediate CONTRACT goldens are
# stale (2026-05-30 vintage, see run_resnet_engine_iso_kpar.sh header) so
# per-case golden mismatch counts are INFO only; the authoritative
# byte-exact check is the deployed e2e gate.
#
# Cases: dense 3x3 IC=256 (246, dispatch 0 — pos-major transposed walk),
# dense 1x1 IC=512/OC=1024 (250, dispatch 1 — chunk rotation + 4 oc passes),
# dense 3x3 IC=512 stride2 (284, dispatch 9 — 2-chunk 3x3), each on 2 vecs.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="/c/Users/User/w64devkit/bin:/c/Users/User/oss-cad-suite/bin:$PATH"
export VERILATOR_ROOT="C:/Users/User/oss-cad-suite/share/verilator"
LOGD="output/reports_integrated/kpar8rn"
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
run_case() { # <tag> <conv> <vec>
  local tag="$1" conv="$2" vec="$3"
  python scripts/gen_resnet_engine_iso_cfg.py "$conv" "$vec" > /dev/null \
    || { echo "  $tag/$conv vec$vec: CFG-GEN FAIL"; fail=1; return; }
  local mdir="$LOGD/obj_iso_${tag}_${conv}_v${vec}"
  case "$tag" in
    kp8)   build "$mdir" -DKPAR8 || true ;;
    kp8ep) build "$mdir" -DKPAR8 -DENG_PIPE || true ;;
    *)     build "$mdir" || true ;;
  esac
  if [ ! -f "$mdir/iso.exe" ]; then echo "  $tag/$conv vec$vec: BUILD FAIL (see build log)"; fail=1; return; fi
  NN2RTL_ISO_DUMP="$LOGD/dump_${tag}_${conv}_v${vec}.bin" \
    "$mdir/iso.exe" > "$LOGD/iso_${tag}_${conv}_v${vec}.log" 2>&1
  local line
  line=$(grep -E "took=|bytes=" "$LOGD/iso_${tag}_${conv}_v${vec}.log" | tr '\n' ' ')
  echo "  $tag/$conv vec$vec: $line"
}

echo "[kpar8-rn-iso] === building + running 3 builds (3 convs x 2 vecs) ==="
for v in 0 1; do
  for c in 246 250 284; do
    run_case kp8   "$c" "$v"
    run_case kp8ep "$c" "$v"
    run_case leg   "$c" "$v"
  done
done

echo "[kpar8-rn-iso] === A/B equivalence (KP8 / KP8EP output bytes == LEGACY) ==="
for v in 0 1; do
  for c in 246 250 284; do
    ref="$LOGD/dump_leg_${c}_v${v}.bin"
    for tag in kp8 kp8ep; do
      a="$LOGD/dump_${tag}_${c}_v${v}.bin"
      if [ ! -s "$a" ] || [ ! -s "$ref" ]; then echo "  $tag/$c vec$v: MISSING DUMP"; fail=1; continue; fi
      if cmp -s "$a" "$ref"; then echo "  $tag/$c vec$v: IDENTICAL ($(stat -c%s "$a") bytes)"; else echo "  $tag/$c vec$v: DIFFER"; fail=1; fi
    done
  done
done

if [ "$fail" = "0" ]; then echo "[kpar8-rn-iso] RESULT: PASS (KP8 and KP8EP == LEGACY byte-identical on all cases)"; else echo "[kpar8-rn-iso] RESULT: FAIL"; fi
exit $fail
