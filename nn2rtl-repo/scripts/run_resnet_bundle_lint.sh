#!/usr/bin/env bash
# run_resnet_bundle_lint.sh — [RESNET-FINAL-BUNDLE 2026-06-11] lint gate.
# verilator --lint-only over the ResNet engine-iso wrapper at every
# elaboration the bundle touches:
#   leg     : K_PAR=1 serial (legacy)
#   kp4     : K_PAR=4 (the pre-bundle ResNet config; still elaborable)
#   kp8     : K_PAR=8 (the new ResNet config)
#   kp8_ep  : K_PAR=8 + ENG_PIPE=1 (the DEPLOYED bundle config)
# Gate = 0 %Error AND 0 %Warning lines in every log.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="/c/Users/User/oss-cad-suite/bin:$PATH"
LOGD="output/reports_integrated/resnet_final_bundle"
mkdir -p "$LOGD"
ENG="output/rtl/shared_engine_skeleton.v output/rtl/engine/address_generator.v \
output/rtl/engine/config_register_block.v output/rtl/engine/mac_array.v \
output/rtl/engine/requant_pipeline.v output/rtl/engine/bram_to_stream_bridge.v"
WNO="-Wno-UNOPTFLAT -Wno-WIDTH -Wno-CASEINCOMPLETE -Wno-UNUSED -Wno-BLKANDNBLK -Wno-PINMISSING -Wno-DECLFILENAME"

fail=0
lint() { # <tag> <defines...>
  local tag="$1"; shift
  verilator_bin.exe --lint-only $WNO \
    --top-module engine_iso_wrap_resnet \
    -DNN2RTL_ENGINE_SUBBLOCKS_PROVIDED "$@" \
    tb/engine_iso_wrap_resnet.v $ENG > "$LOGD/lint_$tag.log" 2>&1
  local rc=$?
  local bad
  bad=$(grep -cE "%(Error|Warning)" "$LOGD/lint_$tag.log")
  if [ "$rc" = "0" ] && [ "$bad" = "0" ]; then
    echo "  lint[$tag]: PASS (0 errors, 0 warnings)"
  else
    echo "  lint[$tag]: FAIL (rc=$rc, $bad diag lines) — see $LOGD/lint_$tag.log"
    fail=1
  fi
}

echo "[bundle-lint] === verilator --lint-only, 4 configs ==="
lint leg
lint kp4    -DKPAR4
lint kp8    -DKPAR8
lint kp8_ep -DKPAR8 -DENG_PIPE
if [ "$fail" = "0" ]; then echo "[bundle-lint] RESULT: PASS"; else echo "[bundle-lint] RESULT: FAIL"; fi
exit $fail
