#!/usr/bin/env bash
# Compile + run + compare the one-layer engine TB.
# Git-Bash compatible (uses Windows iverilog/vvp via /c/... paths).
#
# Exits 0 on PASS, non-zero on FAIL.

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

IVERILOG="${IVERILOG:-/c/Users/User/oss-cad-suite/bin/iverilog}"
VVP="${VVP:-/c/Users/User/oss-cad-suite/bin/vvp}"
PYTHON="${PYTHON:-py}"

# OSS-CAD-Suite needs YOSYSHQ_ROOT set so iverilog can spawn its own
# preprocessor/codegen children (ivlpp / ivl). Without this iverilog
# silently exits 127.
# NOTE: YOSYSHQ_ROOT MUST use Unix-style /c/ path under Git Bash. Windows
# C:/... path causes iverilog to silently exit 127 because the env var
# is passed to the spawned ivlpp child without path translation.
OSS_CAD_ROOT="${OSS_CAD_ROOT:-/c/Users/User/oss-cad-suite}"
export YOSYSHQ_ROOT="${YOSYSHQ_ROOT:-${OSS_CAD_ROOT}/}"
export PATH="${OSS_CAD_ROOT}/bin:${OSS_CAD_ROOT}/lib:${PATH}"

TIMEOUT_CYCLES="${TIMEOUT_CYCLES:-10000000}"

echo "=== [run_engine_one_layer_tb] Compiling testbench ==="
mkdir -p build_engine_one_layer_tb
# -gno-strict-declaration tolerates the forward reference to
# `oc_pass_total_m1` in shared_engine_skeleton.v (used on line 234,
# declared on line 279). Verilator and Vivado accept this naturally;
# iverilog needs the relaxed mode.
"$IVERILOG" -g2012 -gno-strict-declaration \
    -DNN2RTL_ENGINE_SUBBLOCKS_PROVIDED \
    -o build_engine_one_layer_tb/engine_tb.vvp \
    tb/engine_one_layer_tb.v \
    output/rtl/shared_engine_skeleton.v \
    output/rtl/engine/address_generator.v \
    output/rtl/engine/bram_to_stream_bridge.v \
    output/rtl/engine/config_register_block.v \
    output/rtl/engine/mac_array.v \
    output/rtl/engine/requant_pipeline.v

echo "=== [run_engine_one_layer_tb] Running simulation ==="
"$VVP" build_engine_one_layer_tb/engine_tb.vvp "+TIMEOUT_CYCLES=${TIMEOUT_CYCLES}" \
    | tee build_engine_one_layer_tb/sim.log

echo "=== [run_engine_one_layer_tb] Comparing engine output vs .goldout ==="
"$PYTHON" scripts/compare_engine_output.py \
    --observed output/engine_tb_observed.hex \
    --goldout  output/goldens/node_conv_246.goldout \
    --vector-index 0 \
    --n-out-words 196 \
    --word-bytes 256

echo "=== [run_engine_one_layer_tb] PASS ==="
