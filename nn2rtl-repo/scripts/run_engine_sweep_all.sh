#!/usr/bin/env bash
# Sweep the engine one-layer TB across all 14 heavy dispatches.
# Git-Bash compatible.
#
# Exits 0 iff every dispatch passes byte-exact, otherwise non-zero with the
# failures reported in docs/agent_tasks/13_engine_sweep_REPORT.md.

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

IVERILOG="${IVERILOG:-/c/Users/User/oss-cad-suite/bin/iverilog}"
VVP="${VVP:-/c/Users/User/oss-cad-suite/bin/vvp}"
PYTHON="${PYTHON:-py}"

# Same OSS-CAD-Suite environment fix as run_engine_one_layer_tb.sh.
OSS_CAD_ROOT="${OSS_CAD_ROOT:-/c/Users/User/oss-cad-suite}"
export YOSYSHQ_ROOT="${YOSYSHQ_ROOT:-${OSS_CAD_ROOT}/}"
export PATH="${OSS_CAD_ROOT}/bin:${OSS_CAD_ROOT}/lib:${PATH}"

TIMEOUT_CYCLES="${TIMEOUT_CYCLES:-50000000}"

echo "=== [run_engine_sweep_all] starting sweep ==="
echo "    IVERILOG=$IVERILOG"
echo "    VVP=$VVP"
echo "    PYTHON=$PYTHON"
echo "    TIMEOUT_CYCLES=$TIMEOUT_CYCLES"

EXTRA_ARGS=()
if [ -n "${ONLY_DISPATCHES:-}" ]; then
    EXTRA_ARGS+=("--only" "$ONLY_DISPATCHES")
fi

"$PYTHON" scripts/engine_sweep_driver.py \
    --iverilog "$IVERILOG" \
    --vvp "$VVP" \
    --python "$PYTHON" \
    --timeout-cycles "$TIMEOUT_CYCLES" \
    "${EXTRA_ARGS[@]}"
SWEEP_RC=$?

if [ $SWEEP_RC -eq 0 ]; then
    echo "=== [run_engine_sweep_all] ALL DISPATCHES PASS ==="
else
    echo "=== [run_engine_sweep_all] SOME DISPATCHES FAILED (rc=$SWEEP_RC) ==="
fi

exit $SWEEP_RC
