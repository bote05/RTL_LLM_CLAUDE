#!/usr/bin/env bash
# Run the engine isolation TB (conv_246) under Vivado XSim — a THIRD independent
# simulator besides iverilog (byte-exact) and Verilator (in-chain ±1). Resolves
# the simulator-dependent engine MAC discrepancy.
#
#   behavioral : compile the RTL engine directly (fast; answers "is the RTL
#                simulator-ambiguous?" — if XSim==iverilog, Verilator is the outlier)
#   funcsim    : synth shared_engine -> write_verilog -mode funcsim netlist ->
#                simulate the GATE netlist (definitive real-hardware answer)
#
# MUST run from repo root: the TB hardcodes relative paths (output/weights/*.mem,
# output/goldens/*.goldin, output/engine_sweep/*.hex). All xsim work artifacts
# (xsim.dir, *.jou, *.pb, *.log, .Xil) are produced at root then swept into
# build_engine_xsim/.
set -uo pipefail

MODE="${1:-behavioral}"
VIV="/d/vivado/2025.2/Vivado/bin"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WORK="$ROOT/build_engine_xsim"
mkdir -p "$WORK" output/engine_sweep
INCDIR="$ROOT/build_engine_one_layer_tb"
TB="tb/engine_one_layer_tb.v"
EXTRA_DEFS=()
# EARLY_DUMP=1 : dump only the first ED_N_EARLY output pixels then $finish early
# (gate-level funcsim of the full 454K-cycle conv takes ~2h; first pixels suffice).
if [ "${EARLY_DUMP:-0}" = "1" ]; then
  ED_N_EARLY="${ED_N_EARLY:-16}" py scripts/make_earlydump_tb.py
  TB="build_engine_xsim/engine_one_layer_tb_earlydump.v"
fi
# XSim's VRFC parser rejects use-before-declaration of continuous-assign wires
# (oc_pass_total_m1), which iverilog/Verilator tolerate. Build a hoisted COPY.
py scripts/make_xsim_engine_copy.py
ENG_RTL=(
  "build_engine_xsim/shared_engine_skeleton_xsim.v"
  "output/rtl/engine/address_generator.v"
  "output/rtl/engine/config_register_block.v"
  "output/rtl/engine/mac_array.v"
  "output/rtl/engine/requant_pipeline.v"
  "output/rtl/engine/bram_to_stream_bridge.v"
)

echo "=============================================================="
echo "[xsim] MODE=$MODE  ROOT=$ROOT"
echo "=============================================================="

run() { echo "+ $*"; "$@"; local rc=$?; [ $rc -ne 0 ] && { echo "[xsim] STEP FAILED rc=$rc"; exit $rc; }; }

if [ "$MODE" = "behavioral" ]; then
  run "$VIV/xvlog" -sv -d NN2RTL_ENGINE_SUBBLOCKS_PROVIDED "${EXTRA_DEFS[@]}" -i "$INCDIR" \
    --log "$WORK/xvlog_behavioral.log" \
    "$TB" "${ENG_RTL[@]}"
  run "$VIV/xelab" engine_one_layer_tb -s eng_beh --timescale 1ns/1ps -relax \
    --log "$WORK/xelab_behavioral.log"
  SNAP="eng_beh"; OUT_TAG="behavioral"

elif [ "$MODE" = "funcsim" ]; then
  NETLIST="$WORK/shared_engine_funcsim.v"
  if [ ! -f "$NETLIST" ] || [ "${REUSE_NETLIST:-0}" != "1" ]; then
    echo "[xsim] Vivado synth -> funcsim netlist (slow)..."
    run "$VIV/vivado" -mode batch -notrace \
      -source "$ROOT/scripts/engine_funcsim_synth.tcl" \
      -tclargs "$ROOT" "$NETLIST"
  else
    echo "[xsim] REUSE_NETLIST=1 — reusing $NETLIST"
  fi
  [ -f "$NETLIST" ] || { echo "[xsim] FATAL: funcsim netlist not produced"; exit 2; }
  run "$VIV/xvlog" -sv -d NN2RTL_ENGINE_SUBBLOCKS_PROVIDED "${EXTRA_DEFS[@]}" -i "$INCDIR" \
    --log "$WORK/xvlog_funcsim.log" \
    "$TB" "$NETLIST" "$VIV/../data/verilog/src/glbl.v"
  run "$VIV/xelab" engine_one_layer_tb glbl -s eng_func --timescale 1ns/1ps -relax \
    -L unisims_ver -L secureip -L unimacro_ver -L xpm \
    --log "$WORK/xelab_funcsim.log"
  SNAP="eng_func"; OUT_TAG="funcsim"
else
  echo "usage: $0 [behavioral|funcsim]"; exit 2
fi

echo "[xsim] xsim run (cwd=$ROOT)..."
run "$VIV/xsim" "$SNAP" -R --log "$WORK/xsim_run_${OUT_TAG}.log"

# sweep stray artifacts out of repo root
mv -f xsim.dir "$WORK/" 2>/dev/null || true
mv -f *.jou *.pb "$WORK/" 2>/dev/null || true
rm -rf .Xil 2>/dev/null || true

echo "=============================================================="
if [ "${EARLY_DUMP:-0}" = "1" ]; then
  cp output/engine_sweep/early_pixels.txt "output/engine_sweep/early_pixels_${OUT_TAG}.txt" 2>/dev/null || true
  echo "[xsim] comparing early pixels (${OUT_TAG}) vs node_conv_246.goldout"
  py scripts/compare_early_pixels.py
else
  HEX="output/engine_sweep/observed_dispatch00_node_conv_246.hex"
  TAGGED="output/engine_sweep/observed_xsim_${OUT_TAG}_conv246.hex"
  cp "$HEX" "$TAGGED"
  echo "[xsim] comparing $TAGGED vs node_conv_246.goldout"
  py scripts/compare_engine_output.py \
    --observed "$TAGGED" \
    --goldout output/goldens/node_conv_246.goldout \
    --vector-index 0 --n-out-words 196 --word-bytes 256
fi
