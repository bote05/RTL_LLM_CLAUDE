#!/usr/bin/env bash
# Diagnostic compile + run for conv_290 with DBG_DUMP_ACC instrumentation enabled.
set -e
cd "$(dirname "$0")/.."

export YOSYSHQ_ROOT=/c/Users/User/oss-cad-suite/
export PATH="/c/Users/User/oss-cad-suite/bin:/c/Users/User/oss-cad-suite/lib:$PATH"

BUILD=build_engine_one_layer_tb_dbg
mkdir -p $BUILD

# Generate dispatch_cfg.vh for dispatch 10 (conv_290).
cat > $BUILD/dispatch_cfg.vh <<'EOF'
// dispatch_index=10 module_id=node_conv_290

`define CFG_IC          12'd2048
`define CFG_OC          12'd512
`define CFG_KH          3'd1
`define CFG_KW          3'd1
`define CFG_SH          3'd1
`define CFG_SW          3'd1
`define CFG_PH          3'd0
`define CFG_PW          3'd0
`define CFG_IH          8'd7
`define CFG_IW          8'd7
`define CFG_OH          8'd7
`define CFG_OW          8'd7
`define CFG_WEIGHT_BASE 20'd61843
`define CFG_BIAS_BASE   16'd2
`define CFG_SCALE_MULT  32'd768585439
`define CFG_SCALE_SHIFT 6'd36
`define CFG_ZP          8'd0
`define CFG_ACT_IN_BASE  16'd4096
`define CFG_ACT_OUT_BASE 16'd0
`define N_IN_PIXELS  49
`define N_IN_WORDS   392
`define IC_CHUNKS    8
`define N_OUT_WORDS  98
`define N_OUT_PIXELS 49
`define OC_PASSES    2
`define IC_BYTES     2048
`define OC_BYTES     512
`define GOLDIN_PATH "output/goldens/node_conv_290.goldin"
`define OBSERVED_HEX_PATH "output/engine_sweep/observed_dispatch10_node_conv_290_dbg.hex"
EOF

iverilog -g2012 -gno-strict-declaration \
    -DNN2RTL_ENGINE_SUBBLOCKS_PROVIDED \
    -DDBG_DUMP_ACC \
    -DDBG_DUMP_MAC \
    -I $BUILD \
    -o $BUILD/engine_tb_dbg.vvp \
    tb/engine_one_layer_tb.v \
    output/rtl/shared_engine_skeleton.v \
    output/rtl/engine/address_generator.v \
    output/rtl/engine/bram_to_stream_bridge.v \
    output/rtl/engine/config_register_block.v \
    output/rtl/engine/mac_array.v \
    output/rtl/engine/requant_pipeline.v

echo "Compile OK; running..."
vvp $BUILD/engine_tb_dbg.vvp +TIMEOUT_CYCLES=50000000 > $BUILD/sim_dbg.log 2>&1
echo "Sim done. log lines: $(wc -l < $BUILD/sim_dbg.log)"
