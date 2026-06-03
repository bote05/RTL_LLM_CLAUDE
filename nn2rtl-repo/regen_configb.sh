#!/usr/bin/env bash
# Config B (mixed-INT3) regen with the HARDENED pipeline (the steps the all-INT4 regen skipped).
# 18 INT3 layers (4 spatial + 14 engine), rest INT4. Calibration-faithful (CALIB=256) => 77.6% top-1.
set -e
cd /c/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo
PY=/c/Python313/python
export PATH="/c/Users/User/oss-cad-suite/bin:/c/Users/User/w64devkit/bin:$PATH"
log(){ echo "===== [$(printf '%(%H:%M:%S)T')] $* ====="; }
INT3="node_conv_246,node_conv_250,node_conv_254,node_conv_260,node_conv_264,node_conv_266,node_conv_272,node_conv_278,node_conv_282,node_conv_286,node_conv_290,node_conv_294,node_conv_296,node_conv_300,node_conv_284,node_conv_288,node_conv_292,node_conv_298"

log "STEP0 backup all-INT4 byte-exact baseline"
mkdir -p backups/allint4_byteexact
cp output/layer_ir.json backups/allint4_byteexact/layer_ir.json
cp output/weights/bias.mem output/weights/scale.mem backups/allint4_byteexact/ 2>/dev/null || true
cp output/goldens/node_relu_48.goldout backups/allint4_byteexact/relu48_logical.goldout
for c in 284 288 292 298; do cp output/weights/node_conv_${c}_weights_mp_k_*.hex backups/allint4_byteexact/ 2>/dev/null || true; done
cp output/rtl/nn2rtl_top.v backups/allint4_byteexact/nn2rtl_top.v
echo "backup done"

log "STEP1 RTL -> INT3 (engine ENGINE_WGT_W=3 + 4 wrappers WGT_BITS(3))"
sed -i 's/localparam integer ENGINE_WGT_W  = 4;/localparam integer ENGINE_WGT_W  = 3;/' output/rtl/nn2rtl_top.v
grep -n "localparam integer ENGINE_WGT_W" output/rtl/nn2rtl_top.v | head -1
for c in 284 288 292 298; do sed -i 's/\.WGT_BITS(4)/.WGT_BITS(3)/' output/rtl/node_conv_$c.v; echo "conv_$c: $(grep -o 'WGT_BITS([0-9])' output/rtl/node_conv_$c.v | head -1)"; done

log "STEP2 generate_golden Config B (18 INT3, WB=4 CALIB=256)"
NN2RTL_INT3_LAYERS="$INT3" NN2RTL_WEIGHT_BITS=4 NN2RTL_IMAGENET_CALIB=256 $PY scripts/generate_golden.py checkpoints/resnet50_full.onnx 2>&1 | tail -3

log "STEP3 spatial WIDE INT3 (4 convs, single-file --wgt-bits 3)"
for spec in "284:512:4608:9" "288:2048:1024:8" "292:512:4608:9" "298:512:4608:9"; do
  c=${spec%%:*}; r=${spec#*:}; oc=${r%%:*}; r=${r#*:}; kt=${r%%:*}; mpk=${r##*:}
  $PY scripts/repack_weights_wide.py --input output/weights/node_conv_${c}_weights.hex --output output/weights/node_conv_${c}_weights_mp_k_${mpk}.hex --oc $oc --k-total $kt --mp 16 --mp-k $mpk --wgt-bits 3
done
for c in 284 288 292 298; do echo "conv_$c WIDE: $(head -1 $(ls output/weights/node_conv_${c}_weights_mp_k_*.hex|head -1)|tr -d '\r\n'|wc -c) chars (INT3 mpk9=108 mpk8=96)"; done

log "STEP4 engine banks INT3 (build -> dedup -> nibble_int3)"
$PY scripts/build_weight_memory_map.py --network resnet-50
$PY scripts/dedup_engine_banks.py
$PY scripts/nibble_engine_banks_int3.py
echo "bank0 width: $(head -1 output/weights/uram_weights_bank0.mem|tr -d '\r\n'|wc -c) chars (INT3=24)"

log "STEP5 scales (spatial+engine) + BIAS MAP (the fix)"
$PY scripts/build_spatial_scale_mems.py
$PY scripts/build_scale_memory_map.py --network resnet-50
$PY scripts/build_bias_memory_map.py --network resnet-50

log "STEP6 refresh final golden"
$PY scripts/refresh_final_golden.py node_relu_48

log "STEP7 rebuild + e2e"
NN2RTL_VALUE_RUNONLY=0 npx tsx scripts/run_nn2rtl_top_value.ts 0 2>&1 | tail -8
echo "CONFIGB_DONE exit=$?"
