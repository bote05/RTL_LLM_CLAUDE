# OOC synth of banked node_linear to confirm RAMB36 inference (weight ROM no longer LUT).
read_verilog -sv {C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/rtl/node_linear.v}
synth_design -top node_linear -part xcu250-figd2104-2L-e -mode out_of_context
puts "=== UTILIZATION REPORT ==="
report_utilization
puts "OOC_DONE"
