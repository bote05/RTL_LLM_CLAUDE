read_verilog -sv {C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/rtl/node_linear.v}
synth_design -top node_linear -part xcu250-figd2104-2L-e -mode out_of_context
puts "==== REPORT_UTILIZATION BEGIN ===="
report_utilization
puts "==== REPORT_UTILIZATION END ===="
