read_verilog C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/reports_integrated/ooc/sp_int3.v
synth_design -top sp_int3 -part xcu250-figd2104-2L-e -mode out_of_context -generic W=512 -generic DEPTH=16384
report_utilization -file C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/reports_integrated/ooc/sp512.rpt
puts D
exit
