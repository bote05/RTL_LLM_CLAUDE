read_verilog C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/reports_integrated/ooc/sp_r.v
synth_design -top sp_r -part xcu250-figd2104-2L-e -mode out_of_context -generic W=432 -generic DEPTH=256 -generic MEM_INIT_FILE=C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/reports_integrated/ooc/sp_init_432.mem
report_utilization -file C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/reports_integrated/ooc/r432.rpt
puts D
exit
