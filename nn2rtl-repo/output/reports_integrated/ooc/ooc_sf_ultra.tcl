read_verilog C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/reports_integrated/ooc/skip_fifo_ultra.v
synth_design -top skip_fifo_ultra -part xcu250-figd2104-2L-e -mode out_of_context -generic WIDTH=256 -generic DEPTH=1024
report_utilization -file C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/reports_integrated/ooc/sf_ultra.rpt
puts SF_ultra_DONE
exit
