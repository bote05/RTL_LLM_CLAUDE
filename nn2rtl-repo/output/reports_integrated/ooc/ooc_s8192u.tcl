read_verilog C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/reports_integrated/ooc/sfifo_sync.v
synth_design -top sfifo_sync -part xcu250-figd2104-2L-e -mode out_of_context -generic WIDTH=256 -generic DEPTH=8192 -generic STYLE=ultra
report_utilization -file C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/reports_integrated/ooc/s8192u.rpt
puts D
exit
