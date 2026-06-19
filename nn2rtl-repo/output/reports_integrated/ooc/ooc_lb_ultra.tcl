read_verilog C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/reports_integrated/ooc/lbslot.v
synth_design -top lbslot -part xcu250-figd2104-2L-e -mode out_of_context -generic IC=512 -generic MEM_DEPTH=58 -generic STYLE=ultra
report_utilization -file C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/reports_integrated/ooc/lb_ultra.rpt
puts LB_ultra_DONE
exit
