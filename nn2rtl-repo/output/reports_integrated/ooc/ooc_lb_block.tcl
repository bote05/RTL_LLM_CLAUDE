read_verilog C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/reports_integrated/ooc/lbslot.v
synth_design -top lbslot -part xcu250-figd2104-2L-e -mode out_of_context -generic IC=512 -generic MEM_DEPTH=58 -generic STYLE=block
report_utilization -file C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/reports_integrated/ooc/lb_block.rpt
puts LB_block_DONE
exit
