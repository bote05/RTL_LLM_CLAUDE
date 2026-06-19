read_verilog C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/reports_integrated/ooc/uram_weight_bank_only.v
synth_design -top uram_weight_bank -part xcu250-figd2104-2L-e -mode out_of_context -generic DEPTH=39424 -generic ADDR_W=17 -generic MEM_INIT_FILE=C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/uram_weights_bank0.mem
report_utilization -file C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/reports_integrated/ooc/bank_asis_39424.rpt
puts NN2RTL_ASIS_DONE
exit
