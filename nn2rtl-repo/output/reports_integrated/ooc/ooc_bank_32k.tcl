read_verilog C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/reports_integrated/ooc/uram_weight_bank_128.v
synth_design -top uram_weight_bank_128 -part xcu250-figd2104-2L-e -mode out_of_context -generic DEPTH=32768 -generic ADDR_W=15 -generic MEM_INIT_FILE=C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/uram_weights_bank0.mem
report_utilization -file C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/reports_integrated/ooc/bank_32k.rpt
puts NN2RTL_32K_DONE
exit
