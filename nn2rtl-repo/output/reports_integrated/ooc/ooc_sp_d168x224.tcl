read_verilog C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/reports_integrated/ooc/ooc_spatial_rom.v
synth_design -top ooc_spatial_rom -part xcu250-figd2104-2L-e -mode out_of_context -generic DEPTH=168 -generic WIDE_W=224 -generic ADDR_W=8 -generic MEM_INIT=C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_196_weights_mp_k_7.hex
report_utilization -file C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/reports_integrated/ooc/sp_d168x224.rpt
puts EDONE
exit
