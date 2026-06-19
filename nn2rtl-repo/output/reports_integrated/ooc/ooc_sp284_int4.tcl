read_verilog C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/reports_integrated/ooc/ooc_spatial_rom.v
synth_design -top ooc_spatial_rom -part xcu250-figd2104-2L-e -mode out_of_context -generic DEPTH=16384 -generic WIDE_W=576 -generic ADDR_W=14 -generic MEM_INIT=C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/backups/allint4_byteexact/node_conv_284_weights_mp_k_9.hex
report_utilization -file C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/reports_integrated/ooc/sp284_int4.rpt
puts EDONE
exit
