# Synthesize shared_engine out-of-context, then write a functional (funcsim)
# gate-level netlist. The netlist drops into engine_one_layer_tb in place of the
# RTL engine (same module name + ports) for the definitive real-hardware check.
#
#   vivado -mode batch -source engine_funcsim_synth.tcl -tclargs <ROOT> <NETLIST_OUT>
#
# The engine has NO internal memory (act/uram/bias live in the TB), so the
# netlist is pure compute logic + DSP — the URAM-init problem does not apply.
set ROOT    [lindex $argv 0]
set NETLIST [lindex $argv 1]
set PART    xcu250-figd2104-2L-e

# Use the XSim-hoisted skeleton copy (VRFC-clean) and prepend the subblocks
# define so the empty stubs are excluded and engine/*.v implementations win.
set skel_hoist $ROOT/build_engine_xsim/shared_engine_skeleton_xsim.v
set fin  [open $skel_hoist r]; set body [read $fin]; close $fin
set skel_def $ROOT/build_engine_xsim/shared_engine_skeleton_synth.v
set fout [open $skel_def w]
puts $fout "`define NN2RTL_ENGINE_SUBBLOCKS_PROVIDED 1"
puts -nonewline $fout $body
close $fout

read_verilog -sv [list \
  $skel_def \
  $ROOT/output/rtl/engine/address_generator.v \
  $ROOT/output/rtl/engine/config_register_block.v \
  $ROOT/output/rtl/engine/mac_array.v \
  $ROOT/output/rtl/engine/requant_pipeline.v \
  $ROOT/output/rtl/engine/bram_to_stream_bridge.v ]

synth_design -top shared_engine -part $PART -mode out_of_context -flatten_hierarchy rebuilt
puts "NN2RTL_INFO: synth_design complete"
report_utilization
write_verilog -force -mode funcsim $NETLIST
puts "NN2RTL_INFO: wrote funcsim netlist -> $NETLIST"
