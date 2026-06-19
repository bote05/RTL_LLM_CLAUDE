# Resume opt->place->route from a SYNTH checkpoint at a chosen clock period.
# Avoids re-synthesizing on clock-sweep retries.
#   vivado -mode batch -source scripts/resnet8_resume_route.tcl -tclargs <synth.dcp> <clock_ns> <out_prefix>
# Writes <out_prefix>_routed.dcp + <out_prefix>_postroute_{util,timing}.rpt
set synth_dcp [lindex $argv 0]
set clock_ns  [lindex $argv 1]
set out_pref  [lindex $argv 2]
set_param general.maxThreads 8
puts "RESUME: read_checkpoint $synth_dcp  clock=${clock_ns}ns  out=$out_pref"
open_checkpoint $synth_dcp
# Re-assert the clock at the requested period. set_property PERIOD on get_clocks is
# a silent no-op and create_clock -add ADDS a SECOND clock (leaving the old period
# the binding constraint) -> the requested clock would never take effect. Calling
# create_clock WITHOUT -add on the SAME name (clk) REDEFINES/overrides the existing
# constraint, so $clock_ns becomes authoritative (verified by the puts below).
create_clock -name clk -period $clock_ns [get_ports clk]
puts "RESUME: effective clk period = [get_property PERIOD [get_clocks clk]] ns (requested ${clock_ns})"
opt_design
puts "RESUME: place_design"
place_design
puts "RESUME: phys_opt (pre-route)"
catch { phys_opt_design }
puts "RESUME: route_design"
route_design
write_checkpoint -force ${out_pref}_routed.dcp
puts "RESUME: post-route phys_opt"
catch { phys_opt_design }
write_checkpoint -force ${out_pref}_routed.dcp
report_utilization -file ${out_pref}_postroute_util.rpt
report_timing_summary -check_timing_verbose -max_paths 20 -file ${out_pref}_postroute_timing.rpt
puts "RESUME: DONE"
