# Re-time a ROUTED checkpoint at a relaxed clock to find the timing-MET period.
# No re-route; just re-applies the clock constraint and reports timing.
#   vivado -mode batch -source scripts/resnet8_retime.tcl -tclargs <routed.dcp> <clock_ns> <out_timing.rpt>
set routed_dcp [lindex $argv 0]
set clock_ns   [lindex $argv 1]
set out_rpt    [lindex $argv 2]
puts "RETIME: open $routed_dcp  clock=${clock_ns}ns"
open_checkpoint $routed_dcp
# Override the clock period via create_clock (set_property PERIOD is read-only).
# create_clock on the same source net redefines the clock at the new period.
create_clock -name clk -period $clock_ns [get_ports clk]
report_timing_summary -check_timing_verbose -max_paths 10 -file $out_rpt
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "RETIME_RESULT: clock_ns=$clock_ns setup_WNS=$wns"
puts "RETIME: DONE"
