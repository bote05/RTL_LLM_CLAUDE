# Route-only resume from the KPAR8 physopt checkpoint (placed @67.95 MHz, +1.283).
# No RTL changes; just complete routing that timed out at 12h last attempt.
set_param general.maxThreads 8
set CKPT D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/reports_integrated/checkpoints
open_checkpoint $CKPT/first_light_physopt_kpar8_c16.dcp
puts "=== PRE-ROUTE (placed) timing ==="
report_timing_summary -no_detailed_paths -file $CKPT/kpar8_preroute_timing.rpt
route_design -directive Explore
write_checkpoint -force $CKPT/first_light_routed_kpar8_c16.dcp
report_timing_summary -max_paths 10 -file $CKPT/routed_kpar8_c16_timing.rpt
report_utilization -file $CKPT/routed_kpar8_c16_util.rpt
puts "=== ROUTE DONE, running post-route phys_opt for hold ==="
phys_opt_design -directive Explore
write_checkpoint -force $CKPT/first_light_routed_physopt_kpar8_c16.dcp
report_timing_summary -max_paths 10 -file $CKPT/routed_physopt_kpar8_c16_timing.rpt
report_power -file $CKPT/routed_kpar8_c16_power.rpt
puts "KPAR8_ROUTE_DONE"
