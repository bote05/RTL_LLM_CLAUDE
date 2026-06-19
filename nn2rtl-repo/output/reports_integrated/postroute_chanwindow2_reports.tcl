## Post-route bitstream-readiness reports on the routed chanwindow2 checkpoint.
## Read-only: open + report_route_status / report_drc / report_power / report_timing.
set rd C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/reports_integrated
open_checkpoint $rd/checkpoints/first_light_routed_chanwindow2.dcp
puts "NN2RTL_INFO: report_route_status"
report_route_status -file $rd/postroute_chanwindow2_route_status.rpt
puts "NN2RTL_INFO: report_drc (bitstream-readiness gate)"
report_drc -file $rd/postroute_chanwindow2_drc.rpt
puts "NN2RTL_INFO: report_power"
report_power -file $rd/postroute_chanwindow2_power.rpt
puts "NN2RTL_INFO: report_timing_summary"
report_timing_summary -max_paths 10 -file $rd/postroute_chanwindow2_timing.rpt
puts "NN2RTL_INFO: postroute reports complete"
