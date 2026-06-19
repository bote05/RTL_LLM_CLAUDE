# ResNet-50 Config B — Step 0 routability localize (READ-ONLY: open + report, NO route).
# Uses the physopt checkpoint (the placed one logged 'failed integrity check (2)').
set base "C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/reports_integrated"
set ck "$base/checkpoints/first_light_physopt_configB.dcp"
set out "$base/step0"
puts "NN2RTL_STEP0: opening $ck"
open_checkpoint $ck
puts "NN2RTL_STEP0: route status"
report_route_status -file ${out}_route_status.rpt
puts "NN2RTL_STEP0: utilization (incl per-SLR)"
report_utilization -file ${out}_util.rpt
puts "NN2RTL_STEP0: design analysis (congestion + complexity)"
report_design_analysis -congestion -complexity -file ${out}_analysis.rpt
puts "NN2RTL_STEP0: timing summary"
report_timing_summary -max_paths 5 -file ${out}_timing.rpt
puts "NN2RTL_STEP0: DONE"
