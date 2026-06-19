## Read-only congestion/complexity analysis of the chan_window physopt (placed) checkpoint.
## Goal: LOCATE the residual routing-congestion hotspot (the 2022 node overlaps that Explore +
## AggressiveExplore both failed on) so a SURGICAL pblock can spread it WITHOUT the global Fmax
## trade of AltSpreadLogic. No place/route here -> Fmax-neutral, low RAM.
open_checkpoint C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/reports_integrated/checkpoints/first_light_physopt_chanwindow.dcp
puts "NN2RTL_INFO: report_design_analysis -congestion -complexity"
report_design_analysis -congestion -complexity -file C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/reports_integrated/congestion_analysis.rpt
puts "NN2RTL_INFO: per-SLR / per-clock-region utilization"
report_utilization -file C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/reports_integrated/congestion_util.rpt
puts "NN2RTL_INFO: congestion analysis complete"
