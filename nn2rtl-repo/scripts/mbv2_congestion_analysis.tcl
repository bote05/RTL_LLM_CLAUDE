# Fast read-only diagnosis of the placed (physopt) MobileNetV2 checkpoint:
# congestion region map + SLR (4-die) utilization distribution + worst critical paths
# + top high-fanout nets. NO place/route -- just open + report (~10-15 min, low RAM).
# Drives the Fmax/congestion campaign: tells us WHERE to floorplan + WHICH nets to fix.
set_param general.maxThreads 8
set CKDIR "C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/reports/synth/checkpoints"
puts "NN2RTL_INFO: open_checkpoint physopt"
open_checkpoint "$CKDIR/mbv2_route_physopt_c8.dcp"
puts "NN2RTL_INFO: report_utilization (incl SLR distribution for the SSI part)"
report_utilization -file "$CKDIR/mbv2_analysis_util.rpt"
puts "NN2RTL_INFO: report_design_analysis -congestion (region map)"
catch { report_design_analysis -congestion -file "$CKDIR/mbv2_analysis_congestion.rpt" }
puts "NN2RTL_INFO: report_design_analysis -complexity (rent/connectivity per region)"
catch { report_design_analysis -complexity -file "$CKDIR/mbv2_analysis_complexity.rpt" }
puts "NN2RTL_INFO: top high-fanout nets"
catch { report_high_fanout_nets -max_nets 40 -file "$CKDIR/mbv2_analysis_fanout.rpt" }
puts "NN2RTL_INFO: worst critical paths (placement-estimated; logic-depth vs route)"
report_timing_summary -max_paths 15 -file "$CKDIR/mbv2_analysis_timing.rpt"
puts "NN2RTL_INFO: per-SLR CLB distribution"
catch { report_utilization -slr -file "$CKDIR/mbv2_analysis_slr.rpt" }
puts "NN2RTL_INFO: congestion analysis complete"
