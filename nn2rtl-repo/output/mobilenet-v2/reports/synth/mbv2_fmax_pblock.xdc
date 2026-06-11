# =============================================================================
# mbv2_fmax_pblock.xdc -- RETIRED 2026-06-11 [FINAL-BUNDLE] (comment-only file)
# =============================================================================
# The previous SLR floorplan in this file was written against the PRE-engine-
# dispatch netlist. It constrained cells that NO LONGER EXIST after the
# DW-ENGINE-EXT / DW-QUARTET / FC-ENGINE waves moved every deep conv onto the
# shared engine:
#   u_node_conv_854 / 860 / 866 / 872 / 878 / 884 / 890 / 896 / 902 / 908
#   u_br_ldr28 / u_br_ldr30 / u_br_ldr32
# (only u_node_conv_810/812 remain spatial; u_br_ldr22/24/26 + u_node_add_1038/
# 1110 still exist but were floorplanned around setup walls that are gone).
#
# Loading this file crashed place_design with EXCEPTION_ACCESS_VIOLATION on the
# new netlist; the new c8b route closes WITHOUT it (--no-pblock, routed
# 86.67 MHz @8ns target, WNS -3.538). Per the final-bundle decision it is
# retired rather than rewritten: the remaining WNS is being attacked with
# cycle-neutral RTL fanout fixes (see scripts/apply_mbv2_final_bundle.py and
# docs/agent_tasks/MBV2_FINAL_BUNDLE_ANALYSIS.md), and an unproven floorplan on
# the last MBV2 synth is a gamble.
#
# If a future route stalls on the engine_output_fifo URAM -> bridge beat_buf
# distance class (0 logic levels, pure route; -4.003/-3.963ns in
# checkpoints/mbv2_route_postroute_timing_new_c8b.rpt), a MINIMAL replacement
# would be ONE soft pblock keeping u_shared_engine + u_engine_out_fifo + the
# act loaders in the URAM-column SLR -- rebuild it from the LIVE netlist cell
# names (report_utilization -hierarchical) and verify with place_design before
# trusting it.
#
# This file is intentionally constraint-free.
# =============================================================================
