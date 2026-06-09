# =============================================================================
# mbv2_fmax_pblock.xdc  -- MobileNetV2 engine-top SLR floorplan (Fmax)
# Target: xcu250-figd2104-2L-e (4 SLRs, clock-region rows Y0..Y15, 4 rows/SLR)
#   SLR0 = CLOCKREGION X0Y0 : X7Y3
#   SLR1 = CLOCKREGION X0Y4 : X7Y7   (URAM-heavy engine pinned here)
#   SLR2 = CLOCKREGION X0Y8 : X7Y11
#   SLR3 = CLOCKREGION X0Y12: X7Y15  (most CLB headroom in the c8 route: 87.5%)
# (SLR<->CR map verified from c8 placement: SLICE Y462 = SLR1, Y924 = SLR3,
#  and the report labels the killer hops "SLR Crossing[1->3]" / "[3->1]".)
#
# WHY (verified from the c8 physopt postPlace reports, READ-ONLY):
#   * The ENTIRE setup wall is two convs: every top-30 setup path is
#     u_node_conv_866/mac_lane_q2_reg -> acc_reg (27 paths) or
#     u_node_conv_860/mac_lane_q2_reg -> acc_reg (18 paths). Data Path Delay
#     12.54ns = logic 1.15ns (9%) + route 11.38ns (91%). The two killer hops are
#     net acc__0[0] = 4.863ns (SLR Crossing[1->3]) and net
#     scheduler/p_0_out[21] = 5.199ns (SLR Crossing[3->1]): ~10.06ns of pure SLL
#     crossing INSIDE ONE conv's single-cycle accumulate (acc[lane] += sum_comb).
#   * Root cause = CLB-site saturation (avg 95.78%; SLR1 100%, SLR2 99.9%,
#     SLR0 95.65%) while raw LUT is only 76.5%. With no SLR holding room for a
#     whole conv, the placer tears conv_860/866's accumulate datapath across
#     SLR1<->SLR3. The 232 URAM sit in SLR1, pinning u_shared_engine +
#     u_engine_out_fifo there and stuffing SLR1 CLBs to 100%.
#
# FIX (byte-exact by construction -- PLACEMENT ONLY, no netlist/value change):
#   Confine each hot conv to ONE SLR so its acc->scheduler->acc loop cannot
#   cross an SLL. Keep the URAM engine + its input gather bridges in SLR1; push
#   the WNS-owning deep convs (854/860/866/872) to the roomiest die SLR3; spread
#   the mp16 convs (878..908) + the two residual adds across SLR2/SLR0 away from
#   the saturated SLR1. pblocks are SOFT (no EXCLUDE_PLACEMENT / IS_SOFT=false)
#   so the placer may still spill non-critical leaf cells -- they bias, not
#   hard-fence. The crossing being INTRA-conv is the key: once a conv is confined
#   to one SLR, the crossing is physically impossible -> NO RTL pipeline register
#   is needed (and a register here is NOT a free latency-only insert: the accum
#   is a feedback path whose drain gates ST_BIAS, so registering it changes the
#   schedule and is NOT byte-exact without a coordinated FSM + latency-formula +
#   golden rebuild -- see the analysis note. Hence pblock-only.)
#
# NOTE: placement constraints only -> NO logic change -> e2e stays 8/8 byte-exact.
# Run the e2e gate (scripts/run_mbv2_e2e_parallel.sh) after route regardless.
# =============================================================================

# ---- SLR1: URAM engine cluster + its input gather bridges --------------------
# The engine is pinned by the 232 URAM banks (all in SLR1) + the 29-URAM out
# fifo. The retile_gather bridges u_br_ldr22..32 FEED the engine input (2048b
# beats); co-locating them with the engine keeps the wide gather->engine bus
# SLL-local. (These are the REAL bridge instance names -- the prior XDC named
# u_br_878..u_br_908 which DO NOT EXIST in the netlist, so -quiet silently
# dropped them and left the bridges unconstrained.)
create_pblock pblk_engine
resize_pblock pblk_engine -add {CLOCKREGION_X0Y4:CLOCKREGION_X7Y7}
add_cells_to_pblock pblk_engine [get_cells -quiet {
  u_shared_engine u_engine_out_fifo
  u_br_ldr22 u_br_ldr24 u_br_ldr26 u_br_ldr28 u_br_ldr30 u_br_ldr32
}]

# ---- SLR3: the WNS-owning deep convs 854/860/866/872 (roomiest die) ----------
# conv_866 (27 of top-30 setup paths) + conv_860 (the other 18) are the ONLY
# setup-critical owners; 854/872 are their streaming-chain neighbors, glued in
# the same die so the chain stays local. Each conv's whole acc->scheduler->acc
# loop now lives in one SLR -> no SLL inside it.
create_pblock pblk_deep_hi
resize_pblock pblk_deep_hi -add {CLOCKREGION_X0Y12:CLOCKREGION_X7Y15}
add_cells_to_pblock pblk_deep_hi [get_cells -quiet {
  u_node_conv_854 u_node_conv_860 u_node_conv_866 u_node_conv_872
}]

# ---- SLR2: mp16 convs 878/884/890 --------------------------------------------
# Not on the current WNS, but mp16-widened (C=576) -> same intra-conv accumulate
# topology; confine each to one SLR so they cannot become the next crossing.
create_pblock pblk_mp16_a
resize_pblock pblk_mp16_a -add {CLOCKREGION_X0Y8:CLOCKREGION_X7Y11}
add_cells_to_pblock pblk_mp16_a [get_cells -quiet {
  u_node_conv_878 u_node_conv_884 u_node_conv_890
}]

# ---- SLR0: mp16 convs 896/902/908 + the two residual adds --------------------
# C=960 deep convs; node_add_1038/1110 are the final-stage residual joins, kept
# in the same die as 896/902/908 so the add operands stay local.
create_pblock pblk_mp16_b
resize_pblock pblk_mp16_b -add {CLOCKREGION_X0Y0:CLOCKREGION_X7Y3}
add_cells_to_pblock pblk_mp16_b [get_cells -quiet {
  u_node_conv_896 u_node_conv_902 u_node_conv_908
  u_node_add_1038 u_node_add_1110
}]

# pblocks intentionally allow placer spill (soft). Do NOT set:
#   set_property EXCLUDE_PLACEMENT true  [get_pblocks *]
#   set_property IS_SOFT false           [get_pblocks *]
# Tighten to hard ONLY if a route still scatters conv_860/866 across an SLL.
