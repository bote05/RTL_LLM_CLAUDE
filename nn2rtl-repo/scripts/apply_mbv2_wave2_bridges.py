#!/usr/bin/env python3
"""
apply_mbv2_wave2_bridges.py

Implement the WAVE-2 tiled<->flat RETILE BRIDGE for the MobileNet-v2 final
stage in the (already-patched, not-regenerated) top:

    output/mobilenet-v2/rtl/nn2rtl_top.v

WHY
---
build_top_wrapper.ts never implemented the tiled<->full retile bridge ("wave-2",
its own TODO at lines 749/801). The final stage ALTERNATES contracts because the
channel counts (576/960/1280) blow past the 4096-bit flat-bus cap:

    * pointwise convs + relus -> tiled-streaming (256b bus, channel_tile=32)
    * depthwise convs         -> depthwise-conv  (4096b bus, 2 beats/pixel)
    * residual adds / mean     -> flat-bus        (full channel pack per beat)

Adjacent modules whose contracts differ cannot wire directly: a tiled producer
emits N 256b beats/pixel while a full-width consumer wants 1 (or 2) wide
beats/pixel. The wrapper therefore left 11 convs orphaned on PIXEL_IN and wired
every other final-stage consumer one step too early (skipping the orphan).

THIS PATCH
----------
For every boundary whose producer/consumer contracts differ, instantiate a
retile_gather / retile_scatter (rtl_library/retile_bridge.v) between the TRUE
producer and the consumer, and re-point the consumer (and the add main/skip
operands + the mean input) onto the bridge. Shifted same-contract passthroughs
are re-pointed straight to their true tiled producer.

The byte layout is contiguous on both sides (tiled beat k == full[k*256+:256]),
so the bridges are a pure gather/scatter -- NO channel reordering.

Handshake (ALWAYS-ACCEPT PING-PONG bridge, deadlock-free):
    consumer.valid_in = bridge.valid_out & spatial_run_drain_<i>   [per-bridge gate]
    bridge.valid_in   = producer.valid_out                         [RAW, always-accept]
    bridge.ready_down = <consumer's actual-accept condition>       [NO spatial_run!]
    bridge.drain_en   = spatial_run_drain_<i>                       [per-bridge gate]
where the actual-accept condition is the consumer's ready_in (for tiled convs/
relus, the depthwise, and the mean), or ready_in & skip_valid (for the adds,
which only latch when BOTH operands are valid), or skip_in_ready (for the
add_1038 skip FIFO push), and the per-bridge gate is
    spatial_run_drain_<i> = ~(engine_busy | sched_spatial_stall
                              | (any_retile_stall & ~br_<i>_stall_out))
i.e. spatial_run with ONLY this bridge's OWN stall term removed.

================================================================================
DEADLOCK ROOT CAUSE (fixed by this revision) and the INVARIANT it enforces
================================================================================
The previous revision gated the DRAIN and the CONSUMER LATCH by DIFFERENT
signals, differing by exactly the any_retile_stall term:

  - drain advanced on   xfer = valid_out & ready_down & drain_en
        with drain_en = ~(engine_busy | sched_spatial_stall)   [EXCLUDES a.r.s.]
  - consumer latched on valid_in = bridge_valid_out & spatial_run
        with spatial_run = ~(engine_busy | sched_spatial_stall | any_retile_stall)
                                                                [INCLUDES a.r.s.]

With ONE bridge that is self-consistent. With 23 bridges sharing one GLOBAL
any_retile_stall it is fatal: whenever ANY bridge X is full (stall_out=1 ->
any_retile_stall=1 -> spatial_run=0), a DIFFERENT bridge Y that is mid-drain
still sees drain_en=1, so Y fires xfer -- Y frees its read buffer and advances
rsel/e_idx, COUNTING the beat as transferred -- but Y's consumer's
valid_in = Y_valid_out & spatial_run = ...&0 = 0, so the consumer does NOT
latch.  That beat is SILENTLY DROPPED.  Y and its consumer are now permanently
misaligned by one beat -> the consumer never completes a pixel -> upstream
starves -> every full bridge stays full -> any_retile_stall latches high ->
spatial_run latches to 0 forever -> s_axis_tready=0 (input frozen) chain-wide.

THE INVARIANT, now enforced exactly:  for every bridge i,
    (drain advances)  <=>  (consumer latches)   on the SAME cycle, SAME gate.
Both sides are now gated by the identical per-bridge signal spatial_run_drain_i:
    drain xfer          = br_i_valid_out & ready_down_i & spatial_run_drain_i
    consumer latches    = br_i_valid_out & ready_down_i & spatial_run_drain_i
(the consumer internally accepts only when its own ready_in -- which IS
ready_down_i -- is high, and we re-gate its valid_in spatial term to
spatial_run_drain_i).  They are bit-identical -> no lost beat is possible.

SELF-FREEZE is broken the RIGHT way: spatial_run_drain_i masks out ONLY bridge
i's OWN stall.  So:
  - When bridge i is full (br_i_stall_out=1): drain_en_i still excludes only
    i's own term, so i keeps draining into its consumer (whose valid_in uses the
    SAME spatial_run_drain_i, so it keeps latching) -> i self-clears.  No
    self-freeze.
  - When a DIFFERENT bridge X is full (br_X_stall_out=1): any_retile_stall&~br_i
    is high, so spatial_run_drain_i=0 -> bridge i's drain AND its consumer's
    latch freeze TOGETHER, in lockstep -> NO lost beat, NO misalignment.

INTAKE is now TRULY always-accept (the bridge's whole reason for being a
ping-pong): bridge.valid_in is the producer's RAW valid_out with NO spatial_run.
This is REQUIRED because the MobileNet producers FREE-RUN their output and do
NOT honor backpressure:
  - relu (n4_*) free-run valid_out for the entire 18/30-beat send window
    (n4_23.v lines 84-104: sending-branch sets valid_out=1 every cycle, looks at
    no ready);
  - depthwise convs emit beat0/beat1 purely off cyc_cnt/pix_done/em_phase
    (node_conv_878.v lines 217-231: no ready sampled);
so a beat that meets valid_in=0 (because spatial_run dropped for ANY other
bridge) would be SILENTLY DROPPED on the INTAKE side too -- the exact same
lost-beat class.  The bridge already gates do_write = valid_in & wsel_empty, so
RAW valid_in is genuinely always-accept: it writes a beat iff the selected write
buffer is free, and never drops a free-buffer beat.  Backpressure is supplied by
the per-bridge stall throttling the PRODUCER's own NEW-pixel start (the producer
instances keep "& spatial_run" on their valid_in), plus the one-pixel ping-pong
slack so the bridge does not fill mid-send.

No combinational loop: br_i_stall_out = (full0 & full1) is register-derived;
ready_down_i is each consumer's registered status ready_in (add/conv/relu/mean/
FIFO accept logic, never a function of any bridge stall); spatial_run_drain_i
gates only the NEXT-cycle register update via xfer.  any_retile_stall is a pure
OR of register-derived stall_out bits.

Idempotent: re-running is a no-op once the bridge block + re-pointing exist.
"""
import argparse
import os
import re
import sys

# Default target is the running BASELINE top (unchanged behavior).  The engine-
# dispatched top (nn2rtl_top_engine.v) can be targeted via --top or the
# NN2RTL_TOP env var WITHOUT changing the default.
DEFAULT_TOP = r"D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/rtl/nn2rtl_top.v"
TOP = DEFAULT_TOP

MARK_BEGIN = "    // ===== WAVE-2 RETILE BRIDGES (apply_mbv2_wave2_bridges.py) ====="
MARK_END   = "    // ===== END WAVE-2 RETILE BRIDGES ====="

# ---------------------------------------------------------------------------
# Bridge instances.
#   kind     : "gather" (tiled->full) or "scatter" (full->tiled)
#   name     : instance/wire base name (br_*)
#   producer : true upstream module id (drives bridge valid_in/data_in)
#   tile_w/n_tiles/full_beat_w/full_beats/spatial : module params
#   The "full" side width = full_beat_w with full_beats beats/pixel.
#   ready_down : the consumer's real accept qualifier (without spatial_run).
#   consumer   : the single downstream module/FIFO instance this bridge feeds.
#                Its valid_in spatial gate is re-written from the GLOBAL
#                spatial_run to this bridge's PER-BRIDGE spatial_run_drain_<name>
#                so (drain xfer) and (consumer latch) are bit-identical.
#   out_w      : bridge data_out bus width (gather=full_beat_w, scatter=tile_w)
# ---------------------------------------------------------------------------
BRIDGES = [
    # ---- flat -> tiled (SCATTER): feed pointwise conv inputs ----
    dict(kind="scatter", name="br_876", producer="node_conv_874",
         tile_w=256, n_tiles=3,  full_beat_w=768,   full_beats=1, spatial=196,
         ready_down="node_conv_876_ready_in", consumer="node_conv_876"),
    dict(kind="scatter", name="br_882", producer="node_add_828",
         tile_w=256, n_tiles=3,  full_beat_w=768,   full_beats=1, spatial=196,
         ready_down="node_conv_882_ready_in", consumer="node_conv_882"),
    dict(kind="scatter", name="br_888", producer="node_add_900",
         tile_w=256, n_tiles=3,  full_beat_w=768,   full_beats=1, spatial=196,
         ready_down="node_conv_888_ready_in", consumer="node_conv_888"),
    dict(kind="scatter", name="br_900c", producer="node_add_1038",
         tile_w=256, n_tiles=5,  full_beat_w=1280,  full_beats=1, spatial=49,
         ready_down="node_conv_900_ready_in", consumer="node_conv_900"),
    dict(kind="scatter", name="br_906", producer="node_add_1110",
         tile_w=256, n_tiles=5,  full_beat_w=1280,  full_beats=1, spatial=49,
         ready_down="node_conv_906_ready_in", consumer="node_conv_906"),

    # ---- tiled -> depthwise (GATHER): 2 beats/pixel of 4096b ----
    dict(kind="gather", name="br_878", producer="n4_23",
         tile_w=256, n_tiles=18, full_beat_w=4608, full_beats=1, spatial=196,
         ready_down="node_conv_878_ready_in", consumer="node_conv_878"),
    dict(kind="gather", name="br_884", producer="n4_25",
         tile_w=256, n_tiles=18, full_beat_w=4096, full_beats=2, spatial=196,
         ready_down="node_conv_884_ready_in", consumer="node_conv_884"),
    dict(kind="gather", name="br_890", producer="n4_27",
         tile_w=256, n_tiles=18, full_beat_w=4608, full_beats=1, spatial=196,
         ready_down="node_conv_890_ready_in", consumer="node_conv_890"),
    dict(kind="gather", name="br_896", producer="n4_29",
         tile_w=256, n_tiles=30, full_beat_w=7680, full_beats=1, spatial=49,
         ready_down="node_conv_896_ready_in", consumer="node_conv_896"),
    dict(kind="gather", name="br_902", producer="n4_31",
         tile_w=256, n_tiles=30, full_beat_w=7680, full_beats=1, spatial=49,
         ready_down="node_conv_902_ready_in", consumer="node_conv_902"),
    dict(kind="gather", name="br_908", producer="n4_33",
         tile_w=256, n_tiles=30, full_beat_w=4096, full_beats=2, spatial=49,
         ready_down="node_conv_908_ready_in", consumer="node_conv_908"),

    # ---- depthwise -> tiled (SCATTER): feed the relu inputs (2->N) ----
    dict(kind="scatter", name="br_n4_24", producer="node_conv_878",
         tile_w=256, n_tiles=18, full_beat_w=4608, full_beats=1, spatial=196,
         ready_down="n4_24_ready_in", consumer="n4_24"),
    dict(kind="scatter", name="br_n4_26", producer="node_conv_884",
         tile_w=256, n_tiles=18, full_beat_w=4096, full_beats=2, spatial=196,
         ready_down="n4_26_ready_in", consumer="n4_26"),
    dict(kind="scatter", name="br_n4_28", producer="node_conv_890",
         tile_w=256, n_tiles=18, full_beat_w=4608, full_beats=1, spatial=49,
         ready_down="n4_28_ready_in", consumer="n4_28"),
    dict(kind="scatter", name="br_n4_30", producer="node_conv_896",
         tile_w=256, n_tiles=30, full_beat_w=7680, full_beats=1, spatial=49,
         ready_down="n4_30_ready_in", consumer="n4_30"),
    dict(kind="scatter", name="br_n4_32", producer="node_conv_902",
         tile_w=256, n_tiles=30, full_beat_w=7680, full_beats=1, spatial=49,
         ready_down="n4_32_ready_in", consumer="n4_32"),
    dict(kind="scatter", name="br_n4_34", producer="node_conv_908",
         tile_w=256, n_tiles=30, full_beat_w=4096, full_beats=2, spatial=49,
         ready_down="n4_34_ready_in", consumer="n4_34"),

    # ---- tiled -> flat add LHS (GATHER): main path ----
    dict(kind="gather", name="br_828m", producer="node_conv_880",
         tile_w=256, n_tiles=3,  full_beat_w=768,   full_beats=1, spatial=196,
         ready_down="(node_add_828_ready_in & node_add_828_skip_valid)",
         consumer="node_add_828"),
    dict(kind="gather", name="br_900m", producer="node_conv_886",
         tile_w=256, n_tiles=3,  full_beat_w=768,   full_beats=1, spatial=196,
         ready_down="(node_add_900_ready_in & node_add_900_skip_valid)",
         consumer="node_add_900"),
    dict(kind="gather", name="br_1038m", producer="node_conv_898",
         tile_w=256, n_tiles=5,  full_beat_w=1280,  full_beats=1, spatial=49,
         ready_down="(node_add_1038_ready_in & node_add_1038_skip_valid)",
         consumer="node_add_1038"),
    dict(kind="gather", name="br_1110m", producer="node_conv_904",
         tile_w=256, n_tiles=5,  full_beat_w=1280,  full_beats=1, spatial=49,
         ready_down="(node_add_1110_ready_in & node_add_1110_skip_valid)",
         consumer="node_add_1110"),

    # ---- tiled -> flat add SKIP (GATHER): only add_1038 skip is tiled ----
    #   consumer is the skip FIFO PUSH port (u_skip_node_add_1038.in_valid).
    dict(kind="gather", name="br_1038s", producer="node_conv_892",
         tile_w=256, n_tiles=5,  full_beat_w=1280,  full_beats=1, spatial=49,
         ready_down="node_add_1038_skip_in_ready", consumer="u_skip_node_add_1038"),

    # ---- tiled -> flat mean (GATHER): 40 tiles -> one 10240b beat ----
    dict(kind="gather", name="br_mean", producer="n4_35",
         tile_w=256, n_tiles=40, full_beat_w=10240, full_beats=1, spatial=49,
         ready_down="node_mean_ready_in", consumer="node_mean"),
]

# ---------------------------------------------------------------------------
# Consumer re-pointing.
#   For each consumer we set its .valid_in (the leading source term, keeping the
#   "& spatial_run" tail) and .data_in. `src` may be a bridge ("br_*") or a true
#   tiled producer module id (for shifted same-contract passthroughs).
# Entries marked add_main / mean / skip are handled specially below.
# ---------------------------------------------------------------------------
# (consumer_id, valid_term, data_expr, spatial_gate)
#   spatial_gate is the wire that replaces the trailing "& spatial_run" on this
#   consumer's valid_in.  For a BRIDGE-fed consumer it is that bridge's
#   per-bridge "spatial_run_drain_<nm>" (so drain xfer == consumer latch,
#   bit-for-bit).  For a non-bridge passthrough it stays the global "spatial_run".
REPOINT = [
    # SCATTER -> pointwise convs
    ("node_conv_876", "br_876_valid_out",  "br_876_data_out",  "spatial_run_drain_br_876"),
    ("node_conv_882", "br_882_valid_out",  "br_882_data_out",  "spatial_run_drain_br_882"),
    ("node_conv_888", "br_888_valid_out",  "br_888_data_out",  "spatial_run_drain_br_888"),
    ("node_conv_900", "br_900c_valid_out", "br_900c_data_out", "spatial_run_drain_br_900c"),
    ("node_conv_906", "br_906_valid_out",  "br_906_data_out",  "spatial_run_drain_br_906"),
    # GATHER -> depthwise convs
    ("node_conv_878", "br_878_valid_out",  "br_878_data_out",  "spatial_run_drain_br_878"),
    ("node_conv_884", "br_884_valid_out",  "br_884_data_out",  "spatial_run_drain_br_884"),
    ("node_conv_890", "br_890_valid_out",  "br_890_data_out",  "spatial_run_drain_br_890"),
    ("node_conv_896", "br_896_valid_out",  "br_896_data_out",  "spatial_run_drain_br_896"),
    ("node_conv_902", "br_902_valid_out",  "br_902_data_out",  "spatial_run_drain_br_902"),
    ("node_conv_908", "br_908_valid_out",  "br_908_data_out",  "spatial_run_drain_br_908"),
    # SCATTER (from depthwise) -> relus
    ("n4_24", "br_n4_24_valid_out", "br_n4_24_data_out", "spatial_run_drain_br_n4_24"),
    ("n4_26", "br_n4_26_valid_out", "br_n4_26_data_out", "spatial_run_drain_br_n4_26"),
    ("n4_28", "br_n4_28_valid_out", "br_n4_28_data_out", "spatial_run_drain_br_n4_28"),
    ("n4_30", "br_n4_30_valid_out", "br_n4_30_data_out", "spatial_run_drain_br_n4_30"),
    ("n4_32", "br_n4_32_valid_out", "br_n4_32_data_out", "spatial_run_drain_br_n4_32"),
    ("n4_34", "br_n4_34_valid_out", "br_n4_34_data_out", "spatial_run_drain_br_n4_34"),
    # shifted same-contract (tiled->tiled) passthroughs -> true tiled producer.
    # NOT bridge-fed: keep the global spatial_run gate.
    ("n4_23",  "node_conv_876_valid_out", "node_conv_876_data_out", "spatial_run"),
    ("n4_25",  "node_conv_882_valid_out", "node_conv_882_data_out", "spatial_run"),
    ("n4_27",  "node_conv_888_valid_out", "node_conv_888_data_out", "spatial_run"),
    ("n4_31",  "node_conv_900_valid_out", "node_conv_900_data_out", "spatial_run"),
    ("n4_33",  "node_conv_906_valid_out", "node_conv_906_data_out", "spatial_run"),
]

# Add main-path GATHER: replace the {skip, conv_xxx_data_out[W:0]} data_in and
# the valid_in main term with the GATHER output.  The add's valid_in keeps its
# "& <skip_valid>" middle term but its trailing spatial gate is re-written to the
# main bridge's per-bridge spatial_run_drain (so drain == latch: ready_down for
# the main bridge is (add_ready_in & skip_valid), and the add latches when
# valid_in & add_ready_in -> identical boolean).
# (add_id, main_bridge, skip_data_wire, skip_valid_wire, spatial_gate)
ADD_MAIN = [
    ("node_add_828",  "br_828m",  "node_add_828_skip_data",  "node_add_828_skip_valid",
     "spatial_run_drain_br_828m"),
    ("node_add_900",  "br_900m",  "node_add_900_skip_data",  "node_add_900_skip_valid",
     "spatial_run_drain_br_900m"),
    ("node_add_1038", "br_1038m", "node_add_1038_skip_data", "node_add_1038_skip_valid",
     "spatial_run_drain_br_1038m"),
    ("node_add_1110", "br_1110m", "node_add_1110_skip_data", "node_add_1110_skip_valid",
     "spatial_run_drain_br_1110m"),
]

# Mean GATHER re-point (bridge-fed -> per-bridge gate).
MEAN = ("node_mean", "br_mean_valid_out", "br_mean_data_out", "spatial_run_drain_br_mean")

# add_1038 skip FIFO: re-source from the GATHER bridge (5 tiles -> 1280b).
#   The FIFO push (in_valid) is br_1038s's consumer; its spatial gate is
#   re-written to spatial_run_drain_br_1038s so the FIFO latch == bridge drain.
SKIP_FIFO_1038 = dict(add_id="node_add_1038", bridge="br_1038s", width=1280,
                      spatial_gate="spatial_run_drain_br_1038s")


# ---------------------------------------------------------------------------
def build_bridge_block():
    lines = [MARK_BEGIN,
             "    // Pure byte gather/scatter between tiled-streaming (256b, 32ch/beat)",
             "    // and flat/depthwise full-width beats. No channel reordering.",
             "",
             "    // PER-BRIDGE drain enable (== the gate on that bridge's consumer's",
             "    // valid_in spatial term, see the consumer re-pointing below): the global",
             "    // spatial_run with ONLY THIS bridge's OWN stall term removed.  This makes",
             "    // (drain xfer) and (consumer latch) bit-identical -> no lost beat (SAFETY),",
             "    // while a bridge's own full never blocks its own drain (no SELF-FREEZE).",
             "    // any_retile_stall and each br_*_stall_out are declared below; a continuous",
             "    // wire net assign may forward-reference a net (legal Verilog).",
             ""]
    for b in BRIDGES:
        nm = b["name"]
        lines.append(
            f"    wire spatial_run_drain_{nm} = "
            f"~(engine_busy | sched_spatial_stall | (any_retile_stall & ~{nm}_stall_out));")
    lines.append("")
    for b in BRIDGES:
        nm = b["name"]
        if b["kind"] == "gather":
            out_w = b["full_beat_w"]
        else:
            out_w = b["tile_w"]
        lines.append(f"    wire {nm}_valid_out;")
        lines.append(f"    wire [{out_w-1}:0] {nm}_data_out;")
        lines.append(f"    wire {nm}_ready_out;  // toward producer (free-running; observed for completeness)")
        lines.append(f"    wire {nm}_stall_out;  // (full0 & full1) -> spatial_throttle")
    lines.append("")

    for b in BRIDGES:
        nm = b["name"]
        prod = b["producer"]
        # INTAKE is RAW (always-accept).  The MobileNet producers FREE-RUN their
        # output and do not honor backpressure, so an "& spatial_run" on the
        # bridge intake would SILENTLY DROP a producer beat whenever spatial_run
        # dropped for ANY other bridge.  The bridge already gates
        # do_write = valid_in & wsel_empty, so RAW valid_in only writes into a
        # FREE buffer and never drops a free-buffer beat.  Backpressure is the
        # per-bridge stall throttling the PRODUCER's own new-pixel start (the
        # producer instance keeps "& spatial_run" on its valid_in) plus the
        # one-pixel ping-pong slack.
        #
        # DRAIN uses the PER-BRIDGE drain enable spatial_run_drain_<nm>, the
        # SAME signal that gates this bridge's consumer's valid_in spatial term.
        # -> (drain xfer) <=> (consumer latch) bit-for-bit (SAFETY: no lost beat)
        # and it masks out ONLY this bridge's own stall (SELF-FREEZE broken).
        # ready_down is the consumer's RAW accept term (its registered ready_in,
        # plus skip_valid for adds), NEVER a function of any bridge stall.
        rd = b["ready_down"]
        if b["kind"] == "gather":
            lines += [
                f"    retile_gather #(.TILE_W({b['tile_w']}), .N_TILES({b['n_tiles']}), "
                f".OUT_W({b['full_beat_w']}), .OUT_BEATS({b['full_beats']}), .SPATIAL({b['spatial']})) u_{nm} (",
                f"        .clk(clk), .rst_n(rst_n),",
                f"        .valid_in({prod}_valid_out),  // RAW producer valid (always-accept; free-running producers)",
                f"        .ready_out({nm}_ready_out),",
                f"        .data_in({prod}_data_out),",
                f"        .valid_out({nm}_valid_out),",
                f"        .ready_down({rd}),  // consumer-raw accept, NO spatial_run",
                f"        .drain_en(spatial_run_drain_{nm}),  // == consumer valid_in gate; excludes ONLY this bridge's own stall",
                f"        .data_out({nm}_data_out),",
                f"        .stall_out({nm}_stall_out)",
                f"    );",
                "",
            ]
        else:  # scatter
            lines += [
                f"    retile_scatter #(.TILE_W({b['tile_w']}), .N_TILES({b['n_tiles']}), "
                f".IN_W({b['full_beat_w']}), .IN_BEATS({b['full_beats']}), .SPATIAL({b['spatial']})) u_{nm} (",
                f"        .clk(clk), .rst_n(rst_n),",
                f"        .valid_in({prod}_valid_out),  // RAW producer valid (always-accept; free-running producers)",
                f"        .ready_out({nm}_ready_out),",
                f"        .data_in({prod}_data_out),",
                f"        .valid_out({nm}_valid_out),",
                f"        .ready_down({rd}),  // consumer-raw accept, NO spatial_run",
                f"        .drain_en(spatial_run_drain_{nm}),  // == consumer valid_in gate; excludes ONLY this bridge's own stall",
                f"        .data_out({nm}_data_out),",
                f"        .stall_out({nm}_stall_out)",
                f"    );",
                "",
            ]

    # Belt-and-suspenders: OR every bridge stall_out into one wire and feed it
    # into spatial_throttle (patched separately).  When ANY bridge has both
    # ping-pong buffers full it throttles all producers' NEW valid_in.
    stall_terms = " | ".join(f"{b['name']}_stall_out" for b in BRIDGES)
    lines.append(f"    wire any_retile_stall = {stall_terms};")
    lines.append("")
    lines.append(MARK_END)
    lines.append("")
    return "\n".join(lines)


def find_inst(text, mid):
    m = re.search(re.escape(mid) + r"\s+u_" + re.escape(mid) + r"\s*\((?:.|\n)*?\n\s*\);", text)
    if not m:
        raise RuntimeError(f"instance u_{mid} not found")
    return m


# Match the spatial gate (last term of a valid_in / in_valid expression): either
# the global "spatial_run" or any prior "spatial_run_drain_<bridge>" (idempotent
# re-apply).  Must NOT match "spatial_run_drain_..." when we want to *find* a bare
# spatial_run elsewhere, so we anchor on word boundaries.
SPATIAL_GATE_RE = r"spatial_run(?:_drain_[A-Za-z0-9_]+)?"


def set_valid_in(block, new_valid_term, spatial_gate=None):
    """Replace the leading source term of .valid_in(...), keeping the middle
    terms (e.g. '& skip_valid'). The leading term is everything up to the first
    '&'. If spatial_gate is given, the trailing spatial gate term (spatial_run or
    a prior spatial_run_drain_*) is rewritten to '& <spatial_gate>'."""
    def sub(mm):
        inner = mm.group(1)
        parts = inner.split("&")
        parts[0] = " " + new_valid_term + " "
        if spatial_gate is not None:
            # The spatial gate is normally the LAST term of the valid_in expr.
            # IDEMPOTENT re-apply: a BRIDGELESS final-stage hop may already have
            # had its gate STRIPPED by the later UNGATE step (step 6).  In that
            # case there is no spatial term to rewrite and the subsequent
            # strip_spatial_gate would drop it again anyway, so a missing gate is
            # a benign no-op rather than an error.
            if re.search(SPATIAL_GATE_RE, parts[-1]):
                # Rewrite is a no-op when the gate already equals spatial_gate
                # (idempotent re-apply or a deliberately-global passthrough).
                parts[-1] = re.sub(SPATIAL_GATE_RE, spatial_gate, parts[-1], count=1)
            elif spatial_gate.startswith("spatial_run_drain_"):
                # A bridge-fed consumer MUST end up with its per-bridge gate; if it
                # is entirely missing something is wrong (not just an idempotent
                # ungate of a bridgeless hop).
                raise RuntimeError(
                    f"per-bridge spatial gate term not found in valid_in tail: "
                    f"{parts[-1]!r}")
            # else: spatial_gate == 'spatial_run' on an already-ungated bridgeless
            # hop -> leave as-is (UNGATE keeps it ungated).
        return ".valid_in(" + "&".join(parts) + ")"
    new, n = re.subn(r"\.valid_in\(([^\n]*)\)", sub, block, count=1)
    if n != 1:
        raise RuntimeError("valid_in not matched")
    return new


def set_data_in(block, new_data_expr):
    new, n = re.subn(r"\.data_in\([^\n]*\)", f".data_in({new_data_expr})", block, count=1)
    if n != 1:
        raise RuntimeError("data_in not matched")
    return new


# ===========================================================================
# DRAIN-SIDE FIX (2026-06-01): strip the bare global "& spatial_run" gate from
# the FINAL-STAGE BRIDGELESS tiled hops.
# ===========================================================================
# After the input-side per-bridge-except-self gate fixed the INTAKE deadlock,
# the design DRAIN-deadlocked in the TERMINAL tiled chain
#     ... n4_34 -> node_conv_910 -> node_conv_912 -> n4_35 -> u_br_mean(gather)
#         -> node_mean(GAP) -> node_linear(Gemm) -> m_axis_tvalid
# plus the analogous bridgeless hops in the residual blocks.
#
# ROOT CAUSE (lost-beat hazard):  In the final stage the convs and relus are
# FREE-RUNNING producers -- in their emit window they assert valid_out and
# advance their output beat index every cycle OFF A COUNTER, never sampling any
# downstream ready (node_conv_910.v ST_EMIT lines 165-177; n4_35.v sending-branch
# lines 88-113).  Their consumer is a self-throttling module that latches a beat
# only on (valid_in & ready_in) and holds ready_in low while busy.  When such a
# consumer's valid_in is gated by the GLOBAL spatial_run, a transient spatial_run=0
# (caused by ANY OTHER wave-2 bridge momentarily filling both ping-pong buffers,
# any_retile_stall=1) forces valid_in low FOR ONE CYCLE while the producer keeps
# free-running its emission -> the consumer MISSES that beat, which is gone
# forever -> the consumer captures < N beats for that pixel, never completes it,
# never feeds the downstream gather all its tiles -> u_br_mean never fills ->
# valid_out/drain never fire -> node_mean never gets its 49th beat -> node_linear
# never starts -> m_axis_tvalid is permanently 0.  This is EXACTLY the
# free-running-producer-meets-momentary-not-ready hazard the bridge header warns
# about (retile_bridge.v:25-52) -- but these terminal/residual hops are wired
# DIRECTLY tiled-to-tiled with NO bridge to provide always-accept slack.
#
# FIX (the bridge header's own principle):  a free-running producer must meet an
# always-accept consumer.  These bridgeless hops have NO engine dispatch and need
# NO spatial throttle (engine_busy and sched_spatial_stall are 0 in the all-
# spatial final stage; the only skip FIFO -- u_skip_node_add_1038 -- is fed via
# its OWN per-bridge gate spatial_run_drain_br_1038s, NOT these hops; so in the
# final stage spatial_run only ever drops due to any_retile_stall, i.e. the
# spurious bridge-transient dropout that drops the beat).  Removing the global
# "& spatial_run" makes valid_in track the producer's RAW free-running valid_out;
# the consumer's OWN ready_in is the sole accept term and stays high across the
# whole producer burst, so NO beat can ever be dropped.  Byte-exact: same data,
# same order, simply no longer spuriously skipped.  Does NOT touch the INTAKE
# path: PIXEL_IN/node_conv_810 gate (top:893), s_axis_tready (top:480), and every
# bridge's RAW intake + per-bridge-except-self drain gate are all preserved.
#
# ALSO fixes the GAP->Gemm single-pulse handoff: node_mean.valid_out is a 1-cycle
# pulse (node_mean.v:85 default 0, set only in ST_PACK line 108).  AND-ing it with
# the global spatial_run would silently drop the lone GAP result if spatial_run is
# low on that exact cycle.  node_linear self-throttles via its own ready_in, so
# raw node_mean_valid_out is the correct, never-dropped handoff.
#
# Consumers whose valid_in is FED BY A BRIDGE keep their per-bridge
# spatial_run_drain_<br> gate (THE INVARIANT: drain xfer == consumer latch).  We
# strip ONLY the bare global spatial_run from BRIDGELESS final-stage hops.
UNGATE_FINAL_BRIDGELESS = [
    # residual blocks (free-run conv/relu -> self-throttling conv/relu, no bridge)
    "n4_23", "node_conv_880",
    "n4_25", "node_conv_886",
    "n4_27", "node_conv_892", "node_conv_894", "n4_29", "node_conv_898",
    "n4_31", "node_conv_904",
    "n4_33",
    # TERMINAL chain (the proven drain-deadlock locus)
    "node_conv_910", "node_conv_912", "n4_35",
    # GAP -> Gemm single-pulse handoff
    "node_linear",
]


def strip_spatial_gate(text, mid):
    """Remove the trailing global '& spatial_run' (or a prior '& spatial_run_drain_*')
    term from this consumer's .valid_in(...).  Idempotent: if the valid_in has no
    spatial gate term left (already stripped), it is a no-op.  Leaves any
    non-spatial middle terms (e.g. '& skip_valid') untouched -- but the bridgeless
    final-stage hops have none.  Refuses to touch a BRIDGE-fed consumer (whose
    leading term is a br_*_valid_out) so we can never accidentally drop a
    per-bridge drain gate (would break THE INVARIANT)."""
    m = find_inst(text, mid)
    blk = m.group(0)

    def sub(mm):
        inner = mm.group(1)
        parts = [p.strip() for p in inner.split("&")]
        # Safety: never strip a per-bridge gate off a bridge-fed consumer.
        if parts and parts[0].startswith("br_"):
            raise RuntimeError(
                f"refusing to ungate bridge-fed consumer u_{mid} "
                f"(leading term {parts[0]!r}); bridge-fed hops MUST keep their "
                f"per-bridge spatial_run_drain gate")
        # Drop ONLY a trailing pure spatial-gate term (the term that is exactly a
        # spatial_run / spatial_run_drain_* token).  Middle terms are preserved.
        kept = []
        for i, p in enumerate(parts):
            is_pure_spatial = re.fullmatch(SPATIAL_GATE_RE, p) is not None
            if is_pure_spatial:
                continue  # drop this gate term
            kept.append(p)
        if not kept:
            raise RuntimeError(f"u_{mid} valid_in collapsed to empty after ungate")
        return ".valid_in(" + " & ".join(kept) + ")"

    nb, n = re.subn(r"\.valid_in\(([^\n]*)\)", sub, blk, count=1)
    if n != 1:
        raise RuntimeError(f"valid_in not matched in u_{mid}")
    return text[:m.start()] + nb + text[m.end():]


def patch_consumer(text, mid, valid_term, data_expr, spatial_gate):
    m = find_inst(text, mid)
    blk = m.group(0)
    nb = set_data_in(set_valid_in(blk, valid_term, spatial_gate), data_expr)
    return text[:m.start()] + nb + text[m.end():]


def patch_add_main(text, add_id, bridge, skip_data, skip_valid, spatial_gate):
    m = find_inst(text, add_id)
    blk = m.group(0)
    # valid_in: main term -> bridge_valid_out, keep '& skip_valid', re-gate the
    # trailing spatial term to the main bridge's per-bridge spatial_run_drain.
    nb = set_valid_in(blk, f"{bridge}_valid_out", spatial_gate)
    # data_in: {skip_data, bridge_data_out}
    nb = set_data_in(nb, f"{{{skip_data}, {bridge}_data_out}}")
    return text[:m.start()] + nb + text[m.end():]


def patch_skip_fifo(text, add_id, bridge, width, spatial_gate):
    m = re.search(r"u_skip_" + re.escape(add_id) + r"\s*\((?:.|\n)*?\n\s*\);", text)
    if not m:
        raise RuntimeError(f"u_skip_{add_id} not found")
    blk = m.group(0)
    nb = blk
    # in_valid: leading term -> bridge_valid_out; re-gate the spatial term to the
    # bridge's per-bridge spatial_run_drain; keep the '& ..skip_in_ready' tail.
    def vsub(mm):
        inner = mm.group(1)
        parts = inner.split("&")
        parts[0] = " " + f"{bridge}_valid_out" + " "
        # The spatial gate is the MIDDLE term here (tail is skip_in_ready); rewrite
        # the first part that contains a spatial gate token (no-op if already set).
        for i in range(1, len(parts)):
            if re.search(SPATIAL_GATE_RE, parts[i]):
                parts[i] = re.sub(SPATIAL_GATE_RE, spatial_gate, parts[i], count=1)
                break
        else:
            raise RuntimeError(
                f"spatial gate term not found in in_valid of u_skip_{add_id}")
        return ".in_valid(" + "&".join(parts) + ")"
    nb, n1 = re.subn(r"\.in_valid\(([^\n]*)\)", vsub, nb, count=1)
    if n1 != 1:
        raise RuntimeError(f"in_valid not matched in u_skip_{add_id}")
    nb, n2 = re.subn(r"\.in_data\([^\n]*\)", f".in_data({bridge}_data_out[{width-1}:0])", nb, count=1)
    if n2 != 1:
        raise RuntimeError(f"in_data not matched in u_skip_{add_id}")
    return text[:m.start()] + nb + text[m.end():]


def patch_spatial_throttle(text):
    """OR any_retile_stall into spatial_throttle. Idempotent.

    `any_retile_stall` is declared inside the (later) bridge block; a continuous
    `wire x = expr;` net assignment may forward-reference a net, so the ordering
    is legal Verilog. When ANY ping-pong bridge has both buffers full it pulls
    spatial_run low, throttling every producer's NEW valid_in (intake side). The
    bridges' DRAIN sides ignore spatial_run, so the stall always self-clears."""
    if "any_retile_stall" in text and "spatial_throttle = engine_busy | sched_spatial_stall | any_retile_stall" in text:
        return text  # already patched
    pat = r"wire spatial_throttle = engine_busy \| sched_spatial_stall;"
    repl = ("wire spatial_throttle = engine_busy | sched_spatial_stall | any_retile_stall; "
            "// any_retile_stall: OR of every wave-2 bridge stall_out (declared in bridge block)")
    new, n = re.subn(pat, repl, text, count=1)
    if n != 1:
        # maybe already has the any_retile_stall term but reformatted; require it present
        if "any_retile_stall" in text and "spatial_throttle" in text and "any_retile_stall" in \
           re.search(r"wire spatial_throttle = [^\n;]*;", text).group(0):
            return text
        raise RuntimeError("spatial_throttle assignment not found / not patchable")
    return new


def resolve_top():
    """Resolve the target top from --top, then NN2RTL_TOP, then DEFAULT_TOP.
    Default behavior (no arg, no env) is unchanged: the baseline nn2rtl_top.v."""
    ap = argparse.ArgumentParser(description="Apply MobileNetV2 wave-2 retile bridges.")
    ap.add_argument("--top", default=None,
                    help="path to the top .v to patch (default: $NN2RTL_TOP or the baseline "
                         "nn2rtl_top.v). Pass the engine top nn2rtl_top_engine.v to retarget.")
    args = ap.parse_args()
    return args.top or os.environ.get("NN2RTL_TOP") or DEFAULT_TOP


def main():
    top = resolve_top()
    with open(top, "r", encoding="utf-8") as f:
        text = f.read()

    # 1) Insert (or replace) the bridge block, just before the first per-layer
    #    "// ----- spatial module instantiations -----" anchor.
    block = build_bridge_block()
    if MARK_BEGIN in text:
        text = re.sub(re.escape(MARK_BEGIN) + r"(?:.|\n)*?" + re.escape(MARK_END) + r"\n",
                      block, text, count=1)
    else:
        anchor = "    // ----- spatial module instantiations -----\n"
        idx = text.find(anchor)
        if idx < 0:
            raise RuntimeError("spatial-instantiation anchor not found")
        idx += len(anchor)
        text = text[:idx] + block + text[idx:]

    # 2) Re-point plain consumers (re-gating each bridge-fed consumer's spatial
    #    term to that bridge's per-bridge spatial_run_drain).
    for mid, vt, de, sg in REPOINT:
        text = patch_consumer(text, mid, vt, de, sg)

    # 3) Re-point add main paths.
    for add_id, br, sd, sv, sg in ADD_MAIN:
        text = patch_add_main(text, add_id, br, sd, sv, sg)

    # 4) Re-point mean.
    text = patch_consumer(text, *MEAN)

    # 5) Re-source the add_1038 skip FIFO from its GATHER bridge.
    text = patch_skip_fifo(text, SKIP_FIFO_1038["add_id"],
                           SKIP_FIFO_1038["bridge"], SKIP_FIFO_1038["width"],
                           SKIP_FIFO_1038["spatial_gate"])

    # 6) DRAIN-SIDE FIX: strip the bare global "& spatial_run" from the FINAL-
    #    STAGE BRIDGELESS tiled hops (terminal chain + residual hops + GAP->Gemm
    #    single-pulse handoff).  Runs AFTER REPOINT/MEAN so the 5 passthrough
    #    relus (n4_23/25/27/31/33) first get their data/valid re-pointed (with the
    #    placeholder spatial_run gate REPOINT writes) and then have that gate
    #    removed here.  Bridge-fed consumers are NOT in this list -> they keep
    #    their per-bridge spatial_run_drain gate (THE INVARIANT).  See the long
    #    comment above UNGATE_FINAL_BRIDGELESS.
    for mid in UNGATE_FINAL_BRIDGELESS:
        text = strip_spatial_gate(text, mid)

    # 7) OR every bridge stall_out (any_retile_stall) into spatial_throttle.
    text = patch_spatial_throttle(text)

    with open(top, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Wave-2 retile bridges applied to {top}")
    print(f"  bridges instantiated : {len(BRIDGES)}")
    print(f"  consumers re-pointed : {len(REPOINT)} + {len(ADD_MAIN)} adds + mean")
    print(f"  add_1038 skip FIFO re-sourced from GATHER {SKIP_FIFO_1038['bridge']}")
    print(f"  final-stage bridgeless hops ungated : {len(UNGATE_FINAL_BRIDGELESS)} "
          f"(drain-side lost-beat fix; INTAKE gates preserved)")
    print(f"  spatial_throttle now ORs any_retile_stall (deadlock-free drain)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
