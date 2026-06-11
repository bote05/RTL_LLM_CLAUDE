#!/usr/bin/env python3
"""apply_mbv2_final_bundle.py -- MBV2 FINAL-SYNTH BUNDLE (route-forensics fixes).

Targets the routed-WNS classes in the NEW-netlist c8b route report
(output/mobilenet-v2/reports/synth/checkpoints/mbv2_route_postroute_timing_new_c8b.rpt,
routed 86.67 MHz @8ns, worst paths ~98% ROUTE delay):

  PATH CLASS 1 (top-1/2/5, -4.019ns): u_engine_out_node_conv_876/g_tiled.tile_idx_reg
    -> data_out_reg mux. tile_idx (fo=518) selects one 256b slice of the 2048b
    beat_buf in the engine_output_bridge g_tiled branch; the placer stretches the
    select net ~11.5ns across the die. STRUCTURAL FIX: per-32b-slice dont_touch
    shadow replicas of tile_idx (identical update logic, same-cycle value =>
    byte- and cycle-exact by construction), data_out load sliced per replica.
    Applies to all 13 OUT_KIND=1 bridges (conv_876/878/882/884/888/890/894/896/
    900/902/906/908/912).

  PATH CLASS 2 (top-3, -4.010ns): scheduler engine_output_ready broadcast
    (fo=271) -> loader/bridge ready chain -> engine_output_fifo URAM EN (fo=53),
    passing through a g_legacy data_out clock-enable net (fo=1573, conv_836).
    ATTRIBUTE FIX (c232a20 precedent, Verilator-invisible): max_fanout caps on
    - scheduler engine_output_ready (64)
    - engine_output_bridge emit_ready in all 3 generate branches (128)
    - engine_output_fifo load_skid (16)

  PATH CLASS 3 (top-4/6, -4.003ns): engine_output_fifo URAM dout (out_data was
    absorbed into the URAM output register) -> 51 scattered bridge beat_bufs,
    fo=57-63/bit, 0 logic levels, pure route distance. ATTRIBUTE FIX:
    max_fanout=16 on out_data -> synthesis replicates the skid register into
    fabric FF copies the placer can park near bridge clusters. (No RTL register
    can be added without +1 latency, so attribute-only here.)

  PATH CLASS 4 (-3.955ns x ~28): u_shared_engine ag_act_in_ic_byte_idx_d2 ->
    mac_array DSP. OUT OF SCOPE (output/rtl/engine/* is sibling-owned).
    Documented in docs/agent_tasks/MBV2_FINAL_BUNDLE_ANALYSIS.md only.

Also retires the STALE pblock constraint file
output/mobilenet-v2/reports/synth/mbv2_fmax_pblock.xdc: it names cells deleted
from the new netlist (u_node_conv_854/860/866/872/878..908, u_br_ldr28/30/32 --
all engine-dispatched by DW-ENGINE-EXT / DW-QUARTET / FC-ENGINE) and crashed
place_design (EXCEPTION_ACCESS_VIOLATION). The c8b route closes with
--no-pblock, so the file becomes a comment-only tombstone.

All RTL edits are CYCLE-NEUTRAL: shadow registers carry the identical value
every cycle; max_fanout is a synthesis attribute Verilator/iverilog ignore.
Gate: lint 0 errors + 8/8 e2e PASS with e2e_cycles == 1,184,731 EXACTLY.

Idempotent + anchor-asserted; writes .prefinalbundle backups next to each file.
Run from repo root:  python scripts/apply_mbv2_final_bundle.py
"""

import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOP = os.path.join(REPO, "output", "mobilenet-v2", "rtl", "nn2rtl_top_engine.v")
SCHED = os.path.join(REPO, "output", "mobilenet-v2", "rtl", "nn2rtl_scheduler.v")
XDC = os.path.join(REPO, "output", "mobilenet-v2", "reports", "synth",
                   "mbv2_fmax_pblock.xdc")

MARKER = "[FINAL-BUNDLE"


def read(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def write(p, text):
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def backup(p):
    bak = p + ".prefinalbundle"
    if not os.path.exists(bak):
        shutil.copy2(p, bak)
        print(f"  [backup] {os.path.relpath(bak, REPO)}")


def sub_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"ANCHOR FAIL [{label}]: expected exactly 1 occurrence, "
                         f"found {n}. File drifted; aborting (nothing written).")
    print(f"  [edit] {label}")
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------
# Edit set A: nn2rtl_top_engine.v
# ---------------------------------------------------------------------------

def patch_top(text):
    # --- A1. g_tiled: drop the monolithic current_tile mux, cap emit_ready ---
    text = sub_once(
        text,
        "        wire [DATA_W-1:0] current_tile = beat_buf[tile_idx*256 +: 256];\n"
        "        wire emit_ready = active_slot && (!valid_out || ready_out)\n"
        "                        && !drain_complete && buf_valid;\n",
        "        // [FINAL-BUNDLE TIDX-REP 2026-06-11] monolithic current_tile mux\n"
        "        // removed -- data_out is now loaded per 32b slice from dont_touch\n"
        "        // tile_idx shadow replicas (see g_tidx_rep below). max_fanout caps\n"
        "        // the emit_ready clock-enable net (route-report CE-net class).\n"
        "        (* max_fanout = 128 *) wire emit_ready = active_slot && (!valid_out || ready_out)\n"
        "                        && !drain_complete && buf_valid;\n",
        "g_tiled: remove current_tile + cap emit_ready")

    # --- A2. g_tiled: sliced data_out load + tile_idx shadow replicas ---
    text = sub_once(
        text,
        "        // pull_idx/tile_idx are position CONTROL and keep their reset.\n"
        "        always @(posedge clk) begin\n"
        "            if (emit_ready) data_out <= current_tile;\n"
        "            if (fifo_out_ready && fifo_out_valid) beat_buf <= fifo_out_data;\n"
        "        end\n",
        "        // pull_idx/tile_idx are position CONTROL and keep their reset.\n"
        "        always @(posedge clk) begin\n"
        "            if (fifo_out_ready && fifo_out_valid) beat_buf <= fifo_out_data;\n"
        "        end\n"
        "        // [FINAL-BUNDLE TIDX-REP 2026-06-11] route-WNS class fix (top paths\n"
        "        // -4.019/-4.017/-3.975ns, 98.4% ROUTE): tile_idx_reg fo~518 fanning\n"
        "        // into the 256b 8:1 beat_buf output mux stretched ~11.5ns across the\n"
        "        // die. Split data_out into 32b slices, each loaded through its own\n"
        "        // dont_touch SHADOW copy of tile_idx: identical reset + update logic\n"
        "        // (pull-clear wins over emit-increment, exactly like the master), so\n"
        "        // every replica equals tile_idx on every cycle => byte- AND cycle-\n"
        "        // exact by construction. fo per replica <= ~72; the placer can park\n"
        "        // each replica beside its 32 slice FFs. Cost: +(DATA_W/32)x7 FF per\n"
        "        // OUT_KIND=1 bridge (8x7=56 FF for the 13 DATA_W=256 instances).\n"
        "        genvar ts;\n"
        "        for (ts = 0; ts < (DATA_W + 31) / 32; ts = ts + 1) begin : g_tidx_rep\n"
        "            localparam integer LO = ts * 32;\n"
        "            localparam integer SW = (DATA_W - LO < 32) ? (DATA_W - LO) : 32;\n"
        "            (* dont_touch = \"true\" *) reg [6:0] tile_idx_rep;\n"
        "            always @(posedge clk or negedge rst_n) begin\n"
        "                if (!rst_n) tile_idx_rep <= 7'd0;\n"
        "                else begin\n"
        "                    if (emit_ready) begin\n"
        "                        if (last_tile) tile_idx_rep <= 7'd0;\n"
        "                        else           tile_idx_rep <= tile_idx_rep + 7'd1;\n"
        "                    end\n"
        "                    if (fifo_out_ready && fifo_out_valid) tile_idx_rep <= 7'd0;\n"
        "                end\n"
        "            end\n"
        "            always @(posedge clk) begin\n"
        "                if (emit_ready)\n"
        "                    data_out[LO +: SW] <= beat_buf[tile_idx_rep*256 + LO +: SW];\n"
        "            end\n"
        "        end\n",
        "g_tiled: per-slice data_out load + tile_idx replicas")

    # --- A3. g_legacy: cap emit_ready (the conv_836 fo=1573 CE-net class) ---
    text = sub_once(
        text,
        "        wire emit_ready = active_slot && (!valid_out || ready_out)\n"
        "                        && !drain_complete && buf_valid;\n"
        "        wire last_tile  = (tile_idx == (TILES_PER_BEAT[TILE_IDX_W:0] - 1'b1));\n",
        "        // [FINAL-BUNDLE CE-CAP 2026-06-11] emit_ready is the data_out clock\n"
        "        // enable: one LUT drove up to DATA_W+1 FF CE pins (fo=1573 on the\n"
        "        // conv_836 DATA_W=1536 instance, inside the -4.010ns route path).\n"
        "        // max_fanout lets synth replicate the CE LUT per FF cluster.\n"
        "        (* max_fanout = 128 *) wire emit_ready = active_slot && (!valid_out || ready_out)\n"
        "                        && !drain_complete && buf_valid;\n"
        "        wire last_tile  = (tile_idx == (TILES_PER_BEAT[TILE_IDX_W:0] - 1'b1));\n",
        "g_legacy: cap emit_ready CE fanout")

    # --- A4. g_flat: cap emit_ready (DATA_W up to 8000 on node_linear) ---
    text = sub_once(
        text,
        "        wire emit_ready = active_slot && (!valid_out || ready_out)\n"
        "                        && !drain_complete && buf_full;\n",
        "        // [FINAL-BUNDLE CE-CAP 2026-06-11] same CE-net class as g_legacy;\n"
        "        // DATA_W is 3072 (conv_852/854) / 8000 (node_linear) here.\n"
        "        (* max_fanout = 128 *) wire emit_ready = active_slot && (!valid_out || ready_out)\n"
        "                        && !drain_complete && buf_full;\n",
        "g_flat: cap emit_ready CE fanout")

    # --- A5. engine_output_fifo: out_data replication escape from URAM OREG ---
    text = sub_once(
        text,
        "    output reg              out_valid,\n"
        "    output reg  [DATA_W-1:0] out_data,\n",
        "    output reg              out_valid,\n"
        "    // [FINAL-BUNDLE OREG-REP 2026-06-11] out_data was absorbed into the URAM\n"
        "    // output register, so the macro pin drove all 51 bridge beat_bufs\n"
        "    // directly (fo=57-63/bit, 0 logic levels, 10.8ns pure route = the\n"
        "    // -4.003/-3.963ns paths). max_fanout=16 asks synth to replicate the skid\n"
        "    // register into fabric FF copies placeable near bridge clusters. Same\n"
        "    // RTL register either way => cycle count unchanged; attribute-only and\n"
        "    // invisible to the simulator. If synthesis declines, it is a no-op.\n"
        "    (* max_fanout = 16 *) output reg  [DATA_W-1:0] out_data,\n",
        "engine_output_fifo: cap out_data fanout (OREG escape)")

    # --- A6. engine_output_fifo: load_skid -> URAM EN fanout (fo=53) ---
    text = sub_once(
        text,
        "    wire load_skid = !fifo_empty && (!out_valid || (out_valid && out_ready));\n",
        "    // [FINAL-BUNDLE EN-CAP 2026-06-11] load_skid drives the EN of every URAM\n"
        "    // macro of mem (fo=53; the 2.112ns tail segment of the -4.010ns path).\n"
        "    (* max_fanout = 16 *) wire load_skid = !fifo_empty && (!out_valid || (out_valid && out_ready));\n",
        "engine_output_fifo: cap load_skid EN fanout")

    return text


# ---------------------------------------------------------------------------
# Edit set B: nn2rtl_scheduler.v
# ---------------------------------------------------------------------------

def patch_sched(text):
    text = sub_once(
        text,
        "    output reg         engine_output_ready\n",
        "    // [FINAL-BUNDLE RDY-CAP 2026-06-11] broadcast to all 51 bridges'\n"
        "    // start/dispatch_count (fo=271 even after Vivado's own FSM-state\n"
        "    // replica; 1.569ns segment of the -4.010ns route path). max_fanout\n"
        "    // replicates the decode LUT so copies place near bridge clusters.\n"
        "    (* max_fanout = 64 *) output reg engine_output_ready\n",
        "scheduler: cap engine_output_ready broadcast fanout")
    return text


# ---------------------------------------------------------------------------
# Edit set C: pblock tombstone
# ---------------------------------------------------------------------------

XDC_TOMBSTONE = """\
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
"""


def main():
    if not (os.path.exists(TOP) and os.path.exists(SCHED) and os.path.exists(XDC)):
        raise SystemExit("missing target file(s); run from the repo root checkout")

    top_text = read(TOP)
    sched_text = read(SCHED)
    xdc_text = read(XDC)

    already = (MARKER in top_text)
    if already:
        # Idempotency: verify the full marker set, then no-op.
        need = {"TIDX-REP": 2, "CE-CAP": 2, "OREG-REP": 1, "EN-CAP": 1}
        for k, n in need.items():
            have = top_text.count(f"[FINAL-BUNDLE {k}")
            if have != n:
                raise SystemExit(f"PARTIAL APPLY detected in top ({k}: {have}/{n}); "
                                 f"restore .prefinalbundle and rerun.")
        if MARKER not in sched_text or MARKER not in xdc_text:
            raise SystemExit("PARTIAL APPLY: top patched but scheduler/xdc not; "
                             "restore .prefinalbundle and rerun.")
        print("[apply_mbv2_final_bundle] already applied -- no-op.")
        return

    # Compute all patched texts FIRST (any anchor failure aborts before writes).
    new_top = patch_top(top_text)
    new_sched = patch_sched(sched_text)

    backup(TOP)
    backup(SCHED)
    backup(XDC)
    write(TOP, new_top)
    write(SCHED, new_sched)
    write(XDC, XDC_TOMBSTONE)
    print("[apply_mbv2_final_bundle] APPLIED: 6 edits in nn2rtl_top_engine.v, "
          "1 in nn2rtl_scheduler.v, pblock xdc retired.")
    print("Gates: verilator lint (0 errors) + bash scripts/run_mbv2_e2e_parallel.sh "
          "(8/8 PASS, e2e_cycles == 1,184,731 EXACTLY).")


if __name__ == "__main__":
    sys.exit(main())
