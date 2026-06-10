#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_k1_mbv2_ext.py -- K1-MBV2 EXTENSION: async-reset -> no-reset recode of
DATAPATH-ONLY registers in the MobileNetV2-OWN RTL (FDCE -> FDRE).

This extends scripts/apply_k1_fdce_recode.py (ResNet K1, commit be16f61) to the
file set that scripts/run_mbv2_synth.ts collectSources() ships MINUS the 5
already-recoded SHARED files (line_buf_window, conv_datapath_mp_k,
shared_engine_skeleton, mac_array, requant_pipeline -- all already carry
[K1-FDCE] markers and are inherited by MBV2).

METHOD (identical to ResNet K1 -- see docs/agent_tasks/K1_FDCE_RECODE_ANALYSIS.md
and docs/agent_tasks/K1_MBV2_EXT_ANALYSIS.md):
  * only registers provably WRITE-BEFORE-READ per frame/pixel/OC-pass, or whose
    value is only sampled under a reset-kept valid/control bit, are moved out of
    the async-reset block into a separate sync-only "Block A".
  * ALL FSM/valid/ready/beat-counter/pointer/phase state KEEPS its async reset.
  * NBA last-write-wins ordering is preserved when splitting (acc clears placed
    AFTER the accumulate; drain wr_data writes keep their override order).
  * e2e gate runs Verilator --x-initial 0 = FPGA power-on zeros: a no-reset reg
    powers on to 0 (Vivado INIT default 0) -- exactly the old reset value -- and
    no Block-A write can fire during the single t=0 reset window because every
    write-enable traces to still-reset-held control. Machine state at reset
    release is bit-identical -> byte-exact AND cycle-exact by construction.

MECHANICS (same hardening as the ResNet applier):
  * anchor-asserted: every edit is an exact-string match that must occur EXACTLY
    ONCE at its point in the per-file edit sequence. Heterogeneous classes (DW
    convs, n4 relus, adds) derive their anchors FROM THE FILE ITSELF via strict
    regexes, then still assert count==1 -- any drift aborts the whole run BEFORE
    any file is written (two-phase validate-then-commit).
  * idempotent: files containing the [K1-MBV2] marker are skipped.
  * encoding (utf-8 / cp1252) + EOL (CRLF / LF) auto-detected per file and
    preserved on write.
  * backups: <file>.prek1m written once.
  * --dry-run: full validation + per-file register listing, no writes.

SCOPE (MBV2-own, per docs/agent_tasks/K1_MBV2_EXT_ANALYSIS.md):
  C1  17 depthwise convs (inline datapaths): prod_q/acc/biased/scaled/v_tmp +
      staged output pixel (dp_data_out / pix_out / out_pix)
      + A-family (812..872) output skid out_data
      + B/C-family (878..908, NATIVE_TILED=1) tile_acc gather + out_lat drain
  C2  22 single-beat n4 relus: g_legacy data_out_r + g_bp out_data
  C3  13 multi-beat n4 relus (n4_23..n4_35): beat_buf (ResNet P8 pattern)
  C4  10 residual adds: output skid out_data + input_buf + dp_data_out
  C5  node_conv_810 stem wrapper: output skid out_data
  C6  node_linear: output skid out_data (datapath block is ALREADY sync-only)
  C7  node_mean: emit_data (the BRAM-critical acc/scaled/rounded sync-only
      block is NOT touched)
  C8  output_serializer: buf_data + data_out
  C9  nn2rtl_top_engine.v helpers: skip_fifo.out_data_r, engine_output_fifo.
      out_data, stream_to_act_bram_bridge.{wr_data,skid_data,beat_buf} (all 3
      generate branches), engine_output_bridge.{beat_buf,data_out,gather_buf}
      (all 3 OUT_KINDs)
  C10 rtl_library/retile_bridge.v: retile_gather.buf0/buf1
SKIPPED (see analysis doc): all control state; add MAC pipes (lhs/rhs/sum_term);
multi-beat n4 data_out (multi-site, valid_out-interleaved -- ResNet precedent);
DW legacy/bp emitter branches not elaborated by the shipped top (lo_latch/
lo_hold/em_buf/bp_hi); retile_scatter (not instantiated in the engine top);
uram_weight_bank/act_unified_mem/bias_mem (already sync-only); coord_scheduler/
nn2rtl_scheduler/address_generator/config_register_block (control);
bram_to_stream_bridge (not instantiated); nn2rtl_top.v (excluded from synth).

USAGE:
  python scripts/apply_k1_mbv2_ext.py --dry-run
  python scripts/apply_k1_mbv2_ext.py
  python scripts/apply_k1_mbv2_ext.py --repo-root <path>
Rollback: restore *.prek1m (every patched file has one).
"""

import argparse
import re
import sys
from pathlib import Path

MARKER = "[K1-MBV2]"

# ---------------------------------------------------------------------------
# Edit primitives (same two-phase contract as apply_k1_fdce_recode.py)
# ---------------------------------------------------------------------------

class Edit:
    def __init__(self, desc, old, new, count=1):
        self.desc = desc
        self.old = old
        self.new = new
        self.count = count  # exact required number of occurrences (replace all)

class FilePatch:
    def __init__(self, relpath, regs, ff_estimate, edits):
        self.relpath = relpath
        self.regs = regs
        self.ff_estimate = ff_estimate
        self.edits = edits

class Drift(Exception):
    pass

def _find1(text, pattern, what, relpath, flags=0):
    """Exactly-one regex match or Drift."""
    ms = list(re.finditer(pattern, text, flags))
    if len(ms) != 1:
        raise Drift("%s: pattern for '%s' matched %d times (need exactly 1)"
                    % (relpath, what, len(ms)))
    return ms[0]

def _grab_int(text, pattern, what, relpath):
    m = _find1(text, pattern, what, relpath)
    return int(m.group(1))

# ---------------------------------------------------------------------------
# C1: the 17 depthwise convs (inline fork of conv_datapath -- ResNet P2 analog)
# ---------------------------------------------------------------------------

_DW_A = ["node_conv_812", "node_conv_818", "node_conv_824", "node_conv_830",
         "node_conv_836", "node_conv_842", "node_conv_848", "node_conv_854",
         "node_conv_860", "node_conv_866", "node_conv_872"]
_DW_BC = ["node_conv_878", "node_conv_884", "node_conv_890",   # B (pix_out)
          "node_conv_896", "node_conv_902", "node_conv_908"]   # C (out_pix)

_DW_RESET_PACK = """            v_tmp            <= {SCALED_W{1'b0}};
            for (i = 0; i < MP_K; i = i + 1)
                prod_q[i] <= {PROD_W{1'b0}};
            for (i = 0; i < MP; i = i + 1) begin
                acc[i]    <= {ACC_W{1'b0}};
                biased[i] <= {BIASED_W{1'b0}};
                scaled[i] <= {SCALED_W{1'b0}};
            end
"""

_DW_S2 = """            // Stage 2: registered parallel multiplies (one DSP per tap).
            for (i = 0; i < MP_K; i = i + 1)
                prod_q[i] <= $signed(weight_q[i]) * $signed(tap_q[i]);
"""

def _dw_patch(name, text, relpath):
    """Build the per-file Edit list for one depthwise conv by extracting the
    file's own FSM datapath bodies (anchors derived from the file, then
    asserted exactly-once like every other edit)."""
    edits = []
    C  = _grab_int(text, r"localparam integer C\s+=\s+(\d+);", "C", relpath)
    MP = _grab_int(text, r"localparam integer MP\s+=\s+(\d+);", "MP", relpath)

    # --- identify the staged output register (dp_data_out / pix_out / out_pix)
    m = _find1(text, r"\n(\s+)(dp_data_out|pix_out|out_pix)\s+<= (\{[A-Za-z_0-9]+\{1'b0\}\}|\d+'d0);",
               "output-pixel reset line", relpath)
    outreg = m.group(2)
    edits.append(Edit("reset: drop %s" % outreg, m.group(0), ""))

    # --- reset clause: v_tmp/prod_q/acc/biased/scaled (identical in all 17)
    if text.count(_DW_RESET_PACK) != 1:
        raise Drift("%s: DW reset-pack anchor drift" % relpath)
    edits.append(Edit("reset: drop v_tmp/prod_q/acc/biased/scaled", _DW_RESET_PACK, ""))

    # --- Stage 2 (prod_q) -- exact in all 17
    if text.count(_DW_S2) != 1:
        raise Drift("%s: DW stage-2 anchor drift" % relpath)
    edits.append(Edit("FSM: remove stage-2 prod_q (moved to Block A)", _DW_S2, ""))
    s2_loop = ("            for (i = 0; i < MP_K; i = i + 1)\n"
               "                prod_q[i] <= $signed(weight_q[i]) * $signed(tap_q[i]);\n")

    # --- Stage 3 accumulate (C-slice width varies)
    m = _find1(text,
        r"            // Stage 3: accumulator add \(gated by lane validity\)\n"
        r"(            if \(mac_valid_q2 && mac_global_oc_q2 < C\[[^\]]+\]\) begin\n"
        r"                acc\[mac_lane_q2\] <= acc\[mac_lane_q2\] \+ \$signed\(sum_comb\);\n"
        r"            end\n)\n", "stage-3 accumulate", relpath)
    edits.append(Edit("FSM: remove stage-3 accumulate (moved to Block A)", m.group(0), ""))
    s3_block = m.group(1)

    # --- ST_IDLE acc clear
    idle_old = ("                        mac_done_issuing <= 1'b0;\n"
                "                        for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1)\n"
                "                            acc[lane_i] <= {ACC_W{1'b0}};\n"
                "                    end\n")
    if text.count(idle_old) != 1:
        raise Drift("%s: ST_IDLE acc-clear anchor drift" % relpath)
    edits.append(Edit("ST_IDLE: remove acc clear (moved to Block A)", idle_old,
                      "                        mac_done_issuing <= 1'b0;\n"
                      "                    end\n"))

    # --- ST_BIAS body
    m = _find1(text,
        r"                ST_BIAS: begin\n"
        r"(                    for \(lane_i = 0; lane_i < MP; lane_i = lane_i \+ 1\) begin\n"
        r"                        bias_oc = oc_group \* MP \+ lane_i;\n"
        r"                        if \(bias_oc < C\)\n"
        r"                            biased\[lane_i\] <= \$signed\(acc\[lane_i\]\) \+ \$signed\(biases\[bias_oc\]\);\n"
        r"                        else\n"
        r"                            biased\[lane_i\] <= \{BIASED_W\{1'b0\}\};\n"
        r"                    end\n)"
        r"                    state <= ST_SCALE;\n", "ST_BIAS body", relpath)
    bias_loop = m.group(1)
    edits.append(Edit("ST_BIAS: remove biased writes (moved to Block A)", m.group(0),
                      "                ST_BIAS: begin\n"
                      "                    // [K1-MBV2] biased[] writes moved to Block A (sync-only).\n"
                      "                    state <= ST_SCALE;\n"))

    # --- ST_SCALE body
    m = _find1(text,
        r"                ST_SCALE: begin\n"
        r"(                    for \(lane_i = 0; lane_i < MP; lane_i = lane_i \+ 1\) begin\n"
        r"                        sc_oc = oc_group \* MP \+ lane_i;\n"
        r"                        if \(sc_oc < C\)\n"
        r"                            scaled\[lane_i\] <= \$signed\(biased\[lane_i\]\) \* \$signed\(scale_rom\[sc_oc\]\[15:0\]\);\n"
        r"                        else\n"
        r"                            scaled\[lane_i\] <= \{SCALED_W\{1'b0\}\};\n"
        r"                    end\n)"
        r"                    state <= ST_OUTPUT;\n", "ST_SCALE body", relpath)
    scale_loop = m.group(1)
    edits.append(Edit("ST_SCALE: remove scaled writes (moved to Block A)", m.group(0),
                      "                ST_SCALE: begin\n"
                      "                    // [K1-MBV2] scaled[] writes moved to Block A (sync-only).\n"
                      "                    state <= ST_OUTPUT;\n"))

    # --- ST_OUTPUT body: the requant/pack loop (writes out_shift/out_round/
    #     v_tmp temps + the output pixel) followed by the oc_group control.
    m = _find1(text,
        r"                ST_OUTPUT: begin\n"
        r"(                    for \(lane_i = 0; lane_i < MP; lane_i = lane_i \+ 1\) begin\n"
        r"(?:.*\n)*?"
        r"                    end\n)\n"
        r"                    if \(oc_group == ", "ST_OUTPUT pack loop", relpath)
    out_loop = m.group(1)
    if outreg not in out_loop or "v_tmp = (scaled[lane_i]" not in out_loop:
        raise Drift("%s: ST_OUTPUT loop does not look like the requant/pack loop" % relpath)
    if re.search(r"\b(state|oc_group|lane_counter|mac_valid)\s*<=", out_loop):
        raise Drift("%s: ST_OUTPUT loop unexpectedly writes control" % relpath)
    edits.append(Edit("ST_OUTPUT: remove pack loop (moved to Block A)",
                      "                ST_OUTPUT: begin\n" + out_loop + "\n",
                      "                ST_OUTPUT: begin\n"
                      "                    // [K1-MBV2] %s[]/v_tmp writes moved to Block A (sync-only).\n" % outreg))

    # --- ST_OUTPUT oc-advance acc clear (lane_counter/increment widths vary)
    m = _find1(text,
        r"(                        oc_group     <= oc_group \+ (?:1'b1|\d+'d1);\n"
        r"                        lane_counter <= \d+'d0;\n)"
        r"                        for \(lane_i = 0; lane_i < MP; lane_i = lane_i \+ 1\)\n"
        r"                            acc\[lane_i\] <= \{ACC_W\{1'b0\}\};\n",
        "ST_OUTPUT oc-advance acc clear", relpath)
    edits.append(Edit("ST_OUTPUT: remove oc-advance acc clear (moved to Block A)",
                      m.group(0), m.group(1)))

    # --- assemble Block A and insert it before the main FSM block
    block_a = (
        "    // [K1-MBV2] Block A: DATAPATH registers (sync-only, no reset) -- same\n"
        "    // method as ResNet K1 P2 (apply_k1_fdce_recode.py). prod_q is rewritten\n"
        "    // every cycle from the (no-reset) weight_q/tap_q stage and only reaches\n"
        "    // acc under mac_valid_q2 (reset-kept); acc is sync-cleared on ST_IDLE&\n"
        "    // start_mac / ST_OUTPUT oc-advance BEFORE the first gated accumulate of\n"
        "    // every pass; biased/scaled/%s follow strict write(STn)->read(STn+1)\n" % outreg
        + "    // ordering and %s is only consumed under reset-kept valid/busy\n" % outreg
        + "    // control. acc clears are placed LAST (NBA last-write-wins parity with\n"
        "    // the original single block). i/lane_i/bias_oc/sc_oc/out_oc/out_shift/\n"
        "    // out_round/v_tmp are referenced ONLY by this block after the move.\n"
        "    always @(posedge clk) begin\n"
        + s2_loop
        + s3_block
        + "            if (state == ST_BIAS) begin\n"
        + bias_loop
        + "            end\n"
        + "            if (state == ST_SCALE) begin\n"
        + scale_loop
        + "            end\n"
        + "            if (state == ST_OUTPUT) begin\n"
        + out_loop
        + "            end\n"
        + "            // Accumulator clears LAST: textual-order parity with the\n"
        + "            // original single block (clears overrode the accumulate).\n"
        + "            if (state == ST_IDLE && start_mac) begin\n"
        + "                for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1)\n"
        + "                    acc[lane_i] <= {ACC_W{1'b0}};\n"
        + "            end\n"
        + "            if (state == ST_OUTPUT && oc_group != OC_PASSES - 1) begin\n"
        + "                for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1)\n"
        + "                    acc[lane_i] <= {ACC_W{1'b0}};\n"
        + "            end\n"
        + "    end\n\n")
    fsm_anchor = ("    always @(posedge clk or negedge rst_n) begin\n"
                  "        if (!rst_n) begin\n"
                  "            state            <= ST_IDLE;\n")
    if text.count(fsm_anchor) != 1:
        raise Drift("%s: main FSM insertion anchor drift" % relpath)
    edits.append(Edit("insert Block A before the datapath FSM", fsm_anchor,
                      block_a + fsm_anchor))

    regs = ["prod_q[0:MP_K-1]", "acc[0:MP-1]", "biased[0:MP-1]", "scaled[0:MP-1]",
            "v_tmp", "%s[%d]" % (outreg, C * 8)]
    ff = 9 * 16 + MP * (24 + 34 + 50) + 50 + C * 8

    # --- family extras -------------------------------------------------------
    if name in _DW_A:
        sk_ff, sk_edits = _skid_edits(text, relpath, valid_name="dp_valid_out",
                                      data_src="dp_data_out")
        edits += sk_edits
        regs.append("out_data[%d] (output skid)" % sk_ff)
        ff += sk_ff
    else:
        # B/C: NATIVE_TILED=1 in the shipped top -> tile_acc gather + out_lat drain.
        # tile_acc: composite edit = drop reset line + insert in-branch Block A.
        m = _find1(text,
            r"(        always @\(posedge clk or negedge rst_n\) begin\n"
            r"            if \(!rst_n\) begin\n)"
            r"                tile_acc <= \{PIX_W\{1'b0\}\};\n", "tile_acc reset", relpath)
        blk = ("        // [K1-MBV2] tile_acc is gather DATA: every consumed slice is\n"
               "        // rewritten during the pixel's N_TILES-tile gather before the\n"
               "        // last-tile core_valid_in pulse; writes are gated by accept_tile\n"
               "        // (valid_in_t & sched_ready_in, both reset-held). Sync-only -> FDRE.\n"
               "        always @(posedge clk) begin\n"
               "            if (accept_tile) tile_acc[in_tile*TILE_W +: TILE_W] <= data_in_t;\n"
               "        end\n")
        edits.append(Edit("g_in_native: tile_acc reset dropped + Block A",
                          m.group(0), blk + m.group(1)))
        m = _find1(text,
            r"(                if \(accept_tile\) begin\n)"
            r"(?:                    //[^\n]*\n)?"
            r"                    tile_acc\[in_tile\*TILE_W \+: TILE_W\] <= data_in_t;\n",
            "tile_acc write", relpath)
        edits.append(Edit("g_in_native: remove tile_acc write from control block",
                          m.group(0),
                          "                if (accept_tile) begin\n"
                          "                    // ([K1-MBV2] tile_acc data write moved to Block A above.)\n"))

        # out_lat drain (pix_out_ready latches outreg)
        m = _find1(text,
            r"(        always @\(posedge clk or negedge rst_n\) begin\n"
            r"            if \(!rst_n\) begin\n)"
            r"                out_lat  <= \{PIX_W\{1'b0\}\};\n", "out_lat reset", relpath)
        blk = ("        // [K1-MBV2] out_lat is drain DATA: latched whole-pixel under\n"
               "        // pix_out_ready (reset-kept pulse; skid_block guarantees !out_busy)\n"
               "        // and consumed (data_out_t) only while out_busy (reset-kept).\n"
               "        always @(posedge clk) begin\n"
               "            if (pix_out_ready) out_lat <= %s;\n" % outreg
               + "        end\n")
        edits.append(Edit("g_emit_native: out_lat reset dropped + Block A",
                          m.group(0), blk + m.group(1)))
        ol_old = ("                if (pix_out_ready) begin\n"
                  "                    // Capture the full pixel. (skid_block guarantees !out_busy here.)\n"
                  "                    out_lat  <= %s;\n" % outreg)
        if text.count(ol_old) != 1:
            raise Drift("%s: out_lat write anchor drift" % relpath)
        edits.append(Edit("g_emit_native: remove out_lat write from control block", ol_old,
                          "                if (pix_out_ready) begin\n"
                          "                    // ([K1-MBV2] out_lat data write moved to Block A above;\n"
                          "                    // skid_block guarantees !out_busy here.)\n"))
        regs += ["tile_acc[%d]" % (C * 8), "out_lat[%d]" % (C * 8)]
        ff += 2 * C * 8

    return FilePatch("output/mobilenet-v2/rtl/%s.v" % name, regs, ff, edits)

# ---------------------------------------------------------------------------
# Generic 1-deep output skid (dp_valid_out -> out_data), used by DW-A, adds,
# node_linear, node_conv_810. out_data is consumed only under out_full
# (reset-kept); written only under dp_valid_out (reset-kept).
# ---------------------------------------------------------------------------

def _skid_edits(text, relpath, valid_name, data_src):
    edits = []
    m = _find1(text, r"\n(\s+)out_data <= (\d+)'d0;", "skid out_data reset", relpath)
    width = int(m.group(2))
    edits.append(Edit("skid: drop out_data reset", m.group(0), ""))
    old = ("            if (%s) begin\n" % valid_name
           + "                out_data <= %s;\n" % data_src
           + "                out_full <= 1'b1;\n"
           + "            end\n"
           + "        end\n"
           + "    end\n")
    if text.count(old) != 1:
        raise Drift("%s: skid capture anchor drift" % relpath)
    new = ("            if (%s) begin\n" % valid_name
           + "                out_full <= 1'b1;\n"
           + "            end\n"
           + "        end\n"
           + "    end\n"
           + "    // [K1-MBV2] out_data is skid DATA: sampled downstream only under\n"
           + "    // out_full (reset-kept); written only under %s (reset-kept).\n" % valid_name
           + "    always @(posedge clk) begin\n"
           + "        if (%s) out_data <= %s;\n" % (valid_name, data_src)
           + "    end\n")
    edits.append(Edit("skid: out_data -> sync-only block", old, new))
    return width, edits

# ---------------------------------------------------------------------------
# C2: single-beat n4 relus (n4, n4_2 .. n4_22): g_legacy data_out_r + g_bp out_data
# ---------------------------------------------------------------------------

_N4_SINGLE = ["n4"] + ["n4_%d" % i for i in range(2, 23)]

def _n4_single_patch(name, text, relpath):
    edits = []
    m = _find1(text, r"\n(\s+)data_out_r\s+<= (\d+)'d0;", "g_legacy data_out_r reset", relpath)
    width = int(m.group(2))
    edits.append(Edit("g_legacy: drop data_out_r reset", m.group(0), ""))

    m = _find1(text,
        r"[ \t]+if \(valid_in\) begin\n[ \t]+data_out_r <= requant_comb;\n[ \t]+end\n",
        "g_legacy data_out_r write", relpath)
    edits.append(Edit("g_legacy: remove data_out_r write (moved to sync-only block)",
                      m.group(0), ""))
    m = _find1(text,
        r"        always @\(posedge clk or negedge rst_n\) begin\n"
        r"            if \(!rst_n\) begin\n"
        r"                valid_out_r <= 1'b0;\n", "g_legacy insertion anchor", relpath)
    blk = ("        // [K1-MBV2] data_out_r is DATAPATH: consumed downstream only under\n"
           "        // valid_out_r (reset-kept); written under valid_in (upstream valid\n"
           "        // chain is reset-held at t=0). Sync-only write -> FDRE.\n"
           "        always @(posedge clk) begin\n"
           "            if (valid_in) data_out_r <= requant_comb;\n"
           "        end\n")
    edits.append(Edit("g_legacy: insert sync-only data_out_r block", m.group(0),
                      blk + m.group(0)))

    m = _find1(text, r"\n(\s+)out_data <= (\d+)'d0;", "g_bp out_data reset", relpath)
    edits.append(Edit("g_bp: drop out_data reset", m.group(0), ""))
    m = _find1(text,
        r"([ \t]+if \(accept && valid_in\) begin\n)[ \t]+out_data <= requant_comb;\n"
        r"([ \t]+out_full <= 1'b1;\n[ \t]+end\n)", "g_bp out_data write", relpath)
    edits.append(Edit("g_bp: remove out_data write (moved to sync-only block)",
                      m.group(0), m.group(1) + m.group(2)))
    m = _find1(text,
        r"        always @\(posedge clk or negedge rst_n\) begin\n"
        r"            if \(!rst_n\) begin\n"
        r"                out_full <= 1'b0;\n", "g_bp insertion anchor", relpath)
    blk = ("        // [K1-MBV2] out_data is skid DATA: consumed only under out_full\n"
           "        // (reset-kept); written under accept && valid_in (control). -> FDRE.\n"
           "        always @(posedge clk) begin\n"
           "            if (accept && valid_in) out_data <= requant_comb;\n"
           "        end\n")
    edits.append(Edit("g_bp: insert sync-only out_data block", m.group(0),
                      blk + m.group(0)))
    return FilePatch("output/mobilenet-v2/rtl/%s.v" % name,
                     ["data_out_r[%d] (g_legacy)" % width, "out_data[%d] (g_bp)" % width],
                     width,  # elaborated config (EB=1) keeps exactly one branch
                     edits)

# ---------------------------------------------------------------------------
# C3: multi-beat n4 relus (n4_23 .. n4_35): beat_buf -- ResNet P8 pattern.
# Both generate branches carry the identical gather; only one elaborates.
# ---------------------------------------------------------------------------

_N4_MULTI = ["n4_%d" % i for i in range(23, 36)]

def _n4_multi_patch(name, text, relpath):
    edits = []
    bpp = _grab_int(text, r"localparam integer BEATS_PER_PIXEL = (\d+);",
                    "BEATS_PER_PIXEL", relpath)
    rst_loop = ("            for (i = 0; i < BEATS_PER_PIXEL; i = i + 1)\n"
                "                beat_buf[i] <= {BEAT_WIDTH_BITS{1'b0}};\n")
    if text.count(rst_loop) != 2:
        raise Drift("%s: expected the beat_buf reset loop in BOTH generate branches"
                    % relpath)
    edits.append(Edit("both branches: drop beat_buf reset loop", rst_loop, "", count=2))
    wr = ("                if (valid_in && ready_in) begin\n"
          "                    beat_buf[in_beat_count] <= data_in;\n")
    if text.count(wr) != 2:
        raise Drift("%s: expected the beat_buf gather write in BOTH branches" % relpath)
    wr_new = "                if (valid_in && ready_in) begin\n"
    edits.append(Edit("both branches: remove beat_buf write (moved to module Block A)",
                      wr, wr_new, count=2))
    gen_anchor = "    generate\n    if (ENABLE_BACKPRESSURE == 0) begin : g_legacy\n"
    if text.count(gen_anchor) != 1:
        raise Drift("%s: generate anchor drift" % relpath)
    blk = ("    // [K1-MBV2] sync-only memory write -- no reset clause (ResNet K1 P8 /\n"
           "    // node_relu.v precedent): beat_buf is gather DATA, fully rewritten each\n"
           "    // pixel before the sending phase reads it; the guard replicates the\n"
           "    // original nested condition (identical in both generate branches; only\n"
           "    // one elaborates). Also unblocks LUTRAM inference.\n"
           "    always @(posedge clk) begin\n"
           "        if (!sending && valid_in && ready_in) begin\n"
           "            beat_buf[in_beat_count] <= data_in;\n"
           "        end\n"
           "    end\n\n")
    edits.append(Edit("insert module-level beat_buf Block A", gen_anchor, blk + gen_anchor))
    return FilePatch("output/mobilenet-v2/rtl/%s.v" % name,
                     ["beat_buf[0:%d] (256b words)" % (bpp - 1)],
                     bpp * 256, edits)

# ---------------------------------------------------------------------------
# C4: residual adds (10 files, 3 template shapes)
# ---------------------------------------------------------------------------

_ADDS = ["node_add_198", "node_add_336", "node_add_408", "node_add_546",
         "node_add_618", "node_add_690", "node_add_828", "node_add_900",
         "node_add_1038", "node_add_1110"]

def _add_patch(name, text, relpath):
    edits = []
    regs = []
    ff = 0

    # ---- output skid (uniform) ----
    sk_ff, sk_edits = _skid_edits(text, relpath, valid_name="dp_valid_out",
                                  data_src="dp_data_out")
    edits += sk_edits
    regs.append("out_data[%d] (output skid)" % sk_ff)
    ff += sk_ff

    # ---- state label names ----
    idle = "S_IDLE" if re.search(r"\bS_IDLE\b", text) else "ST_IDLE"

    # ---- reset lines: dp_data_out + input_buf ----
    m = _find1(text, r"\n(\s+)dp_data_out\s+<= (\{\w+\{1'b0\}\}|\d+'d0);",
               "dp_data_out reset", relpath)
    edits.append(Edit("reset: drop dp_data_out", m.group(0), ""))
    m = _find1(text, r"\n(\s+)input_buf\s+<= (\{\w+\{1'b0\}\}|\d+'d0);",
               "input_buf reset", relpath)
    edits.append(Edit("reset: drop input_buf", m.group(0), ""))
    # widths (FF accounting only): from the module ports (always numeric)
    m = _find1(text, r"input\s+wire\s+\[(\d+):0\]\s+data_in", "data_in port", relpath)
    in_w = int(m.group(1)) + 1
    m = _find1(text, r"output\s+wire\s+\[(\d+):0\]\s+data_out", "data_out port", relpath)
    out_w = int(m.group(1)) + 1

    # ---- input_buf write site: 3 known guard shapes ----
    m_flat = re.search(
        r"([ \t]+)if \(state == %s && valid_in && !skid_block\) begin\n"
        r"([ \t]+)input_buf    <= data_in;\n" % idle, text)
    m_case = re.search(
        r"([ \t]+)if \((valid_in && (?:ready_in && )?!skid_block)\) begin\n"
        r"([ \t]+)input_buf <= data_in;\n", text)
    if m_flat:
        guard = "state == %s && valid_in && !skid_block" % idle
        edits.append(Edit("remove input_buf gather write (moved to Block A)",
                          m_flat.group(0),
                          m_flat.group(1) + "if (state == %s && valid_in && !skid_block) begin\n" % idle))
    elif m_case:
        guard = "state == %s && %s" % (idle, m_case.group(2))
        edits.append(Edit("remove input_buf gather write (moved to Block A)",
                          m_case.group(0),
                          m_case.group(1) + "if (%s) begin\n" % m_case.group(2)))
    else:
        raise Drift("%s: input_buf write site not recognized" % relpath)
    if text.count("input_buf    <= data_in;") + text.count("input_buf <= data_in;") != 1:
        raise Drift("%s: expected exactly one input_buf write site" % relpath)

    # ---- dp_data_out write site(s): 1-line or 3-arm cascade, both inside
    #      `if (stage2_valid) begin` ----
    casc = re.search(
        r"([ \t]+)if \(stage2_valid\) begin\n"
        r"((?:[ \t]+//[^\n]*\n)*(?:[ \t]+if [^\n]*\n)?(?:[ \t]+//[^\n]*\n)*"
        r"[ \t]+dp_data_out\[stage2_idx\*8 \+: 8\] <= [^\n]+;\n"
        r"(?:[ \t]+else if[^\n]*\n[ \t]+dp_data_out\[stage2_idx\*8 \+: 8\] <= [^\n]+;\n"
        r"[ \t]+else\n[ \t]+dp_data_out\[stage2_idx\*8 \+: 8\] <= [^\n]+;\n)?)", text)
    if not casc:
        raise Drift("%s: dp_data_out write site not recognized" % relpath)
    dp_body = casc.group(2)
    n_writes = dp_body.count("dp_data_out[")
    if n_writes != text.count("dp_data_out["):
        raise Drift("%s: dp_data_out written outside the stage2_valid site" % relpath)
    edits.append(Edit("remove dp_data_out stage-3 write (moved to Block A)",
                      casc.group(0), casc.group(1) + "if (stage2_valid) begin\n"))
    dp_guard = "stage2_valid"  # stage2_valid only pulses during the RUN pipe; in the
    # case-shaped adds the enclosing case arm adds state==RUN, but stage2_valid is
    # 0 in every other state (set only in RUN, cleared otherwise) -> equivalent.
    # To stay strictly conservative we replicate the state term where it exists:
    if re.search(r"case \(state\)", text):
        run = "S_RUN" if idle == "S_IDLE" else "ST_RUN"
        dp_guard = "state == %s && stage2_valid" % run

    # ---- Block A insertion before the FSM block ----
    m = _find1(text,
        r"    always @\(posedge clk or negedge rst_n\) begin\n"
        r"        if \(!rst_n\) begin\n"
        r"            state {1,}<= (?:ST|S)_IDLE;\n", "add FSM insertion anchor", relpath)
    blk = ("    // [K1-MBV2] Block A: array/data writes (sync-only) -- node_add_1\n"
           "    // precedent (ResNet K1 P9/P10 analog). input_buf is fully rewritten on\n"
           "    // the accept edge before the RUN pipe reads it; every consumed\n"
           "    // dp_data_out byte is written by the 3-stage pipe (stage2_valid covers\n"
           "    // ch 0..OC-1) before dp_valid_out pulses; both guards replicate the\n"
           "    // original conditions on reset-kept control. lhs/rhs/sum MAC pipes and\n"
           "    // all stage*_valid/idx control KEEP their async reset.\n"
           "    always @(posedge clk) begin\n"
           "        if (" + guard + ") begin\n"
           "            input_buf <= data_in;\n"
           "        end\n"
           "        if (" + dp_guard + ") begin\n"
           + dp_body
           + "        end\n"
           "    end\n\n")
    edits.append(Edit("insert add Block A", m.group(0), blk + m.group(0)))

    regs += ["input_buf[%d]" % in_w, "dp_data_out[%d]" % out_w]
    ff += in_w + out_w
    return FilePatch("output/mobilenet-v2/rtl/%s.v" % name, regs, ff, edits)

# ---------------------------------------------------------------------------
# C5/C6: node_conv_810 + node_linear -- output skid only
# ---------------------------------------------------------------------------

def _skid_only_patch(name, text, relpath):
    ff, edits = _skid_edits(text, relpath, valid_name="dp_valid_out",
                            data_src="dp_data_out")
    return FilePatch("output/mobilenet-v2/rtl/%s.v" % name,
                     ["out_data[%d] (output skid)" % ff], ff, edits)

# ---------------------------------------------------------------------------
# C7: node_mean -- emit_data (the BRAM-critical sync-only block is untouched)
# ---------------------------------------------------------------------------

def _node_mean_patch(text, relpath):
    edits = []
    old = "            emit_data  <= {(C*8){1'b0}};\n"
    if text.count(old) != 1:
        raise Drift("%s: emit_data reset anchor drift" % relpath)
    edits.append(Edit("reset: drop emit_data", old, ""))

    m = _find1(text,
        r"                ST_PACK: begin\n"
        r"                    // serialized clamp\+pack[^\n]*\n"
        r"(                    for \(plane = 0; plane < SCALE_LANES; plane = plane \+ 1\)\n"
        r"                        emit_data\[\(pack_idx\*SCALE_LANES \+ plane\)\*8 \+: 8\] <=\n"
        r"(?:[^\n]*\n){3})", "ST_PACK emit_data loop", relpath)
    pack_loop = m.group(1)
    edits.append(Edit("ST_PACK: remove emit_data pack loop (moved to sync-only block)",
                      m.group(0),
                      "                ST_PACK: begin\n"
                      "                    // [K1-MBV2] emit_data pack loop moved to the sync-only\n"
                      "                    // block below (serialized clamp+pack, 16 ch/cycle).\n"))

    anchor = "    // ---- FSM + decoupled emitter (single driver for emit_busy/emit_tile/state/counters) ----\n"
    if text.count(anchor) != 1:
        raise Drift("%s: FSM comment anchor drift" % relpath)
    blk = ("    // [K1-MBV2] emit_data is DATAPATH: all 1280 bytes are written during\n"
           "    // ST_PACK (pack_idx 0..79) BEFORE emit_busy rises, and data_out is\n"
           "    // sampled only under valid_out (= emit_busy, reset-kept). This block is\n"
           "    // SEPARATE from (and does not touch) the BRAM-critical tiled-accumulate\n"
           "    // sync-only block above (acc_mem/scaled_mem/rounded_mem untouched).\n"
           "    // `plane` is referenced ONLY by this block after the move.\n"
           "    always @(posedge clk) begin\n"
           "        if (state == ST_PACK) begin\n"
           + pack_loop
           + "        end\n"
           "    end\n\n")
    edits.append(Edit("insert emit_data sync-only block", anchor, blk + anchor))
    return FilePatch("output/mobilenet-v2/rtl/node_mean.v",
                     ["emit_data[10240]"], 10240, edits)

# ---------------------------------------------------------------------------
# C8: output_serializer -- buf_data + data_out
# ---------------------------------------------------------------------------

def _serializer_patch(text, relpath):
    edits = []
    old = ("            data_out  <= {BEATW{1'b0}};\n"
           "            buf_data  <= {BUF_W{1'b0}};\n")
    if text.count(old) != 1:
        raise Drift("%s: serializer reset anchor drift" % relpath)
    edits.append(Edit("reset: drop data_out/buf_data", old, ""))

    old = ("                    buf_data  <= {{(BUF_W-W_IN){1'b0}}, data_in};\n"
           "                    busy      <= 1'b1;\n"
           "                    beat      <= {BCW{1'b0}};\n"
           "                    valid_out <= 1'b1;\n"
           "                    data_out  <= data_in[0 +: BEATW];\n"
           "                    last_out  <= (NBEATS == 1);\n")
    if text.count(old) != 1:
        raise Drift("%s: serializer accept anchor drift" % relpath)
    edits.append(Edit("accept: remove buf_data/data_out writes (moved to Block A)", old,
                      "                    busy      <= 1'b1;\n"
                      "                    beat      <= {BCW{1'b0}};\n"
                      "                    valid_out <= 1'b1;\n"
                      "                    last_out  <= (NBEATS == 1);\n"))

    old = ("                        beat     <= beat + 1'b1;\n"
           "                        data_out <= buf_data[(beat + 1'b1)*BEATW +: BEATW];\n"
           "                        last_out <= ((beat + 1'b1) == NBEATS-1);\n")
    if text.count(old) != 1:
        raise Drift("%s: serializer stream anchor drift" % relpath)
    edits.append(Edit("stream: remove data_out write (moved to Block A)", old,
                      "                        beat     <= beat + 1'b1;\n"
                      "                        last_out <= ((beat + 1'b1) == NBEATS-1);\n"))

    anchor = "    /* verilator lint_off WIDTH */\n"
    if text.count(anchor) != 1:
        raise Drift("%s: lint_off anchor drift" % relpath)
    blk = ("    // [K1-MBV2] buf_data/data_out are stream DATA (sync-only, no reset):\n"
           "    // buf_data is fully written on the accept edge and read strictly after\n"
           "    // (beats 1..NBEATS-1); data_out is sampled by the consumer only under\n"
           "    // valid_out (reset-kept). Guards replicate the original branch\n"
           "    // conditions exactly (busy/valid_in/valid_out/ready_in/beat control all\n"
           "    // keep their async reset). Write sites are mutually exclusive on busy.\n"
           "    always @(posedge clk) begin\n"
           "        if (!busy) begin\n"
           "            if (valid_in) begin\n"
           "                buf_data <= {{(BUF_W-W_IN){1'b0}}, data_in};\n"
           "                data_out <= data_in[0 +: BEATW];\n"
           "            end\n"
           "        end else begin\n"
           "            if (valid_out && ready_in && (beat != NBEATS-1)) begin\n"
           "                data_out <= buf_data[(beat + 1'b1)*BEATW +: BEATW];\n"
           "            end\n"
           "        end\n"
           "    end\n")
    edits.append(Edit("insert serializer Block A", anchor, anchor + blk))
    return FilePatch("output/mobilenet-v2/rtl/output_serializer.v",
                     ["buf_data[8192]", "data_out[256]"], 8448, edits)

# ---------------------------------------------------------------------------
# C9: nn2rtl_top_engine.v helper modules (fixed-text anchors -- the module
# bodies were verified against the live tree; any drift aborts)
# ---------------------------------------------------------------------------

def _top_engine_patch():
    edits = []

    # ---- skip_fifo: out_data_r ----
    edits.append(Edit("skip_fifo: out_data_r -> sync-only block",
"""    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr      <= {(ADDR_W+1){1'b0}};
            rd_ptr      <= {(ADDR_W+1){1'b0}};
            out_valid_r <= 1'b0;
            out_data_r  <= {WIDTH{1'b0}};
        end else begin
            if (wr_fire) wr_ptr <= wr_ptr + 1'b1;
            if (out_valid_r && out_ready) out_valid_r <= 1'b0;
            if (load_skid) begin
                out_data_r  <= mem[rd_ptr[ADDR_W-1:0]];
                out_valid_r <= 1'b1;
                rd_ptr      <= rd_ptr + 1'b1;
            end
        end
    end
""",
"""    // [K1-MBV2] out_data_r is FIFO DATA: sampled downstream only under
    // out_valid_r (reset-kept); written only under load_skid (pointer-
    // derived, pointers reset-kept). Sync-only write -> FDRE + BRAM-friendly.
    always @(posedge clk) begin
        if (load_skid) out_data_r <= mem[rd_ptr[ADDR_W-1:0]];
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr      <= {(ADDR_W+1){1'b0}};
            rd_ptr      <= {(ADDR_W+1){1'b0}};
            out_valid_r <= 1'b0;
        end else begin
            if (wr_fire) wr_ptr <= wr_ptr + 1'b1;
            if (out_valid_r && out_ready) out_valid_r <= 1'b0;
            if (load_skid) begin
                out_valid_r <= 1'b1;
                rd_ptr      <= rd_ptr + 1'b1;
            end
        end
    end
"""))

    # ---- engine_output_fifo: out_data (same recode as ResNet K1 P6) ----
    edits.append(Edit("engine_output_fifo: out_data -> sync-only block",
"""    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr    <= {(ADDR_W+1){1'b0}};
            rd_ptr    <= {(ADDR_W+1){1'b0}};
            out_valid <= 1'b0;
            out_data  <= {DATA_W{1'b0}};
        end else begin
            if (wr_fire) wr_ptr <= wr_ptr + 1'b1;
            // Output handshake: drop valid when consumer accepts.
            if (out_valid && out_ready) begin
                out_valid <= 1'b0;
            end
            // Refill output skid when it is empty (or being consumed
            // this cycle) and the FIFO has data. rd_ptr advances on the
            // same edge so the next refill reads the next entry.
            if (load_skid) begin
                out_data  <= mem[rd_ptr[ADDR_W-1:0]];
                out_valid <= 1'b1;
                rd_ptr    <= rd_ptr + 1'b1;
            end
        end
    end
""",
"""    // [K1-MBV2] out_data is FIFO DATA: sampled only under out_valid (kept
    // async-reset); written only under load_skid. Sync-only write -> FDRE.
    always @(posedge clk) begin
        if (load_skid) out_data <= mem[rd_ptr[ADDR_W-1:0]];
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr    <= {(ADDR_W+1){1'b0}};
            rd_ptr    <= {(ADDR_W+1){1'b0}};
            out_valid <= 1'b0;
        end else begin
            if (wr_fire) wr_ptr <= wr_ptr + 1'b1;
            // Output handshake: drop valid when consumer accepts.
            if (out_valid && out_ready) begin
                out_valid <= 1'b0;
            end
            // Refill output skid when it is empty (or being consumed
            // this cycle) and the FIFO has data. rd_ptr advances on the
            // same edge so the next refill reads the next entry.
            if (load_skid) begin
                out_valid <= 1'b1;
                rd_ptr    <= rd_ptr + 1'b1;
            end
        end
    end
"""))

    # ---- stream_to_act_bram_bridge g_w_eq (identical to the ResNet K1 P6 text) ----
    edits.append(Edit("stream bridge g_w_eq: wr_data/skid_data -> Block A",
"""        assign in_ready = !loaded && (!skid_valid || drain_skid);
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                wr_req     <= 1'b0;
                wr_addr    <= 15'd0;
                wr_data    <= 2048'd0;
                word_count <= 16'd0;
                loaded     <= 1'b0;
                skid_valid <= 1'b0;
                skid_data  <= 2048'd0;
            end else begin
                // (1) Grant retires wr_req and advances count.
                if (wr_req && wr_grant) begin
                    wr_req <= 1'b0;
                    word_count <= next_word_count;
                    if (next_word_count == TOTAL_BRAM_WORDS) loaded <= 1'b1;
                end
                // (2) Drain skid into wr_req when bridge is free.
                if (drain_skid) begin
                    wr_req  <= 1'b1;
                    wr_addr <= next_wr_addr;
                    wr_data <= skid_data;
                end
                // (3) Capture new beat into skid; clear skid if drained and no new.
                if (in_valid && !loaded && (!skid_valid || drain_skid)) begin
                    skid_valid <= 1'b1;
                    skid_data  <= in_data;
                end else if (drain_skid) begin
                    skid_valid <= 1'b0;
                end
            end
        end
""",
"""        assign in_ready = !loaded && (!skid_valid || drain_skid);
        // [K1-MBV2] Block A: stream DATA regs (sync-only, no reset). wr_data is
        // consumed only while wr_req is pending; skid_data only while
        // skid_valid -- both controls stay async-reset below. Reset values dead.
        always @(posedge clk) begin
            if (drain_skid) wr_data <= skid_data;
            if (in_valid && !loaded && (!skid_valid || drain_skid))
                skid_data <= in_data;
        end
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                wr_req     <= 1'b0;
                wr_addr    <= 15'd0;
                word_count <= 16'd0;
                loaded     <= 1'b0;
                skid_valid <= 1'b0;
            end else begin
                // (1) Grant retires wr_req and advances count.
                if (wr_req && wr_grant) begin
                    wr_req <= 1'b0;
                    word_count <= next_word_count;
                    if (next_word_count == TOTAL_BRAM_WORDS) loaded <= 1'b1;
                end
                // (2) Drain skid into wr_req when bridge is free.
                if (drain_skid) begin
                    wr_req  <= 1'b1;
                    wr_addr <= next_wr_addr;
                end
                // (3) Capture new beat into skid; clear skid if drained and no new.
                if (in_valid && !loaded && (!skid_valid || drain_skid)) begin
                    skid_valid <= 1'b1;
                end else if (drain_skid) begin
                    skid_valid <= 1'b0;
                end
            end
        end
"""))

    # ---- stream_to_act_bram_bridge g_w_lt (MBV2 1-px/word variant) ----
    edits.append(Edit("stream bridge g_w_lt: wr_data/skid_data -> Block A",
"""        assign in_ready = !loaded && (!skid_valid || drain_skid);
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                wr_req     <= 1'b0;
                wr_addr    <= 15'd0;
                wr_data    <= 2048'd0;
                word_count <= 16'd0;
                loaded     <= 1'b0;
                skid_valid <= 1'b0;
                skid_data  <= {BUS_W{1'b0}};
            end else begin
                // (1) Grant retires wr_req and advances count.
                if (wr_req && wr_grant) begin
                    wr_req <= 1'b0;
                    word_count <= next_word_count;
                    if (next_word_count == TOTAL_BRAM_WORDS) loaded <= 1'b1;
                end
                // (2) Drain skid -> ONE zero-extended word per beat (1 beat = 1 pixel).
                if (drain_skid) begin
                    wr_req  <= 1'b1;
                    wr_addr <= next_wr_addr;
                    wr_data <= { {(2048-BUS_W){1'b0}}, skid_data };
                end
                // (3) Capture new beat into skid.
                if (in_valid && !loaded && (!skid_valid || drain_skid)) begin
                    skid_valid <= 1'b1;
                    skid_data  <= in_data;
                end else if (drain_skid) begin
                    skid_valid <= 1'b0;
                end
            end
        end
""",
"""        assign in_ready = !loaded && (!skid_valid || drain_skid);
        // [K1-MBV2] Block A: stream DATA regs (sync-only, no reset). Same
        // argument as g_w_eq; the zero-extension is pure data formatting.
        always @(posedge clk) begin
            if (drain_skid) wr_data <= { {(2048-BUS_W){1'b0}}, skid_data };
            if (in_valid && !loaded && (!skid_valid || drain_skid))
                skid_data <= in_data;
        end
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                wr_req     <= 1'b0;
                wr_addr    <= 15'd0;
                word_count <= 16'd0;
                loaded     <= 1'b0;
                skid_valid <= 1'b0;
            end else begin
                // (1) Grant retires wr_req and advances count.
                if (wr_req && wr_grant) begin
                    wr_req <= 1'b0;
                    word_count <= next_word_count;
                    if (next_word_count == TOTAL_BRAM_WORDS) loaded <= 1'b1;
                end
                // (2) Drain skid -> ONE zero-extended word per beat (1 beat = 1 pixel).
                if (drain_skid) begin
                    wr_req  <= 1'b1;
                    wr_addr <= next_wr_addr;
                end
                // (3) Capture new beat into skid.
                if (in_valid && !loaded && (!skid_valid || drain_skid)) begin
                    skid_valid <= 1'b1;
                end else if (drain_skid) begin
                    skid_valid <= 1'b0;
                end
            end
        end
"""))

    # ---- stream_to_act_bram_bridge g_w_gt (MBV2 cont_slice variant) ----
    edits.append(Edit("stream bridge g_w_gt: beat_buf/wr_data/skid_data -> Block A",
"""        assign in_ready = !loaded && (!skid_valid || drain_skid);
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                wr_req     <= 1'b0;
                wr_addr    <= 15'd0;
                wr_data    <= 2048'd0;
                word_count <= 16'd0;
                beat_buf   <= {BUS_W{1'b0}};
                slice_idx  <= {(SLICE_W+1){1'b0}};
                buf_active <= 1'b0;
                loaded     <= 1'b0;
                skid_valid <= 1'b0;
                skid_data  <= {BUS_W{1'b0}};
            end else begin
                // (1) Grant retires wr_req and advances count.
                if (wr_req && wr_grant) begin
                    wr_req <= 1'b0;
                    word_count <= next_word_count;
                    if (next_word_count == TOTAL_BRAM_WORDS) loaded <= 1'b1;
                    if (slice_idx == WORDS_PER_BEAT - 1) begin
                        slice_idx  <= {(SLICE_W+1){1'b0}};
                        buf_active <= 1'b0;
                    end else if (next_word_count != TOTAL_BRAM_WORDS) begin
                        // Continue slicing the current beat (defensive guard
                        // on next_word_count so we don't overrun TOTAL_BRAM_WORDS
                        // when a frame ends mid-beat with a non-divisible total).
                        slice_idx <= slice_idx + 1'b1;
                        wr_req    <= 1'b1;
                        wr_addr   <= BRAM_BASE_ADDR[14:0] + next_word_count[14:0];
                        // [LUT-CUT 2026-06-07] Cut 2: fixed-mux constant slice (byte-exact
                        // to beat_buf[(slice_idx+1)*2048 +: 2048] for all reachable slice_idx)
                        wr_data   <= cont_slice;
                    end
                end
                // (2) Load new beat from skid when buf is free.
                if (drain_skid) begin
                    beat_buf   <= skid_data;
                    buf_active <= 1'b1;
                    slice_idx  <= {(SLICE_W+1){1'b0}};
                    wr_req     <= 1'b1;
                    wr_addr    <= next_wr_addr;
                    wr_data    <= skid_data[2047:0];
                end
                // (3) Capture new beat into skid.
                if (in_valid && !loaded && (!skid_valid || drain_skid)) begin
                    skid_valid <= 1'b1;
                    skid_data  <= in_data;
                end else if (drain_skid) begin
                    skid_valid <= 1'b0;
                end
            end
        end
""",
"""        assign in_ready = !loaded && (!skid_valid || drain_skid);
        // [K1-MBV2] Block A: stream DATA regs (sync-only, no reset), consumed
        // only under wr_req/buf_active/skid_valid (async-reset control below).
        // Textual order preserved: a drain_skid wr_data write overrides the
        // continue-slice write, exactly as in the original single block.
        always @(posedge clk) begin
            if (wr_req && wr_grant && (slice_idx != WORDS_PER_BEAT - 1)
                && (next_word_count != TOTAL_BRAM_WORDS))
                wr_data <= cont_slice;
            if (drain_skid) begin
                beat_buf <= skid_data;
                wr_data  <= skid_data[2047:0];
            end
            if (in_valid && !loaded && (!skid_valid || drain_skid))
                skid_data <= in_data;
        end
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                wr_req     <= 1'b0;
                wr_addr    <= 15'd0;
                word_count <= 16'd0;
                slice_idx  <= {(SLICE_W+1){1'b0}};
                buf_active <= 1'b0;
                loaded     <= 1'b0;
                skid_valid <= 1'b0;
            end else begin
                // (1) Grant retires wr_req and advances count.
                if (wr_req && wr_grant) begin
                    wr_req <= 1'b0;
                    word_count <= next_word_count;
                    if (next_word_count == TOTAL_BRAM_WORDS) loaded <= 1'b1;
                    if (slice_idx == WORDS_PER_BEAT - 1) begin
                        slice_idx  <= {(SLICE_W+1){1'b0}};
                        buf_active <= 1'b0;
                    end else if (next_word_count != TOTAL_BRAM_WORDS) begin
                        // Continue slicing the current beat (defensive guard
                        // on next_word_count so we don't overrun TOTAL_BRAM_WORDS
                        // when a frame ends mid-beat with a non-divisible total).
                        slice_idx <= slice_idx + 1'b1;
                        wr_req    <= 1'b1;
                        wr_addr   <= BRAM_BASE_ADDR[14:0] + next_word_count[14:0];
                    end
                end
                // (2) Load new beat from skid when buf is free.
                if (drain_skid) begin
                    buf_active <= 1'b1;
                    slice_idx  <= {(SLICE_W+1){1'b0}};
                    wr_req     <= 1'b1;
                    wr_addr    <= next_wr_addr;
                end
                // (3) Capture new beat into skid.
                if (in_valid && !loaded && (!skid_valid || drain_skid)) begin
                    skid_valid <= 1'b1;
                end else if (drain_skid) begin
                    skid_valid <= 1'b0;
                end
            end
        end
"""))

    # ---- engine_output_bridge g_legacy ----
    edits.append(Edit("engine_output_bridge g_legacy: remove data_out emit write",
"""                if (emit_ready) begin
                    valid_out     <= 1'b1;
                    data_out      <= current_tile;
                    tiles_emitted <= tiles_emitted + 32'd1;
                    if (tiles_emitted + 32'd1 == EXPECTED_TILES[31:0]) drain_complete <= 1'b1;
                    if (last_tile) begin
                        buf_valid <= 1'b0;
""",
"""                if (emit_ready) begin
                    valid_out     <= 1'b1;
                    tiles_emitted <= tiles_emitted + 32'd1;
                    if (tiles_emitted + 32'd1 == EXPECTED_TILES[31:0]) drain_complete <= 1'b1;
                    if (last_tile) begin
                        buf_valid <= 1'b0;
"""))
    edits.append(Edit("engine_output_bridge g_legacy: remove beat_buf pull write",
"""                if (fifo_out_ready && fifo_out_valid) begin
                    beat_buf  <= fifo_out_data;
                    buf_valid <= 1'b1;
                    tile_idx  <= {(TILE_IDX_W+1){1'b0}};
                end
""",
"""                if (fifo_out_ready && fifo_out_valid) begin
                    buf_valid <= 1'b1;
                    tile_idx  <= {(TILE_IDX_W+1){1'b0}};
                end
"""))
    edits.append(Edit("engine_output_bridge g_legacy: reset drop + Block A",
"""        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                valid_out      <= 1'b0;
                data_out       <= {DATA_W{1'b0}};
                beat_buf       <= {ACT_W{1'b0}};
                buf_valid      <= 1'b0;
                tile_idx       <= {(TILE_IDX_W+1){1'b0}};
                tiles_emitted  <= 32'd0;
                drain_complete <= 1'b0;
""",
"""        // [K1-MBV2] beat_buf/data_out are stream DATA (sync-only, no reset):
        // beat_buf is consumed (current_tile) only while buf_valid; data_out is
        // sampled downstream only under valid_out -- both controls keep their
        // async reset below. Reset values dead.
        always @(posedge clk) begin
            if (emit_ready) data_out <= current_tile;
            if (fifo_out_ready && fifo_out_valid) beat_buf <= fifo_out_data;
        end
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                valid_out      <= 1'b0;
                buf_valid      <= 1'b0;
                tile_idx       <= {(TILE_IDX_W+1){1'b0}};
                tiles_emitted  <= 32'd0;
                drain_complete <= 1'b0;
"""))

    # ---- engine_output_bridge g_tiled ----
    edits.append(Edit("engine_output_bridge g_tiled: remove data_out emit write",
"""                if (emit_ready) begin
                    valid_out     <= 1'b1;
                    data_out      <= current_tile;
                    tiles_emitted <= tiles_emitted + 32'd1;
                    if (tiles_emitted + 32'd1 == EXPECTED_TILES[31:0]) drain_complete <= 1'b1;
                    if (last_tile) begin buf_valid <= 1'b0; tile_idx <= 7'd0; end
""",
"""                if (emit_ready) begin
                    valid_out     <= 1'b1;
                    tiles_emitted <= tiles_emitted + 32'd1;
                    if (tiles_emitted + 32'd1 == EXPECTED_TILES[31:0]) drain_complete <= 1'b1;
                    if (last_tile) begin buf_valid <= 1'b0; tile_idx <= 7'd0; end
"""))
    edits.append(Edit("engine_output_bridge g_tiled: remove beat_buf pull write",
"""                if (fifo_out_ready && fifo_out_valid) begin
                    beat_buf    <= fifo_out_data;
                    buf_valid   <= 1'b1;
""",
"""                if (fifo_out_ready && fifo_out_valid) begin
                    buf_valid   <= 1'b1;
"""))
    edits.append(Edit("engine_output_bridge g_tiled: reset drop + Block A",
"""        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                valid_out<=1'b0; data_out<={DATA_W{1'b0}}; beat_buf<={ACT_W{1'b0}};
                buf_valid<=1'b0; tile_idx<=7'd0; beat_in_pos<=16'd0; pull_idx<=16'd0;
                tiles_emitted<=32'd0; drain_complete<=1'b0;
""",
"""        // [K1-MBV2] beat_buf/data_out are stream DATA (sync-only, no reset);
        // consumed only under buf_valid/valid_out (reset-kept). beat_in_pos/
        // pull_idx/tile_idx are position CONTROL and keep their reset.
        always @(posedge clk) begin
            if (emit_ready) data_out <= current_tile;
            if (fifo_out_ready && fifo_out_valid) beat_buf <= fifo_out_data;
        end
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                valid_out<=1'b0;
                buf_valid<=1'b0; tile_idx<=7'd0; beat_in_pos<=16'd0; pull_idx<=16'd0;
                tiles_emitted<=32'd0; drain_complete<=1'b0;
"""))

    # ---- engine_output_bridge g_flat ----
    edits.append(Edit("engine_output_bridge g_flat: remove data_out emit write",
"""                if (emit_ready) begin
                    valid_out     <= 1'b1;
                    data_out      <= current_tile;
                    tiles_emitted <= tiles_emitted + 32'd1;
                    if (tiles_emitted + 32'd1 == EXPECTED_TILES[31:0]) drain_complete <= 1'b1;
                    buf_full    <= 1'b0;   // release the gather for the next position
""",
"""                if (emit_ready) begin
                    valid_out     <= 1'b1;
                    tiles_emitted <= tiles_emitted + 32'd1;
                    if (tiles_emitted + 32'd1 == EXPECTED_TILES[31:0]) drain_complete <= 1'b1;
                    buf_full    <= 1'b0;   // release the gather for the next position
"""))
    edits.append(Edit("engine_output_bridge g_flat: remove gather_buf pull write",
"""                if (fifo_out_ready && fifo_out_valid) begin
                    gather_buf[beat_in_pos*ACT_W +: ACT_W] <= fifo_out_data;
                    if (beat_in_pos == (BEATS_PER_POS-1)) buf_full <= 1'b1;
""",
"""                if (fifo_out_ready && fifo_out_valid) begin
                    if (beat_in_pos == (BEATS_PER_POS-1)) buf_full <= 1'b1;
"""))
    edits.append(Edit("engine_output_bridge g_flat: reset drop + Block A",
"""        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                valid_out<=1'b0; data_out<={DATA_W{1'b0}}; gather_buf<={GW{1'b0}};
                beat_in_pos<=16'd0; buf_full<=1'b0; tiles_emitted<=32'd0; drain_complete<=1'b0;
""",
"""        // [K1-MBV2] gather_buf/data_out are stream DATA (sync-only, no reset):
        // every consumed gather_buf bit is rewritten during the position's
        // BEATS_PER_POS-beat gather before buf_full rises; data_out is sampled
        // only under valid_out. beat_in_pos/buf_full control keep their reset.
        always @(posedge clk) begin
            if (emit_ready) data_out <= current_tile;
            if (fifo_out_ready && fifo_out_valid)
                gather_buf[beat_in_pos*ACT_W +: ACT_W] <= fifo_out_data;
        end
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                valid_out<=1'b0;
                beat_in_pos<=16'd0; buf_full<=1'b0; tiles_emitted<=32'd0; drain_complete<=1'b0;
"""))

    return FilePatch(
        "output/mobilenet-v2/rtl/nn2rtl_top_engine.v",
        ["skip_fifo.out_data_r (x15)", "engine_output_fifo.out_data (x1)",
         "stream_to_act_bram_bridge.{wr_data,skid_data[,beat_buf]} (x33, 3 branches)",
         "engine_output_bridge.{beat_buf,data_out|gather_buf} (x34, 3 OUT_KINDs)"],
        263808,  # exact, computed from the 83 instantiation params (see analysis doc):
                 # skip_fifo 10,624 + engine_output_fifo 2,048 +
                 # stream_to_act 137,472 + engine_output_bridge 113,664
        edits)

# ---------------------------------------------------------------------------
# C10: rtl_library/retile_bridge.v -- retile_gather buf0/buf1
# ---------------------------------------------------------------------------

def _retile_patch():
    edits = []
    edits.append(Edit("retile_gather: insert buf0/buf1 Block A",
"""    assign data_out = emit_chunk;
""",
"""    assign data_out = emit_chunk;

    // [K1-MBV2] buf0/buf1 are ping-pong gather DATA (sync-only, no reset):
    // every consumed tile slice is rewritten during that pixel's N_TILES-beat
    // gather before full0/full1 (reset-kept) marks the buffer drainable, and
    // the drain reads rbuf only while valid_out (= rsel_full). Writes are
    // gated by do_write = valid_in & wsel_empty (valid_in is the producer's
    // reset-held valid; wsel/full* keep reset). FDCE -> FDRE on 2x FULL_W bits.
    always @(posedge clk) begin
        if (do_write) begin
            if (wsel == 1'b0) buf0[g_idx*TILE_W +: TILE_W] <= data_in;
            else              buf1[g_idx*TILE_W +: TILE_W] <= data_in;
        end
    end
"""))
    edits.append(Edit("retile_gather: reset drop + remove buf writes from control block",
"""            buf0 <= {FULL_W{1'b0}};
            buf1 <= {FULL_W{1'b0}};
            full0 <= 1'b0; full1 <= 1'b0;
            wsel  <= 1'b0; rsel  <= 1'b0;
            g_idx <= {GIDX_W{1'b0}};
            e_idx <= {EIDX_W{1'b0}};
        end else begin
            // ---- GATHER / write side (always-accept into the free buffer) ----
            if (do_write) begin
                if (wsel == 1'b0) buf0[g_idx*TILE_W +: TILE_W] <= data_in;
                else              buf1[g_idx*TILE_W +: TILE_W] <= data_in;
                if (g_idx == N_TILES[GIDX_W-1:0] - 1'b1) begin
""",
"""            full0 <= 1'b0; full1 <= 1'b0;
            wsel  <= 1'b0; rsel  <= 1'b0;
            g_idx <= {GIDX_W{1'b0}};
            e_idx <= {EIDX_W{1'b0}};
        end else begin
            // ---- GATHER / write side (always-accept into the free buffer) ----
            // ([K1-MBV2] buf0/buf1 data writes moved to the sync-only block above.)
            if (do_write) begin
                if (g_idx == N_TILES[GIDX_W-1:0] - 1'b1) begin
"""))
    return FilePatch("rtl_library/retile_bridge.v",
                     ["retile_gather.buf0/buf1 (2x FULL_W; 7 insts, 184 tiles total)"],
                     94208, edits)

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def read_text_smart(path):
    raw = path.read_bytes()
    try:
        text, enc = raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        text, enc = raw.decode("cp1252"), "cp1252"
    eol = "\r\n" if "\r\n" in text else "\n"
    return text.replace("\r\n", "\n"), enc, eol


def build_patch_for(root, relpath, builder, *args):
    fp = root / relpath
    if not fp.is_file():
        raise Drift("MISSING FILE: %s" % relpath)
    text, enc, eol = read_text_smart(fp)
    if MARKER in text:
        return None, (text, enc, eol)
    return builder(*args, text=text) if args else builder(text=text), (text, enc, eol)


def main():
    ap = argparse.ArgumentParser(description="K1-MBV2 extension: FDCE->FDRE datapath recode")
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = Path(args.repo_root).resolve() if args.repo_root else \
        Path(__file__).resolve().parent.parent
    print("[K1-MBV2] repo root: %s" % root)

    # (relpath, patch_builder) -- builders that need the file text get it below.
    jobs = []
    for n in _DW_A + _DW_BC:
        jobs.append(("output/mobilenet-v2/rtl/%s.v" % n,
                     lambda text, n=n, rp="output/mobilenet-v2/rtl/%s.v" % n:
                         _dw_patch(n, text, rp)))
    for n in _N4_SINGLE:
        jobs.append(("output/mobilenet-v2/rtl/%s.v" % n,
                     lambda text, n=n, rp="output/mobilenet-v2/rtl/%s.v" % n:
                         _n4_single_patch(n, text, rp)))
    for n in _N4_MULTI:
        jobs.append(("output/mobilenet-v2/rtl/%s.v" % n,
                     lambda text, n=n, rp="output/mobilenet-v2/rtl/%s.v" % n:
                         _n4_multi_patch(n, text, rp)))
    for n in _ADDS:
        jobs.append(("output/mobilenet-v2/rtl/%s.v" % n,
                     lambda text, n=n, rp="output/mobilenet-v2/rtl/%s.v" % n:
                         _add_patch(n, text, rp)))
    for n in ["node_conv_810", "node_linear"]:
        jobs.append(("output/mobilenet-v2/rtl/%s.v" % n,
                     lambda text, n=n, rp="output/mobilenet-v2/rtl/%s.v" % n:
                         _skid_only_patch(n, text, rp)))
    jobs.append(("output/mobilenet-v2/rtl/node_mean.v",
                 lambda text: _node_mean_patch(text, "output/mobilenet-v2/rtl/node_mean.v")))
    jobs.append(("output/mobilenet-v2/rtl/output_serializer.v",
                 lambda text: _serializer_patch(text, "output/mobilenet-v2/rtl/output_serializer.v")))
    jobs.append(("output/mobilenet-v2/rtl/nn2rtl_top_engine.v",
                 lambda text: _top_engine_patch()))
    jobs.append(("rtl_library/retile_bridge.v",
                 lambda text: _retile_patch()))

    # -------- Phase 1: build + validate everything (no writes) --------
    plan, skipped, errors = [], [], []
    for relpath, builder in jobs:
        fp = root / relpath
        if not fp.is_file():
            errors.append("MISSING FILE: %s" % relpath)
            continue
        text, enc, eol = read_text_smart(fp)
        if MARKER in text:
            skipped.append(relpath)
            continue
        try:
            patch = builder(text=text)
        except Drift as e:
            errors.append("BUILD DRIFT: %s" % e)
            continue
        t = text
        ok = True
        for e in patch.edits:
            n = t.count(e.old)
            if n != e.count:
                errors.append("ANCHOR DRIFT in %s -- edit '%s' matched %d times (need exactly %d)"
                              % (relpath, e.desc, n, e.count))
                ok = False
                break
            t = t.replace(e.old, e.new, e.count)
        if ok:
            # final sanity: the marker must now be present (every patch carries it)
            if MARKER not in t:
                errors.append("%s: patched text carries no %s marker" % (relpath, MARKER))
            else:
                plan.append((patch, relpath, t, enc, eol))

    if errors:
        print("\n[K1-MBV2] ABORT -- validation failed; NO files were modified:")
        for e in errors:
            print("  " + e)
        sys.exit(1)

    # -------- Report --------
    print("\n[K1-MBV2] plan: %d files to patch, %d already applied (marker found)"
          % (len(plan), len(skipped)))
    tot = 0
    for patch, relpath, _, _, _ in plan:
        print("  %-46s %8s FF -> %s"
              % (relpath, patch.ff_estimate, "; ".join(patch.regs)))
        tot += patch.ff_estimate or 0
    if skipped:
        print("[K1-MBV2] skipped (idempotent): %d files" % len(skipped))
    print("\n[K1-MBV2] TOTAL estimated FFs moved off rst_n (elaborated config): ~%d" % tot)

    if args.dry_run:
        print("\n[K1-MBV2] dry-run: no files written.")
        return

    # -------- Phase 2: commit (backups first) --------
    for patch, relpath, new_text, enc, eol in plan:
        fp = root / relpath
        bak = fp.with_name(fp.name + ".prek1m")
        if not bak.exists():
            bak.write_bytes(fp.read_bytes())
        if eol != "\n":
            new_text = new_text.replace("\n", eol)
        fp.write_bytes(new_text.encode(enc))
        print("[K1-MBV2] patched %s (backup: %s)" % (relpath, bak.name))

    print("\n[K1-MBV2] done: %d files patched, %d skipped." % (len(plan), len(skipped)))
    print("[K1-MBV2] next: verilator lint, then bash scripts/run_mbv2_e2e_parallel.sh")
    print("           (must end 'RESULT: PASS (8/8 byte-exact)' with IDENTICAL cycles).")

if __name__ == "__main__":
    main()
