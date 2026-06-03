#!/usr/bin/env python3
# Compression transform for node_conv_818 (MobileNetV2 depthwise 3x3, IC=OC=96).
#
# CRUX MEASUREMENT for the HYBRID-vs-ALL-SPATIAL decision.
#
# Baseline (Vivado synth, xczu9eg part): 336,522 LUT, of which 281,344 are
# "LUT as Memory" (distributed LUTRAM). Vivado reports:
#     [Synth 8-6849] Infeasible attribute ram_style="block" ... using LUTRAM
# for BOTH line_buf_b* (13x RAM64M8x1760 = 22,880) and out_buf_b* (4x
# RAM256X1D x3072 = 12,288). The (* ram_style="block" *) hint SILENTLY FAILED.
#
# Why infeasible:
#   * line_buf is read COMBINATIONALLY (always @(*) mux) -> async read forbids
#     BRAM. Cannot fix without a registered read = +1 latency cycle, which the
#     cycle-exact testbench (pipeline_latency_cycles=1124, hard-enforced)
#     rejects. So line_buf is STRUCTURALLY LOCKED to LUTRAM here.
#   * out_buf is written byte-granular at a runtime offset
#     (out_buf_bN[pix][wb_ch*8 +: 8] <= sat_byte) across a 768-bit word ->
#     runtime partial write forbids BRAM byte-enable.
#
# This transform attacks ONLY the out_buf (the cycle-safe lever): replace the
# 4 banks of [1024 x 768] (byte-written) with 96 per-channel arrays
# [N_OUT_PIX x 8] written as FULL aligned 8-bit words. Each becomes a real
# BRAM18. The registered read assembles all 96 channel-bytes for em_pix into
# the 768-bit data_out word (same cycle as before -> cycle-exact).
#
# Also removes the DEAD `window` register array (written, never read).
#
# Value-exact: identical sat_byte computed, identical addressing (cmp_pix /
# em_pix), identical write/read cycles. Pure storage-organization change.
#
# Writes a NEW file; never overwrites the baseline.

import re
import sys

SRC = "output/mobilenet-v2/rtl/node_conv_818.v"
DST = "output/mobilenet-v2/rtl/improved/node_conv_818.compressed.v"

src = open(SRC, "r", encoding="utf-8").read()
orig = src

# ---------------------------------------------------------------------------
# 1) out_buf declaration: 4 wide banks -> 96 per-channel narrow BRAM arrays.
# ---------------------------------------------------------------------------
old_decl = (
    '    (* ram_style = "block" *) reg [BEAT_W-1:0] out_buf_b0 [0:1023];\n'
    '    (* ram_style = "block" *) reg [BEAT_W-1:0] out_buf_b1 [0:1023];\n'
    '    (* ram_style = "block" *) reg [BEAT_W-1:0] out_buf_b2 [0:1023];\n'
    '    (* ram_style = "block" *) reg [BEAT_W-1:0] out_buf_b3 [0:1023];\n'
)
assert old_decl in src, "out_buf decl anchor not found"
# Per-channel byte arrays, depth N_OUT_PIX (3136). ram_style=block -> BRAM18.
new_decl_lines = ["    // [COMPRESS] out_buf restructured: 96 per-channel BRAM byte arrays\n"
                  "    //           (full-word aligned writes) replace 4x768b byte-written LUTRAM banks.\n"]
for c in range(96):
    new_decl_lines.append(
        f'    (* ram_style = "block" *) reg signed [7:0] out_ch{c:02d} [0:N_OUT_PIX-1];\n'
    )
new_decl = "".join(new_decl_lines)
src = src.replace(old_decl, new_decl)

# ---------------------------------------------------------------------------
# 2) writeback: byte-granular partial write into wide bank -> full-word write
#    into the per-channel array selected by wb_ch.
# ---------------------------------------------------------------------------
old_wb = (
    '                        case (cmp_pix[11:10])\n'
    "                            2'd0: out_buf_b0[cmp_pix[9:0]][wb_ch*8 +: 8] <= sat_byte;\n"
    "                            2'd1: out_buf_b1[cmp_pix[9:0]][wb_ch*8 +: 8] <= sat_byte;\n"
    "                            2'd2: out_buf_b2[cmp_pix[9:0]][wb_ch*8 +: 8] <= sat_byte;\n"
    "                            2'd3: out_buf_b3[cmp_pix[9:0]][wb_ch*8 +: 8] <= sat_byte;\n"
    '                        endcase\n'
)
assert old_wb in src, "writeback anchor not found"
wb_cases = ["                        case (wb_ch)\n"]
for c in range(96):
    wb_cases.append(f"                            7'd{c}: out_ch{c:02d}[cmp_pix] <= sat_byte;\n")
wb_cases.append("                            default: ;\n")
wb_cases.append("                        endcase\n")
new_wb = "".join(wb_cases)
src = src.replace(old_wb, new_wb)

# ---------------------------------------------------------------------------
# 3) read: registered read from wide bank -> registered read assembling all 96
#    per-channel bytes at em_pix into the 768-bit data_out word.
# ---------------------------------------------------------------------------
old_rd = (
    '                case (em_pix[11:10])\n'
    "                    2'd0: data_out <= out_buf_b0[em_pix[9:0]];\n"
    "                    2'd1: data_out <= out_buf_b1[em_pix[9:0]];\n"
    "                    2'd2: data_out <= out_buf_b2[em_pix[9:0]];\n"
    "                    2'd3: data_out <= out_buf_b3[em_pix[9:0]];\n"
    '                endcase\n'
)
assert old_rd in src, "read anchor not found"
# Assemble {ch95, ch94, ..., ch00} (ch00 occupies bits [7:0]).
parts = ", ".join(f"out_ch{c:02d}[em_pix]" for c in range(95, -1, -1))
new_rd = (
    "                data_out <= {" + parts + "};\n"
)
src = src.replace(old_rd, new_rd)

# ---------------------------------------------------------------------------
# 4) Remove the DEAD `window` register array + its write logic (no fanout).
# ---------------------------------------------------------------------------
old_win_decl = "    reg signed [7:0]   window [0:K_TOTAL-1][0:MP-1];\n"
assert old_win_decl in src, "window decl anchor not found"
src = src.replace(old_win_decl, "    // [COMPRESS] dead `window` array removed (written, never read).\n")

old_win_blk = (
    "    always @(posedge clk) begin\n"
    "        if (!rst_n) begin\n"
    "            for (wj = 0; wj < K_TOTAL; wj = wj + 1) begin\n"
    "                window[wj][0] <= 8'sd0;\n"
    "                window[wj][1] <= 8'sd0;\n"
    "                window[wj][2] <= 8'sd0;\n"
    "                window[wj][3] <= 8'sd0;\n"
    "            end\n"
    "        end else if (is_issue) begin\n"
    "            window[cur_k][cur_lane] <= act_byte;\n"
    "        end\n"
    "    end\n"
)
assert old_win_blk in src, "window block anchor not found"
src = src.replace(old_win_blk, "    // [COMPRESS] dead `window` write process removed.\n")

# 'wj' integer is now unused but harmless; leave it (declared as `integer wj;`).

assert src != orig, "no change applied"
open(DST, "w", encoding="utf-8").write(src)
print("wrote", DST)
print("delta bytes:", len(src) - len(orig))
