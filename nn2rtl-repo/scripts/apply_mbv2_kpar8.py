#!/usr/bin/env python3
"""apply_mbv2_kpar8.py — ENGINE K-PARALLEL P=8 (8 taps/cycle/lane) for the
MobileNetV2 shared-engine top, INCLUDING the [FC-PAD] rider (FC base
13413 -> 13416 so node_linear joins the fast walk). Anchor-asserted +
idempotent; writes .prekp8 backups before first mutation of each file.

DESIGN (see docs/agent_tasks/KPAR8_ANALYSIS.md)
-----------------------------------------------
* The K_PAR param space becomes {1, 4, 8}. The SHARED files
  (output/rtl/engine/mac_array.v, address_generator.v,
  output/rtl/shared_engine_skeleton.v) get a NEW `K_PAR==8` generate branch
  INSERTED BETWEEN the K_PAR==1 branch and the existing K_PAR>1 branch:
      if (K_PAR == 1)      : legacy        (VERBATIM, untouched)
      else if (K_PAR == 8) : new 8-tap     (this script)
      else                 : KPAR4 branch  (VERBATIM, untouched — ResNet)
  so the K_PAR==1 and K_PAR==4 elaborations are textually IDENTICAL to
  today (same scope names g_p1/g_p4/g_walk_legacy/g_walk_kpar/
  g_waddr_kpar/g_ktap_kpar). ResNet's top sets K_PAR=4 and must stay bit-
  AND cycle-identical (gate: e2e PASS 0/100352 @ EXACTLY 5,664,715).
* PORT WIDTHS: act_bytes_ext / tap_mask / k_tap_mask widths become
  max(K_PAR,4)-based expressions — (((K_PAR > 4) ? K_PAR : 4)) — which
  evaluate to the ORIGINAL widths (24b / 4b) at K_PAR=1 and K_PAR=4 and
  widen (56b / 8b) only at K_PAR=8.
* K_PAR==8 (set ONLY by output/mobilenet-v2/rtl/nn2rtl_top_engine.v and the
  -DKPAR8 build of tb/engine_iso_wrap_mbv2.v):
  - weight banks repacked 8-taps-per-line (288b->2304b, depth
    18533(+3 FC pad)->2317, scripts/repack_mbv2_kpar8_banks.py carries the
    layout + FC relocation proof P0..P4);
  - the engine's weight_rd_addr export becomes the GROUP address (old>>3);
  - address_generator: FAST walk (8 taps/cycle) for dense layers with
    IC%8==0 and weight_base%8==0 = all 34 MBV2 pointwise dispatches AND
    (post-pad) the FC dispatch 46; the 12 depthwise dispatches keep the
    SERIAL walk (1 tap/cycle, cycle-identical) through a 2-cycle-piped
    3-bit subword select; an 8-bit per-tap mask travels with each group;
  - mac_array: 8 DSP products/lane + COMBINATIONAL 8:1 tree into the same
    32b accumulator. INT8xINT8 products accumulate exactly (no rounding),
    so group order cannot change the sum; masked taps multiply a ZEROED act
    byte (contribution exactly 0). Pipeline SHAPE (stage1 product regs,
    stage2 gated accumulate) is unchanged -> the skeleton's d5 requant
    drain alignment is UNCHANGED (TREE_STAGES=0). Fmax note: the stage-2
    add is now a 9-operand combinational sum (see analysis doc).
* [FC-PAD] scheduler row 46 weight base 13413 -> 13416 (%8==0) +
  gen_dw_engine_iso_cfg.py linear row parses the scheduler (no more
  hardcoded 13413). Bank relocation lives in the repack script.

Why one act read/cycle still suffices (dense): a fast group consumes 8
consecutive ic bytes of ONE pixel; groups are 8-aligned and the act word is
256 bytes with 256%8==0, so all 8 bytes always sit in the SAME 2048b word
(IC>256, e.g. IC=960/1280, just rotates chunks 8x faster — still 1 read/cycle).

Usage: python scripts/apply_mbv2_kpar8.py [--check]
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAC = REPO / "output" / "rtl" / "engine" / "mac_array.v"
AG = REPO / "output" / "rtl" / "engine" / "address_generator.v"
SKEL = REPO / "output" / "rtl" / "shared_engine_skeleton.v"
TOP = REPO / "output" / "mobilenet-v2" / "rtl" / "nn2rtl_top_engine.v"
SCHED = REPO / "output" / "mobilenet-v2" / "rtl" / "nn2rtl_scheduler.v"
ISO = REPO / "tb" / "engine_iso_wrap_mbv2.v"
DWCFG = REPO / "scripts" / "gen_dw_engine_iso_cfg.py"

MARK = "[KPAR8 2026-06-10]"

_backed_up: set[Path] = set()

# max(K_PAR,4) width expressions (evaluate to the ORIGINAL 24b/4b at
# K_PAR<=4, widen only at K_PAR=8)
EXTW = "(((K_PAR > 4) ? K_PAR : 4)-1)*8-1"   # act_bytes_ext MSB
MSKW = "((K_PAR > 4) ? K_PAR : 4)-1"          # tap_mask MSB


def patch(path: Path, old: str, new: str, tag: str, count: int = 1) -> None:
    text = path.read_text(encoding="utf-8")
    # Idempotency: `new` always carries [KPAR8] marker text that the
    # pre-change files cannot contain, so its presence == already applied.
    if new in text:
        print(f"  [skip] {path.name}: {tag} already applied")
        return
    n = text.count(old)
    if n != count:
        raise SystemExit(f"ANCHOR FAIL {path.name} / {tag}: found {n}, want {count}")
    if path not in _backed_up:
        bak = path.with_name(path.name + ".prekp8")
        if not bak.exists():
            bak.write_text(text, encoding="utf-8", newline="\n")
        _backed_up.add(path)
    path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")
    print(f"  [ok]   {path.name}: {tag}")


# ============================================================================
# 1. mac_array.v
# ============================================================================
def patch_mac() -> None:
    # ---- widen the ext ports with max(K_PAR,4) expressions ----
    patch(MAC, """    // [KPAR4] taps 1..3 broadcast act bytes (dense mode; tap0 reuses the
    // legacy act_byte port). The skeleton ties this 24'd0 when K_PAR==1.
    input  wire [23:0]   act_bytes_ext,
""", f"""    // [KPAR4] taps 1..K_PAR-1 broadcast act bytes (dense mode; tap0 reuses
    // the legacy act_byte port). The skeleton ties this 0 when K_PAR==1.
    // {MARK} width = (max(K_PAR,4)-1) bytes: the K_PAR==1 and K_PAR==4
    // elaborations keep their ORIGINAL [23:0] port exactly; only K_PAR==8
    // widens to [55:0] (taps 1..7).
    input  wire [{EXTW}:0] act_bytes_ext,
""", "act_bytes_ext width")

    patch(MAC, """    // [KPAR4] per-tap valid mask aligned with weight_bus/act bytes
    // (fast group: 1111 / partial; serial fallback: 0001). The skeleton
    // ties 4'b0001 when K_PAR==1. A masked tap's act byte is zeroed before
    // the multiply, so its contribution is EXACTLY 0.
    input  wire [3:0]    tap_mask,
""", f"""    // [KPAR4] per-tap valid mask aligned with weight_bus/act bytes
    // (fast group: all-ones / partial; serial fallback: bit0 only). The
    // skeleton ties bit0=1 when K_PAR==1. A masked tap's act byte is zeroed
    // before the multiply, so its contribution is EXACTLY 0.
    // {MARK} width = max(K_PAR,4): [3:0] at K_PAR<=4 (unchanged), [7:0] at 8.
    input  wire [{MSKW}:0] tap_mask,
""", "tap_mask width")

    # ---- insert the K_PAR==8 branch BETWEEN g_p1 and g_p4 ----
    p8_lanes = "\n".join(
        f"            wire signed [7:0]  a{j} = tap_mask[{j}] ? "
        f"$signed(act_bytes_ext[{j*8-1}:{(j-1)*8}]) : 8'sd0;"
        for j in range(1, 8))
    p8_w = "\n".join(
        f"            wire signed [WGT_W-1:0] w{j} = "
        f"$signed(weight_bus[({j}*256 + lane)*WGT_W +: WGT_W]);"
        for j in range(8))
    p8_mulreg = "\n".join(
        f"            (* use_dsp = \"yes\" *) reg signed [15:0] mul_q1_{j};"
        for j in range(8))
    p8_mul = "\n".join(
        f"                mul_q1_{j} <= w{j} * a{j};" for j in range(8))
    p8_sum = " + ".join(f"mul_q1_{j}" for j in range(8))

    patch(MAC, """    end else begin : g_p4
""", f"""    end else if (K_PAR == 8) begin : g_p8
        // ---- {MARK} 8-tap datapath: 8 DSP products/lane/cycle + a
        // COMBINATIONAL 8:1 adder tree into the 32b accumulator. Same
        // pipeline SHAPE as the serial and 4-tap paths (stage-1 product
        // regs, stage-2 gated accumulate) -> mac_busy timing and the
        // skeleton's d5 requant capture are unchanged (TREE_STAGES=0).
        // Fmax note: stage-2 is now a 9-operand (acc + 8x16b) sum — the
        // deepest combinational adder in the engine; see KPAR8_ANALYSIS.md.
        for (lane = 0; lane < 256; lane = lane + 1) begin : g_mac
            wire signed [7:0]  a_byte = $signed(act_byte);
            // tap0: legacy broadcast byte (dense) or this lane's own
            // channel byte (depthwise) — same select as the serial path.
            wire signed [7:0]  a_lane0 = dw_mode ? $signed(act_word[lane*8 +: 8]) : a_byte;
            // per-tap act bytes, ZEROED when the tap is masked (partial
            // last group / serial-fallback dispatches): a 0 act byte makes
            // the tap's product exactly 0, so masked taps cannot perturb acc.
            wire signed [7:0]  a0 = tap_mask[0] ? a_lane0 : 8'sd0;
{p8_lanes}
            // tap-major weight slices: tap j's 256-lane word at [j*256*WGT_W].
{p8_w}
{p8_mulreg}
            reg signed [31:0] acc;

            always @(posedge clk) begin
{p8_mul}
            end

            // [K1-FDCE] same no-reset accumulate as the serial path
            // (mac_clear pulses on every ST_RUN entry). All operands are
            // signed; the 8-way sum is sign-extended into the 32b acc —
            // exact integer math, identical result to 8 serial adds.
            always @(posedge clk) begin
                if (mac_clear)
                    acc <= 32'sd0;
                else if (mac_valid_q1)
                    acc <= acc + {p8_sum};
            end

            assign acc_out[lane*32 +: 32] = acc;
        end
    end else begin : g_p4
""", "insert g_p8 branch")


# ============================================================================
# 2. address_generator.v
# ============================================================================
def patch_ag() -> None:
    patch(AG, """    // [KPAR4] per-tap valid mask of the group issued THIS cycle (tap j =
    // old K-word k_cnt+j), registered in lockstep with weight_rd_addr/en.
    // Constant 4'b0001 when K_PAR==1 (legacy single-tap issue).
    output wire [3:0]  k_tap_mask
""", f"""    // [KPAR4] per-tap valid mask of the group issued THIS cycle (tap j =
    // old K-word k_cnt+j), registered in lockstep with weight_rd_addr/en.
    // Constant bit0-only when K_PAR==1 (legacy single-tap issue).
    // {MARK} width = max(K_PAR,4): [3:0] at K_PAR<=4 (unchanged), [7:0] at 8.
    output wire [{MSKW}:0]  k_tap_mask
""", "k_tap_mask width")

    # ---- derive the K_PAR==8 walk from the CURRENT K_PAR>1 branch text ----
    text = AG.read_text(encoding="utf-8")
    if "g_walk_kpar8" in text:
        print("  [skip] address_generator.v: g_walk_kpar8 already applied")
        return
    open_anchor = "    end else begin : g_walk_kpar\n"
    close_anchor = "\n    end endgenerate\n"
    if text.count(open_anchor) != 1 or text.count(close_anchor) != 1:
        raise SystemExit("ANCHOR FAIL address_generator.v / kpar branch capture")
    b0 = text.index(open_anchor) + len(open_anchor)
    b1 = text.index(close_anchor)
    kp4 = text[b0:b1]   # the VERBATIM K_PAR==4 walk (stays untouched)

    kp8 = kp4

    def sub(old: str, new: str, tag: str) -> None:
        nonlocal kp8
        if kp8.count(old) != 1:
            raise SystemExit(f"ANCHOR FAIL address_generator.v / kp8-sub {tag}")
        kp8 = kp8.replace(old, new)

    sub("""    // [KPAR4] FAST eligibility (all per-layer constants): dense 1x1 with
    // 4-aligned IC and a 4-aligned weight base (and IC>=4). Every MBV2
    // pointwise dispatch qualifies; depthwise (cfg_depthwise=1) and the FC
    // dispatch (base 13413 % 4 == 1) fall back to the SERIAL walk below.""",
        f"""    // {MARK} FAST eligibility (all per-layer constants): dense with
    // 8-aligned IC and an 8-aligned weight base (and IC>=8). Every MBV2
    // pointwise dispatch qualifies AND (post [FC-PAD]) the FC dispatch
    // (base 13416 % 8 == 0); depthwise (cfg_depthwise=1) falls back to the
    // SERIAL walk below (any alignment, 3-bit subword select).""",
        "eligibility comment")
    sub("""                          && (cfg_ic[1:0] == 2'b00) && (cfg_ic[11:2] != 10'd0)
                          && (cfg_weight_uram_base[1:0] == 2'b00);""",
        """                          && (cfg_ic[2:0] == 3'b000) && (cfg_ic[11:3] != 9'd0)
                          && (cfg_weight_uram_base[2:0] == 3'b000);""",
        "eligibility expr")
    sub("""    // Last GROUP: fast mode issues k_cnt..k_cnt+3 per cycle, so the final
    // issue is at k_cnt == K_TOTAL-4 (fast layers have K_TOTAL%4==0 by the
    // eligibility gate: K_TOTAL = IC for 1x1). Serial keeps the m1 compare.
    wire        k_at_last  = kpar_fast ? (k_cnt == (k_total[15:0] - 16'd4))""",
        """    // Last GROUP: fast mode issues k_cnt..k_cnt+7 per cycle, so the final
    // issue is at k_cnt == K_TOTAL-8 (fast layers have K_TOTAL%8==0 by the
    // eligibility gate: K_TOTAL = IC for 1x1). Serial keeps the m1 compare.
    wire        k_at_last  = kpar_fast ? (k_cnt == (k_total[15:0] - 16'd8))""",
        "k_at_last -8")
    sub("""    wire [3:0]  fast_mask  = { (k_cnt_w + 17'd3) < k_total,
                               (k_cnt_w + 17'd2) < k_total,
                               (k_cnt_w + 17'd1) < k_total,
                               1'b1 };
    reg  [3:0]  k_tap_mask_r;""",
        """    wire [7:0]  fast_mask  = { (k_cnt_w + 17'd7) < k_total,
                               (k_cnt_w + 17'd6) < k_total,
                               (k_cnt_w + 17'd5) < k_total,
                               (k_cnt_w + 17'd4) < k_total,
                               (k_cnt_w + 17'd3) < k_total,
                               (k_cnt_w + 17'd2) < k_total,
                               (k_cnt_w + 17'd1) < k_total,
                               1'b1 };
    reg  [7:0]  k_tap_mask_r;""",
        "8-bit fast_mask")
    sub("            k_tap_mask_r        <= 4'b0001;   // [KPAR4]",
        "            k_tap_mask_r        <= 8'b0000_0001;   // [KPAR8]",
        "reset mask")
    sub("                k_tap_mask_r        <= kpar_fast ? fast_mask : 4'b0001;",
        "                k_tap_mask_r        <= kpar_fast ? fast_mask : 8'b0000_0001;",
        "issue mask")
    sub("                    end else if (kpar_fast && (ic_cnt == (loop_ic - 12'd4))) begin",
        "                    end else if (kpar_fast && (ic_cnt == (loop_ic - 12'd8))) begin",
        "ic wrap compare")
    sub("                        k_cnt <= k_cnt + 16'd4;",
        "                        k_cnt <= k_cnt + 16'd8;",
        "ic wrap step")
    sub("                        ic_cnt <= ic_cnt + (kpar_fast ? 12'd4 : 12'd1);   // [KPAR4]",
        "                        ic_cnt <= ic_cnt + (kpar_fast ? 12'd8 : 12'd1);   // [KPAR8]",
        "ic step 8")
    sub("                        k_cnt  <= k_cnt  + (kpar_fast ? 16'd4 : 16'd1);   // [KPAR4]",
        "                        k_cnt  <= k_cnt  + (kpar_fast ? 16'd8 : 16'd1);   // [KPAR8]",
        "k step 8")

    new_text = (text[:b0 - len(open_anchor)]
                + "    end else if (K_PAR == 8) begin : g_walk_kpar8\n"
                + kp8
                + "\n    end else begin : g_walk_kpar\n"
                + kp4
                + text[b1:])
    bak = AG.with_name(AG.name + ".prekp8")
    if not bak.exists():
        bak.write_text(text, encoding="utf-8", newline="\n")
    AG.write_text(new_text, encoding="utf-8", newline="\n")
    print("  [ok]   address_generator.v: g_walk_kpar8 inserted (kp4 branch verbatim)")


# ============================================================================
# 3. shared_engine_skeleton.v
# ============================================================================
def patch_skel() -> None:
    patch(SKEL, """    wire [3:0]                 ag_k_tap_mask;   // [KPAR4] per-tap valid of the issued group
""", f"""    wire [{MSKW}:0] ag_k_tap_mask;   // [KPAR4] per-tap valid of the issued group ({MARK} max(K_PAR,4) wide)
""", "ag_k_tap_mask width")

    patch(SKEL, """    end else begin : g_waddr_kpar
        assign weight_rd_addr  = {2'b00, ag_weight_rd_addr[21:2]};
    end endgenerate
""", f"""    end else if (K_PAR == 8) begin : g_waddr_kpar8
        assign weight_rd_addr  = {{3'b000, ag_weight_rd_addr[21:3]}};   // {MARK} GROUP addr = old>>3
    end else begin : g_waddr_kpar
        assign weight_rd_addr  = {{2'b00, ag_weight_rd_addr[21:2]}};
    end endgenerate
""", "g_waddr_kpar8")

    patch(SKEL, """    wire [3:0]  mac_tap_mask;
    wire [23:0] mac_act_bytes_ext;
""", f"""    wire [{MSKW}:0] mac_tap_mask;        // {MARK} max(K_PAR,4) wide
    wire [{EXTW}:0] mac_act_bytes_ext;  // {MARK} (max(K_PAR,4)-1) bytes
""", "mac tap wire widths")

    kidx8 = "\n".join(
        f"        wire [7:0] kidx{j} = ag_act_in_ic_byte_idx_d2 + 8'd{j};"
        for j in range(1, 8))
    ext8 = "\n".join(
        f"        assign mac_act_bytes_ext[{j*8-1}:{(j-1)*8}]   = "
        f"act_in_rd_data_d[kidx{j}*ACT_W +: ACT_W];"
        for j in range(1, 8))
    pass8 = "\n".join(
        f"""        assign mac_weight_bus[{j}*(MAC_COUNT*WGT_W) +: MAC_COUNT*WGT_W] =
            weight_rd_data[{j}*(MAC_COUNT*WGT_W) +: MAC_COUNT*WGT_W];"""
        for j in range(1, 8))

    patch(SKEL, """    end else begin : g_ktap_kpar
        // subword + mask pipes (2-cycle, mirroring ag_weight_rd_en_d/_d2).
""", f"""    end else if (K_PAR == 8) begin : g_ktap_kpar8
        // {MARK} 3-bit subword + 8-bit mask pipes (2-cycle, mirroring
        // ag_weight_rd_en_d/_d2 = the WLAT=2 URAM alignment). FAST dense
        // groups are 8-aligned so the subsel is 0 and tap0 == slice0;
        // SERIAL dispatches (the 12 depthwise) walk one old word/cycle with
        // mask bit0-only, tap0 tracking the 3-bit subword; taps 1..7 are
        // masked dead inside mac_array.
        reg [2:0] wsub_d1, wsub_d2;
        reg [7:0] ktap_d1, ktap_d2;
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                wsub_d1 <= 3'd0;         wsub_d2 <= 3'd0;
                ktap_d1 <= 8'b0000_0001; ktap_d2 <= 8'b0000_0001;
            end else begin
                wsub_d1 <= ag_weight_rd_addr[2:0]; wsub_d2 <= wsub_d1;
                ktap_d1 <= ag_k_tap_mask;          ktap_d2 <= ktap_d1;
            end
        end
        // tap0 = subword-selected old word (slice 0 for aligned fast groups).
        assign mac_weight_bus[MAC_COUNT*WGT_W-1:0] =
            weight_rd_data[wsub_d2*(MAC_COUNT*WGT_W) +: MAC_COUNT*WGT_W];
{pass8}
        assign mac_tap_mask = ktap_d2;
        // dense taps 1..7 act bytes: consecutive ic bytes of the HELD act
        // word. Fast groups are 8-aligned (idx%8==0 and 256%8==0) so idx+7
        // never crosses the word; the +j adds use 8-bit WRAP intermediates
        // so a serial-mode idx near 255 stays an in-range masked-dead select.
{kidx8}
{ext8}
    end else begin : g_ktap_kpar
        // subword + mask pipes (2-cycle, mirroring ag_weight_rd_en_d/_d2).
""", "g_ktap_kpar8")

    # ---- standalone-parse stubs (NN2RTL_ENGINE_SUBBLOCKS_PROVIDED undef) ----
    patch(SKEL, """    input  wire [23:0]  act_bytes_ext, // [KPAR4]
    input  wire [3:0]   tap_mask,      // [KPAR4]
""", f"""    input  wire [{EXTW}:0] act_bytes_ext, // [KPAR4] ({MARK} max(K_PAR,4)-1 bytes)
    input  wire [{MSKW}:0] tap_mask,      // [KPAR4] ({MARK} max(K_PAR,4) wide)
""", "mac stub widths")

    patch(SKEL, """    output wire [3:0]   k_tap_mask     // [KPAR4]
""", f"""    output wire [{MSKW}:0] k_tap_mask  // [KPAR4] ({MARK} max(K_PAR,4) wide)
""", "AG stub mask width")

    patch(SKEL, """    assign k_tap_mask         = 4'b0001;   // [KPAR4]
""", f"""    assign k_tap_mask         = {{{{(((K_PAR > 4) ? K_PAR : 4)-1){{1'b0}}}}, 1'b1}};   // [KPAR4]/{MARK}
""", "AG stub mask assign")


# ============================================================================
# 4. nn2rtl_top_engine.v (MBV2-only)
# ============================================================================
def patch_top() -> None:
    patch(TOP, """    wire [8191:0]              engine_weight_rd_data;  // [KPAR4] 4 tap-major 2048b words per (group) line
""", f"""    wire [16383:0]             engine_weight_rd_data;  // {MARK} 8 tap-major 2048b words per (group) line
""", "weight bus width")

    patch(TOP, """    wire [12:0] weight_bank_rd_addr = engine_weight_rd_addr[12:0];  // [KPAR4] GROUP-addressed: 4634 wide lines (engine exports old>>2)
""", f"""    wire [11:0] weight_bank_rd_addr = engine_weight_rd_addr[11:0];  // {MARK} GROUP-addressed: 2317 wide lines (engine exports old>>3)
""", "bank addr slice")

    text = TOP.read_text(encoding="utf-8")
    old_wires = "\n".join(
        f"    wire [1151:0] uram_bank{b}_rd_data;  // [KPAR4] 4 x 288b tap-major"
        for b in range(8))
    new_wires = "\n".join(
        f"    wire [2303:0] uram_bank{b}_rd_data;  // {MARK} 8 x 288b tap-major"
        for b in range(8))
    patch(TOP, old_wires + "\n", new_wires + "\n", "bank data wires")

    patch(TOP, """    genvar kp_tap;
    generate for (kp_tap = 0; kp_tap < 4; kp_tap = kp_tap + 1) begin : g_kpar_wbus
""", f"""    // {MARK} 8 tap-major 2048b words per repacked 2304b bank line (old word
    // 8g+j at bits [j*288 +: 288] of line g — proof: repack_mbv2_kpar8_banks.py
    // P1..P4, incl. the FC-PAD relocation 13413->13416).
    genvar kp_tap;
    generate for (kp_tap = 0; kp_tap < 8; kp_tap = kp_tap + 1) begin : g_kpar_wbus
""", "tap loop 8")

    for b in range(8):
        patch(TOP, f"""    uram_weight_bank #(
        .DEPTH(4634),           // [KPAR4] ceil(18533/4) wide lines
        .ADDR_W(13),
        .WORD_W(1152),          // [KPAR4] 4 x 288b tap-major
        .MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank{b}_kp4.mem")
""", f"""    uram_weight_bank #(
        .DEPTH(2317),           // {MARK} (18533+3 FC pad)/8 wide lines
        .ADDR_W(12),
        .WORD_W(2304),          // {MARK} 8 x 288b tap-major
        .MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank{b}_kp8.mem")
""", f"bank{b} params")

    patch(TOP, """        .URAM_DATA_W(8192),     // [KPAR4] 4 tap-major 2048b words per line
        .K_PAR(4),              // [KPAR4] 4 taps/cycle/lane (dense 1x1 fast walk; DW+FC serial fallback)
""", f"""        .URAM_DATA_W(16384),    // {MARK} 8 tap-major 2048b words per line
        .K_PAR(8),              // {MARK} 8 taps/cycle/lane (dense 1x1 + FC fast walk; DW serial fallback)
""", "engine K_PAR=8")


# ============================================================================
# 5. nn2rtl_scheduler.v — [FC-PAD] row 46 base 13413 -> 13416
# ============================================================================
def patch_sched() -> None:
    patch(SCHED, """            6'd46: weight_base_word_rom = 20'd13413;  // [FC-ENGINE] node_linear
""", f"""            6'd46: weight_base_word_rom = 20'd13416;  // [FC-ENGINE] node_linear ({MARK} FC-PAD: %8 -> FAST walk; banks relocated by repack_mbv2_kpar8_banks.py)
""", "FC base pad")


# ============================================================================
# 6. tb/engine_iso_wrap_mbv2.v — KPAR8 build variant
# ============================================================================
def patch_iso() -> None:
    patch(ISO, """`ifdef KPAR4
    wire [8191:0] eng_weight_rd_data;     // [KPAR4] 4 tap-major 2048b words
`else
""", f"""`ifdef KPAR8
    wire [16383:0] eng_weight_rd_data;    // {MARK} 8 tap-major 2048b words
`elsif KPAR4
    wire [8191:0] eng_weight_rd_data;     // [KPAR4] 4 tap-major 2048b words
`else
""", "iso weight bus width")

    kp8_banks = "\n".join(
        f'    iso_uram_bank #(.WORD_W(2304), .MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank{b}_kp8.mem")) '
        f"u{b}(.clk(clk),.rd_addr(wbank_addr),.rd_data(b{b}),.rd_en(eng_weight_rd_en));"
        for b in range(8))
    patch(ISO, """`ifdef KPAR4
    // [KPAR4] repacked 4-taps-per-line banks (group-addressed by the engine).
""", f"""`ifdef KPAR8
    // {MARK} repacked 8-taps-per-line banks (group-addressed by the engine;
    // FC-PAD relocated image — pair with the patched scheduler/cfg base 13416).
    wire [2303:0] b0,b1,b2,b3,b4,b5,b6,b7;
    genvar kp_tap8;
    generate for (kp_tap8 = 0; kp_tap8 < 8; kp_tap8 = kp_tap8 + 1) begin : g_kpar8_wbus
        assign eng_weight_rd_data[kp_tap8*2048 +: 2048] = {{
            b7[kp_tap8*288 +: 256], b6[kp_tap8*288 +: 256],
            b5[kp_tap8*288 +: 256], b4[kp_tap8*288 +: 256],
            b3[kp_tap8*288 +: 256], b2[kp_tap8*288 +: 256],
            b1[kp_tap8*288 +: 256], b0[kp_tap8*288 +: 256]}};
    end endgenerate
{kp8_banks}
`elsif KPAR4
    // [KPAR4] repacked 4-taps-per-line banks (group-addressed by the engine).
""", "iso kpar8 banks")

    patch(ISO, """`ifdef KPAR4
        .URAM_DATA_W(8192),     // [KPAR4] 4 tap-major words per line
        .K_PAR(4),
`else
""", f"""`ifdef KPAR8
        .URAM_DATA_W(16384),    // {MARK} 8 tap-major words per line
        .K_PAR(8),
`elsif KPAR4
        .URAM_DATA_W(8192),     // [KPAR4] 4 tap-major words per line
        .K_PAR(4),
`else
""", "iso engine K_PAR=8")


# ============================================================================
# 7. scripts/gen_dw_engine_iso_cfg.py — linear base parses the scheduler
# ============================================================================
def patch_dwcfg_base() -> None:
    # (a) insert the scheduler-row-46 parser before emit_linear
    patch(DWCFG, '''def emit_linear(vec: int) -> int:
''', '''def _fc_base_from_scheduler() -> int:
    """[KPAR8 FC-PAD] row 46 weight base is patched 13413 -> 13416 by
    apply_mbv2_kpar8.py; parse the deployed scheduler so this cfg can never
    drift from the dispatch table the e2e top actually runs."""
    import re as _re
    t = (REPO / "output" / "mobilenet-v2" / "rtl" / "nn2rtl_scheduler.v").read_text(encoding="utf-8")
    m = _re.search(r"6'd46: weight_base_word_rom = 20'd(\\d+);", t)
    if not m:
        raise SystemExit("scheduler row 46 weight base not found")
    return int(m.group(1))


def emit_linear(vec: int) -> int:
''', "linear base parser")
    # (b) the hardcoded 13413 becomes an f-string interpolation slot
    text = DWCFG.read_text(encoding="utf-8")
    if "CFG_WEIGHT_BASE {fc_base}" in text:
        print("  [skip] gen_dw_engine_iso_cfg.py: linear base placeholder already applied")
        return
    old = "#define CFG_WEIGHT_BASE 13413"
    if text.count(old) != 1:
        raise SystemExit("ANCHOR FAIL gen_dw_engine_iso_cfg.py / linear base")
    text = text.replace(old, "#define CFG_WEIGHT_BASE {fc_base}")
    DWCFG.write_text(text, encoding="utf-8", newline="\n")
    print("  [ok]   gen_dw_engine_iso_cfg.py: linear base -> {fc_base} placeholder")


def patch_dwcfg_format() -> None:
    # the emit_linear f-string must interpolate the parsed base
    text = DWCFG.read_text(encoding="utf-8")
    if "fc_base = _fc_base_from_scheduler()" in text:
        print("  [skip] gen_dw_engine_iso_cfg.py: base interpolation already applied")
        return
    old = '''    hdr = f"""// AUTO-GENERATED by scripts/gen_dw_engine_iso_cfg.py linear {vec} — do not edit.'''
    if text.count(old) != 1:
        raise SystemExit("ANCHOR FAIL gen_dw_engine_iso_cfg.py / emit_linear hdr")
    new = '''    fc_base = _fc_base_from_scheduler()
    hdr = f"""// AUTO-GENERATED by scripts/gen_dw_engine_iso_cfg.py linear {vec} — do not edit.'''
    text = text.replace(old, new)
    DWCFG.write_text(text, encoding="utf-8", newline="\n")
    print("  [ok]   gen_dw_engine_iso_cfg.py: emit_linear parses scheduler base")


def main() -> int:
    check = "--check" in sys.argv
    if check:
        for p in (MAC, AG, SKEL, TOP, SCHED, ISO, DWCFG):
            t = p.read_text(encoding="utf-8")
            print(f"{p.name}: {'APPLIED' if 'KPAR8' in t else 'not applied'}")
        return 0
    print("[kpar8] patching mac_array.v ...");              patch_mac()
    print("[kpar8] patching address_generator.v ...");      patch_ag()
    print("[kpar8] patching shared_engine_skeleton.v ..."); patch_skel()
    print("[kpar8] patching nn2rtl_top_engine.v ...");      patch_top()
    print("[kpar8] patching nn2rtl_scheduler.v (FC-PAD) ..."); patch_sched()
    print("[kpar8] patching engine_iso_wrap_mbv2.v ...");   patch_iso()
    print("[kpar8] patching gen_dw_engine_iso_cfg.py ...")
    patch_dwcfg_base()
    patch_dwcfg_format()
    print("[kpar8] DONE. Next: scripts/repack_mbv2_kpar8_banks.py (if not yet run), "
          "lint (K_PAR=1/4/8), engine-ISO (-DKPAR8 + KPAR4 DW cycle-identity), "
          "8/8 e2e, ResNet inertness e2e (K_PAR=4, must stay 5,664,715).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
