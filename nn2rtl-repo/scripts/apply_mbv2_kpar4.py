#!/usr/bin/env python3
"""apply_mbv2_kpar4.py — ENGINE K-PARALLEL P=4 (4 taps/cycle/lane) for the
MobileNetV2 shared-engine top. Anchor-asserted + idempotent; writes .prekp4
backups before first mutation of each file.

DESIGN (see docs/agent_tasks/KPAR4_ANALYSIS.md)
-----------------------------------------------
* SHARED files (output/rtl/engine/mac_array.v, address_generator.v,
  output/rtl/shared_engine_skeleton.v) get a parameter K_PAR (default 1).
  K_PAR==1 elaborates the ORIGINAL logic VERBATIM inside generate-if
  branches -> every legacy instance (all ResNet tops + harnesses, which
  never set K_PAR) is bit- and cycle-identical by construction.
* K_PAR==4 (set ONLY by output/mobilenet-v2/rtl/nn2rtl_top_engine.v and the
  KPAR4 build of tb/engine_iso_wrap_mbv2.v):
  - weight banks repacked 4-taps-per-line (288b->1152b, depth 18533->4634,
    scripts/repack_mbv2_kpar4_banks.py carries the layout proof);
  - the engine's weight_rd_addr export becomes the GROUP address (old>>2);
  - address_generator: FAST walk (4 taps/cycle) for dense 1x1 layers with
    IC%4==0 and weight_base%4==0 (= all 34 MBV2 pointwise dispatches);
    everything else (12 depthwise dispatches, FC@13413 base%4==1) keeps the
    SERIAL walk (1 tap/cycle) and reads its old word through a 2-cycle-piped
    subword select on the wide line; a 4-bit per-tap mask travels with each
    issued group (fast: 1111 [partial-group capable]; serial: 0001);
  - mac_array: 4 DSP products/lane + COMBINATIONAL 4:1 tree into the same
    32b accumulator. INT8xINT8 products accumulate exactly (no rounding),
    so group order cannot change the sum; masked taps multiply a ZEROED act
    byte (contribution exactly 0). Pipeline SHAPE (stage1 product regs,
    stage2 gated accumulate) is unchanged -> the skeleton's d5 requant
    drain alignment is UNCHANGED (TREE_STAGES=0).

Usage: python scripts/apply_mbv2_kpar4.py [--check]
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAC = REPO / "output" / "rtl" / "engine" / "mac_array.v"
AG = REPO / "output" / "rtl" / "engine" / "address_generator.v"
SKEL = REPO / "output" / "rtl" / "shared_engine_skeleton.v"
TOP = REPO / "output" / "mobilenet-v2" / "rtl" / "nn2rtl_top_engine.v"
ISO = REPO / "tb" / "engine_iso_wrap_mbv2.v"

MARK = "[KPAR4 2026-06-10]"

_backed_up: set[Path] = set()


def patch(path: Path, old: str, new: str, tag: str, count: int = 1) -> None:
    text = path.read_text(encoding="utf-8")
    # Idempotency: `new` always carries [KPAR4] marker text that the
    # pre-change files cannot contain, so its presence == already applied.
    # (Do NOT also require `old not in text`: additive hunks keep `old` as a
    # substring of `new`, which would re-apply and duplicate lines.)
    if new in text:
        print(f"  [skip] {path.name}: {tag} already applied")
        return
    n = text.count(old)
    if n != count:
        raise SystemExit(f"ANCHOR FAIL {path.name} / {tag}: found {n}, want {count}")
    if path not in _backed_up:
        bak = path.with_name(path.name + ".prekp4")
        if not bak.exists():
            bak.write_text(text, encoding="utf-8", newline="\n")
        _backed_up.add(path)
    path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")
    print(f"  [ok]   {path.name}: {tag}")


# ============================================================================
# 1. mac_array.v
# ============================================================================
def patch_mac() -> None:
    patch(MAC, """    parameter integer WGT_W = 4
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          mac_clear,
    input  wire          mac_valid_in,
    input  wire [7:0]    act_byte,
    input  wire [256*WGT_W-1:0] weight_bus,  // WGT_W-packed: 256 lanes * WGT_W bits
""", """    parameter integer WGT_W = 4,
    // [KPAR4 2026-06-10] K-tap parallelism. 1 (DEFAULT) elaborates the
    // ORIGINAL serial datapath via generate-if — every legacy instance
    // (all ResNet tops/harnesses never set K_PAR) is bit- and
    // cycle-identical. 4 = MBV2 engine top: 4 taps/cycle/lane; the 4
    // products are summed by a COMBINATIONAL 4:1 tree into the same 32b
    // accumulator (INT8xINT8 -> 32b accumulation is exact and
    // order-independent), so the accumulate latency — and the skeleton's
    // d5 requant drain — is UNCHANGED (TREE_STAGES=0).
    parameter integer K_PAR = 1
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          mac_clear,
    input  wire          mac_valid_in,
    input  wire [7:0]    act_byte,
    input  wire [K_PAR*256*WGT_W-1:0] weight_bus,  // WGT_W-packed: K_PAR taps x 256 lanes (tap-major, tap0 lowest)
    // [KPAR4] taps 1..3 broadcast act bytes (dense mode; tap0 reuses the
    // legacy act_byte port). The skeleton ties this 24'd0 when K_PAR==1.
    input  wire [23:0]   act_bytes_ext,
    // [KPAR4] per-tap valid mask aligned with weight_bus/act bytes
    // (fast group: 1111 / partial; serial fallback: 0001). The skeleton
    // ties 4'b0001 when K_PAR==1. A masked tap's act byte is zeroed before
    // the multiply, so its contribution is EXACTLY 0.
    input  wire [3:0]    tap_mask,
""", "K_PAR param + ext ports")

    patch(MAC, """    genvar lane;
    generate
        for (lane = 0; lane < 256; lane = lane + 1) begin : g_mac
""", """    genvar lane;
    generate
    if (K_PAR == 1) begin : g_p1
        // ---- [KPAR4] ORIGINAL serial datapath, VERBATIM (legacy default) ----
        for (lane = 0; lane < 256; lane = lane + 1) begin : g_mac
""", "open legacy generate branch")

    patch(MAC, """            assign acc_out[lane*32 +: 32] = acc;
        end
    endgenerate
""", """            assign acc_out[lane*32 +: 32] = acc;
        end
        // [KPAR4] lint tie: ext ports are consumed only by the K_PAR>1 branch.
        /* verilator lint_off UNUSED */
        wire _unused_kpar_ext = &{1'b0, act_bytes_ext, tap_mask};
        /* verilator lint_on UNUSED */
    end else begin : g_p4
        // ---- [KPAR4] 4-tap datapath: 4 DSP products/lane/cycle + a
        // COMBINATIONAL 4:1 adder tree into the 32b accumulator. The
        // pipeline SHAPE matches the serial path exactly (stage-1 product
        // regs, stage-2 gated accumulate), so mac_busy timing and the
        // skeleton's d5 requant capture are unchanged.
        for (lane = 0; lane < 256; lane = lane + 1) begin : g_mac
            wire signed [7:0]  a_byte = $signed(act_byte);
            // tap0: legacy broadcast byte (dense) or this lane's own
            // channel byte (depthwise) — same select as the serial path.
            wire signed [7:0]  a_lane0 = dw_mode ? $signed(act_word[lane*8 +: 8]) : a_byte;
            // per-tap act bytes, ZEROED when the tap is masked (partial
            // last group / serial-fallback dispatches): a 0 act byte makes
            // the tap's product exactly 0, so masked taps cannot perturb acc.
            wire signed [7:0]  a0 = tap_mask[0] ? a_lane0                       : 8'sd0;
            wire signed [7:0]  a1 = tap_mask[1] ? $signed(act_bytes_ext[7:0])   : 8'sd0;
            wire signed [7:0]  a2 = tap_mask[2] ? $signed(act_bytes_ext[15:8])  : 8'sd0;
            wire signed [7:0]  a3 = tap_mask[3] ? $signed(act_bytes_ext[23:16]) : 8'sd0;
            // tap-major weight slices: tap j's 256-lane word at [j*256*WGT_W].
            wire signed [WGT_W-1:0] w0 = $signed(weight_bus[(0*256 + lane)*WGT_W +: WGT_W]);
            wire signed [WGT_W-1:0] w1 = $signed(weight_bus[(1*256 + lane)*WGT_W +: WGT_W]);
            wire signed [WGT_W-1:0] w2 = $signed(weight_bus[(2*256 + lane)*WGT_W +: WGT_W]);
            wire signed [WGT_W-1:0] w3 = $signed(weight_bus[(3*256 + lane)*WGT_W +: WGT_W]);
            (* use_dsp = "yes" *) reg signed [15:0] mul_q1_0;
            (* use_dsp = "yes" *) reg signed [15:0] mul_q1_1;
            (* use_dsp = "yes" *) reg signed [15:0] mul_q1_2;
            (* use_dsp = "yes" *) reg signed [15:0] mul_q1_3;
            reg signed [31:0] acc;

            always @(posedge clk) begin
                mul_q1_0 <= w0 * a0;
                mul_q1_1 <= w1 * a1;
                mul_q1_2 <= w2 * a2;
                mul_q1_3 <= w3 * a3;
            end

            // [K1-FDCE] same no-reset accumulate as the serial path
            // (mac_clear pulses on every ST_RUN entry). All operands are
            // signed; the 4-way sum is sign-extended into the 32b acc —
            // exact integer math, identical result to 4 serial adds.
            always @(posedge clk) begin
                if (mac_clear)
                    acc <= 32'sd0;
                else if (mac_valid_q1)
                    acc <= acc + mul_q1_0 + mul_q1_1 + mul_q1_2 + mul_q1_3;
            end

            assign acc_out[lane*32 +: 32] = acc;
        end
    end
    endgenerate
""", "close legacy + add K_PAR=4 branch")


# ============================================================================
# 2. address_generator.v
# ============================================================================
def patch_ag() -> None:
    patch(AG, """module address_generator (
    input  wire        clk,
""", """module address_generator #(
    // [KPAR4 2026-06-10] K-tap parallelism. 1 (DEFAULT) elaborates the
    // ORIGINAL serial walk VERBATIM via generate-if — bit- and
    // cycle-identical for every legacy instance. 4 = walk the K dimension
    // 4 old-words/cycle for FAST-eligible layers (dense 1x1, IC%4==0,
    // weight_base%4==0 — all 34 MBV2 pointwise dispatches); everything
    // else (depthwise, the FC dispatch at base 13413%4==1) keeps the
    // SERIAL walk and is served by the skeleton's subword select.
    parameter integer K_PAR = 1
) (
    input  wire        clk,
""", "K_PAR param")

    patch(AG, """    output reg  [15:0] k_index,
    output reg         mac_done,
    output reg         pixel_done
);
""", """    output reg  [15:0] k_index,
    output reg         mac_done,
    output reg         pixel_done,

    // [KPAR4] per-tap valid mask of the group issued THIS cycle (tap j =
    // old K-word k_cnt+j), registered in lockstep with weight_rd_addr/en.
    // Constant 4'b0001 when K_PAR==1 (legacy single-tap issue).
    output wire [3:0]  k_tap_mask
);
""", "k_tap_mask port")

    # ---- wrap the original k_at_last + sequential body in generate-if ----
    text = AG.read_text(encoding="utf-8")
    start_anchor = "    wire        k_at_last       = (k_cnt == k_total_m1);\n"
    end_anchor = """            pixel_done <= pixel_done_latch;
        end
    end

endmodule
"""
    if "g_walk_legacy" in text:
        print("  [skip] address_generator.v: walk generate already applied")
        return
    if text.count(start_anchor) != 1 or text.count(end_anchor) != 1:
        raise SystemExit("ANCHOR FAIL address_generator.v / walk capture")
    body_start = text.index(start_anchor)
    body_end = text.index(end_anchor) + end_anchor.index("endmodule")
    legacy = text[body_start:body_end]  # k_at_last wire .. end of always (excl endmodule)

    # ---- derive the K_PAR>1 walk by ASSERTED transforms of the verbatim copy ----
    kpar = legacy

    def sub(old: str, new: str, tag: str) -> None:
        nonlocal kpar
        if kpar.count(old) != 1:
            raise SystemExit(f"ANCHOR FAIL address_generator.v / kpar-sub {tag}")
        kpar = kpar.replace(old, new)

    sub("""    wire        k_at_last       = (k_cnt == k_total_m1);
""", """    // [KPAR4] FAST eligibility (all per-layer constants): dense 1x1 with
    // 4-aligned IC and a 4-aligned weight base (and IC>=4). Every MBV2
    // pointwise dispatch qualifies; depthwise (cfg_depthwise=1) and the FC
    // dispatch (base 13413 % 4 == 1) fall back to the SERIAL walk below.
    wire        kpar_fast = (!cfg_depthwise)
                          && (cfg_kh == 3'd1) && (cfg_kw == 3'd1)
                          && (cfg_ic[1:0] == 2'b00) && (cfg_ic[11:2] != 10'd0)
                          && (cfg_weight_uram_base[1:0] == 2'b00);
    // Last GROUP: fast mode issues k_cnt..k_cnt+3 per cycle, so the final
    // issue is at k_cnt == K_TOTAL-4 (fast layers have K_TOTAL%4==0 by the
    // eligibility gate: K_TOTAL = IC for 1x1). Serial keeps the m1 compare.
    wire        k_at_last  = kpar_fast ? (k_cnt == (k_total[15:0] - 16'd4))
                                       : (k_cnt == k_total_m1);
    // Per-tap valid of the group issued this cycle (general partial-group
    // form kept for safety; always 4'b1111 on MBV2 fast layers).
    wire [16:0] k_cnt_w    = {1'b0, k_cnt};
    wire [3:0]  fast_mask  = { (k_cnt_w + 17'd3) < k_total,
                               (k_cnt_w + 17'd2) < k_total,
                               (k_cnt_w + 17'd1) < k_total,
                               1'b1 };
    reg  [3:0]  k_tap_mask_r;
    assign k_tap_mask = k_tap_mask_r;
""", "fast wires")

    sub("""            k_index             <= 16'd0;
            mac_done            <= 1'b0;
            pixel_done          <= 1'b0;
""", """            k_index             <= 16'd0;
            mac_done            <= 1'b0;
            pixel_done          <= 1'b0;
            k_tap_mask_r        <= 4'b0001;   // [KPAR4]
""", "reset mask")

    sub("""                // k_index mirrors the running k_cnt counter.
                k_index             <= k_cnt;
""", """                // k_index mirrors the running k_cnt counter.
                k_index             <= k_cnt;

                // [KPAR4] mask registered in lockstep with weight_rd_addr.
                k_tap_mask_r        <= kpar_fast ? fast_mask : 4'b0001;
""", "issue mask")

    sub("                    end else if (ic_cnt == (loop_ic - 12'd1)) begin   "
        "// [DW-ENGINE P1] loop_ic==1 in DW mode",
        "                    end else if (!kpar_fast && (ic_cnt == (loop_ic - 12'd1))) begin   "
        "// [DW-ENGINE P1] loop_ic==1 in DW mode; [KPAR4] unreachable in fast walk (1x1: k_at_last fires first)",
        "ic boundary guard")

    sub("""                    end else begin
                        ic_cnt <= ic_cnt + 12'd1;
                        k_cnt  <= k_cnt + 16'd1;
                    end
""", """                    end else begin
                        ic_cnt <= ic_cnt + (kpar_fast ? 12'd4 : 12'd1);   // [KPAR4]
                        k_cnt  <= k_cnt  + (kpar_fast ? 16'd4 : 16'd1);   // [KPAR4]
                    end
""", "step 4")

    wrapped = ("""    // ====================================================================
    // [KPAR4 2026-06-10] K_PAR==1 keeps the ORIGINAL wire + sequential body
    // VERBATIM inside generate-if (legacy elaboration provably unchanged).
    // K_PAR>1 adds the FAST 4-taps/cycle walk for eligible layers; all
    // other layers keep the serial walk, with k_tap_mask telling the
    // mac_array which taps of each issued group are live.
    // ====================================================================
    generate if (K_PAR == 1) begin : g_walk_legacy
    assign k_tap_mask = 4'b0001;
""" + legacy + """
    end else begin : g_walk_kpar
""" + kpar + """
    end endgenerate

""")
    new_text = text[:body_start] + wrapped + text[body_end:]
    AG.write_text(new_text, encoding="utf-8", newline="\n")
    print("  [ok]   address_generator.v: legacy walk wrapped + K_PAR walk added")


# ============================================================================
# 3. shared_engine_skeleton.v
# ============================================================================
def patch_skel() -> None:
    patch(SKEL, """    parameter integer ENABLE_DEPTHWISE = 0
) (
""", """    parameter integer ENABLE_DEPTHWISE = 0,

    // ---- K-tap parallelism (default 1 = bit/cycle-identical legacy) ----
    // [KPAR4 2026-06-10] When 1 (DEFAULT) every K_PAR generate-if in this
    // file and in mac_array/address_generator elaborates the ORIGINAL
    // serial logic VERBATIM — ResNet (which never sets K_PAR) is provably
    // unchanged. When 4 (MBV2 engine top + the KPAR4 iso build):
    //   * URAM_DATA_W must be K_PAR*MAC_COUNT*WGT_W (4 tap-major words per
    //     repacked bank line, tap0 lowest);
    //   * weight_rd_addr exports the GROUP address (old word addr >> 2);
    //   * FAST-eligible dense 1x1 layers run 4 taps/cycle; depthwise and
    //     unaligned-base layers fall back to the serial walk through a
    //     2-cycle-piped subword select (byte-exact, legacy-rate).
    parameter integer K_PAR = 1
) (
""", "K_PAR param")

    patch(SKEL, """    wire [15:0]                ag_k_index;
    wire                       ag_mac_done;
    wire                       ag_pixel_done;
""", """    wire [15:0]                ag_k_index;
    wire                       ag_mac_done;
    wire                       ag_pixel_done;
    wire [3:0]                 ag_k_tap_mask;   // [KPAR4] per-tap valid of the issued group
""", "ag_k_tap_mask wire")

    patch(SKEL, """    wire [MAC_COUNT*WGT_W-1:0]        mac_weight_bus;       // 2048 b
""", """    wire [K_PAR*MAC_COUNT*WGT_W-1:0]  mac_weight_bus;       // [KPAR4] K_PAR taps x (MAC_COUNT*WGT_W); 2048 b at K_PAR=1
""", "mac_weight_bus width")

    patch(SKEL, """    assign weight_rd_addr  = ag_weight_rd_addr;
    assign weight_rd_en    = ag_weight_rd_en;
""", """    // [KPAR4] K_PAR>1 exports the GROUP address (old word addr >> 2): the
    // MBV2 banks are repacked 4-taps-per-line (repack_mbv2_kpar4_banks.py).
    // K_PAR==1 is the verbatim legacy passthrough.
    generate if (K_PAR == 1) begin : g_waddr_legacy
        assign weight_rd_addr  = ag_weight_rd_addr;
    end else begin : g_waddr_kpar
        assign weight_rd_addr  = {2'b00, ag_weight_rd_addr[21:2]};
    end endgenerate
    assign weight_rd_en    = ag_weight_rd_en;
""", "group weight addr")

    patch(SKEL, """    // mac_weight_bus is the full URAM-wide weight read; at N+2 it carries the
    // weights requested at read N (one URAM word = MAC_COUNT INT8 weights = 2048b).
    assign mac_weight_bus = weight_rd_data[MAC_COUNT*WGT_W-1:0];

""", """    // [KPAR4] K_PAR==1 (DEFAULT): the ORIGINAL single-tap hookup, verbatim.
    // K_PAR==4: weight_rd_data carries 4 tap-major words per (group-
    // addressed) line. Tap0 is selected by the OLD address's [1:0], piped
    // 2 cycles exactly like ..._rd_en_d/_d2 to meet the URAM READ_LATENCY=2
    // data: FAST dense groups are 4-aligned so the subsel is 0 and tap0 ==
    // slice0; SERIAL dispatches (depthwise, FC base 13413%4==1) walk one
    // old word/cycle with mask 4'b0001, so tap0 tracks the subword and
    // taps 1..3 are masked dead inside mac_array.
    wire [3:0]  mac_tap_mask;
    wire [23:0] mac_act_bytes_ext;
    generate if (K_PAR == 1) begin : g_ktap_legacy
        // mac_weight_bus is the full URAM-wide weight read; at N+2 it carries the
        // weights requested at read N (one URAM word = MAC_COUNT INT8 weights = 2048b).
        assign mac_weight_bus = weight_rd_data[MAC_COUNT*WGT_W-1:0];
        assign mac_tap_mask      = 4'b0001;
        assign mac_act_bytes_ext = 24'd0;
        /* verilator lint_off UNUSED */
        wire _unused_kpar_skel = &{1'b0, ag_k_tap_mask};
        /* verilator lint_on UNUSED */
    end else begin : g_ktap_kpar
        // subword + mask pipes (2-cycle, mirroring ag_weight_rd_en_d/_d2).
        reg [1:0] wsub_d1, wsub_d2;
        reg [3:0] ktap_d1, ktap_d2;
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                wsub_d1 <= 2'd0;    wsub_d2 <= 2'd0;
                ktap_d1 <= 4'b0001; ktap_d2 <= 4'b0001;
            end else begin
                wsub_d1 <= ag_weight_rd_addr[1:0]; wsub_d2 <= wsub_d1;
                ktap_d1 <= ag_k_tap_mask;          ktap_d2 <= ktap_d1;
            end
        end
        // tap0 = subword-selected old word (slice 0 for aligned fast groups).
        assign mac_weight_bus[MAC_COUNT*WGT_W-1:0] =
            weight_rd_data[wsub_d2*(MAC_COUNT*WGT_W) +: MAC_COUNT*WGT_W];
        assign mac_weight_bus[1*(MAC_COUNT*WGT_W) +: MAC_COUNT*WGT_W] =
            weight_rd_data[1*(MAC_COUNT*WGT_W) +: MAC_COUNT*WGT_W];
        assign mac_weight_bus[2*(MAC_COUNT*WGT_W) +: MAC_COUNT*WGT_W] =
            weight_rd_data[2*(MAC_COUNT*WGT_W) +: MAC_COUNT*WGT_W];
        assign mac_weight_bus[3*(MAC_COUNT*WGT_W) +: MAC_COUNT*WGT_W] =
            weight_rd_data[3*(MAC_COUNT*WGT_W) +: MAC_COUNT*WGT_W];
        assign mac_tap_mask = ktap_d2;
        // dense taps 1..3 act bytes: consecutive ic bytes of the HELD act
        // word. Fast groups are 4-aligned (idx%4==0 and 256%4==0) so idx+3
        // never crosses the word; the +j adds use 8-bit WRAP intermediates
        // so a serial-mode idx=255 stays an in-range (masked-dead) select.
        wire [7:0] kidx1 = ag_act_in_ic_byte_idx_d2 + 8'd1;
        wire [7:0] kidx2 = ag_act_in_ic_byte_idx_d2 + 8'd2;
        wire [7:0] kidx3 = ag_act_in_ic_byte_idx_d2 + 8'd3;
        assign mac_act_bytes_ext[7:0]   = act_in_rd_data_d[kidx1*ACT_W +: ACT_W];
        assign mac_act_bytes_ext[15:8]  = act_in_rd_data_d[kidx2*ACT_W +: ACT_W];
        assign mac_act_bytes_ext[23:16] = act_in_rd_data_d[kidx3*ACT_W +: ACT_W];
    end endgenerate

""", "ktap datapath generate")

    patch(SKEL, """    address_generator u_address_generator (
""", """    address_generator #(.K_PAR(K_PAR)) u_address_generator (
""", "AG inst param")

    patch(SKEL, """        .mac_done              (ag_mac_done),
        .pixel_done            (ag_pixel_done)
    );
""", """        .mac_done              (ag_mac_done),
        .pixel_done            (ag_pixel_done),
        .k_tap_mask            (ag_k_tap_mask)
    );
""", "AG inst mask port")

    patch(SKEL, """    mac_array #(.WGT_W(WGT_W)) u_mac_array (
""", """    mac_array #(.WGT_W(WGT_W), .K_PAR(K_PAR)) u_mac_array (
""", "mac inst param")

    patch(SKEL, """        .act_byte      (mac_act_byte),
        .weight_bus    (mac_weight_bus),
""", """        .act_byte      (mac_act_byte),
        .weight_bus    (mac_weight_bus),
        // [KPAR4] taps 1..3 act bytes + per-tap mask (legacy-inert ties).
        .act_bytes_ext (mac_act_bytes_ext),
        .tap_mask      (mac_tap_mask),
""", "mac inst ext ports")

    # ---- standalone-parse stubs ----
    patch(SKEL, """module mac_array (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         mac_clear,
    input  wire         mac_valid_in,
    input  wire [7:0]   act_byte,
    input  wire [2047:0] weight_bus,
    input  wire         dw_mode,      // [DW-ENGINE P1]
    input  wire [2047:0] act_word,    // [DW-ENGINE P1]
""", """module mac_array #(
    parameter integer WGT_W = 4,
    parameter integer K_PAR = 1
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         mac_clear,
    input  wire         mac_valid_in,
    input  wire [7:0]   act_byte,
    input  wire [K_PAR*256*WGT_W-1:0] weight_bus,
    input  wire [23:0]  act_bytes_ext, // [KPAR4]
    input  wire [3:0]   tap_mask,      // [KPAR4]
    input  wire         dw_mode,      // [DW-ENGINE P1]
    input  wire [2047:0] act_word,    // [DW-ENGINE P1]
""", "mac stub ports")

    patch(SKEL, """module address_generator (
    input  wire         clk,
""", """module address_generator #(
    parameter integer K_PAR = 1
) (
    input  wire         clk,
""", "AG stub param")

    patch(SKEL, """    output wire [15:0]  k_index,
    output wire         mac_done,
    output wire         pixel_done
);
""", """    output wire [15:0]  k_index,
    output wire         mac_done,
    output wire         pixel_done,
    output wire [3:0]   k_tap_mask     // [KPAR4]
);
""", "AG stub mask port")

    patch(SKEL, """    assign mac_done           = 1'b1;
    assign pixel_done         = 1'b1;
endmodule
""", """    assign mac_done           = 1'b1;
    assign pixel_done         = 1'b1;
    assign k_tap_mask         = 4'b0001;   // [KPAR4]
endmodule
""", "AG stub mask assign")


# ============================================================================
# 4. nn2rtl_top_engine.v (MBV2-only)
# ============================================================================
def patch_top() -> None:
    patch(TOP, """    wire [2047:0]              engine_weight_rd_data;
""", """    wire [8191:0]              engine_weight_rd_data;  // [KPAR4] 4 tap-major 2048b words per (group) line
""", "weight bus width")

    patch(TOP, """    wire [14:0] weight_bank_rd_addr = engine_weight_rd_addr[14:0];  // [FC-ENGINE] 18533 > 2^14
""", """    wire [12:0] weight_bank_rd_addr = engine_weight_rd_addr[12:0];  // [KPAR4] GROUP-addressed: 4634 wide lines (engine exports old>>2)
""", "bank addr slice")

    text = TOP.read_text(encoding="utf-8")
    old_wires = "\n".join(f"    wire [287:0] uram_bank{b}_rd_data;" for b in range(8))
    new_wires = "\n".join(f"    wire [1151:0] uram_bank{b}_rd_data;  // [KPAR4] 4 x 288b tap-major"
                          for b in range(8))
    patch(TOP, old_wires + "\n", new_wires + "\n", "bank data wires")

    patch(TOP, """    // MAC bus = concat of the low 256 bits of each bank (bank 0 lowest).
    assign engine_weight_rd_data = {uram_bank7_rd_data[255:0],
        uram_bank6_rd_data[255:0],
        uram_bank5_rd_data[255:0],
        uram_bank4_rd_data[255:0],
        uram_bank3_rd_data[255:0],
        uram_bank2_rd_data[255:0],
        uram_bank1_rd_data[255:0],
        uram_bank0_rd_data[255:0]};
""", """    // [KPAR4] MAC bus = 4 tap-major 2048b words. Each repacked bank line
    // (1152b) packs the 4 old 288b words tap-major (old word 4g+j at bits
    // [j*288 +: 288] of line g — proof: scripts/repack_mbv2_kpar4_banks.py
    // P1/P2/P3). Within a tap, lanes keep the original bank-major order
    // (bank b = lanes 32b..32b+31, low 256 of the 288b word, bank0 lowest).
    genvar kp_tap;
    generate for (kp_tap = 0; kp_tap < 4; kp_tap = kp_tap + 1) begin : g_kpar_wbus
        assign engine_weight_rd_data[kp_tap*2048 +: 2048] = {
            uram_bank7_rd_data[kp_tap*288 +: 256],
            uram_bank6_rd_data[kp_tap*288 +: 256],
            uram_bank5_rd_data[kp_tap*288 +: 256],
            uram_bank4_rd_data[kp_tap*288 +: 256],
            uram_bank3_rd_data[kp_tap*288 +: 256],
            uram_bank2_rd_data[kp_tap*288 +: 256],
            uram_bank1_rd_data[kp_tap*288 +: 256],
            uram_bank0_rd_data[kp_tap*288 +: 256]};
    end endgenerate
""", "tap-major bus concat")

    for b in range(8):
        patch(TOP, f"""    uram_weight_bank #(
        .DEPTH(18533),
        .ADDR_W(15),
        .MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank{b}.mem")
""", f"""    uram_weight_bank #(
        .DEPTH(4634),           // [KPAR4] ceil(18533/4) wide lines
        .ADDR_W(13),
        .WORD_W(1152),          // [KPAR4] 4 x 288b tap-major
        .MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank{b}_kp4.mem")
""", f"bank{b} params")

    patch(TOP, """module uram_weight_bank #(
    parameter integer DEPTH         = 1024,
    parameter integer ADDR_W        = 17,
    parameter         MEM_INIT_FILE = ""
) (
    input  wire                    clk,
    input  wire [ADDR_W-1:0]       rd_addr,
    output wire [287:0]            rd_data,
    input  wire                    rd_en
);
""", """module uram_weight_bank #(
    parameter integer DEPTH         = 1024,
    parameter integer ADDR_W        = 17,
    parameter integer WORD_W        = 288,   // [KPAR4] 1152 = 4 x 288b tap-major lines
    parameter         MEM_INIT_FILE = ""
) (
    input  wire                    clk,
    input  wire [ADDR_W-1:0]       rd_addr,
    output wire [WORD_W-1:0]       rd_data,
    input  wire                    rd_en
);
""", "bank module ports")

    patch(TOP, """        .MEMORY_PRIMITIVE("ultra"),
        .MEMORY_SIZE(DEPTH * 288),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_A(288),
""", """        .MEMORY_PRIMITIVE("ultra"),
        .MEMORY_SIZE(DEPTH * WORD_W),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_A(WORD_W),
""", "bank xpm widths")

    patch(TOP, """    reg [287:0] mem [0:DEPTH-1];
    initial begin
        if (MEM_INIT_FILE != "") $readmemh(MEM_INIT_FILE, mem);
    end
    reg [287:0] rd_data_r1, rd_data_r2;
""", """    reg [WORD_W-1:0] mem [0:DEPTH-1];
    initial begin
        if (MEM_INIT_FILE != "") $readmemh(MEM_INIT_FILE, mem);
    end
    reg [WORD_W-1:0] rd_data_r1, rd_data_r2;
""", "bank behavioral widths")

    patch(TOP, """    shared_engine #(
        .WGT_W(8),
        .URAM_DATA_W(2048),
""", """    shared_engine #(
        .WGT_W(8),
        .URAM_DATA_W(8192),     // [KPAR4] 4 tap-major 2048b words per line
        .K_PAR(4),              // [KPAR4] 4 taps/cycle/lane (dense 1x1 fast walk; DW+FC serial fallback)
""", "engine K_PAR=4")


# ============================================================================
# 5. tb/engine_iso_wrap_mbv2.v — `KPAR4 build variant
# ============================================================================
def patch_iso() -> None:
    patch(ISO, """    wire [21:0]   eng_weight_rd_addr;
    wire          eng_weight_rd_en;
    wire [2047:0] eng_weight_rd_data;     // 2048b = 256 INT8 weights (mbv2)
""", """    wire [21:0]   eng_weight_rd_addr;
    wire          eng_weight_rd_en;
`ifdef KPAR4
    wire [8191:0] eng_weight_rd_data;     // [KPAR4] 4 tap-major 2048b words
`else
    wire [2047:0] eng_weight_rd_data;     // 2048b = 256 INT8 weights (mbv2)
`endif
""", "iso weight bus width")

    patch(ISO, """    wire [16:0]  wbank_addr = eng_weight_rd_addr[16:0];
    wire [287:0] b0,b1,b2,b3,b4,b5,b6,b7;
    assign eng_weight_rd_data = {b7[255:0],b6[255:0],b5[255:0],b4[255:0],
                                 b3[255:0],b2[255:0],b1[255:0],b0[255:0]};
    iso_uram_bank #(.MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank0.mem")) u0(.clk(clk),.rd_addr(wbank_addr),.rd_data(b0),.rd_en(eng_weight_rd_en));
    iso_uram_bank #(.MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank1.mem")) u1(.clk(clk),.rd_addr(wbank_addr),.rd_data(b1),.rd_en(eng_weight_rd_en));
    iso_uram_bank #(.MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank2.mem")) u2(.clk(clk),.rd_addr(wbank_addr),.rd_data(b2),.rd_en(eng_weight_rd_en));
    iso_uram_bank #(.MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank3.mem")) u3(.clk(clk),.rd_addr(wbank_addr),.rd_data(b3),.rd_en(eng_weight_rd_en));
    iso_uram_bank #(.MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank4.mem")) u4(.clk(clk),.rd_addr(wbank_addr),.rd_data(b4),.rd_en(eng_weight_rd_en));
    iso_uram_bank #(.MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank5.mem")) u5(.clk(clk),.rd_addr(wbank_addr),.rd_data(b5),.rd_en(eng_weight_rd_en));
    iso_uram_bank #(.MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank6.mem")) u6(.clk(clk),.rd_addr(wbank_addr),.rd_data(b6),.rd_en(eng_weight_rd_en));
    iso_uram_bank #(.MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank7.mem")) u7(.clk(clk),.rd_addr(wbank_addr),.rd_data(b7),.rd_en(eng_weight_rd_en));
""", """    wire [16:0]  wbank_addr = eng_weight_rd_addr[16:0];
`ifdef KPAR4
    // [KPAR4] repacked 4-taps-per-line banks (group-addressed by the engine).
    wire [1151:0] b0,b1,b2,b3,b4,b5,b6,b7;
    genvar kp_tap;
    generate for (kp_tap = 0; kp_tap < 4; kp_tap = kp_tap + 1) begin : g_kpar_wbus
        assign eng_weight_rd_data[kp_tap*2048 +: 2048] = {
            b7[kp_tap*288 +: 256], b6[kp_tap*288 +: 256],
            b5[kp_tap*288 +: 256], b4[kp_tap*288 +: 256],
            b3[kp_tap*288 +: 256], b2[kp_tap*288 +: 256],
            b1[kp_tap*288 +: 256], b0[kp_tap*288 +: 256]};
    end endgenerate
    iso_uram_bank #(.WORD_W(1152), .MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank0_kp4.mem")) u0(.clk(clk),.rd_addr(wbank_addr),.rd_data(b0),.rd_en(eng_weight_rd_en));
    iso_uram_bank #(.WORD_W(1152), .MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank1_kp4.mem")) u1(.clk(clk),.rd_addr(wbank_addr),.rd_data(b1),.rd_en(eng_weight_rd_en));
    iso_uram_bank #(.WORD_W(1152), .MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank2_kp4.mem")) u2(.clk(clk),.rd_addr(wbank_addr),.rd_data(b2),.rd_en(eng_weight_rd_en));
    iso_uram_bank #(.WORD_W(1152), .MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank3_kp4.mem")) u3(.clk(clk),.rd_addr(wbank_addr),.rd_data(b3),.rd_en(eng_weight_rd_en));
    iso_uram_bank #(.WORD_W(1152), .MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank4_kp4.mem")) u4(.clk(clk),.rd_addr(wbank_addr),.rd_data(b4),.rd_en(eng_weight_rd_en));
    iso_uram_bank #(.WORD_W(1152), .MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank5_kp4.mem")) u5(.clk(clk),.rd_addr(wbank_addr),.rd_data(b5),.rd_en(eng_weight_rd_en));
    iso_uram_bank #(.WORD_W(1152), .MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank6_kp4.mem")) u6(.clk(clk),.rd_addr(wbank_addr),.rd_data(b6),.rd_en(eng_weight_rd_en));
    iso_uram_bank #(.WORD_W(1152), .MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank7_kp4.mem")) u7(.clk(clk),.rd_addr(wbank_addr),.rd_data(b7),.rd_en(eng_weight_rd_en));
`else
    wire [287:0] b0,b1,b2,b3,b4,b5,b6,b7;
    assign eng_weight_rd_data = {b7[255:0],b6[255:0],b5[255:0],b4[255:0],
                                 b3[255:0],b2[255:0],b1[255:0],b0[255:0]};
    iso_uram_bank #(.MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank0.mem")) u0(.clk(clk),.rd_addr(wbank_addr),.rd_data(b0),.rd_en(eng_weight_rd_en));
    iso_uram_bank #(.MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank1.mem")) u1(.clk(clk),.rd_addr(wbank_addr),.rd_data(b1),.rd_en(eng_weight_rd_en));
    iso_uram_bank #(.MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank2.mem")) u2(.clk(clk),.rd_addr(wbank_addr),.rd_data(b2),.rd_en(eng_weight_rd_en));
    iso_uram_bank #(.MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank3.mem")) u3(.clk(clk),.rd_addr(wbank_addr),.rd_data(b3),.rd_en(eng_weight_rd_en));
    iso_uram_bank #(.MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank4.mem")) u4(.clk(clk),.rd_addr(wbank_addr),.rd_data(b4),.rd_en(eng_weight_rd_en));
    iso_uram_bank #(.MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank5.mem")) u5(.clk(clk),.rd_addr(wbank_addr),.rd_data(b5),.rd_en(eng_weight_rd_en));
    iso_uram_bank #(.MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank6.mem")) u6(.clk(clk),.rd_addr(wbank_addr),.rd_data(b6),.rd_en(eng_weight_rd_en));
    iso_uram_bank #(.MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank7.mem")) u7(.clk(clk),.rd_addr(wbank_addr),.rd_data(b7),.rd_en(eng_weight_rd_en));
`endif
""", "iso kpar banks")

    patch(ISO, """    shared_engine #(
        .WGT_W(8),
        .URAM_DATA_W(2048),
""", """    shared_engine #(
        .WGT_W(8),
`ifdef KPAR4
        .URAM_DATA_W(8192),     // [KPAR4] 4 tap-major words per line
        .K_PAR(4),
`else
        .URAM_DATA_W(2048),
`endif
""", "iso engine K_PAR")

    patch(ISO, """module iso_uram_bank #(parameter MEM_INIT_FILE="") (
    input wire clk, input wire [16:0] rd_addr,
    output wire [287:0] rd_data, input wire rd_en);
    reg [287:0] mem [0:131071];
    initial if (MEM_INIT_FILE!="") $readmemh(MEM_INIT_FILE, mem);
    reg [287:0] r1, r2;
""", """module iso_uram_bank #(parameter MEM_INIT_FILE="", parameter integer WORD_W=288) (
    input wire clk, input wire [16:0] rd_addr,
    output wire [WORD_W-1:0] rd_data, input wire rd_en);
    reg [WORD_W-1:0] mem [0:131071];
    initial if (MEM_INIT_FILE!="") $readmemh(MEM_INIT_FILE, mem);
    reg [WORD_W-1:0] r1, r2;
""", "iso bank WORD_W")


def main() -> int:
    check = "--check" in sys.argv
    if check:
        for p in (MAC, AG, SKEL, TOP, ISO):
            t = p.read_text(encoding="utf-8")
            print(f"{p.name}: {'APPLIED' if 'KPAR4' in t else 'not applied'}")
        return 0
    print("[kpar4] patching mac_array.v ...");           patch_mac()
    print("[kpar4] patching address_generator.v ...");   patch_ag()
    print("[kpar4] patching shared_engine_skeleton.v ..."); patch_skel()
    print("[kpar4] patching nn2rtl_top_engine.v ...");    patch_top()
    print("[kpar4] patching engine_iso_wrap_mbv2.v ..."); patch_iso()
    print("[kpar4] DONE. Next: scripts/repack_mbv2_kpar4_banks.py (if not yet run), "
          "lint, engine-ISO (legacy + -DKPAR4), 8/8 e2e, hazard checker.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
