#!/usr/bin/env python3
"""apply_mbv2_fc_engine.py — MBV2 "FC-ON-ENGINE" top + scheduler surgery
(extends DW-ENGINE EXT, commit 2937dbd).

Moves node_linear (the FC/Gemm classifier, M=1000 x K=1280, ~1.28M serial
cycles as an UNOVERLAPPED tail at frame end) onto the shared engine as ONE
dense dispatch — dispatch 46 of 47 (APPENDED after conv_912; no SLOT
renumbering):

  module        IC    OC  geometry  oc_passes  k_total  act_in  act_out  wgt    b/s
  node_linear  1280  1000  1x1 px=1     4        1280    25088   25093   13413   87

Engine config: a 1x1 conv over a 1x1 "image" (KH=KW=1, S=1, P=0, IH=IW=OH=
OW=1). The dense address_generator walk reads act word 24264+(k>>8), byte
k&255 — exactly node_mean's 5-beat channel layout (beat t = channels
256t..256t+255) — and weight word 13413 + oc_pass*1280 + k. Requant = the
engine's per-OC constant-shift path with mult' = 4071<<(23-20) = 32568,
PROVEN byte-identical to node_linear.v's (x*4071 + 2^19)>>>20 (algebraic +
empirical 8-vector proof: docs/agent_tasks/FC_ENGINE_ANALYSIS.md; the engine
output path applies NO relu — clamp [-128,127] only — matching the
signed, un-relu'd logits).

Surgery:
  * u_node_linear spatial instance DELETED (file stays on disk, unused).
  * node_mean's 5-beat output stream -> NEW input loader (the e2e-proven
    stream_to_act_bram_bridge g_w_eq BUS_W=2048 branch) filling act words
    [24264, 24269).
  * NEW engine_output_bridge (OUT_KIND=2 flat-gather, OC=1000, POSITIONS=1,
    DATA_W=8000): gathers the 4 oc_pass beats and re-drives the old
    node_linear_valid_out/data_out nets with ONE 8000b beat = logits 0..999
    (dead lanes 1000..1023 never emitted) — output_serializer/m_axis are
    byte-identical downstream. Consumer handshake follows the established
    bridge convention: ready_out = (ser_ready_out & spatial_run), serializer
    valid_in gains '& spatial_run'.
  * Engine act_out writes go to scratch [25093, 25097) (never read).
  * Act regions [25088, 25097) sit ABOVE the GLOBAL act-mem maximum ever
    used (25088, the frame-start d0/d1 region top) and below ACT_DEPTH
    25600 -> STRICTLY disjoint from EVERYTHING (no lifetime argument
    needed, unlike the P1/EXT DW regions). Proof:
    scripts/check_mbv2_act_region_hazards_fc.py.
  * Weight banks 13413 -> 18533 words: DEPTH/ADDR_W bumped to 18533/15 and
    the bank address slice widened [13:0] -> [14:0] (18533 > 2^14 — the
    one NEW width anchor class this extension introduces).
  * Scheduler: 22 ROMs gain row 6'd46 (append-only; rows 0..45 untouched),
    LAST_DISPATCH 45 -> 46, depthwise_rom UNCHANGED (FC is dense).
  * NUM_DISPATCHES 46 -> 47 on all 46 pre-existing bridges + 47 on the new
    one; all_loaded[46]/all_drain[46] rows rebound from 1'b1.

Map bases must already exist — run scripts/extend_mbv2_engine_maps_fc.py
FIRST (asserted: banks 18533 lines, bias/scale 91).

Usage:
  python scripts/apply_mbv2_fc_engine.py

Anchor-asserted + DECLARATIVE: backups <file>.prefc capture the DW-EXT state
once; a re-run restores from .prefc and re-applies, so the script is safely
re-runnable / bisectable.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOP = REPO / "output" / "mobilenet-v2" / "rtl" / "nn2rtl_top_engine.v"
SCHED = REPO / "output" / "mobilenet-v2" / "rtl" / "nn2rtl_scheduler.v"
WDIR = REPO / "output" / "mobilenet-v2" / "weights"

MARK = "[FC-ENGINE]"
EXTMARK = "[DW-ENGINE EXT]"

NDISP = 47
SLOT = 46
BANK_DEPTH_OLD = 13413
BANK_DEPTH_NEW = 18533
BS_WORDS_NEW = 91
# FC act regions sit ABOVE the GLOBAL act-mem maximum ever used (25088 — the
# frame-start d0-write/d1 in-place region [12544,+12544) is the prior max;
# the DW regions topped out at 24264 INSIDE it under a lifetime argument).
# [25088, 25600) is genuinely never touched by anything -> the FC regions are
# STRICTLY disjoint from every region with NO lifetime argument at all.
ACT_IN_BASE = 25088    # 5 words (GAP vector, 5 x 2048b)
ACT_OUT_BASE = 25093   # 4 words (engine scratch, never read); max used 25097 < 25600


def die(msg: str) -> None:
    print(f"[fc-engine] FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def rep(text: str, old: str, new: str, what: str, count: int = 1) -> str:
    n = text.count(old)
    if n != count:
        die(f"anchor '{what}': expected {count} occurrence(s), found {n}")
    return text.replace(old, new)


def excise_block(text: str, start_marker: str, end_marker: str, new: str,
                 what: str) -> str:
    s = text.find(start_marker)
    if s < 0:
        die(f"excise '{what}': start marker not found")
    e = text.find(end_marker, s + len(start_marker))
    if e < 0:
        die(f"excise '{what}': end marker not found")
    return text[:s] + new + text[e + len(end_marker):]


# ============================================================================
# Pre-flight
# ============================================================================
def preflight() -> None:
    for b in range(8):
        n = sum(1 for _ in (WDIR / f"uram_weights_bank{b}.mem").open())
        if n != BANK_DEPTH_NEW:
            die(f"bank{b} has {n} lines (need {BANK_DEPTH_NEW}) — run extend_mbv2_engine_maps_fc.py")
    for f in ["bias.mem", "scale.mem"]:
        n = sum(1 for _ in (WDIR / f).open())
        if n != BS_WORDS_NEW:
            die(f"{f} has {n} lines (need {BS_WORDS_NEW}) — run extend_mbv2_engine_maps_fc.py")
    # node_linear wrapper geometry/scale must match the proven identity
    t = (REPO / "output/mobilenet-v2/rtl/node_linear.v").read_text(encoding="utf-8")
    lp = dict(re.findall(r"localparam integer (\w+)\s*=\s*(\d+);", t))
    for k, v in [("K", 1280), ("M", 1000), ("SCALE_MULT", 4071), ("SCALE_SHIFT", 20)]:
        if k not in lp or int(lp[k]) != v:
            die(f"node_linear.v {k}={lp.get(k)} != expected {v} (requant identity proof void)")
    print("[fc-engine] preflight OK (extended maps + node_linear K/M/MULT/SHIFT identity anchors)")


# ============================================================================
# TOP surgery
# ============================================================================
def patch_top() -> None:
    text = TOP.read_text(encoding="utf-8")

    # ---- header bookkeeping ----
    text = rep(text,
        "// Layers total: 99, spatial: 53, engine-dispatched: 46 ([DW-ENGINE P1] 896/902/908 + "
        "[DW-ENGINE EXT] 824/836/842/854/860/866/872/878/884 depthwise dispatches), "
        "residual adds: 10, projection convs: 11.",
        f"// Layers total: 99, spatial: 52, engine-dispatched: {NDISP} ([DW-ENGINE P1] 896/902/908 + "
        "[DW-ENGINE EXT] 824/836/842/854/860/866/872/878/884 depthwise dispatches + "
        f"{MARK} node_linear dense dispatch {SLOT}), residual adds: 10, projection convs: 11.",
        "header layer counts")

    # ---- wire decls: node_linear_ready_in is gone (bridge-fed output) ----
    text = rep(text,
        "    wire node_linear_valid_out;\n"
        "    wire [7999:0] node_linear_data_out;\n"
        "    wire node_linear_ready_in;\n",
        "    wire node_linear_valid_out;   // driven by engine_output_bridge SLOT 46 (" + MARK + ")\n"
        "    wire [7999:0] node_linear_data_out;\n",
        "node_linear wire decls")

    # ---- node_mean retarget: its 5-beat GAP stream now fills the FC loader ----
    text = rep(text,
        "        .out_ready_in(node_linear_ready_in & spatial_run),\n"
        "        .valid_out(node_mean_valid_out),",
        f"        // {MARK} node_linear is engine dispatch {SLOT}: node_mean's 5-beat\n"
        "        // GAP stream fills the FC input loader (u_ldr_node_linear).\n"
        "        .out_ready_in(ldr_fc_in_ready & spatial_run),\n"
        "        .valid_out(node_mean_valid_out),",
        "node_mean retarget")

    # ---- delete the serial node_linear instance ----
    text = excise_block(text,
        "    node_linear #(.ENABLE_BACKPRESSURE(1)) u_node_linear (",
        "\n    );\n",
        f"    // node_linear: engine-dispatched ({MARK} dense dispatch {SLOT}; "
        f"data_out driven by shared_engine via engine_output_bridge SLOT {SLOT})\n",
        "u_node_linear instance")

    # ---- input loader (inserted before the arbiter) ----
    loader = f'''
    // {MARK} FC input loader: node_mean's 5-beat GAP stream (beat t = channels
    // 256t..256t+255) -> act words {ACT_IN_BASE}..{ACT_IN_BASE + 4} for the node_linear
    // dense engine dispatch (the engine's 1x1 walk reads word {ACT_IN_BASE}+(k>>8),
    // byte k&255 — the exact same layout).
    wire        ldr_fc_wr_req;
    wire        ldr_fc_wr_grant;
    wire [14:0] ldr_fc_wr_addr;
    wire [2047:0] ldr_fc_wr_data;
    wire        ldr_fc_loaded;
    wire        ldr_fc_in_ready;
    stream_to_act_bram_bridge #(
        .BUS_W(2048),
        .BRAM_BASE_ADDR({ACT_IN_BASE}),
        .TOTAL_BRAM_WORDS(5)
    ) u_ldr_node_linear (
        .clk(clk), .rst_n(rst_n),
        .in_valid(node_mean_valid_out & spatial_run),
        .in_data(node_mean_data_out),
        .wr_req(ldr_fc_wr_req),
        .wr_grant(ldr_fc_wr_grant),
        .wr_addr(ldr_fc_wr_addr),
        .wr_data(ldr_fc_wr_data),
        .loaded(ldr_fc_loaded),
        .in_ready(ldr_fc_in_ready)
    );
'''
    text = rep(text,
        "    // ----- act BRAM write arbiter: engine priority, then bridges -----",
        loader + "\n    // ----- act BRAM write arbiter: engine priority, then bridges -----",
        "loader insertion")

    # ---- arbiter: grant + en/addr/data mux terms (after the EXT DW set) ----
    m = re.search(r"    assign ldr_dw884_wr_grant = ldr_dw884_wr_req & ~\(([^)]+)\);\n", text)
    if not m:
        die("ldr_dw884 grant line not found")
    accum = m.group(1) + " | ldr_dw884_wr_req"
    grant = (f"    // {MARK} FC input loader (lowest priority; frame-end only, nothing\n"
             "    // else writes the act BRAM while node_mean drains).\n"
             f"    assign ldr_fc_wr_grant = ldr_fc_wr_req & ~({accum});\n")
    text = rep(text, m.group(0), m.group(0) + grant, "arbiter grant")
    text = rep(text,
        " | ldr_dw884_wr_req;\n",
        " | ldr_dw884_wr_req | ldr_fc_wr_req;\n",
        "act_wr_en_final")
    text = rep(text,
        "ldr_dw884_wr_req ? ldr_dw884_wr_addr : 15'd0;",
        "ldr_dw884_wr_req ? ldr_dw884_wr_addr : ldr_fc_wr_req ? ldr_fc_wr_addr : 15'd0;",
        "act_wr_addr_final mux")
    text = rep(text,
        "ldr_dw884_wr_req ? ldr_dw884_wr_data : 2048'd0;",
        "ldr_dw884_wr_req ? ldr_dw884_wr_data : ldr_fc_wr_req ? ldr_fc_wr_data : 2048'd0;",
        "act_wr_data_final mux")

    # ---- all_loaded / all_drain row 46 (rows 0..45 untouched) ----
    text = rep(text, "    assign all_loaded[46] = 1'b1;",
               f"    assign all_loaded[46] = ldr_fc_loaded;  // {MARK}",
               "all_loaded[46]")
    text = rep(text, "    assign all_drain[46] = 1'b1;",
               f"    assign all_drain[46] = u_engine_out_node_linear_drain_complete;  // {MARK}",
               "all_drain[46]")

    # ---- output bridge (inserted after the conv_912 bridge) ----
    anchor = ("        .drain_complete(u_engine_out_node_conv_912_drain_complete)\n"
              "    );\n")
    bridge = f'''
    wire u_engine_out_node_linear_fifo_ready;
    wire u_engine_out_node_linear_drain_complete;
    // {MARK} node_linear dense dispatch {SLOT}: flat-gather the 4 oc_pass beats
    // (lane L of pass p = logit 256p+L) into ONE 8000b word = logits 0..999
    // (dead lanes 1000..1023 in beat 3's high bytes are never emitted), re-driving
    // the old node_linear output nets — output_serializer/m_axis byte-identical.
    // Engine requant == node_linear requant (mult'=32568 slot; identity proof in
    // docs/agent_tasks/FC_ENGINE_ANALYSIS.md), and the engine applies NO relu.
    engine_output_bridge #(
        .SLOT({SLOT}),
        .ACT_W(2048),
        .DATA_W(8000),
        .EXPECTED_BEATS(4),
        .NUM_DISPATCHES({NDISP}),
        .OC(1000), .OUT_KIND(2), .POSITIONS(1)
    ) u_engine_out_node_linear (
        .clk(clk), .rst_n(rst_n),
        .start(sched_engine_output_ready),
        .fifo_out_valid(eofifo_out_valid),
        .fifo_out_data(eofifo_out_data),
        .fifo_out_ready(u_engine_out_node_linear_fifo_ready),
        .ready_out((ser_ready_out & spatial_run)),
        .valid_out(node_linear_valid_out),
        .data_out(node_linear_data_out),
        .drain_complete(u_engine_out_node_linear_drain_complete)
    );
'''
    text = rep(text, anchor, anchor + bridge, "bridge insertion")

    # ---- eofifo_out_ready: one more term ----
    text = rep(text,
        " | u_engine_out_node_conv_884_fifo_ready;",
        " | u_engine_out_node_conv_884_fifo_ready | u_engine_out_node_linear_fifo_ready;",
        "eofifo_out_ready")

    # ---- serializer: bridge-fed valid gating (matches ready_out gating) ----
    text = rep(text,
        "        .valid_in(node_linear_valid_out),",
        f"        .valid_in(node_linear_valid_out & spatial_run),  // {MARK} bridge-fed",
        "serializer valid_in")

    # ---- NUM_DISPATCHES on the 46 pre-existing bridges ----
    text = rep(text, ".NUM_DISPATCHES(46)", f".NUM_DISPATCHES({NDISP})",
               "NUM_DISPATCHES", count=46)

    # ---- weight banks: depth 13413 -> 18533, ADDR 14 -> 15 bits ----
    text = rep(text,
        "    // Total MAC cycles = 13413; per-bank depth = 13413. ([DW-ENGINE EXT] +153 stride-1 DW\n"
        "    // words appended after the P1 trio: 824@13260 836@13269 842@13278 854@13287\n"
        "    // 860@13305 866@13323 872@13341 878@13359 884@13386; oc_passes x 9 taps each)",
        f"    // Total MAC cycles = {BANK_DEPTH_NEW}; per-bank depth = {BANK_DEPTH_NEW}. ({MARK} +5120 dense FC\n"
        "    // words appended after the DW-EXT set: node_linear@13413 = 4 oc_passes x 1280 taps.\n"
        "    // Depth now exceeds 2^14 -> bank ADDR_W 14->15 and the rd_addr slice [13:0]->[14:0].)",
        "bank depth comment")
    text = rep(text,
        "    // Address path: engine_weight_rd_addr[13:0] -> each bank's rd_addr.",
        "    // Address path: engine_weight_rd_addr[14:0] -> each bank's rd_addr.",
        "weight_bank_rd_addr comment")
    text = rep(text,
        "    wire [13:0] weight_bank_rd_addr = engine_weight_rd_addr[13:0];",
        f"    wire [14:0] weight_bank_rd_addr = engine_weight_rd_addr[14:0];  // {MARK} 18533 > 2^14",
        "weight_bank_rd_addr width")
    text = rep(text, f".DEPTH({BANK_DEPTH_OLD}),", f".DEPTH({BANK_DEPTH_NEW}),",
               "bank DEPTH", count=8)
    text = rep(text, ".ADDR_W(14),", ".ADDR_W(15),", "bank ADDR_W", count=8)

    verify_top(text)
    TOP.write_text(text, encoding="utf-8", newline="\n")
    print(f"[fc-engine] top patched + verified ({NDISP} dispatches)")


def verify_top(text: str) -> None:
    checks = [
        (f"NUM_DISPATCHES({NDISP})", text.count(f".NUM_DISPATCHES({NDISP})"), NDISP),
        ("NUM_DISPATCHES(46) gone", text.count(".NUM_DISPATCHES(46)"), 0),
        (f"bank DEPTH {BANK_DEPTH_NEW}", text.count(f".DEPTH({BANK_DEPTH_NEW}),"), 8),
        ("bank ADDR_W 15", text.count(".ADDR_W(15),"), 8),  # 8 banks (act_unified_mem's .ADDR_W(15) has no comma)
        ("bank ADDR_W 14 gone", text.count(".ADDR_W(14),"), 0),
        ("spatial node_linear gone", text.count("u_node_linear ("), 0),
        ("node_linear_ready_in gone", text.count("node_linear_ready_in"), 0),
        ("loader", text.count(") u_ldr_node_linear ("), 1),
        ("bridge", text.count("u_engine_out_node_linear_fifo_ready"), 3),
        ("drain wire", text.count("u_engine_out_node_linear_drain_complete"), 3),
        ("loaded row", text.count("assign all_loaded[46] = ldr_fc_loaded;"), 1),
        ("rd_addr slice", text.count("engine_weight_rd_addr[14:0]"), 2),  # comment + wire
        ("old rd_addr slice gone", text.count("engine_weight_rd_addr[13:0]"), 0),
    ]
    for what, got, want in checks:
        if got != want:
            die(f"top verify '{what}': got {got}, want {want}")
    slots = sorted(int(m) for m in re.findall(r"\.SLOT\((\d+)\),", text))
    if slots != list(range(NDISP)):
        die(f"top verify SLOT set: {slots}")
    # every all_loaded/all_drain row 0..45 must be UNCHANGED non-constant rows
    for d in range(46):
        ml = re.search(rf"    assign all_loaded\[{d}\] = (\S+?);", text)
        md = re.search(rf"    assign all_drain\[{d}\] = (\S+?);", text)
        if not ml or not md:
            die(f"all_loaded/all_drain[{d}] missing")
        if d != 28 and (ml.group(1) == "1'b1" or md.group(1) == "1'b1"):
            # (28 = conv_876, legitimately 1'b1-loaded in the EXT baseline)
            die(f"row {d} unexpectedly constant: {ml.group(1)}/{md.group(1)}")
    md = re.search(r"    assign all_drain\[46\] = (\S+?);", text)
    if not md or md.group(1) != "u_engine_out_node_linear_drain_complete":
        die("all_drain[46] not rebound to the FC bridge")
    print(f"[fc-engine] top invariants OK ({NDISP} slots, loader/bridge/banks consistent)")


# ============================================================================
# SCHEDULER surgery (append-only: row 6'd46 on each ROM)
# ============================================================================
ROM_FC_VALUES = {
    "channel_in_rom": "16'd1280",
    "channel_out_rom": "16'd1000",
    "kernel_h_rom": "4'd1", "kernel_w_rom": "4'd1",
    "stride_h_rom": "3'd1", "stride_w_rom": "3'd1",
    "padding_h_rom": "3'd0", "padding_w_rom": "3'd0",
    "input_h_rom": "9'd1", "input_w_rom": "9'd1",
    "output_h_rom": "9'd1", "output_w_rom": "9'd1",
    "weight_base_word_rom": f"20'd{BANK_DEPTH_OLD}",
    "bias_base_word_rom": "16'd87",
    "scale_mult_rom": "32'd0",     # vestigial: requant is per-OC from scale.mem
    "scale_shift_rom": "6'd0",     # vestigial
    "zero_point_rom": "8'd0",
    "input_bank_rom": "3'd0", "output_bank_rom": "3'd0",
    "skip_mask_rom": "6'd0",
    "act_in_base_word_rom": f"16'd{ACT_IN_BASE}",
    "act_out_base_word_rom": f"16'd{ACT_OUT_BASE}",
}


def patch_scheduler() -> None:
    text = SCHED.read_text(encoding="utf-8")

    text = rep(text,
        "// Number of engine dispatches: 46  ([DW-ENGINE P1] + [DW-ENGINE EXT] depthwise @ "
        "824@4, 836@9, 842@12, 854@17, 860@20, 866@23, 872@26, 878@29, 884@32, 896@37, 902@40, 908@43)",
        f"// Number of engine dispatches: {NDISP}  ([DW-ENGINE P1] + [DW-ENGINE EXT] depthwise @ "
        "824@4, 836@9, 842@12, 854@17, 860@20, 866@23, 872@26, 878@29, 884@32, 896@37, 902@40, 908@43"
        f" + {MARK} node_linear dense @ 46)",
        "sched header")

    for name, val in ROM_FC_VALUES.items():
        # append row 46 right after the existing row 45 of THIS rom
        m = re.search(rf"(            6'd45: {name} = [^;]+;\n)(            default:)", text)
        if not m:
            die(f"scheduler ROM {name}: row-45/default anchor not found")
        n_rows = len(re.findall(rf"6'd\d+: {name} = ", text))
        if n_rows != 46:
            die(f"scheduler ROM {name}: {n_rows} rows, want 46 pre-patch")
        text = text[:m.end(1)] + f"            6'd46: {name} = {val};  // {MARK} node_linear\n" \
               + text[m.end(1):]

    text = rep(text,
        "    localparam [5:0] LAST_DISPATCH = 6'd45;  // [DW-ENGINE EXT] 46 dispatches",
        f"    localparam [5:0] LAST_DISPATCH = 6'd46;  // {MARK} {NDISP} dispatches",
        "LAST_DISPATCH")

    verify_sched(text)
    SCHED.write_text(text, encoding="utf-8", newline="\n")
    print(f"[fc-engine] scheduler patched + verified ({NDISP}-entry ROMs)")


def verify_sched(text: str) -> None:
    for name in ROM_FC_VALUES:
        cnt = len(re.findall(rf"6'd\d+: {name} = ", text))
        if cnt != NDISP:
            die(f"sched verify ROM {name}: {cnt} entries, want {NDISP}")
        needle = f"6'd46: {name} = {ROM_FC_VALUES[name]};"
        if text.count(needle) != 1:
            die(f"sched verify: '{needle}' count != 1")
    if text.count(f"LAST_DISPATCH = 6'd46") != 1:
        die("sched verify: LAST_DISPATCH not 46")
    # depthwise_rom must NOT gain a row (FC is dense) — still 12 ones.
    dw_count = len(re.findall(r"depthwise_rom = 1'b1;", text))
    if dw_count != 12:
        die(f"sched verify depthwise_rom: {dw_count} ones, want 12")
    if "6'd46: depthwise_rom" in text:
        die("sched verify: depthwise_rom must not have a row 46")
    # rows 0..45 of a spot-check ROM unchanged (conv_912 row preserved)
    if text.count("6'd45: channel_out_rom = 16'd1280;") != 1:
        die("sched verify: conv_912 row disturbed")
    print(f"[fc-engine] scheduler invariants OK ({NDISP} entries, depthwise rows 12)")


# ============================================================================
# main
# ============================================================================
def main() -> int:
    preflight()

    # declarative: restore the DW-EXT state from .prefc if already applied
    for f in [TOP, SCHED]:
        backup = f.with_suffix(f.suffix + ".prefc")
        cur = f.read_text(encoding="utf-8")
        if MARK in cur:
            if not backup.exists():
                die(f"{f.name} already patched but {backup.name} missing")
            f.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
            print(f"[fc-engine] {f.name}: restored DW-EXT baseline from {backup.name}")
        else:
            if EXTMARK not in cur:
                die(f"{f.name} lacks {EXTMARK} — base must be the DW-ENGINE EXT state")
            if not backup.exists():
                backup.write_text(cur, encoding="utf-8", newline="\n")
                print(f"[fc-engine] {f.name}: saved DW-EXT baseline to {backup.name}")

    patch_top()
    patch_scheduler()
    print("[fc-engine] DONE — next: verilator lint, "
          "scripts/check_mbv2_act_region_hazards_fc.py, engine-ISO (gen_dw_engine_iso_cfg.py linear), "
          "bash scripts/run_mbv2_e2e_parallel.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
