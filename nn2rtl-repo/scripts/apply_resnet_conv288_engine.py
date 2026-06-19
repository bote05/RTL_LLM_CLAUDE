#!/usr/bin/env python3
"""[CONV288-ENGINE 2026-06-12] Move node_conv_288 from SPATIAL fabric onto the
SHARED ENGINE as dispatch 10 (18 dispatches total), mirroring the K5 pattern
(scripts/apply_k5_engine_dispatch.py, conv_284/292/298, byte-exact green).

WHY (route forensics, failed_route_final_c14): the KPAR8 route died on WIRE
DEMAND in SLR1; u_node_conv_288 is the single largest spatial module (192,536
cells) sitting in the epicenter, and its weights cost ~285 BRAM the placer
needs back to co-locate the engine banks. Engine-side, conv_288 is ~16x fewer
cycles than the MP=16 spatial decimator (the block-14 long pole).

GEOMETRY: conv_288 = 1x1 STRIDE-2 projection 1024->2048, 14x14 -> 7x7,
INT3 weights (the 18th Config-B INT3 layer), per-OC scales. The engine runs
it STRIDE-1 over a DECIMATED input: a per-beat gate in front of a new loader
(ldr17) keeps only even-row/even-col pixels of relu_39's stream — the same
decimation the proven spatial wrapper used (apply_conv288_decimator.py),
moved from compute to load. 49 px x 4 words = 196 act words @ base 13184.

DISPATCH ORDER (load-bearing — the deadlock analysis): conv_288 goes at
dispatch 10, BEFORE conv_286 (which shifts 10->11). node_add_13 pairs
conv_286's bridge output (main, DIRECT from the bridge) with conv_288's
output (via the 4096-deep u_skip_node_add_13). conv_288-first lets the skip
fifo absorb all 3136 beats autonomously; conv_286 then drains against the
buffered skip data. The reverse order deadlocks: bridge-286's drain would
need dispatch-11 data that can't start until 286 drains. (Engine-FIFO beat
order is also dispatch-order — same conclusion.)

PREREQS (run in this order BEFORE this script):
  1. python scripts/build_weight_memory_map.py        (fresh full banks)
  2. python scripts/dedup_engine_banks_conv288.py     (PROOF + 18-entry banks)
AFTER this script:
  3. append node_conv_288 to docs/agent_tasks/06_phase1_compression_candidates_HEAVY.txt
     + re-run build_bias_memory_map.py + build_scale_memory_map.py
     (existing base_words INVARIANT — append-only; conv_288 @47, 8 passes)
  4. update scripts/repack_resnet_kpar8_banks.py constants (75264/18) + run
     (its P0 parses the PATCHED scheduler — hence run after this script)
  5. python scripts/apply_resnet_tap0_hardwire.py     (re-asserts P0 on 18)

SCHEDULER EDITS (surgical — the deployed scheduler carries hand-patched
[K5-DEDUP] bases + [OVERLAP] act remaps that build_scheduler.py would lose):
  - all 22 per-dispatch ROM case tables rebuilt 17 -> 18 rows: rows 0..9
    unchanged; NEW row 10 = conv_288; old rows 10..16 shift to 11..17.
  - weight_base_word_rom: conv_288 @32256; old dispatches 11..16 (orig bases
    61843..92563 > conv_288's 53651) shift +8192; conv_286 (28160) unchanged.
  - bias_base_word_rom: conv_288 @47 (append-only bias/scale maps).
  - scale_mult/shift: scale_factor_to_mult_shift(layer scale_factor), the
    build_scheduler.py formula — RE-DERIVED for all 17 existing dispatches
    from layer_ir.json and asserted == deployed ROM (catches drift).
  - act_in_base 13184 (fresh region: no overlap with 0/4096/8192 zones or
    the OVERLAP remaps 12288..13154; act mem DEPTH 24576).
  - act_out_base 4096 (engine act-BRAM write is DEAD post-OVERLAP — scratch).
  - LAST_DISPATCH 5'd16 -> 5'd17; header count comment.

TOP EDITS:
  1. delete u_skid_node_conv_288 (Phase-A decouple skid) + u_node_conv_288
     spatial inst + node_conv_288_ready_in decl (valid_out/data_out decls
     STAY — the bridge drives them)
  2. relu_39 broadcast accept: skid_node_conv_288_in_ready -> dec288_ready
     (keep ? ldr17 ready : 1 — dropped pixels accepted unconditionally)
  3. add dec288 decimation counters + loader u_ldr_node_conv_288 (ldr17,
     BUS_W=256, base 13184, 196 words) fed by relu_39 (kept beats only)
  4. add bridge u_engine_out_node_conv_288 (SLOT=10, EXPECTED_BEATS=392,
     NUM_DISPATCHES=18) driving node_conv_288_valid_out/data_out;
     ready_out = (node_add_13_skip_in_ready & spatial_run) — same consumer
     contract the spatial module had
  5. existing bridges SLOT renumber 286:10->11 290:11->12 292:12->13
     294:13->14 296:14->15 298:15->16 300:16->17; ALL NUM_DISPATCHES 17->18
  6. all_loaded/all_drain -> 18 wide, remapped (10 = ldr17/conv_288);
     all_loaded_d/all_drain_d shadows widened
  7. arbiter: +ldr17 grant (lowest priority); wr addr/data mux tails;
     act_wr_en_final chain
  8. eofifo_out_ready OR-chain += conv_288 bridge
  9. engine weight bank DEPTH 8384 -> 9408 (x8)
 10. dbg-blk14 $display rewritten (referenced deleted skid/inst internals)

Usage: python scripts/apply_resnet_conv288_engine.py [--dry-run]
Idempotent (marker-guarded), anchor-asserted (abort = no writes), backups
in backups/conv288_engine_20260612/ (made by the launch sequence) plus
.preconv288 siblings written here.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOP = ROOT / "output" / "rtl" / "nn2rtl_top.v"
SCHED = ROOT / "output" / "rtl" / "nn2rtl_scheduler.v"
LAYER_IR = ROOT / "output" / "layer_ir.json"
MARK = "// [CONV288-ENGINE]"

# ---- new-dispatch constants (verified against layer_ir + dedup output) ----
C288 = dict(ic=1024, oc=2048, kh=1, kw=1, sh=1, sw=1, ph=0, pw=0,
            ih=7, iw=7, oh=7, ow=7,           # DECIMATED stride-1 geometry
            wbase=32256, bias=47, zp=0,
            in_bank=2, out_bank=1, skip_mask=0,
            act_in=13184, act_out=4096)
NEW_BANK_DEPTH = 9408     # 75264 / 8 wide lines
# old dispatch idx -> module (for scale re-derivation + drain reindex)
OLD_MOD = {0: "node_conv_246", 1: "node_conv_250", 2: "node_conv_254",
           3: "node_conv_260", 4: "node_conv_264", 5: "node_conv_266",
           6: "node_conv_272", 7: "node_conv_278", 8: "node_conv_282",
           9: "node_conv_284", 10: "node_conv_286", 11: "node_conv_290",
           12: "node_conv_292", 13: "node_conv_294", 14: "node_conv_296",
           15: "node_conv_298", 16: "node_conv_300"}
# old dispatches whose ORIGINAL full-bank base > conv_288's 53651 -> dedup
# base shifts +8192 (290/292/294/296/298/300 = old idx 11..16).
WBASE_SHIFT_OLD_IDX = {11, 12, 13, 14, 15, 16}
# bridge SLOT renumber: module -> new slot
SLOT_RENUM = {"node_conv_286": 11, "node_conv_290": 12, "node_conv_292": 13,
              "node_conv_294": 14, "node_conv_296": 15, "node_conv_298": 16,
              "node_conv_300": 17}
EXPECTED_WBASES = {  # new dispatch idx -> dedup base (from dedup_engine_banks_conv288)
    0: 0, 1: 2304, 2: 4352, 3: 6656, 4: 8960, 5: 9984, 6: 12288, 7: 14592,
    8: 16896, 9: 18944, 10: 32256, 11: 28160, 12: 40448, 13: 44544,
    14: 53760, 15: 57856, 16: 61952, 17: 71168}

ROMS = [  # (name, width_bits)
    ("channel_in_rom", 16), ("channel_out_rom", 16),
    ("kernel_h_rom", 4), ("kernel_w_rom", 4),
    ("stride_h_rom", 3), ("stride_w_rom", 3),
    ("padding_h_rom", 3), ("padding_w_rom", 3),
    ("input_h_rom", 9), ("input_w_rom", 9),
    ("output_h_rom", 9), ("output_w_rom", 9),
    ("weight_base_word_rom", 20), ("bias_base_word_rom", 16),
    ("scale_mult_rom", 32), ("scale_shift_rom", 6),
    ("zero_point_rom", 8), ("input_bank_rom", 3), ("output_bank_rom", 3),
    ("skip_mask_rom", 6), ("act_in_base_word_rom", 16),
    ("act_out_base_word_rom", 16),
]
C288_ROM_VALS = {
    "channel_in_rom": C288["ic"], "channel_out_rom": C288["oc"],
    "kernel_h_rom": C288["kh"], "kernel_w_rom": C288["kw"],
    "stride_h_rom": C288["sh"], "stride_w_rom": C288["sw"],
    "padding_h_rom": C288["ph"], "padding_w_rom": C288["pw"],
    "input_h_rom": C288["ih"], "input_w_rom": C288["iw"],
    "output_h_rom": C288["oh"], "output_w_rom": C288["ow"],
    "weight_base_word_rom": C288["wbase"], "bias_base_word_rom": C288["bias"],
    # scale_mult/scale_shift filled in main() via scale_factor_to_mult_shift
    "zero_point_rom": C288["zp"], "input_bank_rom": C288["in_bank"],
    "output_bank_rom": C288["out_bank"], "skip_mask_rom": C288["skip_mask"],
    "act_in_base_word_rom": C288["act_in"],
    "act_out_base_word_rom": C288["act_out"],
}


def need(cond, msg):
    if not cond:
        print(f"[c288] ABORT (no write): {msg}")
        sys.exit(1)


def sub1(t, pat, rep, msg, flags=0):
    new, n = re.subn(pat, rep, t, count=1, flags=flags)
    need(n == 1, f"anchor not found ({n} matches): {msg}")
    return new


def scale_factor_to_mult_shift(scale_factor):
    """VERBATIM copy of build_scheduler.py:scale_factor_to_mult_shift."""
    if scale_factor is None or scale_factor <= 0.0:
        return (0, 0)
    target_bits = 30
    shift = int(round(target_bits - math.log2(scale_factor)))
    shift = max(0, min(62, shift))
    mult = int(round(scale_factor * (1 << shift)))
    if mult >= (1 << 31):
        excess = mult.bit_length() - 31
        shift = max(0, shift - excess)
        mult = int(round(scale_factor * (1 << shift)))
        mult = min(mult, (1 << 31) - 1)
    return (mult & 0xFFFFFFFF, shift & 0x3F)


def parse_rom(text, name, n):
    vals = {int(i): v for i, v in
            re.findall(rf"5'd(\d+): {name} = \d+'d(\d+);", text)}
    need(len(vals) == n, f"ROM {name}: {len(vals)} rows != {n}")
    return {k: int(v) for k, v in vals.items()}


# ----------------------------------------------------------------------------
# Scheduler surgery
# ----------------------------------------------------------------------------
def patch_scheduler(dry: bool) -> None:
    t = SCHED.read_text(encoding="utf-8")
    need(MARK not in t, "scheduler already patched (marker present)")

    # cross-check the deployed scale ROM against the layer_ir re-derivation
    ir = json.loads(LAYER_IR.read_text(encoding="utf-8"))
    by_id = {L["module_id"]: L for L in ir["layers"]}
    sm = parse_rom(t, "scale_mult_rom", 17)
    ss = parse_rom(t, "scale_shift_rom", 17)
    for oi, mod in OLD_MOD.items():
        m, s = scale_factor_to_mult_shift(by_id[mod]["scale_factor"])
        need((m, s) == (sm[oi], ss[oi]),
             f"scale re-derivation drift: {mod} computed ({m},{s}) != "
             f"deployed ({sm[oi]},{ss[oi]})")
    c288_m, c288_s = scale_factor_to_mult_shift(
        by_id["node_conv_288"]["scale_factor"])
    C288_ROM_VALS["scale_mult_rom"] = c288_m
    C288_ROM_VALS["scale_shift_rom"] = c288_s
    print(f"[c288] scale ROM re-derivation: 17/17 match; conv_288 -> "
          f"mult={c288_m} shift={c288_s} "
          f"(sf={by_id['node_conv_288']['scale_factor']})")

    # rebuild every ROM case body: 0..9 keep, 10 = conv_288, 11..17 = old 10..16
    for name, width in ROMS:
        old = parse_rom(t, name, 17)
        new_rows = {}
        for ni in range(18):
            if ni <= 9:
                v = old[ni]
            elif ni == 10:
                v = C288_ROM_VALS[name]
            else:
                oi = ni - 1
                v = old[oi]
                if name == "weight_base_word_rom" and oi in WBASE_SHIFT_OLD_IDX:
                    v += 8192
            new_rows[ni] = v
        if name == "weight_base_word_rom":
            for ni, v in new_rows.items():
                need(v == EXPECTED_WBASES[ni],
                     f"wbase mismatch d{ni}: {v} != {EXPECTED_WBASES[ni]}")
        # splice: replace the contiguous 5'dN-entry run inside this ROM's case
        block_pat = (rf"(case \(dispatch_idx\)\n)((?:\s*5'd\d+: {name} = [^\n]*\n)+)"
                     rf"(\s*default: {name})")
        m = re.search(block_pat, t)
        need(m, f"case block for {name}")
        tag = {"weight_base_word_rom":
               "  // [CONV288-ENGINE] dedup bases, +8192 past conv_288's region",
               "act_in_base_word_rom":
               "  // [CONV288-ENGINE] fresh region (decimated 7x7x1024 = 196 words)"
               }.get(name, "")
        lines = []
        for ni in range(18):
            cmt = ""
            if ni == 10:
                cmt = "  // [CONV288-ENGINE] conv_288 dispatch" + \
                      (tag.replace("  // [CONV288-ENGINE]", ";") if tag else "")
            lines.append(f"            5'd{ni}: {name} = {width}'d{new_rows[ni]};{cmt}")
        body = "\n".join(lines) + "\n"
        t = t[:m.start(2)] + body + t[m.end(2):]

    t = sub1(t, r"localparam \[4:0\] LAST_DISPATCH = 5'd16;",
             f"localparam [4:0] LAST_DISPATCH = 5'd17;  {MARK} 18 dispatches",
             "LAST_DISPATCH")
    t = sub1(t, r"// Number of engine dispatches: 17",
             "// Number of engine dispatches: 18  " + MARK, "header count")

    # post-conditions
    for name, _ in ROMS:
        parse_rom(t, name, 18)
    if dry:
        print("[c288] scheduler DRY RUN ok")
        return
    bak = SCHED.with_name(SCHED.name + ".preconv288")
    if not bak.exists():
        shutil.copy(SCHED, bak)
    SCHED.write_text(t, encoding="utf-8", newline="\n")
    print(f"[c288] scheduler patched: 22 ROMs -> 18 rows, LAST_DISPATCH 17")


# ----------------------------------------------------------------------------
# Top surgery
# ----------------------------------------------------------------------------
LOADER_BLOCK = f"""
    {MARK} loader for node_conv_288 (dispatch 10, DECIMATED input)
    // conv_288 is a 1x1 STRIDE-2 projection (1024->2048, 14x14 -> 7x7). The
    // engine runs it as a STRIDE-1 7x7 1x1 conv over a DECIMATED input: this
    // per-beat gate keeps only even-row/even-col pixels of relu_39's
    // 14x14x1024 stream (the s2 sampling grid) — the same decimation the
    // spatial wrapper it replaces performed (apply_conv288_decimator.py),
    // moved from compute-side to load-side. 32 beats/pixel (1024ch / 32ch
    // per 256b beat); 49 kept px x 4 words = 196 BRAM words @ base 13184
    // (fresh region: disjoint from the 0/4096/8192 zones and the OVERLAP
    // remaps 12288..13154; act mem DEPTH 24576).
    reg [4:0] dec288_beat;   // 0..31 beat-in-pixel
    reg [3:0] dec288_col;    // 0..13 raster col over the full 14x14
    reg [3:0] dec288_row;    // 0..13 raster row
    wire dec288_keep  = ~dec288_row[0] & ~dec288_col[0];
    wire ldr_node_conv_288_in_ready;
    // dropped pixels are accepted+discarded unconditionally; kept pixels
    // defer to the loader (folded into relu_39's broadcast accept, exactly
    // where the deleted Phase-A skid's in_ready used to sit)
    wire dec288_ready = dec288_keep ? ldr_node_conv_288_in_ready : 1'b1;
    wire dec288_adv   = node_relu_39_valid_out & node_relu_39_ready_out_combined;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dec288_beat <= 5'd0; dec288_col <= 4'd0; dec288_row <= 4'd0;
        end else if (dec288_adv) begin
            if (dec288_beat == 5'd31) begin
                dec288_beat <= 5'd0;
                if (dec288_col == 4'd13) begin
                    dec288_col <= 4'd0;
                    dec288_row <= (dec288_row == 4'd13) ? 4'd0 : dec288_row + 4'd1;
                end else begin
                    dec288_col <= dec288_col + 4'd1;
                end
            end else begin
                dec288_beat <= dec288_beat + 5'd1;
            end
        end
    end
    wire        ldr17_wr_req;
    wire        ldr17_wr_grant;
    wire [14:0] ldr17_wr_addr;
    wire [2047:0] ldr17_wr_data;
    wire        ldr17_loaded;
    stream_to_act_bram_bridge #(
        .BUS_W(256),
        .BRAM_BASE_ADDR(13184),
        .TOTAL_BRAM_WORDS(196)
    ) u_ldr_node_conv_288 (
        .clk(clk), .rst_n(rst_n),
        .in_valid(node_relu_39_valid_out & spatial_run & node_relu_39_ready_out_combined & dec288_keep),
        .in_data(node_relu_39_data_out),
        .in_ready(ldr_node_conv_288_in_ready),
        .wr_req(ldr17_wr_req),
        .wr_grant(ldr17_wr_grant),
        .wr_addr(ldr17_wr_addr),
        .wr_data(ldr17_wr_data),
        .loaded(ldr17_loaded)
    );
"""

BRIDGE_BLOCK = f"""
    {MARK} output bridge for node_conv_288 (dispatch 10)
    // MUST dispatch BEFORE conv_286 (now dispatch 11): node_add_13 pairs
    // conv_286's bridge output (main, DIRECT) with conv_288's (via the
    // 4096-deep u_skip_node_add_13). conv_288-first lets the skip fifo
    // absorb all 3136 beats autonomously; conv_286 then drains against the
    // buffered skip data. The reverse order deadlocks (bridge-286's drain
    // would need dispatch-11 data that cannot start until 286 drains).
    wire u_engine_out_node_conv_288_fifo_ready;
    wire u_engine_out_node_conv_288_drain_complete;
    engine_output_bridge #(
        .SLOT(10),
        .ACT_W(2048),
        .DATA_W(256),
        .EXPECTED_BEATS(392),
        .NUM_DISPATCHES(18)
    ) u_engine_out_node_conv_288 (
        .clk(clk), .rst_n(rst_n),
        .start(sched_engine_output_ready),
        .fifo_out_valid(eofifo_out_valid),
        .fifo_out_data(eofifo_out_data),
        .fifo_out_ready(u_engine_out_node_conv_288_fifo_ready),
        .ready_out((node_add_13_skip_in_ready & spatial_run)),
        .valid_out(node_conv_288_valid_out),
        .data_out(node_conv_288_data_out),
        .drain_complete(u_engine_out_node_conv_288_drain_complete)
    );
"""

OLD_DBG14 = '''                $display("[dbg-blk14 @cyc=%0d] r39_vo=%b r39_rdyout=%b ldr8_inrdy=%b ldr8_loaded=%b | skidC288_v=%b skidC288_inrdy=%b c288_rdyin=%b c288_vo=%b add13skip_inrdy=%b add13skip_v=%b c284_vo=%b c286_vo=%b | c288[outbusy=%b mac=%b fst=%0d ibeat=%0d irow=%0d icol=%0d] disp=%0d",
                         log_cycle, node_relu_39_valid_out, node_relu_39_ready_out_combined,
                         ldr_node_conv_282_in_ready, ldr8_loaded,
                         skid_node_conv_288_valid, skid_node_conv_288_in_ready,
                         node_conv_288_ready_in, node_conv_288_valid_out,
                         node_add_13_skip_in_ready, node_add_13_skip_valid,
                         node_conv_284_valid_out, node_conv_286_valid_out,
                         u_node_conv_288.out_busy, u_node_conv_288.mac_busy, u_node_conv_288.frame_state,
                         u_node_conv_288.in_beat_idx, u_node_conv_288.irow, u_node_conv_288.icol,
                         sched_dispatch_idx);'''
NEW_DBG14 = '''                $display("[dbg-blk14 @cyc=%0d] r39_vo=%b r39_rdyout=%b ldr8_inrdy=%b ldr8_loaded=%b | ldr17_inrdy=%b ldr17_loaded=%b dec288[keep=%b beat=%0d row=%0d col=%0d] c288_vo=%b add13skip_inrdy=%b add13skip_v=%b c284_vo=%b c286_vo=%b disp=%0d",
                         log_cycle, node_relu_39_valid_out, node_relu_39_ready_out_combined,
                         ldr_node_conv_282_in_ready, ldr8_loaded,
                         ldr_node_conv_288_in_ready, ldr17_loaded,
                         dec288_keep, dec288_beat, dec288_row, dec288_col,
                         node_conv_288_valid_out,
                         node_add_13_skip_in_ready, node_add_13_skip_valid,
                         node_conv_284_valid_out, node_conv_286_valid_out,
                         sched_dispatch_idx);'''


def patch_top(dry: bool) -> None:
    t = TOP.read_text(encoding="utf-8")
    need(MARK not in t, "top already patched (marker present)")

    # ---- 1. delete the Phase-A skid + spatial inst (one contiguous block) ----
    skid_inst_pat = (
        r"\n\s*// \[Phase A decouple\] conv_288's own input skid[^\n]*\n"
        r"\s*// isn't throttled[^\n]*\n"
        r"\s*wire skid_node_conv_288_in_ready, skid_node_conv_288_valid;\n"
        r"\s*wire \[255:0\] skid_node_conv_288_data;\n"
        r"\s*skip_fifo #\(\.WIDTH\(256\), \.DEPTH\(4096\)\) u_skid_node_conv_288 \(.*?\);\n"
        r"\s*node_conv_288 u_node_conv_288 \(.*?\);\n")
    t = sub1(t, skid_inst_pat,
             f"\n    {MARK} node_conv_288: engine-dispatched (dispatch 10); "
             f"data_out driven by u_engine_out_node_conv_288; input decimated+"
             f"loaded by u_ldr_node_conv_288 (Phase-A skid deleted)\n",
             "conv_288 skid+inst block", flags=re.DOTALL)

    # ---- 1c. delete the now-undriven ready_in decl ----
    t = sub1(t, r"\n\s*wire node_conv_288_ready_in;", "",
             "node_conv_288_ready_in decl")

    # ---- 2. relu_39 broadcast accept: skid ready -> decimation-gated ready ----
    t = sub1(t,
             r"wire node_relu_39_ready_out_combined = \(ldr_node_conv_282_in_ready & skid_node_conv_288_in_ready\) & spatial_run;",
             "wire node_relu_39_ready_out_combined = (ldr_node_conv_282_in_ready & dec288_ready) & spatial_run;  "
             + MARK, "relu_39 combined accept")

    # ---- 3. loader + decimator (insert before the arbiter section) ----
    t = sub1(t, r"\n(\s*// ----- act BRAM write arbiter)",
             "\n" + LOADER_BLOCK + r"\n\1", "loader insertion point")

    # ---- 5a. SLOT renumber on existing bridges (scoped per bridge block) ----
    for mod, new_slot in SLOT_RENUM.items():
        blk_pat = (r"(engine_output_bridge #\(\s*\.SLOT\()(\d+)"
                   r"(\),[^;]*?u_engine_out_" + mod + r" \()")
        m = re.search(blk_pat, t, flags=re.DOTALL)
        need(m, f"bridge block for {mod}")
        need(int(m.group(2)) == new_slot - 1,
             f"bridge {mod} SLOT {m.group(2)} != expected {new_slot - 1}")
        t = t[:m.start(2)] + str(new_slot) + t[m.end(2):]
        print(f"  SLOT {mod}: {m.group(2)} -> {new_slot}")

    # ---- 5b. NUM_DISPATCHES 17 -> 18 on all existing bridges ----
    t, n = re.subn(r"\.NUM_DISPATCHES\(17\)", ".NUM_DISPATCHES(18)", t)
    need(n == 17, f"NUM_DISPATCHES(17) count {n} != 17")

    # ---- 4. output bridge (insert before the eofifo ready assign) ----
    t = sub1(t, r"\n(\s*assign eofifo_out_ready = )",
             "\n" + BRIDGE_BLOCK + r"\n\1", "bridge insertion point")

    # ---- 6a. all_loaded: widen + remap (10 = ldr17, 11..17 = old 10..16) ----
    t = sub1(t, r"wire \[16:0\] all_loaded;", "wire [17:0] all_loaded;",
             "all_loaded decl")
    lm = re.findall(r"assign all_loaded\[(\d+)\]\s*=\s*(ldr\d+)_loaded;", t)
    need(len(lm) == 17, f"all_loaded assigns {len(lm)} != 17")
    old2ldr = {int(i): l for i, l in lm}
    new_assign = {}
    for ni in range(18):
        if ni <= 9:
            new_assign[ni] = old2ldr[ni]
        elif ni == 10:
            new_assign[ni] = "ldr17"
        else:
            new_assign[ni] = old2ldr[ni - 1]
    block = "\n".join(f"    assign all_loaded[{i}] = {new_assign[i]}_loaded;"
                      + (f"  {MARK}" if i == 10 else "") for i in range(18))
    t = sub1(t, r"(\n\s*assign all_loaded\[\d+\]\s*=\s*ldr\d+_loaded;)+",
             "\n" + block, "all_loaded assign block")

    # ---- 6b. all_drain: widen + reindex by module name ----
    t = sub1(t, r"wire \[16:0\] all_drain;", "wire [17:0] all_drain;",
             "all_drain decl")
    dm = re.findall(
        r"assign all_drain\[(\d+)\]\s*=\s*u_engine_out_(node_conv_\d+)_drain_complete;", t)
    need(len(dm) == 17, f"all_drain assigns {len(dm)} != 17")
    old_drain = {int(i): mod for i, mod in dm}
    for oi, mod in OLD_MOD.items():
        need(old_drain[oi] == mod, f"all_drain[{oi}] is {old_drain[oi]} != {mod}")
    new_drain = {}
    for ni in range(18):
        if ni <= 9:
            new_drain[ni] = old_drain[ni]
        elif ni == 10:
            new_drain[ni] = "node_conv_288"
        else:
            new_drain[ni] = old_drain[ni - 1]
    dblock = "\n".join(
        f"    assign all_drain[{i}] = u_engine_out_{new_drain[i]}_drain_complete;"
        + (f"  {MARK}" if i == 10 else "") for i in range(18))
    t = sub1(t, r"(\n\s*assign all_drain\[\d+\]\s*=\s*u_engine_out_node_conv_\d+_drain_complete;)+",
             "\n" + dblock, "all_drain assign block")

    # ---- 6c. debug shadows ----
    t = sub1(t, r"reg \[16:0\] all_loaded_d;", "reg [17:0] all_loaded_d;",
             "all_loaded_d shadow")
    t = sub1(t, r"reg \[16:0\] all_drain_d;", "reg [17:0] all_drain_d;",
             "all_drain_d shadow")

    # ---- 7a. arbiter grant for ldr17 (lowest priority, after ldr16) ----
    base16 = " | ".join(f"ldr{i}_wr_req" for i in range(16))
    anchor16 = f"assign ldr16_wr_grant = ldr16_wr_req & ~({base16});"
    need(anchor16 in t, "ldr16 grant anchor")
    grants = anchor16 + (f"\n    assign ldr17_wr_grant = ldr17_wr_req & "
                         f"~({base16} | ldr16_wr_req);  {MARK}")
    t = t.replace(anchor16, grants, 1)
    # ---- 7b. act write addr/data mux tails ----
    t = sub1(t, r"ldr16_wr_req \? ldr16_wr_addr : 15'd0;",
             "ldr16_wr_req ? ldr16_wr_addr : ldr17_wr_req ? ldr17_wr_addr : 15'd0;",
             "act wr_addr mux tail")
    t = sub1(t, r"ldr16_wr_req \? ldr16_wr_data : 2048'd0;",
             "ldr16_wr_req ? ldr16_wr_data : ldr17_wr_req ? ldr17_wr_data : 2048'd0;",
             "act wr_data mux tail")
    # ---- 7c. wr_en chain ----
    men = re.search(r"assign act_wr_en_final\s*=\s*([^;]+);", t)
    need(men, "act_wr_en_final assign")
    need("ldr16_wr_req" in men.group(1) and "ldr17_wr_req" not in men.group(1),
         "wr_en chain shape")
    t = t[:men.end(1)] + " | ldr17_wr_req" + t[men.end(1):]

    # ---- 8. eofifo_out_ready OR-chain += conv_288 bridge ----
    t = sub1(t, r"(assign eofifo_out_ready = [^;]*u_engine_out_node_conv_298_fifo_ready)",
             r"\1 | u_engine_out_node_conv_288_fifo_ready",
             "eofifo_out_ready chain")

    # ---- 9. engine weight bank DEPTH x8 ----
    t, n = re.subn(r"\.DEPTH\(8384\),           // \[KPAR8-RN\] 67072/8 wide lines",
                   ".DEPTH(9408),           // [KPAR8-RN+C288] 75264/8 wide lines",
                   t)
    need(n == 8, f"bank DEPTH count {n} != 8")

    # ---- 10. dbg-blk14 rewrite (referenced deleted skid/inst internals) ----
    need(OLD_DBG14 in t, "dbg-blk14 display anchor")
    t = t.replace(OLD_DBG14, NEW_DBG14, 1)

    # ---- post-conditions: no dangling references ----
    for bad in ("skid_node_conv_288", "node_conv_288_ready_in",
                "u_node_conv_288."):
        need(bad not in t, f"dangling reference survived: {bad}")
    need(t.count(".NUM_DISPATCHES(18)") == 18, "NUM_DISPATCHES(18) != 18")
    need(t.count(".DEPTH(9408)") == 8, "DEPTH(9408) != 8")
    need(len(re.findall(r"assign all_loaded\[\d+\]", t)) == 18, "all_loaded != 18")
    need(len(re.findall(r"assign all_drain\[\d+\]", t)) == 18, "all_drain != 18")
    need(t.count("u_engine_out_node_conv_288") >= 5, "conv_288 bridge missing")
    need(t.count("u_ldr_node_conv_288") >= 1, "conv_288 loader missing")

    if dry:
        print("[c288] top DRY RUN ok — all anchors matched")
        return
    bak = TOP.with_name(TOP.name + ".preconv288")
    if not bak.exists():
        shutil.copy(TOP, bak)
    TOP.write_text(t, encoding="utf-8", newline="\n")
    print("[c288] top patched")


def main() -> int:
    dry = "--dry-run" in sys.argv
    # prereq sanity: the 18-entry banks must already exist (dedup ran)
    bank0 = (ROOT / "output/weights/uram_weights_bank0.mem")
    nl = sum(1 for _ in bank0.open())
    need(nl == 75264, f"bank0 has {nl} rows != 75264 — run "
         f"build_weight_memory_map.py + dedup_engine_banks_conv288.py first")
    patch_scheduler(dry)
    patch_top(dry)
    print(f"[c288] {'DRY RUN complete' if dry else 'APPLIED'}. Next: HEAVY "
          f"append + bias/scale regen, repack_resnet_kpar8_banks (75264/18), "
          f"tap0 re-check, verilator build, vec0+vec1 gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
