#!/usr/bin/env python3
"""apply_mbv2_dw_engine_p1.py — MBV2 "DW-on-engine PHASE 1" top + scheduler surgery.

Moves the 3 WIDE depthwise convs node_conv_896/902/908 (C=960, 3x3, s1, p1,
7x7) from the spatial fabric onto the shared engine as DEPTHWISE dispatches
28/31/34 (37 dispatches total), deleting the 3 spatial DW conv instances AND
the 3 retile_gather bridges (br_ldr28/30/32) that fed their consumers'
loaders — the pblock-pinned congestion drivers (~376K LUT incl. bridges).

Per conv (chain conv_894 -> n4_29 -> [896] -> n4_30 -> [898]):
  * u_node_conv_896 spatial instance DELETED (file stays on disk, unused).
  * n4_29 (relu) tile stream -> NEW tiled_stream_to_act_bram_bridge loader
    (per-pixel word-aligned: 30x256b tiles -> 3 full + 1 zero-padded 2048b
    word; BYTE-IDENTICAL layout to retile_gather OUT_BEATS=4) filling the DW
    dispatch's act region (8192/8388/8584, 196 words).
  * Engine runs the conv in DEPTHWISE mode (cfg reg 0x3C; per-lane act mux in
    mac_array, K walk = 9 taps, channel chunk = oc_pass; see
    output/rtl/engine/* [DW-ENGINE P1] markers — this script ASSERTS those
    engine-core edits are present, it does not apply them).
  * NEW engine_output_bridge (SLOT 28/31/34, OUT_KIND=1, OC=960, POSITIONS=49)
    re-emits the 30x256b tile stream on the old node_conv_896_valid_out/
    data_out nets -> n4_30 (relu) UNCHANGED.
  * n4_30's stream -> conv_898's loader CONVERTED to the same tiled loader
    (br_ldr28 retile_gather DELETED).
  * Engine act_out writes go to scratch regions 8780/8976/9172 (never read;
    the FIFO->bridge stream is the consumed copy, same as other dispatches).

Scheduler: 34 -> 37 dispatches (renumber 28..33 -> 29/30/32/33/35/36), DW
rows inserted at 28/31/34, new depthwise_rom + write step 13 (reg 0x3C),
LAST_DISPATCH 36. Weight/bias/scale map bases (13152/13188/13224 + 58/62/66)
must already exist — run scripts/extend_mbv2_engine_maps_dw.py FIRST (this
script asserts the extended line counts).

Anchor-asserted + idempotent (re-run detects the [DW-ENGINE P1] marker and
verifies invariants). Backups: <file>.predw (only written once).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOP = REPO / "output" / "mobilenet-v2" / "rtl" / "nn2rtl_top_engine.v"
SCHED = REPO / "output" / "mobilenet-v2" / "rtl" / "nn2rtl_scheduler.v"
WDIR = REPO / "output" / "mobilenet-v2" / "weights"

MARK = "[DW-ENGINE P1]"


def die(msg: str) -> None:
    print(f"[dw-p1] FATAL: {msg}", file=sys.stderr)
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
# Pre-flight: engine-core edits + extended maps must be in place.
# ============================================================================
def preflight() -> None:
    for f, needle in [
        (REPO / "output/rtl/engine/config_register_block.v", "reg_depthwise"),
        (REPO / "output/rtl/engine/address_generator.v", "cfg_depthwise"),
        (REPO / "output/rtl/engine/mac_array.v", "dw_mode"),
        (REPO / "output/rtl/shared_engine_skeleton.v", "ENABLE_DEPTHWISE"),
    ]:
        if needle not in f.read_text(encoding="utf-8"):
            die(f"engine-core edit missing: {f.name} lacks '{needle}'")
    for b in range(8):
        n = sum(1 for _ in (WDIR / f"uram_weights_bank{b}.mem").open())
        if n != 13260:
            die(f"bank{b} has {n} lines (need 13260) — run extend_mbv2_engine_maps_dw.py")
    for f in ["bias.mem", "scale.mem"]:
        n = sum(1 for _ in (WDIR / f).open())
        if n != 70:
            die(f"{f} has {n} lines (need 70) — run extend_mbv2_engine_maps_dw.py")
    print("[dw-p1] preflight OK (engine-core edits + extended maps present)")


# ============================================================================
# TOP surgery
# ============================================================================
NEW_LOADER_MODULE = '''
// ----------------------------------------------------------------------------
// [DW-ENGINE P1] tiled_stream_to_act_bram_bridge — 256b tile stream -> act BRAM
// with PER-POSITION word alignment.  The producer (an n4_* relu) emits
// TILES_PER_POS contiguous 256b tiles per pixel (tile k = channels k*32..k*32+31).
// Tiles are packed 8-per-2048b-word; the LAST word of each position is flushed
// PARTIAL (remaining bytes ZERO) so every position starts on a word boundary —
// byte-identical to the retile_gather(OUT_BEATS=WORDS_PER_POS) + 2048b-loader
// path it replaces, and exactly the ceil(C/256)-chunks-per-pixel layout the
// engine's address_generator reads (dense ic_cnt[11:8] chunks AND depthwise
// oc_pass chunks).  wr_req/wr_grant/loaded protocol identical to
// stream_to_act_bram_bridge.  Word submissions are >= 6 producer beats apart,
// so single-cycle grant denial never drops a beat (in_ready holds the producer
// only on the word-completing tile while the previous word is still pending).
// ----------------------------------------------------------------------------
module tiled_stream_to_act_bram_bridge #(
    parameter integer TILE_W           = 256,
    parameter integer TILES_PER_POS    = 30,   // tiles per pixel (= C/32)
    parameter integer WORDS_PER_POS    = 4,    // ceil(TILES_PER_POS*TILE_W/2048)
    parameter integer BRAM_BASE_ADDR   = 0,
    parameter integer TOTAL_BRAM_WORDS = 196   // positions * WORDS_PER_POS
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              in_valid,
    input  wire [TILE_W-1:0] in_data,
    output reg               wr_req,
    input  wire              wr_grant,
    output reg  [14:0]       wr_addr,
    output reg  [2047:0]     wr_data,
    output reg               loaded,
    output wire              in_ready
);
    localparam integer TILES_PER_WORD = 2048 / TILE_W;   // 8

    reg [2047:0] word_acc;       // tiles 0..k-1 of the in-progress word
    reg [4:0]    tile_in_word;   // 0..TILES_PER_WORD-1
    reg [5:0]    tile_in_pos;    // 0..TILES_PER_POS-1
    reg [15:0]   word_count;     // words granted so far

    wire [15:0] next_word_count = word_count + 16'd1;
    wire bridge_free = !wr_req || (wr_req && wr_grant);
    wire [14:0] next_wr_addr = BRAM_BASE_ADDR[14:0]
                             + ((wr_req && wr_grant) ? next_word_count[14:0]
                                                     : word_count[14:0]);

    // This tile completes a word when it fills slot 7 OR ends the position.
    wire word_last_tile = (tile_in_word == TILES_PER_WORD[4:0] - 5'd1)
                        || (tile_in_pos == TILES_PER_POS[5:0] - 6'd1);
    // Accept a tile unless it would complete a word while the previous word
    // is still waiting for its grant.
    assign in_ready = !loaded && (!word_last_tile || bridge_free);
    wire take   = in_valid && in_ready;
    wire submit = take && word_last_tile;

    // Compose the word: zero base on tile 0 (gives the zero-padded partial
    // last word), OR-in this tile at its slot (slots are disjoint).
    wire [2047:0] tile_shifted = {{(2048-TILE_W){1'b0}}, in_data}
                                 << (tile_in_word * TILE_W);
    wire [2047:0] word_now = ((tile_in_word == 5'd0) ? 2048'd0 : word_acc)
                             | tile_shifted;

    // [K1-MBV2 style] stream DATA regs (sync-only, no reset): word_acc is
    // rebuilt every word; wr_data is consumed only while wr_req is pending.
    always @(posedge clk) begin
        if (take && !word_last_tile) word_acc <= word_now;
        if (submit)                  wr_data  <= word_now;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_req       <= 1'b0;
            wr_addr      <= 15'd0;
            word_count   <= 16'd0;
            loaded       <= 1'b0;
            tile_in_word <= 5'd0;
            tile_in_pos  <= 6'd0;
        end else begin
            // (1) grant retires wr_req and advances the count.
            if (wr_req && wr_grant) begin
                wr_req     <= 1'b0;
                word_count <= next_word_count;
                if (next_word_count == TOTAL_BRAM_WORDS[15:0]) loaded <= 1'b1;
            end
            // (2) a completed word becomes the pending write (textually after
            // the grant clear: submit wins, same as stream_to_act_bram_bridge).
            if (submit) begin
                wr_req  <= 1'b1;
                wr_addr <= next_wr_addr;
            end
            // (3) tile counters.
            if (take) begin
                tile_in_word <= word_last_tile ? 5'd0 : (tile_in_word + 5'd1);
                tile_in_pos  <= (tile_in_pos == TILES_PER_POS[5:0] - 6'd1)
                                ? 6'd0 : (tile_in_pos + 6'd1);
            end
        end
    end
endmodule

'''


def dw_loader_block(tag: str, conv: str, relu: str, base: int) -> str:
    return f'''
    // {MARK} DW input loader: relu {relu}'s 30x256b tile stream -> act words
    // {base}..{base+195} for the conv_{conv} DEPTHWISE engine dispatch.
    wire        ldr_dw{conv}_wr_req;
    wire        ldr_dw{conv}_wr_grant;
    wire [14:0] ldr_dw{conv}_wr_addr;
    wire [2047:0] ldr_dw{conv}_wr_data;
    wire        ldr_dw{conv}_loaded;
    wire        ldr_dw{conv}_in_ready;
    tiled_stream_to_act_bram_bridge #(
        .TILES_PER_POS(30),
        .WORDS_PER_POS(4),
        .BRAM_BASE_ADDR({base}),
        .TOTAL_BRAM_WORDS(196)
    ) u_ldr_node_conv_{conv} (
        .clk(clk), .rst_n(rst_n),
        .in_valid({relu}_valid_out & spatial_run),
        .in_data({relu}_data_out),
        .wr_req(ldr_dw{conv}_wr_req),
        .wr_grant(ldr_dw{conv}_wr_grant),
        .wr_addr(ldr_dw{conv}_wr_addr),
        .wr_data(ldr_dw{conv}_wr_data),
        .loaded(ldr_dw{conv}_loaded),
        .in_ready(ldr_dw{conv}_in_ready)
    );
'''


def dw_bridge_block(conv: str, slot: int, relu: str) -> str:
    return f'''
    wire u_engine_out_node_conv_{conv}_fifo_ready;
    wire u_engine_out_node_conv_{conv}_drain_complete;
    // {MARK} conv_{conv} DEPTHWISE dispatch {slot}: re-emit the engine's act
    // words as the 30x256b tile stream the (unchanged) relu {relu} consumes —
    // identical geometry to the conv_894-style OC=960 g_tiled bridges.
    engine_output_bridge #(
        .SLOT({slot}),
        .ACT_W(2048),
        .DATA_W(256),
        .EXPECTED_BEATS(196),
        .NUM_DISPATCHES(37),
        .OC(960), .OUT_KIND(1), .POSITIONS(49)
    ) u_engine_out_node_conv_{conv} (
        .clk(clk), .rst_n(rst_n),
        .start(sched_engine_output_ready),
        .fifo_out_valid(eofifo_out_valid),
        .fifo_out_data(eofifo_out_data),
        .fifo_out_ready(u_engine_out_node_conv_{conv}_fifo_ready),
        .ready_out(({relu}_ready_in & spatial_run)),
        .valid_out(node_conv_{conv}_valid_out),
        .data_out(node_conv_{conv}_data_out),
        .drain_complete(u_engine_out_node_conv_{conv}_drain_complete)
    );
'''


def patch_top() -> None:
    text = TOP.read_text(encoding="utf-8")
    if MARK in text:
        verify_top(text)
        print("[dw-p1] top already patched + verified — skipping")
        return
    backup = TOP.with_suffix(".v.predw")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8", newline="\n")

    # ---- header bookkeeping ----
    text = rep(text,
        "// Layers total: 99, spatial: 65, engine-dispatched: 34, residual adds: 10, projection convs: 11.",
        "// Layers total: 99, spatial: 62, engine-dispatched: 37 ([DW-ENGINE P1] conv_896/902/908 depthwise dispatches 28/31/34), residual adds: 10, projection convs: 11.",
        "header layer counts")

    # ---- per-conv wire decls: drop ready_in (spatial conv gone) ----
    for conv, slot in [("896", 28), ("902", 31), ("908", 34)]:
        text = rep(text,
            f"    wire [255:0] node_conv_{conv}_data_out;  // [NATIVE_TILED_{conv}] narrowed: native 256b tile bus\n"
            f"    wire node_conv_{conv}_ready_in;\n",
            f"    wire [255:0] node_conv_{conv}_data_out;  // 256b tile bus, driven by engine_output_bridge SLOT {slot} ({MARK})\n",
            f"wire decls {conv}")

    # ---- delete the 3 per-bridge drain gates ----
    text = rep(text,
        "    wire spatial_run_drain_br_ldr28 = ~(engine_busy | sched_spatial_stall | (any_retile_stall & ~br_ldr28_stall_out));\n"
        "    wire spatial_run_drain_br_ldr30 = ~(engine_busy | sched_spatial_stall | (any_retile_stall & ~br_ldr30_stall_out));\n"
        "    wire spatial_run_drain_br_ldr32 = ~(engine_busy | sched_spatial_stall | (any_retile_stall & ~br_ldr32_stall_out));\n",
        f"    // {MARK} br_ldr28/30/32 DELETED: conv_896/902/908 moved onto the engine;\n"
        "    // their consumers' loaders take the n4_30/32/34 tile streams directly via\n"
        "    // tiled_stream_to_act_bram_bridge (per-pixel word-aligned packing).\n",
        "spatial_run_drain_br_ldr* wires")

    # ---- delete the br_ldr28/30/32 wire decl block ----
    decls = ""
    for br in ["br_ldr28", "br_ldr30", "br_ldr32"]:
        decls += (f"    wire {br}_valid_out;\n"
                  f"    wire [2047:0] {br}_data_out;\n"
                  f"    wire {br}_ready_out;  // toward producer (free-running; observed for completeness)\n"
                  f"    wire {br}_stall_out;\n"
                  f"    wire {br}_wr_accept;  // (full0 & full1) -> spatial_throttle\n")
    text = rep(text, decls, "", "br_ldr28/30/32 wire decls")

    # ---- delete the 3 retile_gather instances ----
    for br, relu, ldr in [("br_ldr28", "n4_30", "ldr28"),
                          ("br_ldr30", "n4_32", "ldr30"),
                          ("br_ldr32", "n4_34", "ldr32")]:
        text = excise_block(
            text,
            f"    retile_gather #(.TILE_W(256), .N_TILES(30), .OUT_W(2048), .OUT_BEATS(4), .SPATIAL(49), .SYNTH_FIXED_MUX(1)) u_{br} (",
            f"        .wr_accept({br}_wr_accept)\n    );\n\n",
            "", f"retile_gather u_{br}")

    # ---- any_retile_stall: drop the 3 deleted terms ----
    text = rep(text,
        "    wire any_retile_stall = br_mean_stall_out | br_ldr22_stall_out | br_ldr24_stall_out | br_ldr26_stall_out | br_ldr28_stall_out | br_ldr30_stall_out | br_ldr32_stall_out;",
        f"    wire any_retile_stall = br_mean_stall_out | br_ldr22_stall_out | br_ldr24_stall_out | br_ldr26_stall_out;  // {MARK} br_ldr28/30/32 deleted",
        "any_retile_stall")

    # ---- spatial zone rewiring (x3): n4_X9 retarget, conv delete, n4_X0 retarget
    zones = [
        ("896", "n4_29", "n4_30", "ldr_dw896", "ldr28", 28),
        ("902", "n4_31", "n4_32", "ldr_dw902", "ldr30", 31),
        ("908", "n4_33", "n4_34", "ldr_dw908", "ldr32", 34),
    ]
    for conv, ru, rd, dwldr, cldr, slot in zones:
        # upstream relu: feed the DW loader instead of the spatial conv
        text = rep(text,
            f"        .out_ready_in(node_conv_{conv}_ready_in),\n"
            f"        .valid_out({ru}_valid_out),",
            f"        // {MARK} conv_{conv} is engine dispatch {slot}: this tile stream\n"
            f"        // fills the DW input loader (u_ldr_node_conv_{conv}).\n"
            f"        .out_ready_in({dwldr}_in_ready & spatial_run),\n"
            f"        .valid_out({ru}_valid_out),",
            f"{ru} out_ready_in retarget")
        # delete the spatial conv instance (fused after the relu's ");")
        text = excise_block(
            text,
            f"    );node_conv_{conv} #(.ENABLE_BACKPRESSURE(1), .NATIVE_TILED(1)) u_node_conv_{conv} (",
            "\n    );\n",
            f"    );\n\n    // node_conv_{conv}: engine-dispatched ({MARK} DEPTHWISE dispatch {slot}; data_out driven by shared_engine via engine_output_bridge SLOT {slot})\n",
            f"u_node_conv_{conv} instance")
        # downstream relu: bridge-fed valid gating + loader-direct ready
        text = rep(text,
            f"        .valid_in(node_conv_{conv}_valid_out),\n"
            f"        .ready_in({rd}_ready_in),\n"
            f"        .data_in(node_conv_{conv}_data_out),\n"
            f"        .out_ready_in(br_{cldr}_wr_accept),",
            f"        .valid_in(node_conv_{conv}_valid_out & spatial_run),  // {MARK} bridge-fed\n"
            f"        .ready_in({rd}_ready_in),\n"
            f"        .data_in(node_conv_{conv}_data_out),\n"
            f"        .out_ready_in({cldr}_in_ready & spatial_run),  // {MARK} direct to the tiled loader\n",
            f"{rd} rewire")

    # ---- convert ldr28/30/32 to tiled loaders fed by the relus ----
    for cldr, consumer, rd in [("ldr28", "node_conv_898", "n4_30"),
                               ("ldr30", "node_conv_904", "n4_32"),
                               ("ldr32", "node_conv_910", "n4_34")]:
        text = rep(text,
            f"    stream_to_act_bram_bridge #(\n"
            f"        .BUS_W(2048),\n"
            f"        .BRAM_BASE_ADDR(0),\n"
            f"        .TOTAL_BRAM_WORDS(196)\n"
            f"    ) u_ldr_{consumer} (\n"
            f"        .clk(clk), .rst_n(rst_n),\n"
            f"        .in_valid(br_{cldr}_valid_out & spatial_run_drain_br_{cldr}),\n"
            f"        .in_data(br_{cldr}_data_out),",
            f"    // {MARK} fed DIRECTLY by relu {rd}'s 256b tile stream (retile_gather\n"
            f"    // br_{cldr} DELETED); per-pixel word-aligned packing is byte-identical\n"
            f"    // to the old retile_gather(OUT_BEATS=4) + 2048b loader path.\n"
            f"    tiled_stream_to_act_bram_bridge #(\n"
            f"        .TILES_PER_POS(30),\n"
            f"        .WORDS_PER_POS(4),\n"
            f"        .BRAM_BASE_ADDR(0),\n"
            f"        .TOTAL_BRAM_WORDS(196)\n"
            f"    ) u_ldr_{consumer} (\n"
            f"        .clk(clk), .rst_n(rst_n),\n"
            f"        .in_valid({rd}_valid_out & spatial_run),\n"
            f"        .in_data({rd}_data_out),",
            f"loader {cldr} conversion")

    # ---- insert the 3 DW input loaders before the (converted) ldr28 block ----
    text = rep(text,
        "    wire        ldr28_wr_req;",
        dw_loader_block("dw896", "896", "n4_29", 8192)
        + dw_loader_block("dw902", "902", "n4_31", 8388).replace("conv_902 DEPTHWISE", "conv_902 DEPTHWISE")
        + dw_loader_block("dw908", "908", "n4_33", 8584)
        + "\n    wire        ldr28_wr_req;",
        "DW loader insertion")

    # ---- arbiter: grants + en/addr/data muxes ----
    m = re.search(r"    assign ldr31_wr_grant = ldr31_wr_req & ~\(([^)]+)\);\n", text)
    if not m:
        die("ldr31 grant line not found")
    prior = m.group(1) + " | ldr31_wr_req"
    grants = f"    // {MARK} DW input loaders (lowest priority; the spatial chain is serial\n"
    grants += "    // so at most one loader is ever active at a time).\n"
    accum = prior
    for dwl in ["ldr_dw896", "ldr_dw902", "ldr_dw908"]:
        grants += f"    assign {dwl}_wr_grant = {dwl}_wr_req & ~({accum});\n"
        accum += f" | {dwl}_wr_req"
    text = rep(text, m.group(0), m.group(0) + grants, "arbiter grants")
    text = rep(text,
        " | ldr29_wr_req | ldr31_wr_req;\n",
        " | ldr29_wr_req | ldr31_wr_req | ldr_dw896_wr_req | ldr_dw902_wr_req | ldr_dw908_wr_req;\n",
        "act_wr_en_final")
    text = rep(text,
        "ldr31_wr_req ? ldr31_wr_addr : 15'd0;",
        "ldr31_wr_req ? ldr31_wr_addr : ldr_dw896_wr_req ? ldr_dw896_wr_addr : ldr_dw902_wr_req ? ldr_dw902_wr_addr : ldr_dw908_wr_req ? ldr_dw908_wr_addr : 15'd0;",
        "act_wr_addr_final mux")
    text = rep(text,
        "ldr31_wr_req ? ldr31_wr_data : 2048'd0;",
        "ldr31_wr_req ? ldr31_wr_data : ldr_dw896_wr_req ? ldr_dw896_wr_data : ldr_dw902_wr_req ? ldr_dw902_wr_data : ldr_dw908_wr_req ? ldr_dw908_wr_data : 2048'd0;",
        "act_wr_data_final mux")

    # ---- all_loaded renumber ----
    text = rep(text,
        "    assign all_loaded[28] = ldr28_loaded;\n"
        "    assign all_loaded[29] = ldr29_loaded;\n"
        "    assign all_loaded[30] = ldr30_loaded;\n"
        "    assign all_loaded[31] = ldr31_loaded;\n"
        "    assign all_loaded[32] = ldr32_loaded;\n"
        "    assign all_loaded[33] = ldr33_loaded;\n"
        "    assign all_loaded[34] = 1'b1;\n"
        "    assign all_loaded[35] = 1'b1;\n"
        "    assign all_loaded[36] = 1'b1;\n",
        f"    // {MARK} renumbered: 896@28(DW) 898@29 900@30 902@31(DW) 904@32 906@33 908@34(DW) 910@35 912@36\n"
        "    assign all_loaded[28] = ldr_dw896_loaded;\n"
        "    assign all_loaded[29] = ldr28_loaded;\n"
        "    assign all_loaded[30] = ldr29_loaded;\n"
        "    assign all_loaded[31] = ldr_dw902_loaded;\n"
        "    assign all_loaded[32] = ldr30_loaded;\n"
        "    assign all_loaded[33] = ldr31_loaded;\n"
        "    assign all_loaded[34] = ldr_dw908_loaded;\n"
        "    assign all_loaded[35] = ldr32_loaded;\n"
        "    assign all_loaded[36] = ldr33_loaded;\n",
        "all_loaded renumber")

    # ---- engine instance: arm depthwise ----
    text = rep(text,
        "        .ENABLE_OUTPUT_BACKPRESSURE(1)\n    ) u_shared_engine (",
        "        .ENABLE_OUTPUT_BACKPRESSURE(1),\n"
        f"        // {MARK} arm the depthwise per-lane mode (dispatches 28/31/34).\n"
        "        .ENABLE_DEPTHWISE(1)\n    ) u_shared_engine (",
        "shared_engine ENABLE_DEPTHWISE")

    # ---- weight banks: depth 13152 -> 13260 ----
    text = rep(text,
        "    // Total MAC cycles = 13152; per-bank depth = 13152.",
        f"    // Total MAC cycles = 13260; per-bank depth = 13260. ({MARK} +108 DW words:\n"
        "    // conv_896@13152 conv_902@13188 conv_908@13224, 36 words each = 4 oc_passes x 9 taps)",
        "bank depth comment")
    text = rep(text, ".DEPTH(13152),", ".DEPTH(13260),", "bank DEPTH", count=8)

    # ---- engine_output_bridge: NUM_DISPATCHES 34 -> 37 (all) ----
    text = rep(text, ".NUM_DISPATCHES(34)", ".NUM_DISPATCHES(37)",
               "NUM_DISPATCHES", count=34)

    # ---- SLOT renumber on the 6 shifted bridges (anchored by instance name) --
    slot_moves = [
        ("898", "1280", "49", 28, 29, False),
        ("900", "256", "196", 29, 30, True),
        ("904", "1280", "49", 30, 32, False),
        ("906", "256", "196", 31, 33, True),
        ("910", "2560", "98", 32, 35, "g2"),
        ("912", "256", "245", 33, 36, "g3"),
    ]
    for conv, dw_, eb, old, new, kind in slot_moves:
        if kind is True:
            tail = ",\n        .OC(960), .OUT_KIND(1), .POSITIONS(49)"
        elif kind == "g2":
            tail = ",\n        .OC(320), .OUT_KIND(2), .POSITIONS(49)"
        elif kind == "g3":
            tail = ",\n        .OC(1280), .OUT_KIND(1), .POSITIONS(49)"
        else:
            tail = ""
        old_block = (f"        .SLOT({old}),\n"
                     f"        .ACT_W(2048),\n"
                     f"        .DATA_W({dw_}),\n"
                     f"        .EXPECTED_BEATS({eb}),\n"
                     f"        .NUM_DISPATCHES(37){tail}\n"
                     f"    ) u_engine_out_node_conv_{conv} (")
        new_block = old_block.replace(f".SLOT({old}),", f".SLOT({new}),  // {MARK} was {old}")
        text = rep(text, old_block, new_block, f"bridge SLOT {conv}")

    # ---- insert the 3 DW output bridges ----
    for conv, slot, relu, after in [("896", 28, "n4_30", "894"),
                                    ("902", 31, "n4_32", "900"),
                                    ("908", 34, "n4_34", "906")]:
        anchor = (f"        .drain_complete(u_engine_out_node_conv_{after}_drain_complete)\n"
                  f"    );\n")
        text = rep(text, anchor, anchor + dw_bridge_block(conv, slot, relu),
                   f"DW bridge {conv} insertion")

    # ---- eofifo_out_ready: add the 3 new terms ----
    text = rep(text,
        " | u_engine_out_node_conv_912_fifo_ready;",
        " | u_engine_out_node_conv_912_fifo_ready | u_engine_out_node_conv_896_fifo_ready | u_engine_out_node_conv_902_fifo_ready | u_engine_out_node_conv_908_fifo_ready;",
        "eofifo_out_ready")

    # ---- all_drain renumber ----
    text = rep(text,
        "    assign all_drain[28] = u_engine_out_node_conv_898_drain_complete;\n"
        "    assign all_drain[29] = u_engine_out_node_conv_900_drain_complete;\n"
        "    assign all_drain[30] = u_engine_out_node_conv_904_drain_complete;\n"
        "    assign all_drain[31] = u_engine_out_node_conv_906_drain_complete;\n"
        "    assign all_drain[32] = u_engine_out_node_conv_910_drain_complete;\n"
        "    assign all_drain[33] = u_engine_out_node_conv_912_drain_complete;\n"
        "    assign all_drain[34] = 1'b1;\n"
        "    assign all_drain[35] = 1'b1;\n"
        "    assign all_drain[36] = 1'b1;\n",
        f"    // {MARK} renumbered (see all_loaded)\n"
        "    assign all_drain[28] = u_engine_out_node_conv_896_drain_complete;\n"
        "    assign all_drain[29] = u_engine_out_node_conv_898_drain_complete;\n"
        "    assign all_drain[30] = u_engine_out_node_conv_900_drain_complete;\n"
        "    assign all_drain[31] = u_engine_out_node_conv_902_drain_complete;\n"
        "    assign all_drain[32] = u_engine_out_node_conv_904_drain_complete;\n"
        "    assign all_drain[33] = u_engine_out_node_conv_906_drain_complete;\n"
        "    assign all_drain[34] = u_engine_out_node_conv_908_drain_complete;\n"
        "    assign all_drain[35] = u_engine_out_node_conv_910_drain_complete;\n"
        "    assign all_drain[36] = u_engine_out_node_conv_912_drain_complete;\n",
        "all_drain renumber")

    # ---- new loader module definition ----
    text = rep(text,
        "module act_unified_mem #(",
        NEW_LOADER_MODULE + "module act_unified_mem #(",
        "tiled loader module insertion")

    verify_top(text)
    TOP.write_text(text, encoding="utf-8", newline="\n")
    print("[dw-p1] top patched + verified")


def verify_top(text: str) -> None:
    checks = [
        ("retile_gather instance count", text.count("retile_gather #("), 4),       # mean + ldr22/24/26
        ("tiled loader instances", text.count(") u_ldr_node_conv_896 ("), 1),
        ("tiled loader module", text.count("module tiled_stream_to_act_bram_bridge"), 1),
        ("NUM_DISPATCHES(37)", text.count(".NUM_DISPATCHES(37)"), 37),
        ("NUM_DISPATCHES(34) gone", text.count(".NUM_DISPATCHES(34)"), 0),
        ("bank DEPTH 13260", text.count(".DEPTH(13260),"), 8),
        ("spatial conv inst gone", text.count("u_node_conv_896 ("), 0),
        # functional names only (underscore-suffixed nets + instances);
        # the explanatory "br_ldr28 DELETED" comments are expected to remain.
        ("br_ldr28 nets gone", text.count("br_ldr28_") + text.count("u_br_ldr28"), 0),
        ("br_ldr30 nets gone", text.count("br_ldr30_") + text.count("u_br_ldr30"), 0),
        ("br_ldr32 nets gone", text.count("br_ldr32_") + text.count("u_br_ldr32"), 0),
        ("ENABLE_DEPTHWISE(1)", text.count(".ENABLE_DEPTHWISE(1)"), 1),
        ("DW bridges", text.count("u_engine_out_node_conv_896_fifo_ready"), 3),    # decl + port + or-term
        ("SLOT(36)", text.count(".SLOT(36),"), 1),
    ]
    for what, got, want in checks:
        if got != want:
            die(f"top verify '{what}': got {got}, want {want}")
    # every SLOT 0..36 present exactly once
    slots = sorted(int(m) for m in re.findall(r"\.SLOT\((\d+)\),", text))
    if slots != list(range(37)):
        die(f"top verify SLOT set: {slots}")
    print("[dw-p1] top invariants OK (37 slots, bridges/loaders/banks consistent)")


# ============================================================================
# SCHEDULER surgery
# ============================================================================
ROM_SPECS = {
    # name: (DW values for dispatches 28/31/34) — same value for all three
    # unless a tuple of 3 is given.
    "channel_in_rom": "16'd960",
    "channel_out_rom": "16'd960",
    "kernel_h_rom": "4'd3",
    "kernel_w_rom": "4'd3",
    "stride_h_rom": "3'd1",
    "stride_w_rom": "3'd1",
    "padding_h_rom": "3'd1",
    "padding_w_rom": "3'd1",
    "input_h_rom": "9'd7",
    "input_w_rom": "9'd7",
    "output_h_rom": "9'd7",
    "output_w_rom": "9'd7",
    "weight_base_word_rom": ("20'd13152", "20'd13188", "20'd13224"),
    "bias_base_word_rom": ("16'd58", "16'd62", "16'd66"),
    "scale_mult_rom": "32'd0",      # vestigial: requant is per-OC from scale.mem
    "scale_shift_rom": "6'd0",      # vestigial
    "zero_point_rom": "8'd0",
    "input_bank_rom": "3'd0",
    "output_bank_rom": "3'd0",
    "skip_mask_rom": "6'd0",
    "act_in_base_word_rom": ("16'd8192", "16'd8388", "16'd8584"),
    "act_out_base_word_rom": ("16'd8780", "16'd8976", "16'd9172"),
}
# old dispatch idx -> new dispatch idx (DW rows inserted at 28/31/34)
OLD_TO_NEW = {**{i: i for i in range(28)},
              28: 29, 29: 30, 30: 32, 31: 33, 32: 35, 33: 36}
DW_NEW_IDX = [28, 31, 34]

DEPTHWISE_ROM = '''
    // [DW-ENGINE P1] per-dispatch DEPTHWISE flag -> engine config reg 0x3C.
    // 1 only for the 3 wide depthwise convs (896@28, 902@31, 908@34).
    reg depthwise_rom;
    always @(*) begin
        case (dispatch_idx)
            6'd28: depthwise_rom = 1'b1;
            6'd31: depthwise_rom = 1'b1;
            6'd34: depthwise_rom = 1'b1;
            default: depthwise_rom = 1'b0;
        endcase
    end
'''


def patch_scheduler() -> None:
    text = SCHED.read_text(encoding="utf-8")
    if MARK in text:
        verify_sched(text)
        print("[dw-p1] scheduler already patched + verified — skipping")
        return
    backup = SCHED.with_suffix(".v.predw")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8", newline="\n")

    text = rep(text, "// Number of engine dispatches: 34",
               f"// Number of engine dispatches: 37  ({MARK} conv_896/902/908 depthwise @ 28/31/34)",
               "sched header")

    # rebuild each ROM case body with 37 entries
    for name, dwval in ROM_SPECS.items():
        entries = re.findall(rf"6'd(\d+): {name} = ([^;]+);", text)
        if len(entries) != 34:
            die(f"scheduler ROM {name}: found {len(entries)} entries, want 34")
        old = {int(i): v for i, v in entries}
        new = {}
        for oi, ni in OLD_TO_NEW.items():
            new[ni] = old[oi]
        for k, ni in enumerate(DW_NEW_IDX):
            new[ni] = dwval[k] if isinstance(dwval, tuple) else dwval
        body_old = "".join(f"            6'd{i}: {name} = {old[i]};\n" for i in range(34))
        body_new = "".join(f"            6'd{i}: {name} = {new[i]};\n" for i in range(37))
        text = rep(text, body_old, body_new, f"ROM {name}")

    # depthwise_rom + write step 13
    text = rep(text,
        "    // ------------------------------------------------------------\n"
        "    // Per-step AXI write address + data\n",
        DEPTHWISE_ROM +
        "\n    // ------------------------------------------------------------\n"
        "    // Per-step AXI write address + data\n",
        "depthwise_rom insertion")
    text = rep(text,
        "            4'd12: begin step_addr = 8'h38; step_data = {16'd0, act_out_base_word_rom}; end\n"
        "            default: begin step_addr = 8'h00; step_data = 32'd0; end",
        "            4'd12: begin step_addr = 8'h38; step_data = {16'd0, act_out_base_word_rom}; end\n"
        f"            4'd13: begin step_addr = 8'h3C; step_data = {{31'd0, depthwise_rom}}; end  // {MARK}\n"
        "            default: begin step_addr = 8'h00; step_data = 32'd0; end",
        "step 13")
    text = rep(text,
        "    localparam [5:0] LAST_DISPATCH = 6'd33;\n    localparam [3:0] LAST_STEP     = 4'd12;",
        f"    localparam [5:0] LAST_DISPATCH = 6'd36;  // {MARK} 37 dispatches\n"
        f"    localparam [3:0] LAST_STEP     = 4'd13;  // {MARK} +DEPTHWISE write",
        "LAST_DISPATCH/LAST_STEP")

    verify_sched(text)
    SCHED.write_text(text, encoding="utf-8", newline="\n")
    print("[dw-p1] scheduler patched + verified")


def verify_sched(text: str) -> None:
    for name in ROM_SPECS:
        n = len(re.findall(rf"6'd\d+: {name} = ", text))
        if n != 37:
            die(f"sched verify ROM {name}: {n} entries, want 37")
    spot = [
        ("6'd28: channel_in_rom = 16'd960;", 1),
        ("6'd29: channel_in_rom = 16'd960;", 1),   # conv_898 (was 28): IC=960
        ("6'd36: channel_out_rom = 16'd1280;", 1), # conv_912 (was 33)
        ("6'd28: weight_base_word_rom = 20'd13152;", 1),
        ("6'd31: weight_base_word_rom = 20'd13188;", 1),
        ("6'd34: weight_base_word_rom = 20'd13224;", 1),
        ("6'd36: weight_base_word_rom = 20'd11552;", 1),
        ("6'd34: bias_base_word_rom = 16'd66;", 1),
        ("6'd36: bias_base_word_rom = 16'd53;", 1),
        ("6'd28: kernel_h_rom = 4'd3;", 1),
        ("6'd29: kernel_h_rom = 4'd1;", 1),
        ("6'd28: padding_h_rom = 3'd1;", 1),
        ("6'd28: act_in_base_word_rom = 16'd8192;", 1),
        ("6'd31: act_in_base_word_rom = 16'd8388;", 1),
        ("6'd34: act_in_base_word_rom = 16'd8584;", 1),
        ("6'd34: act_out_base_word_rom = 16'd9172;", 1),
        ("LAST_DISPATCH = 6'd36", 1),
        ("LAST_STEP     = 4'd13", 1),
        ("step_addr = 8'h3C", 1),
        ("6'd28: depthwise_rom = 1'b1;", 1),
        ("6'd31: depthwise_rom = 1'b1;", 1),
        ("6'd34: depthwise_rom = 1'b1;", 1),
    ]
    for needle, want in spot:
        if text.count(needle) != want:
            die(f"sched verify: '{needle}' count {text.count(needle)} != {want}")
    print("[dw-p1] scheduler invariants OK (37-entry ROMs, DW rows, step 13)")


def main() -> int:
    preflight()
    patch_top()
    patch_scheduler()
    print("[dw-p1] DONE — next: verilator lint, engine-ISO x3 (WLAT=2), "
          "8/8 e2e gate, scripts/check_mbv2_act_region_hazards.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
