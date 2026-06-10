#!/usr/bin/env python3
"""apply_mbv2_dw_engine_ext.py — MBV2 "DW-on-engine STRIDE-1 EXTENSION"
top + scheduler surgery (extends DW-ENGINE P1, commit bc67c94).

Moves the remaining STRIDE-1 depthwise convs (all 3x3, s1, p1) from the
spatial fabric onto the shared engine as DEPTHWISE dispatches:

  conv   C    HxW   px  passes  loader kind             act_in  act_out  wgt   b/s
  824   144  56x56 3136   1     flat BUS_W=1152          9368   12504   13260  70
  836   192  28x28  784   1     flat BUS_W=1536         15640   16424   13269  71
  842   192  28x28  784   1     flat BUS_W=1536         17208   17992   13278  72
  854   384  14x14  196   2     flat BUS_W=4096 (pad)   18776   19168   13287  73
  860   384  14x14  196   2     flat BUS_W=4096 (pad)   19560   19952   13305  75
  866   384  14x14  196   2     flat BUS_W=4096 (pad)   20344   20736   13323  77
  872   384  14x14  196   2     flat BUS_W=4096 (pad)   21128   21520   13341  79
  878   576  14x14  196   3     tiled 18 tiles/pos       21912   22500   13359  81
  884   576  14x14  196   3     tiled 18 tiles/pos       23088   23676   13386  84

Per conv (chain <expand bridge> -> <prod relu> -> [DW conv] -> <cons relu> ->
<project loader>, e.g. conv_822 -> n4_5 -> [824] -> n4_6 -> ldr4):
  * u_node_conv_X spatial instance DELETED (file stays on disk, unused).
  * prod relu's tile/flat stream -> NEW input loader filling the DW dispatch's
    act region:
      - flat convs reuse stream_to_act_bram_bridge (1 zero-extended 2048b
        word per pixel for BUS_W<2048; the BUS_W=4096 {1024'b0, data} pad
        form for C=384 — byte-identical to the u_ldr_node_conv_856 pattern);
      - 878/884 (NATIVE_TILED 256b producers) use the P1
        tiled_stream_to_act_bram_bridge (18 tiles/pos -> 3 words/pos).
  * Engine runs the conv in DEPTHWISE mode (config reg 0x3C — the P1
    engine-core edits; this script ASSERTS they are present).
  * NEW engine_output_bridge re-emits the engine beats on the old
    node_conv_X_valid_out/data_out nets — downstream relu byte-identical:
      - 824/836/842: OUT_KIND=0 low-slice (1 beat/pos, like every OC<=256 slot)
      - 854..872:    OUT_KIND=2 flat-gather (OC=384, 2 beats -> 3072b/pos)
      - 878/884:     OUT_KIND=1 tiled-256  (OC=576, 3 beats -> 18 tiles/pos)
  * Engine act_out writes go to scratch regions (never read; the
    FIFO->bridge stream is the consumed copy — same as every dispatch).
  * Act regions live in [9368, 24264) — strictly disjoint from each other,
    from the P1 DW regions [8192, 9368), and from every ping-pong region
    (< 7232); they reuse only the frame-start stem/816 regions whose
    consumers (d0/d1) retire long before the first DW fill opens (the SAME
    lifetime argument that placed P1's regions at 8192+).

Scheduler: 37 -> 37+n dispatches (DW rows inserted before each conv's
project consumer), depthwise_rom extended, LAST_DISPATCH bumped. Weight/
bias/scale map bases must already exist — run
scripts/extend_mbv2_engine_maps_dw_ext.py FIRST (asserted).

Usage:
  python scripts/apply_mbv2_dw_engine_ext.py [--convs 824,836,...]   (default: all 9)

Anchor-asserted + DECLARATIVE: backups <file>.preext capture the P1 state
once; a re-run (any conv list) restores from .preext and re-applies, so the
script is safely re-runnable / bisectable.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOP = REPO / "output" / "mobilenet-v2" / "rtl" / "nn2rtl_top_engine.v"
SCHED = REPO / "output" / "mobilenet-v2" / "rtl" / "nn2rtl_scheduler.v"
WDIR = REPO / "output" / "mobilenet-v2" / "weights"

MARK = "[DW-ENGINE EXT]"
P1MARK = "[DW-ENGINE P1]"

# The P1 dispatch order (37 modules) — asserted against the .preext sources.
MODULES_P1 = [
    "814", "816", "820", "822", "826", "828", "832", "834", "838", "840",
    "844", "846", "850", "852", "856", "858", "862", "864", "868", "870",
    "874", "876", "880", "882", "886", "888", "892", "894", "896", "898",
    "900", "902", "904", "906", "908", "910", "912",
]
P1_DW = {"896", "902", "908"}

ALL_EXT = ["824", "836", "842", "854", "860", "866", "872", "878", "884"]

# Per-conv static info. succ = the project conv the DW conv precedes (its
# dispatch slot is taken over, everything after shifts +1). prev = the
# preceding dispatch module (bridge insertion anchor).
DW = {
    "824": dict(c=144, h=56, px=3136, passes=1, kind="flat",  bus=1152,
                prod="n4_5",  cons="n4_6",  prev="822", succ="826",
                act_in=9368,  act_out=12504, wbase=13260, bbase=70),
    "836": dict(c=192, h=28, px=784,  passes=1, kind="flat",  bus=1536,
                prod="n4_9",  cons="n4_10", prev="834", succ="838",
                act_in=15640, act_out=16424, wbase=13269, bbase=71),
    "842": dict(c=192, h=28, px=784,  passes=1, kind="flat",  bus=1536,
                prod="n4_11", cons="n4_12", prev="840", succ="844",
                act_in=17208, act_out=17992, wbase=13278, bbase=72),
    "854": dict(c=384, h=14, px=196,  passes=2, kind="flatpad", bus=3072,
                prod="n4_15", cons="n4_16", prev="852", succ="856",
                act_in=18776, act_out=19168, wbase=13287, bbase=73),
    "860": dict(c=384, h=14, px=196,  passes=2, kind="flatpad", bus=3072,
                prod="n4_17", cons="n4_18", prev="858", succ="862",
                act_in=19560, act_out=19952, wbase=13305, bbase=75),
    "866": dict(c=384, h=14, px=196,  passes=2, kind="flatpad", bus=3072,
                prod="n4_19", cons="n4_20", prev="864", succ="868",
                act_in=20344, act_out=20736, wbase=13323, bbase=77),
    "872": dict(c=384, h=14, px=196,  passes=2, kind="flatpad", bus=3072,
                prod="n4_21", cons="n4_22", prev="870", succ="874",
                act_in=21128, act_out=21520, wbase=13341, bbase=79),
    "878": dict(c=576, h=14, px=196,  passes=3, kind="tiled", bus=256,
                prod="n4_23", cons="n4_24", prev="876", succ="880",
                act_in=21912, act_out=22500, wbase=13359, bbase=81),
    "884": dict(c=576, h=14, px=196,  passes=3, kind="tiled", bus=256,
                prod="n4_25", cons="n4_26", prev="882", succ="886",
                act_in=23088, act_out=23676, wbase=13386, bbase=84),
}
for _k, _v in DW.items():
    _v["words"] = _v["px"] * _v["passes"]


def die(msg: str) -> None:
    print(f"[dw-ext] FATAL: {msg}", file=sys.stderr)
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


def build_ext_modules(convs: list[str]) -> list[str]:
    """P1 order with each selected DW conv inserted before its `succ`."""
    out = []
    by_succ = {DW[c]["succ"]: c for c in convs}
    for m in MODULES_P1:
        if m in by_succ:
            out.append(by_succ[m])
        out.append(m)
    assert len(out) == len(MODULES_P1) + len(convs)
    return out


# ============================================================================
# Pre-flight
# ============================================================================
def preflight(convs: list[str]) -> None:
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
        if n != 13413:
            die(f"bank{b} has {n} lines (need 13413) — run extend_mbv2_engine_maps_dw_ext.py")
    for f in ["bias.mem", "scale.mem"]:
        n = sum(1 for _ in (WDIR / f).open())
        if n != 87:
            die(f"{f} has {n} lines (need 87) — run extend_mbv2_engine_maps_dw_ext.py")
    # wrapper geometry must match the table (C, IH=OH=H, stride 1, 3x3, pad 1)
    for c in convs:
        t = (REPO / f"output/mobilenet-v2/rtl/node_conv_{c}.v").read_text(encoding="utf-8")
        lp = dict(re.findall(r"localparam integer (\w+)\s*=\s*(\d+);", t))
        want = DW[c]
        checks = [("C", want["c"]), ("IH", want["h"]), ("IW", want["h"]),
                  ("OH", want["h"]), ("OW", want["h"]), ("SH", 1), ("SW", 1),
                  ("KH", 3), ("KW", 3), ("PH", 1), ("PW", 1)]
        for k, v in checks:
            if k in lp and int(lp[k]) != v:
                die(f"conv_{c} wrapper {k}={lp[k]} != expected {v}")
    print(f"[dw-ext] preflight OK (engine-core + extended maps + wrapper geometry, "
          f"{len(convs)} convs)")


# ============================================================================
# Templates
# ============================================================================
def dw_loader_block(conv: str) -> str:
    i = DW[conv]
    if i["kind"] == "tiled":
        inst = f'''    tiled_stream_to_act_bram_bridge #(
        .TILES_PER_POS({i["c"] // 32}),
        .WORDS_PER_POS({i["passes"]}),
        .BRAM_BASE_ADDR({i["act_in"]}),
        .TOTAL_BRAM_WORDS({i["words"]})
    ) u_ldr_node_conv_{conv} (
        .clk(clk), .rst_n(rst_n),
        .in_valid({i["prod"]}_valid_out & spatial_run),
        .in_data({i["prod"]}_data_out),'''
    elif i["kind"] == "flatpad":
        inst = f'''    stream_to_act_bram_bridge #(
        .BUS_W(4096),
        .BRAM_BASE_ADDR({i["act_in"]}),
        .TOTAL_BRAM_WORDS({i["words"]})
    ) u_ldr_node_conv_{conv} (
        .clk(clk), .rst_n(rst_n),
        .in_valid({i["prod"]}_valid_out & spatial_run),
        .in_data({{1024'b0, {i["prod"]}_data_out}}),'''
    else:
        inst = f'''    stream_to_act_bram_bridge #(
        .BUS_W({i["bus"]}),
        .BRAM_BASE_ADDR({i["act_in"]}),
        .TOTAL_BRAM_WORDS({i["words"]})
    ) u_ldr_node_conv_{conv} (
        .clk(clk), .rst_n(rst_n),
        .in_valid({i["prod"]}_valid_out & spatial_run),
        .in_data({i["prod"]}_data_out),'''
    return f'''
    // {MARK} DW input loader: relu {i["prod"]}'s stream -> act words
    // {i["act_in"]}..{i["act_in"] + i["words"] - 1} for the conv_{conv} DEPTHWISE engine dispatch.
    wire        ldr_dw{conv}_wr_req;
    wire        ldr_dw{conv}_wr_grant;
    wire [14:0] ldr_dw{conv}_wr_addr;
    wire [2047:0] ldr_dw{conv}_wr_data;
    wire        ldr_dw{conv}_loaded;
    wire        ldr_dw{conv}_in_ready;
{inst}
        .wr_req(ldr_dw{conv}_wr_req),
        .wr_grant(ldr_dw{conv}_wr_grant),
        .wr_addr(ldr_dw{conv}_wr_addr),
        .wr_data(ldr_dw{conv}_wr_data),
        .loaded(ldr_dw{conv}_loaded),
        .in_ready(ldr_dw{conv}_in_ready)
    );
'''


def dw_bridge_block(conv: str, slot: int, ndisp: int) -> str:
    i = DW[conv]
    if i["kind"] == "tiled":
        kind_params = (f",\n        .OC({i['c']}), .OUT_KIND(1), .POSITIONS({i['px']})")
        geom = f"{i['c'] // 32}x256b tile stream"
    elif i["kind"] == "flatpad":
        kind_params = (f",\n        .OC({i['c']}), .OUT_KIND(2), .POSITIONS({i['px']})")
        geom = f"flat {i['bus']}b/pos (2-beat gather)"
    else:
        kind_params = ""
        geom = f"flat {i['bus']}b low-slice (1 beat/pos)"
    return f'''
    wire u_engine_out_node_conv_{conv}_fifo_ready;
    wire u_engine_out_node_conv_{conv}_drain_complete;
    // {MARK} conv_{conv} DEPTHWISE dispatch {slot}: re-emit the engine's act
    // words as the {geom} the (unchanged) relu {i["cons"]} consumes.
    engine_output_bridge #(
        .SLOT({slot}),
        .ACT_W(2048),
        .DATA_W({i["bus"]}),
        .EXPECTED_BEATS({i["words"]}),
        .NUM_DISPATCHES({ndisp}){kind_params}
    ) u_engine_out_node_conv_{conv} (
        .clk(clk), .rst_n(rst_n),
        .start(sched_engine_output_ready),
        .fifo_out_valid(eofifo_out_valid),
        .fifo_out_data(eofifo_out_data),
        .fifo_out_ready(u_engine_out_node_conv_{conv}_fifo_ready),
        .ready_out(({i["cons"]}_ready_in & spatial_run)),
        .valid_out(node_conv_{conv}_valid_out),
        .data_out(node_conv_{conv}_data_out),
        .drain_complete(u_engine_out_node_conv_{conv}_drain_complete)
    );
'''


# ============================================================================
# TOP surgery
# ============================================================================
def patch_top(convs: list[str], modules_ext: list[str]) -> None:
    text = TOP.read_text(encoding="utf-8")
    n = len(convs)
    ndisp = 37 + n
    new_idx = {m: i for i, m in enumerate(modules_ext)}

    # ---- header bookkeeping ----
    text = rep(text,
        "// Layers total: 99, spatial: 62, engine-dispatched: 37 ([DW-ENGINE P1] conv_896/902/908 depthwise dispatches 28/31/34), residual adds: 10, projection convs: 11.",
        f"// Layers total: 99, spatial: {62 - n}, engine-dispatched: {ndisp} "
        f"([DW-ENGINE P1] 896/902/908 + {MARK} {'/'.join(convs)} depthwise dispatches), "
        "residual adds: 10, projection convs: 11.",
        "header layer counts")

    # ---- parse the P1 all_loaded / all_drain rows (old idx -> signal) ----
    old_loaded = {}
    old_drain = {}
    for d in range(37):
        m = re.search(rf"    assign all_loaded\[{d}\] = (\S+?);", text)
        if not m:
            die(f"all_loaded[{d}] not found")
        old_loaded[d] = m.group(1)
        m = re.search(rf"    assign all_drain\[{d}\] = (\S+?);", text)
        if not m:
            die(f"all_drain[{d}] not found")
        old_drain[d] = m.group(1)

    # ---- per-conv zone surgery ----
    for conv in convs:
        i = DW[conv]
        slot = new_idx[conv]
        if i["kind"] == "tiled":
            # wire decls: drop ready_in
            text = rep(text,
                f"    wire [255:0] node_conv_{conv}_data_out;  // [NATIVE_TILED_{conv}] narrowed: native 256b tile bus\n"
                f"    wire node_conv_{conv}_ready_in;\n",
                f"    wire [255:0] node_conv_{conv}_data_out;  // 256b tile bus, driven by engine_output_bridge SLOT {slot} ({MARK})\n",
                f"wire decls {conv}")
            # producer relu: feed the DW loader (raw NATIVE_TILED handshake -> gated)
            text = rep(text,
                f"        .out_ready_in(node_conv_{conv}_ready_in),\n"
                f"        .valid_out({i['prod']}_valid_out),",
                f"        // {MARK} conv_{conv} is engine dispatch {slot}: this tile stream\n"
                f"        // fills the DW input loader (u_ldr_node_conv_{conv}).\n"
                f"        .out_ready_in(ldr_dw{conv}_in_ready & spatial_run),\n"
                f"        .valid_out({i['prod']}_valid_out),",
                f"producer {i['prod']} retarget ({conv})")
            # delete the spatial conv instance
            text = excise_block(text,
                f"    node_conv_{conv} #(.ENABLE_BACKPRESSURE(1), .NATIVE_TILED(1)) u_node_conv_{conv} (",
                "\n    );\n",
                f"    // node_conv_{conv}: engine-dispatched ({MARK} DEPTHWISE dispatch {slot}; "
                f"data_out driven by shared_engine via engine_output_bridge SLOT {slot})\n",
                f"u_node_conv_{conv} instance")
            # consumer relu: bridge-fed valid gating (was RAW under NATIVE_TILED)
            text = rep(text,
                f"        .valid_in(node_conv_{conv}_valid_out),\n"
                f"        .ready_in({i['cons']}_ready_in),",
                f"        .valid_in(node_conv_{conv}_valid_out & spatial_run),  // {MARK} bridge-fed\n"
                f"        .ready_in({i['cons']}_ready_in),",
                f"consumer {i['cons']} rewire ({conv})")
        else:
            text = rep(text,
                f"    wire [{i['bus'] - 1}:0] node_conv_{conv}_data_out;\n"
                f"    wire node_conv_{conv}_ready_in;\n",
                f"    wire [{i['bus'] - 1}:0] node_conv_{conv}_data_out;  // driven by engine_output_bridge SLOT {slot} ({MARK})\n",
                f"wire decls {conv}")
            text = rep(text,
                f"        .out_ready_in(node_conv_{conv}_ready_in & spatial_run),\n"
                f"        .valid_out({i['prod']}_valid_out),",
                f"        // {MARK} conv_{conv} is engine dispatch {slot}: this flat stream\n"
                f"        // fills the DW input loader (u_ldr_node_conv_{conv}).\n"
                f"        .out_ready_in(ldr_dw{conv}_in_ready & spatial_run),\n"
                f"        .valid_out({i['prod']}_valid_out),",
                f"producer {i['prod']} retarget ({conv})")
            text = excise_block(text,
                f"    node_conv_{conv} #(.ENABLE_BACKPRESSURE(1)) u_node_conv_{conv} (",
                "\n    );\n",
                f"    // node_conv_{conv}: engine-dispatched ({MARK} DEPTHWISE dispatch {slot}; "
                f"data_out driven by shared_engine via engine_output_bridge SLOT {slot})\n",
                f"u_node_conv_{conv} instance")
            # consumer relu valid_in is ALREADY `& spatial_run` gated — unchanged.

        # input loader (inserted before the arbiter)
        text = rep(text,
            "    // ----- act BRAM write arbiter: engine priority, then bridges -----",
            dw_loader_block(conv)
            + "\n    // ----- act BRAM write arbiter: engine priority, then bridges -----",
            f"loader insertion ({conv})")

        # output bridge (inserted after the preceding dispatch's bridge)
        anchor = (f"        .drain_complete(u_engine_out_node_conv_{i['prev']}_drain_complete)\n"
                  f"    );\n")
        text = rep(text, anchor, anchor + dw_bridge_block(conv, slot, ndisp),
                   f"bridge insertion ({conv})")

    # ---- eofifo_out_ready: one term per new bridge ----
    fifo_terms = "".join(f" | u_engine_out_node_conv_{c}_fifo_ready" for c in convs)
    text = rep(text,
        " | u_engine_out_node_conv_908_fifo_ready;",
        f" | u_engine_out_node_conv_908_fifo_ready{fifo_terms};",
        "eofifo_out_ready")

    # ---- arbiter: grants + en/addr/data mux terms (after the P1 DW trio) ----
    m = re.search(r"    assign ldr_dw908_wr_grant = ldr_dw908_wr_req & ~\(([^)]+)\);\n", text)
    if not m:
        die("ldr_dw908 grant line not found")
    accum = m.group(1) + " | ldr_dw908_wr_req"
    grants = (f"    // {MARK} stride-1 DW input loaders (lowest priority; the spatial\n"
              "    // chain is serial so at most one loader is ever active at a time).\n")
    for conv in convs:
        grants += f"    assign ldr_dw{conv}_wr_grant = ldr_dw{conv}_wr_req & ~({accum});\n"
        accum += f" | ldr_dw{conv}_wr_req"
    text = rep(text, m.group(0), m.group(0) + grants, "arbiter grants")
    en_terms = "".join(f" | ldr_dw{c}_wr_req" for c in convs)
    text = rep(text,
        " | ldr_dw896_wr_req | ldr_dw902_wr_req | ldr_dw908_wr_req;\n",
        f" | ldr_dw896_wr_req | ldr_dw902_wr_req | ldr_dw908_wr_req{en_terms};\n",
        "act_wr_en_final")
    addr_terms = "".join(f"ldr_dw{c}_wr_req ? ldr_dw{c}_wr_addr : " for c in convs)
    text = rep(text,
        "ldr_dw908_wr_req ? ldr_dw908_wr_addr : 15'd0;",
        f"ldr_dw908_wr_req ? ldr_dw908_wr_addr : {addr_terms}15'd0;",
        "act_wr_addr_final mux")
    data_terms = "".join(f"ldr_dw{c}_wr_req ? ldr_dw{c}_wr_data : " for c in convs)
    text = rep(text,
        "ldr_dw908_wr_req ? ldr_dw908_wr_data : 2048'd0;",
        f"ldr_dw908_wr_req ? ldr_dw908_wr_data : {data_terms}2048'd0;",
        "act_wr_data_final mux")

    # ---- all_loaded / all_drain renumber (rows 0..45 rebuilt) ----
    new_loaded = {d: "1'b1" for d in range(46)}
    new_drain = {d: "1'b1" for d in range(46)}
    p1_idx = {m_: i_ for i_, m_ in enumerate(MODULES_P1)}
    for d, mod in enumerate(modules_ext):
        if mod in convs:
            new_loaded[d] = f"ldr_dw{mod}_loaded"
            new_drain[d] = f"u_engine_out_node_conv_{mod}_drain_complete"
        else:
            new_loaded[d] = old_loaded[p1_idx[mod]]
            new_drain[d] = old_drain[p1_idx[mod]]

    def rebuild(vec: str, rows: dict[int, str]) -> None:
        nonlocal text
        pat = re.compile(rf"    assign {vec}\[0\] = [\s\S]*?    assign {vec}\[45\] = 1'b1;\n")
        mm = pat.search(text)
        if not mm:
            die(f"{vec} block not found")
        body = (f"    // {MARK} renumbered for {ndisp} dispatches "
                f"(DW inserts: {', '.join(f'{c}@{new_idx[c]}' for c in convs)})\n")
        body += "".join(f"    assign {vec}[{d}] = {rows[d]};\n" for d in range(46))
        text = text[:mm.start()] + body + text[mm.end():]

    rebuild("all_loaded", new_loaded)
    rebuild("all_drain", new_drain)

    # ---- SLOT renumber on every pre-existing bridge ----
    for mod in MODULES_P1:
        old_slot = p1_idx[mod]
        ns = new_idx[mod]
        pat = re.compile(
            rf"\.SLOT\((\d+)\),[^\n]*(?=[\s\S]{{0,400}}?\) u_engine_out_node_conv_{mod} \()")
        mm = pat.search(text)
        if not mm:
            die(f"bridge SLOT for conv_{mod} not found")
        if int(mm.group(1)) != old_slot:
            die(f"bridge conv_{mod}: SLOT {mm.group(1)} != expected P1 slot {old_slot}")
        repl = f".SLOT({ns})," + (f"  // {MARK} was {old_slot}" if ns != old_slot else "")
        text = text[:mm.start()] + repl + text[mm.end():]

    # ---- NUM_DISPATCHES on the 37 pre-existing bridges ----
    text = rep(text, ".NUM_DISPATCHES(37)", f".NUM_DISPATCHES({ndisp})",
               "NUM_DISPATCHES", count=37)

    # ---- weight banks: depth 13260 -> 13413 ----
    text = rep(text,
        "    // Total MAC cycles = 13260; per-bank depth = 13260. ([DW-ENGINE P1] +108 DW words:\n"
        "    // conv_896@13152 conv_902@13188 conv_908@13224, 36 words each = 4 oc_passes x 9 taps)",
        f"    // Total MAC cycles = 13413; per-bank depth = 13413. ({MARK} +153 stride-1 DW\n"
        "    // words appended after the P1 trio: 824@13260 836@13269 842@13278 854@13287\n"
        "    // 860@13305 866@13323 872@13341 878@13359 884@13386; oc_passes x 9 taps each)",
        "bank depth comment")
    text = rep(text, ".DEPTH(13260),", ".DEPTH(13413),", "bank DEPTH", count=8)

    verify_top(text, convs, modules_ext)
    TOP.write_text(text, encoding="utf-8", newline="\n")
    print(f"[dw-ext] top patched + verified ({n} convs, {ndisp} dispatches)")


def verify_top(text: str, convs: list[str], modules_ext: list[str]) -> None:
    ndisp = len(modules_ext)
    checks = [
        (f"NUM_DISPATCHES({ndisp})", text.count(f".NUM_DISPATCHES({ndisp})"), ndisp),
        ("NUM_DISPATCHES(37) gone", text.count(".NUM_DISPATCHES(37)"), 0),
        ("bank DEPTH 13413", text.count(".DEPTH(13413),"), 8),
        ("tiled loader module", text.count("module tiled_stream_to_act_bram_bridge"), 1),
    ]
    for conv in convs:
        checks += [
            (f"spatial conv {conv} gone", text.count(f"u_node_conv_{conv} ("), 0),
            (f"loader {conv}", text.count(f") u_ldr_node_conv_{conv} ("), 1),
            (f"bridge {conv}", text.count(f"u_engine_out_node_conv_{conv}_fifo_ready"), 3),
            (f"ready_in {conv} gone", text.count(f"node_conv_{conv}_ready_in"), 0),
        ]
    for what, got, want in checks:
        if got != want:
            die(f"top verify '{what}': got {got}, want {want}")
    slots = sorted(int(m) for m in re.findall(r"\.SLOT\((\d+)\),", text))
    if slots != list(range(ndisp)):
        die(f"top verify SLOT set: {slots}")
    # every all_loaded/all_drain row maps to the EXT schedule
    for d, mod in enumerate(modules_ext):
        ml = re.search(rf"    assign all_loaded\[{d}\] = (\S+?);", text)
        md = re.search(rf"    assign all_drain\[{d}\] = (\S+?);", text)
        if not ml or not md:
            die(f"all_loaded/all_drain[{d}] missing after rebuild")
        if md.group(1) != f"u_engine_out_node_conv_{mod}_drain_complete":
            die(f"all_drain[{d}] = {md.group(1)} != bridge of conv_{mod}")
        if mod in convs and ml.group(1) != f"ldr_dw{mod}_loaded":
            die(f"all_loaded[{d}] = {ml.group(1)} != ldr_dw{mod}_loaded")
    print(f"[dw-ext] top invariants OK ({ndisp} slots, bridges/loaders/banks consistent)")


# ============================================================================
# SCHEDULER surgery
# ============================================================================
ROM_NAMES = [
    "channel_in_rom", "channel_out_rom", "kernel_h_rom", "kernel_w_rom",
    "stride_h_rom", "stride_w_rom", "padding_h_rom", "padding_w_rom",
    "input_h_rom", "input_w_rom", "output_h_rom", "output_w_rom",
    "weight_base_word_rom", "bias_base_word_rom", "scale_mult_rom",
    "scale_shift_rom", "zero_point_rom", "input_bank_rom", "output_bank_rom",
    "skip_mask_rom", "act_in_base_word_rom", "act_out_base_word_rom",
]


def dw_rom_value(name: str, conv: str) -> str:
    i = DW[conv]
    return {
        "channel_in_rom": f"16'd{i['c']}",
        "channel_out_rom": f"16'd{i['c']}",
        "kernel_h_rom": "4'd3", "kernel_w_rom": "4'd3",
        "stride_h_rom": "3'd1", "stride_w_rom": "3'd1",
        "padding_h_rom": "3'd1", "padding_w_rom": "3'd1",
        "input_h_rom": f"9'd{i['h']}", "input_w_rom": f"9'd{i['h']}",
        "output_h_rom": f"9'd{i['h']}", "output_w_rom": f"9'd{i['h']}",
        "weight_base_word_rom": f"20'd{i['wbase']}",
        "bias_base_word_rom": f"16'd{i['bbase']}",
        "scale_mult_rom": "32'd0",     # vestigial: requant is per-OC from scale.mem
        "scale_shift_rom": "6'd0",     # vestigial
        "zero_point_rom": "8'd0",
        "input_bank_rom": "3'd0", "output_bank_rom": "3'd0",
        "skip_mask_rom": "6'd0",
        "act_in_base_word_rom": f"16'd{i['act_in']}",
        "act_out_base_word_rom": f"16'd{i['act_out']}",
    }[name]


def patch_scheduler(convs: list[str], modules_ext: list[str]) -> None:
    text = SCHED.read_text(encoding="utf-8")
    n = len(convs)
    ndisp = 37 + n
    new_idx = {m: i for i, m in enumerate(modules_ext)}
    p1_idx = {m: i for i, m in enumerate(MODULES_P1)}

    text = rep(text,
        "// Number of engine dispatches: 37  ([DW-ENGINE P1] conv_896/902/908 depthwise @ 28/31/34)",
        f"// Number of engine dispatches: {ndisp}  ([DW-ENGINE P1] + {MARK} depthwise @ "
        + ", ".join(f"{m}@{new_idx[m]}" for m in modules_ext if m in convs or m in P1_DW) + ")",
        "sched header")

    # rebuild each ROM case body with `ndisp` entries
    for name in ROM_NAMES:
        entries = re.findall(rf"6'd(\d+): {name} = ([^;]+);", text)
        if len(entries) != 37:
            die(f"scheduler ROM {name}: found {len(entries)} entries, want 37")
        old = {int(i_): v for i_, v in entries}
        new = {}
        for d, mod in enumerate(modules_ext):
            if mod in convs:
                new[d] = dw_rom_value(name, mod)
            else:
                new[d] = old[p1_idx[mod]]
        body_old = "".join(f"            6'd{i_}: {name} = {old[i_]};\n" for i_ in range(37))
        body_new = "".join(f"            6'd{i_}: {name} = {new[i_]};\n" for i_ in range(ndisp))
        text = rep(text, body_old, body_new, f"ROM {name}")

    # depthwise_rom: rebuild the case entries (P1 trio renumbered + the new set)
    dw_idx = sorted(new_idx[m] for m in modules_ext if m in convs or m in P1_DW)
    body_new = "".join(f"            6'd{d}: depthwise_rom = 1'b1;\n" for d in dw_idx)
    text = excise_block(text,
        "            6'd28: depthwise_rom = 1'b1;\n",
        "            6'd34: depthwise_rom = 1'b1;\n",
        body_new, "depthwise_rom entries")
    text = rep(text,
        "    // 1 only for the 3 wide depthwise convs (896@28, 902@31, 908@34).",
        f"    // 1 for every depthwise dispatch ({MARK}): "
        + ", ".join(f"{m}@{new_idx[m]}" for m in modules_ext if m in convs or m in P1_DW) + ".",
        "depthwise_rom comment")

    text = rep(text,
        "    localparam [5:0] LAST_DISPATCH = 6'd36;  // [DW-ENGINE P1] 37 dispatches",
        f"    localparam [5:0] LAST_DISPATCH = 6'd{ndisp - 1};  // {MARK} {ndisp} dispatches",
        "LAST_DISPATCH")

    verify_sched(text, convs, modules_ext)
    SCHED.write_text(text, encoding="utf-8", newline="\n")
    print(f"[dw-ext] scheduler patched + verified ({ndisp}-entry ROMs)")


def verify_sched(text: str, convs: list[str], modules_ext: list[str]) -> None:
    ndisp = len(modules_ext)
    new_idx = {m: i for i, m in enumerate(modules_ext)}
    for name in ROM_NAMES:
        cnt = len(re.findall(rf"6'd\d+: {name} = ", text))
        if cnt != ndisp:
            die(f"sched verify ROM {name}: {cnt} entries, want {ndisp}")
    spot = [(f"LAST_DISPATCH = 6'd{ndisp - 1}", 1)]
    for c in convs:
        d = new_idx[c]
        i = DW[c]
        spot += [
            (f"6'd{d}: channel_in_rom = 16'd{i['c']};", 1),
            (f"6'd{d}: kernel_h_rom = 4'd3;", 1),
            (f"6'd{d}: input_h_rom = 9'd{i['h']};", 1),
            (f"6'd{d}: weight_base_word_rom = 20'd{i['wbase']};", 1),
            (f"6'd{d}: bias_base_word_rom = 16'd{i['bbase']};", 1),
            (f"6'd{d}: act_in_base_word_rom = 16'd{i['act_in']};", 1),
            (f"6'd{d}: act_out_base_word_rom = 16'd{i['act_out']};", 1),
            (f"6'd{d}: depthwise_rom = 1'b1;", 1),
        ]
    # the P1 trio renumbered + the final dispatch (912) row preserved
    for m in ["896", "902", "908"]:
        spot.append((f"6'd{new_idx[m]}: depthwise_rom = 1'b1;", 1))
    spot.append((f"6'd{ndisp - 1}: channel_out_rom = 16'd1280;", 1))   # conv_912
    dw_count = len(re.findall(r"depthwise_rom = 1'b1;", text))
    if dw_count != len(convs) + 3:
        die(f"sched verify depthwise_rom: {dw_count} ones, want {len(convs) + 3}")
    for needle, want in spot:
        if text.count(needle) != want:
            die(f"sched verify: '{needle}' count {text.count(needle)} != {want}")
    print(f"[dw-ext] scheduler invariants OK ({ndisp} entries, "
          f"{len(convs) + 3} depthwise rows)")


# ============================================================================
# main
# ============================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", default=",".join(ALL_EXT),
                    help="comma list of conv ids to move (default: all 9)")
    args = ap.parse_args()
    convs = [c.strip() for c in args.convs.split(",") if c.strip()]
    for c in convs:
        if c not in DW:
            die(f"unknown conv {c}; valid: {ALL_EXT}")
    convs = [c for c in ALL_EXT if c in convs]   # canonical order

    preflight(convs)

    # declarative: restore the P1 state from .preext if already applied
    for f in [TOP, SCHED]:
        backup = f.with_suffix(f.suffix + ".preext")
        cur = f.read_text(encoding="utf-8")
        if MARK in cur:
            if not backup.exists():
                die(f"{f.name} already patched but {backup.name} missing")
            f.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
            print(f"[dw-ext] {f.name}: restored P1 baseline from {backup.name}")
        else:
            if P1MARK not in cur:
                die(f"{f.name} lacks {P1MARK} — base must be the DW-ENGINE P1 state")
            if not backup.exists():
                backup.write_text(cur, encoding="utf-8", newline="\n")
                print(f"[dw-ext] {f.name}: saved P1 baseline to {backup.name}")

    modules_ext = build_ext_modules(convs)
    print(f"[dw-ext] EXT schedule ({len(modules_ext)} dispatches): "
          + " ".join(f"{m}@{i}" + ("*" if m in convs else "")
                     for i, m in enumerate(modules_ext) if m in convs or m in P1_DW))
    patch_top(convs, modules_ext)
    patch_scheduler(convs, modules_ext)
    print("[dw-ext] DONE — next: verilator lint, "
          "scripts/check_mbv2_act_region_hazards_ext.py, 8/8 e2e gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
