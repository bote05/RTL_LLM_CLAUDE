#!/usr/bin/env python3
"""apply_mbv2_dw_engine_quartet.py — MBV2 "QUARTET-ON-ENGINE": move the 4
STRIDE-2 depthwise convs (818/830/848/890) from the spatial fabric onto the
shared engine as DEPTHWISE dispatches. Extends DW-ENGINE P1/EXT (base state =
ENG-PIPE commit 5fe7327: 47 dispatches incl. [FC-ENGINE] node_linear @46,
KPAR8 banks, ENG_PIPE FSM).

WHY STRIDE-2 IS CONFIG-ONLY: output/rtl/engine/address_generator.v already
computes base_r = pixel_h*cfg_stride_h (+kh-pad, bounds vs cfg_ih/iw) and, in
DW mode, act word = act_in_base + (in_r*cfg_iw + in_c)*ic_chunks + oc_pass.
The FSM iterates pixels over cfg_oh/cfg_ow. So a stride-2 DW dispatch needs
ONLY: the full IHxIW input image in the act BRAM (loader TOTAL = IH*IW*chunks
words) + scheduler rows with stride=2, ih=2*oh + weight/bias/scale map entries
(scripts/extend_mbv2_engine_maps_dw_quartet.py — run FIRST, asserted).

  conv   C   IHxIH->OHxOH  in_wds out_beats passes loader kind     act_in  act_out  wgt(kp8)  b/s
  818    96  112 -> 56     12544   3136      1     flat BUS_W=768   12544       0    18536     91
  830   144   56 -> 28      3136    784      1     flat BUS_W=1152   9368   12504    18545     92
  848   192   28 -> 14       784    196      1     flat BUS_W=1536  15640   16424    18554     93
  890   576   14 ->  7       588    147      3     tiled 18 t/pos   21912   22500    18563     94

ACT-REGION PLAN (no act-mem growth; ACT_DEPTH stays 25600). Under the proven
concurrency model (while dispatch d runs/drains, the ONLY loader filling is
dispatch d+1's; loaders never write after `loaded`), every region only has to
be checked against its ADJACENT dispatch windows:
  * 830/848/890 regions REUSE retired EXT windows (824's / 836's / 878's
    in+scratch windows) — every touched adjacent pair is STRICTLY DISJOINT.
  * 818 reads [12544,+12544) = d1(816)'s in-place region: ldr_dw818 re-fills
    it with the RELU'D copy of d1's own bridge stream WHILE d1 runs. Safe by
    the lag argument (engine act write of word i == FIFO push cycle of beat i
    [engine_act_wr_commit = wr_en & eofifo_in_ready]; the loader writes word
    i >= 3 cycles later through bridge->n4_3->arbiter, and d1's 1x1 walk
    never re-reads word i after pixel i) — the same argument as the e2e-
    proven "lag-safe-1x1" rxf class, extended to the wxf pair. 818's scratch
    [0,+3136) is dead (next fill of [0,..) opens during d3 for d4's loader).
Proof: scripts/check_mbv2_act_region_hazards_quartet.py (PART A strict +
lag classes on touched pairs, PART B baseline-identical inherited rows).

Scheduler: 47 -> 47+n dispatches (each DW conv inserted before its project
consumer: 818@2 [stage 2], 830@7, 848@16, 890@37 [stage-1 indices]), all 22
ROMs rebuilt, depthwise_rom extended, LAST_DISPATCH bumped, DEPTH(2317->2324)
on the 8 kp8 banks.

Usage:
  python scripts/apply_mbv2_dw_engine_quartet.py --convs 830,848,890   (STAGE 1)
  python scripts/apply_mbv2_dw_engine_quartet.py                       (all 4 = STAGE 2)

Anchor-asserted + DECLARATIVE: backups <file>.prequartet capture the FC/
ENG-PIPE state once; a re-run (any conv list) restores from .prequartet and
re-applies, so the script is safely re-runnable / bisectable.
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

MARK = "[DW-QUARTET]"
FCMARK = "[FC-ENGINE]"

# The FC dispatch order (47 modules; "linear" = node_linear FC) — asserted
# against the .prequartet sources.
MODULES_FC = [
    "814", "816", "820", "822", "824", "826", "828", "832", "834", "836",
    "838", "840", "842", "844", "846", "850", "852", "854", "856", "858",
    "860", "862", "864", "866", "868", "870", "872", "874", "876", "878",
    "880", "882", "884", "886", "888", "892", "894", "896", "898", "900",
    "902", "904", "906", "908", "910", "912", "linear",
]
FC_DW = {"824", "836", "842", "854", "860", "866", "872", "878", "884",
         "896", "902", "908"}

ALL_Q = ["818", "830", "848", "890"]

# Per-conv static info. in_words = IH*IW*passes (loader fill), out_beats =
# OH*OW*passes (engine FIFO beats). succ = the project conv whose dispatch
# slot the DW conv is inserted before; prev = preceding dispatch module
# (bridge insertion anchor).
DW = {
    "818": dict(c=96,  ih=112, oh=56, passes=1, kind="flat",  bus=768,
                prod="n4_3",  cons="n4_4",  prev="816", succ="820",
                act_in=12544, act_out=0,     wbase=18536, bbase=91),
    "830": dict(c=144, ih=56,  oh=28, passes=1, kind="flat",  bus=1152,
                prod="n4_7",  cons="n4_8",  prev="828", succ="832",
                act_in=9368,  act_out=12504, wbase=18545, bbase=92),
    "848": dict(c=192, ih=28,  oh=14, passes=1, kind="flat",  bus=1536,
                prod="n4_13", cons="n4_14", prev="846", succ="850",
                act_in=15640, act_out=16424, wbase=18554, bbase=93),
    "890": dict(c=576, ih=14,  oh=7,  passes=3, kind="tiled", bus=256,
                prod="n4_27", cons="n4_28", prev="888", succ="892",
                act_in=21912, act_out=22500, wbase=18563, bbase=94),
}
for _k, _v in DW.items():
    _v["in_words"] = _v["ih"] * _v["ih"] * _v["passes"]
    _v["out_beats"] = _v["oh"] * _v["oh"] * _v["passes"]


def die(msg: str) -> None:
    print(f"[dw-q] FATAL: {msg}", file=sys.stderr)
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


def build_q_modules(convs: list[str]) -> list[str]:
    """FC order with each selected DW conv inserted before its `succ`."""
    out = []
    by_succ = {DW[c]["succ"]: c for c in convs}
    for m in MODULES_FC:
        if m in by_succ:
            out.append(by_succ[m])
        out.append(m)
    assert len(out) == len(MODULES_FC) + len(convs)
    return out


def bridge_inst(mod: str) -> str:
    return ("u_engine_out_node_linear" if mod == "linear"
            else f"u_engine_out_node_conv_{mod}")


# ============================================================================
# Pre-flight
# ============================================================================
def preflight(convs: list[str]) -> None:
    for f, needle in [
        (REPO / "output/rtl/engine/config_register_block.v", "reg_depthwise"),
        (REPO / "output/rtl/engine/address_generator.v", "cfg_depthwise"),
        (REPO / "output/rtl/engine/address_generator.v", "cfg_stride_h"),
        (REPO / "output/rtl/engine/mac_array.v", "dw_mode"),
        (REPO / "output/rtl/shared_engine_skeleton.v", "ENABLE_DEPTHWISE"),
    ]:
        if needle not in f.read_text(encoding="utf-8"):
            die(f"engine-core edit missing: {f.name} lacks '{needle}'")
    for b in range(8):
        n = sum(1 for _ in (WDIR / f"uram_weights_bank{b}_kp8.mem").open())
        if n != 2324:
            die(f"bank{b}_kp8 has {n} lines (need 2324) — run extend_mbv2_engine_maps_dw_quartet.py")
    for f in ["bias.mem", "scale.mem"]:
        n = sum(1 for _ in (WDIR / f).open())
        if n != 97:
            die(f"{f} has {n} lines (need 97) — run extend_mbv2_engine_maps_dw_quartet.py")
    # wrapper geometry must match the table (C, IH=2*OH, stride 2, 3x3, pad 1)
    for c in convs:
        t = (REPO / f"output/mobilenet-v2/rtl/node_conv_{c}.v").read_text(encoding="utf-8")
        lp = dict(re.findall(r"localparam integer (\w+)\s*=\s*(\d+);", t))
        want = DW[c]
        checks = [("C", want["c"]), ("IH", want["ih"]), ("IW", want["ih"]),
                  ("OH", want["oh"]), ("OW", want["oh"]), ("SH", 2), ("SW", 2),
                  ("KH", 3), ("KW", 3), ("PH", 1), ("PW", 1)]
        for k, v in checks:
            if k in lp and int(lp[k]) != v:
                die(f"conv_{c} wrapper {k}={lp[k]} != expected {v}")
    print(f"[dw-q] preflight OK (engine-core + quartet maps + wrapper geometry, "
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
        .TOTAL_BRAM_WORDS({i["in_words"]})
    ) u_ldr_node_conv_{conv} (
        .clk(clk), .rst_n(rst_n),
        .in_valid({i["prod"]}_valid_out & spatial_run),
        .in_data({i["prod"]}_data_out),'''
    else:
        inst = f'''    stream_to_act_bram_bridge #(
        .BUS_W({i["bus"]}),
        .BRAM_BASE_ADDR({i["act_in"]}),
        .TOTAL_BRAM_WORDS({i["in_words"]})
    ) u_ldr_node_conv_{conv} (
        .clk(clk), .rst_n(rst_n),
        .in_valid({i["prod"]}_valid_out & spatial_run),
        .in_data({i["prod"]}_data_out),'''
    return f'''
    // {MARK} STRIDE-2 DW input loader: relu {i["prod"]}'s stream -> the FULL
    // {i["ih"]}x{i["ih"]} input image, act words {i["act_in"]}..{i["act_in"] + i["in_words"] - 1},
    // for the conv_{conv} DEPTHWISE engine dispatch (engine reads rows 2*oh_r..+2).
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
        kind_params = (f",\n        .OC({i['c']}), .OUT_KIND(1), .POSITIONS({i['oh'] * i['oh']})")
        geom = f"{i['c'] // 32}x256b tile stream"
    else:
        kind_params = ""
        geom = f"flat {i['bus']}b low-slice (1 beat/pos)"
    return f'''
    wire u_engine_out_node_conv_{conv}_fifo_ready;
    wire u_engine_out_node_conv_{conv}_drain_complete;
    // {MARK} conv_{conv} STRIDE-2 DEPTHWISE dispatch {slot}: re-emit the engine's
    // {i["oh"]}x{i["oh"]} output as the {geom} the (unchanged) relu {i["cons"]} consumes.
    engine_output_bridge #(
        .SLOT({slot}),
        .ACT_W(2048),
        .DATA_W({i["bus"]}),
        .EXPECTED_BEATS({i["out_beats"]}),
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
def patch_top(convs: list[str], modules_q: list[str]) -> None:
    text = TOP.read_text(encoding="utf-8")
    n = len(convs)
    ndisp = 47 + n
    new_idx = {m: i for i, m in enumerate(modules_q)}

    # ---- header bookkeeping ----
    text = rep(text,
        "// Layers total: 99, spatial: 52, engine-dispatched: 47 ([DW-ENGINE P1] 896/902/908 + [DW-ENGINE EXT] 824/836/842/854/860/866/872/878/884 depthwise dispatches + [FC-ENGINE] node_linear dense dispatch 46), residual adds: 10, projection convs: 11.",
        f"// Layers total: 99, spatial: {52 - n}, engine-dispatched: {ndisp} "
        f"([DW-ENGINE P1]+[DW-ENGINE EXT] 12 stride-1 DW + {MARK} stride-2 DW {'/'.join(convs)} "
        f"+ [FC-ENGINE] node_linear), residual adds: 10, projection convs: 11.",
        "header layer counts")

    # ---- parse the FC-state all_loaded / all_drain rows (old idx -> signal) ----
    old_loaded = {}
    old_drain = {}
    for d in range(47):
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
            text = rep(text,
                f"    wire [255:0] node_conv_{conv}_data_out;  // [NATIVE_TILED_{conv}] narrowed: native 256b tile bus\n"
                f"    wire node_conv_{conv}_ready_in;\n",
                f"    wire [255:0] node_conv_{conv}_data_out;  // 256b tile bus, driven by engine_output_bridge SLOT {slot} ({MARK})\n",
                f"wire decls {conv}")
            text = rep(text,
                f"        .out_ready_in(node_conv_{conv}_ready_in),\n"
                f"        .valid_out({i['prod']}_valid_out),",
                f"        // {MARK} conv_{conv} is engine dispatch {slot}: this tile stream\n"
                f"        // fills the DW input loader (u_ldr_node_conv_{conv}).\n"
                f"        .out_ready_in(ldr_dw{conv}_in_ready & spatial_run),\n"
                f"        .valid_out({i['prod']}_valid_out),",
                f"producer {i['prod']} retarget ({conv})")
            text = excise_block(text,
                f"    node_conv_{conv} #(.ENABLE_BACKPRESSURE(1), .NATIVE_TILED(1)) u_node_conv_{conv} (",
                "\n    );\n",
                f"    // node_conv_{conv}: engine-dispatched ({MARK} STRIDE-2 DEPTHWISE dispatch {slot}; "
                f"data_out driven by shared_engine via engine_output_bridge SLOT {slot})\n",
                f"u_node_conv_{conv} instance")
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
                f"    // node_conv_{conv}: engine-dispatched ({MARK} STRIDE-2 DEPTHWISE dispatch {slot}; "
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

    # ---- eofifo_out_ready: one term per new bridge (before the FC term) ----
    fifo_terms = "".join(f" | u_engine_out_node_conv_{c}_fifo_ready" for c in convs)
    text = rep(text,
        " | u_engine_out_node_linear_fifo_ready;",
        f"{fifo_terms} | u_engine_out_node_linear_fifo_ready;",
        "eofifo_out_ready")

    # ---- arbiter: grants + en/addr/data mux terms (after the FC loader) ----
    m = re.search(r"    assign ldr_fc_wr_grant = ldr_fc_wr_req & ~\(([^)]+)\);\n", text)
    if not m:
        die("ldr_fc grant line not found")
    accum = m.group(1) + " | ldr_fc_wr_req"
    grants = (f"    // {MARK} stride-2 DW input loaders (lowest priority; the spatial\n"
              "    // chain is serial so at most one loader is ever active at a time).\n")
    for conv in convs:
        grants += f"    assign ldr_dw{conv}_wr_grant = ldr_dw{conv}_wr_req & ~({accum});\n"
        accum += f" | ldr_dw{conv}_wr_req"
    text = rep(text, m.group(0), m.group(0) + grants, "arbiter grants")
    en_terms = "".join(f" | ldr_dw{c}_wr_req" for c in convs)
    text = rep(text,
        " | ldr_fc_wr_req;\n",
        f" | ldr_fc_wr_req{en_terms};\n",
        "act_wr_en_final")
    addr_terms = "".join(f"ldr_dw{c}_wr_req ? ldr_dw{c}_wr_addr : " for c in convs)
    text = rep(text,
        "ldr_fc_wr_req ? ldr_fc_wr_addr : 15'd0;",
        f"ldr_fc_wr_req ? ldr_fc_wr_addr : {addr_terms}15'd0;",
        "act_wr_addr_final mux")
    data_terms = "".join(f"ldr_dw{c}_wr_req ? ldr_dw{c}_wr_data : " for c in convs)
    text = rep(text,
        "ldr_fc_wr_req ? ldr_fc_wr_data : 2048'd0;",
        f"ldr_fc_wr_req ? ldr_fc_wr_data : {data_terms}2048'd0;",
        "act_wr_data_final mux")

    # ---- all_loaded / all_drain renumber (rows 0..63 rebuilt) ----
    new_loaded = {d: "1'b1" for d in range(64)}
    new_drain = {d: "1'b1" for d in range(64)}
    fc_idx = {m_: i_ for i_, m_ in enumerate(MODULES_FC)}
    for d, mod in enumerate(modules_q):
        if mod in convs:
            new_loaded[d] = f"ldr_dw{mod}_loaded"
            new_drain[d] = f"u_engine_out_node_conv_{mod}_drain_complete"
        else:
            new_loaded[d] = old_loaded[fc_idx[mod]]
            new_drain[d] = old_drain[fc_idx[mod]]

    def rebuild(vec: str, rows: dict[int, str]) -> None:
        nonlocal text
        pat = re.compile(rf"    assign {vec}\[0\] = [\s\S]*?    assign {vec}\[63\] = 1'b1;\n")
        mm = pat.search(text)
        if not mm:
            die(f"{vec} block not found")
        body = (f"    // {MARK} renumbered for {ndisp} dispatches "
                f"(stride-2 DW inserts: {', '.join(f'{c}@{new_idx[c]}' for c in convs)})\n")
        body += "".join(f"    assign {vec}[{d}] = {rows[d]};\n" for d in range(64))
        text = text[:mm.start()] + body + text[mm.end():]

    rebuild("all_loaded", new_loaded)
    rebuild("all_drain", new_drain)

    # ---- SLOT renumber on every pre-existing bridge ----
    for mod in MODULES_FC:
        old_slot = fc_idx[mod]
        ns = new_idx[mod]
        inst = bridge_inst(mod)
        pat = re.compile(
            rf"\.SLOT\((\d+)\),[^\n]*(?=[\s\S]{{0,400}}?\) {inst} \()")
        mm = pat.search(text)
        if not mm:
            die(f"bridge SLOT for {mod} not found")
        if int(mm.group(1)) != old_slot:
            die(f"bridge {mod}: SLOT {mm.group(1)} != expected FC slot {old_slot}")
        repl = f".SLOT({ns})," + (f"  // {MARK} was {old_slot}" if ns != old_slot else "")
        text = text[:mm.start()] + repl + text[mm.end():]

    # ---- NUM_DISPATCHES on the 47 pre-existing bridges ----
    text = rep(text, ".NUM_DISPATCHES(47)", f".NUM_DISPATCHES({ndisp})",
               "NUM_DISPATCHES", count=47)

    # ---- weight banks: kp8 depth 2317 -> 2324 ----
    text = rep(text,
        "    // Total MAC cycles = 18533; per-bank depth = 18533. ([FC-ENGINE] +5120 dense FC\n"
        "    // words appended after the DW-EXT set: node_linear@13413 = 4 oc_passes x 1280 taps.\n"
        "    // Depth now exceeds 2^14 -> bank ADDR_W 14->15 and the rd_addr slice [13:0]->[14:0].)",
        f"    // Total MAC cycles = 18587 (old domain; +54 {MARK} stride-2 DW words appended\n"
        "    // after the FC region: 818@18533 830@18542 848@18551 890@18560 = oc_passes x 9\n"
        "    // taps each; RELOCATED kp8 bases = +3 FC-pad: 818@18536 830@18545 848@18554\n"
        "    // 890@18563). kp8 wide lines: (18536+54+2 zero-tail)/8 = 2324 per bank.",
        "bank depth comment")
    text = rep(text,
        ".DEPTH(2317),           // [KPAR8 2026-06-10] (18533+3 FC pad)/8 wide lines",
        f".DEPTH(2324),           // [KPAR8] + {MARK}: (18533+3 FC pad+54 DW+2 tail)/8 wide lines",
        "bank DEPTH", count=8)

    verify_top(text, convs, modules_q)
    TOP.write_text(text, encoding="utf-8", newline="\n")
    print(f"[dw-q] top patched + verified ({n} convs, {ndisp} dispatches)")


def verify_top(text: str, convs: list[str], modules_q: list[str]) -> None:
    ndisp = len(modules_q)
    checks = [
        (f"NUM_DISPATCHES({ndisp})", text.count(f".NUM_DISPATCHES({ndisp})"), ndisp),
        ("NUM_DISPATCHES(47) gone", text.count(".NUM_DISPATCHES(47)"), 0),
        ("bank DEPTH 2324", text.count(".DEPTH(2324),"), 8),
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
    new_idx = {m: i for i, m in enumerate(modules_q)}
    for d, mod in enumerate(modules_q):
        ml = re.search(rf"    assign all_loaded\[{d}\] = (\S+?);", text)
        md = re.search(rf"    assign all_drain\[{d}\] = (\S+?);", text)
        if not ml or not md:
            die(f"all_loaded/all_drain[{d}] missing after rebuild")
        want_drain = (f"{bridge_inst(mod)}_drain_complete")
        if md.group(1) != want_drain:
            die(f"all_drain[{d}] = {md.group(1)} != {want_drain}")
        if mod in convs and ml.group(1) != f"ldr_dw{mod}_loaded":
            die(f"all_loaded[{d}] = {ml.group(1)} != ldr_dw{mod}_loaded")
    print(f"[dw-q] top invariants OK ({ndisp} slots, bridges/loaders/banks consistent)")


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
        "stride_h_rom": "3'd2", "stride_w_rom": "3'd2",   # STRIDE-2
        "padding_h_rom": "3'd1", "padding_w_rom": "3'd1",
        "input_h_rom": f"9'd{i['ih']}", "input_w_rom": f"9'd{i['ih']}",
        "output_h_rom": f"9'd{i['oh']}", "output_w_rom": f"9'd{i['oh']}",
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


def patch_scheduler(convs: list[str], modules_q: list[str]) -> None:
    text = SCHED.read_text(encoding="utf-8")
    n = len(convs)
    ndisp = 47 + n
    new_idx = {m: i for i, m in enumerate(modules_q)}
    fc_idx = {m: i for i, m in enumerate(MODULES_FC)}

    text = rep(text,
        "// Number of engine dispatches: 47  ([DW-ENGINE P1] + [DW-ENGINE EXT] depthwise @ 824@4, 836@9, 842@12, 854@17, 860@20, 866@23, 872@26, 878@29, 884@32, 896@37, 902@40, 908@43 + [FC-ENGINE] node_linear dense @ 46)",
        f"// Number of engine dispatches: {ndisp}  ({MARK} stride-2 depthwise @ "
        + ", ".join(f"{m}@{new_idx[m]}" for m in modules_q if m in convs)
        + " + [DW-ENGINE P1/EXT] stride-1 depthwise @ "
        + ", ".join(f"{m}@{new_idx[m]}" for m in modules_q if m in FC_DW)
        + f" + [FC-ENGINE] node_linear dense @ {ndisp - 1})",
        "sched header")

    # rebuild each ROM case body with `ndisp` entries (regex-span replace —
    # rows may carry trailing comments in the FC state)
    for name in ROM_NAMES:
        entries = re.findall(rf"6'd(\d+): {name} = ([^;]+);", text)
        if len(entries) != 47:
            die(f"scheduler ROM {name}: found {len(entries)} entries, want 47")
        old = {int(i_): v for i_, v in entries}
        new = {}
        for d, mod in enumerate(modules_q):
            if mod in convs:
                new[d] = dw_rom_value(name, mod)
            else:
                new[d] = old[fc_idx[mod]]
        pat = re.compile(rf"            6'd0: {name} = [\s\S]*?\n(?=            default:)")
        mm = pat.search(text)
        if not mm:
            die(f"ROM {name}: case body span not found")
        body_new = "".join(f"            6'd{i_}: {name} = {new[i_]};\n"
                           for i_ in range(ndisp))
        text = text[:mm.start()] + body_new + text[mm.end():]

    # depthwise_rom: rebuild the 1'b1 rows (12 stride-1 renumbered + the new set)
    dw_idx = sorted(new_idx[m] for m in modules_q if m in convs or m in FC_DW)
    body_new = "".join(f"            6'd{d}: depthwise_rom = 1'b1;\n" for d in dw_idx)
    pat = re.compile(r"(?:            6'd\d+: depthwise_rom = 1'b1;\n)+")
    mm = pat.search(text)
    if not mm:
        die("depthwise_rom rows not found")
    text = text[:mm.start()] + body_new + text[mm.end():]
    text = rep(text,
        "    // 1 for every depthwise dispatch ([DW-ENGINE EXT]): 824@4, 836@9, 842@12, 854@17, 860@20, 866@23, 872@26, 878@29, 884@32, 896@37, 902@40, 908@43.",
        f"    // 1 for every depthwise dispatch ({MARK}): "
        + ", ".join(f"{m}@{new_idx[m]}" for m in modules_q if m in convs or m in FC_DW) + ".",
        "depthwise_rom comment")

    text = rep(text,
        "    localparam [5:0] LAST_DISPATCH = 6'd46;  // [FC-ENGINE] 47 dispatches",
        f"    localparam [5:0] LAST_DISPATCH = 6'd{ndisp - 1};  // {MARK} {ndisp} dispatches",
        "LAST_DISPATCH")

    verify_sched(text, convs, modules_q)
    SCHED.write_text(text, encoding="utf-8", newline="\n")
    print(f"[dw-q] scheduler patched + verified ({ndisp}-entry ROMs)")


def verify_sched(text: str, convs: list[str], modules_q: list[str]) -> None:
    ndisp = len(modules_q)
    new_idx = {m: i for i, m in enumerate(modules_q)}
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
            (f"6'd{d}: stride_h_rom = 3'd2;", 1),
            (f"6'd{d}: stride_w_rom = 3'd2;", 1),
            (f"6'd{d}: input_h_rom = 9'd{i['ih']};", 1),
            (f"6'd{d}: output_h_rom = 9'd{i['oh']};", 1),
            (f"6'd{d}: weight_base_word_rom = 20'd{i['wbase']};", 1),
            (f"6'd{d}: bias_base_word_rom = 16'd{i['bbase']};", 1),
            (f"6'd{d}: act_in_base_word_rom = 16'd{i['act_in']};", 1),
            (f"6'd{d}: act_out_base_word_rom = 16'd{i['act_out']};", 1),
            (f"6'd{d}: depthwise_rom = 1'b1;", 1),
        ]
    # stride-2 rows are EXACTLY the quartet rows
    s2 = len(re.findall(r"stride_h_rom = 3'd2;", text))
    if s2 != len(convs):
        die(f"sched verify stride-2 rows: {s2} != {len(convs)}")
    for m in sorted(FC_DW):
        spot.append((f"6'd{new_idx[m]}: depthwise_rom = 1'b1;", 1))
    spot.append((f"6'd{ndisp - 1}: channel_out_rom = 16'd1000;", 1))   # node_linear
    dw_count = len(re.findall(r"depthwise_rom = 1'b1;", text))
    if dw_count != len(convs) + 12:
        die(f"sched verify depthwise_rom: {dw_count} ones, want {len(convs) + 12}")
    for needle, want in spot:
        if text.count(needle) != want:
            die(f"sched verify: '{needle}' count {text.count(needle)} != {want}")
    print(f"[dw-q] scheduler invariants OK ({ndisp} entries, "
          f"{len(convs) + 12} depthwise rows, {len(convs)} stride-2 rows)")


# ============================================================================
# main
# ============================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", default=",".join(ALL_Q),
                    help="comma list of conv ids to move (default: all 4). "
                         "STAGE 1 = 830,848,890; STAGE 2 adds 818.")
    args = ap.parse_args()
    convs = [c.strip() for c in args.convs.split(",") if c.strip()]
    for c in convs:
        if c not in DW:
            die(f"unknown conv {c}; valid: {ALL_Q}")
    convs = [c for c in ALL_Q if c in convs]   # canonical order

    preflight(convs)

    # declarative: restore the FC/ENG-PIPE state from .prequartet if applied
    for f in [TOP, SCHED]:
        backup = f.with_suffix(f.suffix + ".prequartet")
        cur = f.read_text(encoding="utf-8")
        if MARK in cur:
            if not backup.exists():
                die(f"{f.name} already patched but {backup.name} missing")
            f.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
            print(f"[dw-q] {f.name}: restored FC baseline from {backup.name}")
        else:
            if FCMARK not in cur:
                die(f"{f.name} lacks {FCMARK} — base must be the FC-ENGINE state")
            if not backup.exists():
                backup.write_text(cur, encoding="utf-8", newline="\n")
                print(f"[dw-q] {f.name}: saved FC baseline to {backup.name}")

    modules_q = build_q_modules(convs)
    print(f"[dw-q] QUARTET schedule ({len(modules_q)} dispatches): "
          + " ".join(f"{m}@{i}" + ("*" if m in convs else "")
                     for i, m in enumerate(modules_q) if m in convs))
    patch_top(convs, modules_q)
    patch_scheduler(convs, modules_q)
    print("[dw-q] DONE — next: verilator lint, "
          "scripts/check_mbv2_act_region_hazards_quartet.py, engine-ISO, 8/8 e2e gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
