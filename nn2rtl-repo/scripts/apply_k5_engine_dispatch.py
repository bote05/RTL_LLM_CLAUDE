#!/usr/bin/env python3
"""[K5 ENGINE-DISPATCH 2026-06-09] Move node_conv_284/292/298 from SPATIAL fabric
onto the SHARED ENGINE (dispatches 9/12/15), mirroring the proven conv_246 pattern.
Deletes the ~200K-LUT congestion wall (the 3 lbw chan-select muxes) + ~1161 BRAM
from the netlist; engine is also FASTER for these (-~1.07M cycles).

Prereqs (already done, all verified):
  banks 17-dispatch (dedup_engine_banks_k5.py, PROOF vs live), bias/scale maps 17,
  scheduler regenerated 17/5-bit + weight_base_word_rom patched to dedup bases.

Top edits (per conv c in {284: relu_40->skid_relu_41, 292: relu_43->skid_relu_44,
298: relu_46->skid_relu_47}):
  1. delete input skid u_skid_node_conv_c + wires; delete spatial inst u_node_conv_c;
     delete node_conv_c_ready_in decl (valid_out/data_out decls STAY - bridge drives)
  2. upstream relu .ready_out: skid_node_conv_c_in_ready -> ldr_node_conv_c_in_ready
  3. add loader u_ldr_node_conv_c (ldr14/15/16, BRAM_BASE_ADDR=8192 [bank2],
     TOTAL_BRAM_WORDS=392/98/98) fed by the upstream relu
  4. add bridge u_engine_out_node_conv_c (SLOT=9/12/15, EXPECTED_BEATS=98,
     NUM_DISPATCHES=17) driving node_conv_c_valid_out/data_out
Global edits:
  5. existing bridges SLOT renumber 286:9->10 290:10->11 294:11->13 296:12->14
     300:13->16; ALL NUM_DISPATCHES 14->17
  6. all_loaded/all_drain -> 17 wide, remapped to NEW dispatch indices
  7. sched_dispatch_idx [3:0]->[4:0] (decl + current_loaded index)
  8. arbiter: +ldr14/15/16 grants; act wr addr/data muxes + wr_en chain extended
  9. eofifo_out_ready OR-chain += 3 bridges
 10. engine weight bank DEPTH 39424 -> 67072 (x8)

Usage: python scripts/apply_k5_engine_dispatch.py [--dry-run]
"""
from __future__ import annotations
import re, sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOP = ROOT / "output" / "rtl" / "nn2rtl_top.v"
BK = ROOT / "backups" / "k5_engine_dispatch_20260609" / "nn2rtl_top.pre_surgery.v"
MARK = "// [K5-ENGINE-DISPATCH]"

# conv: (upstream_relu, downstream_skid_ready, slot, total_bram_words, ldr_idx)
CONVS = [
    ("node_conv_284", "node_relu_40", "skid_node_relu_41_ready", 9, 392, 14),
    ("node_conv_292", "node_relu_43", "skid_node_relu_44_ready", 12, 98, 15),
    ("node_conv_298", "node_relu_46", "skid_node_relu_47_ready", 15, 98, 16),
]
# existing bridge SLOT renumber (old dispatch idx -> new)
SLOT_RENUM = {"node_conv_286": 10, "node_conv_290": 11, "node_conv_294": 13,
              "node_conv_296": 14, "node_conv_300": 16}
# module -> NEW dispatch index (for all_drain reindex)
NEW_IDX = {"node_conv_246": 0, "node_conv_250": 1, "node_conv_254": 2, "node_conv_260": 3,
           "node_conv_264": 4, "node_conv_266": 5, "node_conv_272": 6, "node_conv_278": 7,
           "node_conv_282": 8, "node_conv_284": 9, "node_conv_286": 10, "node_conv_290": 11,
           "node_conv_292": 12, "node_conv_294": 13, "node_conv_296": 14, "node_conv_298": 15,
           "node_conv_300": 16}


def need(cond, msg):
    if not cond:
        print(f"[k5] ABORT (no write): {msg}")
        sys.exit(1)


def sub1(t, pat, rep, msg, flags=0):
    new, n = re.subn(pat, rep, t, count=1, flags=flags)
    need(n == 1, f"anchor not found: {msg}")
    return new


def main() -> int:
    dry = "--dry-run" in sys.argv
    t = TOP.read_text()
    need(MARK not in t, "already applied (marker present)")
    shutil.copy(TOP, BK)
    print(f"[k5] backup -> {BK}")

    loaders_add, bridges_add = [], []

    for conv, up_relu, dn_ready, slot, words, li in CONVS:
        # 1a. delete the input skid block (wires + skip_fifo inst)
        skid_pat = (r"\n\s*wire skid_" + conv + r"_in_ready, skid_" + conv + r"_valid;\s*\n"
                    r"\s*wire \[255:0\] skid_" + conv + r"_data;\s*\n"
                    r"\s*skip_fifo #\(.*?\) u_skid_" + conv + r" \(.*?\);\n")
        t = sub1(t, skid_pat, "\n", f"{conv} input skid", flags=re.DOTALL)
        # 1b. delete the spatial instantiation
        inst_pat = r"\n\s*" + conv + r" u_" + conv + r" \(.*?\);\n"
        t = sub1(t, inst_pat,
                 f"\n    {MARK} {conv}: engine-dispatched (slot {slot}); data_out driven by "
                 f"u_engine_out_{conv}; input loaded by u_ldr_{conv}\n",
                 f"{conv} spatial inst", flags=re.DOTALL)
        # 1c. delete the now-unused ready_in decl
        t = sub1(t, r"\n\s*wire " + conv + r"_ready_in;", "", f"{conv}_ready_in decl")
        # 2. upstream relu ready_out -> loader in_ready
        t = sub1(t, r"\.ready_out\(skid_" + conv + r"_in_ready & spatial_run\)",
                 f".ready_out(ldr_{conv}_in_ready & spatial_run)", f"{up_relu} ready_out")
        # 3. loader block (mirrors u_ldr_node_conv_246; bank 2 base 8192)
        loaders_add.append(f"""
    {MARK} loader for {conv} (dispatch {slot}, input bank 2)
    wire        ldr{li}_wr_req;
    wire        ldr{li}_wr_grant;
    wire [14:0] ldr{li}_wr_addr;
    wire [2047:0] ldr{li}_wr_data;
    wire        ldr{li}_loaded;
    wire ldr_{conv}_in_ready;
    stream_to_act_bram_bridge #(
        .BUS_W(256),
        .BRAM_BASE_ADDR(8192),
        .TOTAL_BRAM_WORDS({words})
    ) u_ldr_{conv} (
        .clk(clk), .rst_n(rst_n),
        .in_valid({up_relu}_valid_out & spatial_run & ldr_{conv}_in_ready),
        .in_data({up_relu}_data_out),
        .in_ready(ldr_{conv}_in_ready),
        .wr_req(ldr{li}_wr_req),
        .wr_grant(ldr{li}_wr_grant),
        .wr_addr(ldr{li}_wr_addr),
        .wr_data(ldr{li}_wr_data),
        .loaded(ldr{li}_loaded)
    );
""")
        # 4. output bridge (mirrors u_engine_out_node_conv_290; 7x7x512 out = 98 beats)
        bridges_add.append(f"""
    {MARK} output bridge for {conv} (dispatch {slot})
    wire u_engine_out_{conv}_fifo_ready;
    wire u_engine_out_{conv}_drain_complete;
    engine_output_bridge #(
        .SLOT({slot}),
        .ACT_W(2048),
        .DATA_W(256),
        .EXPECTED_BEATS(98),
        .NUM_DISPATCHES(17)
    ) u_engine_out_{conv} (
        .clk(clk), .rst_n(rst_n),
        .start(sched_engine_output_ready),
        .fifo_out_valid(eofifo_out_valid),
        .fifo_out_data(eofifo_out_data),
        .fifo_out_ready(u_engine_out_{conv}_fifo_ready),
        .ready_out(({dn_ready} & spatial_run)),
        .valid_out({conv}_valid_out),
        .data_out({conv}_data_out),
        .drain_complete(u_engine_out_{conv}_drain_complete)
    );
""")

    # 5a. SLOT renumber on existing bridges (scoped per bridge block)
    for mod, new_slot in SLOT_RENUM.items():
        blk_pat = r"(engine_output_bridge #\(\s*\.SLOT\()(\d+)(\),[^;]*?u_engine_out_" + mod + r" \()"
        m = re.search(blk_pat, t, flags=re.DOTALL)
        need(m, f"bridge block for {mod}")
        t = t[:m.start(2)] + str(new_slot) + t[m.end(2):]
        print(f"  SLOT {mod}: {m.group(2)} -> {new_slot}")
    # 5b. NUM_DISPATCHES 14 -> 17 on all existing bridges
    t, n = re.subn(r"\.NUM_DISPATCHES\(14\)", ".NUM_DISPATCHES(17)", t)
    need(n == 14, f"NUM_DISPATCHES(14) count {n} != 14")

    # 6a. all_loaded: widen ([15:0] w/ [15:14] tied 1) + remap to new dispatch indices
    t = sub1(t, r"wire \[15:0\] all_loaded;", "wire [16:0] all_loaded;", "all_loaded decl")
    lm = dict(re.findall(r"assign all_loaded\[(\d+)\]\s*=\s*(ldr\d+)_loaded;", t))
    need(len(lm) == 14, f"all_loaded assigns {len(lm)} != 14")
    # old dispatch idx -> ldr wire (identity for 0..13); rebuild by NEW idx
    old2ldr = {int(k): v for k, v in lm.items()}
    shift = {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:10,10:11,11:13,12:14,13:16}
    new_assign = {}
    for oi, ni in shift.items():
        new_assign[ni] = old2ldr[oi]
    for conv, _, _, slot, _, li in CONVS:
        new_assign[slot] = f"ldr{li}"
    block = "\n".join(f"    assign all_loaded[{i}] = {new_assign[i]}_loaded;" for i in range(17))
    t = sub1(t, r"(\n\s*assign all_loaded\[\d+\]\s*=\s*ldr\d+_loaded;)+",
             "\n" + block, "all_loaded assign block")
    t = sub1(t, r"\n\s*assign all_loaded\[14\] = 1'b1;\s*\n\s*assign all_loaded\[15\] = 1'b1;",
             "", "all_loaded tie-offs (now real dispatch bits)")
    # 6b. all_drain: widen + reindex by module name
    t = sub1(t, r"wire \[15:0\] all_drain;", "wire [16:0] all_drain;", "all_drain decl")
    dm = re.findall(r"assign all_drain\[(\d+)\]\s*=\s*u_engine_out_(node_conv_\d+)_drain_complete;", t)
    need(len(dm) == 14, f"all_drain assigns {len(dm)} != 14")
    dblock = "\n".join(
        f"    assign all_drain[{NEW_IDX[mod]}] = u_engine_out_{mod}_drain_complete;"
        for _, mod in sorted(dm, key=lambda x: NEW_IDX[x[1]])) + "\n" + "\n".join(
        f"    assign all_drain[{slot}] = u_engine_out_{conv}_drain_complete;"
        for conv, _, _, slot, _, _ in CONVS)
    t = sub1(t, r"(\n\s*assign all_drain\[\d+\]\s*=\s*u_engine_out_node_conv_\d+_drain_complete;)+",
             "\n" + dblock, "all_drain assign block")
    t = sub1(t, r"\n\s*assign all_drain\[14\] = 1'b1;\s*\n\s*assign all_drain\[15\] = 1'b1;",
             "", "all_drain tie-offs (now real dispatch bits)")

    # 7. dispatch idx width (decl + BOTH muxes + debug shadow regs)
    t = sub1(t, r"wire \[3:0\]\s*sched_dispatch_idx;", "wire [4:0] sched_dispatch_idx;",
             "sched_dispatch_idx decl")
    t = sub1(t, r"all_loaded\[sched_dispatch_idx\[3:0\]\]", "all_loaded[sched_dispatch_idx[4:0]]",
             "current_loaded index")
    t = sub1(t, r"all_drain\[sched_dispatch_idx\[3:0\]\]", "all_drain[sched_dispatch_idx[4:0]]",
             "current_drain_complete index")
    t = sub1(t, r"reg \[15:0\] all_loaded_d;", "reg [16:0] all_loaded_d;", "all_loaded_d shadow")
    t = sub1(t, r"reg \[15:0\] all_drain_d;", "reg [16:0] all_drain_d;", "all_drain_d shadow")

    # 8a. arbiter grants for ldr14/15/16 (chained priority after ldr13)
    base13 = ("engine_act_out_wr_en | ldr0_wr_req | ldr1_wr_req | ldr2_wr_req | ldr3_wr_req | "
              "ldr4_wr_req | ldr5_wr_req | ldr6_wr_req | ldr7_wr_req | ldr8_wr_req | ldr9_wr_req | "
              "ldr10_wr_req | ldr11_wr_req | ldr12_wr_req")
    anchor13 = f"assign ldr13_wr_grant = ldr13_wr_req & ~({base13});"
    need(anchor13 in t, "ldr13 grant anchor")
    grants = anchor13
    prev = base13 + " | ldr13_wr_req"
    for i in (14, 15, 16):
        grants += f"\n    assign ldr{i}_wr_grant = ldr{i}_wr_req & ~({prev});"
        prev += f" | ldr{i}_wr_req"
    t = t.replace(anchor13, grants, 1)
    # 8b. act write addr/data muxes: insert new terms before the constant tail
    t = sub1(t, r"ldr13_wr_req \? ldr13_wr_addr : 15'd0;",
             "ldr13_wr_req ? ldr13_wr_addr : ldr14_wr_req ? ldr14_wr_addr : "
             "ldr15_wr_req ? ldr15_wr_addr : ldr16_wr_req ? ldr16_wr_addr : 15'd0;",
             "act wr_addr mux tail")
    t = sub1(t, r"ldr13_wr_req \? ldr13_wr_data : 2048'd0;",
             "ldr13_wr_req ? ldr13_wr_data : ldr14_wr_req ? ldr14_wr_data : "
             "ldr15_wr_req ? ldr15_wr_data : ldr16_wr_req ? ldr16_wr_data : 2048'd0;",
             "act wr_data mux tail")
    # 8c. wr_en chain (engine | ldrN_wr_req ...)
    men = re.search(r"assign act_wr_en_final\s*=\s*([^;]+);", t)
    need(men, "act_wr_en_final assign")
    need("ldr13_wr_req" in men.group(1) and "ldr14_wr_req" not in men.group(1), "wr_en chain shape")
    t = t[:men.end(1)] + " | ldr14_wr_req | ldr15_wr_req | ldr16_wr_req" + t[men.end(1):]

    # 9. eofifo_out_ready OR-chain += 3
    t = sub1(t, r"(assign eofifo_out_ready = [^;]*u_engine_out_node_conv_300_fifo_ready)",
             r"\1 | u_engine_out_node_conv_284_fifo_ready | u_engine_out_node_conv_292_fifo_ready | "
             r"u_engine_out_node_conv_298_fifo_ready", "eofifo_out_ready chain")

    # 10. engine weight bank DEPTH x8
    t, n = re.subn(r"\.DEPTH\(39424\)", ".DEPTH(67072)", t)
    need(n == 8, f"bank DEPTH count {n} != 8")

    # insert loader blocks before the arbiter section, bridges before eofifo assign
    t = sub1(t, r"\n(\s*// ----- act BRAM write arbiter)", "\n" + "\n".join(loaders_add) + r"\n\1",
             "loader insertion point")
    t = sub1(t, r"\n(\s*assign eofifo_out_ready = )", "\n" + "\n".join(bridges_add) + r"\n\1",
             "bridge insertion point")

    if dry:
        print("[k5] DRY RUN ok — all anchors matched")
        return 0
    TOP.write_text(t, newline="\n")
    print("[k5] applied. Verify: verilator lint -> iso conv_284 -> e2e 0/100352")
    return 0


if __name__ == "__main__":
    sys.exit(main())
