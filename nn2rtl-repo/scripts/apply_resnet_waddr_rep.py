#!/usr/bin/env python3
"""apply_resnet_waddr_rep.py — [WADDR-REP 2026-06-11] per-bank replication
of the engine weight-address register (the ROUTE-data fix from the
kp4mp32_c16 post-route forensics). Anchor-asserted + idempotent; .prewrep
backups.

THE PROBLEM (output/reports_integrated/checkpoints/
first_light_postroute_timing_kp4mp32_c16.rpt): the design's worst routed
paths are `u_address_generator/g_walk_kpar.weight_rd_addr_reg[*]` ->
`u_uram_weight_bank{4,5,6}/mem_bram_*/{CASDOMUXA,ADDRARDADDR}` at 98.9-99.3%
ROUTE delay (slack +0.102 @16ns): ONE 22b address register fans out to 8
banks x ~200 cascaded RAMB36 address/cascade-select pins scattered across
the die. Vivado's own late replication (weight_rd_addr_reg[8]_replica_1)
appears IN the worst paths — post-placement cloning came too late to help.

THE FIX (structural, cycle-IDENTICAL): give each bank its own REGISTER copy
fed by the SAME D / same enable / same reset as the original — 1/8 the
fanout each, placeable next to its bank. Three layers:

* output/rtl/engine/address_generator.v (SHARED): new parameter
  WADDR_REP (default 1) + output `weight_rd_addr_rep`
  (WADDR_REP x 22b). Each walk branch gains a replication generate:
  WADDR_REP==1 elaborates a pure passthrough assign of the original
  register (ZERO new FFs -> MBV2 and every iso harness are bit- and
  FF-identical; MBV2 8/8 inertness gate mandatory and run). WADDR_REP>1
  elaborates (* dont_touch *) replicas whose always-block mirrors the
  original's exact update protocol (reset 0; update only under
  run_active with the same fast/legacy mux; hold otherwise).
* output/rtl/shared_engine_skeleton.v (SHARED): forwards WADDR_REP, adds
  the `weight_rd_addr_rep` output (group-shifted per K_PAR exactly like
  the scalar export: >>3 at K_PAR=8, >>2 at K_PAR=4, passthrough at 1).
* output/rtl/nn2rtl_top.v (ResNet-own): WADDR_REP=8; each
  u_uram_weight_bank{b} takes its address from replica b
  (`engine_weight_rd_addr_rep[b*22 +: 14]` = that replica's group
  address). The scalar weight_rd_addr export keeps driving only the
  3-bit serial-subword pipe inside the skeleton (fanout 3).

The original register's value and every replica's value are EQUAL every
cycle by construction, so e2e must be byte-exact AND cycle-exact.

Gates: lint 0; ResNet e2e vec0+vec1 PASS 0/100352 at UNCHANGED cycles;
MBV2 8/8 PASS at EXACTLY 1,184,731 (shared files touched).

Usage: python scripts/apply_resnet_waddr_rep.py [--check]
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AG = REPO / "output" / "rtl" / "engine" / "address_generator.v"
SKEL = REPO / "output" / "rtl" / "shared_engine_skeleton.v"
TOP = REPO / "output" / "rtl" / "nn2rtl_top.v"

_backed_up: set[Path] = set()


def patch(path: Path, old: str, new: str, tag: str, count: int = 1) -> None:
    """Anchor-asserted replace. Idempotent: presence of `new` == applied."""
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"  [skip] {path.name}: {tag} already applied")
        return
    n = text.count(old)
    if n != count:
        raise SystemExit(f"ANCHOR FAIL {path.name} / {tag}: found {n}, want {count}")
    if path not in _backed_up:
        bak = path.with_name(path.name + ".prewrep")
        if not bak.exists():
            bak.write_text(text, encoding="utf-8", newline="\n")
        _backed_up.add(path)
    path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")
    print(f"  [ok]   {path.name}: {tag}")


def rep_block(d_expr: str) -> str:
    """The per-branch replication generate (same D protocol as the original
    weight_rd_addr register in that branch)."""
    return f"""
    // [WADDR-REP 2026-06-11] per-bank replicas of the weight_rd_addr
    // register: SAME D ({d_expr}), same run_active
    // enable, same reset -> every replica equals the original register
    // every cycle. WADDR_REP==1 (default; MBV2 + iso harnesses) elaborates
    // a passthrough of the original register: ZERO new FFs, bit-identical.
    // (* dont_touch *) keeps Vivado from re-merging the copies; each copy
    // feeds ONE weight bank in the ResNet top (1/8 the fanout, placeable
    // next to its bank's BRAM column).
    if (WADDR_REP == 1) begin : g_wrep1
        assign weight_rd_addr_rep = weight_rd_addr;
    end else begin : g_wrep
        for (wr_i = 0; wr_i < WADDR_REP; wr_i = wr_i + 1) begin : g_r
            (* dont_touch = "true" *) reg [21:0] waddr_rep_q;
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n)          waddr_rep_q <= 22'd0;
                else if (run_active) waddr_rep_q <= {d_expr};
            end
            assign weight_rd_addr_rep[wr_i*22 +: 22] = waddr_rep_q;
        end
    end
"""


D_FAST = "kpar_fast ? weight_addr_next_fast : weight_addr_next"
D_LEG = "weight_addr_next"


def patch_ag() -> None:
    # 1. parameter
    patch(AG, """    parameter integer K_PAR = 1
) (""", """    parameter integer K_PAR = 1,
    // [WADDR-REP 2026-06-11] number of replicated copies of the
    // weight_rd_addr register exported on weight_rd_addr_rep (per-bank
    // fanout relief; see apply_resnet_waddr_rep.py). 1 (DEFAULT) =
    // passthrough of the original register — zero new FFs, bit-identical
    // for every legacy instance (MBV2, iso harnesses).
    parameter integer WADDR_REP = 1
) (""", "WADDR_REP parameter")

    # 2. output port (k_tap_mask is currently the last port)
    patch(AG, """    output wire [((K_PAR > 4) ? K_PAR : 4)-1:0]  k_tap_mask
);""", """    output wire [((K_PAR > 4) ? K_PAR : 4)-1:0]  k_tap_mask,

    // [WADDR-REP 2026-06-11] replicated copies of weight_rd_addr (copy i at
    // [i*22 +: 22]); equal to weight_rd_addr every cycle by construction.
    output wire [WADDR_REP*22-1:0] weight_rd_addr_rep
);""", "weight_rd_addr_rep output port")

    # 3. module-scope genvar (single decl reused by all branches)
    patch(AG, """    // Latched layer-completion flag (pixel_done semantics — see header).
    reg        pixel_done_latch;
""", """    // Latched layer-completion flag (pixel_done semantics — see header).
    reg        pixel_done_latch;

    // [WADDR-REP 2026-06-11] genvar for the per-branch replica loops.
    genvar wr_i;
""", "module-scope genvar")

    # 4. replica blocks at the tail of each walk branch
    patch(AG, """
    end else if (K_PAR == 8) begin : g_walk_kpar8""",
          rep_block(D_LEG) + """
    end else if (K_PAR == 8) begin : g_walk_kpar8""",
          "legacy-branch replicas")

    patch(AG, """
    end else begin : g_walk_kpar""",
          rep_block(D_FAST) + """
    end else begin : g_walk_kpar""",
          "kpar8-branch replicas")

    patch(AG, """
    end endgenerate

endmodule""",
          rep_block(D_FAST) + """
    end endgenerate

endmodule""",
          "kpar4-branch replicas")


def patch_skel() -> None:
    # 1. parameter (after ENG_PIPE, the current last parameter)
    patch(SKEL, """    parameter integer ENG_PIPE = 0
) (""", """    parameter integer ENG_PIPE = 0,

    // ---- weight-address replication (default 1 = bit/FF-identical) ----
    // [WADDR-REP 2026-06-11] forwarded to address_generator; the ResNet top
    // sets 8 so each uram_weight_bank gets its own (* dont_touch *) copy of
    // the weight address register (post-route fanout fix; see
    // apply_resnet_waddr_rep.py). weight_rd_addr_rep carries the copies,
    // group-shifted per K_PAR exactly like the scalar weight_rd_addr.
    parameter integer WADDR_REP = 1
) (""", "WADDR_REP parameter")

    # 2. output port next to the scalar export
    patch(SKEL, """    output wire [URAM_ADDR_W-1:0]        weight_rd_addr,
    output wire                          weight_rd_en,""", """    output wire [URAM_ADDR_W-1:0]        weight_rd_addr,
    // [WADDR-REP 2026-06-11] replicated weight addresses (copy i at
    // [i*URAM_ADDR_W +: URAM_ADDR_W]), each == weight_rd_addr every cycle.
    output wire [WADDR_REP*URAM_ADDR_W-1:0] weight_rd_addr_rep,
    output wire                          weight_rd_en,""", "weight_rd_addr_rep output port")

    # 3. internal wire
    patch(SKEL, """    wire [URAM_ADDR_W-1:0]     ag_weight_rd_addr;
    wire                       ag_weight_rd_en;""", """    wire [URAM_ADDR_W-1:0]     ag_weight_rd_addr;
    wire [WADDR_REP*22-1:0]    ag_weight_rd_addr_rep;   // [WADDR-REP 2026-06-11]
    wire                       ag_weight_rd_en;""", "ag rep wire")

    # 4. group-shifted per-replica export (mirrors the scalar g_waddr_*)
    patch(SKEL, """    generate if (K_PAR == 1) begin : g_waddr_legacy
        assign weight_rd_addr  = ag_weight_rd_addr;
    end else if (K_PAR == 8) begin : g_waddr_kpar8
        assign weight_rd_addr  = {3'b000, ag_weight_rd_addr[21:3]};   // [KPAR8 2026-06-10] GROUP addr = old>>3
    end else begin : g_waddr_kpar
        assign weight_rd_addr  = {2'b00, ag_weight_rd_addr[21:2]};
    end endgenerate""", """    generate if (K_PAR == 1) begin : g_waddr_legacy
        assign weight_rd_addr  = ag_weight_rd_addr;
    end else if (K_PAR == 8) begin : g_waddr_kpar8
        assign weight_rd_addr  = {3'b000, ag_weight_rd_addr[21:3]};   // [KPAR8 2026-06-10] GROUP addr = old>>3
    end else begin : g_waddr_kpar
        assign weight_rd_addr  = {2'b00, ag_weight_rd_addr[21:2]};
    end endgenerate
    // [WADDR-REP 2026-06-11] per-replica export, group-shifted exactly like
    // the scalar weight_rd_addr above (same K_PAR generate split).
    genvar wrp;
    generate if (K_PAR == 1) begin : g_waddr_rep_legacy
        for (wrp = 0; wrp < WADDR_REP; wrp = wrp + 1) begin : g_w
            assign weight_rd_addr_rep[wrp*URAM_ADDR_W +: URAM_ADDR_W]
                 = ag_weight_rd_addr_rep[wrp*22 +: 22];
        end
    end else if (K_PAR == 8) begin : g_waddr_rep_kpar8
        for (wrp = 0; wrp < WADDR_REP; wrp = wrp + 1) begin : g_w
            assign weight_rd_addr_rep[wrp*URAM_ADDR_W +: URAM_ADDR_W]
                 = {3'b000, ag_weight_rd_addr_rep[wrp*22+3 +: 19]};
        end
    end else begin : g_waddr_rep_kpar
        for (wrp = 0; wrp < WADDR_REP; wrp = wrp + 1) begin : g_w
            assign weight_rd_addr_rep[wrp*URAM_ADDR_W +: URAM_ADDR_W]
                 = {2'b00, ag_weight_rd_addr_rep[wrp*22+2 +: 20]};
        end
    end endgenerate""", "per-replica group-shift export")

    # 5. AG instantiation: forward param + connect port
    patch(SKEL, """    address_generator #(.K_PAR(K_PAR)) u_address_generator (""",
          """    address_generator #(.K_PAR(K_PAR), .WADDR_REP(WADDR_REP)) u_address_generator (""",
          "AG inst param")
    patch(SKEL, """        .weight_rd_addr        (ag_weight_rd_addr),
        .weight_rd_en          (ag_weight_rd_en),""", """        .weight_rd_addr        (ag_weight_rd_addr),
        .weight_rd_addr_rep    (ag_weight_rd_addr_rep),   // [WADDR-REP 2026-06-11]
        .weight_rd_en          (ag_weight_rd_en),""", "AG inst rep port")

    # 6. the standalone-parse AG stub (NN2RTL_ENGINE_SUBBLOCKS_PROVIDED
    #    undefined): mirror the new param + port so `iverilog -t null
    #    shared_engine_skeleton.v` still parses.
    patch(SKEL, """module address_generator #(
    parameter integer K_PAR = 1
) (""", """module address_generator #(
    parameter integer K_PAR = 1,
    parameter integer WADDR_REP = 1   // [WADDR-REP 2026-06-11]
) (""", "stub param")
    patch(SKEL, """    output wire [((K_PAR > 4) ? K_PAR : 4)-1:0] k_tap_mask  // [KPAR4] ([KPAR8 2026-06-10] max(K_PAR,4) wide)
);""", """    output wire [((K_PAR > 4) ? K_PAR : 4)-1:0] k_tap_mask,  // [KPAR4] ([KPAR8 2026-06-10] max(K_PAR,4) wide)
    output wire [WADDR_REP*22-1:0] weight_rd_addr_rep   // [WADDR-REP 2026-06-11]
);""", "stub port")
    patch(SKEL, """    assign k_tap_mask         = {{(((K_PAR > 4) ? K_PAR : 4)-1){1'b0}}, 1'b1};   // [KPAR4]/[KPAR8 2026-06-10]
endmodule""", """    assign k_tap_mask         = {{(((K_PAR > 4) ? K_PAR : 4)-1){1'b0}}, 1'b1};   // [KPAR4]/[KPAR8 2026-06-10]
    assign weight_rd_addr_rep = {WADDR_REP{22'd0}};   // [WADDR-REP 2026-06-11]
endmodule""", "stub tie")


def patch_top() -> None:
    # 1. localparam + rep wire (right after the engine weight-data wire)
    patch(TOP, """    wire [ENGINE_WBUS_W-1:0]   engine_weight_rd_data;   // WGT-packed: 256 lanes * ENGINE_WGT_W b
""", """    wire [ENGINE_WBUS_W-1:0]   engine_weight_rd_data;   // WGT-packed: 256 lanes * ENGINE_WGT_W b
    // [WADDR-REP 2026-06-11] one (* dont_touch *) copy of the engine weight
    // address register PER BANK (post-route fanout fix: the single register
    // fanning to 8 banks x ~200 cascaded RAMB36s was the design's worst
    // routed path class, 98.9% route delay). Replica b feeds bank b only.
    localparam integer ENGINE_WADDR_REP = 8;
    wire [ENGINE_WADDR_REP*22-1:0] engine_weight_rd_addr_rep;
""", "rep localparam + wire")

    # 2. retire the shared bank-address wire (each bank now takes its own
    #    replica slice).
    patch(TOP, """    wire [13:0] weight_bank_rd_addr = engine_weight_rd_addr[13:0];  // [KPAR8-RN] GROUP address (engine exports old>>3; 8384 wide lines)
""", """    // [WADDR-REP 2026-06-11] the shared weight_bank_rd_addr wire is GONE:
    // bank b's address is its own replica engine_weight_rd_addr_rep[b*22+:14]
    // ([KPAR8-RN] GROUP address, engine exports old>>3; 8384 wide lines).
""", "retire shared bank addr wire")

    # 3. per-bank replica hookup
    for b in range(8):
        patch(TOP, f"""    ) u_uram_weight_bank{b} (
        .clk(clk),
        .rd_addr(weight_bank_rd_addr),
""", f"""    ) u_uram_weight_bank{b} (
        .clk(clk),
        .rd_addr(engine_weight_rd_addr_rep[{b}*22 +: 14]),   // [WADDR-REP] replica {b}
""", f"bank{b} replica addr")

    # 4. engine instantiation: param + port
    patch(TOP, """        .ENG_PIPE(1)
    ) u_shared_engine (
""", """        .ENG_PIPE(1),
        // [WADDR-REP 2026-06-11] one weight-address register copy per bank.
        .WADDR_REP(ENGINE_WADDR_REP)
    ) u_shared_engine (
""", "engine WADDR_REP param")
    patch(TOP, """        .weight_rd_addr(engine_weight_rd_addr),
        .weight_rd_en(engine_weight_rd_en),""", """        .weight_rd_addr(engine_weight_rd_addr),
        .weight_rd_addr_rep(engine_weight_rd_addr_rep),   // [WADDR-REP 2026-06-11]
        .weight_rd_en(engine_weight_rd_en),""", "engine rep port")


def main() -> int:
    if "--check" in sys.argv:
        for f in (AG, SKEL, TOP):
            t = f.read_text(encoding="utf-8")
            print(f"{f.name}: WADDR-REP markers = {t.count('[WADDR-REP')}")
        return 0
    print("[waddr-rep] patching address_generator.v (replica registers) ...")
    patch_ag()
    print("[waddr-rep] patching shared_engine_skeleton.v (export plumbing) ...")
    patch_skel()
    print("[waddr-rep] patching nn2rtl_top.v (per-bank hookup) ...")
    patch_top()
    print("[waddr-rep] done. Backups: *.prewrep. Re-run is a no-op.")
    print("[waddr-rep] GATES: lint; ResNet e2e vec0+vec1 (cycle-EXACT); "
          "MBV2 8/8 @ 1,184,731 EXACT (shared files touched).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
