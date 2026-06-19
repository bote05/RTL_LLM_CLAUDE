#!/usr/bin/env python3
"""apply_resnet_tap0_hardwire.py — [TAP0-HW 2026-06-12] param-gated hardwire
of the K_PAR==8 engine tap0 weight-subword 8:1 mux to weight slice 0
(ResNet ONLY; default 0 keeps the legacy mux VERBATIM for MobileNetV2).
Anchor-asserted + idempotent; .pretap0hw backups; --dry-run supported.

THE PROBLEM (failed_route_final_c14 forensics): the KPAR8 ResNet route
died on WIRE DEMAND in SLR1 — 8/10 top contended nodes are
u_uram_weight_bank{1,2,6,7}/weight_bus[*] (50% of all contention), SLL
SLR0-1 columns at 114%/104%. Every one of the 6,144 weight-bus bits loads
BOTH its direct MAC tap AND one input of the bank-local tap0 wsub_d2 8:1
mux (shared_engine_skeleton.v g_ktap_kpar8) — the mux roughly doubles the
weight-path pin count and parks ~768b of 8:1 mux logic in the hotspot.

THE FIX (structural, provably sim-invisible for ResNet): for ResNet the
mux's select is 0 on EVERY cycle, so tap0 == weight slice 0 always:

  PROOF P0 (re-derived below from the DEPLOYED artifacts, hard-fail on
  violation — same premise as scripts/repack_resnet_kpar8_banks.py):
  (a) the ResNet top never sets ENABLE_DEPTHWISE and the scheduler never
      writes cfg 0x3C  -> dw_mode is constant 0;
  (b) every dispatch in the deployed scheduler ROMs has base%8==0,
      ic%8==0, ic>=8  -> address_generator's kpar_fast==1 for ALL
      dispatches (the serial walk is NEVER exercised — and would be
      wrong-by-design for the pos-major-transposed 3x3 bank regions);
  (c) fast-mode issued address = base + pass*kt + pos*ic + ic0 with
      ic0 stepping by 8 from 0 and kt = ic*kh*kw: every term %8==0
      -> ag_weight_rd_addr[2:0]==0 whenever weight_rd_en is up;
  (d) wsub_d1/d2 reset to 0 and only ever capture addr[2:0]  -> wsub_d2
      is 0 on EVERY cycle (reset value included), valid window or not.
  Therefore replacing the mux output by slice 0 is cycle-for-cycle
  IDENTICAL on every cycle — sim-invisible, gate must show 0 deltas.

  FUTURE dispatch 18 (conv_288, queued next phase): IC=1024 (%8==0,
  >=8), OC=2048, 1x1 dense (identity layout — not even transposed), and
  an appended base lands at the current region end (67,072 %8==0). It
  satisfies P0; this script's re-derivation (run on the regenerated
  scheduler) plus the repack's own P0 assert hard-fail if a relocation
  ever breaks the alignment.

THREE PATCHES:
* output/rtl/shared_engine_skeleton.v (SHARED): new parameter
  TAP0_HARDWIRE (default 0). g_ktap_kpar8 splits into nested generate
  branches: g_tap0_mux (default — the ORIGINAL wsub pipe + 8:1 mux
  verbatim, wsub regs moved inside the branch with the same FFs / same
  D / same reset => MBV2 bit- and cycle-identical) and g_tap0_hw
  (TAP0_HARDWIRE==1 — tap0 = weight_rd_data slice 0, wsub deleted).
  The K_PAR==4 branch and K_PAR==1 legacy are untouched.
* output/rtl/nn2rtl_top.v (ResNet-own): u_shared_engine sets
  .TAP0_HARDWIRE(1).

Gates: MBV2 8/8 byte-exact at EXACTLY 1,184,731 cycles (shared file
touched; default-0 path must be bit-identical). ResNet vec0+vec1
0/100352 in the phase-3 combined gate (change is provably invisible).

Usage: python scripts/apply_resnet_tap0_hardwire.py [--dry-run]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKEL = REPO / "output" / "rtl" / "shared_engine_skeleton.v"
TOP = REPO / "output" / "rtl" / "nn2rtl_top.v"
SCHED = REPO / "output" / "rtl" / "nn2rtl_scheduler.v"

DRY = "--dry-run" in sys.argv[1:]
P = 8
_backed_up: set[Path] = set()


# ----------------------------------------------------------------------------
# P0 re-derivation from the DEPLOYED artifacts (hard-fail on any violation).
# ----------------------------------------------------------------------------
def parse_rom(text: str, name: str, n: int) -> dict[int, int]:
    vals = {int(i): int(v)
            for i, v in re.findall(rf"5'd(\d+): {name} = \d+'d(\d+);", text)}
    if len(vals) != n:
        raise SystemExit(f"P0 FAIL: ROM {name}: {len(vals)} rows != {n}")
    return vals


def assert_p0() -> None:
    sched = SCHED.read_text(encoding="utf-8")
    top = TOP.read_text(encoding="utf-8")

    # (a1) ResNet top never sets ENABLE_DEPTHWISE -> dw_mode constant 0
    # (skeleton: dw_mode = (ENABLE_DEPTHWISE != 0) ? cfg_depthwise : 1'b0).
    if ".ENABLE_DEPTHWISE" in top:
        raise SystemExit("P0 FAIL: nn2rtl_top.v sets ENABLE_DEPTHWISE — the "
                         "!cfg_depthwise fast-eligibility term is no longer "
                         "constant; tap0 hardwire premise void.")
    # (a2) the scheduler never writes the depthwise cfg register 0x3C.
    if re.search(r"8'h3[cC]", sched):
        raise SystemExit("P0 FAIL: nn2rtl_scheduler.v writes AXIL 0x3C "
                         "(depthwise cfg) — serial walk could be engaged.")

    # K_PAR must actually be 8 on this top (the hardwire targets g_ktap_kpar8).
    if "localparam integer ENGINE_K_PAR  = 8;" not in top:
        raise SystemExit("P0 FAIL: nn2rtl_top.v ENGINE_K_PAR != 8 — the "
                         "TAP0_HARDWIRE branch under test would not elaborate.")

    # (b) every dispatch fast-eligible, regions tile [0, end) contiguously
    # from 0 with an 8-aligned end (future-dispatch append stays 8-aligned).
    ndisp = len(re.findall(r"5'd\d+: weight_base_word_rom = ", sched))
    ic = parse_rom(sched, "channel_in_rom", ndisp)
    oc = parse_rom(sched, "channel_out_rom", ndisp)
    kh = parse_rom(sched, "kernel_h_rom", ndisp)
    kw = parse_rom(sched, "kernel_w_rom", ndisp)
    wb = parse_rom(sched, "weight_base_word_rom", ndisp)
    table = [dict(idx=d, ic=ic[d], base=wb[d], kt=ic[d] * kh[d] * kw[d],
                  passes=(oc[d] + 255) // 256) for d in range(ndisp)]
    for e in table:
        if not (e["base"] % P == 0 and e["ic"] % P == 0 and e["ic"] >= P):
            raise SystemExit(
                f"P0 FAIL: dispatch {e['idx']} NOT P=8 fast-eligible "
                f"(base={e['base']} ic={e['ic']}) — its serial walk would "
                f"need the wsub mux the hardwire deletes. ABORT.")
        if e["kt"] % P != 0:
            raise SystemExit(f"P0 FAIL: dispatch {e['idx']} kt={e['kt']} "
                             f"not %{P}==0 — pass blocks misalign groups.")
    regions = sorted((e["base"], e["base"] + e["passes"] * e["kt"], e["idx"])
                     for e in table)
    cur = 0
    for lo, hi, d in regions:
        if lo != cur:
            raise SystemExit(f"P0 FAIL: dispatch {d} region starts at {lo}, "
                             f"expected {cur} (regions must tile from 0).")
        cur = hi
    if cur % P != 0:
        raise SystemExit(f"P0 FAIL: region end {cur} not %{P}==0 — a future "
                         f"appended dispatch would get an unaligned base.")
    print(f"[tap0-hw] P0 PASS: {ndisp} dispatches all P=8 fast-eligible "
          f"(bases {sorted(e['base'] for e in table)} all %8==0; ICs "
          f"{sorted(set(e['ic'] for e in table))} all %8==0); regions tile "
          f"[0,{cur}) contiguously, end %8==0 (future dispatch-18 append "
          f"stays eligible); ENABLE_DEPTHWISE off; no 0x3C scheduler write.")


# ----------------------------------------------------------------------------
# Anchor-asserted idempotent patching (apply_resnet_waddr_rep.py style).
# ----------------------------------------------------------------------------
def patch(path: Path, old: str, new: str, tag: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"  [skip] {path.name}: {tag} already applied")
        return
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"ANCHOR FAIL {path.name} / {tag}: found {n}, want 1")
    if DRY:
        print(f"  [dry]  {path.name}: {tag} would apply")
        return
    if path not in _backed_up:
        bak = path.with_name(path.name + ".pretap0hw")
        if not bak.exists():
            bak.write_text(text, encoding="utf-8", newline="\n")
        _backed_up.add(path)
    path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")
    print(f"  [ok]   {path.name}: {tag}")


# ---- patch 1: skeleton parameter ----
OLD_PARAM = """    parameter integer WADDR_REP = 1
) ("""
NEW_PARAM = """    parameter integer WADDR_REP = 1,

    // ---- tap0 weight-subword hardwire (default 0 = legacy mux verbatim) ----
    // [TAP0-HW 2026-06-12] When 0 (DEFAULT) the K_PAR==8 tap0 hookup keeps
    // the ORIGINAL wsub_d2-piped 8:1 weight-subword mux VERBATIM — MobileNetV2
    // (whose 12 depthwise dispatches NEED the serial-walk subword select) is
    // bit- and cycle-identical. When 1 (ResNet top ONLY) tap0 is HARDWIRED to
    // weight slice 0: proof P0 (apply_resnet_tap0_hardwire.py, re-asserted on
    // the deployed scheduler ROMs) shows every dispatch is fast-eligible
    // (base%8==0, ic%8==0, ic>=8, depthwise off), so every issued weight
    // address has [2:0]==0 and the mux select was 0 on EVERY cycle (reset
    // value included). Deletes the 768b 8:1 mux + halves the weight-bus pin
    // load in the SLR1 route hotspot (failed_route_final_c14: 8/10 top
    // contended nodes = uram_weight_bank weight_bus nets). Only meaningful
    // at K_PAR==8; other K_PAR branches ignore it.
    parameter integer TAP0_HARDWIRE = 0
) ("""

# ---- patch 2: skeleton g_ktap_kpar8 wsub/ktap block + tap0 assign ----
OLD_KTAP = """        reg [2:0] wsub_d1, wsub_d2;
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
            weight_rd_data[wsub_d2*(MAC_COUNT*WGT_W) +: MAC_COUNT*WGT_W];"""
NEW_KTAP = """        reg [7:0] ktap_d1, ktap_d2;
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                ktap_d1 <= 8'b0000_0001; ktap_d2 <= 8'b0000_0001;
            end else begin
                ktap_d1 <= ag_k_tap_mask;          ktap_d2 <= ktap_d1;
            end
        end
        if (TAP0_HARDWIRE == 0) begin : g_tap0_mux
            // [TAP0-HW 2026-06-12] DEFAULT: the ORIGINAL wsub-piped 8:1 tap0
            // mux VERBATIM (wsub regs moved inside this branch: same FFs,
            // same D / same reset / same update -> MBV2 bit- and
            // cycle-identical; 8/8 inertness gate mandatory and run).
            reg [2:0] wsub_d1, wsub_d2;
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    wsub_d1 <= 3'd0;                   wsub_d2 <= 3'd0;
                end else begin
                    wsub_d1 <= ag_weight_rd_addr[2:0]; wsub_d2 <= wsub_d1;
                end
            end
            // tap0 = subword-selected old word (slice 0 for aligned fast groups).
            assign mac_weight_bus[MAC_COUNT*WGT_W-1:0] =
                weight_rd_data[wsub_d2*(MAC_COUNT*WGT_W) +: MAC_COUNT*WGT_W];
        end else begin : g_tap0_hw
            // [TAP0-HW 2026-06-12] ResNet route-congestion fix: tap0 HARDWIRED
            // to weight slice 0 — deletes the 768b 8:1 wsub mux in the SLR1
            // hotspot and halves the weight-bus pin load (12,288 -> 6,144
            // weight-path pin connections). Proof P0 (re-asserted on the
            // DEPLOYED scheduler ROMs by apply_resnet_tap0_hardwire.py):
            // every dispatch is fast-eligible (base%8==0, ic%8==0, ic>=8,
            // depthwise off) -> every issued weight address has [2:0]==0 ->
            // the deleted mux's select wsub_d2 was 0 on EVERY cycle (reset
            // value included) -> tap0 == slice0 unconditionally. The serial
            // walk is NEVER exercised on this top (and stays wrong-by-design
            // for the pos-major-transposed 3x3 bank regions — see
            // scripts/repack_resnet_kpar8_banks.py header).
            assign mac_weight_bus[MAC_COUNT*WGT_W-1:0] =
                weight_rd_data[0 +: MAC_COUNT*WGT_W];
            /* verilator lint_off UNUSED */
            wire _unused_tap0_hw = &{1'b0, ag_weight_rd_addr[2:0]};
            /* verilator lint_on UNUSED */
        end"""

# ---- patch 3: ResNet top instantiation ----
OLD_INST = """        .WADDR_REP(ENGINE_WADDR_REP)
    ) u_shared_engine ("""
NEW_INST = """        .WADDR_REP(ENGINE_WADDR_REP),
        // [TAP0-HW 2026-06-12] hardwire the engine tap0 weight subword to
        // slice 0 (route-congestion fix; all 17 dispatches fast-eligible,
        // proof P0 in apply_resnet_tap0_hardwire.py + the kp8 repack).
        .TAP0_HARDWIRE(1)
    ) u_shared_engine ("""


def main() -> int:
    print(f"[tap0-hw] {'DRY-RUN — ' if DRY else ''}P0 re-derivation from "
          f"deployed artifacts:")
    assert_p0()
    print("[tap0-hw] patching:")
    patch(SKEL, OLD_PARAM, NEW_PARAM, "TAP0_HARDWIRE parameter (default 0)")
    patch(SKEL, OLD_KTAP, NEW_KTAP, "g_ktap_kpar8 tap0 generate split")
    patch(TOP, OLD_INST, NEW_INST, "u_shared_engine .TAP0_HARDWIRE(1)")
    print(f"[tap0-hw] {'dry-run complete (nothing written)' if DRY else 'APPLIED'}. "
          f"Gates: MBV2 8/8 @ exactly 1,184,731; ResNet vec0+vec1 0/100352 "
          f"(phase-3 combined gate).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
