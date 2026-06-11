#!/usr/bin/env python3
"""apply_resnet_fanout_hints.py — [FO-HINT 2026-06-11] synthesis-only
(* max_fanout *) hints for the remaining 94-99%-ROUTE path classes in the
kp4mp32_c16 post-route report where STRUCTURAL replication is awkward.
Anchor-asserted + idempotent; .prefoh backups. ResNet-OWN files only
(nn2rtl_scheduler.v + node_conv_*.v) — no shared file, no MBV2 exposure.

PATH CLASSES ADDRESSED (output/reports_integrated/checkpoints/
first_light_postroute_timing_kp4mp32_c16.rpt):

1. u_scheduler/FSM_onehot_state_reg[5] -> u_node_conv_276/out_pix_reg[*]/CE
   (13 of the top 40 paths, slack +0.240, 94.5% route, BUFGCE in path).
   The scheduler's `state` register feeds the spatial_stall Moore output
   whose top-level spatial_run net already carries (* max_fanout = 32 *)
   [FMAX-FANOUT] — the driver LUT is replicated, but every replica's
   INPUT still converges on the single state register. Hinting the state
   register itself lets synthesis clone the FSM bits feeding distant
   replica groups. Sim-inert (attributes are comments to Verilator).

2. u_node_conv_288/in_beat_idx_reg[0] -> in_lo_reg[*]/D (slack +0.213,
   99.0% route). The gather beat counter decodes into per-slice write
   selects of the (multi-thousand-bit, physically spread) in_lo register.
   ~IN_BEATS compare-LUT loads scattered across the register's spread =
   long routes from one small counter. max_fanout=8 gives the placer
   region-local counter copies. Applied UNIFORMLY to all spatial conv
   nodes with the tiled-streaming gather (same generated pattern, same
   class) — synthesis only replicates where actual fanout exceeds the
   bound, so narrow nodes are no-ops.

CLASSES OBSERVED BUT NOT RTL-FIXABLE (documented in
docs/agent_tasks/RESNET_FINAL_BUNDLE_ANALYSIS.md): point-to-point DATA
paths (dp/data_out_reg -> next node in_lo / requant g_lane -> engine FIFO /
loader accumulator paths) — ~0.24-0.30 slack, pure placement distance, no
broadcast source to replicate; Vivado-side levers only.

Usage: python scripts/apply_resnet_fanout_hints.py [--check]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RTL = REPO / "output" / "rtl"
SCHED = RTL / "nn2rtl_scheduler.v"

_backed_up: set[Path] = set()


def patch(path: Path, old: str, new: str, tag: str, count: int = 1,
          enc: str = "utf-8") -> None:
    # enc="latin-1" for the generated node_conv files: 12 of them carry
    # cp1252 em-dashes in header comments; latin-1 round-trips every byte
    # 1:1 so the patch is byte-preserving outside the (ASCII) hunk.
    text = path.read_text(encoding=enc)
    if new in text:
        print(f"  [skip] {path.name}: {tag} already applied")
        return
    n = text.count(old)
    if n != count:
        raise SystemExit(f"ANCHOR FAIL {path.name} / {tag}: found {n}, want {count}")
    if path not in _backed_up:
        bak = path.with_name(path.name + ".prefoh")
        if not bak.exists():
            bak.write_text(text, encoding=enc, newline="\n")
        _backed_up.add(path)
    path.write_text(text.replace(old, new), encoding=enc, newline="\n")
    print(f"  [ok]   {path.name}: {tag}")


def main() -> int:
    convs = sorted(RTL.glob("node_conv_*.v"))
    targets = [p for p in convs
               if "reg [INB_W-1:0] in_beat_idx;" in p.read_text(encoding="latin-1")
               and not p.name.endswith(".preimprove")]
    if "--check" in sys.argv:
        t = SCHED.read_text(encoding="utf-8")
        print(f"{SCHED.name}: FO-HINT markers = {t.count('[FO-HINT')}")
        done = sum("[FO-HINT" in p.read_text(encoding="latin-1") for p in targets)
        print(f"node_conv in_beat_idx hints: {done}/{len(targets)}")
        return 0

    # 1. scheduler FSM state register (split decl so the hint does not land
    #    on the combinational next_state).
    print("[fo-hint] patching nn2rtl_scheduler.v (FSM state register) ...")
    patch(SCHED, """    reg [3:0]           state, next_state;
""", """    // [FO-HINT 2026-06-11] state feeds the spatial_stall Moore output whose
    // top-level spatial_run replicas (already max_fanout=32) all converge on
    // these 4 FFs — the kp4mp32_c16 post-route report's #2 path class
    // (FSM_onehot_state_reg -> conv_276 out_pix CE, 13/40 worst paths).
    // Hint synthesis to clone the state bits per consumer region. Synth-only
    // attribute: Verilator/iverilog ignore it -> byte- and cycle-exact.
    (* max_fanout = 16 *) reg [3:0] state;
    reg [3:0]           next_state;
""", "state reg max_fanout")

    # 2. in_beat_idx gather counters (uniform class fix).
    print(f"[fo-hint] patching {len(targets)} node_conv files (in_beat_idx) ...")
    for p in targets:
        patch(p, """    reg [INB_W-1:0] in_beat_idx;
""", """    (* max_fanout = 8 *) reg [INB_W-1:0] in_beat_idx;   // [FO-HINT 2026-06-11] kp4mp32_c16 #3 path class (beat-select decode into the spread in_lo register; 99% route on conv_288). Synth-only.
""", "in_beat_idx max_fanout", enc="latin-1")
    print("[fo-hint] done. Backups: *.prefoh. Re-run is a no-op.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
