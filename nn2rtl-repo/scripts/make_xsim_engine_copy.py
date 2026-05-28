#!/usr/bin/env python3
"""Produce an XSim-parser-compatible COPY of shared_engine_skeleton.v.

Vivado's VRFC parser (unlike iverilog/Verilator) rejects use-before-declaration
of continuously-assigned wires (VRFC 10-3380). `oc_pass_total_m1` is declared
~line 279 but used in the FSM combinational block ~line 234. We hoist the two
derived-wire declarations (oc_pass_total, oc_pass_total_m1) to immediately after
the `wire [11:0] cfg_oc;` declaration so they precede every use.

This does NOT touch the canonical RTL — it writes a transient build copy used
only for the XSim compile (same copy-and-modify pattern as run_engine_only_synth.ts).
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "output/rtl/shared_engine_skeleton.v"
DST = ROOT / "build_engine_xsim/shared_engine_skeleton_xsim.v"


def main() -> None:
    lines = SRC.read_text().splitlines(keepends=True)

    # Find the contiguous declaration block: the `oc_pass_total` decl through the
    # end (first ';') of the `oc_pass_total_m1` decl (which spans 2 lines).
    start = m1 = None
    for i, l in enumerate(lines):
        if start is None and re.match(r"\s*wire\s*\[3:0\]\s*oc_pass_total\s+=", l):
            start = i
        if re.match(r"\s*wire\s*\[3:0\]\s*oc_pass_total_m1\s*=", l):
            m1 = i
            break
    if start is None or m1 is None:
        sys.exit("FATAL: could not locate oc_pass_total / oc_pass_total_m1 declarations")
    end = next((j for j in range(m1, len(lines)) if lines[j].rstrip().endswith(";")), None)
    if end is None:
        sys.exit("FATAL: could not find end (';') of oc_pass_total_m1 declaration")
    block = lines[start:end + 1]

    # Remove from original position.
    rest = lines[:start] + lines[end + 1:]

    # Insert right after `wire [11:0] cfg_oc;`.
    anchor = None
    for i, l in enumerate(rest):
        if re.match(r"\s*wire\s*\[11:0\]\s*cfg_oc\s*;", l):
            anchor = i
            break
    if anchor is None:
        sys.exit("FATAL: could not locate `wire [11:0] cfg_oc;` anchor")

    out = (rest[:anchor + 1]
           + ["    // [xsim-hoist] moved above first use (FSM ST_REQUANT) for VRFC\n"]
           + block
           + rest[anchor + 1:])

    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text("".join(out))
    print(f"[xsim-copy] wrote {DST.relative_to(ROOT)}")
    print(f"[xsim-copy] hoisted lines {start+1}-{end+1} to after line {anchor+1} (cfg_oc)")


if __name__ == "__main__":
    main()
