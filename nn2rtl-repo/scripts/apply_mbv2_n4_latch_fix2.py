#!/usr/bin/env python3
"""[CLEANUP 2026-06-08] Complete the n4 LATCH cleanup that #5b (apply_mbv2_n4_latch_fix.py) left
PARTIAL: its temp-detection regex required `reg [` (one space) and so missed relu_idx/relu_byte,
which are declared `reg        [6:0] relu_idx;` (many spaces). 8 latches survived (relu_idx in
n4_4/9/11/13/14, relu_byte in n4_10/20/21) and are the only lint flags + the prime --threads-4
MT-scheduling-ambiguity suspect. This adds the missing relu-temp '0 default right before the
`if (valid_in)` guard in the requant_comb always block. BYTE-EXACT (the temp is assigned-before-use
inside the guarded loop; requant_comb unchanged). Idempotent, function-replacement.

Usage: python scripts/apply_mbv2_n4_latch_fix2.py [--dry-run]
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "output" / "mobilenet-v2" / "rtl"
# CORRECTED detection: variable whitespace between reg / [W:0] / name.
DECL = re.compile(r"\breg\s+(?:signed\s+)?\[\d+:\d+\]\s+(relu_idx|relu_byte)\s*;")
# requant_comb zero + optional already-inserted in_byte default, then the combinational guard.
BLOCK = re.compile(r"(\n        requant_comb = \d+'d0;\n(?:        in_byte   = '0;[^\n]*\n)?)(        if \(valid_in\) begin\n)")


def main() -> int:
    dry = "--dry-run" in sys.argv
    bk = ROOT / "backups" / "mbv2_n4_latch_fix2"
    if not dry:
        bk.mkdir(parents=True, exist_ok=True)
    done, skip = 0, 0
    for f in sorted(RTL.glob("n4*.v")):
        t = f.read_text()
        if "always @(*)" not in t or "requant_comb" not in t:
            continue
        m = DECL.search(t)
        if not m:
            continue  # no relu_idx/relu_byte temp -> not one of the 8
        temp = m.group(1)
        if re.search(rf"\b{temp}\s*=\s*'0;", t):
            print(f"  {f.name}: {temp} already defaulted -> SKIP"); skip += 1; continue
        bm = BLOCK.search(t)
        if not bm or len(BLOCK.findall(t)) != 1:
            print(f"  {f.name}: block anchor count={len(BLOCK.findall(t))} (expect 1) -> SKIP"); skip += 1; continue

        def _repl(mm: "re.Match", _temp=temp) -> str:
            return (mm.group(1)
                    + f"        {_temp:<9} = '0;   // [LATCH-FIX2 2026-06-08] unconditional default (no inferred latch)\n"
                    + mm.group(2))
        new = BLOCK.sub(_repl, t)
        if dry:
            print(f"  {f.name}: OK ({temp})")
        else:
            (bk / f.name).write_text(t, newline="\n")
            f.write_text(new, newline="\n")
            print(f"  {f.name}: APPLIED ({temp})")
        done += 1
    print(f"[n4-latch-fix2] {'validated' if dry else 'applied'}={done} skipped={skip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
