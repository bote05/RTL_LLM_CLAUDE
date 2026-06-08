#!/usr/bin/env python3
"""[CLEANUP 2026-06-08] Remove inferred LATCHes in the n4_* relu modules.

In the shared combinational requant `always @(*)`, the per-channel loop temporaries (in_byte +
relu_byte OR relu_idx, depending on the module variant) are assigned ONLY inside `if (valid_in)`
-> inferred latch. requant_comb is unconditionally zeroed and is the only downstream output, so
the latch is functionally harmless, but it is a real lint/synthesis cleanliness issue. Fix = give
each declared temporary an UNCONDITIONAL width-agnostic ('0) default before the guard. BYTE-EXACT
(temps are used only inside the guarded for-loop; requant_comb unchanged).

Robust to the two n4 variants (relu_byte / relu_idx). Idempotent, anchor-validated, backs up first.
Usage: python scripts/apply_mbv2_n4_latch_fix.py [--dry-run]
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "output" / "mobilenet-v2" / "rtl"
ANCHOR = re.compile(r"\n        requant_comb = (\d+)'d0;\n        if \(valid_in\) begin\n")
CANDIDATE_TEMPS = ("in_byte", "relu_byte", "relu_idx")


def main() -> int:
    dry = "--dry-run" in sys.argv
    bk = ROOT / "backups" / "mbv2_n4_latch_fix"
    if not dry:
        bk.mkdir(parents=True, exist_ok=True)
    done, skip = 0, 0
    for f in sorted(RTL.glob("n4*.v")):
        t = f.read_text()
        if "always @(*)" not in t or "requant_comb" not in t:
            continue
        if "[LATCH-FIX" in t:
            print(f"  {f.name}: already fixed -> SKIP"); skip += 1; continue
        m = ANCHOR.search(t)
        if not m or len(ANCHOR.findall(t)) != 1:
            print(f"  {f.name}: anchor count={len(ANCHOR.findall(t))} (expect 1) -> SKIP"); skip += 1; continue
        # only default temporaries that are actually DECLARED in this module
        temps = [n for n in CANDIDATE_TEMPS
                 if re.search(rf"\breg (signed )?\[\d+:\d+\] +{n};", t)]
        if not temps:
            print(f"  {f.name}: no known temps declared -> SKIP"); skip += 1; continue
        defaults = "".join(
            f"        {n:<9} = '0;   // [LATCH-FIX 2026-06-08] unconditional default (no inferred latch)\n"
            for n in temps)
        new_block = f"\n        requant_comb = {m.group(1)}'d0;\n{defaults}        if (valid_in) begin\n"
        new = t.replace(m.group(0), new_block)
        if dry:
            print(f"  {f.name}: OK (temps: {', '.join(temps)})")
        else:
            (bk / f.name).write_text(t, newline="\n")
            f.write_text(new, newline="\n")
            print(f"  {f.name}: APPLIED (temps: {', '.join(temps)})")
        done += 1
    print(f"[n4-latch-fix] {'validated' if dry else 'applied'}={done} skipped={skip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
