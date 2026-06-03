#!/usr/bin/env python3
"""FIT FIX 1 (PREP — does NOT build): map the skip_fifo memory by DEPTH so the
deep FIFOs land in URAM instead of distributed LUT-RAM.

WHY (measured, first full Vivado synth 2026-05-30):
- CLB LUT = 1,983,938 / 1,728,000 = 115% OVER. Of that, LUT-as-Distributed-RAM
  = 434,324, and ~386K (91%) is ONE undirected array: skip_fifo.mem
  (nn2rtl_top.v, 107 instances, all WIDTH=256, no ram_style). The deep instances
  dominate: 5xDEPTH8192 + 5x4096 + 7x2048 + 12x1024 + 7x512 = 36 deep FIFOs.
- BRAM = 4663 / 2688 = 174% OVER -> BRAM is the BINDING constraint.
- URAM288 = 203 / 1280 = 16% (84% EMPTY).

THE CORRECT TARGET IS URAM, NOT BRAM. The naive "ram_style=block" would push 386K
LUT-RAM into BRAM and make the binding (BRAM) constraint WORSE. Instead, route the
DEEP FIFOs (DEPTH>=512) to URAM (ram_style="ultra"):
  * removes ~370K of the 434K LUT-RAM -> CLB LUT total ~1.98M -> ~1.6M = ~92% = FITS
  * costs ZERO BRAM (the binding constraint is untouched)
  * adds ~164 URAM (256b x DEPTH: 4 URAM/instance for DEPTH<=4096, 8 for 8192)
    -> 203 + ~164 = ~367 / 1280 = ~29% (huge headroom)
Shallow FIFOs (DEPTH<512, mostly the 65 DEPTH=2 skids) STAY distributed LUT-RAM:
they are tiny (~16K LUT total, well within the 791K LUT-RAM budget) and putting
them in URAM/BRAM would waste whole dense blocks on 2-entry skids.

BYTE-EXACT: ram_style is a SYNTHESIS-ONLY mapping hint. Verilator (the byte-exact
gate) and iverilog ignore it entirely -> simulation behavior is bit-identical.
The FIFO read/write/pointer logic is unchanged. skip_fifo has NO $readmemh/initial
(verified) so it is runtime/zero-init data -> URAM (which cannot be bitstream-
initialized on this device) is SAFE for it (unlike weights/scale_rom, which have
init and must NOT go to URAM).

NOTE — this fixes the LUT overage ONLY. The BRAM 174% is a SEPARATE problem
(weights ~3200 tiles + line buffers ~1177) handled by other fit fixes
(line-buffer re-aspect + engine-weight-bank -> URAM + runtime loader).

This patch is IDEMPOTENT and REVERSIBLE (--revert). It does NOT rebuild.

USAGE:
  python scripts/apply_fit_skipfifo_uram.py --dry-run
  python scripts/apply_fit_skipfifo_uram.py
  python scripts/apply_fit_skipfifo_uram.py --revert
"""
from __future__ import annotations
import sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOP = ROOT / "output/rtl/nn2rtl_top.v"
BK = ROOT / "backups/fit_skipfifo_uram_20260530"
BK.mkdir(parents=True, exist_ok=True)
BKFILE = BK / "nn2rtl_top.v"

DRY = "--dry-run" in sys.argv
REVERT = "--revert" in sys.argv

# Anchor: the skip_fifo memory declaration (with its preceding ADDR_W localparam,
# to disambiguate from other `mem` arrays in the file). This is the ONLY place
# skip_fifo's mem is declared (one module, 107 instances).
OLD = """    localparam integer ADDR_W = clog2(DEPTH);

    reg [WIDTH-1:0] mem [0:DEPTH-1];"""

NEW = """    localparam integer ADDR_W = clog2(DEPTH);

    // [FIT-FIX 2026-05-30 apply_fit_skipfifo_uram.py] Map by DEPTH: deep FIFOs
    // (>=512) -> URAM (BRAM is the binding constraint at 174%; URAM is 84% empty
    // and skip_fifo is runtime/no-init data so URAM zero-init is safe). Shallow
    // FIFOs stay distributed LUT-RAM (tiny; would waste a whole dense block).
    // Byte-exact: ram_style is a synth-only hint; Verilator/iverilog ignore it.
    localparam RAM_STYLE = (DEPTH >= 512) ? "ultra" : "distributed";
    (* ram_style = RAM_STYLE *) reg [WIDTH-1:0] mem [0:DEPTH-1];"""


def main():
    if REVERT:
        if not BKFILE.exists():
            print(f"ERROR: no backup at {BKFILE} — nothing to revert"); sys.exit(1)
        shutil.copy(BKFILE, TOP)
        print(f"[revert] restored {TOP} from {BKFILE}"); return

    txt = TOP.read_text()

    if "[FIT-FIX 2026-05-30 apply_fit_skipfifo_uram.py]" in txt:
        print("[skip] already patched. Use --revert to undo."); return

    c = txt.count(OLD)
    if c != 1:
        print(f"ERROR: skip_fifo mem anchor found {c} times (expected 1). RTL drifted — aborting, no change."); sys.exit(1)

    new = txt.replace(OLD, NEW)

    # report the DEPTH split this will produce
    import re
    depths = [int(m) for m in re.findall(r"skip_fifo #\(\.WIDTH\(\d+\), \.DEPTH\((\d+)\)\)", txt)]
    deep = sorted([d for d in depths if d >= 512])
    shallow = sorted([d for d in depths if d < 512])
    # URAM estimate: 256b -> 4 URAM wide; depth>4096 -> x2 stack
    uram = sum(4 * (2 if d > 4096 else 1) for d in deep)
    print(f"  skip_fifo instances: {len(depths)} total")
    print(f"  -> URAM (DEPTH>=512): {len(deep)} instances, depths {sorted(set(deep))}, ~{uram} URAM288 added (203 -> {203+uram} / 1280)")
    print(f"  -> distributed LUT-RAM (DEPTH<512): {len(shallow)} instances (mostly DEPTH=2 skids)")

    if DRY:
        print("\n=== DRY RUN — anchor matched once. NOT applied, NOT built. ===")
        return

    if not BKFILE.exists():
        shutil.copy(TOP, BKFILE); print(f"[backup] {TOP} -> {BKFILE}")
    TOP.write_text(new, newline="\n")
    print(f"\n[ok] patched {TOP}. NOT built.")
    print("NEXT: rebuild + e2e byte-exact gate (run_nn2rtl_top_value.ts 0 AND 1 = 0 mismatch) to confirm")
    print("      Verilator behavior unchanged; then re-synth to MEASURE the LUT drop (~1.98M -> ~1.6M).")


if __name__ == "__main__":
    main()
