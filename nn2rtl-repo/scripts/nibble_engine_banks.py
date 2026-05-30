#!/usr/bin/env python3
"""Phase 3 FIT (engine INT4 nibble-pack): convert the 8 deduped engine weight banks
from 288-bit lines (low 256 = 32 INT8-stored INT4 weights) to 144-bit lines
(low 128 = 32 nibbles). Halves engine weight BRAM (with dedup -> ~1131 BRAM36).
Each weight's low nibble IS its signed INT4 value (matches mac_array $signed(4-bit)).
Run AFTER scripts/dedup_engine_banks.py. Pairs with RTL edits: mac_array weight_bus
2048->1024 + slice *8->*4; shared_engine WGT_W 8->4/URAM_DATA_W 2048->1024; nn2rtl_top
bank rd_data 288->144, concat [255:0]->[127:0], uram_weight_bank width 288->144
(READ_LATENCY_A=2 PRESERVED). Byte-exact gate: run_nn2rtl_top_probe relu_48==0.00%.
"""
from __future__ import annotations
import sys
from pathlib import Path

WDIR = Path(__file__).resolve().parent.parent / "output/weights"

def repack_line(line: str) -> str:
    val = int(line, 16)                       # 288-bit word
    out = 0
    for j in range(32):                       # 32 weights in low 256 bits
        nib = (val >> (j * 8)) & 0xF          # low nibble of byte j = signed INT4 value
        out |= nib << (j * 4)                 # nibble j at bit j*4 (low 128 bits)
    return format(out, "036x")                # 144 bits = 36 hex chars (top 16 bits = 0, unused)

def main() -> int:
    dry = "--dry-run" in sys.argv
    for b in range(8):
        p = WDIR / f"uram_weights_bank{b}.mem"
        lines = p.read_text().splitlines()
        assert all(len(l) == 72 for l in lines), f"bank{b}: expected 72-char lines"
        new = [repack_line(l) for l in lines]
        # offline self-check: nibble round-trip == original low-256 bytes (signed INT4)
        for li in (0, len(lines)//2, len(lines)-1):
            ov = int(lines[li], 16); nv = int(new[li], 16)
            for j in range(32):
                ob = (ov >> (j*8)) & 0xFF; o = ob-256 if ob > 127 else ob
                nn = (nv >> (j*4)) & 0xF; n = nn-16 if nn & 0x8 else nn
                assert n == o, f"bank{b} line{li} wt{j}: {n} != {o}"
        if dry:
            print(f"  [dry] bank{b}: {len(lines)} lines 72->36 chars, round-trip OK")
        else:
            p.write_text("\n".join(new) + "\n", newline="\n")
            print(f"  [ok] bank{b}: {len(new)} lines, 72->36 chars (288->144 bit), nibble round-trip OK")
    print(f"\n{'(dry) ' if dry else ''}nibble-packed 8 engine banks. low 128 bits = 32 INT4 nibbles.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
