#!/usr/bin/env python3
"""Phase-2 FIT (engine INT3 tri-bit pack): convert the 8 engine weight banks
from 288-bit lines (low 256 = 32 INT8-stored weights, here INT3-range [-4,3])
to 96-bit lines (32 weights x 3 bits). This is the INT3 analogue of
nibble_engine_banks.py (which packs to 144-bit/INT4) and REPLACES it in the
INT3 regen flow.

Pipeline position (INT3/Config B engine regen):
    generate_golden (NN2RTL_INT3_LAYERS set -> engine convs quantized INT3)
    -> build_weight_memory_map.py   (writes 288-bit/72-char banks, INT3-range bytes)
    -> THIS script                  (288-bit -> 96-bit/24-char, 32x3 packed)
    -> dedup_engine_banks.py        (row dedup)

The RTL reads weight_bus[lane*WGT_W +: WGT_W] with WGT_W=3 (mac_array, via
shared_engine, via nn2rtl_top ENGINE_WGT_W=3 -> uram_weight_bank WORD_W=96), so
each weight occupies bits [j*3 +: 3] of the 96-bit word, and weight j's low 3
bits ARE its signed INT3 value (matches $signed(3-bit slice)). Run AFTER
build_weight_memory_map.py and BEFORE dedup. 96-bit words (vs 144) shrink the
engine weight BRAM ~1/3 — the engine half of Config B's fit.

Run: python scripts/nibble_engine_banks_int3.py [--dry-run]
"""
from __future__ import annotations
import sys
from pathlib import Path

WDIR = Path(__file__).resolve().parent.parent / "output/weights"


def repack_line(line: str) -> str:
    """288-bit INT8 word (low 256 = 32 bytes) -> 96-bit word (32 x 3 bits).
    Each byte holds an INT3-range value [-4,3]; its low 3 bits are the signed
    INT3 field the RTL slice $signed(word[j*3+:3]) reads back."""
    val = int(line, 16)
    out = 0
    for j in range(32):
        tri = (val >> (j * 8)) & 0x7        # low 3 bits of byte j
        out |= tri << (j * 3)               # weight j at bit j*3 (96 bits total)
    return format(out, "024x")              # 96 bits = 24 hex chars


def main() -> int:
    dry = "--dry-run" in sys.argv
    for b in range(8):
        p = WDIR / f"uram_weights_bank{b}.mem"
        lines = p.read_text().splitlines()
        # build_weight_memory_map emits 288-bit (72-char) lines; verify we are
        # NOT accidentally re-running on already-packed banks.
        assert all(len(l) == 72 for l in lines), (
            f"bank{b}: expected 72-char (288-bit) lines from build_weight_memory_map; "
            f"got {len(lines[0]) if lines else 0} — did this already run / was it nibble-packed?")
        new = [repack_line(l) for l in lines]
        # self-check: tri-bit round-trip == signed-INT8 byte (must be in [-4,3]).
        for li in (0, len(lines) // 2, len(lines) - 1):
            ov = int(lines[li], 16); nv = int(new[li], 16)
            for j in range(32):
                ob = (ov >> (j * 8)) & 0xFF
                o = ob - 256 if ob > 127 else ob
                assert -4 <= o <= 3, f"bank{b} line{li} wt{j}: byte {o} OUT OF INT3 RANGE [-4,3]"
                nn = (nv >> (j * 3)) & 0x7
                n = nn - 8 if (nn & 0x4) else nn   # signed 3-bit
                assert n == o, f"bank{b} line{li} wt{j}: {n} != {o}"
        if dry:
            print(f"  [dry] bank{b}: {len(lines)} lines 72->24 chars (288->96 bit), INT3 round-trip OK")
        else:
            p.write_text("\n".join(new) + "\n", newline="\n")
            print(f"  [ok] bank{b}: {len(new)} lines, 72->24 chars (288->96 bit), INT3 round-trip OK")
    print(f"\n{'(dry) ' if dry else ''}tri-bit packed 8 engine banks. 96-bit words = 32 INT3 weights "
          f"(bits [j*3+:3]). Next: dedup_engine_banks.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
