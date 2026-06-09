#!/usr/bin/env python3
"""[K5 ENGINE-DISPATCH 2026-06-09] 17-dispatch engine-bank compaction with a
byte-exact PROOF against the live 14-dispatch banks.

K5 moves node_conv_284/292/298 from spatial fabric onto the shared engine
(deleting the ~200K-LUT congestion wall). Their weights were ALWAYS packed by
build_weight_memory_map.py (bases 40339/65939/83347) — the old 14-entry
dedup_engine_banks.py simply dropped them. This script, run on FRESH full
(96659-row, 288-bit) banks from build_weight_memory_map.py:

  PROOF : compact(OLD_14) + int3-pack  ==  live banks byte-for-byte
          (backups/k5_engine_dispatch_20260609/uram_weights_bank*.mem).
          Proves the regenerated source is identical to what produced the
          e2e-byte-exact deployed banks.
  BUILD : compact(NEW_17) + int3-pack  ->  output/weights/uram_weights_bank*.mem
          (67072 rows x 96 bits), prints the 17 new prefix-sum bases for
          nn2rtl_scheduler.v weight_base_word_rom + the new bank DEPTH for
          nn2rtl_top.v.

Run AFTER build_weight_memory_map.py (72-char lines required).
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WDIR = ROOT / "output/weights"
LIVE_BACKUP = ROOT / "backups/k5_engine_dispatch_20260609"

OLD_14 = [
    ("node_conv_246", 11155, 2304), ("node_conv_250", 14483, 2048),
    ("node_conv_254", 17555, 2304), ("node_conv_260", 21907, 2304),
    ("node_conv_264", 25235, 1024), ("node_conv_266", 26259, 2304),
    ("node_conv_272", 30611, 2304), ("node_conv_278", 34963, 2304),
    ("node_conv_282", 38291, 2048), ("node_conv_286", 49555, 4096),
    ("node_conv_290", 61843, 4096), ("node_conv_294", 75155, 4096),
    ("node_conv_296", 79251, 4096), ("node_conv_300", 92563, 4096),
]
# 17 dispatches in ORIGINAL-base order (base order == block-copy order).
NEW_17 = [
    ("node_conv_246", 11155, 2304), ("node_conv_250", 14483, 2048),
    ("node_conv_254", 17555, 2304), ("node_conv_260", 21907, 2304),
    ("node_conv_264", 25235, 1024), ("node_conv_266", 26259, 2304),
    ("node_conv_272", 30611, 2304), ("node_conv_278", 34963, 2304),
    ("node_conv_282", 38291, 2048), ("node_conv_284", 40339, 9216),
    ("node_conv_286", 49555, 4096), ("node_conv_290", 61843, 4096),
    ("node_conv_292", 65939, 9216), ("node_conv_294", 75155, 4096),
    ("node_conv_296", 79251, 4096), ("node_conv_298", 83347, 9216),
    ("node_conv_300", 92563, 4096),
]


def repack_line(line: str) -> str:
    # identical to nibble_engine_banks_int3.repack_line
    val = int(line, 16)
    out = 0
    for j in range(32):
        out |= ((val >> (j * 8)) & 0x7) << (j * 3)
    return format(out, "024x")


def compact_pack(full: list[str], dispatches) -> list[str]:
    out: list[str] = []
    for _, base, size in dispatches:
        out.extend(full[base:base + size])
    return [repack_line(l) for l in out]


def main() -> int:
    dry = "--dry-run" in sys.argv
    exp17 = sum(s for _, _, s in NEW_17)
    print(f"NEW_17 total rows (bank DEPTH) = {exp17}")
    bases = []
    acc = 0
    for name, _, s in NEW_17:
        bases.append((name, acc)); acc += s

    for b in range(8):
        p = WDIR / f"uram_weights_bank{b}.mem"
        full = p.read_text().splitlines()
        assert len(full) == 96659 and len(full[0]) == 72, \
            f"bank{b}: need FRESH full banks from build_weight_memory_map (got {len(full)} x {len(full[0])}ch)"
        # PROOF: 14-entry compaction reproduces the live deployed banks exactly
        live = (LIVE_BACKUP / f"uram_weights_bank{b}.mem").read_text().splitlines()
        proof = compact_pack(full, OLD_14)
        assert proof == live, f"bank{b}: PROOF FAILED — 14-entry compaction != live deployed bank"
        # BUILD the 17-entry banks
        new = compact_pack(full, NEW_17)
        assert len(new) == exp17
        if dry:
            print(f"  [dry] bank{b}: PROOF OK (14->live byte-exact); 17-entry = {len(new)} rows")
        else:
            p.write_text("\n".join(new) + "\n", newline="\n")
            print(f"  [ok] bank{b}: PROOF OK; wrote {len(new)} rows x 24ch")

    print("\nweight_base_word_rom (17 dispatches, dedup'd bases):")
    for name, nb in bases:
        print(f"  {name}: {nb}")
    print(f"nn2rtl_top.v engine bank DEPTH: 39424 -> {exp17}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
