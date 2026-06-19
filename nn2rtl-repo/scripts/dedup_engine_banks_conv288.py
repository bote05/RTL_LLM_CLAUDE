#!/usr/bin/env python3
"""[CONV288-ENGINE 2026-06-12] 18-dispatch engine-bank compaction with a
byte-exact PROOF against the live 17-dispatch banks (K5-lineage; extends
scripts/dedup_engine_banks_k5.py).

conv_288 (1x1 1024->2048 s2 projection, weight_bits=3 = the 18th INT3
Config-B layer) moves from SPATIAL fabric onto the shared engine as
dispatch 10 (before conv_286 — add_13's main-vs-skip drain order demands
it). Its weights were ALWAYS packed by build_weight_memory_map.py at
base 53651 (size 8192 = 8 oc_passes x kt 1024) — the 17-entry dedup
simply dropped them, exactly like K5 found for 284/292/298.

Run on FRESH full (96659-row, 288-bit) banks from build_weight_memory_map:

  PROOF : compact(OLD_17) + int3-pack  ==  live banks byte-for-byte
          (backups/conv288_engine_20260612/uram_weights_bank*.mem).
          Proves the regenerated source is identical to what produced the
          vec0+vec1-byte-exact deployed banks.
  INT3  : every byte of conv_288's region decodes to signed int8 in
          [-4, 3] — the &0x7 3-bit pack is value-preserving (the engine
          datapath is WGT_W=3; an INT4 layer here would be CORRUPTED).
  BUILD : compact(NEW_18) + int3-pack  ->  output/weights/uram_weights_bank*.mem
          (75264 rows x 96 bits), prints the 18 new prefix-sum bases for
          nn2rtl_scheduler.v weight_base_word_rom + the new bank DEPTH.

NOTE: the deployed banks are the KPAR8-repacked *_kp8.mem files — after
this script, scripts/repack_resnet_kpar8_banks.py (constants updated to
75264/18) MUST be re-run on the new 18-entry banks, AFTER the scheduler
is patched to 18 dispatches (its P0 parses the deployed scheduler ROMs).

Usage: python scripts/dedup_engine_banks_conv288.py [--dry-run]
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WDIR = ROOT / "output/weights"
LIVE_BACKUP = ROOT / "backups/conv288_engine_20260612"

# The deployed 17-dispatch compaction (== dedup_engine_banks_k5.NEW_17).
OLD_17 = [
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
# 18 dispatches in ORIGINAL-base order (base order == block-copy order).
# conv_288 @53651 (= 49555 + 4096, contiguous after conv_286's region).
NEW_18 = [
    ("node_conv_246", 11155, 2304), ("node_conv_250", 14483, 2048),
    ("node_conv_254", 17555, 2304), ("node_conv_260", 21907, 2304),
    ("node_conv_264", 25235, 1024), ("node_conv_266", 26259, 2304),
    ("node_conv_272", 30611, 2304), ("node_conv_278", 34963, 2304),
    ("node_conv_282", 38291, 2048), ("node_conv_284", 40339, 9216),
    ("node_conv_286", 49555, 4096), ("node_conv_288", 53651, 8192),
    ("node_conv_290", 61843, 4096), ("node_conv_292", 65939, 9216),
    ("node_conv_294", 75155, 4096), ("node_conv_296", 79251, 4096),
    ("node_conv_298", 83347, 9216), ("node_conv_300", 92563, 4096),
]
C288_BASE, C288_SIZE = 53651, 8192


def repack_line(line: str) -> str:
    # identical to dedup_engine_banks_k5.repack_line (nibble_engine_banks_int3)
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
    exp18 = sum(s for _, _, s in NEW_18)
    assert exp18 == 75264 and exp18 % 8 == 0
    print(f"NEW_18 total rows (bank DEPTH) = {exp18} (KPAR8 wide lines = {exp18 // 8})")
    bases = []
    acc = 0
    for name, _, s in NEW_18:
        bases.append((name, acc)); acc += s

    int3_checked = 0
    for b in range(8):
        p = WDIR / f"uram_weights_bank{b}.mem"
        full = p.read_text().splitlines()
        assert len(full) == 96659 and len(full[0]) == 72, \
            f"bank{b}: need FRESH full banks from build_weight_memory_map (got {len(full)} x {len(full[0])}ch)"
        # PROOF: 17-entry compaction reproduces the live deployed banks exactly
        live = (LIVE_BACKUP / f"uram_weights_bank{b}.mem").read_text().splitlines()
        proof = compact_pack(full, OLD_17)
        assert proof == live, f"bank{b}: PROOF FAILED — 17-entry compaction != live deployed bank"
        # INT3 range check on conv_288's region: every 8-bit slot must be a
        # signed value in [-4, 3] or the 3-bit pack would corrupt it.
        for w in range(C288_BASE, C288_BASE + C288_SIZE):
            v = int(full[w], 16)
            for j in range(32):
                byte = (v >> (j * 8)) & 0xFF
                sv = byte - 256 if byte >= 128 else byte
                if not (-4 <= sv <= 3):
                    print(f"INT3 FAIL bank{b} word={w} slot={j}: int8={sv} "
                          f"outside [-4,3] — conv_288 is NOT INT3-packable")
                    return 1
                int3_checked += 1
        # BUILD the 18-entry banks
        new = compact_pack(full, NEW_18)
        assert len(new) == exp18
        if dry:
            print(f"  [dry] bank{b}: PROOF OK (17->live byte-exact); INT3 OK; 18-entry = {len(new)} rows")
        else:
            p.write_text("\n".join(new) + "\n", newline="\n")
            print(f"  [ok] bank{b}: PROOF OK; INT3 OK; wrote {len(new)} rows x 24ch")

    print(f"\nINT3 range check: {int3_checked} conv_288 weight slots all in [-4,3]")
    print("weight_base_word_rom (18 dispatches, dedup'd bases):")
    for name, nb in bases:
        print(f"  {name}: {nb}")
    print(f"nn2rtl_top.v engine bank DEPTH (wide): 8384 -> {exp18 // 8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
