#!/usr/bin/env python3
"""Phase 3 FIT (engine dedup): compact the 8 engine weight banks from 96659 ->
39424 rows by keeping ONLY the 14 engine-dispatch weight blocks (the other 57235
rows are the 39 spatial convs' weights, which build_weight_memory_map packs into
the banks but the engine NEVER reads). Each dispatch reads a contiguous hole-free
[base, base+size) block, so dedup = block-copy in base order + rebase. NO address-
generator change (weight_addr = cfg_weight_uram_base + dim-offset; only the base
moves). The new bases (prefix sums) go into nn2rtl_scheduler.v weight_base_word_rom
and the bank DEPTH (96659->39424) into nn2rtl_top.v.

Verified low-risk by workflow wcjp2inex. Byte-exact gate: run_nn2rtl_top_probe.ts
relu_48 multiset==0.00% + conv_246/conv_250 byte-exact.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WDIR = ROOT / "output/weights"
# 14 engine dispatches in base order (from nn2rtl_scheduler_schedule.json + weight_memory_map.json)
DISPATCHES = [
    ("node_conv_246", 11155, 2304), ("node_conv_250", 14483, 2048),
    ("node_conv_254", 17555, 2304), ("node_conv_260", 21907, 2304),
    ("node_conv_264", 25235, 1024), ("node_conv_266", 26259, 2304),
    ("node_conv_272", 30611, 2304), ("node_conv_278", 34963, 2304),
    ("node_conv_282", 38291, 2048), ("node_conv_286", 49555, 4096),
    ("node_conv_290", 61843, 4096), ("node_conv_294", 75155, 4096),
    ("node_conv_296", 79251, 4096), ("node_conv_300", 92563, 4096),
]

def main() -> int:
    dry = "--dry-run" in sys.argv
    total = sum(s for _, _, s in DISPATCHES)
    assert total == 39424, f"size sum {total} != 39424"
    new_bases = []; acc = 0
    for _, _, s in DISPATCHES:
        new_bases.append(acc); acc += s
    print(f"new bases (prefix sums): {new_bases}")
    for b in range(8):
        p = WDIR / f"uram_weights_bank{b}.mem"
        lines = p.read_text().splitlines()
        assert len(lines) == 96659, f"bank{b}: {len(lines)} lines != 96659"
        out = []
        for _, base, size in DISPATCHES:
            out.extend(lines[base:base + size])
        assert len(out) == 39424, f"bank{b}: compacted {len(out)} != 39424"
        if dry:
            print(f"  [dry] bank{b}: 96659 -> {len(out)}")
        else:
            p.write_text("\n".join(out) + "\n", newline="\n")
            print(f"  [ok] bank{b}: 96659 -> {len(out)} rows")
    print(f"\n{'(dry) ' if dry else ''}deduped 8 banks to 39424 rows.")
    print(f"ROM update (nn2rtl_scheduler.v weight_base_word_rom): {new_bases}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
