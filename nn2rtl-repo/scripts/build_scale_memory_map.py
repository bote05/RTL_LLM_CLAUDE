#!/usr/bin/env python3
"""Build the engine's PER-OUTPUT-CHANNEL requant scale ROM (Phase 2 / INT4-GPTQ).

Mirrors build_bias_memory_map.py EXACTLY so the engine can read it with the same
per-oc_pass addressing as the bias ROM. One wide scale word per oc_pass per
dispatched heavy layer = MAC_COUNT(256) lanes x 32 bits = 8192 bits = 2048 hex.
Per-lane 32-bit slot: bits[15:0]=SCALE_MULT (15-bit), bits[21:16]=SCALE_SHIFT
(<=23), from golden_impl.compute_scale_approx(layer.scale_factor_per_oc[ch]).
Slot 255 rendered first (MSB-first $readmemh fill), big-endian per slot — same
convention as the bias word so `scale_in[lane*32 +: 32]` recovers the slot.

Emits weights/scale.mem + scale_memory_map.json (+ .vh). base_word per layer =
running sum of oc_passes (identical to bias_base_word), so the scheduler can
reuse the bias base or carry an identical scale_base_word_rom.
"""
from __future__ import annotations
import json, struct, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root (for scripts.*)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_bias_memory_map import (  # noqa: E402
    detect_repo_root, load_network_config, layer_oc_passes, MAC_COUNT,
)
from golden_impl import compute_scale_approx  # noqa: E402

DEFAULT_HEAVY = "docs/agent_tasks/06_phase1_compression_candidates_HEAVY.txt"


def pack_scale_word(packed32: list[int]) -> str:
    """256 x 32-bit slots -> 8192-bit MSB-first hex (slot 255 first, BE per slot)."""
    if len(packed32) != MAC_COUNT:
        raise ValueError(f"need {MAC_COUNT} slots, got {len(packed32)}")
    raw = bytearray()
    for slot in range(MAC_COUNT - 1, -1, -1):
        raw.extend(struct.pack(">I", packed32[slot] & 0xFFFFFFFF))
    return raw.hex()


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--network", default="resnet-50")
    ap.add_argument("--heavy-list", default=DEFAULT_HEAVY)
    ap.add_argument("--out-mem", default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    repo = detect_repo_root(Path(__file__))
    net = load_network_config(repo, args.network)
    out_dir = (repo / net["outputDir"]).resolve()
    wdir = out_dir / "weights"
    ir = json.loads((out_dir / "layer_ir.json").read_text())
    by_id = {L["module_id"]: L for L in ir.get("layers", [])}

    hp = Path(args.heavy_list)
    if not hp.is_absolute():
        hp = repo / hp
    heavy = [l.strip() for l in hp.read_text().splitlines() if l.strip()]

    out_mem = Path(args.out_mem) if args.out_mem else wdir / "scale.mem"
    out_json = Path(args.out_json) if args.out_json else wdir / "scale_memory_map.json"

    lines, layers_out, cur = [], [], 0
    for mid in heavy:
        L = by_id.get(mid)
        if L is None or L.get("op_type") != "conv2d":
            raise SystemExit(f"{mid}: missing/not conv2d in LayerIR")
        oc = L["weight_shape"][0]
        oc_passes = layer_oc_passes(L)
        per_oc = L.get("scale_factor_per_oc")
        if per_oc is None:
            raise SystemExit(f"{mid}: no scale_factor_per_oc (regenerate goldens with GPTQ)")
        if len(per_oc) != oc:
            raise SystemExit(f"{mid}: scale_factor_per_oc len {len(per_oc)} != oc {oc}")
        packed = []
        for ch in range(oc):
            mult, shift = compute_scale_approx(float(per_oc[ch]))
            packed.append(((shift & 0x3F) << 16) | (mult & 0xFFFF))
        packed += [(0 << 16) | 1] * (oc_passes * MAC_COUNT - len(packed))  # pad: mult=1,shift=0
        for op in range(oc_passes):
            lines.append(pack_scale_word(packed[op * MAC_COUNT:(op + 1) * MAC_COUNT]))
        layers_out.append({"module_id": mid, "oc": oc, "oc_passes": oc_passes,
                           "base_word": cur, "size_words": oc_passes})
        cur += oc_passes

    out_mem.parent.mkdir(parents=True, exist_ok=True)
    out_mem.write_text("".join(l + "\n" for l in lines), encoding="utf-8", newline="\n")
    out_json.write_text(json.dumps({"schema": "scale_memory_map_v1",
                                    "wide_word_bits": MAC_COUNT * 32, "total_words": cur,
                                    "layers": layers_out}, indent=2), encoding="utf-8")
    print(f"[scale-mem] wrote {out_mem.relative_to(repo)} ({cur} wide words, {len(lines)} lines)")
    print(f"[scale-mem] layers: " + ", ".join(f"{l['module_id']}@{l['base_word']}({l['oc_passes']})" for l in layers_out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
