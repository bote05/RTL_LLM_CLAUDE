#!/usr/bin/env python3
"""Build the on-chip bias memory map (task 13a Path D, Bundle A follow-up).

The engine's requantisation pipeline consumes one wide bias word per
oc_pass per dispatched heavy layer. A wide bias word = MAC_COUNT (256)
INT32 biases packed into 8192 bits, byte 0 (lowest channel slot) at
bits [31:0], byte 1 at bits [63:32], etc.

The scheduler reads each layer's bias_base_word from its ROM and writes
it into the engine's config register at the start of each dispatch.
This script:
  1. Walks the dispatched heavy layer list (from
     06_phase1_compression_candidates_HEAVY.txt) in scheduler order.
  2. For each dispatched layer, reads its bias .hex file (one INT32 per
     line, 8 hex chars each), packs into ceil(oc / MAC_COUNT) wide bias
     words of 256 INT32 each. Out-of-range channel slots get zero.
  3. Concatenates all dispatched layers' wide bias words in dispatch
     order into a single bias.mem file. One wide bias word per line,
     2048 hex chars per line (one wide word = 8192 bits = 2048 hex).
  4. Emits a bias_memory_map.json with per-layer base_word offset (= sum
     of oc_passes for prior layers) and oc_passes count.
  5. Emits a Verilog header with localparam BIAS_BASE_<module_id>_WORDS
     for each dispatched layer.

The wrapper's `u_bias_mem` reads from this file. The scheduler's
bias_base_word_rom is populated from `bias_memory_map.json`'s per-layer
base_word field.

Hard gate: exits non-zero if the total number of wide bias words
exceeds the bias memory's SIZE_WORDS (256 by default in the wrapper).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path

MAC_COUNT = 256
BIAS_BYTES_PER_CHANNEL = 4   # INT32
BIAS_WIDE_WORD_BITS = MAC_COUNT * BIAS_BYTES_PER_CHANNEL * 8  # 8192
BIAS_WIDE_WORD_HEX_CHARS = BIAS_WIDE_WORD_BITS // 4           # 2048

# Default cap matches u_bias_mem's SIZE_WORDS in the wrapper generator.
DEFAULT_BIAS_MEM_CAPACITY = 256


def detect_repo_root(script_path: Path) -> Path:
    override = os.environ.get("NN2RTL_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return script_path.resolve().parent.parent


def load_network_config(repo_root: Path, network_id: str) -> dict:
    with (repo_root / "networks.json").open("r", encoding="utf-8") as fh:
        registry = json.load(fh)
    for net in registry["networks"]:
        if net["id"] == network_id:
            return net
    raise SystemExit(f"unknown network '{network_id}'")


def resolve_bias_path(bias_path: str, weights_dir: Path) -> Path:
    p = Path(bias_path)
    if p.is_file():
        return p
    rebased = weights_dir / p.name
    if rebased.is_file():
        return rebased
    raise FileNotFoundError(f"could not resolve bias hex: {bias_path}")


def read_int32_hex(path: Path) -> list[int]:
    out: list[int] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            tok = line.strip()
            if not tok:
                continue
            v = int(tok, 16)
            # Hex files are written as unsigned 32-bit; interpret two's-complement
            # so we can re-pack as signed little-endian later without sign loss.
            if v >= (1 << 31):
                v -= (1 << 32)
            out.append(v)
    return out


def pack_oc_pass_word(biases: list[int]) -> str:
    """Pack 256 INT32 biases into one 8192-bit wide word as MSB-first hex.

    Layout: channel slot 0 at bits [31:0], slot 1 at bits [63:32], ...,
    slot 255 at bits [8191:8160]. Within each 32-bit slot the value is
    laid out MSB-first: bits [31:24]=MSByte, ..., bits [7:0]=LSByte, so
    requant_pipeline.v reading `bias_in[lane*32 +: 32]` as `$signed`
    recovers the original signed INT32.

    Slot ordering: render slot 255 first (lands at high bits via $readmemh's
    MSB-first fill), slot 0 last. Byte ordering WITHIN each slot is
    big-endian (".pack(\">i\")") so the natural $signed read works.
    """
    if len(biases) != MAC_COUNT:
        raise ValueError(f"need {MAC_COUNT} bias values per wide word, got {len(biases)}")
    raw = bytearray()
    for slot in range(MAC_COUNT - 1, -1, -1):
        raw.extend(struct.pack(">i", biases[slot]))  # 4 bytes, big-endian
    return raw.hex()


def layer_oc_passes(layer: dict) -> int:
    weight_shape = layer.get("weight_shape", [])
    if len(weight_shape) != 4:
        raise SystemExit(f"{layer['module_id']}: weight_shape malformed: {weight_shape}")
    oc = weight_shape[0]
    return math.ceil(oc / MAC_COUNT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", default="resnet-50")
    parser.add_argument(
        "--heavy-list",
        default="docs/agent_tasks/06_phase1_compression_candidates_HEAVY.txt",
        help="newline-delimited list of dispatched heavy module ids",
    )
    parser.add_argument("--capacity", type=int, default=DEFAULT_BIAS_MEM_CAPACITY)
    parser.add_argument("--out-mem", default=None)
    parser.add_argument("--out-header", default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args(argv)

    repo_root = detect_repo_root(Path(__file__))
    net = load_network_config(repo_root, args.network)
    output_dir = (repo_root / net["outputDir"]).resolve()
    weights_dir = output_dir / "weights"
    layer_ir_path = output_dir / "layer_ir.json"

    out_mem = Path(args.out_mem) if args.out_mem else weights_dir / "bias.mem"
    out_header = Path(args.out_header) if args.out_header else weights_dir / "bias_memory_map.vh"
    out_json = Path(args.out_json) if args.out_json else weights_dir / "bias_memory_map.json"

    heavy_list_path = Path(args.heavy_list)
    if not heavy_list_path.is_absolute():
        heavy_list_path = repo_root / heavy_list_path
    with heavy_list_path.open("r", encoding="utf-8") as fh:
        heavy_modules = [line.strip() for line in fh if line.strip()]
    if not heavy_modules:
        raise SystemExit(f"empty heavy list: {heavy_list_path}")

    with layer_ir_path.open("r", encoding="utf-8") as fh:
        ir = json.load(fh)
    layers_by_id = {L["module_id"]: L for L in ir.get("layers", [])}

    lines: list[str] = []
    layers_out: list[dict] = []
    cur_word = 0

    for module_id in heavy_modules:
        layer = layers_by_id.get(module_id)
        if layer is None:
            raise SystemExit(f"heavy module '{module_id}' not in LayerIR")
        if layer.get("op_type") != "conv2d":
            raise SystemExit(f"heavy module '{module_id}' is not a conv2d")
        bias_path = layer.get("bias_path")
        oc = layer["weight_shape"][0]
        oc_passes = layer_oc_passes(layer)
        if not bias_path:
            # Conv with no bias: pad with zero biases.
            biases: list[int] = [0] * oc
        else:
            bias_hex = resolve_bias_path(bias_path, weights_dir)
            biases = read_int32_hex(bias_hex)
            if len(biases) != oc:
                raise SystemExit(
                    f"{module_id}: bias hex has {len(biases)} entries, "
                    f"layer claims oc={oc}"
                )

        # Pad biases to the next 256-multiple with zeros.
        padded_biases = biases + [0] * (oc_passes * MAC_COUNT - len(biases))
        for op in range(oc_passes):
            slice_ = padded_biases[op * MAC_COUNT:(op + 1) * MAC_COUNT]
            lines.append(pack_oc_pass_word(slice_))

        layers_out.append({
            "module_id": module_id,
            "oc": oc,
            "oc_passes": oc_passes,
            "base_word": cur_word,
            "size_words": oc_passes,
        })
        cur_word += oc_passes

    total_words = cur_word
    if total_words > args.capacity:
        print(
            f"ERROR: total wide bias words ({total_words}) exceeds bias mem "
            f"capacity ({args.capacity}). Increase u_bias_mem.SIZE_WORDS in "
            f"build_top_wrapper.ts.",
            file=sys.stderr,
        )
        return 1

    out_mem.parent.mkdir(parents=True, exist_ok=True)
    with out_mem.open("w", encoding="utf-8", newline="\n") as fh:
        for line in lines:
            fh.write(line)
            fh.write("\n")

    header_lines = [
        "// Auto-generated by scripts/build_bias_memory_map.py - do not hand-edit.",
        f"// network: {args.network}",
        f"// One wide bias word = {MAC_COUNT} INT32 biases = {BIAS_WIDE_WORD_BITS} bits.",
        f"// Total wide bias words: {total_words} of {args.capacity} capacity",
        "",
        f"localparam BIAS_TOTAL_WORDS = {total_words};",
        "",
    ]
    for layer in layers_out:
        mid = layer["module_id"]
        header_lines.append(f"localparam BIAS_BASE_{mid}_WORDS = {layer['base_word']};")
        header_lines.append(f"localparam BIAS_SIZE_{mid}_WORDS = {layer['size_words']};")
    with out_header.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(header_lines))
        fh.write("\n")

    sidecar = {
        "schema": "bias_memory_map_v1",
        "mac_count": MAC_COUNT,
        "bias_bytes_per_channel": BIAS_BYTES_PER_CHANNEL,
        "wide_word_bits": BIAS_WIDE_WORD_BITS,
        "wide_word_hex_chars": BIAS_WIDE_WORD_HEX_CHARS,
        "total_wide_bias_words": total_words,
        "capacity_wide_words": args.capacity,
        "utilisation_pct": round((total_words / args.capacity) * 100.0, 2),
        "layers": layers_out,
        "notes": (
            "Heavy layers in dispatch order. The scheduler emits "
            "bias_base_word = base_word for each dispatched layer; the "
            "engine's address generator computes "
            "bias_rd_addr = bias_base_word + oc_pass_idx."
        ),
    }
    with out_json.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(sidecar, fh, indent=2)
        fh.write("\n")

    print(f"Wrote {out_mem.relative_to(repo_root)}")
    print(f"  Wide bias words      : {total_words} / {args.capacity}")
    print(f"  Bytes per wide word  : {BIAS_WIDE_WORD_BITS // 8}")
    print(f"  Total bias bytes     : {total_words * (BIAS_WIDE_WORD_BITS // 8):,}")
    print(f"  Layers covered       : {len(layers_out)}")
    for layer in layers_out:
        print(f"    {layer['module_id']:<22} base_word={layer['base_word']:>3}  oc_passes={layer['oc_passes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
