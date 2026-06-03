#!/usr/bin/env python3
"""Build a deterministic URAM weight memory subsystem from the LayerIR.

Task 13a Path D — banked memory subsystem with native URAM widths.

For each conv2d layer in `output/layer_ir.json`, this script:
  1. Reads the per-layer INT8 weight `.hex` file (PyTorch [oc, ic, kh, kw] order,
     one byte per line MSB-first hex).
  2. Re-orders the weights into MAC-cycle order: for each (oc_pass, ic, kh, kw)
     coordinate the engine reads 256 weights in lockstep, one per output
     channel slot within the current oc_pass.
  3. Distributes those 256 weights across 8 parallel banks (32 weights per
     bank per MAC cycle). Bank N stores output-channel slots [N*32 .. N*32+31].
  4. Emits one `.mem` file per bank at 288 bits per line (native URAM cascade
     width). Each line: 32 zero-pad bits at the high end + 32 weight bytes
     in the low 256 bits, byte 0 at bits[7:0].

Deliverables (all under `<weights_dir>/`):
  - `uram_weights_bank<N>.mem`     N=0..7. The 8 bank images.
  - `weight_memory_map.vh`         Verilog header with per-layer base
                                    MAC-cycle indices.
  - `weight_memory_map.json`       Machine-readable sidecar (layers,
                                    total MAC cycles, URAM accounting).

Hard gate: exits non-zero if the per-bank URAM block count + bank count
would exceed the U250 URAM288 primitive budget (1,280).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Engine architectural constants — task 13a Path D commitments.
# ---------------------------------------------------------------------------
MAC_COUNT = 256                           # output-channel lanes per MAC cycle
WGT_W = 8                                 # INT8 weights
NUM_BANKS = 8                             # parallel URAM read banks
WEIGHTS_PER_BANK = MAC_COUNT // NUM_BANKS # = 32 bytes per bank per MAC cycle
BANK_USEFUL_BITS = WEIGHTS_PER_BANK * WGT_W  # = 256 bits of real data per bank line

# ---------------------------------------------------------------------------
# URAM physical primitive accounting.
# UltraScale+ URAM288 = 4,096 words × 72 bits per port, dual-port.
# A 288-bit-wide bank stripe = 4 URAM288 primitives in parallel
# (same 4,096 depth, 4× width).
# ---------------------------------------------------------------------------
URAM_WORD_BITS = 288                      # native cascade width
URAM_PRIMITIVE_BITS = 72
URAM_PRIMITIVE_DEPTH = 4096
URAM_CASCADE_FACTOR = URAM_WORD_BITS // URAM_PRIMITIVE_BITS  # 4 primitives per stripe
URAM_WORDS_PER_LANE = URAM_PRIMITIVE_DEPTH
URAM_PHYSICAL_BUDGET = 1280


def detect_repo_root(script_path: Path) -> Path:
    override = os.environ.get("NN2RTL_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return script_path.resolve().parent.parent


def load_network_config(repo_root: Path, network_id: str) -> dict:
    networks_file = repo_root / "networks.json"
    with networks_file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    for net in data.get("networks", []):
        if net.get("id") == network_id:
            return net
    raise SystemExit(f"network '{network_id}' not found in {networks_file}")


def resolve_hex_path(weights_path: str, weights_dir: Path) -> Path:
    candidate = Path(weights_path)
    if candidate.is_file():
        return candidate
    rebased = weights_dir / candidate.name
    if rebased.is_file():
        return rebased
    raise FileNotFoundError(
        f"could not resolve weights file: tried '{weights_path}' and '{rebased}'"
    )


def read_hex_bytes(hex_path: Path) -> bytes:
    out = bytearray()
    with hex_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            token = line.strip()
            if not token:
                continue
            out.append(int(token, 16))
    return bytes(out)


def layer_mac_cycles(oc: int, ic: int, kh: int, kw: int) -> int:
    """How many engine MAC cycles a layer takes = ceil(oc/MAC_COUNT) × ic × kh × kw."""
    oc_passes = math.ceil(oc / MAC_COUNT)
    return oc_passes * ic * kh * kw


def encode_bank_line(weights_for_bank: list[int]) -> str:
    """Render 32 INT8 weight bytes (one bank's slice of one MAC cycle) into a
    288-bit URAM word as 72 MSB-first hex characters.

    Bit layout in the URAM word:
      bits [255:0]  = the 32 weight bytes, byte 0 at bits [7:0]
                       (matches the canonical "byte i is at bits [i*8+7:i*8]"
                       convention used by node_conv_288's $readmemh setup).
      bits [287:256] = zero padding (URAM native is 288 bits; the MAC array
                       only consumes the low 256 bits per bank).

    `weights_for_bank` is the list of 32 INT8 byte values for this bank,
    with index 0 = output-channel slot N*32+0 (lowest-address slot in
    bank N).
    """
    if len(weights_for_bank) != WEIGHTS_PER_BANK:
        raise ValueError(
            f"bank line needs exactly {WEIGHTS_PER_BANK} bytes; got {len(weights_for_bank)}"
        )
    # Pack low 256 bits: reverse(weights) → byte 31 at bits[255:248], byte 0
    # at bits[7:0]. Then prepend 32 zero pad bits (= 8 hex chars '0').
    low_256_hex = bytes(reversed(weights_for_bank)).hex()
    return "00000000" + low_256_hex  # 8 + 64 = 72 hex chars = 288 bits


def layer_mac_cycle_image(
    data: bytes, oc: int, ic: int, kh: int, kw: int
) -> list[list[list[int]]]:
    """Re-order a layer's PyTorch [oc, ic, kh, kw] hex bytes into MAC-cycle
    × bank × byte-within-bank order.

    Returns a 3D list `image[mac_cycle][bank][byte_slot]` of INT8 ints.

    The engine's read order per layer:
        for oc_pass in 0..ceil(oc/MAC_COUNT):
          for ic_i  in 0..ic:
            for kh_i in 0..kh:
              for kw_i in 0..kw:
                # one MAC cycle: 256 output channels at one (ic, kh, kw)
                for bank in 0..NUM_BANKS:
                  for slot in 0..WEIGHTS_PER_BANK:
                    oc_idx = oc_pass * MAC_COUNT + bank * WEIGHTS_PER_BANK + slot
                    w = (data[oc_idx][ic_i][kh_i][kw_i]) if oc_idx < oc else 0
    """
    if len(data) != oc * ic * kh * kw:
        raise ValueError(
            f"weight tensor size mismatch: data has {len(data)} bytes, "
            f"layer claims {oc}x{ic}x{kh}x{kw} = {oc*ic*kh*kw}"
        )
    # Index helper: PyTorch [oc, ic, kh, kw] row-major.
    def w_byte(o: int, i_c: int, h: int, w: int) -> int:
        return data[((o * ic + i_c) * kh + h) * kw + w]

    oc_passes = math.ceil(oc / MAC_COUNT)
    image: list[list[list[int]]] = []
    for op in range(oc_passes):
        for i_c in range(ic):
            for h in range(kh):
                for w in range(kw):
                    cycle: list[list[int]] = []
                    for bank in range(NUM_BANKS):
                        bank_bytes: list[int] = []
                        for slot in range(WEIGHTS_PER_BANK):
                            oc_idx = op * MAC_COUNT + bank * WEIGHTS_PER_BANK + slot
                            bank_bytes.append(
                                w_byte(oc_idx, i_c, h, w) if oc_idx < oc else 0
                            )
                        cycle.append(bank_bytes)
                    image.append(cycle)
    return image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", default="resnet-50")
    parser.add_argument("--layer-ir", default=None)
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument("--out-header", default=None)
    parser.add_argument("--out-json", default=None)
    parser.add_argument(
        "--engine-modules",
        "--heavy-list",
        dest="engine_modules",
        default=None,
        help=(
            "Optional newline-delimited file of engine-dispatched (heavy) "
            "module ids. When given, ONLY these convs are packed into the "
            "URAM weight banks (their base_mac_cycle offsets are computed in "
            "the file's listed order). When omitted, ALL conv2d layers are "
            "packed (legacy ResNet flow). Required for networks where some "
            "convs stay spatial (e.g. MobileNetV2 depthwise) so they do not "
            "pollute the engine banks."
        ),
    )
    args = parser.parse_args(argv)

    repo_root = detect_repo_root(Path(__file__))
    net = load_network_config(repo_root, args.network)
    output_dir = (repo_root / net["outputDir"]).resolve()
    weights_dir = Path(args.weights_dir) if args.weights_dir else output_dir / "weights"

    layer_ir_path = Path(args.layer_ir) if args.layer_ir else output_dir / "layer_ir.json"
    out_header = Path(args.out_header) if args.out_header else weights_dir / "weight_memory_map.vh"
    out_json = Path(args.out_json) if args.out_json else weights_dir / "weight_memory_map.json"

    with layer_ir_path.open("r", encoding="utf-8") as fh:
        ir = json.load(fh)

    # Optional engine-module (heavy-list) filter. When provided, only the
    # listed convs are packed into the URAM banks, walked in the file's order
    # so the base_mac_cycle layout matches the bias/scale maps' dispatch order.
    # When absent, every conv2d in LayerIR order is packed (legacy behaviour).
    engine_order: list[str] | None = None
    engine_set: set[str] | None = None
    if args.engine_modules:
        em_path = Path(args.engine_modules)
        if not em_path.is_absolute():
            em_path = repo_root / em_path
        with em_path.open("r", encoding="utf-8") as fh:
            engine_order = [
                ln.strip() for ln in fh
                if ln.strip() and not ln.strip().startswith("#")
            ]
        if not engine_order:
            raise SystemExit(f"engine-modules list is empty: {em_path}")
        engine_set = set(engine_order)

    layers_by_id = {L["module_id"]: L for L in ir.get("layers", [])}
    if engine_order is not None:
        layer_iter = []
        for mid in engine_order:
            L = layers_by_id.get(mid)
            if L is None:
                raise SystemExit(
                    f"engine module '{mid}' not found in LayerIR ({layer_ir_path})"
                )
            if L.get("op_type") != "conv2d":
                raise SystemExit(f"engine module '{mid}' is not a conv2d")
            layer_iter.append(L)
    else:
        layer_iter = ir.get("layers", [])

    # bank_lines[bank] = list of 72-hex-char strings, one per MAC cycle.
    bank_lines: list[list[str]] = [[] for _ in range(NUM_BANKS)]
    layers_out: list[dict] = []
    cur_mac_cycle = 0

    for layer in layer_iter:
        if layer.get("op_type") != "conv2d":
            continue
        weights_path = layer.get("weights_path")
        if not weights_path:
            continue
        module_id = layer["module_id"]
        weight_shape = layer.get("weight_shape")
        if not weight_shape or len(weight_shape) != 4:
            raise SystemExit(
                f"{module_id}: weight_shape must be 4D [oc, ic, kh, kw]; got {weight_shape}"
            )
        oc, ic, kh, kw = weight_shape
        hex_path = resolve_hex_path(weights_path, weights_dir)
        data = read_hex_bytes(hex_path)
        if not data:
            continue

        image = layer_mac_cycle_image(data, oc, ic, kh, kw)
        layer_cycles = len(image)

        for cycle in image:
            for bank_idx, bank_bytes in enumerate(cycle):
                bank_lines[bank_idx].append(encode_bank_line(bank_bytes))

        layers_out.append({
            "module_id": module_id,
            "weight_shape": [oc, ic, kh, kw],
            "oc_passes": math.ceil(oc / MAC_COUNT),
            "base_mac_cycle": cur_mac_cycle,
            "size_mac_cycles": layer_cycles,
        })
        cur_mac_cycle += layer_cycles

    total_mac_cycles = cur_mac_cycle

    # URAM block accounting per bank.
    # Each bank is a 288-bit-wide × `total_mac_cycles`-deep memory.
    # Vivado places it as a stripe of 4 URAM288 primitives wide × cascade
    # depth of ceil(total_mac_cycles / 4096) primitives deep.
    per_bank_depth_lanes = max(1, math.ceil(total_mac_cycles / URAM_WORDS_PER_LANE))
    per_bank_uram_blocks = per_bank_depth_lanes * URAM_CASCADE_FACTOR
    total_uram_blocks = per_bank_uram_blocks * NUM_BANKS
    utilisation = (total_uram_blocks / URAM_PHYSICAL_BUDGET) * 100.0

    # Write the 8 .mem files.
    weights_dir.mkdir(parents=True, exist_ok=True)
    for bank_idx in range(NUM_BANKS):
        out_mem = weights_dir / f"uram_weights_bank{bank_idx}.mem"
        with out_mem.open("w", encoding="utf-8", newline="\n") as fh:
            for line in bank_lines[bank_idx]:
                fh.write(line)
                fh.write("\n")

    header_lines = [
        "// Auto-generated by scripts/build_weight_memory_map.py — do not hand-edit.",
        f"// network: {args.network}",
        f"// Path D banked layout: {NUM_BANKS} parallel banks,",
        f"//   each {URAM_WORD_BITS} bits wide (native URAM cascade), used as",
        f"//   {BANK_USEFUL_BITS} useful bits + {URAM_WORD_BITS - BANK_USEFUL_BITS} zero-pad bits per line.",
        f"// MAC array consumes {MAC_COUNT} × {WGT_W} = {MAC_COUNT * WGT_W} bits per cycle.",
        f"//",
        f"// Total MAC cycles    : {total_mac_cycles}",
        f"// Per-bank depth (lanes of {URAM_WORDS_PER_LANE}): {per_bank_depth_lanes}",
        f"// URAM288 primitives/bank: {per_bank_uram_blocks}",
        f"// Total URAM288 primitives: {total_uram_blocks} of {URAM_PHYSICAL_BUDGET} "
        f"({utilisation:.2f}% of U250 URAM budget)",
        "",
        f"localparam URAM_WEIGHT_BANKS = {NUM_BANKS};",
        f"localparam URAM_BANK_BITS    = {URAM_WORD_BITS};",
        f"localparam URAM_BANK_USEFUL_BITS = {BANK_USEFUL_BITS};",
        f"localparam URAM_TOTAL_MAC_CYCLES = {total_mac_cycles};",
        "",
    ]
    for layer in layers_out:
        mid = layer["module_id"]
        header_lines.append(
            f"localparam MAC_CYCLE_BASE_{mid} = {layer['base_mac_cycle']};"
        )
        header_lines.append(
            f"localparam MAC_CYCLES_{mid}     = {layer['size_mac_cycles']};"
        )
    with out_header.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(header_lines))
        fh.write("\n")

    sidecar = {
        "schema": "weight_memory_map_v2_banked",
        "engine_modules_filter": (
            str(Path(args.engine_modules)) if args.engine_modules else None
        ),
        "engine_module_count": (len(engine_order) if engine_order is not None else None),
        "mac_count": MAC_COUNT,
        "num_banks": NUM_BANKS,
        "weights_per_bank_per_cycle": WEIGHTS_PER_BANK,
        "bank_useful_bits": BANK_USEFUL_BITS,
        "bank_line_bits": URAM_WORD_BITS,
        "uram_primitive_bits": URAM_PRIMITIVE_BITS,
        "uram_primitive_depth": URAM_PRIMITIVE_DEPTH,
        "cascade_factor": URAM_CASCADE_FACTOR,
        "total_mac_cycles": total_mac_cycles,
        "per_bank_depth_lanes": per_bank_depth_lanes,
        "per_bank_uram_blocks": per_bank_uram_blocks,
        "total_uram_blocks_required": total_uram_blocks,
        "uram_capacity_blocks": URAM_PHYSICAL_BUDGET,
        "utilisation_pct": round(utilisation, 2),
        "notes": (
            "Path D layout (task 13a). The engine reads weights in MAC-cycle "
            "order: for each (oc_pass, ic, kh, kw) coordinate, all NUM_BANKS "
            "banks are read in lockstep at the same MAC-cycle index. Bank N "
            "holds output-channel slots [N*32 .. N*32+31] for the current "
            "oc_pass. The wrapper concatenates the low BANK_USEFUL_BITS of "
            "each bank into a MAC_COUNT*WGT_W = 2048-bit weight bus. The top "
            "32 bits of each 288-bit bank line are zero-padded; URAM "
            "primitives are 100% physically used (each block is 4096 entries "
            "× 72 bits and the cascade is configured at native width). The "
            "engine_weight_rd_addr emitted by the address_generator is "
            "already in MAC-cycle units, so there is no '>>3' conversion."
        ),
        "layers": layers_out,
    }
    with out_json.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(sidecar, fh, indent=2)
        fh.write("\n")

    total_bytes_written = total_mac_cycles * MAC_COUNT  # 256 useful weight bytes per cycle (across all 8 banks)
    print(f"Banked layout (Path D):")
    print(f"  Total MAC cycles            : {total_mac_cycles}")
    print(f"  Useful weight bytes packed  : {total_bytes_written:,}")
    print(f"  Per-bank lines              : {total_mac_cycles} × {URAM_WORD_BITS} bits")
    print(f"  Per-bank URAM288 primitives : {per_bank_uram_blocks}")
    print(f"  Total URAM288 primitives    : {total_uram_blocks} / {URAM_PHYSICAL_BUDGET}")
    print(f"  URAM utilisation            : {utilisation:.2f}%")
    print(f"  Conv layers placed          : {len(layers_out)}")

    if total_uram_blocks > URAM_PHYSICAL_BUDGET:
        print(
            f"ERROR: total URAM288 primitives required ({total_uram_blocks}) "
            f"exceeds U250 budget ({URAM_PHYSICAL_BUDGET}).",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
