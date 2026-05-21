#!/usr/bin/env python3
"""Verify the Path-D banked weight memory map by reconstructing bytes
from the 8 .mem files and comparing them bit-exact against the original
PyTorch per-layer .hex files.

The test:
  For one or more heavy layers, walk the MAC-cycle range that
  `weight_memory_map.json` allocates to that layer. For each MAC cycle:
    * Read line `base_mac_cycle + offset` from each of the 8 bank .mem
      files (72 hex chars = 288 bits per line).
    * Decode the low 256 bits of each bank line into 32 INT8 bytes
      (byte 0 at bits[7:0]). Concatenate bank 0 .. bank 7 to get the
      256 weights for this MAC cycle (one per output-channel slot).
    * For each of the 256 slots, compute (oc_pass, slot_within_pass) and
      cross-check against `weight[oc, ic, kh, kw]` from the original
      .hex file (where oc = oc_pass*256 + slot_within_pass; out-of-range
      oc slots are expected to read 0 — padding).

Exits non-zero on ANY mismatch and prints the first failing coordinate.

This is the strongest possible local verification of Path D, short of
firing up a Verilator harness against the actual engine. It catches:
  * byte-order errors (LSB / MSB swap),
  * MAC-cycle iteration order mistakes (oc_pass × ic × kh × kw),
  * per-bank slot allocation errors,
  * pad-byte placement bugs,
  * base_mac_cycle layout bugs across multiple layers.

Usage:
    python scripts/verify_weight_memory_map.py [--layer node_conv_298 ...]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

NUM_BANKS = 8
MAC_COUNT = 256
WEIGHTS_PER_BANK = MAC_COUNT // NUM_BANKS  # 32


def detect_repo_root(script_path: Path) -> Path:
    override = os.environ.get("NN2RTL_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return script_path.resolve().parent.parent


def load_bank_lines(weights_dir: Path) -> list[list[str]]:
    out: list[list[str]] = []
    for b in range(NUM_BANKS):
        p = weights_dir / f"uram_weights_bank{b}.mem"
        with p.open("r", encoding="utf-8") as fh:
            lines = [line.strip() for line in fh if line.strip()]
        out.append(lines)
    return out


def decode_bank_line(line: str) -> list[int]:
    """A 72-hex-char URAM word: top 8 chars = zero pad, low 64 chars = 32
    weight bytes with byte 0 at bits[7:0] (rightmost two hex chars)."""
    if len(line) != 72:
        raise ValueError(f"bank line must be 72 hex chars, got {len(line)}: '{line}'")
    pad = line[:8]
    if pad != "00000000":
        raise ValueError(f"top 32 pad bits must be zero, got '{pad}'")
    low_256_hex = line[8:]
    raw = bytes.fromhex(low_256_hex)  # 32 bytes, MSB-first (byte 31 first)
    return list(reversed(raw))  # now byte 0 is at index 0


def read_pytorch_hex(weights_path: Path) -> bytes:
    out = bytearray()
    with weights_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            tok = line.strip()
            if tok:
                out.append(int(tok, 16))
    return bytes(out)


def resolve_layer_hex(layer_ir_path: Path, weights_dir: Path, module_id: str) -> Path:
    with layer_ir_path.open("r", encoding="utf-8") as fh:
        ir = json.load(fh)
    for layer in ir.get("layers", []):
        if layer.get("module_id") != module_id:
            continue
        wp = layer.get("weights_path")
        if not wp:
            break
        candidate = Path(wp)
        if candidate.is_file():
            return candidate
        rebased = weights_dir / candidate.name
        if rebased.is_file():
            return rebased
        break
    raise FileNotFoundError(f"could not resolve weights .hex for {module_id}")


def verify_layer(
    layer_entry: dict,
    bank_lines: list[list[str]],
    layer_ir_path: Path,
    weights_dir: Path,
) -> tuple[int, int, int]:
    """Returns (mac_cycles_checked, slots_compared, mismatches)."""
    module_id = layer_entry["module_id"]
    oc, ic, kh, kw = layer_entry["weight_shape"]
    base = layer_entry["base_mac_cycle"]
    n_cycles = layer_entry["size_mac_cycles"]

    pytorch_hex = resolve_layer_hex(layer_ir_path, weights_dir, module_id)
    data = read_pytorch_hex(pytorch_hex)
    if len(data) != oc * ic * kh * kw:
        raise SystemExit(
            f"{module_id}: pytorch hex has {len(data)} bytes but layer claims "
            f"{oc}x{ic}x{kh}x{kw} = {oc*ic*kh*kw}"
        )

    def w_byte(o: int, i_c: int, h: int, w_idx: int) -> int:
        return data[((o * ic + i_c) * kh + h) * kw + w_idx]

    oc_passes = math.ceil(oc / MAC_COUNT)
    expected_n = oc_passes * ic * kh * kw
    if expected_n != n_cycles:
        raise SystemExit(
            f"{module_id}: weight_memory_map says {n_cycles} MAC cycles but "
            f"layer shape implies {expected_n}"
        )

    mismatches = 0
    slots_compared = 0
    cycle_idx = 0
    for op in range(oc_passes):
        for i_c in range(ic):
            for h in range(kh):
                for w_idx in range(kw):
                    mac_cycle = base + cycle_idx
                    # Read all 8 banks at this MAC cycle.
                    for bank in range(NUM_BANKS):
                        bank_bytes = decode_bank_line(bank_lines[bank][mac_cycle])
                        for slot in range(WEIGHTS_PER_BANK):
                            oc_idx = op * MAC_COUNT + bank * WEIGHTS_PER_BANK + slot
                            expected = w_byte(oc_idx, i_c, h, w_idx) if oc_idx < oc else 0
                            actual = bank_bytes[slot]
                            if expected != actual:
                                mismatches += 1
                                if mismatches <= 5:
                                    print(
                                        f"  MISMATCH {module_id} @ cycle={mac_cycle} "
                                        f"bank={bank} slot={slot} oc={oc_idx} "
                                        f"ic={i_c} kh={h} kw={w_idx}: "
                                        f"expected=0x{expected:02x} actual=0x{actual:02x}",
                                        file=sys.stderr,
                                    )
                            slots_compared += 1
                    cycle_idx += 1
    return n_cycles, slots_compared, mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", default="resnet-50")
    parser.add_argument(
        "--layer",
        action="append",
        default=[],
        help="layer module_id to verify (can be passed multiple times). "
             "Default: verify the first heavy layer and the largest layer.",
    )
    args = parser.parse_args(argv)

    repo_root = detect_repo_root(Path(__file__))
    with (repo_root / "networks.json").open("r", encoding="utf-8") as fh:
        registry = json.load(fh)
    net = next(n for n in registry["networks"] if n["id"] == args.network)
    output_dir = (repo_root / net["outputDir"]).resolve()
    weights_dir = output_dir / "weights"
    layer_ir_path = output_dir / "layer_ir.json"

    with (weights_dir / "weight_memory_map.json").open("r", encoding="utf-8") as fh:
        wmap = json.load(fh)
    if wmap.get("schema") != "weight_memory_map_v2_banked":
        raise SystemExit(
            f"weight_memory_map.json has schema "
            f"'{wmap.get('schema')}' (expected weight_memory_map_v2_banked). "
            f"Re-run scripts/build_weight_memory_map.py."
        )
    layers = wmap["layers"]
    if not layers:
        raise SystemExit("weight_memory_map.json has no layers")

    print(f"Loading {NUM_BANKS} bank .mem files from {weights_dir} ...")
    bank_lines = load_bank_lines(weights_dir)
    expected_lines = wmap["total_mac_cycles"]
    for b, lines in enumerate(bank_lines):
        if len(lines) != expected_lines:
            raise SystemExit(
                f"bank {b} has {len(lines)} lines but weight_memory_map says "
                f"{expected_lines}"
            )

    if args.layer:
        targets = [lay for lay in layers if lay["module_id"] in args.layer]
        missing = set(args.layer) - {lay["module_id"] for lay in targets}
        if missing:
            raise SystemExit(f"unknown layers: {sorted(missing)}")
    else:
        layers_by_size = sorted(layers, key=lambda L: -L["size_mac_cycles"])
        targets = [layers[0], layers_by_size[0]]
        targets = [
            L for i, L in enumerate(targets)
            if L["module_id"] not in {t["module_id"] for t in targets[:i]}
        ]

    total_mismatches = 0
    for layer in targets:
        cycles, slots, miss = verify_layer(layer, bank_lines, layer_ir_path, weights_dir)
        print(
            f"  {layer['module_id']:<22} cycles={cycles:>6}  "
            f"slots_checked={slots:>10}  mismatches={miss}"
        )
        total_mismatches += miss

    if total_mismatches > 0:
        print(f"FAIL: {total_mismatches} mismatch(es).")
        return 2
    print(f"OK: {len(targets)} layer(s) reconstruct bit-exact from the 8 bank .mem files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
