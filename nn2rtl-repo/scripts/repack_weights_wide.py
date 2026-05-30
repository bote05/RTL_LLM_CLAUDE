#!/usr/bin/env python3
"""Repack per-module weight hex files into the wide layout used by
conv_datapath_parallel.

Input (flat, 1 byte hex per line):
    line i = weight at (oc * K_TOTAL + k) where i = oc*K_TOTAL + k.
    Sequential OC then K.

Output (wide, MP bytes packed per line as a single hex of length MP*2):
    line i = packed{w[oc_group*MP+MP-1, k], ..., w[oc_group*MP+0, k]}
    where i = oc_group * K_TOTAL + k, oc_group in [0..OC_PASSES-1],
    k in [0..K_TOTAL-1]. Padding with zero when oc_group*MP+lane >= OC.

Lanes are packed least-significant-byte first to match Verilog's
`weight_word_q[i*8 +: 8]` indexing convention in conv_datapath_parallel.

Usage:
    python scripts/repack_weights_wide.py \
        --input output/weights/node_conv_196_weights.hex \
        --output output/weights/node_conv_196_weights_wide.hex \
        --oc 64 --k-total 147 --mp 8

Or batch mode (auto-derives shape from sidecar):
    python scripts/repack_weights_wide.py --batch \
        --sidecar-dir output/tb --weights-dir output/weights \
        --mp 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


def read_flat_weights(path: Path) -> list[int]:
    """Read a flat weight hex file (one byte hex per line) as INT8 signed.

    Returned values are in [-128, 127], matching the byte interpretation
    expected by Verilog $readmemh into reg signed [7:0].
    """
    out: list[int] = []
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("//"):
            continue
        # one byte hex (00..ff) per line, signed interpretation
        v = int(s, 16)
        if v > 0x7f:
            v -= 0x100  # two's complement
        out.append(v)
    return out


def write_wide_weights(out_path: Path, weights: list[int], oc: int, k_total: int, mp: int, mp_k: int = 1) -> tuple[int, int]:
    """Write the MP*MP_K-byte-packed hex file. Returns (entries_written, padded_zeros).

    For mp_k=1 this is the original MP-byte-per-word format (consumed by
    conv_datapath_parallel). For mp_k>1, each entry contains MP*MP_K bytes
    laid out as: bits [(lane*MP_K + kpos)*8 +: 8] = weight at
    (oc=g*mp+lane, k=k_group*mp_k+kpos). Consumed by conv_datapath_mp_k.
    """
    if len(weights) != oc * k_total:
        raise ValueError(
            f"weight count mismatch: file has {len(weights)} bytes, expected {oc * k_total} (oc={oc}, k_total={k_total})"
        )
    if k_total % mp_k != 0:
        raise ValueError(f"k_total ({k_total}) must be divisible by mp_k ({mp_k})")
    oc_passes = (oc + mp - 1) // mp
    k_groups = k_total // mp_k
    bytes_per_word = mp * mp_k
    # INT4 NIBBLE-PACKING: each weight is stored as a 4-bit nibble (2 weights/byte),
    # halving weight BRAM. One hex char per weight => word = mp*mp_k hex chars.
    # The RTL (conv_datapath_mp_k.v) reads weight_word_q[(lane*mp_k+kpos)*4 +: 4] (signed).
    nibbles_per_word = bytes_per_word
    lines: list[str] = []
    padded = 0
    for g in range(oc_passes):
        for kg in range(k_groups):
            packed = 0
            for lane in range(mp):
                for kpos in range(mp_k):
                    global_oc = g * mp + lane
                    k_lin = kg * mp_k + kpos
                    if global_oc < oc:
                        byte_val = weights[global_oc * k_total + k_lin] & 0xf
                    else:
                        byte_val = 0
                        padded += 1
                    packed |= byte_val << ((lane * mp_k + kpos) * 4)
            lines.append(format(packed, f"0{nibbles_per_word}x"))
    out_path.write_text("\n".join(lines) + "\n")
    return len(lines), padded


def derive_shape_from_rtl(rtl_path: Path) -> tuple[int, int, int, int, int] | None:
    """Parse IC/OC/KH/KW from a node_conv_*.v file's localparams.

    Supports both `localparam integer IC = N;` (standard) and
    `localparam IC = N, OC = M, ...;` (pointwise modules use multi-decl
    form) by scanning anywhere in the file.
    """
    import re
    if not rtl_path.exists():
        return None
    txt = rtl_path.read_text()
    def grab(name: str) -> int | None:
        # Try standard form first, then multi-decl form.
        m = re.search(rf"localparam\s+(?:integer\s+)?{name}\s*=\s*(\d+)", txt)
        return int(m.group(1)) if m else None
    ic = grab("IC")
    oc = grab("OC")
    kh = grab("KH")
    kw = grab("KW")
    if None in (ic, oc, kh, kw):
        return None
    return oc, ic, kh, kw, ic * kh * kw  # type: ignore


def find_mp_in_rtl(rtl_path: Path) -> int | None:
    """Heuristically scrape MP from a node_conv_*.v file. Looks for
    `localparam [integer] MP = N` (with or without integer keyword)."""
    import re
    if not rtl_path.exists():
        return None
    with rtl_path.open() as f:
        head = "".join(f.readline() for _ in range(140))
    # Match either `localparam MP = N` or `localparam integer MP = N`,
    # with optional whitespace.
    m = re.search(r"localparam\s+(?:integer\s+)?MP\s*=\s*(\d+)", head)
    if m:
        return int(m.group(1))
    return None


def batch(sidecar_dir: Path, weights_dir: Path, rtl_dir: Path, default_mp: int, dry_run: bool) -> None:
    repacked = 0
    skipped = 0
    failed = 0
    for sidecar_path in sorted(sidecar_dir.glob("node_conv_*.sidecar.json")):
        module_id = sidecar_path.stem.replace(".sidecar", "")
        weight_path = weights_dir / f"{module_id}_weights.hex"
        meta_path = rtl_dir / f"{module_id}.meta.json"
        rtl_path = rtl_dir / f"{module_id}.v"
        out_path = weights_dir / f"{module_id}_weights_wide.hex"

        if not weight_path.exists():
            print(f"[skip] {module_id}: weight file missing", file=sys.stderr)
            skipped += 1
            continue
        if not meta_path.exists():
            print(f"[skip] {module_id}: meta.json missing", file=sys.stderr)
            skipped += 1
            continue

        try:
            # Parse shape from the actual RTL file (meta.json is stale and
            # has the original Foundry verilog, not the current edits).
            shape = derive_shape_from_rtl(rtl_path)
            if shape is None:
                print(f"[skip] {module_id}: not a conv2d or no weight_shape", file=sys.stderr)
                skipped += 1
                continue
            oc, ic, kh, kw, k_total = shape
            mp = find_mp_in_rtl(rtl_path) or default_mp
            weights = read_flat_weights(weight_path)
            if dry_run:
                print(f"[dry-run] {module_id}: OC={oc} K_TOTAL={k_total} MP={mp} weights={len(weights)}")
                continue
            entries, padded = write_wide_weights(out_path, weights, oc, k_total, mp)
            print(f"[ok]   {module_id}: OC={oc} K_TOTAL={k_total} MP={mp} entries={entries} padded_zeros={padded}")
            repacked += 1
        except Exception as e:
            print(f"[fail] {module_id}: {e}", file=sys.stderr)
            failed += 1

    print(f"\n[summary] repacked={repacked} skipped={skipped} failed={failed}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, help="single-file mode input hex")
    p.add_argument("--output", type=Path, help="single-file mode output hex")
    p.add_argument("--oc", type=int, help="single-file mode OC")
    p.add_argument("--k-total", type=int, help="single-file mode K_TOTAL")
    p.add_argument("--mp", type=int, default=8, help="MP parameter (default 8)")
    p.add_argument("--mp-k", type=int, default=1, help="MP_K kernel parallelism (default 1)")
    p.add_argument("--output-suffix", default="_weights_wide.hex", help="output suffix (e.g. _weights_mp_k_9.hex)")
    p.add_argument("--batch", action="store_true", help="batch mode: process all weights from sidecar dir")
    p.add_argument("--sidecar-dir", type=Path, default=Path("output/tb"))
    p.add_argument("--weights-dir", type=Path, default=Path("output/weights"))
    p.add_argument("--rtl-dir", type=Path, default=Path("output/rtl"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.batch:
        batch(args.sidecar_dir.resolve(), args.weights_dir.resolve(), args.rtl_dir.resolve(),
              args.mp, args.dry_run)
        return

    if not (args.input and args.output and args.oc is not None and args.k_total is not None):
        p.error("single-file mode requires --input, --output, --oc, --k-total")

    weights = read_flat_weights(args.input)
    if args.dry_run:
        print(f"[dry-run] would repack {args.input} -> {args.output}, "
              f"OC={args.oc} K_TOTAL={args.k_total} MP={args.mp} MP_K={args.mp_k}, weights={len(weights)}")
        return
    entries, padded = write_wide_weights(args.output, weights, args.oc, args.k_total, args.mp, args.mp_k)
    print(f"[ok] wrote {entries} entries to {args.output} (padded_zeros={padded})")


if __name__ == "__main__":
    main()
