#!/usr/bin/env python3
"""Switch spatial conv modules from conv_datapath_parallel to conv_datapath_mp_k.

For each target module with KH*KW divisible by mp_k:
  - Rewrite `conv_datapath_parallel #(` -> `conv_datapath_mp_k #(`
  - Add `.MP_K(<mp_k>),` parameter override before `.MP(`
  - Rewrite WEIGHTS_PATH suffix `_weights_wide.hex` -> `_weights_mp_k_<N>.hex`

Default MP_K per kernel shape:
  3x3 -> MP_K=9 (full kernel-parallel)
  7x7 -> MP_K=7 (one row per cycle)

Run scripts/repack_weights_wide.py with --mp-k first to produce the
*_weights_mp_k_<N>.hex files for the modules being patched.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


KH_RE = re.compile(r"localparam\s+integer\s+KH\s*=\s*(\d+)")
KW_RE = re.compile(r"localparam\s+integer\s+KW\s*=\s*(\d+)")
OC_RE = re.compile(r"localparam\s+integer\s+OC\s*=\s*(\d+)")
IC_RE = re.compile(r"localparam\s+integer\s+IC\s*=\s*(\d+)")
MP_RE = re.compile(r"localparam\s+integer\s+MP\s*=\s*(\d+)")


def derive_shape(txt: str) -> tuple[int, int, int, int, int] | None:
    """Return (IC, OC, KH, KW, MP) from localparam declarations."""
    parts = {}
    for name, regex in (("IC", IC_RE), ("OC", OC_RE), ("KH", KH_RE), ("KW", KW_RE), ("MP", MP_RE)):
        m = regex.search(txt)
        if not m:
            return None
        parts[name] = int(m.group(1))
    return parts["IC"], parts["OC"], parts["KH"], parts["KW"], parts["MP"]


def patch_one(path: Path, mp_k: int, dry_run: bool) -> tuple[bool, str]:
    """Patch one node_conv_*.v to use conv_datapath_mp_k. Returns (changed, msg)."""
    txt = path.read_text()
    if "conv_datapath_mp_k" in txt:
        return (False, "already-mp_k")
    if "conv_datapath_parallel" not in txt:
        return (False, "no-parallel-instance")

    new = re.sub(r"\bconv_datapath_parallel\s*#\(", "conv_datapath_mp_k #(", txt)
    new = re.sub(
        r"(\.K_TOTAL\(K_TOTAL\),\s*\.MP\(MP\),)",
        rf"\1\n        .MP_K({mp_k}),",
        new,
        count=1,
    )
    new = re.sub(r"_weights_wide\.hex\"", f'_weights_mp_k_{mp_k}.hex"', new)

    if new == txt:
        return (False, "no-change")
    if dry_run:
        return (True, "would-patch")
    path.write_text(new)
    return (True, "patched")


def repack_for_module(path: Path, mp_k: int, weights_dir: Path, dry_run: bool) -> tuple[bool, str]:
    """Invoke repack_weights_wide.py for one module with --mp-k=<mp_k>."""
    txt = path.read_text()
    shape = derive_shape(txt)
    if shape is None:
        return (False, "no-shape")
    ic, oc, kh, kw, mp = shape
    k_total = ic * kh * kw
    if k_total % mp_k != 0:
        return (False, f"k_total={k_total} not divisible by mp_k={mp_k}")
    in_path = weights_dir / f"{path.stem}_weights.hex"
    out_path = weights_dir / f"{path.stem}_weights_mp_k_{mp_k}.hex"
    if not in_path.exists():
        return (False, f"input-missing:{in_path.name}")
    # Always regenerate — stale files with the wrong MP have bitten us
    # (conv_196 has MP=8 but a manual MP=4 test file shadowed it).

    cmd = [
        sys.executable,
        "scripts/repack_weights_wide.py",
        "--input", str(in_path),
        "--output", str(out_path),
        "--oc", str(oc),
        "--k-total", str(k_total),
        "--mp", str(mp),
        "--mp-k", str(mp_k),
    ]
    if dry_run:
        return (True, f"would-repack ({' '.join(cmd[2:])})")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return (False, f"repack-failed:{res.stderr.strip()}")
    return (True, "repacked")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rtl-dir", type=Path, default=Path("output/rtl"))
    p.add_argument("--weights-dir", type=Path, default=Path("output/weights"))
    p.add_argument("--only", help="comma-separated module IDs")
    p.add_argument("--skip", help="comma-separated module IDs to skip")
    p.add_argument("--mp-k-3x3", type=int, default=9)
    p.add_argument("--mp-k-7x7", type=int, default=7)
    p.add_argument("--repack-only", action="store_true", help="only generate weights, don't patch RTL")
    p.add_argument("--patch-only", action="store_true", help="only patch RTL, assume weights exist")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    only = set(args.only.split(",")) if args.only else None
    skip = set(args.skip.split(",")) if args.skip else set()

    rtl_dir = args.rtl_dir.resolve()
    weights_dir = args.weights_dir.resolve()
    patched = 0
    repacked = 0
    skipped = 0
    failed = 0
    for path in sorted(rtl_dir.glob("node_conv_*.v")):
        if path.name.endswith(".preimprove"):
            continue
        mid = path.stem
        if only is not None and mid not in only:
            continue
        if mid in skip:
            print(f"[skip-explicit] {mid}")
            skipped += 1
            continue

        txt = path.read_text()
        shape = derive_shape(txt)
        if shape is None:
            print(f"[skip] {mid}: no-shape")
            skipped += 1
            continue
        ic, oc, kh, kw, mp = shape
        if kh == 3 and kw == 3:
            mp_k = args.mp_k_3x3
        elif kh == 7 and kw == 7:
            mp_k = args.mp_k_7x7
        else:
            print(f"[skip] {mid}: KH={kh} KW={kw} (not 3x3 or 7x7)")
            skipped += 1
            continue

        if not args.patch_only:
            ok, msg = repack_for_module(path, mp_k, weights_dir, args.dry_run)
            print(f"[{'repack-dry' if args.dry_run else 'repack'}] {mid} mp_k={mp_k}: {msg}")
            if not ok:
                failed += 1
                continue
            repacked += 1

        if not args.repack_only:
            ok, msg = patch_one(path, mp_k, args.dry_run)
            print(f"[{'patch-dry' if args.dry_run else 'patch'}] {mid} mp_k={mp_k}: {msg}")
            if ok:
                patched += 1
            elif msg == "already-mp_k":
                pass
            else:
                failed += 1

    print(f"\n[summary] repacked={repacked} patched={patched} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
