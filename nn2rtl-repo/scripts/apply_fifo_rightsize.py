#!/usr/bin/env python3
"""Right-size every skip_fifo in nn2rtl_top.v to its EMPIRICALLY-MEASURED peak
occupancy (Phase-2 BRAM reclamation, Step 2).

Reads the [fifo-peak] high-water marks from the e2e log (A15), sets each
instance's DEPTH = smallest power-of-2 STRICTLY GREATER than its peak. Strictly
greater (not >=) guarantees the FIFO NEVER asserts `full` (occupancy max = peak <
depth for every FIFO), so in_ready stays high exactly as it did at DEPTH=8192 ->
bit-identical dataflow -> SAME cycle count, no deadlock. (If depth==peak when peak
is a power of 2, the FIFO could go full at peak and perturb timing.)

Patches the LIVE nn2rtl_top.v in place (does NOT regenerate from
build_top_wrapper.ts, which would wipe this session's hand/script fixes).

Verification: re-run e2e; if cycles==13,348,787 and beats==3136, no FIFO hit its
new limit (dynamics preserved). Any deviation => a FIFO was undersized => bump it.

Usage: python scripts/apply_fifo_rightsize.py [--log <e2e_log>] [--dry-run]
"""
from __future__ import annotations
import argparse, math, re
from pathlib import Path

TOP = Path("output/rtl/nn2rtl_top.v")
DEFAULT_LOG = Path("output/reports_integrated/verilator_nn2rtl_top/e2e_phaseA15.log")


def pow2_gt(n: int) -> int:
    """Smallest power of 2 strictly greater than n (min 2)."""
    p = 2
    while p <= n:
        p *= 2
    return p


def bram36(depth: int, w: int = 256) -> int:
    if depth <= 64:
        return 0  # LUTRAM
    return 4 * math.ceil(depth / 512)  # 256-bit width -> 4 cols of 512-deep BRAM36


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    peaks: dict[str, int] = {}
    for line in args.log.read_text(errors="ignore").splitlines():
        m = re.search(r"\.(u_\w+) DEPTH=(\d+) peak=(\d+)", line)
        if m and "fifo-peak" in line:
            peaks[m.group(1)] = int(m.group(3))
    if not peaks:
        raise SystemExit(f"no [fifo-peak] lines in {args.log}")
    print(f"[info] parsed {len(peaks)} FIFO peaks")

    txt = TOP.read_text()
    old_total = new_total = 0
    patched = 0
    changes = []

    # ...#(.WIDTH(W), .DEPTH(<d>)) u_NAME   ->  .DEPTH(d) then )) closes #(...)
    pat = re.compile(r"(\.DEPTH\()(\d+)(\)\)\s*)(u_\w+)")
    def repl2(m: re.Match) -> str:
        nonlocal old_total, new_total, patched
        old_depth, name = int(m.group(2)), m.group(4)
        if name not in peaks:
            return m.group(0)  # not an audited skip_fifo
        nd = pow2_gt(peaks[name])
        old_total += bram36(old_depth); new_total += bram36(nd); patched += 1
        changes.append((name, peaks[name], old_depth, nd))
        return f"{m.group(1)}{nd}{m.group(3)}{m.group(4)}"

    txt2 = pat.sub(repl2, txt)
    print(f"[info] patched {patched} skip_fifo instances")
    print(f"[BRAM36] FIFOs {old_total} -> {new_total}  (reclaim {old_total-new_total}, {100*(old_total-new_total)/old_total:.0f}%)")
    # show the ones that stay large
    for name, pk, od, nd in sorted(changes, key=lambda c: -c[3]):
        if nd >= 1024:
            print(f"   {name}: peak={pk} {od}->{nd} ({bram36(nd)} BRAM36)")
    if args.dry_run:
        print("[dry-run] no file written")
        return
    if patched != len(peaks):
        raise SystemExit(f"[FAIL] patched {patched} but expected {len(peaks)} — aborting (no write)")
    TOP.write_text(txt2)
    print("[written] nn2rtl_top.v")


if __name__ == "__main__":
    main()
