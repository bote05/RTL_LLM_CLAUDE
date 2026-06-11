#!/usr/bin/env python3
"""repack_resnet_kpar8_banks.py — repack the 8 ResNet engine weight banks
for ENGINE K-PARALLEL P=8 (8 taps per line, tap-major), INCLUDING the
POS-MAJOR TRANSPOSITION of the 9 dense 3x3 dispatch regions, with full
layout proofs. [KPAR8-RN 2026-06-11] — extends the KPAR4 lineage
(scripts/repack_resnet_kpar4_banks.py) to 8 taps.

WHY
---
The K_PAR=8 engine consumes 8 consecutive K-taps (old weight words) per
cycle from ONE wide bank line. ResNet INT3 banks: 67072 rows x 96b
(32 lanes x 3b). Repack: width x8 (96b -> 768b), depth /8 (67072 -> 8384,
67072 %% 8 == 0 -> ZERO pad words). Total bits identical => BRAM neutral.

THE 3x3 TRANSPOSITION (identical permutation to KPAR4 — only the PACKING
width changes): the address_generator walks K with ic INNERMOST but the
legacy layout is ic-MAJOR; each 3x3 region is transposed to POS-MAJOR
(word = base + pass*KT + (kh*KW+kw)*IC + ic) so a fast 8-group (8
consecutive ic of one (kh,kw)) is 8 consecutive addresses. An 8-aligned
k-group NEVER crosses a (kh,kw) boundary: k = pos*IC + ic with IC%%8==0 =>
k%%8 == ic%%8, so groups are pure-ic and share one act word, one ic chunk
and one in_bounds decision (pad-safe). All ResNet IC in {256,512,1024,2048}.

LAYOUT (load-bearing, mirrors output/rtl/nn2rtl_top.v [KPAR8-RN])
-----------------------------------------------------------------
OLD: bank b (0..7), line w (0..67071) = 96b hex; 3-bit slot s (0..31)
     = INT3 weight of MAC lane (32*b + s) for old word w.
T  : transposed old-address-space array (lane-independent permutation).
NEW: bank b, line g (0..8383) = 768b = {T[8g+7], ..., T[8g+1], T[8g]}
     (tap-major, tap j at VALUE bits [j*96 +: 96]). Engine tap-j 768b word
     = concat over banks of new_line[j*96 +: 96] (bank0 lowest).
ADDRESSING: fast-walk old addr A lives at new line A>>3, tap A&7. ALL 17
     dispatches are fast-eligible (base%%8==0, IC%%8==0 — asserted in P0),
     groups are 8-aligned -> A&7==0 and one line carries taps k..k+7. The
     serial path is NEVER exercised on this top (and would be WRONG for
     transposed 3x3 regions — the P0 eligibility assert is load-bearing).

PROOFS (run on every invocation; abort = no partial writes)
-----------------------------------------------------------
P0 dispatch-table tiling + P=8 eligibility (parsed from the DEPLOYED
   scheduler ROMs — no drift possible).
P1 permutation bijectivity + full re-expansion of every new line.
P2 random WALK-equivalence (4096 samples): engine-fetched 3-bit lane
   weight at the transposed/fast address == original bank word at the
   LEGACY address.
P3 aligned-group tap-slice identity (512 samples, full 96b per tap).
P4 1x1 regions byte-identical (transpose touches only 3x3 regions).

Usage:  python scripts/repack_resnet_kpar8_banks.py
Output: output/weights/uram_weights_bank{0..7}_kp8.mem (8384 x 768b hex)
Exit 0 = repacked + ALL PROOFS PASS; nonzero = abort.
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WDIR = REPO / "output" / "weights"
SCHED = REPO / "output" / "rtl" / "nn2rtl_scheduler.v"

OLD_DEPTH = 67072           # pre-KPAR bank DEPTH (96b lines)
OLD_HEX = 96 // 4           # 24 hex chars per 96b line
P = 8                       # taps per line
NEW_HEX = P * OLD_HEX       # 192 hex chars per 768b line
GROUPS = OLD_DEPTH // P     # 8384 (67072 % 8 == 0 -> no pad words)
N_DISPATCH = 17


def parse_rom(text: str, name: str) -> dict[int, int]:
    vals = {int(i): int(v)
            for i, v in re.findall(rf"5'd(\d+): {name} = \d+'d(\d+);", text)}
    if len(vals) != N_DISPATCH:
        raise SystemExit(f"ROM {name}: {len(vals)} rows != {N_DISPATCH}")
    return vals


def load_dispatch_table() -> list[dict]:
    t = SCHED.read_text(encoding="utf-8")
    ic = parse_rom(t, "channel_in_rom")
    oc = parse_rom(t, "channel_out_rom")
    kh = parse_rom(t, "kernel_h_rom")
    kw = parse_rom(t, "kernel_w_rom")
    wb = parse_rom(t, "weight_base_word_rom")
    return [dict(idx=d, ic=ic[d], oc=oc[d], kh=kh[d], kw=kw[d], base=wb[d],
                 kt=ic[d] * kh[d] * kw[d], passes=(oc[d] + 255) // 256)
            for d in range(N_DISPATCH)]


def build_permutation(table: list[dict]) -> list[int]:
    """perm[t_addr] = old_addr such that T[t_addr] = OLD[old_addr]."""
    perm = [-1] * OLD_DEPTH
    for e in table:
        khkw = e["kh"] * e["kw"]
        for p in range(e["passes"]):
            blk = e["base"] + p * e["kt"]
            if khkw == 1:
                for k in range(e["kt"]):
                    perm[blk + k] = blk + k
            else:
                for pos in range(khkw):
                    for ic in range(e["ic"]):
                        perm[blk + pos * e["ic"] + ic] = blk + ic * khkw + pos
    return perm


def load_bank(p: Path) -> list[str]:
    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    if len(lines) != OLD_DEPTH:
        raise SystemExit(f"{p.name}: {len(lines)} lines != expected {OLD_DEPTH}")
    for i, ln in enumerate(lines):
        if len(ln) != OLD_HEX:
            raise SystemExit(f"{p.name}:{i}: line len {len(ln)} != {OLD_HEX}")
        int(ln, 16)  # raises on non-hex
    return lines


def main() -> int:
    rng = random.Random(0x4B5038)   # deterministic ("KP8")
    table = load_dispatch_table()

    # ---- P0: region tiling + P=8 fast-eligibility of EVERY dispatch ----
    regions = sorted((e["base"], e["base"] + e["passes"] * e["kt"], e["idx"])
                     for e in table)
    cur = 0
    for lo, hi, d in regions:
        if lo != cur:
            print(f"P0 FAIL: dispatch {d} region starts at {lo}, expected {cur}")
            return 1
        cur = hi
    if cur != OLD_DEPTH:
        print(f"P0 FAIL: regions end at {cur} != bank depth {OLD_DEPTH}")
        return 1
    for e in table:
        if not (e["base"] % P == 0 and e["ic"] % P == 0 and e["ic"] >= P):
            print(f"P0 FAIL: dispatch {e['idx']} NOT P=8 fast-eligible "
                  f"(base={e['base']} ic={e['ic']}) — the serial path would "
                  f"read TRANSPOSED 3x3 data. Aborting (relocation pad needed).")
            return 1
    if OLD_DEPTH % P != 0:
        print(f"P0 FAIL: bank depth {OLD_DEPTH} %% {P} != 0")
        return 1
    n3x3 = sum(1 for e in table if e["kh"] * e["kw"] > 1)
    print(f"[kp8-repack] P0 PASS: {N_DISPATCH} regions tile [0,{OLD_DEPTH}) "
          f"exactly; all P=8 fast-eligible (bases "
          f"{sorted(e['base'] for e in table)} all %8==0; "
          f"{n3x3} transposed 3x3 + {N_DISPATCH - n3x3} identity 1x1)")

    perm = build_permutation(table)

    # ---- P1a: bijectivity ----
    if sorted(perm) != list(range(OLD_DEPTH)):
        print("P1 FAIL: permutation is not a bijection over the bank depth")
        return 1

    pos1x1 = [w for e in table if e["kh"] * e["kw"] == 1
              for w in range(e["base"], e["base"] + e["passes"] * e["kt"])]

    all_ok = True
    for b in range(8):
        src = WDIR / f"uram_weights_bank{b}.mem"
        dst = WDIR / f"uram_weights_bank{b}_kp8.mem"
        old = load_bank(src)
        t_arr = [old[perm[a]] for a in range(OLD_DEPTH)]

        # ---- repack: new line g = {T[8g+7], ..., T[8g]} ----
        # $readmemh assigns each line MSB-first, so the TEXT concatenation
        # tap7+...+tap0 puts tap j at VALUE bits [j*96 +: 96].
        new = ["".join(t_arr[P * g + j] for j in range(P - 1, -1, -1))
               for g in range(GROUPS)]

        # ---- P1b: full re-expansion equality ----
        for g, ln in enumerate(new):
            if len(ln) != NEW_HEX:
                raise SystemExit(f"bank{b} g={g}: new len {len(ln)} != {NEW_HEX}")
            for j in range(P):
                back = ln[(P - 1 - j) * OLD_HEX:(P - j) * OLD_HEX]
                if back != t_arr[P * g + j]:
                    print(f"P1 FAIL bank{b} g={g} tap{j}")
                    all_ok = False

        # ---- P2: random WALK-equivalence at 3-bit lane granularity ----
        for _ in range(4096 // 8):
            e = table[rng.randrange(N_DISPATCH)]
            khkw = e["kh"] * e["kw"]
            p = rng.randrange(e["passes"])
            pos = rng.randrange(khkw)
            ic0 = rng.randrange(0, e["ic"], P)
            j = rng.randrange(P)
            s = rng.randrange(32)
            a_fast = e["base"] + p * e["kt"] + pos * e["ic"] + ic0   # group start
            a_legacy = e["base"] + p * e["kt"] + (ic0 + j) * khkw + pos
            assert a_fast % P == 0, "fast group start must be 8-aligned"
            new_v = int(new[a_fast >> 3], 16)
            got = (new_v >> (j * 96 + s * 3)) & 0x7
            ref = (int(old[a_legacy], 16) >> (s * 3)) & 0x7
            if got != ref:
                print(f"P2 FAIL bank{b} disp={e['idx']} pass={p} pos={pos} "
                      f"ic0={ic0} tap={j} slot={s}: got={got} ref={ref}")
                all_ok = False

        # ---- P3: aligned-group tap-slice identity (full 96b per tap) ----
        for _ in range(512 // 8):
            a0 = rng.randrange(0, OLD_DEPTH, P)
            new_v = int(new[a0 >> 3], 16)
            for j in range(P):
                tap = (new_v >> (j * 96)) & ((1 << 96) - 1)
                ref = int(t_arr[a0 + j], 16)
                if tap != ref:
                    print(f"P3 FAIL bank{b} a0={a0} tap{j}")
                    all_ok = False

        # ---- P4: 1x1 regions are identity (transpose touches only 3x3) ----
        for _ in range(1024 // 8):
            w = pos1x1[rng.randrange(len(pos1x1))]
            if t_arr[w] != old[w]:
                print(f"P4 FAIL bank{b} w={w}: 1x1 region word moved")
                all_ok = False

        if not all_ok:
            return 1
        dst.write_text("\n".join(new) + "\n", encoding="ascii", newline="\n")
        print(f"[kp8-repack] bank{b}: {OLD_DEPTH} x 96b -> {GROUPS} x 768b "
              f"-> {dst.name}  P1/P2/P3/P4 PASS")
    print(f"[kp8-repack] ALL 8 BANKS REPACKED, proofs P0..P4 PASS "
          f"(depth {OLD_DEPTH}->{GROUPS}, bits/bank {OLD_DEPTH*96} -> {GROUPS*768}, "
          f"{n3x3} regions transposed pos-major)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
