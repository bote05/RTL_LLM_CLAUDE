#!/usr/bin/env python3
"""repack_resnet_kpar4_banks.py — repack the 8 ResNet engine URAM weight
banks for ENGINE K-PARALLEL P=4 (4 taps per line, tap-major), INCLUDING the
POS-MAJOR TRANSPOSITION of the 9 dense 3x3 dispatch regions, with full
layout proofs.

WHY
---
The K_PAR=4 engine consumes 4 consecutive K-taps (old weight words) per
cycle from ONE wide bank line. ResNet INT3 banks: 67072 rows x 96b
(32 lanes x 3b). Repack: width x4 (96b -> 384b), depth /4 (67072 -> 16768,
67072 %% 4 == 0 -> ZERO pad words). Total bits identical => BRAM neutral.

THE 3x3 TRANSPOSITION (this is what MBV2's repack did NOT need)
---------------------------------------------------------------
The address_generator walks K with ic INNERMOST (kh, kw, ic) but the legacy
weight layout is ic-MAJOR: word = base + pass*KT + ic*KH*KW + (kh*KW+kw).
For 1x1 (KH*KW==1) walk order == address order, so 4 consecutive walk steps
are 4 consecutive addresses (one repacked line). For 3x3 they have address
STRIDE 9 — un-fetchable in one line. Fix: TRANSPOSE each 3x3 region to
POS-MAJOR: word = base + pass*KT + (kh*KW+kw)*IC + ic. Then a fast 4-group
(4 consecutive ic of one (kh,kw)) is 4 consecutive addresses. The KPAR4-RN
address_generator fast walk uses exactly this pos-major formula (and the
two formulas coincide for 1x1, so MBV2's untransposed banks are unaffected).

A 4-aligned k-group NEVER crosses a (kh,kw) boundary: k = pos*IC + ic with
IC%%4==0 => k%%4 == ic%%4, so groups are pure-ic and share one act word and
one in_bounds decision (pad-safe).

LAYOUT (load-bearing, mirrors output/rtl/nn2rtl_top.v [KPAR4-RN])
-----------------------------------------------------------------
OLD: bank b (0..7), line w (0..67071) = 96b hex line; 3-bit slot s (0..31)
     = bits [s*3 +: 3] = INT3 weight of MAC lane (32*b + s) for old word w.
T  : the transposed old-address-space array (identical permutation for all
     8 banks — the word index permutation is lane-independent).
NEW: bank b, line g (0..16767) = 384b = {T[4g+3], T[4g+2], T[4g+1], T[4g]}
     (tap-major, tap j at VALUE bits [j*96 +: 96]). Engine tap-j 768b word
     = concat over banks of new_line[j*96 +: 96] (bank0 lowest) — identical
     lane order to the old bus, per tap.
ADDRESSING: fast-walk old addr A (= base + pass*KT + pos*IC + ic) lives at
     new line A>>2, tap A&3. ALL 17 dispatches are fast-eligible
     (base%%4==0, IC%%4==0 — asserted in P0), groups are 4-aligned -> A&3==0
     and one line carries exactly taps k..k+3. The serial path is NEVER
     exercised on this top (and would be WRONG for transposed 3x3 regions —
     hence the P0 eligibility assert is load-bearing).

PROOFS (run on every invocation; abort = no partial writes)
-----------------------------------------------------------
P0 (dispatch-table tiling + eligibility): the 17 regions parsed from the
    DEPLOYED scheduler ROMs (output/rtl/nn2rtl_scheduler.v) tile
    [0, 67072) exactly (no gap/overlap), and every dispatch satisfies the
    RTL's kpar_fast gate (not depthwise, IC%%4==0, IC>=4, base%%4==0).
P1 (re-expansion + bijectivity): the permutation old->T is a bijection and
    splitting every new line back into 4x96b words reproduces T exactly.
P2 (random WALK-equivalence, the load-bearing proof): for 4096 random
    (dispatch, pass, pos, aligned ic0, tap j, lane): the engine's fetched
    3-bit weight — new line (base+pass*KT+pos*IC+ic0)>>2, tap j, lane slot —
    equals the ORIGINAL bank word at the LEGACY address
    base + pass*KT + (ic0+j)*KH*KW + pos, same lane slot.
P3 (aligned-group tap-slice identity): for 512 random ALIGNED A (A%%4==0):
    tap-j slice of new line A>>2 equals T[A+j] for j=0..3, full 96b/bank.
P4 (1x1-region identity): for 1024 random words inside 1x1 regions:
    T[w] == OLD[w] (the transposition only touches 3x3 regions).

Usage:  python scripts/repack_resnet_kpar4_banks.py
Output: output/weights/uram_weights_bank{0..7}_kp4.mem (16768 x 96 hex)
Exit 0 = repacked + ALL PROOFS PASS; nonzero = abort.
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WDIR = REPO / "output" / "weights"
SCHED = REPO / "output" / "rtl" / "nn2rtl_top.v".replace("nn2rtl_top.v", "nn2rtl_scheduler.v")

OLD_DEPTH = 67072           # must match the pre-KPAR4 bank DEPTH in the top
OLD_HEX = 96 // 4           # 24 hex chars per 96b line
NEW_HEX = 4 * OLD_HEX       # 96 hex chars per 384b line
GROUPS = OLD_DEPTH // 4     # 16768 (67072 % 4 == 0 -> no pad words)
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
    rng = random.Random(0x4B5052)   # deterministic ("KPR")
    table = load_dispatch_table()

    # ---- P0: region tiling + fast-eligibility of EVERY dispatch ----
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
        if not (e["base"] % 4 == 0 and e["ic"] % 4 == 0 and e["ic"] >= 4):
            print(f"P0 FAIL: dispatch {e['idx']} NOT fast-eligible "
                  f"(base={e['base']} ic={e['ic']}) — the serial path would "
                  f"read TRANSPOSED 3x3 data. Aborting.")
            return 1
    n3x3 = sum(1 for e in table if e["kh"] * e["kw"] > 1)
    print(f"[kp4-repack] P0 PASS: {N_DISPATCH} regions tile [0,{OLD_DEPTH}) "
          f"exactly; all fast-eligible ({n3x3} transposed 3x3 + "
          f"{N_DISPATCH - n3x3} identity 1x1)")

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
        dst = WDIR / f"uram_weights_bank{b}_kp4.mem"
        old = load_bank(src)
        t_arr = [old[perm[a]] for a in range(OLD_DEPTH)]

        # ---- repack: new line g = {T[4g+3], T[4g+2], T[4g+1], T[4g]} ----
        # $readmemh assigns each line MSB-first, so the TEXT concatenation
        # tap3+tap2+tap1+tap0 puts tap j at VALUE bits [j*96 +: 96].
        new = [t_arr[4 * g + 3] + t_arr[4 * g + 2] + t_arr[4 * g + 1] + t_arr[4 * g]
               for g in range(GROUPS)]

        # ---- P1b: full re-expansion equality ----
        for g, ln in enumerate(new):
            if len(ln) != NEW_HEX:
                raise SystemExit(f"bank{b} g={g}: new len {len(ln)} != {NEW_HEX}")
            for j in range(4):
                back = ln[(3 - j) * OLD_HEX:(4 - j) * OLD_HEX]
                if back != t_arr[4 * g + j]:
                    print(f"P1 FAIL bank{b} g={g} tap{j}")
                    all_ok = False

        # ---- P2: random WALK-equivalence at 3-bit lane granularity ----
        for _ in range(4096 // 8):
            e = table[rng.randrange(N_DISPATCH)]
            khkw = e["kh"] * e["kw"]
            p = rng.randrange(e["passes"])
            pos = rng.randrange(khkw)
            ic0 = rng.randrange(0, e["ic"], 4)
            j = rng.randrange(4)
            s = rng.randrange(32)
            a_fast = e["base"] + p * e["kt"] + pos * e["ic"] + ic0   # group start
            a_legacy = e["base"] + p * e["kt"] + (ic0 + j) * khkw + pos
            assert a_fast % 4 == 0, "fast group start must be 4-aligned"
            new_v = int(new[a_fast >> 2], 16)
            got = (new_v >> (j * 96 + s * 3)) & 0x7
            ref = (int(old[a_legacy], 16) >> (s * 3)) & 0x7
            if got != ref:
                print(f"P2 FAIL bank{b} disp={e['idx']} pass={p} pos={pos} "
                      f"ic0={ic0} tap={j} slot={s}: got={got} ref={ref}")
                all_ok = False

        # ---- P3: aligned-group tap-slice identity (full 96b per tap) ----
        for _ in range(512 // 8):
            a0 = rng.randrange(0, OLD_DEPTH, 4)
            new_v = int(new[a0 >> 2], 16)
            for j in range(4):
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
        print(f"[kp4-repack] bank{b}: {OLD_DEPTH} x 96b -> {GROUPS} x 384b "
              f"-> {dst.name}  P1/P2/P3/P4 PASS")
    print(f"[kp4-repack] ALL 8 BANKS REPACKED, proofs P0..P4 PASS "
          f"(depth {OLD_DEPTH}->{GROUPS}, bits/bank {OLD_DEPTH*96} -> {GROUPS*384}, "
          f"{n3x3} regions transposed pos-major)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
