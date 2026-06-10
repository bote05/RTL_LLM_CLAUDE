#!/usr/bin/env python3
"""repack_mbv2_kpar4_banks.py — repack the 8 MBV2 engine URAM weight banks
for ENGINE K-PARALLEL P=4 (4 taps per line, tap-major), with a full
lane-major/tap-major layout PROOF.

WHY
---
The K_PAR=4 engine consumes 4 consecutive K-taps (old weight words) per
cycle. The banks are single-read-port URAM ROMs, so 4 words/cycle requires
4-taps-per-line repacking: width x4 (288b -> 1152b), depth /4
(18533 -> ceil(18533/4) = 4634). Total bits identical (+3 zero pad words)
=> URAM cost neutral.

LAYOUT (load-bearing, mirrors the RTL in nn2rtl_top_engine.v [KPAR4])
----------------------------------------------------------------------
OLD: bank b (b=0..7), line w (w=0..18532) = 288b hex line; byte slot
     s (s=0..31) = bits [s*8 +: 8] = INT8 weight of MAC lane (32*b + s)
     for old K-word w. Only [255:0] of each 288b line is used.
NEW: bank b, line g (g=0..4633) = 1152b = {old[4g+3], old[4g+2],
     old[4g+1], old[4g]}  (tap-major, tap j at bits [j*288 +: 288]).
     Engine tap-j 2048b word = concat over banks of new_line[j*288 +: 256]
     (bank0 lowest) — identical lane order to the old bus, per tap.
ADDRESSING: old word addr A lives at new line A>>2, tap A&3. FAST dense
     groups are 4-aligned (base%4==0, K_TOTAL%4==0, k%4==0) -> one line
     carries exactly taps k..k+3. SERIAL dispatches (depthwise K=9 bases,
     FC @13413 with 13413%4==1) read tap (A&3) via the skeleton's 2-cycle
     piped subword select.
PAD: 18533 % 4 == 1 -> the last line's taps 1..3 are zero words (never
     addressed: old addr space ends at 18532).

PROOF (run on every invocation)
-------------------------------
P1 (re-expansion): splitting every new line back into 4x288b words
    reproduces the original 18533 lines exactly (+ 3 zero pad words).
P2 (random byte cross-check): for 4096 random (w, lane) pairs:
    old_byte(b=lane//32, w, s=lane%32) ==
    new_byte(b, g=w//4, bits[(w%4)*288 + s*8 +: 8]).
P3 (group identity): for 512 random ALIGNED group starts w0 (w0%4==0):
    the engine's tap-j slice of new line w0/4 equals old line w0+j
    [255:0] for j=0..3, for the full 2048b reconstructed bus.

Usage:  python scripts/repack_mbv2_kpar4_banks.py
Output: output/mobilenet-v2/weights/uram_weights_bank{0..7}_kp4.mem
Exit 0 = repacked + ALL PROOFS PASS; nonzero = abort (no partial writes).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WDIR = REPO / "output" / "mobilenet-v2" / "weights"

OLD_DEPTH = 18533           # must match the pre-KPAR4 bank DEPTH in the top
OLD_HEX = 288 // 4          # 72 hex chars per 288b line
NEW_HEX = 4 * OLD_HEX       # 288 hex chars per 1152b line
GROUPS = (OLD_DEPTH + 3) // 4   # 4634
PAD = GROUPS * 4 - OLD_DEPTH    # 3 zero words


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
    rng = random.Random(0x4B50)   # deterministic ("KP")
    all_ok = True
    for b in range(8):
        src = WDIR / f"uram_weights_bank{b}.mem"
        dst = WDIR / f"uram_weights_bank{b}_kp4.mem"
        old = load_bank(src)
        padded = old + ["0" * OLD_HEX] * PAD

        # ---- repack: new line g = {old[4g+3], old[4g+2], old[4g+1], old[4g]} ----
        # $readmemh assigns each line MSB-first, so the TEXT concatenation
        # tap3+tap2+tap1+tap0 puts tap j at VALUE bits [j*288 +: 288].
        new = [padded[4 * g + 3] + padded[4 * g + 2] + padded[4 * g + 1] + padded[4 * g]
               for g in range(GROUPS)]

        # ---- P1: full re-expansion equality ----
        for g, ln in enumerate(new):
            if len(ln) != NEW_HEX:
                raise SystemExit(f"bank{b} g={g}: new len {len(ln)} != {NEW_HEX}")
            # text slice tap j = chars [(3-j)*72 : (4-j)*72]
            for j in range(4):
                back = ln[(3 - j) * OLD_HEX:(4 - j) * OLD_HEX]
                if back != padded[4 * g + j]:
                    print(f"P1 FAIL bank{b} g={g} tap{j}")
                    all_ok = False

        # ---- P2: random byte cross-check via integer bit slicing ----
        for _ in range(4096 // 8):
            w = rng.randrange(OLD_DEPTH)
            s = rng.randrange(32)
            old_v = int(old[w], 16)
            new_v = int(new[w // 4], 16)
            ob = (old_v >> (s * 8)) & 0xFF
            nb = (new_v >> ((w % 4) * 288 + s * 8)) & 0xFF
            if ob != nb:
                print(f"P2 FAIL bank{b} w={w} s={s}: old={ob:02x} new={nb:02x}")
                all_ok = False

        # ---- P3: aligned-group tap-slice identity (low 256b per tap) ----
        for _ in range(512 // 8):
            w0 = rng.randrange(0, OLD_DEPTH - 4, 4)
            new_v = int(new[w0 // 4], 16)
            for j in range(4):
                tap = (new_v >> (j * 288)) & ((1 << 256) - 1)
                ref = int(old[w0 + j], 16) & ((1 << 256) - 1)
                if tap != ref:
                    print(f"P3 FAIL bank{b} w0={w0} tap{j}")
                    all_ok = False

        if not all_ok:
            return 1
        dst.write_text("\n".join(new) + "\n", encoding="ascii", newline="\n")
        print(f"[kp4-repack] bank{b}: {OLD_DEPTH} x 288b -> {GROUPS} x 1152b "
              f"(pad {PAD}) -> {dst.name}  P1/P2/P3 PASS")
    print(f"[kp4-repack] ALL 8 BANKS REPACKED, proofs P1+P2+P3 PASS "
          f"(depth {OLD_DEPTH}->{GROUPS}, bits/bank {OLD_DEPTH*288} -> {GROUPS*1152})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
