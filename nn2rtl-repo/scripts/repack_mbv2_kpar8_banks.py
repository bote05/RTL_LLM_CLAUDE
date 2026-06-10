#!/usr/bin/env python3
"""repack_mbv2_kpar8_banks.py — repack the 8 MBV2 engine URAM weight banks
for ENGINE K-PARALLEL P=8 (8 taps per line, tap-major), WITH the [FC-PAD]
relocation (FC base 13413 -> 13416 so node_linear joins the fast walk), and
a full layout PROOF (P0..P4).

WHY
---
The K_PAR=8 engine consumes 8 consecutive K-taps (old weight words) per
cycle. The banks are single-read-port URAM ROMs, so 8 words/cycle requires
8-taps-per-line repacking: width x8 (288b -> 2304b), depth /8. Relocated
old-domain depth = 18533 + 3 FC pad = 18536 = 8 * 2317 EXACTLY (no extra
tail pad). Total bits/bank = 2317*2304 = 5,338,368 — IDENTICAL to the KPAR4
banks (4634*1152), so URAM cost is neutral vs KPAR4 (and +0.016% vs the
original 18533*288).

FC-PAD RELOCATION (load-bearing; pairs with apply_mbv2_kpar8.py)
----------------------------------------------------------------
Old image (per bank): words 0..18532; FC (node_linear, 4 oc_passes x 1280
taps) occupies 13413..18532. 13413 % 8 == 5, so at K_PAR=8 the FC would be
stuck on the SERIAL walk. RELOCATED image (depth 18536):
    reloc[w] = old[w]        for w <  13413   (dense + DW regions untouched)
    reloc[w] = ZERO          for w in 13413..13415  (3 pad words, never read)
    reloc[w] = old[w-3]      for w >= 13416   (FC region, base now %8==0)
The scheduler row 46 weight base moves 13413 -> 13416 (apply_mbv2_kpar8.py);
every other dispatch base is < 13413 and UNCHANGED.

LAYOUT (mirrors the RTL in nn2rtl_top_engine.v [KPAR8])
-------------------------------------------------------
OLD: bank b (b=0..7), line w = 288b hex line; byte slot s (s=0..31) =
     bits [s*8 +: 8] = INT8 weight of MAC lane (32*b + s) for old K-word w.
     Only [255:0] of each 288b line is used.
NEW: bank b, line g (g=0..2316) = 2304b = {reloc[8g+7], ..., reloc[8g+1],
     reloc[8g]} (tap-major, tap j at VALUE bits [j*288 +: 288]).
     Engine tap-j 2048b word = concat over banks of new_line[j*288 +: 256]
     (bank0 lowest) — identical lane order to the old bus, per tap.
ADDRESSING: relocated old word addr A lives at new line A>>3, tap A&7.
     FAST dense groups are 8-aligned (base%8==0, K_TOTAL=IC%8==0, k%8==0)
     -> one line carries exactly taps k..k+7. SERIAL dispatches (the 12
     depthwise bases, any alignment) read tap (A&7) via the skeleton's
     2-cycle piped 3-bit subword select.

PROOF (run on every invocation; abort = no partial writes)
----------------------------------------------------------
P0 (eligibility): parse the deployed scheduler weight_base_word_rom; assert
    all 34 dense pointwise bases are %8==0, and report the FC row state
    (13413 = apply script not yet run, 13416 = padded).
P1 (re-expansion): splitting every new line back into 8x288b words
    reproduces the RELOCATED image exactly, and the relocated image maps
    1:1 back onto the original 18533 lines + 3 zeros at 13413..13415.
P2 (random byte cross-check): for 4096 random (w_old, lane) pairs:
    old_byte(b=lane//32, w_old, s=lane%32) ==
    new_byte(b, g=w_new//8, bits[(w_new%8)*288 + s*8 +: 8])
    where w_new = w_old (+3 iff w_old >= 13413).
P3 (group identity): for 512 random 8-ALIGNED group starts w0 in the
    UNRELOCATED region (w0+7 < 13413): the engine's tap-j slice of new
    line w0/8 equals old line w0+j [255:0] for j=0..7.
P4 (FC fast-walk identity): for ALL 4 oc_passes and 32 sampled 8-aligned k
    offsets: tap-j slice of new line (13416+p*1280+k)/8 equals
    old[13413 + p*1280 + k + j][255:0] — i.e. the padded FC base presents
    the exact same weights the serial walk used to read.

Usage:  python scripts/repack_mbv2_kpar8_banks.py
Output: output/mobilenet-v2/weights/uram_weights_bank{0..7}_kp8.mem
Exit 0 = repacked + ALL PROOFS PASS; nonzero = abort (no partial writes).
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WDIR = REPO / "output" / "mobilenet-v2" / "weights"
SCHED = REPO / "output" / "mobilenet-v2" / "rtl" / "nn2rtl_scheduler.v"

OLD_DEPTH = 18533             # pre-KPAR4 bank depth (original 288b image)
FC_OLD_BASE = 13413           # node_linear base in the original image
FC_PAD = 3                    # zero words inserted so FC base % 8 == 0
FC_NEW_BASE = FC_OLD_BASE + FC_PAD   # 13416
RELOC_DEPTH = OLD_DEPTH + FC_PAD     # 18536
K = 8
OLD_HEX = 288 // 4            # 72 hex chars per 288b line
NEW_HEX = K * OLD_HEX         # 576 hex chars per 2304b line
GROUPS = RELOC_DEPTH // K     # 2317
assert GROUPS * K == RELOC_DEPTH, "FC pad must make the image depth %8==0"

# the 12 depthwise dispatch bases (serial walk; alignment irrelevant) — from
# scripts/gen_dw_engine_iso_cfg.py DW_TABLE / extend_mbv2_engine_maps_dw*.py
DW_BASES = {13152, 13188, 13224, 13260, 13269, 13278, 13287, 13305,
            13323, 13341, 13359, 13386}


def reloc_index(w_old: int) -> int:
    return w_old if w_old < FC_OLD_BASE else w_old + FC_PAD


def load_bank(p: Path) -> list[str]:
    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    if len(lines) != OLD_DEPTH:
        raise SystemExit(f"{p.name}: {len(lines)} lines != expected {OLD_DEPTH}")
    for i, ln in enumerate(lines):
        if len(ln) != OLD_HEX:
            raise SystemExit(f"{p.name}:{i}: line len {len(ln)} != {OLD_HEX}")
        int(ln, 16)  # raises on non-hex
    return lines


def proof_p0() -> None:
    t = SCHED.read_text(encoding="utf-8")
    rows = {int(i): int(v) for i, v in
            re.findall(r"6'd(\d+): weight_base_word_rom = 20'd(\d+);", t)}
    if len(rows) != 47:
        raise SystemExit(f"P0 FAIL: scheduler has {len(rows)} base rows != 47")
    fc = rows.pop(46)
    bad = [(d, b) for d, b in rows.items()
           if b not in DW_BASES and b % 8 != 0]
    if bad:
        raise SystemExit(f"P0 FAIL: dense bases not %8-aligned: {bad}")
    n_dense = sum(1 for b in rows.values() if b not in DW_BASES)
    if n_dense != 34:
        raise SystemExit(f"P0 FAIL: {n_dense} dense rows != 34")
    if fc == FC_NEW_BASE:
        state = "PADDED (apply_mbv2_kpar8.py applied)"
    elif fc == FC_OLD_BASE:
        state = "NOT YET PADDED — run apply_mbv2_kpar8.py (banks built for 13416)"
    else:
        raise SystemExit(f"P0 FAIL: FC base {fc} is neither {FC_OLD_BASE} nor {FC_NEW_BASE}")
    print(f"[kp8-repack] P0 PASS: 34/34 dense bases %8==0; FC row 46 = {fc} ({state})")


def main() -> int:
    rng = random.Random(0x4B58)   # deterministic ("KX")
    proof_p0()
    zero = "0" * OLD_HEX
    all_ok = True
    for b in range(8):
        src = WDIR / f"uram_weights_bank{b}.mem"
        dst = WDIR / f"uram_weights_bank{b}_kp8.mem"
        old = load_bank(src)

        # ---- relocate: insert 3 zero words at the FC boundary ----
        reloc = old[:FC_OLD_BASE] + [zero] * FC_PAD + old[FC_OLD_BASE:]
        assert len(reloc) == RELOC_DEPTH

        # ---- repack: new line g = {reloc[8g+7], ..., reloc[8g]} ----
        # $readmemh assigns each line MSB-first, so the TEXT concatenation
        # tap7+...+tap0 puts tap j at VALUE bits [j*288 +: 288].
        new = ["".join(reloc[K * g + j] for j in range(K - 1, -1, -1))
               for g in range(GROUPS)]

        # ---- P1: full re-expansion equality (incl. the relocation map) ----
        for g, ln in enumerate(new):
            if len(ln) != NEW_HEX:
                raise SystemExit(f"bank{b} g={g}: new len {len(ln)} != {NEW_HEX}")
            for j in range(K):
                back = ln[(K - 1 - j) * OLD_HEX:(K - j) * OLD_HEX]
                if back != reloc[K * g + j]:
                    print(f"P1 FAIL bank{b} g={g} tap{j}")
                    all_ok = False
        # relocated image maps 1:1 back to the original
        for w in range(OLD_DEPTH):
            if reloc[reloc_index(w)] != old[w]:
                print(f"P1 FAIL bank{b}: reloc map broken at w={w}")
                all_ok = False
                break
        if reloc[FC_OLD_BASE:FC_NEW_BASE] != [zero] * FC_PAD:
            print(f"P1 FAIL bank{b}: FC pad words not zero")
            all_ok = False

        # ---- P2: random byte cross-check via integer bit slicing ----
        for _ in range(4096 // 8):
            w = rng.randrange(OLD_DEPTH)
            s = rng.randrange(32)
            wn = reloc_index(w)
            old_v = int(old[w], 16)
            new_v = int(new[wn // K], 16)
            ob = (old_v >> (s * 8)) & 0xFF
            nb = (new_v >> ((wn % K) * 288 + s * 8)) & 0xFF
            if ob != nb:
                print(f"P2 FAIL bank{b} w={w} s={s}: old={ob:02x} new={nb:02x}")
                all_ok = False

        # ---- P3: aligned-group tap-slice identity, unrelocated region ----
        for _ in range(512 // 8):
            w0 = rng.randrange(0, FC_OLD_BASE - K, K)
            new_v = int(new[w0 // K], 16)
            for j in range(K):
                tap = (new_v >> (j * 288)) & ((1 << 256) - 1)
                ref = int(old[w0 + j], 16) & ((1 << 256) - 1)
                if tap != ref:
                    print(f"P3 FAIL bank{b} w0={w0} tap{j}")
                    all_ok = False

        # ---- P4: FC fast-walk identity at the PADDED base ----
        for p in range(4):
            for k in rng.sample(range(0, 1280, K), 32):
                wn0 = FC_NEW_BASE + p * 1280 + k
                assert wn0 % K == 0
                new_v = int(new[wn0 // K], 16)
                for j in range(K):
                    tap = (new_v >> (j * 288)) & ((1 << 256) - 1)
                    ref = int(old[FC_OLD_BASE + p * 1280 + k + j], 16) & ((1 << 256) - 1)
                    if tap != ref:
                        print(f"P4 FAIL bank{b} pass{p} k={k} tap{j}")
                        all_ok = False

        if not all_ok:
            return 1
        dst.write_text("\n".join(new) + "\n", encoding="ascii", newline="\n")
        print(f"[kp8-repack] bank{b}: {OLD_DEPTH} x 288b -> {GROUPS} x 2304b "
              f"(FC pad {FC_PAD} @ {FC_OLD_BASE}) -> {dst.name}  P1/P2/P3/P4 PASS")
    print(f"[kp8-repack] ALL 8 BANKS REPACKED, proofs P0..P4 PASS "
          f"(depth {OLD_DEPTH}->{GROUPS}, bits/bank {OLD_DEPTH*288} -> {GROUPS*2304})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
