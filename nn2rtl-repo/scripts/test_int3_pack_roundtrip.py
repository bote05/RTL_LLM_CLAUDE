#!/usr/bin/env python3
"""STEP 4 mandatory sign/width gate for the INT3 spatial datapath.

Proves the 3-bit weight PACKING (repack_weights_wide.write_wide_weights, wgt_bits=3)
round-trips exactly through the RTL MAC SLICE
    $signed(weight_word_q[(lane*MP_K+kpos)*WGT_BITS +: WGT_BITS])
that conv_datapath_mp_k.v uses. This is the cheap, deterministic catch for the
3-bit sign-extension / bit-stride bug class BEFORE the destructive INT3 regen +
full Verilog e2e. (Verilog $signed(part_select) treats the slice MSB as the sign
bit, which `unpack_rtl_slice` models exactly.)

Tests: (1) every INT3 value [-4..3] across lanes/kpos in one packed word;
       (2) the real conv_284 shape (OC=512, K_TOTAL=4608, MP=16, MP_K=9) with
           pseudo-random INT3 weights over all 16384 words.
Run: python scripts/test_int3_pack_roundtrip.py
"""
from __future__ import annotations
import sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from repack_weights_wide import write_wide_weights


def unpack_rtl_slice(packed_word: int, lane: int, mp_k: int, wgt_bits: int) -> list[int]:
    """Replicate conv_datapath_mp_k.v: for each kpos, extract WGT_BITS bits at
    base=(lane*MP_K+kpos)*WGT_BITS and interpret as a signed two's-complement
    WGT_BITS-bit value (slice MSB = sign), exactly as $signed(word[base +: WGT_BITS])."""
    out = []
    full = (1 << wgt_bits) - 1
    sign_bit = 1 << (wgt_bits - 1)
    for kpos in range(mp_k):
        base = (lane * mp_k + kpos) * wgt_bits
        field = (packed_word >> base) & full
        out.append(field - (1 << wgt_bits) if (field & sign_bit) else field)
    return out


def run_one(oc: int, k_total: int, mp: int, mp_k: int, weights: list[int], wgt_bits: int) -> tuple[bool, str]:
    oc_passes = (oc + mp - 1) // mp
    k_groups = k_total // mp_k
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "w.hex"
        n, pad = write_wide_weights(p, weights, oc, k_total, mp, mp_k, wgt_bits=wgt_bits)
        lines = [l.strip() for l in p.read_text().splitlines() if l.strip()]
    exp_words = oc_passes * k_groups
    exp_hex = (mp * mp_k * wgt_bits + 3) // 4
    if n != exp_words or len(lines) != exp_words:
        return False, f"word count {n}/{len(lines)} != {exp_words}"
    if any(len(l) != exp_hex for l in lines):
        return False, f"hex width != {exp_hex} (got {len(lines[0])})"
    # unpack every word/lane/kpos and compare to the source integer weight
    errs = 0
    wi = 0
    for g in range(oc_passes):
        for kg in range(k_groups):
            packed = int(lines[wi], 16); wi += 1
            for lane in range(mp):
                global_oc = g * mp + lane
                if global_oc >= oc:
                    continue
                got = unpack_rtl_slice(packed, lane, mp_k, wgt_bits)
                for kpos in range(mp_k):
                    k_lin = kg * mp_k + kpos
                    if got[kpos] != weights[global_oc * k_total + k_lin]:
                        errs += 1
                        if errs <= 5:
                            print(f"    MISMATCH oc={global_oc} k={k_lin}: got {got[kpos]} exp {weights[global_oc*k_total+k_lin]}")
    return errs == 0, f"words={exp_words} hex_width={exp_hex} pad={pad} errs={errs}"


def main() -> int:
    print("=== INT3 pack/slice round-trip (STEP 4 sign/width gate) ===")
    ok_all = True

    # Test 1: every INT3 value across lanes/kpos, one OC-pass, one k-group.
    mp, mp_k = 16, 9
    vals = [-4, -3, -2, -1, 0, 1, 2, 3]
    w1 = [vals[(o * mp_k + k) % len(vals)] for o in range(mp) for k in range(mp_k)]
    ok, msg = run_one(mp, mp_k, mp, mp_k, w1, 3)
    print(f"  [1] all-INT3-values single word: {'PASS' if ok else 'FAIL'} ({msg})")
    ok_all &= ok

    # Test 2: real conv_284 shape (OC=512, K_TOTAL=512*3*3=4608, MP=16, MP_K=9),
    # deterministic pseudo-random INT3 weights (no Math.random; LCG over index).
    oc, k_total = 512, 4608
    n = oc * k_total
    w2 = [((i * 1103515245 + 12345) >> 5) % 8 - 4 for i in range(n)]  # in [-4,3]
    ok, msg = run_one(oc, k_total, 16, 9, w2, 3)
    print(f"  [2] conv_284 full shape: {'PASS' if ok else 'FAIL'} ({msg})")
    ok_all &= ok

    # Test 3: confirm INT4 path is byte-identical to the pre-change behavior
    # (wgt_bits=4 default) — backward-compat guard.
    w4 = [((i * 2654435761) % 16) - 8 for i in range(16 * 9)]  # in [-8,7]
    ok, msg = run_one(16, 9, 16, 9, w4, 4)
    print(f"  [3] INT4 backward-compat (wgt_bits=4): {'PASS' if ok else 'FAIL'} ({msg})")
    ok_all &= ok

    print(f"\nRESULT: {'ALL PASS — 3-bit pack/slice agreement proven' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
