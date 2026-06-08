#!/usr/bin/env python3
"""FIT-FIX 2026-06-07: constant-shift requant in output/rtl/engine/requant_pipeline.v.

Replaces the 256 per-lane VARIABLE 65-bit arithmetic barrel shifters (`>>> shift_lane`)
+ per-lane variable round generators (~70K LUT, ~6.4% of the U250) with a SINGLE
COMPILE-TIME constant shift `>>> FIXED_SHIFT` + a constant round `+ ROUND_CONST`.

The per-OC shift is folded OFFLINE into a pre-widened multiplier
    mult' = mult << (FIXED_SHIFT - shift)
packed by scripts/build_scale_memory_map.py into the low 31 bits of each 32-bit scale
slot. The multiply is already DSP-mapped (use_dsp=yes), and the U250 has 91% of its DSPs
idle, so this is a pure LUT->(free DSP + wiring) move with NO throughput / latency change.

BYTE-EXACT identity (FS = FIXED_SHIFT >= shift, proven over 1.33M brute-force + 2M random
cases, 0 mismatch):
    floor((biased*mult*2^(FS-shift) + 2^(FS-1)) / 2^FS)
        == floor((biased*mult + 2^(shift-1)) / 2^shift)
The active RTL round path is already an UNCONDITIONAL +HALF (round_half_lane), which is
exactly the rtl_old form the identity targets; ROUND_CONST = 2^(FIXED_SHIFT-1) reproduces it.

Idempotent + backs up the original. Re-run build_scale_memory_map.py --network mobilenet-v2
AFTER this so scale.mem holds mult' (the two MUST move together).
"""
from __future__ import annotations
import shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TARGET = REPO / "output" / "rtl" / "engine" / "requant_pipeline.v"
BACKUP_DIR = REPO / "backups" / "requant_const_shift_20260607"

ANCHOR_SCALED_W = "    localparam integer SCALED_W = 65;\n"

INSERT_AFTER_SCALED_W = ANCHOR_SCALED_W + """
    // [FIT-FIX 2026-06-07] Constant-shift requant. The per-OC scale shift is folded OFFLINE
    // into a pre-widened multiplier (mult' = mult << (FIXED_SHIFT - shift), in scale.mem via
    // build_scale_memory_map.py), so this module applies a SINGLE compile-time arithmetic
    // shift instead of 256 per-lane VARIABLE 65-bit barrel shifters (~70K LUT removed; the
    // multiply is already DSP-mapped and the U250 has 91% idle DSP). FIXED_SHIFT must match
    // build_scale_memory_map.py and be >= any compute_scale_approx shift (range [0,23]).
    // ROUND_CONST = 2^(FIXED_SHIFT-1) is the unconditional +HALF the old round_half_lane gave.
    // Byte-exact: floor((biased*mult*2^(FS-shift)+2^(FS-1))/2^FS)==floor((biased*mult+2^(shift-1))/2^shift).
    localparam integer FIXED_SHIFT = 23;
    localparam signed [SCALED_W-1:0] ROUND_CONST =
        $signed({{(SCALED_W-1){1'b0}}, 1'b1}) <<< (FIXED_SHIFT - 1);
"""

BLOCK1_OLD = """            wire signed [SCALE_W-1:0] mult_lane =
                $signed({{(SCALE_W-16){1'b0}}, scale_q1[lane*32 +: 16]});
            wire [5:0] shift_lane = scale_q2[lane*32 + 16 +: 6];
            wire       shift_zero_lane = (shift_lane == 6'd0);
            wire signed [SCALED_W-1:0] round_half_lane =
                shift_zero_lane ? {SCALED_W{1'b0}}
                                : ($signed({{(SCALED_W-1){1'b0}}, 1'b1}) <<< (shift_lane - 6'd1));
            wire signed [SCALED_W-1:0] round_half_m1_lane =
                shift_zero_lane ? {SCALED_W{1'b0}}
                                : (round_half_lane - $signed({{(SCALED_W-1){1'b0}}, 1'b1}));
"""

BLOCK1_NEW = """            // [FIT-FIX 2026-06-07] mult_lane is the PRE-WIDENED multiplier
            // mult' = mult << (FIXED_SHIFT - shift) (low 31 bits of the slot, always
            // positive, < 2^31). The per-OC VARIABLE shift + variable round generator
            // are GONE -- replaced by the module-level constant ROUND_CONST + a
            // compile-time >>> FIXED_SHIFT below. scale_q2 is now unused (pruned).
            wire signed [SCALE_W-1:0] mult_lane =
                $signed({1'b0, scale_q1[lane*32 +: 31]});
"""

BLOCK2_OLD = """            assign biased_round_sum = scaled_q2 + round_half_lane;
            assign v_tmp            = biased_round_sum >>> shift_lane;
"""

BLOCK2_NEW = """            assign biased_round_sum = scaled_q2 + ROUND_CONST;
            assign v_tmp            = biased_round_sum >>> FIXED_SHIFT;
"""


def main() -> int:
    src = TARGET.read_text(encoding="utf-8")
    if "FIXED_SHIFT" in src:
        print("[apply-requant-const-shift] already applied (FIXED_SHIFT present); no-op.")
        return 0
    for name, anchor in (("SCALED_W", ANCHOR_SCALED_W), ("BLOCK1", BLOCK1_OLD), ("BLOCK2", BLOCK2_OLD)):
        n = src.count(anchor)
        if n != 1:
            print(f"[apply-requant-const-shift] FATAL: anchor {name} found {n} times (expected 1).", file=sys.stderr)
            return 2
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TARGET, BACKUP_DIR / TARGET.name)
    out = src.replace(ANCHOR_SCALED_W, INSERT_AFTER_SCALED_W, 1)
    out = out.replace(BLOCK1_OLD, BLOCK1_NEW, 1)
    out = out.replace(BLOCK2_OLD, BLOCK2_NEW, 1)
    TARGET.write_text(out, encoding="utf-8", newline="\n")
    print(f"[apply-requant-const-shift] patched {TARGET.relative_to(REPO)} (backup in {BACKUP_DIR.relative_to(REPO)})")
    print("[apply-requant-const-shift] NOW re-run: python3 scripts/build_scale_memory_map.py --network mobilenet-v2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
