#!/usr/bin/env python3
"""[DW-CONSTSHIFT 2026-06-10] MBV2 depthwise CONSTANT-SHIFT requant conversion.

Converts the 17 inlined MobileNetV2 depthwise wrappers (node_conv_812..908, the
[PER-OC 2026-06-08] per-channel scale-ROM form) from a per-channel VARIABLE
arithmetic shift requant tail to the FIT-FIX CONSTANT-SHIFT form proven on the
ResNet engine (output/rtl/engine/requant_pipeline.v [FIT-FIX 2026-06-07] +
scripts/build_scale_memory_map.py):

  OLD (per lane, ~272 lanes = 17 convs x MP16):
      scaled = biased * scale_rom[oc][15:0]                   (34b x 16b)
      shift  = scale_rom[oc][21:16]                            (variable, 0..23)
      round  = (shift==0) ? 0 : 1 << (shift-1)                 (variable decode)
      v      = (scaled + round) >>> shift                      (50b VARIABLE barrel shifter)

  NEW:
      mult'  = mult << (DW_FIXED_SHIFT - shift)                (folded OFFLINE, slot[30:0])
      scaled = biased * $signed({1'b0, mult'})                 (34b x 32b, DSP)
      v      = (scaled + DW_ROUND_CONST) >>> DW_FIXED_SHIFT    (compile-time 23, ROUND=1<<22)

  Byte-exact identity (shift s in [0,23], mult in [1,32767], FS=23):
      floor((x*mult + 2^(s-1))/2^s) == floor((x*(mult<<(FS-s)) + 2^(FS-1))/2^FS)
  because x*mult*2^(FS-s) + 2^(FS-1) = 2^(FS-s) * (x*mult + 2^(s-1)) exactly, and
  arithmetic >>> on signed values is floor division by a power of two. The s==0
  slots (round==0 in the old RTL) are covered too: floor((x*mult*2^FS + 2^(FS-1))/2^FS)
  == x*mult since 2^(FS-1) < 2^FS.

THE CRITICAL HAZARD this script is built around (ResNet "2953", memory
project_resnet_2953_stale_scalemem): the RTL slot format and the .mem files MUST
flip together. This script is ATOMIC: it validates EVERY RTL anchor and EVERY
mem slot of all 17 convs FIRST, and only then writes anything. It refuses to run
on a mixed state (some RTL new + mem old or vice versa).

Per-slot cross-checks (all 7,136 slots = sum of C over the 17 DW convs):
  * old slot bits [31:22] must be 0 (format sanity)
  * shift <= DW_FIXED_SHIFT(23) and 0 <= mult <= 32767 (compute_scale_approx range)
  * mult' = mult << (23-shift) < 2^31 (fits slot[30:0], signed-positive as 32b operand)
  * EXACT effective-factor identity: mult' * 2^shift == mult * 2^23 (integer, no float)
  * (default, --skip-ir-check to disable) old (mult,shift) must equal
    compute_scale_approx(layer_ir scale_factor_per_oc[ch]) -- proves the on-disk
    mems are in sync with the authoritative quantization source before converting.

REGEN-PIPELINE NOTE (documented in docs/agent_tasks/DW_CONSTSHIFT_ANALYSIS.md):
the authoritative generator of these mems is scripts/build_spatial_scale_mems.py
(run with NN2RTL_GOLDEN_BASE=output/mobilenet-v2). It has been extended to detect
the [DW-CONSTSHIFT] marker in the consuming RTL and emit the NEW format for those
modules automatically, so a future regen cannot silently revert the mems to the
old {shift,mult} format under constant-shift RTL.

Idempotent: RTL marker = "[DW-CONSTSHIFT" in the .v; mem marker = a leading
"// [DW-CONSTSHIFT" comment line ($readmemh skips // comments). Re-running is a
no-op. Backups of every touched file go to backups/mbv2_dw_constshift_<UTC ts>/.

Usage:
  python scripts/apply_mbv2_dw_constshift.py [--dry-run] [--skip-ir-check]
                                             [--repo-root PATH]
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

DEPTHWISE = [812, 818, 824, 830, 836, 842, 848, 854, 860, 866, 872,
             878, 884, 890, 896, 902, 908]

FIXED_SHIFT = 23                      # must equal DW_FIXED_SHIFT in the patched RTL
MARKER = "[DW-CONSTSHIFT"             # idempotency marker (RTL and .mem)
DATE = "2026-06-10"

# --------------------------------------------------------------------------
# The 6 RTL edits. Anchors verified byte-identical (count==1) across all 17
# DW wrappers on 2026-06-10 (post K1-MBV2 / post [PER-OC 2026-06-08] state).
# Any drift => count != 1 => hard abort BEFORE anything is written.
# --------------------------------------------------------------------------
E1_OLD = (
    "    // [PER-OC 2026-06-08] per-output-channel requant ROM: {shift[21:16], mult[15:0]} per OC\n"
    "    // (compute_scale_approx of the composite per-OC scale). Replaces the per-tensor SCALE_*.\n"
)
E1_NEW = (
    f"    // [PER-OC 2026-06-08][DW-CONSTSHIFT {DATE}] per-output-channel requant ROM. Slot is\n"
    "    // the PRE-WIDENED multiplier mult' = mult << (DW_FIXED_SHIFT - shift), bits [30:0]\n"
    "    // (< 2^31; the per-OC shift is folded OFFLINE -- scripts/apply_mbv2_dw_constshift.py /\n"
    "    // build_spatial_scale_mems.py). RTL applies ONE compile-time >>> DW_FIXED_SHIFT with a\n"
    "    // CONSTANT round, replacing the per-lane variable barrel shifter + round decode.\n"
)

E2_OLD = (
    "    localparam integer SCALE_CONST_W = 16;\n"
    "    localparam integer SCALED_W      = BIASED_W + SCALE_CONST_W; // 50\n"
)
E2_NEW = (
    "    localparam integer SCALE_CONST_W = 16;\n"
    f"    // [DW-CONSTSHIFT {DATE}] constant-shift requant (FIT-FIX form proven on the ResNet\n"
    "    // engine requant_pipeline.v 2026-06-07): the scale .mem now holds the pre-widened\n"
    "    // mult' = mult << (DW_FIXED_SHIFT - shift) so the variable per-OC shift + variable\n"
    "    // round decode collapse into ONE compile-time arithmetic shift + constant round.\n"
    "    // Byte-exact identity (shift in [0,23], mult in [1,32767]):\n"
    "    //   floor((x*mult + 2^(s-1))/2^s) == floor((x*(mult<<(23-s)) + 2^22)/2^23).\n"
    "    localparam integer MULTP_W       = 32; // signed operand width for mult' ({1'b0, slot[30:0]})\n"
    "    localparam integer SCALED_W      = BIASED_W + MULTP_W; // 66 (34b x 32b product, no truncation)\n"
    "    localparam integer DW_FIXED_SHIFT = 23;\n"
    "    localparam signed [SCALED_W-1:0] DW_ROUND_CONST =\n"
    "        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (DW_FIXED_SHIFT - 1);\n"
)

E3_OLD = (
    "    reg        [5:0]          out_shift;  // [PER-OC] per-OC shift (OUTPUT stage)\n"
    "    reg signed [SCALED_W-1:0] out_round;  // [PER-OC] per-OC round bias (OUTPUT stage)\n"
)
E3_NEW = ""  # the per-OC OUTPUT-stage shift/round temporaries are gone

E4_OLD = (
    "    // the original single block). i/lane_i/bias_oc/sc_oc/out_oc/out_shift/\n"
    "    // out_round/v_tmp are referenced ONLY by this block after the move.\n"
)
E4_NEW = (
    "    // the original single block). i/lane_i/bias_oc/sc_oc/out_oc/v_tmp\n"
    "    // are referenced ONLY by this block after the move.\n"
)

E5_OLD = (
    "                            scaled[lane_i] <= $signed(biased[lane_i]) * $signed(scale_rom[sc_oc][15:0]);\n"
)
E5_NEW = (
    "                            // [DW-CONSTSHIFT] slot = pre-widened mult' (bits [30:0], positive)\n"
    "                            scaled[lane_i] <= $signed(biased[lane_i]) * $signed({1'b0, scale_rom[sc_oc][30:0]});\n"
)

E6_OLD = (
    "                            out_shift = scale_rom[out_oc][21:16];\n"
    "                            out_round = (out_shift == 6'd0) ? {SCALED_W{1'b0}}\n"
    "                                      : ({{(SCALED_W-1){1'b0}}, 1'b1} <<< (out_shift - 6'd1));\n"
    "                            v_tmp = (scaled[lane_i] + out_round) >>> out_shift;\n"
)
E6_NEW = (
    "                            // [DW-CONSTSHIFT] per-OC shift folded offline into mult' ->\n"
    "                            // constant round + compile-time shift (no barrel shifter)\n"
    "                            v_tmp = (scaled[lane_i] + DW_ROUND_CONST) >>> DW_FIXED_SHIFT;\n"
)

EDITS = [("E1", E1_OLD, E1_NEW), ("E2", E2_OLD, E2_NEW), ("E3", E3_OLD, E3_NEW),
         ("E4", E4_OLD, E4_NEW), ("E5", E5_OLD, E5_NEW), ("E6", E6_OLD, E6_NEW)]


def transform_rtl(text: str, cid: int) -> str:
    for tag, old, new in EDITS:
        c = text.count(old)
        if c != 1:
            raise RuntimeError(f"node_conv_{cid}: RTL anchor {tag} count={c} (expect 1) -> ABORT")
        text = text.replace(old, new)
    return text


def decode_old_mem(lines: list[str], cid: int) -> list[tuple[int, int, int]]:
    """Decode OLD-format slots -> [(mult, shift, mult_prime)], with all cross-checks."""
    out = []
    for ch, ln in enumerate(lines):
        v = int(ln, 16)
        if v >> 22:
            raise RuntimeError(f"node_conv_{cid} ch{ch}: slot {ln} has bits[31:22] set "
                               f"-> NOT old {{shift,mult}} format -> ABORT")
        mult = v & 0xFFFF
        shift = (v >> 16) & 0x3F
        if shift > FIXED_SHIFT:
            raise RuntimeError(f"node_conv_{cid} ch{ch}: shift={shift} > {FIXED_SHIFT} -> ABORT")
        if mult > 0x7FFF:
            raise RuntimeError(f"node_conv_{cid} ch{ch}: mult={mult} > 32767 (15-bit cap) -> ABORT")
        mp = mult << (FIXED_SHIFT - shift)
        if mp >= 1 << 31:
            raise RuntimeError(f"node_conv_{cid} ch{ch}: mult'={mp} >= 2^31 -> ABORT")
        # EXACT effective-factor identity, integer arithmetic (no float):
        #   mult' / 2^23 == mult / 2^shift  <=>  mult' * 2^shift == mult * 2^23
        if mp * (1 << shift) != mult * (1 << FIXED_SHIFT):
            raise RuntimeError(f"node_conv_{cid} ch{ch}: effective-factor identity FAILED -> ABORT")
        out.append((mult, shift, mp))
    return out


def render_new_mem(cid: int, slots: list[tuple[int, int, int]]) -> str:
    hdr = [
        f"// [DW-CONSTSHIFT {DATE}] node_conv_{cid}_scale.mem -- CONSTANT-SHIFT format.",
        f"// slot[30:0] = mult' = mult << ({FIXED_SHIFT} - shift)   "
        "(was {shift[21:16], mult[15:0]})",
        f"// consumed by node_conv_{cid}.v: v = (biased * mult' + 2^{FIXED_SHIFT-1}) "
        f">>> {FIXED_SHIFT}   (DW_FIXED_SHIFT = {FIXED_SHIFT})",
        "// regen: scripts/build_spatial_scale_mems.py (auto-detects the [DW-CONSTSHIFT] RTL marker)",
    ]
    return "\n".join(hdr) + "\n" + "\n".join(f"{mp:08X}" for _, _, mp in slots) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="validate + audit only, write nothing")
    ap.add_argument("--skip-ir-check", action="store_true",
                    help="skip the layer_ir.json/compute_scale_approx provenance cross-check")
    ap.add_argument("--repo-root", default=None,
                    help="nn2rtl repo root (default: parent of this script's directory)")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parent.parent
    rtl_dir = root / "output" / "mobilenet-v2" / "rtl"
    w_dir = root / "output" / "mobilenet-v2" / "weights"
    if not rtl_dir.is_dir() or not w_dir.is_dir():
        print(f"ERROR: {rtl_dir} / {w_dir} not found (wrong --repo-root?)", file=sys.stderr)
        return 2

    # Optional provenance cross-check source
    ir_by_mid = None
    if not args.skip_ir_check:
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / "scripts"))
        from golden_impl import compute_scale_approx  # noqa: E402
        ir = json.loads((root / "output" / "mobilenet-v2" / "layer_ir.json").read_text())
        ir_by_mid = {L["module_id"]: L for L in ir["layers"]}

    # ---------------- PHASE 0: validate EVERYTHING, write NOTHING ----------------
    plan = []           # (cid, new_rtl_text|None, new_mem_text|None, slots)
    audit_rows = []
    n_already = 0
    for cid in DEPTHWISE:
        rtl_f = rtl_dir / f"node_conv_{cid}.v"
        mem_f = w_dir / f"node_conv_{cid}_scale.mem"
        rtl_text = rtl_f.read_text()
        mem_text = mem_f.read_text()
        rtl_done = MARKER in rtl_text
        mem_done = mem_text.lstrip().startswith("// " + MARKER)
        if rtl_done != mem_done:
            print(f"ERROR: node_conv_{cid} MIXED STATE (RTL constshift={rtl_done}, "
                  f"mem constshift={mem_done}). This is exactly the ResNet-2953 hazard. "
                  "Restore from backups/ and re-run.", file=sys.stderr)
            return 3
        if rtl_done:
            n_already += 1
            plan.append((cid, None, None, None))
            continue

        new_rtl = transform_rtl(rtl_text, cid)        # raises on any bad anchor
        lines = [x for x in mem_text.split() if x]
        slots = decode_old_mem(lines, cid)            # raises on any bad slot

        if ir_by_mid is not None:
            L = ir_by_mid.get(f"node_conv_{cid}")
            per = None if L is None else L.get("scale_factor_per_oc")
            if per is None or len(per) != len(slots):
                raise RuntimeError(f"node_conv_{cid}: layer_ir scale_factor_per_oc missing/len "
                                   f"mismatch ({None if per is None else len(per)} vs {len(slots)})")
            for ch, (sf, (m, s, _)) in enumerate(zip(per, slots)):
                em, es = compute_scale_approx(float(sf))
                if (em, es) != (m, s):
                    raise RuntimeError(f"node_conv_{cid} ch{ch}: on-disk (mult,shift)=({m},{s}) != "
                                       f"layer_ir compute_scale_approx ({em},{es}) -> STALE MEM, ABORT")

        sh = [s for _, s, _ in slots]
        mp = [p for _, _, p in slots]
        audit_rows.append((cid, len(slots), min(sh), max(sh), sum(1 for s in sh if s == 0),
                           max(mp), max(mp).bit_length()))
        plan.append((cid, new_rtl, render_new_mem(cid, slots), slots))

    # ---------------- audit table ----------------
    if audit_rows:
        print(f"{'conv':>5} {'C':>4} {'shift_min':>9} {'shift_max':>9} {'#shift0':>7} "
              f"{'max_mult_prime':>14} {'bits':>4}")
        tot = 0
        for r in audit_rows:
            print(f"{r[0]:>5} {r[1]:>4} {r[2]:>9} {r[3]:>9} {r[4]:>7} {r[5]:>14} {r[6]:>4}")
            tot += r[1]
        print(f"[dw-constshift] audit: {tot} slots, ALL shift<=23, ALL mult'<2^31, "
              f"effective factors EXACT, layer_ir provenance "
              f"{'SKIPPED' if args.skip_ir_check else 'VERIFIED'}")
    if n_already:
        print(f"[dw-constshift] {n_already}/17 already converted (marker present) -> skipped")

    if args.dry_run:
        print("[dw-constshift] DRY RUN: validation complete, nothing written")
        return 0
    if n_already == len(DEPTHWISE):
        print("[dw-constshift] all 17 already converted -- nothing to do")
        return 0

    # ---------------- PHASE 1: backup + write RTL AND mem together ----------------
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    bkdir = root / "backups" / f"mbv2_dw_constshift_{ts}"
    bkdir.mkdir(parents=True, exist_ok=True)
    n_app = 0
    for cid, new_rtl, new_mem, _ in plan:
        if new_rtl is None:
            continue
        rtl_f = rtl_dir / f"node_conv_{cid}.v"
        mem_f = w_dir / f"node_conv_{cid}_scale.mem"
        shutil.copy2(rtl_f, bkdir / rtl_f.name)
        shutil.copy2(mem_f, bkdir / mem_f.name)
        rtl_f.write_text(new_rtl, newline="\n")
        mem_f.write_text(new_mem, newline="\n")
        n_app += 1
        print(f"  node_conv_{cid}: RTL + scale.mem converted (backup -> {bkdir.name}/)")
    print(f"[dw-constshift] APPLIED {n_app} convs atomically (RTL+mem together); "
          f"backups in {bkdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
