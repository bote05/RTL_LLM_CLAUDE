#!/usr/bin/env python3
"""TIER-1 byte-exact fix: align the RTL's requant SCALE constants to the golden's
15-bit compute_scale_approx (the e2e goldens' representation).

Two edits (RTL constants only; no logic change):
  1. Spatial convs (node_conv_*.v): the SDK's computeScaleApprox embedded
     SCALE_MULT/SCALE_SHIFT that disagree with the Python compute_scale_approx
     the goldens use, causing per-layer +/-1 that accumulates. Re-derive each
     conv's (mult,shift) from its layer_ir scale_factor via compute_scale_approx
     and patch the localparams.
  2. Engine scale ROM (nn2rtl_scheduler.v scale_mult_rom/scale_shift_rom): the
     engine carried a 32-bit near-exact scale (more precise than, hence != the
     golden's 15-bit). Patch each of the 14 dispatch entries to the golden's
     (mult,shift). The engine requant derives its round bias from scale_shift
     (HALF=1<<(shift-1)), so a smaller shift is safe.

Backs up every file to <file>.prescalefix before patching. Idempotent.
Usage: python scripts/apply_scale_constant_fix.py [--dry-run]
"""
from __future__ import annotations
import argparse, json, re, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IR = {l["module_id"]: l for l in json.loads((ROOT / "output/layer_ir.json").read_text())["layers"]}
SCHED = ROOT / "output/rtl/nn2rtl_scheduler.v"
DISPATCH = [246, 250, 254, 260, 264, 266, 272, 278, 282, 286, 290, 294, 296, 300]


def approx(sf: float) -> tuple[int, int]:
    best = (1, 0, float("inf"))
    for sh in range(0, 24):
        m = round(sf * (2 ** sh))
        if 1 <= m < 32768:
            e = abs(m / 2 ** sh - sf) / sf
            if e < best[2]:
                best = (m, sh, e)
    return best[0], best[1]


def backup(p: Path):
    b = p.with_suffix(p.suffix + ".prescalefix")
    if not b.exists():
        shutil.copy2(p, b)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # ---- 1. spatial conv localparams ----
    # Target the localparam declaration (NOT the human comment line). Two forms:
    #   combined : localparam integer SCALE_MULT=491, SCALE_SHIFT=16;
    #   separate : localparam integer SCALE_MULT  = 5855;
    #              localparam integer SCALE_SHIFT = 22;
    LP_COMBINED = re.compile(r"(localparam\s+integer\s+SCALE_MULT\s*=\s*)\d+(\s*,\s*SCALE_SHIFT\s*=\s*)\d+")
    LP_MULT = re.compile(r"(localparam\s+integer\s+SCALE_MULT\s*=\s*)(\d+)")
    LP_SHIFT = re.compile(r"(localparam\s+integer\s+SCALE_SHIFT\s*=\s*)(\d+)")
    conv_changes = []
    for vf in sorted((ROOT / "output/rtl").glob("node_conv_*.v")):
        mid = vf.stem
        l = IR.get(mid)
        if not l or "scale_factor" not in l:
            continue
        txt = vf.read_text()
        m_mult = LP_MULT.search(txt)
        if not m_mult:
            continue
        comb = LP_COMBINED.search(txt)
        if comb:
            rtl_shift = int(re.search(r"SCALE_SHIFT\s*=\s*(\d+)", comb.group(0)).group(1))
        else:
            m_sh = LP_SHIFT.search(txt)
            if not m_sh:
                continue
            rtl_shift = int(m_sh.group(2))
        rtl = (int(m_mult.group(2)), rtl_shift)
        gm, gs = approx(float(l["scale_factor"]))
        if rtl == (gm, gs):
            continue
        if comb:
            new = LP_COMBINED.sub(rf"\g<1>{gm}\g<2>{gs}", txt, count=1)
        else:
            new = LP_MULT.sub(rf"\g<1>{gm}", txt, count=1)
            new = LP_SHIFT.sub(rf"\g<1>{gs}", new, count=1)
        # keep the human comment consistent (harmless if absent)
        new = re.sub(r"SCALE_MULT=\d+,\s*SCALE_SHIFT=\d+", f"SCALE_MULT={gm}, SCALE_SHIFT={gs}", new)
        conv_changes.append((mid, rtl, (gm, gs)))
        if not args.dry_run:
            backup(vf); vf.write_text(new)
    print(f"[conv] patched {len(conv_changes)} spatial conv .v files:")
    for mid, o, n in conv_changes:
        print(f"   {mid:16s} {o} -> {n}")

    # ---- 2. engine scale ROM in scheduler ----
    sch = SCHED.read_text()
    rom_changes = []
    for d, c in enumerate(DISPATCH):
        sf = float(IR[f"node_conv_{c}"]["scale_factor"])
        gm, gs = approx(sf)
        mpat = rf"(4'd{d}:\s*scale_mult_rom\s*=\s*32'd)\d+(;)"
        spat = rf"(4'd{d}:\s*scale_shift_rom\s*=\s*6'd)\d+(;)"
        om = re.search(mpat, sch); os_ = re.search(spat, sch)
        oldm = int(re.search(r"\d+", om.group(0).split("32'd")[1]).group()) if om else None
        olds = int(re.search(r"6'd(\d+)", os_.group(0)).group(1)) if os_ else None
        sch = re.sub(mpat, rf"\g<1>{gm}\g<2>", sch)
        sch = re.sub(spat, rf"\g<1>{gs}\g<2>", sch)
        rom_changes.append((d, c, (oldm, olds), (gm, gs)))
    print(f"[engine] patched {len(rom_changes)} scale_mult/shift_rom entries:")
    for d, c, o, n in rom_changes:
        print(f"   d{d:>2} conv_{c}: mult/shift {o} -> {n}")
    if not args.dry_run:
        backup(SCHED); SCHED.write_text(sch)

    print("\n[dry-run] no files written" if args.dry_run else "\n[written] RTL scale constants aligned to golden compute_scale_approx")


if __name__ == "__main__":
    main()
