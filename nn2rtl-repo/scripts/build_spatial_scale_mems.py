#!/usr/bin/env python3
"""Per-conv per-OC scale .mem for the SPATIAL convs (Phase 2 INT4-GPTQ).

conv_datapath_mp_k/parallel now read a per-OC scale ROM via SCALE_PATH: one
32-bit hex entry per OC (index 0 = OC 0, $readmemh top-of-file order), layout
bits[15:0]=mult (15-bit compute_scale_approx), bits[21:16]=shift. This writes
output/weights/node_conv_<id>_scale.mem for every conv2d in layer_ir that is NOT
an engine dispatch (engine convs use the wide scale.mem instead).

Run after the INT4-GPTQ goldens are regenerated (needs scale_factor_per_oc).

[DW-CONSTSHIFT 2026-06-10] FORMAT IS NOW PER-MODULE, decided by the CONSUMING RTL:
if BASE/rtl/<module_id>.v contains the "[DW-CONSTSHIFT" marker (the MBV2 depthwise
constant-shift requant, scripts/apply_mbv2_dw_constshift.py), the mem is emitted in
the CONSTANT-SHIFT format instead: slot[30:0] = mult' = mult << (23 - shift), one
leading // comment header as the format marker, and the RTL applies a single
compile-time >>> 23 with round 2^22. This keeps RTL and .mem format locked together
across regens (the ResNet-2953 stale-scale.mem hazard class): regenerating the mems
can never silently revert them to the {shift[21:16], mult[15:0]} layout while the
RTL expects mult'. Asserts shift<=23 and mult' < 2^31 per slot.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from golden_impl import compute_scale_approx  # noqa: E402

# [MBV2 2026-06-08] per-network base-dir override (e.g. output/mobilenet-v2) so this works for
# any network; default = ROOT/output (ResNet legacy layout) is unchanged.
BASE = Path(os.environ.get("NN2RTL_GOLDEN_BASE", str(ROOT / "output")))
if not BASE.is_absolute():
    BASE = ROOT / BASE


def main() -> int:
    ir = json.loads((BASE / "layer_ir.json").read_text())
    sched = json.loads((BASE / "rtl/nn2rtl_scheduler_schedule.json").read_text())
    engine_ids = {d["module_id"] for d in sched["dispatches"]}
    wdir = BASE / "weights"
    n = 0
    for L in ir["layers"]:
        if L.get("op_type") != "conv2d":
            continue
        mid = L["module_id"]
        if mid in engine_ids:
            continue  # engine conv -> wide scale.mem
        per_oc = L.get("scale_factor_per_oc")
        if per_oc is None:
            print(f"  WARN {mid}: no scale_factor_per_oc; skip"); continue
        oc = L["weight_shape"][0]
        # [DW-CONSTSHIFT 2026-06-10] format follows the CONSUMING RTL (see module docstring):
        # marker present -> emit pre-widened mult' (constant-shift); absent -> legacy layout.
        rtl_f = BASE / "rtl" / f"{mid}.v"
        constshift = rtl_f.exists() and "[DW-CONSTSHIFT" in rtl_f.read_text()
        lines = []
        if constshift:
            FS = 23  # must equal DW_FIXED_SHIFT in the consuming RTL
            lines.append(f"// [DW-CONSTSHIFT] {mid}_scale.mem -- CONSTANT-SHIFT format.")
            lines.append(f"// slot[30:0] = mult' = mult << ({FS} - shift)   "
                         "(was {shift[21:16], mult[15:0]})")
            lines.append(f"// consumed by {mid}.v: v = (biased * mult' + 2^{FS-1}) >>> {FS}")
            lines.append("// regen: scripts/build_spatial_scale_mems.py "
                         "(auto-detects the [DW-CONSTSHIFT] RTL marker)")
            for ch in range(oc):
                mult, shift = compute_scale_approx(float(per_oc[ch]))
                assert 0 <= shift <= FS, f"{mid} ch{ch}: shift={shift} outside [0,{FS}]"
                mp = mult << (FS - shift)
                assert mp < (1 << 31), f"{mid} ch{ch}: mult'={mp} >= 2^31 (slot[30:0] overflow)"
                lines.append(f"{mp:08X}")
        else:
            for ch in range(oc):
                mult, shift = compute_scale_approx(float(per_oc[ch]))
                lines.append(f"{((shift & 0x3F) << 16) | (mult & 0xFFFF):08X}")
        (wdir / f"{mid}_scale.mem").write_text("\n".join(lines) + "\n", newline="\n")
        n += 1
    print(f"[spatial-scale] wrote {n} per-conv scale .mem files to {wdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
