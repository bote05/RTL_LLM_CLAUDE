#!/usr/bin/env python3
"""Per-conv per-OC scale .mem for the SPATIAL convs (Phase 2 INT4-GPTQ).

conv_datapath_mp_k/parallel now read a per-OC scale ROM via SCALE_PATH: one
32-bit hex entry per OC (index 0 = OC 0, $readmemh top-of-file order), layout
bits[15:0]=mult (15-bit compute_scale_approx), bits[21:16]=shift. This writes
output/weights/node_conv_<id>_scale.mem for every conv2d in layer_ir that is NOT
an engine dispatch (engine convs use the wide scale.mem instead).

Run after the INT4-GPTQ goldens are regenerated (needs scale_factor_per_oc).
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
        lines = []
        for ch in range(oc):
            mult, shift = compute_scale_approx(float(per_oc[ch]))
            lines.append(f"{((shift & 0x3F) << 16) | (mult & 0xFFFF):08X}")
        (wdir / f"{mid}_scale.mem").write_text("\n".join(lines) + "\n", newline="\n")
        n += 1
    print(f"[spatial-scale] wrote {n} per-conv scale .mem files to output/weights/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
