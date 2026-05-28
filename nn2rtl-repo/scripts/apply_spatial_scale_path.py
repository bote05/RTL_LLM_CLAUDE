#!/usr/bin/env python3
"""Wire per-OC SCALE_PATH into the spatial conv wrappers (Phase 2 INT4-GPTQ).

For each spatial conv that has a per-conv scale .mem (output/weights/node_conv_
<id>_scale.mem), add `.SCALE_PATH("...node_conv_<id>_scale.mem")` to its
conv_datapath_mp_k / conv_datapath_parallel instantiation (idempotent). The
datapath's per-tensor fallback means this only switches it to per-OC; without
SCALE_PATH the wrapper still behaves per-tensor.
"""
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "output/rtl"


def main() -> int:
    patched = skipped = already = 0
    for mem in sorted((ROOT / "output/weights").glob("node_conv_*_scale.mem")):
        mid = mem.name[:-len("_scale.mem")]
        vf = RTL / f"{mid}.v"
        if not vf.exists():
            print(f"  skip {mid}: no {vf.name}"); skipped += 1; continue
        src = vf.read_text()
        # ABSOLUTE path (match WEIGHTS_PATH/BIAS_PATH) so $readmemh resolves
        # regardless of the sim cwd. A relative path leaves scale_rom uninit -> 0.
        scale_path = f"{ROOT.as_posix()}/output/weights/{mid}_scale.mem"
        if ".SCALE_PATH(" in src:
            # replace any existing (e.g. earlier relative) SCALE_PATH.
            new, n = re.subn(r'\.SCALE_PATH\("[^"]*"\)',
                             f'.SCALE_PATH("{scale_path}")', src, count=1)
            if n == 1 and new != src:
                vf.write_text(new); patched += 1
            else:
                already += 1
            continue
        # insert .SCALE_PATH(...) right after the .SCALE_SHIFT(...) param.
        new, n = re.subn(r"(\.SCALE_SHIFT\s*\([^)]*\)\s*,)",
                         r'\1.SCALE_PATH("' + scale_path + r'"),',
                         src, count=1)
        if n != 1:
            print(f"  WARN {mid}: .SCALE_SHIFT param not found; not patched"); skipped += 1; continue
        vf.write_text(new)
        patched += 1
    print(f"[scale-path] patched={patched} already={already} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
