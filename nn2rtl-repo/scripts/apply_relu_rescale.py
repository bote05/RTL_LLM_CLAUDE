#!/usr/bin/env python3
"""FIX: add the missing activation rescale to the 22 ReLU nodes whose
input_scale != output_scale.

ROOT CAUSE (2026-05-30): the RTL ReLU template emits pure max(0,x), but 22 of 48
relus must REQUANTIZE (golden = round_half_up(max(0,x) * input_scale/output_scale)).
Missing the rescale fed every downstream conv an input at the wrong scale
(relu_1 was x3 too small -> conv_200 93.9% wrong). Proven: triangulate_conv200
(conv_200.goldin -> scale.mem) == golden byte-exact; recompute(relu=max-only) == RTL.

This patches each node_relu_<id>.v to apply, per channel:
    relu = max(0, x)                       # >= 0
    out  = clamp((relu * RS_MULT + RS_ROUND) >>> RS_SHIFT, 0, 127)
with (RS_MULT, RS_SHIFT) recovered byte-exact from each relu's goldin->goldout
(scripts: exact bounding-interval recovery -> relu_rescale_params.json).

The 26 scale-preserving relus (ratio 1.0) are left as pure max(0,x) -- correct.
"""
from __future__ import annotations
import json, shutil, sys, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARAMS = json.loads((ROOT / "output/reports_integrated/relu_rescale_params.json").read_text())
RTL = ROOT / "output/rtl"
BK = ROOT / "backups" / "relu_rescale_20260530"
BK.mkdir(parents=True, exist_ok=True)

WRITE_PAT = "data_out[ch*8 +: 8] <= (tmp_byte > 8'sd0) ? tmp_byte : 8'sd0;"
TMP_DECL = "reg signed [7:0] tmp_byte;"

def repl_write() -> str:
    return ("begin\n"
            "                            rs_in  = (tmp_byte > 8'sd0) ? $signed(tmp_byte) : 32'sd0;\n"
            "                            rs_out = (rs_in * RS_MULT + RS_ROUND) >>> RS_SHIFT;\n"
            "                            data_out[ch*8 +: 8] <= (rs_out > 32'sd127) ? 8'sd127 : rs_out[7:0];\n"
            "                        end")

def main() -> int:
    dry = "--dry-run" in sys.argv
    n = 0
    for mid, p in PARAMS.items():
        mult, shift = int(p["mult"]), int(p["shift"])
        rnd = (1 << (shift - 1)) if shift > 0 else 0
        f = RTL / f"{mid}.v"
        if not f.exists():
            print(f"  SKIP missing {mid}"); continue
        txt = f.read_text()
        if "RS_MULT" in txt:
            print(f"  SKIP already patched {mid}"); continue
        if txt.count(WRITE_PAT) != 2 or txt.count(TMP_DECL) != 1:
            print(f"  WARN {mid}: pattern×{txt.count(WRITE_PAT)} decl×{txt.count(TMP_DECL)} -- SKIP"); continue
        decl_block = (f"localparam integer RS_MULT  = {mult};\n"
                      f"    localparam integer RS_SHIFT = {shift};\n"
                      f"    localparam integer RS_ROUND = {rnd};\n"
                      f"    {TMP_DECL}\n"
                      f"    reg signed [31:0] rs_in, rs_out;")
        new = txt.replace(TMP_DECL, decl_block, 1).replace(WRITE_PAT, repl_write())
        if new == txt:
            print(f"  WARN {mid}: no change"); continue
        if dry:
            print(f"  [dry] {mid}: MULT={mult} SHIFT={shift} ROUND={rnd}")
        else:
            shutil.copy2(f, BK / f"{mid}.v")
            f.write_text(new, newline="\n")
            print(f"  [ok] {mid}: MULT={mult} SHIFT={shift} ROUND={rnd} (eff={mult/(1<<shift):.4f})")
        n += 1
    print(f"\n{'(dry) ' if dry else ''}patched {n}/22 relu nodes; backups -> {BK}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
