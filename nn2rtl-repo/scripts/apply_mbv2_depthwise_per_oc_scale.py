#!/usr/bin/env python3
"""[ACCURACY 2026-06-08] Convert the 17 inlined MobileNetV2 depthwise wrappers from a
PER-TENSOR requant (localparam SCALE_MULT/SCALE_SHIFT -> SCALE_MULT_CONST/SCALE_ROUND_BIAS)
to a PER-OUTPUT-CHANNEL scale ROM, deploying the proven +4.13% top-1 win.

Each wrapper gains:  reg [31:0] scale_rom [0:C-1]; + $readmemh node_conv_<id>_scale.mem
ST_SCALE uses per-lane mult = scale_rom[oc_group*MP+lane_i][15:0]
ST_OUTPUT uses per-OC shift = scale_rom[out_oc][21:16] (variable >>> shift, round = 1<<(shift-1))
matching rtl_library/conv_datapath_mp_k.v SCALE_PATH exactly (and golden_impl
requantize_tensor_with_scale_per_oc, which is bit-identical).

The requant region is byte-identical across all 17 DW (verified). Idempotent (skips if
scale_rom already present), anchor-validated (aborts if any anchor count != 1), backs up first.

Usage:  python scripts/apply_mbv2_depthwise_per_oc_scale.py [--dry-run]
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "output" / "mobilenet-v2" / "rtl"
DEPTHWISE = [812, 818, 824, 830, 836, 842, 848, 854, 860, 866, 872,
             878, 884, 890, 896, 902, 908]

# ---- the 6 exact string edits (byte-identical anchors across all 17 DW) ----
# (A1) scale_rom declaration after the biases ROM decl
A1_OLD = (
    "    reg signed [31:0] biases  [0:C-1];\n"
    "\n"
    "    initial begin\n"
)
A1_NEW = (
    "    reg signed [31:0] biases  [0:C-1];\n"
    "    // [PER-OC 2026-06-08] per-output-channel requant ROM: {shift[21:16], mult[15:0]} per OC\n"
    "    // (compute_scale_approx of the composite per-OC scale). Replaces the per-tensor SCALE_*.\n"
    "    (* rom_style = \"block\", ram_style = \"block\" *)\n"
    "    reg [31:0]        scale_rom [0:C-1];\n"
    "\n"
    "    initial begin\n"
)
# (B1) per-OC OUTPUT-stage regs after v_tmp decl
B1_OLD = "    reg signed [SCALED_W-1:0] v_tmp;\n"
B1_NEW = (
    "    reg signed [SCALED_W-1:0] v_tmp;\n"
    "    reg        [5:0]          out_shift;  // [PER-OC] per-OC shift (OUTPUT stage)\n"
    "    reg signed [SCALED_W-1:0] out_round;  // [PER-OC] per-OC round bias (OUTPUT stage)\n"
)
# (B2) sc_oc loop var
B2_OLD = "    integer bias_oc, out_oc;\n"
B2_NEW = "    integer bias_oc, out_oc, sc_oc;\n"
# (C) ST_SCALE per-OC mult (anchor = the for + the unique SCALE_MULT_CONST line)
C_OLD = (
    "                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1)\n"
    "                        scaled[lane_i] <= $signed(biased[lane_i]) * $signed(SCALE_MULT_CONST);\n"
)
C_NEW = (
    "                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin\n"
    "                        sc_oc = oc_group * MP + lane_i;\n"
    "                        if (sc_oc < C)\n"
    "                            scaled[lane_i] <= $signed(biased[lane_i]) * $signed(scale_rom[sc_oc][15:0]);\n"
    "                        else\n"
    "                            scaled[lane_i] <= {SCALED_W{1'b0}};\n"
    "                    end\n"
)
# (D) ST_OUTPUT per-OC shift/round (out_oc already computed just above this line)
D_OLD = "                            v_tmp = (scaled[lane_i] + SCALE_ROUND_BIAS) >>> SCALE_SHIFT;\n"
D_NEW = (
    "                            out_shift = scale_rom[out_oc][21:16];\n"
    "                            out_round = (out_shift == 6'd0) ? {SCALED_W{1'b0}}\n"
    "                                      : ({{(SCALED_W-1){1'b0}}, 1'b1} <<< (out_shift - 6'd1));\n"
    "                            v_tmp = (scaled[lane_i] + out_round) >>> out_shift;\n"
)

EDITS = [("A1", A1_OLD, A1_NEW), ("B1", B1_OLD, B1_NEW), ("B2", B2_OLD, B2_NEW),
         ("C", C_OLD, C_NEW), ("D", D_OLD, D_NEW)]


def transform(text: str, cid: int) -> str:
    # (A2) scale_rom $readmemh after the bias $readmemh (derive the abs path from the bias line)
    bias_marker = f'_bias.hex", biases);\n'
    if bias_marker not in text:
        raise RuntimeError(f"node_conv_{cid}: bias $readmemh anchor not found")
    # find the bias readmemh line to lift its absolute path stem
    import re
    m = re.search(r'\$readmemh\("([^"]+/node_conv_\d+)_bias\.hex", biases\);', text)
    if not m:
        raise RuntimeError(f"node_conv_{cid}: could not parse bias $readmemh path")
    stem = m.group(1)
    a2_old = f'        $readmemh("{stem}_bias.hex", biases);\n'
    a2_new = a2_old + f'        $readmemh("{stem}_scale.mem", scale_rom);\n'
    if text.count(a2_old) != 1:
        raise RuntimeError(f"node_conv_{cid}: A2 anchor count={text.count(a2_old)} (expect 1)")
    text = text.replace(a2_old, a2_new)
    # the 5 fixed edits
    for tag, old, new in EDITS:
        c = text.count(old)
        if c != 1:
            raise RuntimeError(f"node_conv_{cid}: edit {tag} anchor count={c} (expect 1)")
        text = text.replace(old, new)
    return text


def main() -> int:
    dry = "--dry-run" in sys.argv
    ts = time.strftime if False else None  # avoid Date.now-style nondeterminism warnings
    bkdir = ROOT / "backups" / "mbv2_depthwise_per_oc"
    if not dry:
        bkdir.mkdir(parents=True, exist_ok=True)
    done, skipped = 0, 0
    for cid in DEPTHWISE:
        f = RTL / f"node_conv_{cid}.v"
        text = f.read_text()
        if "scale_rom" in text:
            print(f"  node_conv_{cid}: already has scale_rom -> SKIP")
            skipped += 1
            continue
        new = transform(text, cid)  # raises on any bad anchor
        if dry:
            print(f"  node_conv_{cid}: OK (all 6 edits anchored)")
        else:
            (bkdir / f"node_conv_{cid}.v").write_text(text, newline="\n")
            f.write_text(new, newline="\n")
            print(f"  node_conv_{cid}: APPLIED (+ backup)")
        done += 1
    print(f"[per-oc-dw] {'validated' if dry else 'applied'}={done} skipped={skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
