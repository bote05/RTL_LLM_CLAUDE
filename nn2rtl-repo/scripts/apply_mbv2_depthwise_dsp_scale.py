#!/usr/bin/env python3
"""FIT-FIX 2026-06-07: DSP-offload the per-tensor requant multiply in the 17 instantiated
MobileNetV2 depthwise (3x3) conv wrappers.

Each wrapper computes `scaled[lane] <= $signed(biased[lane]) * $signed(SCALE_MULT_CONST)`
in ST_SCALE, but the `scaled` reg carries NO use_dsp attribute, so the 34x16 CONSTANT
multiply (SCALE_MULT_CONST has many set bits -> expensive shift-add) maps to LUTs. The 9
tap products already use DSP; only this scale multiply leaks into LUTs.

Adding `(* use_dsp = "yes" *)` forces it into the 91%-idle DSP pool. This is purely a
cell-MAPPING change: same value, same width (SCALED_W), same pipeline stage -> BYTE-EXACT
and latency-neutral. It can only REMOVE LUTs (or be a no-op if Vivado already chose DSP);
it never adds LUTs. DSP cost: MP(=4) lanes x 17 convs = ~68 DSP48E2, trivial vs 12288.

Targets ONLY the 17 convs actually instantiated in nn2rtl_top_engine.v (the 3x3 depthwise;
confirmed as u_node_conv_* in the synth hier). The pointwise convs are handled by the shared
engine and are NOT instantiated (pruned), so they are skipped. The stem (node_conv_810) uses
the shared conv_datapath_mp_k and is skipped to keep ResNet bit-identical.

Idempotent + backs up each original. Verify with: npx tsx scripts/verify_mbv2_batch.ts
(per-module Verilator vs sidecar golden, mismatch=0).
"""
from __future__ import annotations
import shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RTL = REPO / "output" / "mobilenet-v2" / "rtl"
BACKUP_DIR = REPO / "backups" / "mbv2_depthwise_dsp_scale_20260607"

DEPTHWISE = [812, 818, 824, 830, 836, 842, 848, 854, 860, 866,
             872, 878, 884, 890, 896, 902, 908]

ANCHOR = "    reg signed [SCALED_W-1:0] scaled [0:MP-1];\n"
REPLACE = '    (* use_dsp = "yes" *) reg signed [SCALED_W-1:0] scaled [0:MP-1];\n'


def main() -> int:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    patched, skipped = 0, 0
    for n in DEPTHWISE:
        f = RTL / f"node_conv_{n}.v"
        if not f.exists():
            print(f"[dsp-scale] FATAL: {f.relative_to(REPO)} missing", file=sys.stderr)
            return 2
        src = f.read_text(encoding="utf-8")
        if REPLACE.strip() in src:
            print(f"[dsp-scale] node_conv_{n}: already tagged; skip")
            skipped += 1
            continue
        if src.count(ANCHOR) != 1:
            print(f"[dsp-scale] FATAL: node_conv_{n}: anchor found {src.count(ANCHOR)} times (expected 1)", file=sys.stderr)
            return 2
        shutil.copy2(f, BACKUP_DIR / f.name)
        f.write_text(src.replace(ANCHOR, REPLACE, 1), encoding="utf-8", newline="\n")
        patched += 1
        print(f"[dsp-scale] node_conv_{n}: use_dsp applied")
    print(f"[dsp-scale] done: {patched} patched, {skipped} already-applied (backups in {BACKUP_DIR.relative_to(REPO)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
