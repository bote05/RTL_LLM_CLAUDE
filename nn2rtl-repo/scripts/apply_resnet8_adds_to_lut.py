#!/usr/bin/env python3
"""Move the ResNet-8 residual-add fused-scale multiplies from DSP to LUT, to FREE
DSP for the K_PAR=16 conv MAC arrays.

WHY
---
node_add_25/56/87 carry  (* use_dsp = "yes" *)  on their per-channel fused-scale
products (lhs_term / rhs_term). Post-synth that costs 48 + 128 + 256 = 432 DSP.
The adds are tiny 3-cycle free-running INT8*const stages and are NOT on any
critical path; their products fit cheaply in LUT. Freeing those 432 DSP lets the
six K_PAR=16 3x3 convs (256 mult each) ALL map to the DSP array instead of
spilling ~120K LUT (conv_2 alone spilled 20K, conv_5 29K).

BYTE-EXACT: changing the implementation HINT (use_dsp yes->no) does not change
the arithmetic; LUT and DSP compute the identical integer products. Cycle-
identical, value-identical. Idempotent; writes .preadddsp backups.

VERIFY: NN2RTL_VALUE_THREADS=1 NN2RTL_VALUE_XINIT=0 \
        npx tsx scripts/run_resnet8_top_value.ts 0   -> result=PASS mismatch_bytes=0
"""
from __future__ import annotations
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RTL = REPO / "output" / "resnet8" / "rtl"
ADDS = ["node_add_25", "node_add_56", "node_add_87"]


def main():
    n = 0
    for mid in ADDS:
        p = RTL / f"{mid}.v"
        txt = p.read_text()
        if '(* use_dsp = "yes" *)' not in txt:
            print(f"{mid}: no use_dsp=yes (already LUT or changed); skip")
            continue
        bak = p.with_suffix(".v.preadddsp")
        if not bak.exists():
            bak.write_bytes(p.read_bytes())
        txt = txt.replace('(* use_dsp = "yes" *)', '(* use_dsp = "no" *)')
        p.write_text(txt)
        n += 1
        print(f"{mid}: residual-add products use_dsp yes->no (DSP -> LUT)")
    print(f"done: {n} add modules moved to LUT.")


if __name__ == "__main__":
    main()
