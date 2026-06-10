#!/usr/bin/env python3
"""apply_conv_output_pend.py — close the conv output-streamer pixel-drop race
(found by the engine-overlap forensics run; see ENGINE_OVERLAP_ANALYSIS.md).

The backpressured output streamer in every spatial conv node
(apply_3x3_backpressure.py template) latches a completed pixel with

    if (lib_valid_out_w && !out_busy) ...   // NO else => silent drop

The conv MAC pipeline is NOT spatial_run-gated, but the streamer IS (its
ready_out carries spatial_run). So whenever the chain freezes (scheduler
config-write windows; in the pre-overlap design also every engine run) while
a pixel is mid-stream (out_busy=1) AND the next pixel's MAC completes during
the freeze, that completed pixel is silently discarded -> downstream beat
counts come up short -> residual-join deadlock. PROVEN in the overlap e2e:
conv_248 pxout=196 but emit=6240=195*32 -> add_7 starved at 6240/6272 ->
S_WAIT_DRAIN(d1) hang. The class is LATENT in the serialized baseline (its
fixed schedule happens never to land a MAC completion inside a freeze with
the streamer mid-pixel); the overlap's 17 short config freezes expose it.

FIX: one-deep pending slot. A pixel completing while the streamer is busy is
PARKED (pend_pix/out_pend) and reloaded the moment the streamer finishes its
current pixel. One slot is provably sufficient: stall_in = mac_busy ||
out_busy blocks the next MAC start until the streamer is free, so at most
one completed-unstreamed pixel can exist. VALUE-PRESERVING: in any execution
where the drop never fires (e.g. the passing baseline schedule), out_pend
never sets and behavior is cycle-identical.

Usage: python scripts/apply_conv_output_pend.py [--dry-run]
Idempotent ([OVL-PEND] marker), per-file anchors asserted, backups to
backups/conv_output_pend/.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RTL = REPO / "output" / "rtl"
MARKER = "[OVL-PEND]"

DECL_OLD = "    reg                      out_busy;"
DECL_NEW = (
    "    reg                      out_busy;\n"
    f"    // {MARKER} one-deep pending slot: a pixel completing while the streamer\n"
    "    // is busy is parked here (never dropped) and reloaded when it frees.\n"
    "    reg                      out_pend;\n"
    "    reg [OUT_PIXEL_BITS-1:0] pend_pix;"
)

RESET_OLD = "out_idx<=0; out_busy<=1'b0;"
RESET_NEW = "out_idx<=0; out_busy<=1'b0; out_pend<=1'b0;"

DATA_OLD = (
    "        if (lib_valid_out_w && !out_busy)\n"
    "            out_pix <= lib_data_out_w;\n"
    "    end"
)
DATA_NEW = (
    "        if (lib_valid_out_w && !out_busy)\n"
    "            out_pix <= lib_data_out_w;\n"
    f"        // {MARKER} park / reload datapath\n"
    "        if (lib_valid_out_w && out_busy)\n"
    "            pend_pix <= lib_data_out_w;\n"
    "        if (out_busy && ready_out && (out_idx == OUT_BEATS-1) && out_pend)\n"
    "            out_pix <= pend_pix;\n"
    "    end"
)

CTRL_OLD = (
    "            if (lib_valid_out_w && !out_busy) begin\n"
    "                out_idx  <= 0;\n"
    "                out_busy <= 1'b1;\n"
    "            end else if (out_busy && ready_out) begin\n"
    "                if (out_idx == OUT_BEATS-1) out_busy <= 1'b0;\n"
    "                else                        out_idx  <= out_idx + 1'b1;\n"
    "            end"
)
CTRL_NEW = (
    "            if (lib_valid_out_w && !out_busy) begin\n"
    "                out_idx  <= 0;\n"
    "                out_busy <= 1'b1;\n"
    "            end else if (out_busy && ready_out) begin\n"
    f"                // {MARKER} reload the parked pixel (stay busy) instead of idling\n"
    "                if (out_idx == OUT_BEATS-1) begin\n"
    "                    if (out_pend) begin out_idx <= 0; out_pend <= 1'b0; end\n"
    "                    else out_busy <= 1'b0;\n"
    "                end else out_idx <= out_idx + 1'b1;\n"
    "            end\n"
    f"            // {MARKER} a pixel completing while busy is parked, never dropped\n"
    "            if (lib_valid_out_w && out_busy)\n"
    "                out_pend <= 1'b1;"
)


def main() -> None:
    dry = "--dry-run" in sys.argv
    files = sorted(RTL.glob("node_*.v"))
    targets = [f for f in files if "lib_valid_out_w && !out_busy" in f.read_bytes().decode("latin-1")]
    if not targets:
        print("[conv-output-pend] no target files found — nothing to do")
        sys.exit(1)
    bdir = REPO / "backups" / "conv_output_pend"
    top_text = (RTL / "nn2rtl_top.v").read_bytes().decode("latin-1")
    patched = skipped = 0
    for f in targets:
        mod = f.stem
        if f"{mod} u_{mod} (" not in top_text:
            # engine-dispatched node (bridge drives its output); module is
            # compiled but never instantiated -> streamer never runs. Skip.
            print(f"[conv-output-pend] SKIP {f.name} (not instantiated in top: engine-dispatched)")
            continue
        raw = f.read_bytes().decode("latin-1")
        eol = "\r\n" if "\r\n" in raw else "\n"
        t = raw.replace("\r\n", "\n")
        if MARKER in t:
            skipped += 1
            continue
        # invariant the one-slot proof relies on:
        if not re.search(r"stall_in\s*=\s*mac_busy\s*\|\|\s*out_busy", t):
            print(f"[conv-output-pend] FAIL {f.name}: stall_in invariant text not found")
            sys.exit(1)
        for name, old in (("decl", DECL_OLD), ("reset", RESET_OLD),
                          ("data", DATA_OLD), ("ctrl", CTRL_OLD)):
            if t.count(old) != 1:
                print(f"[conv-output-pend] FAIL {f.name}: anchor '{name}' count={t.count(old)} (expected 1)")
                sys.exit(1)
        t = (t.replace(DECL_OLD, DECL_NEW, 1)
               .replace(RESET_OLD, RESET_NEW, 1)
               .replace(DATA_OLD, DATA_NEW, 1)
               .replace(CTRL_OLD, CTRL_NEW, 1))
        if not dry:
            bdir.mkdir(parents=True, exist_ok=True)
            bak = bdir / f.name
            if not bak.exists():
                shutil.copy(f, bak)
            f.write_bytes(t.replace("\n", eol).encode("latin-1"))
        patched += 1
        print(f"[conv-output-pend] {'would patch' if dry else 'patched'} {f.name}")
    print(f"[conv-output-pend] {'DRY-RUN ' if dry else ''}done: patched={patched} already={skipped} of {len(targets)} targets")


if __name__ == "__main__":
    main()
