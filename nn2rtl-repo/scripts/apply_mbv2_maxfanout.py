#!/usr/bin/env python3
"""FMAX/CONGESTION 2026-06-08: cap fanout on the genuine high-fanout broadcast nets in the
17 MobileNetV2 depthwise convs.

The placed-checkpoint fanout report (mbv2_analysis_fanout.rpt) found the REAL high-fanout
nets are NOT node_linear (a red herring) but the per-conv control broadcasts:
  - start_pulse  : 47,071 fanout on conv_896/902/908 (28,277 on 878/884/890) -- the frame_start
    pulse that clears the huge per-channel window shift-register array in line_buf_window.
  - current_global_oc : ~25,453 fanout -- the channel-select net feeding the C:1 window mux,
    weight address, and the MAC lane.
A single driver reaching 25-47K loads forces long, congestion-inducing broadcast routes.
`(* max_fanout = N *)` tells Vivado to REPLICATE the driver so each replica drives a local
subset -> short local routes -> less congestion. This is a SYNTHESIS ATTRIBUTE only: Verilator
ignores it, so module output is BIT-IDENTICAL (byte-exact by construction). Replicas add a few
hundred FFs/LUTs total -> fit stays green (LUT 78.1%, FF 38.5%). NOT chan-shift (which added
per-FF rotate control muxes = +210K LUT and was rejected); max_fanout only clones the driver.

Idempotent + backs up each original. No regen, no golden change (attribute-only).
"""
from __future__ import annotations
import re, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RTL = REPO / "output" / "mobilenet-v2" / "rtl"
BACKUP_DIR = REPO / "backups" / "mbv2_maxfanout_20260608"
MAXFAN = 256

DEPTHWISE = [812, 818, 824, 830, 836, 842, 848, 854, 860, 866,
             872, 878, 884, 890, 896, 902, 908]

# start_pulse is declared bundled: "reg started, start_pulse, pending_rearm;"
SP_OLD = "    reg started, start_pulse, pending_rearm;\n"
SP_NEW = ("    reg started, pending_rearm;\n"
          f"    (* max_fanout = {MAXFAN} *) reg start_pulse;\n")

# current_global_oc is "wire [W:0] current_global_oc = ...;" (W varies per conv channel count)
CGO_RE = re.compile(r"^(\s*)wire (\[\d+:0\] current_global_oc\s*=)", re.MULTILINE)


def main() -> int:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    patched, skipped = 0, 0
    for n in DEPTHWISE:
        f = RTL / f"node_conv_{n}.v"
        if not f.exists():
            print(f"[maxfanout] FATAL: {f.relative_to(REPO)} missing", file=sys.stderr); return 2
        src = f.read_text(encoding="utf-8")
        if "max_fanout" in src:
            print(f"[maxfanout] node_conv_{n}: already applied; skip"); skipped += 1; continue
        orig = src
        # start_pulse
        if src.count(SP_OLD) == 1:
            src = src.replace(SP_OLD, SP_NEW, 1)
        else:
            print(f"[maxfanout] WARN node_conv_{n}: start_pulse anchor count={src.count(SP_OLD)} (skipping start_pulse)", file=sys.stderr)
        # current_global_oc (regex; insert attribute before the wire)
        cgo_hits = len(CGO_RE.findall(src))
        if cgo_hits == 1:
            src = CGO_RE.sub(rf"\1(* max_fanout = {MAXFAN} *) wire \2", src, count=1)
        else:
            print(f"[maxfanout] WARN node_conv_{n}: current_global_oc hits={cgo_hits} (skipping cgo)", file=sys.stderr)
        if src == orig:
            print(f"[maxfanout] node_conv_{n}: no change (anchors not found)", file=sys.stderr); continue
        shutil.copy2(f, BACKUP_DIR / f.name)
        f.write_text(src, encoding="utf-8", newline="\n")
        patched += 1
        print(f"[maxfanout] node_conv_{n}: max_fanout={MAXFAN} applied")
    print(f"[maxfanout] done: {patched} patched, {skipped} already (backups in {BACKUP_DIR.relative_to(REPO)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
