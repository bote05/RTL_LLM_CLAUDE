#!/usr/bin/env python3
# [FIT-FIX 2026-06-07] apply_mbv2_chanshift.py
# Opt the 17 MobileNetV2 depthwise convs into line_buf_window's CHAN_SHIFT=1 rotation scheme,
# which removes the per-tap wide C:1 channel-select MUX (the ~450K-LUT block) for the 7 FF-resident
# taps (6 shift-column + 1 bypass) and keeps the legacy small mux only for the 2 window_kwm1_wire
# (BRAM-sourced) taps. BYTE-EXACT (verified by output/mobilenet-v2/verify_chanshift_c960 [C=960] and
# verify_chanshift_c192 [C=192 stride-2], EQUIV_RESULT PASS mismatch=0), ZERO added latency, NO BRAM.
#
# Per node it makes exactly 3 edits (idempotent -- re-running is a no-op):
#   1. add `.CHAN_SHIFT(1),` as the first param of the `line_buf_window #(` instance.
#   2. add `.chan_advance(chan_issue),` right after the `.channel_select(current_global_oc),` port.
#   3. add `wire chan_issue = (state == ST_MAC) && !mac_done_issuing;` just before `endmodule`
#      (the per-channel issuing strobe: HIGH exactly the cycles the datapath consumes one channel's
#      chan_window_flat, in lockstep with current_global_oc's 0,1,...,C-1 walk).
#
# The default CHAN_SHIFT=0 path of line_buf_window.v is bit-identical to the prior mux, so ResNet
# callers (which omit the param) are UNCHANGED.

import os, re, sys

RTL = os.path.join(os.path.dirname(__file__), "..", "output", "mobilenet-v2", "rtl")
NODES = [812, 818, 824, 830, 836, 842, 848, 854, 860, 866, 872, 878, 884, 890, 896, 902, 908]

STROBE = "    // [FIT-FIX 2026-06-07] per-channel issuing strobe (ST_MAC, !mac_done_issuing) for the\n" \
         "    // line_buf_window CHAN_SHIFT rotation bank: HIGH exactly when the datapath consumes one\n" \
         "    // channel's chan_window_flat (current_global_oc walks 0,1,...,C-1 at these cycles).\n" \
         "    wire chan_issue = (state == ST_MAC) && !mac_done_issuing;\n"

def patch(path):
    src = open(path, "r", encoding="utf-8").read()
    if "CHAN_SHIFT(1)" in src and ".chan_advance(chan_issue)" in src and "wire chan_issue" in src:
        return "skip (already patched)"

    # 1. CHAN_SHIFT param: first param after `line_buf_window #(`
    if "CHAN_SHIFT(1)" not in src:
        src, n = re.subn(r"(line_buf_window\s*#\(\n)",
                         r"\1        // [FIT-FIX 2026-06-07] remove the per-tap C:1 channel-select mux\n"
                         r"        // (rotation bank, FF-resident taps). Byte-exact, no added latency, no BRAM.\n"
                         r"        .CHAN_SHIFT(1),\n", src, count=1)
        if n != 1:
            raise RuntimeError(f"{path}: could not find `line_buf_window #(`")

    # 2. chan_advance port after channel_select
    if ".chan_advance(chan_issue)" not in src:
        src, n = re.subn(r"(\.channel_select\(current_global_oc\),\n)",
                         r"\1        .chan_advance(chan_issue),\n", src, count=1)
        if n != 1:
            raise RuntimeError(f"{path}: could not find `.channel_select(current_global_oc),`")

    # 3. chan_issue strobe wire before the LAST `endmodule`
    if "wire chan_issue" not in src:
        idx = src.rfind("endmodule")
        if idx < 0:
            raise RuntimeError(f"{path}: no endmodule")
        src = src[:idx] + STROBE + "\n" + src[idx:]

    open(path, "w", encoding="utf-8").write(src)
    return "patched"

def main():
    for n in NODES:
        p = os.path.normpath(os.path.join(RTL, f"node_conv_{n}.v"))
        print(f"node_conv_{n}.v: {patch(p)}")

if __name__ == "__main__":
    main()
