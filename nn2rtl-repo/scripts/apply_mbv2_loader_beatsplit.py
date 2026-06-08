#!/usr/bin/env python3
"""apply_mbv2_loader_beatsplit.py (2026-06-03) — Phase 1 of the over-cap-bus beat-split.

Tile the 6 over-cap LOADER-GATHER buses so no data wire exceeds 4096 bits:
  u_br_ldr22/24/26 : OUT_W=6144 OUT_BEATS=1 -> OUT_W=2048 OUT_BEATS=3 (576ch = 3x256ch words)
  u_br_ldr28/30/32 : OUT_W=8192 OUT_BEATS=1 -> OUT_W=2048 OUT_BEATS=4 (960ch = 4x256ch words)
  matching loader BUS_W 6144/8192 -> 2048 (selects the existing g_w_eq 1-word-per-beat branch).

Byte-exact: each gather beat == exactly one 2048b BRAM word; the loader writes the identical
2048b slices to the identical contiguous addresses (base+0,base+1,...), TOTAL_BRAM_WORDS unchanged.
NO module change (g_w_eq is multi-beat-capable). Beat-splitting only re-times the same bytes.

Idempotent (skips if already 2048), atomic (asserts every anchor==1 before writing), backed up.
"""
import os, sys, time, shutil

REPO = r"C:\Users\User\Desktop\RTL_LLM_CLAUDE\nn2rtl-repo"
TOP = os.path.join(REPO, "output", "mobilenet-v2", "rtl", "nn2rtl_top_engine.v")

def gth(ow, ob, sp, name):
    return f".OUT_W({ow}), .OUT_BEATS({ob}), .SPATIAL({sp})) {name} ("

def busblk(bw, base, tot, name):
    return (f"        .BUS_W({bw}),\n        .BRAM_BASE_ADDR({base}),\n"
            f"        .TOTAL_BRAM_WORDS({tot})\n    ) {name} (")

EDITS = [
    # --- gather OUT_W / OUT_BEATS (anchored by instance name) ---
    (gth(6144, 1, 196, "u_br_ldr22"), gth(2048, 3, 196, "u_br_ldr22")),
    (gth(6144, 1, 196, "u_br_ldr24"), gth(2048, 3, 196, "u_br_ldr24")),
    (gth(6144, 1, 49,  "u_br_ldr26"), gth(2048, 3, 49,  "u_br_ldr26")),
    (gth(8192, 1, 49,  "u_br_ldr28"), gth(2048, 4, 49,  "u_br_ldr28")),
    (gth(8192, 1, 49,  "u_br_ldr30"), gth(2048, 4, 49,  "u_br_ldr30")),
    (gth(8192, 1, 49,  "u_br_ldr32"), gth(2048, 4, 49,  "u_br_ldr32")),
    # --- gather output wire widths ---
    ("wire [6143:0] br_ldr22_data_out;", "wire [2047:0] br_ldr22_data_out;"),
    ("wire [6143:0] br_ldr24_data_out;", "wire [2047:0] br_ldr24_data_out;"),
    ("wire [6143:0] br_ldr26_data_out;", "wire [2047:0] br_ldr26_data_out;"),
    ("wire [8191:0] br_ldr28_data_out;", "wire [2047:0] br_ldr28_data_out;"),
    ("wire [8191:0] br_ldr30_data_out;", "wire [2047:0] br_ldr30_data_out;"),
    ("wire [8191:0] br_ldr32_data_out;", "wire [2047:0] br_ldr32_data_out;"),
    # --- loader BUS_W (anchored by the 4-line block incl. instance name; base/total UNCHANGED) ---
    (busblk(6144, 4096, 588, "u_ldr_node_conv_880"), busblk(2048, 4096, 588, "u_ldr_node_conv_880")),
    (busblk(6144, 4096, 588, "u_ldr_node_conv_886"), busblk(2048, 4096, 588, "u_ldr_node_conv_886")),
    (busblk(6144, 4096, 147, "u_ldr_node_conv_892"), busblk(2048, 4096, 147, "u_ldr_node_conv_892")),
    (busblk(8192, 0,    196, "u_ldr_node_conv_898"), busblk(2048, 0,    196, "u_ldr_node_conv_898")),
    (busblk(8192, 0,    196, "u_ldr_node_conv_904"), busblk(2048, 0,    196, "u_ldr_node_conv_904")),
    (busblk(8192, 0,    196, "u_ldr_node_conv_910"), busblk(2048, 0,    196, "u_ldr_node_conv_910")),
]

def main():
    with open(TOP, "rb") as f:
        raw = f.read()
    crlf = b"\r\n" in raw
    text = raw.decode("utf-8")
    if crlf:
        text = text.replace("\r\n", "\n")

    if gth(2048, 3, 196, "u_br_ldr22") in text:
        print("SKIP: loader beat-split already applied (u_br_ldr22 OUT_W=2048)")
        sys.exit(0)

    work = text
    for i, (old, new) in enumerate(EDITS):
        c = work.count(old)
        if c != 1:
            print(f"FAIL edit#{i}: anchor count={c} (need 1): {old[:70]!r}")
            sys.exit(1)
        work = work.replace(old, new, 1)

    bk = os.path.join(REPO, "backups", f"mbv2_loader_beatsplit_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(bk, exist_ok=True)
    shutil.copy2(TOP, os.path.join(bk, os.path.basename(TOP)))
    out = work.replace("\n", "\r\n") if crlf else work
    with open(TOP, "wb") as f:
        f.write(out.encode("utf-8"))
    print(f"OK: applied {len(EDITS)} loader beat-split edits (backup {bk})")

if __name__ == "__main__":
    main()
