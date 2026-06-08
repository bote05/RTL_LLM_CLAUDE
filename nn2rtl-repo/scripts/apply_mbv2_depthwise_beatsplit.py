#!/usr/bin/env python3
"""apply_mbv2_depthwise_beatsplit.py (2026-06-03) — Phase 2 of the over-cap-bus beat-split.

Convert the 4 over-cap DEPTHWISE blocks from flat 1-beat (>4096b) to the PROVEN 2-beat
<=4096b architecture of node_conv_884 (C=576) / node_conv_908 (C=960):
  node_conv_878 (4608b) <- byte-copy node_conv_884   (C=576 14x14 s1; MULT 19655->27056, SHIFT 20)
  node_conv_890 (4608b) <- byte-copy node_conv_884   (C=576;  MULT->8549 SHIFT 20->19; OH/OW 14->7; SH/SW 1->2)
  node_conv_896 (7680b) <- byte-copy node_conv_908   (C=960 7x7 s1; MULT->12275 SHIFT 22->19)
  node_conv_902 (7680b) <- byte-copy node_conv_908   (C=960; MULT->16987 SHIFT 22->19)

Byte-exact: the copied module IS the proven 884/908 datapath (incl. the ENABLE_BACKPRESSURE
generate branch verbatim, addressing review defect D2); only module-name, weight/bias $readmemh
paths, the requant SCALE_MULT/SCALE_SHIFT, and (890) the stride/output geometry differ, which is
exactly what distinguishes these layers. MP drops 16->4 (inherent to the proven arch; ~5% slower
on these 4 non-bottleneck layers, user-accepted). The top's instances are already structurally
identical to 884/908 (verified), so only bridge params + wire widths change in the top.

TOP edits (16): 4 gather OUT_W/OUT_BEATS, 4 scatter IN_W/IN_BEATS, 8 data_out wire widths.
Idempotent (skips if 878 wire already [4095:0]), atomic (asserts every anchor==1), backed up.
"""
import os, re, sys, time, shutil

REPO = r"C:\Users\User\Desktop\RTL_LLM_CLAUDE\nn2rtl-repo"
RTL = os.path.join(REPO, "output", "mobilenet-v2", "rtl")
TOP = os.path.join(RTL, "nn2rtl_top_engine.v")

# ---------------- TOP edits ----------------
def g(ow, ob, sp, name):  # retile_gather param substring (anchored by instance name)
    return f".OUT_W({ow}), .OUT_BEATS({ob}), .SPATIAL({sp})) {name} ("
def s(iw, ib, sp, name):  # retile_scatter param substring
    return f".IN_W({iw}), .IN_BEATS({ib}), .SPATIAL({sp})) {name} ("

TOP_EDITS = [
    # gathers feeding the depthwise (OUT_W/OUT_BEATS)
    (g(4608, 1, 196, "u_br_878"), g(4096, 2, 196, "u_br_878")),
    (g(4608, 1, 196, "u_br_890"), g(4096, 2, 196, "u_br_890")),
    (g(7680, 1, 49,  "u_br_896"), g(4096, 2, 49,  "u_br_896")),
    (g(7680, 1, 49,  "u_br_902"), g(4096, 2, 49,  "u_br_902")),
    # scatters consuming the depthwise (IN_W/IN_BEATS)
    (s(4608, 1, 196, "u_br_n4_24"), s(4096, 2, 196, "u_br_n4_24")),
    (s(4608, 1, 49,  "u_br_n4_28"), s(4096, 2, 49,  "u_br_n4_28")),
    (s(7680, 1, 49,  "u_br_n4_30"), s(4096, 2, 49,  "u_br_n4_30")),
    (s(7680, 1, 49,  "u_br_n4_32"), s(4096, 2, 49,  "u_br_n4_32")),
    # data_out wire widths (node_conv + bridge); anchor on the decl substring (comment, if any, stays)
    ("wire [4607:0] node_conv_878_data_out;", "wire [4095:0] node_conv_878_data_out;"),
    ("wire [4607:0] node_conv_890_data_out;", "wire [4095:0] node_conv_890_data_out;"),
    ("wire [7679:0] node_conv_896_data_out;", "wire [4095:0] node_conv_896_data_out;"),
    ("wire [7679:0] node_conv_902_data_out;", "wire [4095:0] node_conv_902_data_out;"),
    ("wire [4607:0] br_878_data_out;", "wire [4095:0] br_878_data_out;"),
    ("wire [4607:0] br_890_data_out;", "wire [4095:0] br_890_data_out;"),
    ("wire [7679:0] br_896_data_out;", "wire [4095:0] br_896_data_out;"),
    ("wire [7679:0] br_902_data_out;", "wire [4095:0] br_902_data_out;"),
]

# ---------------- module byte-copies ----------------
# (target, template, name_from, mult_from, mult_to, shift_from, shift_to, geom{name:(old,new)})
COPIES = [
    ("node_conv_878", "node_conv_884", "node_conv_884", 19655, 27056, None, None, {}),
    ("node_conv_890", "node_conv_884", "node_conv_884", 19655, 8549,  20,   19,   {"OH": (14, 7), "OW": (14, 7), "SH": (1, 2), "SW": (1, 2)}),
    ("node_conv_896", "node_conv_908", "node_conv_908", 30167, 12275, 22,   19,   {}),
    ("node_conv_902", "node_conv_908", "node_conv_908", 30167, 16987, 22,   19,   {}),
]

def sub1(text, pat, repl, what):
    new, n = re.subn(pat, repl, text)
    if n != 1:
        raise SystemExit(f"FAIL module sub {what}: count={n} (need 1)")
    return new

def make_module(target, template, name_from, mult_from, mult_to, shift_from, shift_to, geom):
    src = os.path.join(RTL, template + ".v")
    with open(src, "rb") as f:
        raw = f.read()
    crlf = b"\r\n" in raw
    text = raw.decode("utf-8")
    if crlf:
        text = text.replace("\r\n", "\n")
    # module name + $readmemh paths (node_conv_884_weights.hex etc.) — all occurrences
    if text.count(name_from) < 1:
        raise SystemExit(f"FAIL {target}: template name {name_from} not found")
    text = text.replace(name_from, target)
    # requant scale mult: the magnitude is a unique large constant -> replace all (localparam + comment)
    text = sub1(text, r"(localparam integer SCALE_MULT\s*=\s*)%d\b" % mult_from,
                r"\g<1>%d" % mult_to, f"{target} SCALE_MULT")
    text = text.replace(str(mult_from), str(mult_to))  # also fix the header-comment occurrence(s)
    if shift_from is not None:
        text = sub1(text, r"(localparam integer SCALE_SHIFT\s*=\s*)%d\b" % shift_from,
                    r"\g<1>%d" % shift_to, f"{target} SCALE_SHIFT")
    for nm, (old, new) in geom.items():
        text = sub1(text, r"(localparam integer %s\s*=\s*)%d\b" % (nm, old),
                    r"\g<1>%d" % new, f"{target} {nm}")
    out = text.replace("\n", "\r\n") if crlf else text
    with open(os.path.join(RTL, target + ".v"), "wb") as f:
        f.write(out.encode("utf-8"))

def main():
    with open(TOP, "rb") as f:
        raw = f.read()
    crlf = b"\r\n" in raw
    text = raw.decode("utf-8")
    if crlf:
        text = text.replace("\r\n", "\n")
    if "wire [4095:0] node_conv_878_data_out;" in text:
        print("SKIP: depthwise beat-split already applied")
        sys.exit(0)

    work = text
    for i, (old, new) in enumerate(TOP_EDITS):
        c = work.count(old)
        if c != 1:
            print(f"FAIL top edit#{i}: anchor count={c} (need 1): {old[:70]!r}")
            sys.exit(1)
        work = work.replace(old, new, 1)

    bk = os.path.join(REPO, "backups", f"mbv2_depthwise_beatsplit_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(bk, exist_ok=True)
    for fn in ("nn2rtl_top_engine.v", "node_conv_878.v", "node_conv_890.v", "node_conv_896.v", "node_conv_902.v"):
        shutil.copy2(os.path.join(RTL, fn), os.path.join(bk, fn))

    # write the 4 byte-copied modules (replaces the flat originals)
    for c in COPIES:
        make_module(*c)
    out = work.replace("\n", "\r\n") if crlf else work
    with open(TOP, "wb") as f:
        f.write(out.encode("utf-8"))
    print(f"OK: applied 16 top edits + byte-copied 4 depthwise modules (backup {bk})")

if __name__ == "__main__":
    main()
