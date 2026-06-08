#!/usr/bin/env python3
"""apply_mbv2_depthwise_tiled_linebuf.py (2026-06-06)

MobileNet-v2 FIT fix: enable line_buf_window TILE_STORAGE=32 (deep-narrow per-slot
storage, 32 ch/tile) on the depthwise 3x3 convs C>=96 so their per-slot line buffers
map to RAMB36 deep-narrow tiles instead of wide-bound BRAM. line_buf_window.v already
implements TILE_STORAGE (default 0 = legacy, byte-exact); this script wires it on per
node. The burst-serialized tiled R/W stalls the node scheduler via a new mem_busy port
folded into stall_in -> atomic window read -> byte-exact vs legacy (TILE_STORAGE=0).
Verified byte-exact on node_conv_896 (C=960) by verify_lbw_c960/tb_equiv (EQUIV PASS).

Targets (depthwise 3x3, C>=96): 818 824 830 836 842 848 854 860 866 872 878 884 890 902 908.
SKIPPED: node_conv_896 (already patched / template), node_conv_810 (stem), node_conv_812 (C=32).

Per-file edits (idempotent, atomic-per-file, backed up first):
  1. lbw param list: append ", .TILE_STORAGE(32)" after the last param .LINE_BUF_USE_URAM(0).
  2. lbw port list:  add ".mem_busy(lbw_mem_busy)" after the .window_flat() port.
  3. declare "wire lbw_mem_busy;" just before the "wire stall_in = mac_busy || skid_block;"
     line, and OR lbw_mem_busy into that stall_in.

Idempotent: a file already containing `lbw_mem_busy` is skipped. latin-1 encoding,
CRLF/LF preserved. Each edit anchor must appear EXACTLY once or the file is left
untouched (atomic).
"""
import sys, os, time, shutil

REPO = r"C:\Users\User\Desktop\RTL_LLM_CLAUDE\nn2rtl-repo"
RTL = os.path.join(REPO, "output", "mobilenet-v2", "rtl")
TARGETS = [818, 824, 830, 836, 842, 848, 854, 860, 866, 872, 878, 884, 890, 902, 908]

# (old, new) edits. Anchors use LF newlines; the file is normalized to LF before
# matching and restored to its original lineending on write.
EDITS = [
    # 1) lbw param list -- append TILE_STORAGE(32) after the last param
    ("        .LINE_BUF_USE_URAM(0)\n    ) lbw (",
     "        .LINE_BUF_USE_URAM(0),\n"
     "        // [FIT-FIX 2026-06-06] deep-narrow tiled per-slot storage: 32 ch/tile.\n"
     "        // Burst-serialized R/W stalls the scheduler via mem_busy -> atomic ->\n"
     "        // byte-exact vs legacy (TILE_STORAGE=0). Verified by verify_lbw_c960/tb_equiv.\n"
     "        .TILE_STORAGE(32)\n"
     "    ) lbw ("),

    # 2) lbw port list -- add mem_busy after window_flat()
    ("        .window_flat()\n    );",
     "        .window_flat(),\n"
     "        .mem_busy(lbw_mem_busy)\n"
     "    );"),

    # 3) declare lbw_mem_busy and OR it into stall_in
    ("    wire stall_in = mac_busy || skid_block;",
     "    // [FIT-FIX 2026-06-06] line_buf_window tiled-storage burst stall (TILE_STORAGE>0).\n"
     "    wire lbw_mem_busy;\n"
     "    wire stall_in = mac_busy || skid_block || lbw_mem_busy;"),
]


def apply_file(path, bk):
    name = os.path.basename(path)
    with open(path, "rb") as f:
        raw = f.read()
    crlf = b"\r\n" in raw
    text = raw.decode("latin-1")
    if crlf:
        text = text.replace("\r\n", "\n")

    # idempotency
    if "lbw_mem_busy" in text:
        print(f"SKIP {name}: already has lbw_mem_busy")
        return True, False

    # assert all anchors exist exactly once BEFORE mutating
    for i, (old, new) in enumerate(EDITS):
        c = text.count(old)
        if c != 1:
            print(f"FAIL {name} edit#{i}: anchor count={c} (need 1): {old[:50]!r}")
            return False, False

    work = text
    for old, new in EDITS:
        work = work.replace(old, new, 1)

    shutil.copy2(path, os.path.join(bk, name))
    out = work.replace("\n", "\r\n") if crlf else work
    with open(path, "wb") as f:
        f.write(out.encode("latin-1"))
    print(f"OK   {name}: applied {len(EDITS)} edits")
    return True, True


def main():
    bk = os.path.join(REPO, "backups", f"depthwise_tiled_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(bk, exist_ok=True)
    ok = True
    changed = []
    for n in TARGETS:
        path = os.path.join(RTL, f"node_conv_{n}.v")
        if not os.path.isfile(path):
            print(f"FAIL node_conv_{n}.v: file not found")
            ok = False
            continue
        good, did = apply_file(path, bk)
        ok &= good
        if did:
            changed.append(n)
    print("-" * 50)
    print(f"backup dir: {bk}")
    print(f"changed ({len(changed)}): {changed}")
    print("ALL OK" if ok else "FAILED (see above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
