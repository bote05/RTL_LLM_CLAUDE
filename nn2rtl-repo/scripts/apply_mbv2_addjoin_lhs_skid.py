#!/usr/bin/env python3
"""apply_mbv2_addjoin_lhs_skid.py (2026-06-03) — harden the 4 final-block residual joins.

Root cause of the depthwise beat-split failure: node_add_828/900/1038/1110 have a BUFFERED skip
(RHS) but an UNBUFFERED main path (LHS = live node_conv_880/886/898/904). The join pairs
skip[i] with the LIVE main beat, gated by spatial_run -> any main-path LATENCY change
(depthwise beat-split / MP) de-syncs it (dup/drop a beat) -> wrong values / wedge.
node_add_198 already solved this with a symmetric u_lhs_ buffer; replicate that for the 4.

Per add: insert an LHS skip_fifo (frame-sized, == its skip FIFO depth) buffering the main conv,
repoint the skip FIFO out_ready + the add valid_in/data_in to the buffered lhs. Byte-exact:
transparent in-order buffer; the add still pairs (lhs[i], skip[i]) -- only pop timing changes.
This is GOOD on its own (hardening) and is the prerequisite for the depthwise beat-split.

Idempotent (skips if u_lhs_node_add_828 present), atomic (asserts every anchor==1), backed up.
"""
import os, sys, time, shutil

REPO = r"C:\Users\User\Desktop\RTL_LLM_CLAUDE\nn2rtl-repo"
TOP = os.path.join(REPO, "output", "mobilenet-v2", "rtl", "nn2rtl_top_engine.v")

# (add_num, main_conv_src, width, depth)
ADDS = [
    (828, 880, 768, 256),
    (900, 886, 768, 256),
    (1038, 898, 1280, 64),
    (1110, 904, 1280, 64),
]

def lhs_fifo(n, src, w, d):
    return (
        f"    // [LHS-SKID] symmetric LHS buffer on the node_conv_{src} main arm (mirrors "
        f"u_skip_node_add_{n} + u_lhs_node_add_198).\n"
        f"    // Makes the residual join fully elastic so a main-path latency change (depthwise "
        f"beat-split / MP) cannot de-sync it.\n"
        f"    // Byte-exact: transparent in-order buffer; the add still pairs (lhs[i], skip[i]) "
        f"-- only pop TIMING changes.\n"
        f"    wire node_add_{n}_lhs_valid;\n"
        f"    wire [{w-1}:0] node_add_{n}_lhs_data;\n"
        f"    wire node_add_{n}_lhs_in_ready;\n"
        f"    skip_fifo #(.WIDTH({w}), .DEPTH({d})) u_lhs_node_add_{n} (\n"
        f"        .clk(clk), .rst_n(rst_n),\n"
        f"        .in_valid(node_conv_{src}_valid_out & spatial_run & node_add_{n}_lhs_in_ready),\n"
        f"        .in_data(node_conv_{src}_data_out[{w-1}:0]),\n"
        f"        .in_ready(node_add_{n}_lhs_in_ready),\n"
        f"        .out_valid(node_add_{n}_lhs_valid),\n"
        f"        .out_data(node_add_{n}_lhs_data),\n"
        f"        .out_ready(node_add_{n}_ready_in & node_add_{n}_skip_valid & spatial_run)\n"
        f"    );\n"
    )

def build_edits():
    edits = []
    for n, src, w, d in ADDS:
        # 1) skip FIFO out_ready: gate on buffered lhs_valid instead of the live main conv;
        #    AND append the new LHS FIFO right after the skip FIFO's closing ");".
        old_or = (f"        .out_ready(node_add_{n}_ready_in & node_conv_{src}_valid_out & spatial_run)\n"
                  f"    );")
        new_or = (f"        .out_ready(node_add_{n}_ready_in & node_add_{n}_lhs_valid & spatial_run)\n"
                  f"    );\n\n" + lhs_fifo(n, src, w, d))
        edits.append((old_or, new_or))
        # 2) add instance valid_in
        edits.append((f".valid_in(node_conv_{src}_valid_out & node_add_{n}_skip_valid & spatial_run)",
                      f".valid_in(node_add_{n}_lhs_valid & node_add_{n}_skip_valid & spatial_run)"))
        # 3) add instance data_in
        edits.append((f".data_in({{node_conv_{src}_data_out[{w-1}:0], node_add_{n}_skip_data}})",
                      f".data_in({{node_add_{n}_lhs_data, node_add_{n}_skip_data}})"))
        # 4) CRITICAL: repoint the engine-output-bridge backpressure (ready_out) for the main conv
        #    from the ADD to the new LHS FIFO's in_ready (mirrors node_add_198 / conv_826:
        #    .ready_out((node_add_198_lhs_in_ready & spatial_run))). Without this the engine bridge
        #    advances on the add's terms while the FIFO captures on its own -> wedge/deadlock.
        edits.append((f".ready_out((node_add_{n}_ready_in & node_add_{n}_skip_valid & spatial_run))",
                      f".ready_out((node_add_{n}_lhs_in_ready & spatial_run))"))
    return edits

def main():
    with open(TOP, "rb") as f:
        raw = f.read()
    crlf = b"\r\n" in raw
    text = raw.decode("utf-8")
    if crlf:
        text = text.replace("\r\n", "\n")
    if "u_lhs_node_add_828" in text:
        print("SKIP: add-join lhs skids already applied")
        sys.exit(0)
    work = text
    for i, (old, new) in enumerate(build_edits()):
        c = work.count(old)
        if c != 1:
            print(f"FAIL edit#{i}: anchor count={c} (need 1): {old[:72]!r}")
            sys.exit(1)
        work = work.replace(old, new, 1)
    bk = os.path.join(REPO, "backups", f"mbv2_addjoin_lhs_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(bk, exist_ok=True)
    shutil.copy2(TOP, os.path.join(bk, os.path.basename(TOP)))
    out = work.replace("\n", "\r\n") if crlf else work
    with open(TOP, "wb") as f:
        f.write(out.encode("utf-8"))
    print(f"OK: added lhs skids to node_add_828/900/1038/1110 (backup {bk})")

if __name__ == "__main__":
    main()
