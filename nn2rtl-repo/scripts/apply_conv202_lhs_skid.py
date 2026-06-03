#!/usr/bin/env python3
"""MP-increase deadlock fix (PREP — does NOT build): add a skip_fifo on the
conv_202 -> node_add LHS path so it is buffered SYMMETRICALLY with the RHS
(conv_204 -> u_skip_node_add, DEPTH=512).

ROOT CAUSE (workflow wza06rbrh, 2026-05-30): node_add is a synchronized 2-stream
join (8 beats/pixel, advances only when BOTH inputs valid the same cycle). The
RHS arm (conv_204) is buffered by a 512-deep skip_fifo (u_skip_node_add); the LHS
arm (conv_202) drains DIRECTLY into the add with no buffer. At MP=16 the arms stay
beat-locked; at MP=32 conv_202 produces ~2x faster, its valid_out PHASE slips vs
the RHS fifo, one beat de-syncs, and the join wedges permanently (circular
hold-off). Giving the LHS the same elastic slack the RHS already has absorbs the
phase slip.

BYTE-EXACT: skip_fifo is a transparent in-order elastic buffer (in_ready=~full,
out_valid=~empty, FIFO order preserved). It changes only WHEN each beat arrives,
never its value or order. It byte-for-byte mirrors the existing RHS u_skip_node_add.
So results are identical; only timing/handshake changes.

This patch is IDEMPOTENT and REVERSIBLE (writes a backup; --revert restores it).
It does NOT rebuild — run the e2e byte-exact gate yourself after.

USAGE:
  python scripts/apply_conv202_lhs_skid.py --dry-run   # show the planned edit
  python scripts/apply_conv202_lhs_skid.py             # apply (with backup)
  python scripts/apply_conv202_lhs_skid.py --revert     # restore from backup
"""
from __future__ import annotations
import sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOP = ROOT / "output/rtl/nn2rtl_top.v"
BK = ROOT / "backups/conv202_lhs_skid_20260530"
BK.mkdir(parents=True, exist_ok=True)
BKFILE = BK / "nn2rtl_top.v"

DRY = "--dry-run" in sys.argv
REVERT = "--revert" in sys.argv

# ---- the conv_202 instantiation as it currently stands (LHS drains directly) ----
OLD_CONV202 = """node_conv_202 u_node_conv_202 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(skid_node_conv_202_valid & spatial_run),
        .ready_in(node_conv_202_ready_in),
        .data_in(skid_node_conv_202_data),
        .valid_out(node_conv_202_valid_out),
        .ready_out(node_add_skip_valid & spatial_run & node_add_ready_in),
        .data_out(node_conv_202_data_out)
    );"""

# ---- replacement: conv_202 now drains into a NEW lhs skip_fifo (symmetric w/ rhs) ----
# The conv's ready_out is gated by the new fifo's in_ready (so the conv only
# advances when the lhs fifo can accept), instead of the combinational join tie.
NEW_CONV202 = """node_conv_202 u_node_conv_202 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(skid_node_conv_202_valid & spatial_run),
        .ready_in(node_conv_202_ready_in),
        .data_in(skid_node_conv_202_data),
        .valid_out(node_conv_202_valid_out),
        .ready_out(node_add_main_in_ready & spatial_run),
        .data_out(node_conv_202_data_out)
    );

    // [MP-DEADLOCK FIX 2026-05-30 apply_conv202_lhs_skid.py] LHS elastic buffer,
    // symmetric with the RHS u_skip_node_add (DEPTH=512). conv_202 (the 1x1 MAIN
    // expand) drains here instead of directly into the join; this absorbs the
    // valid_out phase slip that wedges the synchronized add when MP=32 makes
    // conv_202 produce ~2x faster than the RHS. Byte-exact: FIFO preserves beat
    // value+order, changes only timing.
    wire node_add_main_in_ready;
    wire node_add_main_valid;
    wire [255:0] node_add_main_data;
    skip_fifo #(.WIDTH(256), .DEPTH(512)) u_skip_node_add_main (
        .clk(clk), .rst_n(rst_n),
        .in_valid(node_conv_202_valid_out & spatial_run & node_add_main_in_ready),
        .in_data(node_conv_202_data_out[255:0]),
        .in_ready(node_add_main_in_ready),
        .out_valid(node_add_main_valid),
        .out_data(node_add_main_data),
        .out_ready(node_add_ready_in & node_add_skip_valid & spatial_run)
    );"""

# ---- the node_add join: consume the buffered lhs instead of the raw conv_202 ----
OLD_ADD = """    node_add u_node_add (
        .clk(clk), .rst_n(rst_n),
        .valid_in(node_conv_202_valid_out & node_add_skip_valid & spatial_run),
        .ready_in(node_add_ready_in),
        .data_in({node_add_skip_data, node_conv_202_data_out[255:0]}),
        .valid_out(node_add_valid_out),
        .ready_out(skid_node_relu_3_ready & spatial_run),   // [BP-FIX] hold output until accepted
        .data_out(node_add_data_out)
    );"""

NEW_ADD = """    node_add u_node_add (
        .clk(clk), .rst_n(rst_n),
        .valid_in(node_add_main_valid & node_add_skip_valid & spatial_run),
        .ready_in(node_add_ready_in),
        .data_in({node_add_skip_data, node_add_main_data}),
        .valid_out(node_add_valid_out),
        .ready_out(skid_node_relu_3_ready & spatial_run),   // [BP-FIX] hold output until accepted
        .data_out(node_add_data_out)
    );"""

# ---- the RHS skip_fifo out_ready: was gated by raw conv_202_valid_out; now by
#      the buffered lhs valid (node_add_main_valid) so both arms pop together. ----
OLD_RHS = ".out_ready(node_add_ready_in & node_conv_202_valid_out & spatial_run)"
NEW_RHS = ".out_ready(node_add_ready_in & node_add_main_valid & spatial_run)"


def main():
    if REVERT:
        if not BKFILE.exists():
            print(f"ERROR: no backup at {BKFILE} — nothing to revert"); sys.exit(1)
        shutil.copy(BKFILE, TOP)
        print(f"[revert] restored {TOP} from {BKFILE}"); return

    txt = TOP.read_text()

    # idempotency guard
    if "u_skip_node_add_main" in txt:
        print("[skip] already patched (u_skip_node_add_main present). Use --revert to undo."); return

    # locate all three anchors exactly once each
    for name, old in [("conv_202 inst", OLD_CONV202), ("node_add inst", OLD_ADD), ("rhs out_ready", OLD_RHS)]:
        c = txt.count(old)
        if c != 1:
            print(f"ERROR: anchor '{name}' found {c} times (expected 1). RTL drifted — aborting, no change."); sys.exit(1)

    new = txt.replace(OLD_CONV202, NEW_CONV202).replace(OLD_ADD, NEW_ADD).replace(OLD_RHS, NEW_RHS)

    if DRY:
        print("=== DRY RUN — planned edits (all 3 anchors matched exactly once) ===")
        print("  1. conv_202.ready_out -> node_add_main_in_ready & spatial_run; insert u_skip_node_add_main (DEPTH=512)")
        print("  2. node_add.valid_in/data_in -> buffered lhs (node_add_main_valid / node_add_main_data)")
        print("  3. u_skip_node_add.out_ready -> gated by node_add_main_valid (was raw node_conv_202_valid_out)")
        print(f"\n  net line delta: +{new.count(chr(10)) - txt.count(chr(10))} lines")
        print("  NOT applied (dry-run). NOT built.")
        return

    if not BKFILE.exists():
        shutil.copy(TOP, BKFILE)
        print(f"[backup] {TOP} -> {BKFILE}")
    TOP.write_text(new, newline="\n")
    print(f"[ok] patched {TOP} (+{new.count(chr(10))-txt.count(chr(10))} lines). NOT built.")
    print("NEXT: rebuild + e2e byte-exact gate (run_nn2rtl_top_value.ts 0 AND 1, both mismatch_bytes=0).")
    print("      This fix alone is for MP=16 (must stay byte-exact). To test the cycle win, ALSO apply")
    print("      MP=32 (apply_mp32.py) + regen + the analogous skid on the OTHER asymmetric add joins.")


if __name__ == "__main__":
    main()
