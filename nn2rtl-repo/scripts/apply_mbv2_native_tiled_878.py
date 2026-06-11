#!/usr/bin/env python3
"""
apply_mbv2_native_tiled_878.py

NATIVE-256b-TILED re-architecture of node_conv_878 in the engine top:

    output/mobilenet-v2/rtl/nn2rtl_top_engine.v   (default target)

WHY
---
node_conv_878 (depthwise, C=576) sits on the #1 congestion/area driver: a pair of
retile bridges that round-trip its native-256b neighbours through a wide 4096b /
2-beat port:

    relu n4_23 (256b tiled) --> retile_gather u_br_878 (18 tiles -> 4096b x2)
        --> node_conv_878 (2 beats -> 4608b pixel -> line_buf_window)
        --> 2-beat splitter (4608b -> 4096b x2) --> retile_scatter u_br_n4_24
        --> relu n4_24 (256b tiled)

n4_23, node_conv_878 and n4_24 are ALL ENABLE_BACKPRESSURE=1, so they honour
ready/valid in lockstep. node_conv_878 has been re-architected with a param-gated
NATIVE_TILED=1 path that talks 18x256b tiles DIRECTLY to n4_23 / n4_24 (an INTERNAL
18-tile gather still assembles the full 4608b pixel for line_buf_window, which
re-tiles it for BRAM storage; only the EXTERNAL bridges + the 4096b 2-beat adapter
are removed). Byte layout is contiguous (tile k = channels k*32..k*32+31 =
wide[k*256+:256]) so it is logical-pixel-identical = BYTE-EXACT.

THIS PATCH (idempotent, atomic, backs up first)
------------------------------------------------
(a) DELETE the u_br_878 retile_gather instance + the u_br_n4_24 retile_scatter
    instance and their 5-line wire-decl groups (br_878_*, br_n4_24_*).
(b) DELETE the now-dead spatial_run_drain_br_878 / spatial_run_drain_br_n4_24
    wires (they reference the deleted bridges' stall_out).
(c) Narrow `wire [4095:0] node_conv_878_data_out;` -> `[255:0]`.
(d) REMOVE br_878_stall_out + br_n4_24_stall_out from the any_retile_stall OR.
(e) RE-WIRE the three instances to talk native-tiled DIRECTLY, bridgeless:
      n4_23  : .out_ready_in(node_conv_878_ready_in)   (was br_878_wr_accept)
      conv878: #(.ENABLE_BACKPRESSURE(1), .NATIVE_TILED(1))
               .valid_in_t(n4_23_valid_out), .ready_in_t(node_conv_878_ready_in),
               .data_in_t(n4_23_data_out),
               .out_ready_in_t(n4_24_ready_in),
               .valid_out_t(node_conv_878_valid_out),
               .data_out_t(node_conv_878_data_out)
               (legacy wide ports left UNCONNECTED -> tied off inside the module)
      n4_24  : .valid_in(node_conv_878_valid_out), .data_in(node_conv_878_data_out)
               (was br_n4_24_valid_out & spatial_run_drain_br_n4_24 / br_n4_24_data_out)

DEADLOCK SAFETY (proven; see the design + retile_bridge.v THE INVARIANT)
------------------------------------------------------------------------
Both hops enforce advance-iff-latch by SHARING the same ready boolean, RAW (no
spatial_run), exactly like the wave-2 UNGATE of bridgeless free-running-producer
hops (apply_mbv2_wave2_bridges.py step 6):
  INPUT  edge: n4_23 advances iff out_ready_in == node_conv_878_ready_in; the conv
               latches a tile iff (valid_in_t & ready_in_t == node_conv_878_ready_in).
               SAME boolean, same cycle -> no lost tile.
  OUTPUT edge: the conv advances a tile iff (valid_out_t & out_ready_in_t ==
               n4_24_ready_in); n4_24 latches iff (valid_in & ready_in ==
               n4_24_ready_in). SAME boolean, same cycle -> no lost/dup tile.
These hops have NO engine dispatch; spatial_run only ever drops here for a
transient any_retile_stall (some OTHER bridge full) -> that is exactly the
lost-beat trigger the wave-2 UNGATE removes, so valid_in_t / n4_24.valid_in are
wired RAW (no & spatial_run). Backpressure is the shared ready boolean
(sched_ready_in / n4_24.ready_in) plus n4_23 keeping "& spatial_run" on its OWN
new-pixel start (unchanged).

Idempotent: re-running is a no-op once the bridges are gone and the native wiring
is in place. Atomic: every anchor is asserted to match exactly the expected count
BEFORE any write; on any mismatch the file is left untouched.
"""
import argparse
import os
import re
import shutil
import sys

DEFAULT_TOP = r"D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/rtl/nn2rtl_top_engine.v"

NATIVE_MARK = "// [NATIVE_TILED_878] node_conv_878 bridgeless native-256b re-arch applied"


def find_inst(text, mid):
    """Match a full `<mid> [#(...)] u_<mid> ( ... );` instance block (with an
    optional parameter override list between the module name and the instance)."""
    m = re.search(re.escape(mid) + r"\s+(?:#\([^\n]*\)\s+)?u_" + re.escape(mid)
                  + r"\s*\((?:.|\n)*?\n\s*\);", text)
    if not m:
        raise RuntimeError(f"instance u_{mid} not found")
    return m


def find_retile_inst(text, kind, inst):
    """Match a full `retile_<kind> #(...) u_<inst> ( ... );` block."""
    m = re.search(r"retile_" + re.escape(kind) + r"\s+#\([^\n]*\)\s+u_" + re.escape(inst)
                  + r"\s*\((?:.|\n)*?\n\s*\);\n", text)
    if not m:
        raise RuntimeError(f"retile_{kind} u_{inst} not found")
    return m


def delete_wire_group(text, base):
    """Delete the 5-line wire-decl group for a bridge base name (br_878 / br_n4_24)."""
    # Match the contiguous block of 5 decls starting at `_valid_out`.
    pat = (r"[ \t]*wire " + re.escape(base) + r"_valid_out;\n"
           r"[ \t]*wire \[\d+:0\] " + re.escape(base) + r"_data_out;\n"
           r"[ \t]*wire " + re.escape(base) + r"_ready_out;[^\n]*\n"
           r"[ \t]*wire " + re.escape(base) + r"_stall_out;\n"
           r"[ \t]*wire " + re.escape(base) + r"_wr_accept;[^\n]*\n")
    new, n = re.subn(pat, "", text, count=1)
    if n != 1:
        raise RuntimeError(f"wire group for {base} not matched (n={n})")
    return new


def delete_drain_wire(text, base):
    """Delete the `wire spatial_run_drain_<base> = ...;` line (references deleted stall_out)."""
    pat = r"[ \t]*wire spatial_run_drain_" + re.escape(base) + r" = [^\n;]*;\n"
    new, n = re.subn(pat, "", text, count=1)
    if n != 1:
        raise RuntimeError(f"spatial_run_drain_{base} wire not matched (n={n})")
    return new


def remove_stall_terms(text, terms):
    """Remove the given `<term>` from the any_retile_stall OR-tree."""
    m = re.search(r"wire any_retile_stall = ([^\n;]*);", text)
    if not m:
        raise RuntimeError("any_retile_stall assignment not found")
    expr = m.group(1)
    parts = [p.strip() for p in expr.split("|")]
    before = len(parts)
    parts = [p for p in parts if p not in terms]
    removed = before - len(parts)
    if removed != len(terms):
        # Idempotent: maybe already removed.
        if all(t not in expr for t in terms):
            return text
        raise RuntimeError(
            f"expected to remove {len(terms)} stall terms, removed {removed} "
            f"(terms={terms}, expr={expr!r})")
    new_expr = " | ".join(parts)
    return text[:m.start(1)] + new_expr + text[m.end(1):]


def patch_n4_23(text):
    """n4_23.out_ready_in: br_878_wr_accept -> node_conv_878_ready_in."""
    m = find_inst(text, "n4_23")
    blk = m.group(0)
    # Replace ONLY the connected net; preserve any trailing comma/comment exactly.
    nb, n = re.subn(r"\.out_ready_in\(br_878_wr_accept\)",
                    ".out_ready_in(node_conv_878_ready_in)",
                    blk, count=1)
    if n != 1:
        if "node_conv_878_ready_in" in blk:
            return text  # already patched
        raise RuntimeError(".out_ready_in(br_878_wr_accept) not found in u_n4_23")
    return text[:m.start()] + nb + text[m.end():]


CONV878_NATIVE = """node_conv_878 #(.ENABLE_BACKPRESSURE(1), .NATIVE_TILED(1)) u_node_conv_878 (
.clk(clk), .rst_n(rst_n),
        // [NATIVE_TILED_878] bridgeless native 256b tiled ports wired DIRECTLY
        // n4_23 -> 878 -> n4_24 (legacy wide ports unconnected -> tied off inside).
        // RAW valid (no & spatial_run): advance-iff-latch via the shared ready
        // boolean (node_conv_878_ready_in on the input edge, n4_24_ready_in on the
        // output edge); see retile_bridge.v THE INVARIANT and apply_mbv2_native_tiled_878.py.
        .valid_in_t(n4_23_valid_out),
        .ready_in_t(node_conv_878_ready_in),
        .data_in_t(n4_23_data_out),
        .out_ready_in_t(n4_24_ready_in),
        .valid_out_t(node_conv_878_valid_out),
        .data_out_t(node_conv_878_data_out)
    );"""


def patch_conv878(text):
    """Replace the whole u_node_conv_878 instance with the native-tiled wiring."""
    m = find_inst(text, "node_conv_878")
    blk = m.group(0)
    if ".NATIVE_TILED(1)" in blk and ".valid_in_t(" in blk:
        return text  # already patched
    return text[:m.start()] + CONV878_NATIVE + text[m.end():]


def patch_n4_24(text):
    """n4_24: valid_in/data_in re-sourced from node_conv_878 (bridgeless, RAW valid)."""
    m = find_inst(text, "n4_24")
    blk = m.group(0)
    if "node_conv_878_valid_out" in blk:
        return text  # already patched
    nb, n1 = re.subn(r"\.valid_in\([^\n]*?\)(?=[,\s])",
                     ".valid_in(node_conv_878_valid_out)",
                     blk, count=1)
    if n1 != 1:
        raise RuntimeError(".valid_in not matched in u_n4_24")
    nb, n2 = re.subn(r"\.data_in\([^\n]*?\)(?=[,\s])",
                     ".data_in(node_conv_878_data_out)",
                     nb, count=1)
    if n2 != 1:
        raise RuntimeError(".data_in not matched in u_n4_24")
    return text[:m.start()] + nb + text[m.end():]


def narrow_data_out_wire(text):
    """Narrow node_conv_878_data_out from [4095:0] to [255:0]."""
    pat = r"wire \[4095:0\] node_conv_878_data_out;[^\n]*"
    repl = "wire [255:0] node_conv_878_data_out;  // [NATIVE_TILED_878] narrowed: native 256b tile bus"
    new, n = re.subn(pat, repl, text, count=1)
    if n != 1:
        if re.search(r"wire \[255:0\] node_conv_878_data_out;", text):
            return text  # already narrowed
        raise RuntimeError("node_conv_878_data_out [4095:0] decl not found")
    return new


def resolve_top():
    ap = argparse.ArgumentParser(description="Apply MobileNetV2 native-256b-tiled re-arch for node_conv_878.")
    ap.add_argument("--top", default=None,
                    help="path to the top .v to patch (default: $NN2RTL_TOP or the engine top).")
    args = ap.parse_args()
    return args.top or os.environ.get("NN2RTL_TOP") or DEFAULT_TOP


def main():
    top = resolve_top()
    with open(top, "r", encoding="utf-8") as f:
        text = f.read()

    already = NATIVE_MARK in text

    # Apply each transform (all are individually idempotent).
    # 1) Delete the two retile instances.
    if re.search(r"u_br_878\s*\(", text):
        m = find_retile_inst(text, "gather", "br_878")
        text = text[:m.start()] + text[m.end():]
    if re.search(r"u_br_n4_24\s*\(", text):
        m = find_retile_inst(text, "scatter", "br_n4_24")
        text = text[:m.start()] + text[m.end():]

    # 2) Delete the dead per-bridge drain wires (reference deleted stall_out).
    if re.search(r"wire spatial_run_drain_br_878 ", text):
        text = delete_drain_wire(text, "br_878")
    if re.search(r"wire spatial_run_drain_br_n4_24 ", text):
        text = delete_drain_wire(text, "br_n4_24")

    # 3) Delete the dead bridge wire-decl groups.
    if re.search(r"wire br_878_valid_out;", text):
        text = delete_wire_group(text, "br_878")
    if re.search(r"wire br_n4_24_valid_out;", text):
        text = delete_wire_group(text, "br_n4_24")

    # 4) Remove the two stall terms from any_retile_stall.
    text = remove_stall_terms(text, ["br_878_stall_out", "br_n4_24_stall_out"])

    # 5) Narrow the data_out wire.
    text = narrow_data_out_wire(text)

    # 6) Re-wire the three instances.
    text = patch_n4_23(text)
    text = patch_conv878(text)
    text = patch_n4_24(text)

    # 7) Sanity: no dangling references to the deleted nets remain.
    for dead in ["br_878_valid_out", "br_878_data_out", "br_878_ready_out",
                 "br_878_stall_out", "br_878_wr_accept", "spatial_run_drain_br_878",
                 "u_br_878",
                 "br_n4_24_valid_out", "br_n4_24_data_out", "br_n4_24_ready_out",
                 "br_n4_24_stall_out", "br_n4_24_wr_accept", "spatial_run_drain_br_n4_24",
                 "u_br_n4_24"]:
        if re.search(r"\b" + re.escape(dead) + r"\b", text):
            raise RuntimeError(f"dangling reference to deleted net/inst '{dead}' remains")
    if not re.search(r"wire \[255:0\] node_conv_878_data_out;", text):
        raise RuntimeError("node_conv_878_data_out not narrowed to [255:0]")
    if ".NATIVE_TILED(1)" not in text:
        raise RuntimeError("node_conv_878 not instantiated with .NATIVE_TILED(1)")

    # Stamp the marker so a re-run is a clean no-op detector.
    if NATIVE_MARK not in text:
        text = text.replace(
            "    // ===== WAVE-2 RETILE BRIDGES (apply_mbv2_wave2_bridges.py) =====",
            "    " + NATIVE_MARK + "\n    // ===== WAVE-2 RETILE BRIDGES (apply_mbv2_wave2_bridges.py) =====",
            1)

    # Atomic backup + write.
    bak = top + ".pre_native_tiled_878"
    if not os.path.exists(bak):
        shutil.copyfile(top, bak)
    with open(top, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"native-tiled node_conv_878 re-arch applied to {top}")
    print(f"  backup: {bak}")
    print(f"  state : {'re-applied (no-op)' if already else 'applied'}")
    print("  deleted: u_br_878 (gather), u_br_n4_24 (scatter), their wire groups,")
    print("           spatial_run_drain_br_878 / _br_n4_24, 2 any_retile_stall terms")
    print("  rewired: n4_23.out_ready_in -> node_conv_878_ready_in; conv878 #(.NATIVE_TILED(1))")
    print("           native 256b ports n4_23 -> 878 -> n4_24; n4_24 valid_in/data_in from conv878")
    print("  narrowed: node_conv_878_data_out [4095:0] -> [255:0]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
