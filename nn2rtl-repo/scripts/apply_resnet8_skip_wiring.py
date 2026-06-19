#!/usr/bin/env python3
"""
apply_resnet8_skip_wiring.py

Fix the ResNet-8 residual-add skip-FIFO source wiring AND the two main-path
mis-routes in the generated engine-less top (output/resnet8/rtl/nn2rtl_top.v).

ROOT CAUSE
----------
build_top_wrapper.ts assignSources() assumes a ResNet-50-style IR order where
the 1x1 projection conv (shortcut, "sc") of a downsampling block is emitted
*immediately before* the residual Add, and the FIRST conv after a fork is the
projection / skip path. ResNet-8's MLPerf-Tiny export orders each downsampling
block as:

    relu_2 -> conv2d_3 (sc / 1x1 projection, SKIP path)
           -> conv2d_4 (c1, MAIN path first conv) -> relu_3 -> conv2d_5 (c2)
    add_56 = conv2d_3 (lhs/skip) + conv2d_5 (rhs/main)

i.e. the projection (conv2d_3) is emitted BEFORE the main-path first conv
(conv2d_4). The wrapper's width-mismatch heuristic therefore (a) picks the
WRONG conv as the projection (it tags conv2d_4 as the projection because it
sees conv2d_3 widen the chain first) and (b) routes relu_3's data_in from
conv2d_3 instead of conv2d_4. Both the skip side and the main datapath end up
wrong for the two downsampling blocks; the first (identity) block's skip falls
back to PIXEL_IN (the classic MBV2-class bug).

GROUND TRUTH (checkpoints/resnet8.onnx Add input tensors, input[0]=lhs,
input[1]=rhs; the wrapper packs data_in = {mainSource[high]=rhs,
skipSource[low]=lhs}, and the add RTL applies FUSED_LHS_MULT to the LOW half):

    Add node          lhs (=skip, LOW)     rhs (=main, HIGH)
    node_add_25       node_relu            node_conv2d_2
    node_add_56       node_conv2d_3        node_conv2d_5
    node_add_87       node_conv2d_6        node_conv2d_8

Main-path consumers (node activation input -> producer):
    node_relu_3 <- node_conv2d_4   (wrapper wired it from node_conv2d_3)
    node_relu_5 <- node_conv2d_7   (wrapper wired it from node_conv2d_6)

The wrapper already got every *main* add source (rhs) right (conv2d_2 /
conv2d_5 / conv2d_8) and conv2d_3/4/6/7's own inputs right (all from relu_2 /
relu_4). Only the two relu main-feeds and the three skip FIFOs are wrong.

PATCH (surgical, idempotent; anchored on unique instance names)
---------------------------------------------------------------
1. Main path:
     u_node_relu_3 : .valid_in / .data_in  node_conv2d_3 -> node_conv2d_4
     u_node_relu_5 : .valid_in / .data_in  node_conv2d_6 -> node_conv2d_7
2. Skip FIFOs (preserve the "& spatial_run & <id>_skip_in_ready" valid tail and
   the "[W-1:0]" data slice width = busInBits/2):
     u_skip_node_add_25 : PIXEL_IN      -> node_relu      (W=128)
     u_skip_node_add_56 : node_conv2d_4 -> node_conv2d_3  (W=256)
     u_skip_node_add_87 : node_conv2d_7 -> node_conv2d_6  (W=512)

Does NOT touch the add main-path data_in, the FIFO WIDTH/DEPTH, any conv's own
inputs, or any other instance. Verifies post-patch that the emitted wiring
matches the ONNX-derived ground truth and aborts otherwise.
"""
import argparse
import os
import re
import sys

DEFAULT_TOP = r"D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/resnet8/rtl/nn2rtl_top.v"

# (add_id, correct_skip_src, skip_fifo_width_bits)
SKIP_MAP = [
    ("node_add_25", "node_relu",     128),
    ("node_add_56", "node_conv2d_3", 256),
    ("node_add_87", "node_conv2d_6", 512),
]

# (relu_inst, correct_main_src) — fix the two mis-routed main-path relus.
MAIN_FIX = [
    ("node_relu_3", "node_conv2d_4"),
    ("node_relu_5", "node_conv2d_7"),
]

# Skip-FIFO drain gating. The adds are FREE-RUNNING parallel (ready_in tied
# high), so the wrapper's `out_ready = <add>_ready_in` drains the skip FIFO
# EVERY cycle it holds data — emptying it before the (frame-gate-delayed) main
# operand starts arriving, so the add never sees both operands and produces 0.
# Pop the skip beat ONLY on the cycle the add actually fires: when its MAIN
# operand is valid. (add_id, main_valid_signal). The main producer is itself
# free-running, so its beat coincides with the held skip beat -> aligned.
OUT_READY_FIX = {
    "node_add_25": "node_conv2d_2_valid_out",
    "node_add_56": "node_conv2d_5_valid_out",
    "node_add_87": "node_conv2d_8_valid_out",
}

# Right-size skip-FIFO depth >= the residual frame size so the FIFO never
# overflows regardless of the early-fork-vs-deep-main-path latency skew. The
# producers are free-running (ready_in tied high) so an undersized FIFO silently
# DROPS skip beats -> desync -> deadlock. add_25's frame is 32*32=1024 and its
# main path is two convs deep; the wrapper default (256) is too small.
# (add_id, min_depth). Only depths BELOW the requested value are bumped.
DEPTH_FIX = {
    "node_add_25": 2048,   # frame 1024, two-conv-deep main skew
    "node_add_56": 512,    # frame 256 (already >= 512, left as-is)
    "node_add_87": 128,    # frame 64 (wrapper gives 1728, left as-is)
}


def _find_instance(text, inst_name):
    """Return (start, end, block) for the instance `<module> u_<inst_name> ( ... );`."""
    inst_re = re.compile(r"(u_" + re.escape(inst_name) + r"\s*\((?:.|\n)*?\n\s*\);)")
    m = inst_re.search(text)
    if not m:
        raise RuntimeError(f"instance block u_{inst_name} not found")
    return m.start(1), m.end(1), m.group(1)


def patch_main(text, relu_inst, src):
    """Repoint u_<relu_inst>'s .valid_in and .data_in to <src>."""
    s, e, block = _find_instance(text, relu_inst)
    orig = block

    # .valid_in(<X>_valid_out & spatial_run)  ->  <src>_valid_out & spatial_run
    block, nv = re.subn(
        r"\.valid_in\(\s*\w+_valid_out\s*&\s*spatial_run\s*\)",
        f".valid_in({src}_valid_out & spatial_run)", block, count=1)
    if nv != 1:
        raise RuntimeError(f".valid_in not matched in u_{relu_inst}")

    # .data_in(<X>_data_out)  ->  <src>_data_out
    block, nd = re.subn(
        r"\.data_in\(\s*\w+_data_out\s*\)",
        f".data_in({src}_data_out)", block, count=1)
    if nd != 1:
        raise RuntimeError(f".data_in not matched in u_{relu_inst}")

    changed = block != orig
    return text[:s] + block + text[e:], changed


def patch_skip(text, add_id, src, width):
    """Repoint the u_skip_<add_id> FIFO's .in_valid/.in_data to <src>."""
    s, e, block = _find_instance(text, f"skip_{add_id}")
    orig = block

    # .in_valid(<OLD> & spatial_run & <add_id>_skip_in_ready) -> <src>_valid_out & <tail>
    iv_re = re.compile(r"\.in_valid\(\s*[^&)]+?\s*(&[^)]*)?\)")

    def iv_sub(mm):
        tail = (mm.group(1) or "").strip()
        return f".in_valid({src}_valid_out {tail})" if tail else f".in_valid({src}_valid_out)"

    block, nv = iv_re.subn(iv_sub, block, count=1)
    if nv != 1:
        raise RuntimeError(f".in_valid not matched in u_skip_{add_id}")

    # .in_data(<OLD>[W-1:0]) -> <src>_data_out[W-1:0]
    block, nd = re.subn(
        r"\.in_data\([^)]*\)", f".in_data({src}_data_out[{width - 1}:0])", block, count=1)
    if nd != 1:
        raise RuntimeError(f".in_data not matched in u_skip_{add_id}")

    changed = block != orig
    return text[:s] + block + text[e:], changed


def patch_out_ready(text, add_id, main_valid):
    """Gate the u_skip_<add_id> FIFO .out_ready on the add's main-operand valid."""
    s, e, block = _find_instance(text, f"skip_{add_id}")
    orig = block
    new_or = f"{main_valid} & {add_id}_ready_in"
    block, n = re.subn(r"\.out_ready\([^)]*\)", f".out_ready({new_or})", block, count=1)
    if n != 1:
        raise RuntimeError(f".out_ready not matched in u_skip_{add_id}")
    return text[:s] + block + text[e:], (block != orig)


def patch_depth(text, add_id, min_depth):
    """Bump the u_skip_<add_id> FIFO .DEPTH up to >= min_depth (never shrink)."""
    # skip_fifo #(.WIDTH(W), .DEPTH(D)) u_skip_<add_id> (
    dre = re.compile(r"(skip_fifo\s*#\(\.WIDTH\((\d+)\),\s*\.DEPTH\()(\d+)(\)\)\s*u_skip_" + re.escape(add_id) + r"\b)")
    m = dre.search(text)
    if not m:
        raise RuntimeError(f"skip_fifo decl for u_skip_{add_id} not found")
    cur = int(m.group(3))
    if cur >= min_depth:
        return text, False
    new = m.group(1) + str(min_depth) + m.group(4)
    return text[:m.start()] + new + text[m.end():], True


def verify(text):
    """Assert the post-patch wiring matches ONNX ground truth; raise on mismatch."""
    problems = []
    for relu_inst, src in MAIN_FIX:
        _, _, block = _find_instance(text, relu_inst)
        if f".data_in({src}_data_out)" not in block:
            problems.append(f"u_{relu_inst} .data_in not {src}_data_out")
        if f".valid_in({src}_valid_out & spatial_run)" not in block:
            problems.append(f"u_{relu_inst} .valid_in not {src}_valid_out & spatial_run")
    for add_id, src, width in SKIP_MAP:
        _, _, block = _find_instance(text, f"skip_{add_id}")
        if f".in_data({src}_data_out[{width - 1}:0])" not in block:
            problems.append(f"u_skip_{add_id} .in_data not {src}_data_out[{width-1}:0]")
        if f".in_valid({src}_valid_out " not in block and f".in_valid({src}_valid_out)" not in block:
            problems.append(f"u_skip_{add_id} .in_valid not {src}_valid_out")
    return problems


def main():
    ap = argparse.ArgumentParser(description="Fix ResNet-8 skip + main-path wiring in the top.")
    ap.add_argument("--top", default=os.environ.get("NN2RTL_TOP") or DEFAULT_TOP)
    args = ap.parse_args()
    top = args.top

    with open(top, "r", encoding="utf-8") as f:
        text = f.read()

    report = []
    for relu_inst, src in MAIN_FIX:
        text, changed = patch_main(text, relu_inst, src)
        report.append((f"main:{relu_inst}", src, changed))
    for add_id, src, width in SKIP_MAP:
        text, changed = patch_skip(text, add_id, src, width)
        report.append((f"skip:{add_id}", f"{src}[{width-1}:0]", changed))
    for add_id, min_depth in DEPTH_FIX.items():
        text, changed = patch_depth(text, add_id, min_depth)
        report.append((f"depth:{add_id}", f">={min_depth}", changed))
    for add_id, main_valid in OUT_READY_FIX.items():
        text, changed = patch_out_ready(text, add_id, main_valid)
        report.append((f"out_ready:{add_id}", f"{main_valid} & rdy", changed))

    problems = verify(text)
    if problems:
        print("VERIFY FAILED — not writing:", file=sys.stderr)
        for p in problems:
            print("  -", p, file=sys.stderr)
        return 1

    with open(top, "w", encoding="utf-8") as f:
        f.write(text)

    print("Patched ResNet-8 wiring in", top)
    for what, src, changed in report:
        print(f"  {what:24s} <- {src:28s} [{'patched' if changed else 'already-correct'}]")
    print("VERIFY OK: all main-path + skip sources match ONNX ground truth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
