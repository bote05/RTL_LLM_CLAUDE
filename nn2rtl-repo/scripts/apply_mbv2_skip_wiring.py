#!/usr/bin/env python3
"""
apply_mbv2_skip_wiring.py

Fix MobileNetV2 residual-add skip-FIFO source wiring in the generated top.

BUG
---
build_top_wrapper.ts wires every residual-add skip FIFO's `.in_data` (and, for
the spatial blocks, `.in_valid`) from PIXEL_IN_data / PIXEL_IN_valid (the 24-bit
network RGB input), or from a stale wrong conv. Its skipSource logic
    m.skipSource = pendingProj ?? lastFork ?? "PIXEL_IN"
(build_top_wrapper.ts ~lines 360-362) is ResNet-tuned: it expects a *projection*
conv right before each add. MobileNetV2 inverted-residual skips are IDENTITY (the
block INPUT, with no projection), so pendingProj is null, lastFork is stale, and
it falls back to PIXEL_IN (or a wrong depthwise/expand conv).

CORRECT SKIP SOURCE (derived from checkpoints/mobilenet_v2.onnx Add-node input
tensors + output/mobilenet-v2/layer_ir.json topological order, channel-count
verified). For each add, skip = block input = the output of the layer feeding
the block's expand (1x1) conv = (for stride-1 same-channel blocks) the PREVIOUS
block's output:

    node_add_198  -> node_conv_820   (24ch,  192b)
    node_add_336  -> node_conv_832   (32ch,  256b)
    node_add_408  -> node_add_336    (32ch,  256b)
    node_add_546  -> node_conv_850   (64ch,  512b)
    node_add_618  -> node_add_546    (64ch,  512b)
    node_add_690  -> node_add_618    (64ch,  512b)
    node_add_828  -> node_conv_874   (96ch,  768b)
    node_add_900  -> node_add_828    (96ch,  768b)
    node_add_1038 -> node_conv_892   (160ch, 1280b skip-FIFO; tiled-streaming src
                                       256b bus, sliced [1279:0] to match FIFO
                                       WIDTH, same convention as the main-path
                                       add rhs node_conv_898_data_out[1279:0])
    node_add_1110 -> node_add_1038   (160ch, 1280b)

PATCH
-----
For each of the 10 skip-FIFO instances (u_skip_node_add_<id>) replace
    .in_valid(<OLD> & spatial_run & node_add_<id>_skip_in_ready)
    .in_data(<OLD>)
with
    .in_valid(<src>_valid_out & spatial_run & node_add_<id>_skip_in_ready)
    .in_data(<src>_data_out[<W-1>:0])
where <W> is the FIFO's declared WIDTH (kept consistent). The
`& spatial_run & ..._skip_in_ready` qualifiers and DEPTH are left untouched.

This is a surgical, idempotent on-disk patch of the already-patched working top
(nn2rtl_top.v is patched, not regenerated). It does NOT touch the main-path add
data_in, the FIFO WIDTH/DEPTH, or any other instance.
"""
import argparse
import os
import re
import sys

# Default target is the running BASELINE top (unchanged behavior).  The engine-
# dispatched top (nn2rtl_top_engine.v) can be targeted via --top or the
# NN2RTL_TOP env var WITHOUT changing the default.
DEFAULT_TOP = r"C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/rtl/nn2rtl_top.v"
TOP = DEFAULT_TOP

# add_id -> (skip_source_module_id, skip_fifo_width_bits)
SKIP_MAP = [
    ("node_add_198",  "node_conv_820",  192),
    ("node_add_336",  "node_conv_832",  256),
    ("node_add_408",  "node_add_336",   256),
    ("node_add_546",  "node_conv_850",  512),
    ("node_add_618",  "node_add_546",   512),
    ("node_add_690",  "node_add_618",   512),
    ("node_add_828",  "node_conv_874",  768),
    ("node_add_900",  "node_add_828",   768),
    ("node_add_1038", "node_conv_892",  1280),
    ("node_add_1110", "node_add_1038",  1280),
]


def patch_instance(text, add_id, src, width):
    """Replace .in_valid(...) and .in_data(...) inside the u_skip_<add_id> block."""
    # Locate the instance block: from the instance name "u_skip_<add_id> (" up
    # to the matching ");". Anchor on the unique instance name (the parameter
    # "#( .WIDTH(..) )" section before it contains parentheses, so we do not try
    # to span it).
    inst_re = re.compile(
        r"(u_skip_" + re.escape(add_id) + r"\s*\((?:.|\n)*?\n\s*\);)"
    )
    m = inst_re.search(text)
    if not m:
        raise RuntimeError(f"instance block u_skip_{add_id} not found")
    block = m.group(1)
    orig_block = block

    # --- .in_valid(...) : preserve the "& spatial_run & ..._skip_in_ready" tail ---
    new_valid = f"{src}_valid_out"
    valid_re = re.compile(r"\.in_valid\(\s*([^&\)]+?)\s*(&[^\)]*)?\)")

    def valid_sub(mm):
        tail = mm.group(2) or ""
        return f".in_valid({new_valid} {tail.strip()})" if tail.strip() else f".in_valid({new_valid})"

    block, nv = valid_re.subn(valid_sub, block, count=1)
    if nv != 1:
        raise RuntimeError(f".in_valid not matched in u_skip_{add_id}")

    # --- .in_data(...) ---
    new_data = f"{src}_data_out[{width - 1}:0]"
    data_re = re.compile(r"\.in_data\([^\)]*\)")
    block, nd = data_re.subn(f".in_data({new_data})", block, count=1)
    if nd != 1:
        raise RuntimeError(f".in_data not matched in u_skip_{add_id}")

    if block == orig_block:
        # already correct; nothing changed
        return text, False
    return text[: m.start(1)] + block + text[m.end(1):], True


def resolve_top():
    """Resolve the target top from --top, then NN2RTL_TOP, then DEFAULT_TOP.
    Default behavior (no arg, no env) is unchanged: the baseline nn2rtl_top.v."""
    ap = argparse.ArgumentParser(description="Patch MobileNetV2 skip-FIFO source wiring.")
    ap.add_argument("--top", default=None,
                    help="path to the top .v to patch (default: $NN2RTL_TOP or the baseline "
                         "nn2rtl_top.v). Pass the engine top nn2rtl_top_engine.v to retarget.")
    args = ap.parse_args()
    return args.top or os.environ.get("NN2RTL_TOP") or DEFAULT_TOP


def main():
    top = resolve_top()
    with open(top, "r", encoding="utf-8") as f:
        text = f.read()

    changed_any = False
    report = []
    for add_id, src, width in SKIP_MAP:
        text, changed = patch_instance(text, add_id, src, width)
        report.append((add_id, src, width, changed))
        changed_any = changed_any or changed

    if changed_any:
        with open(top, "w", encoding="utf-8") as f:
            f.write(text)

    print("Patched skip-FIFO wiring in", top)
    for add_id, src, width, changed in report:
        state = "patched" if changed else "already-correct"
        print(f"  {add_id:14s} skip <- {src}_data_out[{width-1}:0] (valid {src}_valid_out)  [{state}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
