#!/usr/bin/env python3
"""Apply stage-transition block re-routing for ResNet-50 channel-expansion
blocks (blocks 1, 4). Also expand all DEPTH=64 skip_fifos to 8192.

PROBLEM:
    In each stage-transition block of this chain, the SKIP-PROJECTION 1x1
    conv (e.g. conv_204 for block 1) was wired in SERIES after the main
    expand (e.g. conv_202), instead of in parallel taking the BLOCK INPUT.
    Effect: add.main sees the projection's output, and add.skip gets the raw
    pre-projection channels (mismatch, e.g. 64ch vs 256ch). add then
    starves on rhs after only 25% of expected frame.

THIS SCRIPT swaps the wiring for blocks where both convs are LOCAL:
  - Block 1: conv_202 (main expand) <-> conv_204 (skip projection)
  - Block 4: conv_222 (main expand) <-> conv_224 (skip projection)

Blocks 8 and 14 are NOT touched because their projection convs (conv_250,
conv_288 vs conv_286) are engine-dispatched, requiring loader rewiring
that's out of scope here.

Also: expands all skip_fifos with DEPTH<8192 to DEPTH=8192 so they don't
overflow during the long stage-1 startup.

USAGE: python scripts/apply_block_reroute.py
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

TOP = Path('output/rtl/nn2rtl_top.v')
BACKUP = Path('output/rtl/nn2rtl_top.v.preroute')

if not BACKUP.exists():
    shutil.copy2(TOP, BACKUP)
    print(f'[backup] saved {BACKUP.name}')

txt = TOP.read_text()


# ---------------------------------------------------------------------------
# Step 1: Expand any DEPTH<8192 skip_fifo to 8192 (scoped to skip_fifo instances).
# ---------------------------------------------------------------------------
def expand_small_depth(match):
    return match.group(0).replace(f'DEPTH({match.group(2)})', 'DEPTH(8192)')

n_depth = 0
for d in [64, 128, 256, 512, 1024, 2048, 4096]:
    pat = re.compile(rf'(skip_fifo\s+#\(\s*\.WIDTH\(\d+\)\s*,\s*\.DEPTH\(({d})\)\)\s+u_skip_node_add\w*)')
    new_txt, c = pat.subn(lambda m, d=d: m.group(0).replace(f'DEPTH({d})', 'DEPTH(8192)'), txt)
    if c:
        txt = new_txt
        n_depth += c
        print(f'[depth] expanded {c} skip_fifo(s) from DEPTH={d} to 8192')

print(f'[depth] total expansions: {n_depth}')


# ---------------------------------------------------------------------------
# Step 2: Block 4 re-route (block 1 already done by hand).
#   Old wiring:
#     skid_node_conv_224.in = node_conv_222_*    (wrong - taking expand into projection)
#     node_add_3.main = node_conv_224_*          (wrong - projection as main)
#     skip_fifo_3.in = node_relu_9_*             (wrong - raw 256ch input as skip)
#     skip_fifo_3.out_ready ... & node_conv_224  (wrong sync)
#   New wiring (projection takes BLOCK INPUT = relu_9; expand stays as main):
#     skid_node_conv_224.in = node_relu_9_*      (projection takes block input)
#     node_add_3.main = node_conv_222_*          (expand as main)
#     skip_fifo_3.in = node_conv_224_*           (projection output as skip)
#     skip_fifo_3.out_ready ... & node_conv_222
# ---------------------------------------------------------------------------

# 2a) Re-route skid_node_conv_224 to read from node_relu_9
old_skid = (
    'skip_fifo #(.WIDTH(256), .DEPTH(8192)) u_skid_node_conv_224 (\n'
    '        .clk(clk), .rst_n(rst_n),\n'
    '        .in_valid(node_conv_222_valid_out & spatial_run & skid_node_conv_224_ready),\n'
    '        .in_data(node_conv_222_data_out),'
)
new_skid = (
    'skip_fifo #(.WIDTH(256), .DEPTH(8192)) u_skid_node_conv_224 (\n'
    '        .clk(clk), .rst_n(rst_n),\n'
    '        // BLOCK-4 RE-ROUTE (2026-05-26): conv_224 is the 1x1 256->512 skip\n'
    '        // PROJECTION; takes block 4 input (relu_9 = end of stage 1), not\n'
    '        // the main expand conv_222.\n'
    '        .in_valid(node_relu_9_valid_out & spatial_run & skid_node_conv_224_ready),\n'
    '        .in_data(node_relu_9_data_out),'
)
if old_skid in txt:
    txt = txt.replace(old_skid, new_skid)
    print('[block-4] re-routed skid_node_conv_224.in from conv_222 to relu_9')
else:
    print('[block-4] WARNING: skid_node_conv_224 already re-routed or pattern not found')

# 2b) node_add_3: take main from conv_222 (was conv_224)
old_add3 = (
    'node_add_3 u_node_add_3 (\n'
    '        .clk(clk), .rst_n(rst_n),\n'
    '        .valid_in(node_conv_224_valid_out & node_add_3_skip_valid & spatial_run),\n'
    '        .ready_in(node_add_3_ready_in),\n'
    '        .data_in({node_add_3_skip_data, node_conv_224_data_out[255:0]}),'
)
new_add3 = (
    'node_add_3 u_node_add_3 (\n'
    '        .clk(clk), .rst_n(rst_n),\n'
    '        // BLOCK-4 RE-ROUTE: main = conv_222 (1x1 expand), skip = conv_224\n'
    '        // (1x1 projection) via skip_fifo.\n'
    '        .valid_in(node_conv_222_valid_out & node_add_3_skip_valid & spatial_run),\n'
    '        .ready_in(node_add_3_ready_in),\n'
    '        .data_in({node_add_3_skip_data, node_conv_222_data_out[255:0]}),'
)
if old_add3 in txt:
    txt = txt.replace(old_add3, new_add3)
    print('[block-4] re-routed node_add_3 main from conv_224 to conv_222')
else:
    print('[block-4] WARNING: node_add_3 pattern not found or already changed')

# 2c) skip_fifo_3: in_data from conv_224 (was relu_9), out_ready synced with conv_222
old_skip3 = (
    'u_skip_node_add_3 (\n'
    '        .clk(clk), .rst_n(rst_n),\n'
    '        .in_valid(node_relu_9_valid_out & spatial_run & node_add_3_skip_in_ready),\n'
    '        .in_data(node_relu_9_data_out[255:0]),\n'
    '        .in_ready(node_add_3_skip_in_ready),\n'
    '        .out_valid(node_add_3_skip_valid),\n'
    '        .out_data(node_add_3_skip_data),\n'
    '        .out_ready(node_add_3_ready_in & node_conv_224_valid_out)\n'
    '    );'
)
new_skip3 = (
    'u_skip_node_add_3 (\n'
    '        .clk(clk), .rst_n(rst_n),\n'
    '        // BLOCK-4 RE-ROUTE: carries projected 512ch from conv_224.\n'
    '        .in_valid(node_conv_224_valid_out & spatial_run & node_add_3_skip_in_ready),\n'
    '        .in_data(node_conv_224_data_out[255:0]),\n'
    '        .in_ready(node_add_3_skip_in_ready),\n'
    '        .out_valid(node_add_3_skip_valid),\n'
    '        .out_data(node_add_3_skip_data),\n'
    '        .out_ready(node_add_3_ready_in & node_conv_222_valid_out)\n'
    '    );'
)
if old_skip3 in txt:
    txt = txt.replace(old_skip3, new_skip3)
    print('[block-4] re-routed skip_fifo_3 in_data from relu_9 to conv_224, out_ready synced with conv_222')
else:
    print('[block-4] WARNING: skip_fifo_3 pattern not found or already changed')


TOP.write_text(txt)
print(f'\n[written] {TOP}')
