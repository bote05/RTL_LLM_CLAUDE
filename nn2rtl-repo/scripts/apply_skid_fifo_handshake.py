#!/usr/bin/env python3
"""Apply skid_fifo handshake pattern at fast->slow boundaries in nn2rtl_top.v.

PROBLEM:
    Most chain producers in nn2rtl_top emit pulse-based valid_out (1-2 cycles
    per pixel). Many consumers (conv modules with long compute windows) only
    have ready_in=1 for a narrow window per pixel. Without a buffer between
    them, the producer's pulse and the consumer's accept window rarely
    align, so beats are silently lost. We measured 89% loss at the
    max_pool -> conv_198 boundary alone.

PROVEN APPROACH:
    A single skid_fifo (DEPTH=8192) at max_pool -> conv_198 recovered all
    6272 beats (vs 672 before). The skid captures producer pulses with its
    own in_ready handshake, then feeds the consumer with held valid_out
    until consumed. Result: chain proceeded past add (which had been
    deadlocked) and processed full-frame data through conv_204.

THIS SCRIPT replicates that pattern across the chain:
  1) For every single-producer chain module (conv/relu/maxpool/etc. whose
     valid_in reads `node_X_valid_out & spatial_run`), insert a skid_fifo
     between the producer and consumer.
  2) For every existing residual skip_fifo (DEPTH=1024), increase to
     DEPTH=8192 so the skip path can buffer a full frame.

SCOPED REGEX:
    Each transformation matches a single instantiation block (.clk through
    the closing `);`) and replaces only that block. No global rewrites that
    could corrupt unrelated modules (that bug class bit apply_parallel_data-
    path.py earlier in this session).

BACKUP:
    Saves nn2rtl_top.v.preskid before modifying. If it already exists, it
    is NOT overwritten (so multiple runs are idempotent).

USAGE:
    python scripts/apply_skid_fifo_handshake.py [--dry-run]
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


TOP_PATH = Path('output/rtl/nn2rtl_top.v')
BACKUP_PATH = Path('output/rtl/nn2rtl_top.v.preskid')


# ---------------------------------------------------------------------------
# Step 1: expand existing residual skip_fifos from DEPTH=1024 to 8192.
# Scoped to skip_fifo instance bodies, so we won't touch random "1024"
# tokens elsewhere in the file.
# ---------------------------------------------------------------------------
SKIP_FIFO_DEPTH_RE = re.compile(
    r'(skip_fifo\s+#\(\s*\.WIDTH\(256\)\s*,\s*\.DEPTH\()1024(\)\s*\)\s+u_skip_node_add)',
)


def expand_skip_fifo_depths(txt: str) -> tuple[str, int]:
    new_txt, n = SKIP_FIFO_DEPTH_RE.subn(r'\g<1>8192\g<2>', txt)
    return new_txt, n


# ---------------------------------------------------------------------------
# Step 2: insert skid_fifo at each single-producer chain instantiation.
# Match pattern:
#   node_KIND_NN u_node_KIND_NN (
#       .clk(clk), .rst_n(rst_n),
#       .valid_in(node_PROD_valid_out & spatial_run),
#       .ready_in(node_KIND_NN_ready_in),
#       .data_in(node_PROD_data_out),
#       .valid_out(node_KIND_NN_valid_out),
#       .data_out(node_KIND_NN_data_out)
#   );
#
# NOT matched (intentionally):
#   - node_conv_196 (input is raw s_axis pixel bus, not node_X_valid_out)
#   - node_add* (input has 3 AND-ed terms including skip_valid, plus
#     concatenated data_in {skip_data, main_data})
#   - max_pool already has my hand-applied skid between it and conv_198
#     (valid_in reads mp_to_c198_valid, not node_X_valid_out & spatial_run)
#   - Engine-dispatched modules (no instantiation; bridges drive their
#     downstream wires)
# ---------------------------------------------------------------------------
CHAIN_INST_RE = re.compile(
    r'(node_(?:conv|relu|max_pool2d)(?:_\d+)?)\s+'        # module type
    r'(u_node_(?:conv|relu|max_pool2d)(?:_\d+)?)\s*\(\s*' # instance name
    r'\.clk\(clk\)\s*,\s*\.rst_n\(rst_n\)\s*,\s*'
    r'\.valid_in\(\s*(node_\w+)_valid_out\s*&\s*spatial_run\s*\)\s*,\s*'
    r'\.ready_in\(\s*\3?\w*_ready_in\s*\)\s*,\s*'
    r'\.data_in\(\s*(node_\w+)_data_out\s*\)\s*,\s*'
    r'\.valid_out\(\s*\w+_valid_out\s*\)\s*,\s*'
    r'\.data_out\(\s*\w+_data_out\s*\)\s*\)\s*;',
    re.DOTALL,
)


def build_skid_block(cons_name: str, prod_valid_base: str, prod_data_base: str) -> str:
    """Generate Verilog snippet to insert before the consumer instantiation."""
    skid = f'skid_{cons_name}'
    return (
        f'    // [skid_fifo inserted by apply_skid_fifo_handshake.py]\n'
        f'    wire {skid}_valid;\n'
        f'    wire [255:0] {skid}_data;\n'
        f'    wire {skid}_ready;\n'
        f'    skip_fifo #(.WIDTH(256), .DEPTH(8192)) u_{skid} (\n'
        f'        .clk(clk), .rst_n(rst_n),\n'
        f'        .in_valid({prod_valid_base}_valid_out & spatial_run & {skid}_ready),\n'
        f'        .in_data({prod_data_base}_data_out),\n'
        f'        .in_ready({skid}_ready),\n'
        f'        .out_valid({skid}_valid),\n'
        f'        .out_data({skid}_data),\n'
        f'        .out_ready({cons_name}_ready_in & spatial_run)\n'
        f'    );\n'
    )


def rewrite_chain_instantiations(txt: str, dry_run: bool) -> tuple[str, list[tuple[str, str]]]:
    """Find every matching chain instantiation and prepend a skid_fifo.

    Returns (new_text, list of (consumer, producer)) for logging.
    """
    fixes: list[tuple[str, str]] = []
    cursor = 0
    parts: list[str] = []
    for m in CHAIN_INST_RE.finditer(txt):
        cons_module = m.group(1)            # e.g. node_conv_198
        prod_valid_base = m.group(3)        # e.g. node_max_pool2d (from .._valid_out)
        prod_data_base = m.group(4)         # should match prod_valid_base
        if prod_valid_base != prod_data_base:
            # Wiring oddity (different valid/data sources). Skip for safety.
            continue
        # Skip if a skid for this consumer was already inserted by a prior run.
        if f'skid_{cons_module}_valid' in txt:
            continue
        skid_block = build_skid_block(cons_module, prod_valid_base, prod_data_base)
        inst_block = m.group(0)
        # Rewrite consumer's valid_in / data_in to read from the skid wires.
        inst_block_new = re.sub(
            r'\.valid_in\([^)]*\)',
            f'.valid_in(skid_{cons_module}_valid)',
            inst_block,
            count=1,
        )
        inst_block_new = re.sub(
            r'\.data_in\([^)]*\)',
            f'.data_in(skid_{cons_module}_data)',
            inst_block_new,
            count=1,
        )
        parts.append(txt[cursor:m.start()])
        parts.append(skid_block)
        parts.append(inst_block_new)
        cursor = m.end()
        fixes.append((cons_module, prod_valid_base))
    parts.append(txt[cursor:])
    return (''.join(parts) if not dry_run else txt, fixes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='Print plan without writing.')
    args = ap.parse_args()

    if not TOP_PATH.exists():
        print(f'[err] {TOP_PATH} not found (run from nn2rtl-repo/)', file=sys.stderr)
        sys.exit(1)

    txt = TOP_PATH.read_text()

    if not BACKUP_PATH.exists() and not args.dry_run:
        shutil.copy2(TOP_PATH, BACKUP_PATH)
        print(f'[backup] saved {BACKUP_PATH.name}')
    elif BACKUP_PATH.exists():
        print(f'[backup] {BACKUP_PATH.name} already exists, not overwriting')

    # Step 1: skip_fifo depth expansion.
    txt, n_depth = expand_skip_fifo_depths(txt)
    print(f'[depth] expanded {n_depth} residual skip_fifos from DEPTH=1024 to 8192')

    # Step 2: insert skid_fifos at chain boundaries.
    new_txt, fixes = rewrite_chain_instantiations(txt, args.dry_run)
    print(f'[skid] inserted {len(fixes)} skid_fifos:')
    for cons, prod in fixes:
        print(f'        {prod} -> {cons}')

    if args.dry_run:
        print('[dry-run] no files written')
        return

    TOP_PATH.write_text(new_txt)
    print(f'[written] {TOP_PATH}')


if __name__ == '__main__':
    main()
