#!/usr/bin/env python3
"""Gate each skip_fifo's out_ready by both add_ready_in AND the main path's
valid_out. This prevents the FIFO from silently draining while waiting for
the main path to deliver its first beat — the bug that caused node_add to
never fire because skip_fifo was empty by the time conv_204 first emitted.
"""
import re
from pathlib import Path

# Map of skip_fifo instance name -> main path valid_out signal
mains = {
    'u_skip_node_add':    'node_conv_204_valid_out',
    'u_skip_node_add_1':  'node_conv_210_valid_out',
    'u_skip_node_add_2':  'node_conv_216_valid_out',
    'u_skip_node_add_3':  'node_conv_224_valid_out',
    'u_skip_node_add_4':  'node_conv_230_valid_out',
    'u_skip_node_add_5':  'node_conv_236_valid_out',
    'u_skip_node_add_6':  'node_conv_242_valid_out',
    'u_skip_node_add_7':  'node_conv_250_valid_out',
    'u_skip_node_add_8':  'node_conv_256_valid_out',
    'u_skip_node_add_9':  'node_conv_262_valid_out',
    'u_skip_node_add_10': 'node_conv_268_valid_out',
    'u_skip_node_add_11': 'node_conv_274_valid_out',
    'u_skip_node_add_12': 'node_conv_280_valid_out',
    'u_skip_node_add_13': 'node_conv_288_valid_out',
    'u_skip_node_add_14': 'node_conv_294_valid_out',
    'u_skip_node_add_15': 'node_conv_300_valid_out',
}

p = Path('output/rtl/nn2rtl_top.v')
txt = p.read_text()
fixed = 0
for inst, main_v in mains.items():
    # Find the .out_ready(node_add_N_ready_in) line inside this instance.
    # Split by instance start, find first .out_ready(...), replace.
    add_name = inst[len('u_skip_'):]  # node_add or node_add_N
    old_pat = rf'(({re.escape(inst)}\s*\(.*?\.out_ready\()\s*{add_name}_ready_in\s*(\)))'
    m = re.search(old_pat, txt, re.DOTALL)
    if not m:
        print(f'[skip] {inst}: pattern not found')
        continue
    new_chunk = m.group(2) + f'{add_name}_ready_in & {main_v}' + m.group(3)
    txt = txt[:m.start()] + new_chunk + txt[m.end():]
    fixed += 1
    print(f'[fixed] {inst}: out_ready = {add_name}_ready_in & {main_v}')

p.write_text(txt)
print(f'\nTotal fixed: {fixed}')
