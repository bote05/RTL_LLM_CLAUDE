#!/usr/bin/env python3
"""For each node_add[_N], find the main path valid_out source."""
import re
with open('output/rtl/nn2rtl_top.v') as f:
    txt = f.read()
# Split by node_add instantiation, then find valid_in
chunks = re.split(r'(?=node_add(?:_\d+)?\s+u_node_add)', txt)
for c in chunks:
    if not c.lstrip().startswith('node_add'):
        continue
    name_m = re.match(r'(node_add(?:_\d+)?)\s+(u_node_add(?:_\d+)?)', c)
    valid_m = re.search(r'\.valid_in\((\w+_valid_out)', c[:500])
    if name_m and valid_m:
        print(f'{name_m.group(1)}: main_valid = {valid_m.group(1)}')
