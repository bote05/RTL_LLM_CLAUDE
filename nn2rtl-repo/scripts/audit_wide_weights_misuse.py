#!/usr/bin/env python3
"""Find pointwise modules wrongly using _weights_wide.hex with flat 8-bit array."""
import re
from pathlib import Path
broken = []
for v in sorted(Path('output/rtl').glob('node_conv_*.v')):
    txt = v.read_text()
    if 'conv_datapath_parallel' in txt or 'conv_datapath_mp_k' in txt:
        continue
    paths = re.findall(r'\$readmemh\(\s*"([^"]+)"', txt)
    w_match = re.search(r'reg signed \[(\d+):0\]\s+weights', txt)
    if not w_match:
        continue
    w_width = int(w_match.group(1)) + 1
    for p in paths:
        fname = p.split('/')[-1]
        if 'weights_wide' in fname:
            broken.append((v.stem, w_width, fname))
            break
print(f'Files with wide weight path but flat {[b[1] for b in broken]}-bit array: {len(broken)}')
for m, w, f in broken[:40]:
    print(f'  {m}: weights[{w-1}:0], path={f}')
