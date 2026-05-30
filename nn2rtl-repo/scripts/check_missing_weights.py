#!/usr/bin/env python3
"""Find all readmemh paths in node_conv_*.v and report missing files."""
import re
from pathlib import Path

missing = []
present = []
for v in sorted(Path('output/rtl').glob('node_conv_*.v')):
    txt = v.read_text()
    # Find all $readmemh("...")
    for m in re.finditer(r'\$readmemh\(\s*"([^"]+)"', txt):
        path = m.group(1)
        fname = path.split('/')[-1]
        wf = Path('output/weights') / fname
        if not wf.exists():
            missing.append((v.stem, fname))
        else:
            present.append((v.stem, fname))

print(f'Total readmemh paths: {len(missing)+len(present)}')
print(f'Missing: {len(missing)}')
print(f'Present: {len(present)}')
print()
print('Missing files (first 30):')
for m, f in missing[:30]:
    print(f'  {m}: {f}')
