#!/usr/bin/env python3
"""Audit ALL $readmemh paths in EVERY RTL file. Check element width vs file
content width to find silent truncation."""
import re
from pathlib import Path

problems = []
for v in sorted(Path('output/rtl').rglob('*.v')):
    txt = v.read_text()
    # Find every $readmemh("path", array_name) instance
    for m in re.finditer(r'\$readmemh\(\s*"([^"]+)"\s*,\s*(\w+)\s*\)', txt):
        path = m.group(1)
        array = m.group(2)
        fname = path.split('/')[-1]
        f = Path('output/weights') / fname
        if not f.exists():
            problems.append((v.stem, array, fname, 'MISSING'))
            continue
        # Find array declaration width
        decl = re.search(rf'reg\s+(?:signed\s+)?\[(\d+):0\]\s+{array}\b', txt)
        if not decl:
            problems.append((v.stem, array, fname, 'NO_DECL'))
            continue
        elem_width = int(decl.group(1)) + 1
        # Sample first non-empty non-comment line of file
        sample = None
        for line in f.read_text().splitlines():
            s = line.strip()
            if s and not s.startswith('//'):
                sample = s
                break
        if sample is None:
            problems.append((v.stem, array, fname, 'EMPTY'))
            continue
        # Hex digits in sample = file's "logical" element width in nibbles
        file_nibbles = len(sample)
        elem_nibbles = (elem_width + 3) // 4
        if file_nibbles != elem_nibbles:
            problems.append((v.stem, array, fname,
                f'WIDTH_MISMATCH elem={elem_width}b/{elem_nibbles}nibble file={file_nibbles}nibble sample={sample}'))

print(f'Total problems: {len(problems)}')
for p in problems:
    print(f'  {p[0]}: array={p[1]} file={p[2]} [{p[3]}]')
