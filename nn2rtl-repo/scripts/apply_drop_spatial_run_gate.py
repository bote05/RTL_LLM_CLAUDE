#!/usr/bin/env python3
"""Option B: drop `& spatial_run` from skid_fifo / skip_fifo `in_valid` gates.

The `spatial_run` gate on `in_valid` silently drops producer pulses whenever
`spatial_throttle = engine_busy | sched_spatial_stall` goes high. Producers
(relu_*, conv_*) emit pulse-style valid_out without honoring ready_in, so a
gated `in_valid=0` cycle = a lost beat.

This patch removes `& spatial_run` from every `in_valid` line whose pattern
is `<producer>_valid_out & spatial_run & <fifo>_ready` (or `_in_ready`).

We KEEP `& spatial_run` on `out_ready` lines — those are the correct freeze
mechanism that keeps the chain stalled during engine windows without losing
beats from the producer side.

USAGE: python scripts/apply_drop_spatial_run_gate.py
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

TOP = Path('output/rtl/nn2rtl_top.v')
BACKUP = Path('output/rtl/nn2rtl_top.v.prebspatial')

if not BACKUP.exists():
    shutil.copy2(TOP, BACKUP)
    print(f'[backup] saved {BACKUP.name}')

txt = TOP.read_text()

# Pattern matches: `<prod>_valid_out & spatial_run & <consumer>_<ready|in_ready>`
# In all matching cases, this is a skid_fifo or skip_fifo `in_valid` gate.
# We remove the `spatial_run & ` middle.
pat = re.compile(r'(\w+_valid_out)\s+&\s+spatial_run\s+&\s+(\w+_(?:in_ready|ready))')

matches = list(pat.finditer(txt))
print(f'[patch] found {len(matches)} in_valid gates to patch')

# Apply substitution
new_txt = pat.sub(r'\1 & \2', txt)

# Sanity: should be fewer `spatial_run` references now
n_before = txt.count('spatial_run')
n_after = new_txt.count('spatial_run')
print(f'[patch] spatial_run references: {n_before} -> {n_after} (delta {n_before - n_after})')

TOP.write_text(new_txt)
print(f'[written] {TOP}')
