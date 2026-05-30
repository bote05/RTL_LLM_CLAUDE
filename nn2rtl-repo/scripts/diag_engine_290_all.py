#!/usr/bin/env python3
"""Enumerate ALL byte mismatches for conv_290 and group them by
(pixel, oc_pass) tuple to find structural patterns.
"""
import struct
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parent.parent
IC = 2048
OC = 512
IH = IW = 7
OH = OW = 7
OC_PASSES = (OC + 255) // 256


def signed_int8(b):
    return b - 256 if b >= 128 else b


# Golden output
goldout_path = REPO / "output/goldens/node_conv_290.goldout"
godata = goldout_path.read_bytes()
m, v, nv, spv, bps = struct.unpack("<4sIIII", godata[:20])
gvec0 = godata[20:20 + spv * ((bps + 3)//4) * 4]


def get_gold(r, c, ch):
    sample = r * OW + c
    return signed_int8(gvec0[sample * bps + ch])


# Observed
obs_path = REPO / "output/engine_sweep/observed_dispatch10_node_conv_290.hex"
obs_lines = obs_path.read_text().strip().split('\n')


def get_observed(r, c, ch):
    pixel_idx = r * OW + c
    oc_pass = ch // 256
    word_idx = pixel_idx * OC_PASSES + oc_pass
    line = obs_lines[word_idx]
    slot = ch % 256
    hex_len = len(line)
    byte_hex = line[hex_len - 2*(slot+1): hex_len - 2*slot]
    b = int(byte_hex, 16)
    return signed_int8(b)


# Iterate all (pixel, ch) and report mismatches
mismatches = []
for r in range(OH):
    for c in range(OW):
        for ch in range(OC):
            g = get_gold(r, c, ch)
            o = get_observed(r, c, ch)
            if g != o:
                mismatches.append((r, c, ch, g, o))

print(f"Total mismatches: {len(mismatches)}")
print()
# Group by (pixel, oc_pass)
by_pp = defaultdict(list)
for r, c, ch, g, o in mismatches:
    pp = (r, c, ch // 256)
    by_pp[pp].append((ch, g, o))

print("Mismatches grouped by (pixel_row, pixel_col, oc_pass):")
for pp in sorted(by_pp.keys()):
    r, c, pp_idx = pp
    chs = by_pp[pp]
    chans = [item[0] for item in chs]
    print(f"  ({r},{c}) pass={pp_idx}: {len(chs)} mismatches, channels={chans[:20]}{'...' if len(chans)>20 else ''}")

# Also check: how many mismatches are 'engine = gold - 1' vs 'engine = gold + 1'
delta_minus = sum(1 for _, _, _, g, o in mismatches if o == g - 1)
delta_plus  = sum(1 for _, _, _, g, o in mismatches if o == g + 1)
delta_other = sum(1 for _, _, _, g, o in mismatches if o != g - 1 and o != g + 1)
print()
print(f"engine = gold - 1: {delta_minus}")
print(f"engine = gold + 1: {delta_plus}")
print(f"engine off by other amount: {delta_other}")
