#!/usr/bin/env python3
"""Diagnostic: compute the golden accumulator and intermediates for
node_conv_246 at output pixel [1,4] for channels 124 and 238, and also
verify the weight bytes the engine should be reading per MAC cycle and
the bias values it should be summing in.
"""
import struct
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent

# Layer geometry for node_conv_246.
IC = 256
OC = 256
KH = KW = 3
SH = SW = 2
PH = PW = 1
IH = IW = 28
OH = OW = 14
SCALE_MULT = 1284434803
SCALE_SHIFT = 39

WEIGHT_BASE_WORD = 11155
BIAS_BASE_WORD = 31
ACT_IN_BASE = 8192
ACT_OUT_BASE = 4096

# ---- Load input activations (vector 0) ----
goldin_path = REPO / "output/goldens/node_conv_246.goldin"
data = goldin_path.read_bytes()
# 20-byte header
magic, version, num_vectors, samples_per_vector, bytes_per_sample = struct.unpack(
    "<4sIIII", data[:20]
)
assert magic == b"NN2V" and version == 2
assert samples_per_vector == 784   # 28*28
assert bytes_per_sample == 256
# vector 0
words_per_sample = (bytes_per_sample + 3) // 4
bytes_per_vector = samples_per_vector * words_per_sample * 4
vec0 = data[20:20 + bytes_per_vector]
# Each sample = 256 bytes, byte[c] = channel c (signed int8)
# Build a [28, 28, 256] array indexed as (h, w, c)
def signed_int8(b):
    return b - 256 if b >= 128 else b

# input[r, c, ic]
def get_input(r, c, ic):
    sample_idx = r * IW + c
    b = vec0[sample_idx * 256 + ic]
    return signed_int8(b)

# ---- Load weights ----
wfile = (REPO / "output/weights/node_conv_246_weights.hex").read_text().split()
# Weight bytes in [oc, ic, kh, kw] order.
weights = [int(t, 16) for t in wfile]
assert len(weights) == OC * IC * KH * KW

def get_weight(oc, ic, kh, kw):
    idx = ((oc * IC + ic) * KH + kh) * KW + kw
    return signed_int8(weights[idx])

# ---- Load bias ----
bfile = (REPO / "output/weights/node_conv_246_bias.hex").read_text().split()
biases_raw = [int(t, 16) for t in bfile]
assert len(biases_raw) == OC
def signed_int32(v):
    return v - (1 << 32) if v >= (1 << 31) else v
biases = [signed_int32(v) for v in biases_raw]

# ---- Golden output ----
goldout_path = REPO / "output/goldens/node_conv_246.goldout"
godata = goldout_path.read_bytes()
m, v, nv, spv, bps = struct.unpack("<4sIIII", godata[:20])
assert m == b"NN2V" and v == 2
gvec0 = godata[20:20 + spv * ((bps + 3)//4) * 4]
# samples 0..195 = 14*14, each 256 bytes; sample[r*14+c][ch] = output [r,c,ch]
def get_gold(r, c, ch):
    sample = r * OW + c
    return signed_int8(gvec0[sample * 256 + ch])

# ---- Compute accumulator for OC channels 124, 238 at pixel [1, 4] ----
PIXEL_R, PIXEL_C = 1, 4

for oc_target in [124, 238]:
    acc = 0
    # for kh in 0..KH-1, kw in 0..KW-1, ic in 0..IC-1:
    for kh in range(KH):
        for kw in range(KW):
            in_r = PIXEL_R * SH + kh - PH
            in_c = PIXEL_C * SW + kw - PW
            for ic in range(IC):
                if 0 <= in_r < IH and 0 <= in_c < IW:
                    a = get_input(in_r, in_c, ic)
                else:
                    a = 0
                w = get_weight(oc_target, ic, kh, kw)
                acc += a * w
    bias = biases[oc_target]
    biased = acc + bias
    # round_half_up_toward_pos_inf: floor(x*scale + 0.5)
    # Integer form: scaled = (biased * SCALE_MULT + (1 << (SHIFT-1))) >> SHIFT
    raw = biased * SCALE_MULT + (1 << (SCALE_SHIFT - 1))
    scaled = raw >> SCALE_SHIFT
    if scaled > 127:
        out = 127
    elif scaled < -128:
        out = -128
    else:
        out = scaled
    gold_val = get_gold(PIXEL_R, PIXEL_C, oc_target)
    print(f"== OC channel {oc_target} at pixel [{PIXEL_R},{PIXEL_C}] ==")
    print(f"  acc                 = {acc}")
    print(f"  bias                = {bias}")
    print(f"  biased = acc+bias   = {biased}")
    print(f"  raw = biased*M + H  = {raw}")
    print(f"  scaled = raw >> {SCALE_SHIFT} = {scaled}")
    print(f"  clamped out         = {out}")
    print(f"  GOLD vector value   = {gold_val}")
    print(f"  ENGINE observed value (from observed.hex): see TB output")
    print()

# Show the per-MAC-cycle weights for first few kernel positions for ch 124
print("== Per-MAC-cycle weight bytes the engine should read ==")
print("For OC pass 0 of node_conv_246, each MAC cycle k corresponds to")
print("a unique (ic, kh, kw) tuple. The 256 channel weights are split across")
print("8 banks of 32 weights each. Lane B*32+S = channel B*32+S.")
print()
# For lane 124 = bank 3 slot 28; lane 238 = bank 7 slot 14.
print("Lane 124 in bank 3, slot 28")
print("Lane 238 in bank 7, slot 14")
print()
# For kh=0, kw=0, ic=0, the MAC-cycle index within layer is 0
# So weight_rd_addr = WEIGHT_BASE_WORD + 0 = 11155
# Inside that line, lane 124's weight = weight[124, 0, 0, 0]
# Let's print the first 9 (kh=0,kw=0,ic=0..8) for both channels.
for oc_target in [124, 238]:
    print(f"Lane {oc_target}: first 9 ic at (kh=0,kw=0):")
    vals = [get_weight(oc_target, ic, 0, 0) for ic in range(9)]
    print(f"  {vals}")

# Compute the expected mac_array accumulator order in engine:
# The engine walks ic-innermost, then kw, then kh:
#   for kh in 0..2:
#     for kw in 0..2:
#       for ic in 0..255:
# That matches the python loop. Order is identical so the integer acc value
# should match bit-for-bit.

# Check edge: at pixel [1,4] with stride=2, padding=1:
#   in_r = 1*2 - 1 + kh = 1, 2, 3
#   in_c = 4*2 - 1 + kw = 7, 8, 9
# All in bounds. No padding involved.
print("\nReceptive field for pixel [1,4]:")
for kh in range(KH):
    for kw in range(KW):
        in_r = PIXEL_R * SH + kh - PH
        in_c = PIXEL_C * SW + kw - PW
        in_bounds = (0 <= in_r < IH) and (0 <= in_c < IW)
        print(f"  kh={kh} kw={kw}: in_r={in_r} in_c={in_c} in_bounds={in_bounds}")
