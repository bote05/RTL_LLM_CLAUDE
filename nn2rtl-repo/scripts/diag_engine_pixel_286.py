#!/usr/bin/env python3
"""Diagnostic for node_conv_286: 1x1 conv, IC=512, OC=2048, HW=7x7.
First mismatches reported by sweep are at pixel [0,6] across many channels:
ch10, 13, 42, 49, 64, 111, 157, 174, 197, 220 — all off-by-one.

This script computes the GOLDEN intermediate values per channel so we can
see if mismatches cluster around half-ties (RTL uses sign-aware rounding
+HALF / +HALF-1; golden uses unconditional +HALF).
"""
import struct
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Layer geometry for node_conv_286.
IC = 512
OC = 2048
KH = KW = 1
SH = SW = 1
PH = PW = 0
IH = IW = 7
OH = OW = 7
SCALE_MULT = 1191746838
SCALE_SHIFT = 38

import sys
PIXEL_R = int(sys.argv[1]) if len(sys.argv) > 1 else 0
PIXEL_C = int(sys.argv[2]) if len(sys.argv) > 2 else 6

# ---- Load input activations (vector 0) ----
goldin_path = REPO / "output/goldens/node_conv_286.goldin"
data = goldin_path.read_bytes()
magic, version, num_vectors, samples_per_vector, bytes_per_sample = struct.unpack(
    "<4sIIII", data[:20]
)
assert magic == b"NN2V" and version == 2
assert samples_per_vector == IH * IW, f"got {samples_per_vector}"
assert bytes_per_sample == IC, f"got {bytes_per_sample}"

words_per_sample = (bytes_per_sample + 3) // 4
bytes_per_vector = samples_per_vector * words_per_sample * 4

# Pick vector 6 — that's where the mismatches are (sample=6 in result json).
# Wait — "sample" in the result is which OUTPUT byte-pixel.
# Actually re-reading: byte 12298 = sample 6, channel 10. sample = byte / bytes_per_oc_sample.
# Let me check format.
# n_out_pixels=49, oc_passes=8, oc_bytes=2048, word_bytes_total=256
# n_total_bytes = 100352 = 49 * 8 * 256 = 7*7 * 8 * 256
# So layout is: [pixel(49)] [oc_pass(8)] [word_bytes(256)]
# byte 12298 = pixel * 8*256 + pass*256 + lane
# 12298 / 2048 = 6 (pixel idx), remainder 138; 138/256=0 (pass), remainder 138; lane=138? But ch=10.
# Hmm let me recompute: 6 * 2048 = 12288. 12298 - 12288 = 10. So pass=0, lane=10 == channel 10. Good.
# pixel idx 6 -> (r=0, c=6) since OW=7.

# But wait — the input vector for testing was likely vector 0 of goldin.
# The OUTPUT goldin/goldout is for the same vector. Let me use vector 0.

def signed_int8(b):
    return b - 256 if b >= 128 else b

vec0 = data[20:20 + bytes_per_vector]

def get_input(r, c, ic):
    sample_idx = r * IW + c
    b = vec0[sample_idx * bytes_per_sample + ic]
    return signed_int8(b)

# ---- Load weights ----
wfile = (REPO / "output/weights/node_conv_286_weights.hex").read_text().split()
weights = [int(t, 16) for t in wfile]
assert len(weights) == OC * IC * KH * KW, f"got {len(weights)} expected {OC*IC*KH*KW}"

def get_weight(oc, ic, kh, kw):
    idx = ((oc * IC + ic) * KH + kh) * KW + kw
    return signed_int8(weights[idx])

# ---- Load bias ----
bfile = (REPO / "output/weights/node_conv_286_bias.hex").read_text().split()
biases_raw = [int(t, 16) for t in bfile]
assert len(biases_raw) == OC

def signed_int32(v):
    return v - (1 << 32) if v >= (1 << 31) else v
biases = [signed_int32(v) for v in biases_raw]

# ---- Golden output ----
goldout_path = REPO / "output/goldens/node_conv_286.goldout"
godata = goldout_path.read_bytes()
m, v, nv, spv, bps = struct.unpack("<4sIIII", godata[:20])
gvec0 = godata[20:20 + spv * ((bps + 3)//4) * 4]

OUT_BPS = bps  # = 2048 here
def get_gold(r, c, ch):
    sample = r * OW + c
    return signed_int8(gvec0[sample * OUT_BPS + ch])

# Check both rounding modes for first 10 mismatch channels:
mismatch_chs = [10, 13, 42, 49, 64, 111, 157, 174, 197, 220]

print(f"== node_conv_286 pixel [{PIXEL_R},{PIXEL_C}] ==")
print(f"  geometry: IC={IC} OC={OC} K={KH}x{KW} HW={IH}x{IW}")
print(f"  scale_mult={SCALE_MULT} scale_shift={SCALE_SHIFT}")
print(f"  HALF = 1<<{SCALE_SHIFT-1} = {1 << (SCALE_SHIFT-1)}")
print()

print(f"{'ch':>4} {'acc':>10} {'bias':>10} {'biased':>10} {'raw':>20} "
      f"{'gold_round':>10} {'rtl_round':>10} {'gold':>8} {'obs':>6} {'diff':>5}")
half = 1 << (SCALE_SHIFT - 1)
half_m1 = half - 1

for oc_target in mismatch_chs:
    acc = 0
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

    # Golden path: unconditional +HALF, arithmetic shift right (floor div)
    raw_pos = biased * SCALE_MULT + half
    gold_round = raw_pos >> SCALE_SHIFT

    # RTL path: sign-aware (+HALF for pos, +HALF-1 for neg)
    scaled = biased * SCALE_MULT
    if scaled < 0:
        rtl_round = (scaled + half_m1) >> SCALE_SHIFT
    else:
        rtl_round = (scaled + half) >> SCALE_SHIFT

    def clamp(x):
        return max(-128, min(127, x))
    gold_clamped = clamp(gold_round)
    rtl_clamped = clamp(rtl_round)

    gold_val = get_gold(PIXEL_R, PIXEL_C, oc_target)
    # Read observed
    obs_lines = [ln.strip() for ln in open(REPO/'output/engine_sweep/observed_dispatch09_node_conv_286.hex') if ln.strip()]
    pixel_idx = PIXEL_R * OW + PIXEL_C
    oc_pass = oc_target // 256
    lane = oc_target % 256
    line = obs_lines[pixel_idx * 8 + oc_pass]
    word_int = int(line, 16)
    obs_bytes = word_int.to_bytes(256, byteorder='big')[::-1]
    obs = signed_int8(obs_bytes[lane])
    diff = gold_val - obs
    print(f"{oc_target:>4} {acc:>10} {bias:>10} {biased:>10} {raw_pos:>20} "
          f"{gold_clamped:>10} {rtl_clamped:>10} {gold_val:>8} {obs:>6} {diff:>5}")
