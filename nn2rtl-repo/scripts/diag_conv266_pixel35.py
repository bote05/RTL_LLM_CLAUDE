#!/usr/bin/env python3
"""Diagnostic for conv_266 pixel (3,5) channels 9 and 58 mismatches.

Compute golden accumulator and intermediates step-by-step, and compare
to expected golden value.
"""
import struct
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Layer geometry for node_conv_266 (dispatch_index=5).
IC = 256
OC = 256
KH = KW = 3
SH = SW = 1
PH = PW = 1
IH = IW = 14
OH = OW = 14
SCALE_MULT = 1016213653
SCALE_SHIFT = 38

# Load input activations (vector 0).
goldin_path = REPO / "output/goldens/node_conv_266.goldin"
data = goldin_path.read_bytes()
magic, version, num_vectors, samples_per_vector, bytes_per_sample = struct.unpack(
    "<4sIIII", data[:20]
)
assert magic == b"NN2V" and version == 2
assert samples_per_vector == 196, samples_per_vector  # 14*14
assert bytes_per_sample == 256
words_per_sample = (bytes_per_sample + 3) // 4
bytes_per_vector = samples_per_vector * words_per_sample * 4
vec0 = data[20:20 + bytes_per_vector]


def signed_int8(b):
    return b - 256 if b >= 128 else b


def get_input(r, c, ic):
    sample_idx = r * IW + c
    b = vec0[sample_idx * 256 + ic]
    return signed_int8(b)


# Load weights — flat byte stream.
wfile = (REPO / "output/weights/node_conv_266_weights.hex").read_text().split()
weights = [int(t, 16) for t in wfile]
assert len(weights) == OC * IC * KH * KW


def get_weight(oc, ic, kh, kw):
    idx = ((oc * IC + ic) * KH + kh) * KW + kw
    return signed_int8(weights[idx])


# Load bias.
bfile = (REPO / "output/weights/node_conv_266_bias.hex").read_text().split()
biases_raw = [int(t, 16) for t in bfile]
assert len(biases_raw) == OC


def signed_int32(v):
    return v - (1 << 32) if v >= (1 << 31) else v


biases = [signed_int32(v) for v in biases_raw]


# Load golden output.
goldout_path = REPO / "output/goldens/node_conv_266.goldout"
godata = goldout_path.read_bytes()
m, v, nv, spv, bps = struct.unpack("<4sIIII", godata[:20])
assert m == b"NN2V" and v == 2
gvec0 = godata[20:20 + spv * ((bps + 3)//4) * 4]


def get_gold(r, c, ch):
    sample = r * OW + c
    return signed_int8(gvec0[sample * 256 + ch])


# Load observed bytes from engine.
obs_path = REPO / "output/engine_sweep/observed_dispatch05_node_conv_266.hex"
obs_lines = [ln.strip() for ln in obs_path.read_text().splitlines() if ln.strip()]
# 196 lines, 256 bytes each (single OC pass for OC=256).
assert len(obs_lines) == 196


def get_observed(r, c, ch):
    line_idx = r * OW + c
    # 2048-bit hex (512 chars), big-endian. channel 0 = LSByte.
    line = obs_lines[line_idx]
    word_int = int(line, 16)
    word_bytes = word_int.to_bytes(256, byteorder="big")
    # reverse so channel 0 is first
    rev = word_bytes[::-1]
    return signed_int8(rev[ch])


# Compute accumulator for OC channels 9, 58 at pixel [3, 5].
PIXEL_R, PIXEL_C = 3, 5

for oc_target in [9, 58]:
    acc = 0
    # Engine walk: kh outermost, kw, ic innermost (matches python loop).
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
    raw = biased * SCALE_MULT + (1 << (SCALE_SHIFT - 1))
    if biased * SCALE_MULT >= 0:
        scaled = raw >> SCALE_SHIFT
    else:
        scaled = raw >> SCALE_SHIFT  # python's >> is arithmetic for negative ints
    if scaled > 127:
        out = 127
    elif scaled < -128:
        out = -128
    else:
        out = scaled
    gold_val = get_gold(PIXEL_R, PIXEL_C, oc_target)
    obs_val = get_observed(PIXEL_R, PIXEL_C, oc_target)
    print(f"== OC channel {oc_target} at pixel [{PIXEL_R},{PIXEL_C}] ==")
    print(f"  acc                 = {acc}")
    print(f"  bias                = {bias}")
    print(f"  biased = acc+bias   = {biased}")
    print(f"  raw = biased*M + H  = {raw}")
    print(f"  scaled = raw >> {SCALE_SHIFT} = {scaled}")
    print(f"  clamped out         = {out}")
    print(f"  GOLD vector value   = {gold_val}")
    print(f"  OBSERVED (engine)   = {obs_val}")
    diff = obs_val - gold_val
    print(f"  diff (obs - gold)   = {diff}")
    print()


print("Receptive field for pixel [3,5]:")
for kh in range(KH):
    for kw in range(KW):
        in_r = PIXEL_R * SH + kh - PH
        in_c = PIXEL_C * SW + kw - PW
        in_bounds = (0 <= in_r < IH) and (0 <= in_c < IW)
        print(f"  kh={kh} kw={kw}: in_r={in_r} in_c={in_c} in_bounds={in_bounds}")


# Now check what happens if the engine drops a MAC term:
# Hypothesis: engine drops the FIRST or LAST MAC product.
# Try simulating "engine dropped the very first MAC (kh=0, kw=0, ic=0)" for each.
print("\n=== Hypothesis tests for OC=9 (off-by 1) ===")
for oc_target in [9, 58]:
    print(f"\n-- OC {oc_target} --")
    # Full acc
    acc_full = 0
    walk = []
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
                acc_full += a * w
                walk.append((kh, kw, ic, a, w, a * w))
    bias = biases[oc_target]
    gold_val = get_gold(PIXEL_R, PIXEL_C, oc_target)
    obs_val = get_observed(PIXEL_R, PIXEL_C, oc_target)
    print(f"  full acc={acc_full} bias={bias}")
    # If engine dropped a MAC term whose product is p, the engine acc = acc_full - p.
    # The engine biased = acc_full - p + bias.
    # For OC=9: gold=0, obs=-1 -> engine output is 1 less.
    # For OC=58: gold=1, obs=0 -> engine output is 1 less.
    # Both have obs = gold - 1.
    # Search for products p where dropping p yields exactly the wrong output.
    bias_acc = acc_full + bias
    # what acc - p value yields the observed output?
    # try all walk items
    print(f"  trying each MAC dropped:")
    dropped_matches = []
    for k, w_item in enumerate(walk):
        kh, kw, ic, a, w, p = w_item
        engine_acc = acc_full - p
        engine_biased = engine_acc + bias
        raw_e = engine_biased * SCALE_MULT + (1 << (SCALE_SHIFT - 1))
        scaled_e = raw_e >> SCALE_SHIFT
        if scaled_e > 127:
            out_e = 127
        elif scaled_e < -128:
            out_e = -128
        else:
            out_e = scaled_e
        if out_e == obs_val:
            dropped_matches.append((k, kh, kw, ic, a, w, p))
    print(f"  dropping any one MAC matches observed in {len(dropped_matches)} positions out of {len(walk)}")
    # show first 5 and last 5
    for m in dropped_matches[:5]:
        k, kh, kw, ic, a, w, p = m
        print(f"    k={k} (kh={kh},kw={kw},ic={ic}) a={a} w={w} p={p}")
    if len(dropped_matches) > 10:
        print(f"    ...")
    for m in dropped_matches[-5:]:
        k, kh, kw, ic, a, w, p = m
        print(f"    k={k} (kh={kh},kw={kw},ic={ic}) a={a} w={w} p={p}")
