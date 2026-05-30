#!/usr/bin/env python3
"""Diagnostic: compute the golden accumulator and intermediates for
node_conv_290 (1x1, IC=2048, OC=512, 7x7) at all the mismatched
(pixel, channel) coordinates and compare to the observed engine
output bytes.
"""
import struct
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Layer geometry for node_conv_290 (from nn2rtl_scheduler_schedule.json dispatch 10).
IC = 2048
OC = 512
KH = KW = 1
SH = SW = 1
PH = PW = 0
IH = IW = 7
OH = OW = 7
SCALE_MULT = 768585439
SCALE_SHIFT = 36


def signed_int8(b):
    return b - 256 if b >= 128 else b


def signed_int32(v):
    return v - (1 << 32) if v >= (1 << 31) else v


# ---- Load input activations (vector 0) ----
goldin_path = REPO / "output/goldens/node_conv_290.goldin"
data = goldin_path.read_bytes()
# 20-byte header
magic, version, num_vectors, samples_per_vector, bytes_per_sample = struct.unpack(
    "<4sIIII", data[:20]
)
assert magic == b"NN2V" and version == 2
assert samples_per_vector == IH * IW, f"samples_per_vector={samples_per_vector}, expected {IH*IW}"
assert bytes_per_sample == IC, f"bytes_per_sample={bytes_per_sample}, expected {IC}"
words_per_sample = (bytes_per_sample + 3) // 4
bytes_per_vector = samples_per_vector * words_per_sample * 4
vec0 = data[20:20 + bytes_per_vector]


# input[r, c, ic]
def get_input(r, c, ic):
    sample_idx = r * IW + c
    b = vec0[sample_idx * bytes_per_sample + ic]
    return signed_int8(b)


# ---- Load weights (OC, IC, KH, KW = 512, 2048, 1, 1) ----
wfile = (REPO / "output/weights/node_conv_290_weights.hex").read_text().split()
weights = [int(t, 16) for t in wfile]
assert len(weights) == OC * IC * KH * KW, f"weights len={len(weights)}"


def get_weight(oc, ic, kh, kw):
    idx = ((oc * IC + ic) * KH + kh) * KW + kw
    return signed_int8(weights[idx])


# ---- Load bias ----
bfile = (REPO / "output/weights/node_conv_290_bias.hex").read_text().split()
biases_raw = [int(t, 16) for t in bfile]
assert len(biases_raw) == OC

biases = [signed_int32(v) for v in biases_raw]


# ---- Golden output ----
goldout_path = REPO / "output/goldens/node_conv_290.goldout"
godata = goldout_path.read_bytes()
m, v, nv, spv, bps = struct.unpack("<4sIIII", godata[:20])
assert m == b"NN2V" and v == 2
gvec0 = godata[20:20 + spv * ((bps + 3)//4) * 4]


def get_gold(r, c, ch):
    sample = r * OW + c
    return signed_int8(gvec0[sample * bps + ch])


# ---- Load observed engine output ----
obs_path = REPO / "output/engine_sweep/observed_dispatch10_node_conv_290.hex"
obs_lines = obs_path.read_text().strip().split('\n')
# Each line is 512 hex chars = 2048 bits = 256 bytes
# Layout: pixel_idx * OC_PASSES + oc_pass_idx
# OC_PASSES = OC / 256 = 2
OC_PASSES = (OC + 255) // 256


def get_observed(r, c, ch):
    pixel_idx = r * OW + c
    oc_pass = ch // 256
    word_idx = pixel_idx * OC_PASSES + oc_pass
    line = obs_lines[word_idx]
    # 256 bytes, channel `ch % 256` is at byte slot
    # Endianness: $fdisplay("%0512h", obs_word). obs_word is 2048-bit value
    # byte0 = LSByte. In hex, that's the LAST 2 hex chars.
    # So byte n is at hex_pos = (len - 2*(n+1))..(len - 2*n)
    slot = ch % 256
    hex_len = len(line)
    byte_hex = line[hex_len - 2*(slot+1): hex_len - 2*slot]
    b = int(byte_hex, 16)
    return signed_int8(b)


def compute_engine_output(r, c, oc_target):
    """Compute what the engine SHOULD produce using sign-aware rounding."""
    acc = 0
    for kh in range(KH):
        for kw in range(KW):
            in_r = r * SH + kh - PH
            in_c = c * SW + kw - PW
            for ic in range(IC):
                if 0 <= in_r < IH and 0 <= in_c < IW:
                    a = get_input(in_r, in_c, ic)
                else:
                    a = 0
                w = get_weight(oc_target, ic, kh, kw)
                acc += a * w
    bias = biases[oc_target]
    biased = acc + bias
    scaled = biased * SCALE_MULT
    # Sign-aware rounding (engine):
    HALF = 1 << (SCALE_SHIFT - 1)
    if scaled < 0:
        sign_bias = HALF - 1
    else:
        sign_bias = HALF
    raw_eng = scaled + sign_bias
    # Arithmetic shift right (floors toward -inf in Python for negatives)
    scaled_eng = raw_eng >> SCALE_SHIFT
    if scaled_eng > 127:
        out_eng = 127
    elif scaled_eng < -128:
        out_eng = -128
    else:
        out_eng = scaled_eng

    # Golden rounding (unconditional +HALF, floor):
    raw_gold = scaled + HALF
    scaled_gold = raw_gold >> SCALE_SHIFT
    if scaled_gold > 127:
        out_gold = 127
    elif scaled_gold < -128:
        out_gold = -128
    else:
        out_gold = scaled_gold

    return acc, biased, scaled, out_eng, out_gold


# ---- Iterate through the mismatch list ----
result = json.loads((REPO / "output/engine_sweep/result_dispatch10_node_conv_290.json").read_text())
mismatches = result["first_mismatches"]

print(f"=== conv_290 diagnostic ===")
print(f"Geometry: IC={IC} OC={OC} K={KH}x{KW} S={SH}x{SW} P={PH}x{PW} IH={IH} IW={IW} OH={OH} OW={OW}")
print(f"SCALE_MULT={SCALE_MULT} SCALE_SHIFT={SCALE_SHIFT}")
print()
HALF = 1 << (SCALE_SHIFT - 1)
print(f"HALF = 1 << {SCALE_SHIFT - 1} = {HALF}")
print()

# Examine each reported mismatch
for mm in mismatches:
    r = mm["pixel_row"]
    c = mm["pixel_col"]
    ch = mm["channel"]
    exp = mm["expected_s"]
    got = mm["got_s"]

    acc, biased, scaled, out_eng, out_gold = compute_engine_output(r, c, ch)
    obs = get_observed(r, c, ch)
    gold = get_gold(r, c, ch)
    print(f"pixel=({r},{c}) ch={ch}  exp={exp} got={got}  obs={obs} gold={gold}")
    print(f"  acc={acc}  biased={biased}  scaled={scaled}")
    print(f"  engine-sim={out_eng} (sign-aware bias)  gold-sim={out_gold} (always +HALF)")
    if obs == out_eng and gold == out_gold:
        if out_eng != out_gold:
            print(f"  --> ROUNDING TIE: engine={out_eng} gold={out_gold} (sign-aware vs always+HALF)")
        else:
            print(f"  --> NO TIE: engine matches gold; mismatch comes from elsewhere")
    else:
        print(f"  --> SIM MISMATCH: my engine-sim={out_eng} but obs={obs}; my gold-sim={out_gold} but gold={gold}")
    print()
