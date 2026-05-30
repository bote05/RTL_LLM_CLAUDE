#!/usr/bin/env python3
"""Parse the TRACE log from engine_one_layer_tb (DBG_TRACE_PIXEL) and
compare it against the python golden MAC walk for conv_266 pixel (3,5),
lanes 9 and 58.

The TRACE log lines look like:
  [TRACE] cyc=N seq=K ic_byte_idx_d=I act=A w[9]=W9 p[9]=P9 w[58]=W58 p[58]=P58

We extract (seq, ic_byte_idx_d, act, w9, p9, w58, p58) for the requant pass
and compare to the golden walk (kh, kw, ic) where ic_byte_idx = ic for IC<=256.
"""
import re
import struct
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent
IC = 256
OC = 256
KH = KW = 3
SH = SW = 1
PH = PW = 1
IH = IW = 14
PIXEL_R, PIXEL_C = 3, 5
LANE_A, LANE_B = 9, 58


def signed_int8(b):
    return b - 256 if b >= 128 else b


def signed_int32(v):
    return v - (1 << 32) if v >= (1 << 31) else v


# Load goldin/weights/bias.
data = (REPO / "output/goldens/node_conv_266.goldin").read_bytes()
vec0 = data[20:20 + 196*256]

def get_input(r, c, ic):
    return signed_int8(vec0[(r*IW+c)*256 + ic])

wfile = (REPO / "output/weights/node_conv_266_weights.hex").read_text().split()
weights = [int(t, 16) for t in wfile]

def get_weight(oc, ic, kh, kw):
    return signed_int8(weights[((oc*IC+ic)*KH+kh)*KW+kw])

# Build golden walk: list of (kh, kw, ic, act_byte_idx, act, w_a, w_b, p_a, p_b)
golden = []
for kh in range(KH):
    for kw in range(KW):
        in_r = PIXEL_R*SH + kh - PH
        in_c = PIXEL_C*SW + kw - PW
        for ic in range(IC):
            in_bounds = 0 <= in_r < IH and 0 <= in_c < IW
            if in_bounds:
                a = get_input(in_r, in_c, ic)
            else:
                a = 0
            w_a = get_weight(LANE_A, ic, kh, kw)
            w_b = get_weight(LANE_B, ic, kh, kw)
            golden.append({
                "kh": kh, "kw": kw, "ic": ic,
                "byte_idx": ic,  # IC=256, single chunk, byte_idx == ic
                "act": a, "in_bounds": in_bounds,
                "w_a": w_a, "p_a": a * w_a,
                "w_b": w_b, "p_b": a * w_b,
            })

trace_path = REPO / "output/engine_sweep/trace_conv266.log"
log = trace_path.read_text(errors="replace")
trace_lines = [ln for ln in log.splitlines() if ln.startswith("[TRACE]") and "seq=" in ln and "act=" in ln]

# Parse trace lines.
trace_pattern = re.compile(
    r"\[TRACE\] cyc=(\d+) seq=(\d+) ic_byte_idx_d=(\d+) act=(-?\d+) "
    r"w\[(\d+)\]=(-?\d+) p\[\d+\]=(-?\d+) w\[(\d+)\]=(-?\d+) p\[\d+\]=(-?\d+)"
)
trace_macs = []
for ln in trace_lines:
    m = trace_pattern.match(ln)
    if m:
        cyc, seq, bi, a, lane_a, w_a, p_a, lane_b, w_b, p_b = m.groups()
        trace_macs.append({
            "cyc": int(cyc),
            "seq": int(seq),
            "byte_idx": int(bi),
            "act": int(a),
            "w_a": int(w_a),
            "p_a": int(p_a),
            "w_b": int(w_b),
            "p_b": int(p_b),
        })

print(f"Golden walk MAC count   = {len(golden)} (expected K_TOTAL = {KH*KW*IC})")
print(f"Trace mac_valid_in pulses = {len(trace_macs)}")
print()

# Compare per-cycle.
# Filter trace to only MACs we expect to see for the pixel.
# Each MAC corresponds to a (kh, kw, ic) tuple, in walk order.
# Issue: trace_macs include only mac_valid_in pulses; the address generator
# gates valid on (in_bounds & ~mac_done) — so padding MACs (in_bounds=0)
# are SKIPPED in the trace but PRESENT in the golden walk with a=0.
# So we need to walk golden and skip out-of-bounds positions when consuming
# trace items. But for interior pixel (3,5), in_bounds is always True.
# So lengths should match exactly.

g_iter = iter(golden)
mismatches = []
n_compared = 0

# Each trace seq should map to the next IN-BOUNDS golden walk entry.
g_idx = 0
for t_idx, t in enumerate(trace_macs):
    # find next in-bounds golden
    while g_idx < len(golden) and not golden[g_idx]["in_bounds"]:
        g_idx += 1
    if g_idx >= len(golden):
        mismatches.append((t_idx, "extra trace MAC beyond golden", t))
        continue
    g = golden[g_idx]
    n_compared += 1
    diff = []
    if t["byte_idx"] != g["byte_idx"]:
        diff.append(f"byte_idx t={t['byte_idx']} g={g['byte_idx']}")
    if t["act"] != g["act"]:
        diff.append(f"act t={t['act']} g={g['act']}")
    if t["w_a"] != g["w_a"]:
        diff.append(f"w_a t={t['w_a']} g={g['w_a']}")
    if t["w_b"] != g["w_b"]:
        diff.append(f"w_b t={t['w_b']} g={g['w_b']}")
    if t["p_a"] != g["p_a"]:
        diff.append(f"p_a t={t['p_a']} g={g['p_a']}")
    if t["p_b"] != g["p_b"]:
        diff.append(f"p_b t={t['p_b']} g={g['p_b']}")
    if diff:
        mismatches.append((t_idx, g_idx, g["kh"], g["kw"], g["ic"], diff, t, g))
    g_idx += 1

# Check for missing tail golden positions.
while g_idx < len(golden):
    if golden[g_idx]["in_bounds"]:
        mismatches.append(("missing", g_idx, golden[g_idx]))
    g_idx += 1

print(f"Compared {n_compared} MAC cycles. Mismatches: {len(mismatches)}")

if mismatches:
    print("\nFirst 20 mismatches:")
    for m in mismatches[:20]:
        print("  ", m)
else:
    print("ALL MACs MATCH.")

# Compute acc from trace and compare to golden total.
trace_acc_a = sum(t["p_a"] for t in trace_macs)
trace_acc_b = sum(t["p_b"] for t in trace_macs)
gold_acc_a = sum(g["p_a"] for g in golden)
gold_acc_b = sum(g["p_b"] for g in golden)
print()
print(f"Trace sum p_a (lane {LANE_A}) = {trace_acc_a}")
print(f"Gold  sum p_a (lane {LANE_A}) = {gold_acc_a}")
print(f"Trace sum p_b (lane {LANE_B}) = {trace_acc_b}")
print(f"Gold  sum p_b (lane {LANE_B}) = {gold_acc_b}")

# Extract requant_valid_in line to see actual engine acc.
rq_pattern = re.compile(
    r"\[TRACE\] cyc=(\d+) REQUANT_VALID_IN seq=(\d+) acc\[\d+\]=(-?\d+) acc\[\d+\]=(-?\d+) "
    r"bias\[\d+\]=(-?\d+) bias\[\d+\]=(-?\d+)"
)
for ln in log.splitlines():
    m = rq_pattern.search(ln)
    if m:
        cyc, seq, acc_a, acc_b, bias_a, bias_b = m.groups()
        print(f"\nREQUANT_VALID_IN at cyc={cyc} seq={seq}:")
        print(f"  acc[{LANE_A}] = {acc_a}  bias = {bias_a}")
        print(f"  acc[{LANE_B}] = {acc_b}  bias = {bias_b}")

# Extract REQUANT_VALID_OUT
out_pattern = re.compile(
    r"\[TRACE\] cyc=(\d+) REQUANT_VALID_OUT out\[\d+\]=(-?\d+) out\[\d+\]=(-?\d+)"
)
for ln in log.splitlines():
    m = out_pattern.search(ln)
    if m:
        cyc, out_a, out_b = m.groups()
        print(f"REQUANT_VALID_OUT at cyc={cyc}: out[{LANE_A}]={out_a} out[{LANE_B}]={out_b}")
