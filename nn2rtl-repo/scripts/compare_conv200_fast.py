#!/usr/bin/env python3
"""Compare the FAST iverilog conv_200 capture (first NCAP accepted beats) to the
conv_200 golden. Decides the artifact-vs-real verdict:
  multiset(cap) == multiset(golden first N) AND position-wise byte-exact
    => iverilog's REAL conv_200 output matches golden => the Verilator e2e deficit
       was a Verilator artifact (NOT a real line_buf bug).
  large mismatch => REAL line_buf window-delivery bug.

cap line = %064x of data_out[255:0] (MSB-first: hexpair0=byte31 ... hexpair31=byte0).
golden (NN2V) frame0 = raw[20:20+spv*bps] int8, stored byte0..byte31 per 32B beat."""
import struct, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CAP = ROOT / "output/reports_integrated/conv200_iverilog_cap_FAST.hex"
GOLD = ROOT / "output/goldens/node_conv_200.goldout"

def s8(b): return b - 256 if b >= 128 else b

# --- cap: each line -> 32 bytes in data_out order (byte0..byte31) via hexpair reversal
cap_beats = []
for ln in CAP.read_text().split():
    ln = ln.strip()
    if len(ln) != 64: continue
    pairs = [int(ln[k:k+2], 16) for k in range(0, 64, 2)]   # hexpair0=byte31 ... hexpair31=byte0
    cap_beats.append([s8(b) for b in pairs[::-1]])           # reverse -> byte0..byte31
cap = np.array(cap_beats, dtype=np.int32)                    # [N,32]
N = cap.shape[0]
print(f"cap: {N} beats x 32 bytes ({N*32} values)")

# --- golden frame0
raw = GOLD.read_bytes()
_, nv, _, spv, bps = struct.unpack("<4sIIII", raw[:20])
g = np.frombuffer(raw[20:20+spv*bps], dtype=np.int8).astype(np.int32)
print(f"golden: spv={spv} bps={bps} -> {g.size} bytes; comparing first {N} beats ({N*32} values)")
gN = g[:N*32].reshape(N, 32)

# --- position-wise (exact stream alignment)
d = np.abs(cap - gN)
pos_mis = int((d != 0).sum()); pos_max = int(d.max()) if d.size else 0
print(f"\nPOSITION-WISE: mismatch={pos_mis}/{cap.size} ({100*pos_mis/cap.size:.2f}%)  max|err|={pos_max}")

# --- multiset (order-invariant; robust to any in-chain reordering)
ms_cap = np.sort(cap.ravel()); ms_g = np.sort(gN.ravel())
ms_equal = np.array_equal(ms_cap, ms_g)
ms_diff = int((ms_cap != ms_g).sum())
print(f"MULTISET: equal={ms_equal}  sorted-diff={ms_diff}/{cap.size} ({100*ms_diff/cap.size:.2f}%)")
print(f"  cap  range[{cap.min()},{cap.max()}] mean={cap.mean():.3f} #zero={int((cap==0).sum())}")
print(f"  gold range[{gN.min()},{gN.max()}] mean={gN.mean():.3f} #zero={int((gN==0).sum())}")

print("\n=== VERDICT ===")
if ms_equal and pos_mis == 0:
    print("BYTE-EXACT: iverilog real conv_200 output == golden -> the Verilator e2e deficit")
    print("            is a VERILATOR ARTIFACT, not a real line_buf bug.")
elif ms_equal and pos_mis > 0:
    print("MULTISET-EQUAL but position-shifted: values correct, in-chain BEAT ORDER differs")
    print("            (capture/backpressure reordering) -> values are RIGHT; not a compute bug.")
else:
    print(f"REAL DIVERGENCE ({100*ms_diff/cap.size:.1f}% multiset): conv_200's REAL output differs from")
    print("            golden in iverilog too -> a REAL line_buf window-delivery bug.")
