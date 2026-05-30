#!/usr/bin/env python3
"""Analyze the NATIVE e2e relu_48 mismatch PATTERN (reliable: native forward pass,
full frame, real m_axis output). Localizes the ~2.7% deficit by clustering:
  - by PIXEL (49 spatial, 7x7)  -> spatial clustering => padding/edge/pixel-drop bug
  - by CHANNEL (2048)           -> channel clustering => specific OC / conv / engine bug
  - by error magnitude          -> peak-loss vs uniform
beat -> pixel=beat//64, tile=beat%64, channel=tile*32+byte (64 tiles/pixel, 32 ch/tile)."""
import struct, glob
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DUMP = ROOT / "output/reports_integrated/relu48_native_dump.bin"
GOLD = glob.glob(str(ROOT / "output/goldens/contracts/node_relu_48_*/node_relu_48.goldout"))[0]

cap = np.frombuffer(DUMP.read_bytes(), dtype=np.int8).astype(np.int32)
raw = Path(GOLD).read_bytes()
_, nv, _, spv, bps = struct.unpack("<4sIIII", raw[:20])
g = np.frombuffer(raw[20:20+spv*bps], dtype=np.int8).astype(np.int32)
n = min(cap.size, g.size); cap = cap[:n]; g = g[:n]
NB, BPS = n // bps, bps           # beats, bytes/beat
print(f"relu_48: {NB} beats x {BPS} B = {n} values; spv={spv} bps={bps}")

cap2 = cap.reshape(NB, BPS); g2 = g.reshape(NB, BPS)
diff = (cap2 != g2)
err = np.abs(cap2 - g2)
tot_mis = int(diff.sum())
print(f"\nTOTAL mismatch={tot_mis}/{n} ({100*tot_mis/n:.2f}%)  max|err|={int(err.max())}  mean|err@mis|={err[diff].mean():.2f}")

# map beat -> pixel (49), channel(2048)
beats = np.arange(NB)
pix_of_beat = beats // 64
# per-PIXEL mismatch
pix_mis = np.zeros(49, dtype=int)
for b in range(NB):
    pix_mis[pix_of_beat[b]] += int(diff[b].sum())
print("\n=== by PIXEL (49 spatial, 7x7) ===")
print("  mismatches/pixel:", pix_mis.tolist())
nzpix = np.nonzero(pix_mis)[0]
print(f"  pixels-with-error: {len(nzpix)}/49  -> {'SPATIALLY CLUSTERED' if len(nzpix)<=12 else 'spatially diffuse'}")
if len(nzpix): print(f"  worst pixels: {sorted(zip(pix_mis[nzpix], nzpix.tolist()), reverse=True)[:8]}")

# per-CHANNEL mismatch (channel = (beat%64)*32 + byte)
ch_mis = np.zeros(2048, dtype=int)
for b in range(NB):
    base = (b % 64) * 32
    for j in np.nonzero(diff[b])[0]:
        ch_mis[base + j] += 1
print("\n=== by CHANNEL (2048) ===")
nzch = np.nonzero(ch_mis)[0]
print(f"  channels-with-error: {len(nzch)}/2048  -> {'CHANNEL-CLUSTERED' if len(nzch)<=128 else 'channel-diffuse'}")
if len(nzch): print(f"  worst channels: {sorted(zip(ch_mis[nzch], nzch.tolist()), reverse=True)[:10]}")

print("\n=== HINT ===")
sp = len(nzpix) <= 12; ch = len(nzch) <= 128
if sp and not ch: print("  SPATIAL cluster -> upstream padding/edge/pixel-drop/line_buf bug (spatial layers).")
elif ch and not sp: print("  CHANNEL cluster -> specific OC: trace channel->producing conv (engine dispatch or spatial OC).")
elif sp and ch: print("  SPATIAL+CHANNEL cluster -> a specific (pixel,OC) region; trace both.")
else: print("  DIFFUSE -> systematic rounding/scale/saturation; revisit requant or a broadly-used op.")
