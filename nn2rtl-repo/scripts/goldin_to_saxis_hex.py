#!/usr/bin/env python3
"""Convert the conv_196 contract goldin (NN2V, 256-bit/sample s_axis input, frame 0)
to a $readmemh hex (one 64-hex-char line per 256-bit beat, MSB-first byte31..byte0)
so an iverilog top-level TB can feed the REAL network input."""
import struct, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
import glob
gp = glob.glob(str(ROOT/"output/goldens/contracts/node_conv_196_*/node_conv_196.goldin"))[0]
raw = open(gp, "rb").read()
m, nv, _, spv, bps = struct.unpack("<4sIIII", raw[:20])
assert bps == 32, f"expected 32 bytes/sample, got {bps}"
data = raw[20:20 + spv * bps]  # frame 0 only
out = ROOT/"output/conv196_saxis_f0.hex"
with open(out, "w") as f:
    for i in range(spv):
        beat = data[i*32:(i+1)*32]              # byte0..byte31 (byte0 = bits[7:0])
        f.write("".join("%02x" % beat[31-k] for k in range(32)) + "\n")  # MSB-first
print(f"wrote {out}: {spv} beats (256-bit), from {gp}")
