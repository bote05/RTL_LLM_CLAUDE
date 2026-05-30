#!/usr/bin/env python3
"""Cycle-accurate simulation of what the engine SHOULD see for conv_290
at pixel(0,0) pass 1. Reads the actual URAM weight banks and the
activation BRAM (built from goldin) and reproduces what each MAC cycle
should produce. Then compares the per-MAC accumulator buildup."""
import struct
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
IC=2048; OC=512; KH=KW=1
def signed_int8(b):
    return b-256 if b>=128 else b

# Load activation BRAM. The TB writes:
# tb_act_wr_addr = ACT_IN_BASE + (pixel_idx * IC_CHUNKS + chunk_idx)
# Each BRAM word is 2048 bits = 256 bytes (256 channels per chunk).
# So channel ic of pixel p is in BRAM at addr `ACT_IN_BASE + p*8 + ic//256`,
# byte slot ic%256.
#
# We rebuild this in Python from the .goldin.
goldin = (REPO / 'output/goldens/node_conv_290.goldin').read_bytes()
m,v,nv,spv,bps = struct.unpack('<4sIIII', goldin[:20])
assert m == b'NN2V'
vec0 = goldin[20:20+spv*((bps+3)//4)*4]
# act_bram[bram_word_idx][byte_idx] = byte
# Load all 49*8 = 392 words.
act_bram = {}
ACT_IN_BASE = 4096
for p in range(49):
    # IC_BYTES = 2048
    pixel_bytes = vec0[p*2048: (p+1)*2048]
    for chunk in range(8):
        chunk_bytes = pixel_bytes[chunk*256:(chunk+1)*256]
        # If short (last chunk), pad to 256
        if len(chunk_bytes) < 256:
            chunk_bytes = chunk_bytes + bytes(256 - len(chunk_bytes))
        act_bram[ACT_IN_BASE + p*8 + chunk] = chunk_bytes

# Load 8 weight banks. Each bank's .mem has 288-bit lines (72 hex chars).
# For conv_290, weight_base_word = 61843. Mac cycles 0..4095 of conv_290.
# Bank b at mac_cycle = global_cycle: weight for OC = pass*256 + b*32 + slot
banks = []
for b in range(8):
    p = REPO / f'output/weights/uram_weights_bank{b}.mem'
    lines = p.read_text().strip().split('\n')
    banks.append(lines)

WEIGHT_BASE = 61843
# Total mac cycles for conv_290 = 2 passes * 2048 = 4096
# Address into bank = WEIGHT_BASE + mac_cycle

def get_weight_bank(bank, mac_cycle, slot):
    """slot 0..31 within bank. Return signed INT8."""
    line = banks[bank][WEIGHT_BASE + mac_cycle]
    # 72 hex chars: top 8 are pad, low 64 hex = 32 bytes (256 bits)
    # slot 0 = byte 0 = lowest byte = last 2 hex chars of low 64
    low = line[-64:]  # 64 hex chars = 32 bytes
    # slot s: bytes at positions [s*2 : s*2+2] from LSB
    # In hex string written MSB-first: byte 0 = chars [62:64], byte 1 = [60:62], etc.
    byte_hex = low[64 - 2*(slot+1): 64 - 2*slot]
    v = int(byte_hex, 16)
    return signed_int8(v)

# Simulate engine for pixel (0,0), pass 1 (oc 256..511 = lanes 0..255)
# K_TOTAL = 2048, 1x1 conv, so MAC cycles = 2048 (one per ic).
# pass 1 starts at mac_cycle global = 2048 (for conv_290's local indexing)
# Activation reads: addr = ACT_IN_BASE + pixel*8 + ic_chunk_idx
# Byte slot = ic % 256

pixel_r, pixel_c = 0, 0
pixel_idx = pixel_r * 7 + pixel_c

print(f'Pixel ({pixel_r},{pixel_c}) idx={pixel_idx}, pass 1')
print(f'Predicted lane 21 = OC 277 acc trace (first 10 IC, IC 250-265, last 10):')

# Accumulator for each lane
acc = [0] * 256
for k in range(2048):  # ic = k
    ic = k
    chunk = ic // 256
    byte = ic % 256
    addr = ACT_IN_BASE + pixel_idx * 8 + chunk
    act_byte = act_bram[addr][byte]
    act = signed_int8(act_byte)
    # Read weight from each bank, slot
    mac_cycle_global = 2048 + k  # pass 1 starts at local mac 2048
    for lane in range(256):
        bank = lane // 32
        slot = lane % 32
        w = get_weight_bank(bank, mac_cycle_global, slot)
        acc[lane] += act * w
    # Trace some specific lanes
    if (k < 10) or (250 <= k < 266) or (k >= 2038):
        print(f'  ic={k:4d}: act={act:4d} '
              f'w_lane21={get_weight_bank(0, mac_cycle_global, 21):4d} '
              f'w_lane0={get_weight_bank(0, mac_cycle_global, 0):4d} '
              f'acc[21]={acc[21]} acc[0]={acc[0]}')

print()
print(f'Final acc:')
for lane in [0, 1, 2, 4, 14, 21, 24, 33, 44, 47, 64, 75, 120, 146, 173, 185]:
    print(f'  lane={lane} (oc={256+lane}): acc={acc[lane]}')
