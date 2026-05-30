#!/usr/bin/env python3
"""Check the engine's dispatch-0 (conv_246) config-write sequence + weight reads
captured in-chain, to pinpoint the engine in-chain bug:
  - config writes  -> compare to expected (scheduler ROM-derived). A missing/wrong
                      accepted write = config-delivery bug.
  - weight reads   -> compare to URAM .mem content at the read addr. A mismatch =
                      URAM-read/banking bug; a match = engine reads correct weights.
"""
from __future__ import annotations
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "output/reports_integrated/verilator_nn2rtl_top_probe"

# expected dispatch-0 config writes (addr -> 32-bit data), conv_246 geometry
EXP_CFG = {
    0x00: 256, 0x04: 256,
    0x08: (3 << 4) | 3,          # kernel_h<<4 | kernel_w
    0x0C: (2 << 3) | 2,          # stride
    0x10: (1 << 3) | 1,          # padding
    0x14: (28 << 16) | 28,       # input_h<<16 | input_w
    0x18: (14 << 16) | 14,       # output_h<<16 | output_w
    0x1C: 11155,                 # weight_base
    0x20: 31,                    # bias_base
    0x24: 1284434803,            # scale_mult
    0x28: 39,                    # {zero_point<<6 | scale_shift}
    0x34: 8192,                  # act_in_base
    0x38: 4096,                  # act_out_base
}


def check_config():
    p = D / "cfg_writes_d0.txt"
    print("=== CONFIG WRITES (dispatch 0) ===")
    if not p.exists():
        print("  (no cfg_writes_d0.txt)"); return
    writes = {}
    for line in p.read_text().split("\n"):
        line = line.strip()
        if not line:
            continue
        a, d = line.split()
        writes[int(a, 16)] = int(d, 16)   # last write per addr wins
    bad = 0
    for addr, exp in sorted(EXP_CFG.items()):
        got = writes.get(addr)
        ok = (got == exp)
        if not ok:
            bad += 1
        print(f"  0x{addr:02x}: got={got if got is None else hex(got)}  exp={hex(exp)}  {'OK' if ok else '<<< MISMATCH/MISSING'}")
    extra = set(writes) - set(EXP_CFG)
    if extra:
        print("  extra writes:", {hex(a): hex(writes[a]) for a in extra})
    print(f"  config mismatches: {bad}/{len(EXP_CFG)}  (0 => config delivery correct)")


def load_bank_low256(name):
    # each line = 72 hex chars (288-bit); low 256 bits = rightmost 64 hex chars
    lines = (ROOT / f"output/weights/{name}").read_text().split()
    return lines


def check_weights(latency="d1"):
    p = D / f"engreads_weights_{latency}.bin"
    print(f"\n=== WEIGHT READS latency={latency} (dispatch 0) vs URAM .mem ===")
    if not p.exists():
        print(f"  (no engreads_weights_{latency}.bin)"); return
    raw = p.read_bytes(); rec = 4 + 256
    reads = {}
    for i in range(len(raw) // rec):
        off = i * rec
        addr = struct.unpack_from("<I", raw, off)[0]
        reads[addr] = raw[off + 4:off + 4 + 256]   # 2048-bit data as 256 bytes
    print(f"  captured {len(reads)} unique weight addrs")
    banks = [load_bank_low256(f"uram_weights_bank{k}.mem") for k in range(8)]
    depth = min(len(b) for b in banks)
    match = mismatch = oor = 0
    first = []
    for addr, data in sorted(reads.items()):
        a = addr & 0x1FFFF   # weight_bank_rd_addr = engine_weight_rd_addr[16:0]
        if a >= depth:
            oor += 1; continue
        # expected 2048-bit = {bank7_low256,...,bank0_low256}; bank0 = low bytes 0..31
        exp = bytearray(256)
        for k in range(8):
            hexline = banks[k][a]
            low64 = hexline[-64:]            # rightmost 64 hex = low 256 bits
            bb = bytes.fromhex(low64)        # 32 bytes, big-endian; byte0 = MS
            # low 256 bits little-endian byte layout: data byte j of bank k = bb[31-j]
            for j in range(32):
                exp[k * 32 + j] = bb[31 - j]
        if bytes(exp) == data:
            match += 1
        else:
            mismatch += 1
            if len(first) < 4:
                nd = sum(1 for j in range(256) if exp[j] != data[j])
                first.append((a, nd))
    print(f"  match: {match}  mismatch: {mismatch}  out-of-range: {oor}")
    for a, nd in first:
        print(f"   addr {a}: {nd}/256 bytes differ")
    print("  (match => engine reads correct weights; mismatch => URAM read/banking bug)")


if __name__ == "__main__":
    check_config()
    for lat in ("same", "d1", "d2"):
        check_weights(lat)
