#!/usr/bin/env python3
"""ENGINE BANK CONSUMPTION GATE for node_conv_246 (first engine-dispatched conv).

Independently verifies that the weight the ENGINE WOULD READ out of the deduped+
nibbled URAM banks equals conv_246's per-OC INT4 weight in the engine's term order.

Chain under test (NOT previously gated end-to-end):
  build_weight_memory_map.py  (PyTorch [oc,ic,kh,kw] -> MAC-cycle x bank x slot)
  dedup_engine_banks.py        (block-copy 14 dispatch blocks; conv_246 base 11155->0)
  nibble_engine_banks.py       (288b/line, low256 INT8-stored -> 144b/line, low128 nibbles)

Engine read model (mac_array.v:83, nn2rtl_top.v:3078):
  mac_weight_bus = {bank7[127:0]..bank0[127:0]}   (1024 bits = 256 lanes * 4b)
  lane L weight  = $signed(mac_weight_bus[L*4 +: 4])
  lane L drives output channel L (acc_out[L*32+:32])
  -> bank b = L//32, slot s = L%32, and per build_weight_memory_map oc_idx=b*32+s=L

Term order within conv_246 (build_weight_memory_map.layer_mac_cycle_image, oc_passes=1):
  for ic in 0..256: for kh in 0..3: for kw in 0..3   => k = ((ic*3)+kh)*3+kw
WEIGHT_BASE (deduped) = 0 (conv_246 is dispatch_idx 0), K_TOTAL = 2304.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path("D:/RTL_LLM_CLAUDE/nn2rtl-repo")
WDIR = ROOT / "output/weights"

MODULE = "node_conv_246"
WEIGHT_BASE = 0      # deduped row offset (dispatch_idx 0 -> weight_base_word_rom 20'd0)
OC, IC, KH, KW = 256, 256, 3, 3
K_TOTAL = IC * KH * KW          # 2304 = 1 oc_pass * 256 * 3 * 3
NUM_BANKS = 8
LANES = 256
SLOTS_PER_BANK = 32             # LANES // NUM_BANKS
WGT_W = 4


def s4(nib: int) -> int:
    """interpret 4-bit value as signed."""
    nib &= 0xF
    return nib - 16 if (nib & 0x8) else nib


def s8(byte: int) -> int:
    byte &= 0xFF
    return byte - 256 if (byte & 0x80) else byte


def load_bank(b: int) -> list[int]:
    """Return list of 144-bit ints, one per row, for bank b (low 128 = 32 nibbles)."""
    p = WDIR / f"uram_weights_bank{b}.mem"
    out = []
    with p.open() as fh:
        for line in fh:
            t = line.strip()
            if not t:
                continue
            out.append(int(t, 16))
    return out


def engine_weight(banks: list[list[int]], row: int, lane: int) -> int:
    """The signed INT4 the engine reads for output-channel `lane` at deduped `row`.

    mac_weight_bus[lane*4 +: 4]; bus = {bank7..bank0} each 128b.
    -> bank = lane//32, nibble position within that bank = (lane%32)*4.
    """
    b = lane // SLOTS_PER_BANK
    slot = lane % SLOTS_PER_BANK
    word = banks[b][row]              # 144-bit, low 128 = 32 nibbles, nibble s at bit s*4
    nib = (word >> (slot * 4)) & 0xF
    return s4(nib)


def golden_weight(ir_data: bytes, oc: int, ic: int, kh: int, kw: int) -> int:
    """conv_246 weight from the layer hex (PyTorch [oc,ic,kh,kw], 1 byte/line).

    The .mem packing stored the low nibble of each INT8 byte as the INT4 value
    (nibble_engine_banks repack_line). So the golden INT4 = signed-4 of low nibble
    of the int8 byte. We verify against the SAME definition the engine consumes.
    """
    idx = ((oc * IC + ic) * KH + kh) * KW + kw
    return ir_data[idx]


def main() -> int:
    # 1) Confirm conv_246 IR shape.
    ir = json.load((ROOT / "output/layer_ir.json").open())
    layer = next(L for L in ir["layers"] if L.get("module_id") == MODULE)
    shp = layer["weight_shape"]
    assert shp == [OC, IC, KH, KW], f"shape {shp} != {[OC,IC,KH,KW]}"
    hex_path = Path(layer["weights_path"])
    if not hex_path.is_file():
        hex_path = WDIR / hex_path.name
    raw = bytearray()
    with hex_path.open() as fh:
        for line in fh:
            t = line.strip()
            if t:
                raw.append(int(t, 16))
    assert len(raw) == OC * IC * KH * KW, f"hex {len(raw)} != {OC*IC*KH*KW}"

    banks = [load_bank(b) for b in range(NUM_BANKS)]
    for b in range(NUM_BANKS):
        assert len(banks[b]) == 39424, f"bank{b} {len(banks[b])} != 39424"

    # 2) Walk every term k and every lane (OC), compare engine-read vs golden.
    mismatches = []
    examples_match = []
    int8_overflow = 0   # bytes whose value doesn't fit signed INT4 (sanity on the source)
    checked = 0
    for k in range(K_TOTAL):
        ic = k // (KH * KW)
        rem = k % (KH * KW)
        kh = rem // KW
        kw = rem % KW
        row = WEIGHT_BASE + k
        for lane in range(LANES):
            oc = lane                       # oc_pass=0, oc_idx = bank*32+slot = lane
            eng = engine_weight(banks, row, lane)
            byte = golden_weight(raw, oc, ic, kh, kw)
            gold_full = s8(byte)            # full INT8 interpretation of the source byte
            gold_int4 = s4(byte & 0xF)      # what nibble pack would store/engine reads
            if (gold_full < -8) or (gold_full > 7):
                int8_overflow += 1
            checked += 1
            if eng != gold_int4:
                if len(mismatches) < 40:
                    mismatches.append({
                        "k": k, "ic": ic, "kh": kh, "kw": kw, "lane(oc)": lane,
                        "row": row, "engine_int4": eng,
                        "golden_low_nibble_int4": gold_int4,
                        "golden_byte": byte, "golden_full_int8": gold_full,
                    })
            elif len(examples_match) < 8 and (k in (0, 1, K_TOTAL // 2, K_TOTAL - 1)):
                examples_match.append({
                    "k": k, "ic": ic, "kh": kh, "kw": kw, "lane(oc)": lane,
                    "row": row, "engine_int4": eng, "golden_int4": gold_int4,
                })

    print(f"=== ENGINE BANK GATE: {MODULE} ===")
    print(f"WEIGHT_BASE(dedup)={WEIGHT_BASE}  K_TOTAL={K_TOTAL}  lanes={LANES}")
    print(f"checked terms*lanes = {checked}")
    print(f"INT8 source bytes outside signed-INT4 range [-8,7]: {int8_overflow}")
    print(f"MISMATCHES (engine_int4 != golden_low_nibble_int4): {len(mismatches)} "
          f"(showing up to 40)")
    for m in mismatches:
        print("  MISMATCH", m)
    print("sample MATCHES:")
    for e in examples_match[:8]:
        print("  match", e)
    n_mis = len(mismatches)
    print(f"\nVERDICT: {'PASS (engine reads correct conv_246 weights)' if n_mis == 0 else 'FAIL'}")
    return 0 if n_mis == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
