#!/usr/bin/env python3
"""Identify engine-top loaders whose input positions are NOT a multiple of
BEATS_PER_WORD (=2048/BUS_W) -> the stream_to_act_bram_bridge drops the final
partial word -> S_WAIT_DRAIN/S_WAIT_LOAD wedge (observed at ldr5/dispatch 4).

For each loader u_ldr_node_conv_<nodeid> (feeds dispatch <nodeid> as its input):
  dispatch_idx = the bridge SLOT whose node == <nodeid>
  positions    = input_h_rom[idx] * input_w_rom[idx]   (the producer's beat count)
  BPW          = 2048 // BUS_W
  need_words   = ceil(positions / BPW)
  partial      = positions % BPW != 0
Prints the full table + the patch list (loaders needing TOTAL_BEATS + ceil size).
"""
import re, math, pathlib

TOP = pathlib.Path("output/mobilenet-v2/rtl/nn2rtl_top_engine.v").read_text()
SCH = pathlib.Path("output/mobilenet-v2/rtl/nn2rtl_scheduler.v").read_text()

def rom(name, src):
    d = {}
    for m in re.finditer(rf"6'd(\d+):\s*{name}\s*=\s*9'd(\d+)", src):
        d[int(m.group(1))] = int(m.group(2))
    return d
ih = rom("input_h_rom", SCH); iw = rom("input_w_rom", SCH)

# bridge SLOT -> nodeid : "engine_output_bridge #(\n .SLOT(N), ... ) u_engine_out_node_conv_<id> ("
slot_node = {}
for m in re.finditer(r"\.SLOT\((\d+)\)[\s\S]{0,400}?u_engine_out_node_conv_(\d+)\s*\(", TOP):
    slot_node[int(m.group(1))] = int(m.group(2))
node_slot = {v: k for k, v in slot_node.items()}

# loaders: BUS_W + TOTAL_BRAM_WORDS + nodeid
loaders = []
for m in re.finditer(r"\.BUS_W\((\d+)\)[\s\S]{0,200}?\.TOTAL_BRAM_WORDS\((\d+)\)\s*\)\s*u_ldr_node_conv_(\d+)\s*\(", TOP):
    busw, words, node = int(m.group(1)), int(m.group(2)), int(m.group(3))
    loaders.append((node, busw, words))

print(f"parsed {len(slot_node)} bridges, {len(loaders)} loaders, ih/iw {len(ih)} dispatches\n")
print(f"{'loader(node)':>14} {'disp':>4} {'BUS_W':>6} {'pos':>6} {'BPW':>4} {'cur_words':>9} {'need(ceil)':>10} {'partial':>8}")
patch = []
for node, busw, words in sorted(loaders, key=lambda x: node_slot.get(x[0], 999)):
    idx = node_slot.get(node)
    if idx is None or idx not in ih:
        print(f"{node:>14} {'?':>4} {busw:>6}  (no dispatch/ROM mapping)")
        continue
    pos = ih[idx]*iw[idx]
    bpw = 2048 // busw if busw < 2048 else 1
    need = math.ceil(pos / bpw)
    partial = (pos % bpw) != 0
    flag = "PARTIAL" if partial else ""
    print(f"{node:>14} {idx:>4} {busw:>6} {pos:>6} {bpw:>4} {words:>9} {need:>10} {flag:>8}")
    if partial or words != need:
        patch.append((node, idx, busw, pos, bpw, words, need))

print("\n=== LOADERS NEEDING FIX (partial-word OR mis-sized) ===")
for node, idx, busw, pos, bpw, words, need in patch:
    print(f"  u_ldr_node_conv_{node} (disp {idx}): TOTAL_BEATS={pos}, TOTAL_BRAM_WORDS {words} -> {need}")
