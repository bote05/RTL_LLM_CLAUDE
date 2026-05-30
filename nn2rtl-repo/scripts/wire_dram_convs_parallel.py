#!/usr/bin/env python3
"""Phase A wiring: the 4 stage-4 convs (284/288/292/298) were re-architected from
DRAM-backed (AXI weight ports, free-running output, no ready_out) to on-chip
split-arch (backpressured output WITH a ready_out input port, weights via
$readmemh). This rewires nn2rtl_top.v to match:

  1. Strip the legacy AXI weights_* port connections from each conv instance.
  2. Append a .ready_out(<downstream-accept> & spatial_run) connection.
  3. Delete the now-orphan sim-only axi_weight_rom (u_wrom_N) blocks + their wires.

ready_out targets (downstream consumer's accept signal, same pattern as the
working spatial 3x3/1x1 convs):
  284 -> skid_node_relu_41 ; 292 -> skid_node_relu_44 ; 298 -> skid_node_relu_47
  288 -> add_13 skip FIFO  (block-14 projection; output buffered in u_skip_node_add_13)

Idempotent: skips a conv whose u_wrom_N block is already gone.
Run AFTER apply_dram_conv3x3.py + apply_conv288_decimator.py have emitted the
real (ready_out-port) module .v files.
"""
from __future__ import annotations
import re
from pathlib import Path

TOP = Path("output/rtl/nn2rtl_top.v")

READY_OUT = {
    284: "skid_node_relu_41_ready & spatial_run",
    292: "skid_node_relu_44_ready & spatial_run",
    298: "skid_node_relu_47_ready & spatial_run",
    288: "node_add_13_skip_in_ready & spatial_run",
}

txt = TOP.read_text()
done = 0
for n, ready in READY_OUT.items():
    if f"u_wrom_{n}" not in txt:
        print(f"[skip] conv_{n}: weight rom already removed")
        continue

    # 1+2. Replace the weights_* port block (comment + 8 lines) with .ready_out().
    weights_block = (
        "        // Tie-offs for legacy DRAM AXI4 weights_* read channel.\n"
        f"        .weights_arvalid(w{n}_arvalid),\n"
        f"        .weights_arready(w{n}_arready),\n"
        f"        .weights_araddr(w{n}_araddr),\n"
        f"        .weights_arlen(w{n}_arlen),\n"
        f"        .weights_rvalid(w{n}_rvalid),\n"
        f"        .weights_rready(w{n}_rready),\n"
        f"        .weights_rdata(w{n}_rdata),\n"
        f"        .weights_rlast(w{n}_rlast)\n"
    )
    if txt.count(weights_block) != 1:
        print(f"[FAIL] conv_{n}: weights port block not found exactly once "
              f"(found {txt.count(weights_block)}) — aborting this conv")
        continue
    txt = txt.replace(weights_block, f"        .ready_out({ready})\n", 1)

    # 3. Delete the sim-only ROM block (comment .. rom instantiation `));`).
    rom_re = re.compile(
        rf"\n    // \[sim-only DRAM weight model for conv_{n}\].*?\.rlast\(w{n}_rlast\)\);\n",
        re.DOTALL,
    )
    new_txt, k = rom_re.subn("\n", txt)
    if k != 1:
        print(f"[FAIL] conv_{n}: rom block matched {k} times (need 1) — aborting this conv")
        continue
    txt = new_txt
    done += 1
    print(f"[ok] conv_{n}: stripped AXI weights + rom, ready_out=({ready})")

TOP.write_text(txt)
print(f"\n[written] rewired {done} convs")
