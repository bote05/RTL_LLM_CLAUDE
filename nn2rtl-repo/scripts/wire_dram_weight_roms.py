#!/usr/bin/env python3
"""Wire a sim-only axi_weight_rom to each DRAM-backed conv (284/288/292/298),
replacing the tied-off weights_* AXI read channel so they can load weights in
the Verilator e2e sim. Idempotent.
"""
from __future__ import annotations
import re
from pathlib import Path

TOP = Path("output/rtl/nn2rtl_top.v")
WROOT = (Path.cwd() / "output" / "weights").as_posix()

# conv id -> weight byte count (= num 8-bit values in node_conv_N_weights.hex)
CONVS = {284: 2359296, 288: 2097152, 292: 2359296, 298: 2359296}

txt = TOP.read_text()
done = 0
for n, wbytes in CONVS.items():
    if f"u_wrom_{n}" in txt:
        print(f"[skip] conv_{n}: weight rom already wired")
        continue
    # locate the conv instantiation block
    m = re.search(rf"node_conv_{n}\s+u_node_conv_{n}\s*\(", txt)
    if not m:
        print(f"[WARN] conv_{n}: instantiation not found")
        continue
    close = txt.find(");", m.start())
    block = txt[m.start():close]
    # replace the tied-off weights_* connections with wired ones
    repl = block
    repl = re.sub(r"\.weights_arvalid\(\)",      f".weights_arvalid(w{n}_arvalid)", repl)
    repl = re.sub(r"\.weights_arready\(1'b0\)",   f".weights_arready(w{n}_arready)", repl)
    repl = re.sub(r"\.weights_araddr\(\)",        f".weights_araddr(w{n}_araddr)", repl)
    repl = re.sub(r"\.weights_arlen\(\)",         f".weights_arlen(w{n}_arlen)", repl)
    repl = re.sub(r"\.weights_rvalid\(1'b0\)",    f".weights_rvalid(w{n}_rvalid)", repl)
    repl = re.sub(r"\.weights_rready\(\)",        f".weights_rready(w{n}_rready)", repl)
    repl = re.sub(r"\.weights_rdata\(64'd0\)",    f".weights_rdata(w{n}_rdata)", repl)
    repl = re.sub(r"\.weights_rlast\(1'b0\)",     f".weights_rlast(w{n}_rlast)", repl)
    if repl == block:
        print(f"[WARN] conv_{n}: tie-off pattern not matched (already wired?)")
        continue
    # build the rom instantiation to insert after the conv's `);`
    wpath = f"{WROOT}/node_conv_{n}_weights.hex"
    rom = (
        f"\n    // [sim-only DRAM weight model for conv_{n}]\n"
        f"    wire w{n}_arvalid, w{n}_arready, w{n}_rvalid, w{n}_rready, w{n}_rlast;\n"
        f"    wire [31:0] w{n}_araddr; wire [7:0] w{n}_arlen; wire [63:0] w{n}_rdata;\n"
        f"    axi_weight_rom #(.WEIGHT_BYTES({wbytes}), .WEIGHTS_PATH(\"{wpath}\")) u_wrom_{n} (\n"
        f"        .clk(clk), .rst_n(rst_n),\n"
        f"        .arvalid(w{n}_arvalid), .arready(w{n}_arready), .araddr(w{n}_araddr), .arlen(w{n}_arlen),\n"
        f"        .rvalid(w{n}_rvalid), .rready(w{n}_rready), .rdata(w{n}_rdata), .rlast(w{n}_rlast));\n"
    )
    # splice: replaced block + ");" + rom
    txt = txt[:m.start()] + repl + ");" + rom + txt[close + 2:]
    done += 1
    print(f"[ok] conv_{n}: wired axi_weight_rom (WEIGHT_BYTES={wbytes})")

TOP.write_text(txt)
print(f"\n[written] wired {done} weight roms")
