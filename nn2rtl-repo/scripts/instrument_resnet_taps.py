#!/usr/bin/env python3
"""[RESNET 2953 LOCALIZER] Instrument nn2rtl_top.v with passive $fwrite taps on
EVERY node_relu's data_out, firing on each accepted output beat (valid_out &
ready_out). One e2e run then dumps every relu's real in-chain output to
output/taps/<node>.bin; compare_all_taps.py compares each to its contract
golden and reports the FIRST-DIVERGING node. Because the final 1x1 engine convs
smear any upstream corruption across all 2048 output channels, the e2e relu_48
output is broadly wrong but unlocalizable -- only intermediate taps localize it.

The instrumentation is PURELY ADDITIVE (a marked block before `endmodule`, plus
nothing else) so it cannot change RTL behaviour; --revert restores the backup.
relu_48 is tapped too as a SELF-CHECK: its tap must reproduce the known m_axis
dump (2953 mismatch) or the tap methodology is wrong.

Usage:
  python scripts/instrument_resnet_taps.py            # instrument (backup first)
  python scripts/instrument_resnet_taps.py --revert    # restore backup
"""
from __future__ import annotations
import re, sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOP = ROOT / "output" / "rtl" / "nn2rtl_top.v"
BK = ROOT / "backups" / "resnet_taps" / "nn2rtl_top.v"
BEGIN = "// [TAP-INSTRUMENT BEGIN]"
END = "// [TAP-INSTRUMENT END]"


def parse_relus(text: str):
    """Return list of (node_name, valid_expr, ready_expr, data_expr)."""
    out = []
    # match `node_relu` or `node_relu_<n>` instantiation block up to the closing ');'
    for m in re.finditer(r"\b(node_relu(?:_\d+)?)\s+u_\1\s*\((.*?)\);", text, re.DOTALL):
        node, body = m.group(1), m.group(2)
        v = re.search(r"\.valid_out\(\s*(.*?)\s*\)", body)
        r = re.search(r"\.ready_out\(\s*(.*?)\s*\)", body)
        d = re.search(r"\.data_out\(\s*(.*?)\s*\)", body)
        if not (v and r and d):
            print(f"  WARN {node}: missing port (v={bool(v)} r={bool(r)} d={bool(d)}) -- skipped")
            continue
        out.append((node, v.group(1).strip(), r.group(1).strip(), d.group(1).strip()))
    return out


def gen_block(relus) -> str:
    L = [BEGIN, "// passive in-chain relu taps (additive; see instrument_resnet_taps.py)"]
    for node, _v, _r, _d in relus:
        L.append(f"    integer _tapfd_{node};")
    L.append("    initial begin")
    for node, _v, _r, _d in relus:
        L.append(f'        _tapfd_{node} = $fopen("output/taps/{node}.bin", "wb");')
    L.append("    end")
    L.append("    always @(posedge clk) if (rst_n) begin")
    for node, v, r, d in relus:
        # byte-LSB-first to match NN2V golden byte order: byte b = data[8b+7:8b]
        slices = ", ".join(f"{d}[{8*b+7}:{8*b}]" for b in range(32))
        fmt = "%c" * 32
        L.append(f"        if (({v}) && ({r})) $fwrite(_tapfd_{node}, \"{fmt}\", {slices});")
    L.append("    end")
    L.append(END)
    return "\n".join(L) + "\n"


def main() -> int:
    if "--revert" in sys.argv:
        if not BK.exists():
            print("[taps] no backup to revert"); return 1
        shutil.copy(BK, TOP)
        print(f"[taps] reverted {TOP.name} from backup")
        return 0

    text = TOP.read_text()
    if BEGIN in text:
        print("[taps] already instrumented (marker present); run --revert first")
        return 0
    BK.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(TOP, BK)
    print(f"[taps] backed up -> {BK}")

    relus = parse_relus(text)
    print(f"[taps] found {len(relus)} relu nodes to tap")
    if len(relus) < 40:
        print(f"[taps] ABORT: expected ~49 relus, found {len(relus)} (parse drift)"); return 1

    block = gen_block(relus)
    # nn2rtl_top.v appends helper modules (skip_fifo, engine_output_bridge, ...)
    # AFTER the top. Insert before the endmodule that closes module nn2rtl_top
    # (the FIRST endmodule after its declaration), NOT the file's last one --
    # else the top's wires are out of scope.
    mstart = text.find("module nn2rtl_top")
    if mstart < 0:
        print("[taps] ABORT: 'module nn2rtl_top' not found"); return 1
    idx = text.find("endmodule", mstart)
    if idx < 0:
        print("[taps] ABORT: no endmodule after nn2rtl_top"); return 1
    new = text[:idx] + block + "\n" + text[idx:]
    TOP.write_text(new, newline="\n")
    (ROOT / "output" / "taps").mkdir(parents=True, exist_ok=True)
    print(f"[taps] instrumented {len(relus)} taps; output dir output/taps/ ready")
    print("  taps:", ", ".join(n for n, *_ in relus))
    return 0


if __name__ == "__main__":
    sys.exit(main())
