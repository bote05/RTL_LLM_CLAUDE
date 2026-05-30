#!/usr/bin/env python3
"""Wire each residual-add instance's NEW .ready_out port in nn2rtl_top.v.

The 16 node_add* modules got an output backpressure port (ready_out) added (BP-FIX:
they were dropping output beats under downstream stall). Each add instance in the top
must now connect .ready_out to its downstream consumer's ready, gated by spatial_run —
so the add HOLDS its output when the downstream skid is full OR during engine dispatches
(spatial_run=0), instead of overrunning.

For each `node_add<N> u_node_add<N> ( ... );` instance:
  - find the consumer: the skip_fifo whose .in_data(...) references node_add<N>_data_out
  - extract that consumer's ready signal from its .in_valid(node_add<N>_valid_out & ... & <X>_ready)
  - insert `.ready_out(<X>_ready & spatial_run),` after the instance's .valid_out(...) line
(idempotent: skips instances that already have .ready_out)

Usage: python scripts/apply_add_ready_out_wiring.py [--apply]   (default = dry-run)
"""
from __future__ import annotations
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOP = ROOT / "output/rtl/nn2rtl_top.v"
APPLY = "--apply" in sys.argv


def main() -> int:
    src = TOP.read_text()
    lines = src.split("\n")
    # add module names: node_add, node_add_1 .. node_add_15
    adds = ["node_add"] + [f"node_add_{i}" for i in range(1, 16)]
    plan = []
    for name in adds:
        # find the consumer ready: a line  .in_valid(<name>_valid_out & ... & <X>_ready)
        # whose instance also has .in_data(<name>_data_out)
        ready_sig = None
        for i, ln in enumerate(lines):
            if f".in_data({name}_data_out" in ln:
                # look back a few lines for the in_valid of this skip_fifo
                for j in range(max(0, i - 4), i + 2):
                    m = re.search(rf"\.in_valid\({name}_valid_out\s*&[^)]*?(\w+_ready)\s*\)", lines[j])
                    if m:
                        ready_sig = m.group(1)
                        break
                if ready_sig:
                    break
        plan.append((name, ready_sig))

    # apply: insert .ready_out after the instance's .valid_out(<name>_valid_out) line
    out = list(lines)
    applied = 0
    for name, ready_sig in plan:
        if ready_sig is None:
            print(f"  [SKIP] {name}: could not find downstream ready")
            continue
        # find the instance's .valid_out(<name>_valid_out) line
        inst_re = re.compile(rf"\.valid_out\({name}_valid_out\)")
        idx = next((k for k, ln in enumerate(out) if inst_re.search(ln)), None)
        if idx is None:
            print(f"  [SKIP] {name}: instance .valid_out line not found")
            continue
        # already wired?
        nearby = "\n".join(out[idx:idx + 3])
        if ".ready_out(" in nearby:
            print(f"  [SKIP] {name}: already has .ready_out")
            continue
        indent = re.match(r"\s*", out[idx]).group(0)
        ins = f"{indent}.ready_out({ready_sig} & spatial_run),   // [BP-FIX] hold output until accepted"
        out.insert(idx + 1, ins)
        applied += 1
        print(f"  [WIRE] {name}: .ready_out({ready_sig} & spatial_run)")

    print(f"\n{'APPLIED' if APPLY else 'DRY-RUN'}: {applied}/{len(adds)} adds wired")
    if APPLY and applied:
        TOP.write_text("\n".join(out))
        print(f"  wrote {TOP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
