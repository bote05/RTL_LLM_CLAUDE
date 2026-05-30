#!/usr/bin/env python3
"""Print transition timeline for specific 1-bit signals: when they went 0→1 and 1→0."""
import re
import sys
from pathlib import Path


def main(vcd_path: str, names: list[str]) -> None:
    name_to_id: dict[str, str] = {}
    scope_stack: list[str] = []
    with open(vcd_path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("$scope"):
                parts = s.split()
                if len(parts) >= 3:
                    scope_stack.append(parts[2])
                continue
            if s.startswith("$upscope"):
                if scope_stack:
                    scope_stack.pop()
                continue
            if s.startswith("$enddefinitions"):
                break
            if s.startswith("$var"):
                parts = s.split()
                if len(parts) >= 5:
                    sig_id = parts[3]
                    sig_name = parts[4]
                    full_name = ".".join(scope_stack + [sig_name]) if scope_stack else sig_name
                    if full_name in names:
                        name_to_id[full_name] = sig_id

    targets = {name_to_id[n]: n for n in names if n in name_to_id}
    print(f"[timeline] tracking {len(targets)} signals", file=sys.stderr)

    transitions: dict[str, list[tuple[int, str]]] = {sid: [] for sid in targets}
    prev = {sid: "0" for sid in targets}
    last_time = 0
    in_data = False
    with open(vcd_path) as f:
        for line in f:
            s = line.strip()
            if not in_data:
                if "enddefinitions" in s:
                    in_data = True
                continue
            if not s:
                continue
            if s[0] == "#":
                last_time = int(s[1:])
                continue
            if s[0] in ("0", "1"):
                val = s[0]
                sid = s[1:]
                if sid in targets:
                    if val != prev[sid]:
                        transitions[sid].append((last_time, val))
                        prev[sid] = val

    for sid, name in targets.items():
        edges = transitions[sid]
        cycles = [t // 2 for t, _ in edges]
        print(f"\n=== {name} ({len(edges)} edges) ===")
        for i in range(min(30, len(edges))):
            t, v = edges[i]
            print(f"  cycle {t//2:>8}  -> {v}")
        if len(edges) > 30:
            print(f"  ... and {len(edges)-30} more")
            # Also print last 5
            print(f"  -- last 5 --")
            for i in range(max(0, len(edges)-5), len(edges)):
                t, v = edges[i]
                print(f"  cycle {t//2:>8}  -> {v}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: signal_timeline.py VCD name1 ...", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2:])
