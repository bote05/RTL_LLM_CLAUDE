#!/usr/bin/env python3
"""Count rising and falling transitions of named 1-bit signals in a VCD."""
import re
import sys
from pathlib import Path


def main(vcd_path: str, names: list[str]) -> None:
    vcd = Path(vcd_path)
    # Map name -> signal id
    name_to_id: dict[str, str] = {}
    scope_stack: list[str] = []
    with vcd.open() as f:
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
                    if sig_name in names or full_name in names:
                        name_to_id.setdefault(sig_name, sig_id)
                        name_to_id.setdefault(full_name, sig_id)

    # Find ids matching requested names
    targets = {}
    for n in names:
        if n in name_to_id:
            targets[name_to_id[n]] = n

    print(f"[count] tracking {len(targets)} ids for {len(names)} names")
    rising = {sid: 0 for sid in targets}
    falling = {sid: 0 for sid in targets}
    prev = {sid: "0" for sid in targets}
    last_time = 0

    in_data = False
    with vcd.open() as f:
        for line in f:
            s = line.strip()
            if not in_data:
                if "enddefinitions" in s:
                    in_data = True
                continue
            if not s:
                continue
            if s[0] == "#":
                try:
                    last_time = int(s[1:])
                except ValueError:
                    pass
                continue
            if s[0] in ("0", "1", "x", "z"):
                val = s[0]
                sid = s[1:]
                if sid in targets:
                    p = prev[sid]
                    if val == "1" and p == "0":
                        rising[sid] += 1
                    elif val == "0" and p == "1":
                        falling[sid] += 1
                    prev[sid] = val

    print(f"[count] last_time={last_time} ({last_time // 2} cycles)")
    for sid, name in targets.items():
        print(f"  {name}: rising={rising[sid]} falling={falling[sid]} final={prev[sid]}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: count_signal_toggles.py VCD name1 name2 ...", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2:])
