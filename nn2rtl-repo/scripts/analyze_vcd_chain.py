#!/usr/bin/env python3
"""Find when each *_valid_out signal in nn2rtl_top first asserts.

Reads a VCD file, builds the signal-name -> signal-id map from the header,
then scans value-change events for the first '1' on each tracked signal.
Prints them in trace-order so we can see where propagation stops along the
spatial+engine chain.
"""
import sys
from pathlib import Path


def main(vcd_path: str) -> None:
    name_to_id = {}
    id_to_name = {}
    in_header = True
    scope_stack = []

    # Pass 1: read header, build id <-> name map for all *_valid_out signals
    # (and a few others we want to see).
    with open(vcd_path, "r", encoding="utf-8", errors="replace") as f:
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
                in_header = False
                break
            if s.startswith("$var"):
                parts = s.split()
                # $var wire <width> <id> <name> [bitrange] $end
                if len(parts) >= 5:
                    sig_id = parts[3]
                    sig_name = parts[4]
                    full_name = ".".join(scope_stack + [sig_name]) if scope_stack else sig_name
                    # Track valid_out / tvalid / ready_in / m_axis_t* / s_axis_t*
                    track = (
                        sig_name.endswith("_valid_out")
                        or sig_name.endswith("_ready_in")
                        or sig_name.startswith("m_axis_t")
                        or sig_name.startswith("s_axis_t")
                        or "sched_" in sig_name
                        or sig_name == "valid_out"
                        or sig_name == "loaded"
                        or "engine_dispatch" in sig_name
                        or "mac_busy" in sig_name
                        or "wr_req" in sig_name
                    )
                    if track:
                        name_to_id[full_name] = sig_id
                        id_to_name[sig_id] = full_name

    print(f"[vcd] tracking {len(name_to_id)} signals from header", file=sys.stderr)

    # Pass 2: walk value changes; record FIRST cycle each tracked signal goes 1.
    first_high = {}        # name -> cycle
    last_state = {}        # id -> str(value)
    high_count = {}        # name -> count of times it went 1
    current_time = 0
    max_seen_time = 0

    with open(vcd_path, "r", encoding="utf-8", errors="replace") as f:
        # Skip header
        for line in f:
            if line.strip().startswith("$enddefinitions"):
                break

        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                try:
                    current_time = int(s[1:])
                    max_seen_time = current_time
                except ValueError:
                    pass
                continue

            # Value change: <value><id> for 1-bit, b<bits> <id> for multi-bit
            if s[0] in ("0", "1", "x", "z"):
                # 1-bit assignment
                val = s[0]
                sig_id = s[1:]
                if sig_id in id_to_name:
                    name = id_to_name[sig_id]
                    prev = last_state.get(sig_id, "0")
                    if val == "1" and prev != "1":
                        if name not in first_high:
                            first_high[name] = current_time
                        high_count[name] = high_count.get(name, 0) + 1
                    last_state[sig_id] = val

    # 2 events per cycle (rising + falling). Cycle = time / 2.
    cycle = lambda t: t // 2

    print(f"[vcd] max trace time: {max_seen_time} ({cycle(max_seen_time)} cycles)")
    print(f"[vcd] tracked signals: {len(name_to_id)}, signals seen high: {len(first_high)}")
    print()
    print("=== First-time-high for tracked signals (sorted by first-high cycle) ===")

    sorted_items = sorted(first_high.items(), key=lambda kv: kv[1])
    for name, t in sorted_items:
        c = cycle(t)
        cnt = high_count.get(name, 0)
        print(f"  cycle {c:>9}  count={cnt:>6}  {name}")

    print()
    print("=== Signals never went HIGH in trace ===")
    never_high = sorted(set(name_to_id.keys()) - set(first_high.keys()))
    for name in never_high[:200]:
        print(f"  {name}")
    if len(never_high) > 200:
        print(f"  ... and {len(never_high) - 200} more")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: analyze_vcd_chain.py path/to/trace.vcd", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
