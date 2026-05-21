"""Verify that an engine sub-block's port list matches task 00's authoritative PORTS spec.

Used by the Wave 2 review gate. Runs after each engine sub-block lands.

Usage:
    python scripts/check_subblock_ports.py \
        --subblock=mac_array \
        --rtl=output/rtl/engine/mac_array.v \
        --spec=docs/agent_tasks/00_engine_skeleton_spec_PORTS.md

Exits 0 if the sub-block's module port list is exactly the set declared in the
spec for that sub-block. Exits non-zero with a unified diff if not.

The spec document is expected to use markdown sections of the form:

    ## SUBBLOCK: mac_array

    | Port | Direction | Width | Owning sub-block |
    | --- | --- | --- | --- |
    | clk          | input  | 1                 | mac_array |
    | rst_n        | input  | 1                 | mac_array |
    | mac_valid    | input  | 1                 | mac_array |
    | ...

The script extracts the port list for the requested sub-block and parses the
sub-block's Verilog module declaration to extract its declared ports. The two
lists must match exactly (same set, same widths, same directions). Any drift
fails the check.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from dataclasses import dataclass
from pathlib import Path


PORT_HEADER_RE = re.compile(r"##\s+SUBBLOCK:\s+(\S+)\s*$")
TABLE_ROW_RE = re.compile(r"^\|\s*([A-Za-z_][A-Za-z_0-9]*)\s*\|\s*(input|output|inout)\s*\|\s*(.+?)\s*\|")
MODULE_DECL_RE = re.compile(r"^\s*module\s+([A-Za-z_][A-Za-z_0-9]*)\s*\((.*?)\)\s*;", re.DOTALL | re.MULTILINE)
PORT_DECL_RE = re.compile(
    r"\b(input|output|inout)\b(?:\s+wire|\s+reg)?(?:\s+signed)?(?:\s*\[\s*([0-9]+)\s*:\s*0\s*\])?\s+([A-Za-z_][A-Za-z_0-9]*)"
)


@dataclass(frozen=True)
class Port:
    name: str
    direction: str
    width: int  # 0 = scalar, N = [N-1:0]

    def render(self) -> str:
        w = f"[{self.width - 1}:0]" if self.width > 1 else ""
        return f"{self.direction:6s} {w:12s} {self.name}"


def parse_spec(spec_path: Path, subblock: str) -> list[Port]:
    """Extract the expected port set for one sub-block from the spec markdown."""
    text = spec_path.read_text(encoding="utf-8")
    lines = text.split("\n")
    in_block = False
    seen_header = False
    ports: list[Port] = []
    for line in lines:
        m = PORT_HEADER_RE.match(line)
        if m:
            in_block = m.group(1) == subblock
            seen_header = seen_header or in_block
            continue
        if not in_block:
            continue
        # Stop when the next top-level section begins
        if line.startswith("## "):
            in_block = False
            continue
        m2 = TABLE_ROW_RE.match(line)
        if not m2:
            continue
        name = m2.group(1)
        direction = m2.group(2)
        width_raw = m2.group(3).strip()
        if name in ("Port", "---", "name"):
            continue
        if width_raw in ("---", "Width"):
            continue
        width = parse_width(width_raw)
        ports.append(Port(name=name, direction=direction, width=width))
    if not seen_header:
        raise SystemExit(f"spec '{spec_path}' has no '## SUBBLOCK: {subblock}' section")
    return ports


def parse_width(s: str) -> int:
    # Accept "1", "8", "256*8", "256 x 8", or "[N-1:0]" style
    s = s.replace(" ", "")
    if s in ("", "1"):
        return 1
    try:
        return int(s)
    except ValueError:
        pass
    # Try N*M
    m = re.match(r"^([0-9]+)\*([0-9]+)$", s)
    if m:
        return int(m.group(1)) * int(m.group(2))
    # Fallback: try eval-safe arithmetic
    if re.match(r"^[0-9+\-*/]+$", s):
        try:
            return int(eval(s))
        except Exception:
            pass
    raise SystemExit(f"could not parse width '{s}'")


def parse_rtl(rtl_path: Path, expected_module: str) -> list[Port]:
    text = rtl_path.read_text(encoding="utf-8")
    decl = MODULE_DECL_RE.search(text)
    if not decl:
        raise SystemExit(f"no module declaration found in {rtl_path}")
    name = decl.group(1)
    if name != expected_module:
        raise SystemExit(
            f"module name mismatch: file has 'module {name}', expected 'module {expected_module}'"
        )
    body = decl.group(2)
    ports: list[Port] = []
    for m in PORT_DECL_RE.finditer(body):
        direction = m.group(1)
        msb_raw = m.group(2)
        name = m.group(3)
        width = int(msb_raw) + 1 if msb_raw is not None else 1
        ports.append(Port(name=name, direction=direction, width=width))
    return ports


def diff_ports(expected: list[Port], actual: list[Port]) -> list[str]:
    expected_sorted = sorted(expected, key=lambda p: p.name)
    actual_sorted = sorted(actual, key=lambda p: p.name)
    expected_render = [p.render() for p in expected_sorted]
    actual_render = [p.render() for p in actual_sorted]
    return list(
        difflib.unified_diff(
            expected_render, actual_render, fromfile="spec", tofile="rtl", lineterm=""
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subblock", required=True, help="sub-block name, e.g. mac_array")
    ap.add_argument("--rtl", required=True, type=Path)
    ap.add_argument(
        "--spec",
        type=Path,
        default=Path("docs/agent_tasks/00_engine_skeleton_spec_PORTS.md"),
    )
    args = ap.parse_args()

    expected = parse_spec(args.spec, args.subblock)
    actual = parse_rtl(args.rtl, args.subblock)
    diffs = diff_ports(expected, actual)
    if not diffs:
        print(f"OK: {args.subblock} port list matches spec ({len(expected)} ports)")
        return 0
    print(f"FAIL: {args.subblock} port list does not match spec")
    print()
    for d in diffs:
        print(d)
    return 2


if __name__ == "__main__":
    sys.exit(main())
