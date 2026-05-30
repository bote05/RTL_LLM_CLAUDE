#!/usr/bin/env python3
"""Audit nn2rtl_top.v for silent width mismatches at module port connections.

For each module instantiation in nn2rtl_top.v, find:
  - the wire declared in nn2rtl_top.v that is connected to each port
  - the port's declared width inside the instantiated module's .v file
  - flag any mismatch (Verilog truncates silently — these are bugs in waiting)

Limitations:
  - Doesn't handle complex expressions on the connection (e.g. {a, b}).
  - Doesn't follow parameterized widths through hierarchies.
  - Pure regex-based, not a real Verilog parser. Good enough for the
    direct `.port(wire)` pattern used throughout nn2rtl_top.v.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


WIRE_DECL_RE = re.compile(
    r"^\s*(?:wire|reg)\s*(\[\s*(\d+)\s*:\s*0\s*\])?\s+([a-zA-Z_]\w*)\s*[;,=]",
    re.MULTILINE,
)
PORT_DECL_RE = re.compile(
    r"^\s*(input|output|inout)\s+(?:wire|reg)?\s*(?:signed\s+)?(\[\s*([^]]+?)\s*\])?\s+([a-zA-Z_]\w*)\s*[,;)]",
    re.MULTILINE,
)
INSTANTIATION_RE = re.compile(
    r"^\s*([a-zA-Z_]\w*)\s+(u_\w+)\s*\(([^)]*(?:\([^)]*\)[^)]*)*)\)\s*;",
    re.MULTILINE | re.DOTALL,
)
PORTMAP_RE = re.compile(
    r"\.(\w+)\s*\(\s*([^)]+?)\s*\)",
)


def parse_widths_from_module(path: Path) -> dict[str, int]:
    """Parse a module .v file and return port -> width (in bits, 1 for scalar)."""
    txt = path.read_text()
    ports: dict[str, int] = {}
    for m in PORT_DECL_RE.finditer(txt):
        _direction = m.group(1)
        range_expr = m.group(3)
        port_name = m.group(4)
        if range_expr is None:
            width = 1
        else:
            # range_expr might be "255:0", "OC*8-1:0", etc. Try the simple
            # number form; otherwise look up a localparam in the same file.
            range_expr = range_expr.strip()
            mm = re.match(r"^(\d+)\s*:\s*0$", range_expr)
            if mm:
                width = int(mm.group(1)) + 1
            else:
                width = resolve_param_width(txt, range_expr)
        ports[port_name] = width
    return ports


def resolve_param_width(module_txt: str, expr: str) -> int:
    """Best-effort resolution of `[<expr>:0]` where <expr> uses localparams."""
    # Build a small symbol table from localparams. We don't handle full Verilog
    # arithmetic; we evaluate simple expressions via Python's eval after
    # substituting names.
    locals_dict: dict[str, int] = {}
    for m in re.finditer(
        r"localparam\s+(?:integer\s+)?([A-Z_]\w*)\s*=\s*(\d+)\s*[;,]", module_txt
    ):
        try:
            locals_dict[m.group(1)] = int(m.group(2))
        except ValueError:
            pass
    # Substitute names then eval. Strip "-1" if present to keep symmetry.
    e = expr
    # Replace symbols (longest first to avoid partial replacement).
    for k in sorted(locals_dict, key=len, reverse=True):
        e = re.sub(rf"\b{k}\b", str(locals_dict[k]), e)
    # Only use safe builtins
    try:
        # noqa: S307 - bounded to numeric expressions after substitution
        val = eval(e, {"__builtins__": {}}, {})
        return int(val) + 1
    except Exception:
        return -1  # unknown


def parse_top_wires(top_txt: str) -> dict[str, int]:
    """Return wire_name -> width from top-level declarations."""
    wires: dict[str, int] = {}
    for m in WIRE_DECL_RE.finditer(top_txt):
        range_msb = m.group(2)
        name = m.group(3)
        width = (int(range_msb) + 1) if range_msb is not None else 1
        wires[name] = width
    return wires


def parse_instantiations(top_txt: str) -> list[tuple[str, str, dict[str, str]]]:
    """Return list of (module_name, instance_name, {port: connected_signal_expr}).

    Only the simple `.port(wire_name)` pattern is matched. Tuples like
    `.port({a, b})` get the raw text as the connected expr.
    """
    instances: list[tuple[str, str, dict[str, str]]] = []
    for m in INSTANTIATION_RE.finditer(top_txt):
        module_name = m.group(1)
        instance_name = m.group(2)
        body = m.group(3)
        if not module_name.startswith("node_"):
            continue
        ports: dict[str, str] = {}
        for pm in PORTMAP_RE.finditer(body):
            port = pm.group(1)
            sig = pm.group(2).strip()
            ports[port] = sig
        instances.append((module_name, instance_name, ports))
    return instances


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    top_path = repo_root / "output" / "rtl" / "nn2rtl_top.v"
    rtl_dir = repo_root / "output" / "rtl"

    top_txt = top_path.read_text()
    top_wires = parse_top_wires(top_txt)
    print(f"[audit] parsed {len(top_wires)} wires in nn2rtl_top.v", file=sys.stderr)

    # Parse port widths for every unique module type we see.
    module_widths: dict[str, dict[str, int]] = {}
    instances = parse_instantiations(top_txt)
    print(f"[audit] found {len(instances)} node_* instantiations", file=sys.stderr)

    mismatch_count = 0
    for module_name, instance_name, ports in instances:
        if module_name not in module_widths:
            mod_path = rtl_dir / f"{module_name}.v"
            if not mod_path.exists():
                continue
            module_widths[module_name] = parse_widths_from_module(mod_path)
        port_widths = module_widths[module_name]

        for port_name, conn_expr in ports.items():
            if port_name not in port_widths:
                continue
            port_w = port_widths[port_name]
            # Try to identify the connected signal width.
            # Strip simple gating like `& spatial_run` to get the bare wire.
            base = re.match(r"^([a-zA-Z_]\w*)\b", conn_expr)
            if not base:
                continue
            wire_name = base.group(1)
            wire_w = top_wires.get(wire_name)
            if wire_w is None:
                # Could be a top-level port (clk/rst_n/s_axis*/m_axis*) - skip
                continue
            if port_w == -1:
                continue  # unresolved
            if port_w != wire_w:
                mismatch_count += 1
                arrow = "->" if port_widths.get(port_name, 0) > 0 else "<-"
                print(
                    f"  MISMATCH {instance_name}({module_name}).{port_name} "
                    f"[{port_w}b] <=> wire {wire_name} [{wire_w}b]"
                )

    print(f"\n[audit] total port-width mismatches: {mismatch_count}", file=sys.stderr)


if __name__ == "__main__":
    main()
