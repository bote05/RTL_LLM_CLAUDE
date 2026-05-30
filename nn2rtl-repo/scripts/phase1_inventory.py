#!/usr/bin/env python3
"""Phase 1 inventory: find every producer in nn2rtl_top.v whose valid_out
is referenced as input by 2+ downstream consumers (fan-out producers).

Also lists every loader bridge (u_ldr_*) and its single producer for the
Step 1.5 audit (loader bridges have 1-deep internal skids that can drop
pulses when grant is denied mid-word).

Output: phase1_fanout.json with structure:
  {
    "fanout_producers": [
      {"name": "node_relu_9", "valid_signal": "node_relu_9_valid_out",
       "consumers": ["skid_node_conv_218", "skid_node_conv_224"]}
    ],
    "loader_bridges": [
      {"bridge": "u_ldr_node_conv_246", "producer_valid": "node_relu_22_valid_out"}
    ]
  }
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

TOP = Path('output/rtl/nn2rtl_top.v')
txt = TOP.read_text()

# Find every "(<producer>_valid_out & ...)" reference inside a .in_valid() or
# .valid_in() port connection. We're looking for places where a producer's
# valid is being SAMPLED by a consumer.
#
# Patterns of interest:
#   .in_valid(node_X_valid_out & ... )       <- skid_fifo / skip_fifo input
#   .valid_in(node_X_valid_out & ... )       <- direct module valid input
#   .valid_in(skid_node_X_valid)             <- consumer of a skid output (DOESN'T count
#                                                as a fan-out of the upstream producer)
#
# Each fan-out target gets a "consumer name" — the enclosing skid_fifo / skip_fifo
# / module instance name.

# Match instantiations and capture their range so we can attribute references.
# Simpler approach: walk line by line, track current instance name from the most
# recent "u_<name> (" header.
producer_consumers: dict[str, list[str]] = defaultdict(list)
loader_bridges: list[dict] = []

# Instance header. Two cases:
#   "module_type u_name ("
#   "module_type #(.PARAM(...), ...) u_name ("   (params may span lines but
#                                                 the closing ") u_name (" is
#                                                 typically on one line at end)
# We use a forgiving pattern: "<any chars>) u_<name> (" OR start-of-line module
# instantiation "<word> u_<name> (".
inst_re_simple = re.compile(r'^\s*[\w]+\s+(u_[\w]+)\s*\(', re.M)
inst_re_param  = re.compile(r'\)\s*(u_[\w]+)\s*\(\s*$', re.M)
in_valid_re = re.compile(r'\.(?:in_valid|valid_in)\s*\(\s*([^)]*)\)')

# Walk through the file, tracking last-seen instance name.
lines = txt.splitlines()
current_inst: str | None = None
for line in lines:
    m_inst = inst_re_simple.match(line) or inst_re_param.search(line)
    if m_inst:
        current_inst = m_inst.group(1)
    m_iv = in_valid_re.search(line)
    if m_iv and current_inst:
        expr = m_iv.group(1)
        # Find producer valid_out signal in expr.
        # We want names of the form node_X_valid_out (the module producers).
        # Skip skid_X_valid (those are skid OUTPUTS being read by consumers).
        prod_matches = re.findall(r'(node_[\w]+_valid_out)', expr)
        for prod in prod_matches:
            producer_consumers[prod].append(current_inst)
        # Record loader-bridge producer if applicable
        if current_inst.startswith('u_ldr_'):
            # Find the producer's valid signal — typically node_X_valid_out
            if prod_matches:
                loader_bridges.append({
                    'bridge': current_inst,
                    'producer_valid': prod_matches[0],
                })

# Fan-out producers = ones with 2+ unique consumers.
fanout = []
for prod, cons_list in producer_consumers.items():
    unique = sorted(set(cons_list))
    if len(unique) >= 2:
        producer_name = prod.replace('_valid_out', '')
        fanout.append({
            'name': producer_name,
            'valid_signal': prod,
            'consumers': unique,
        })
fanout.sort(key=lambda x: x['name'])

# De-duplicate loader bridges (each appears once per in_valid line)
seen = set()
unique_bridges = []
for b in loader_bridges:
    k = (b['bridge'], b['producer_valid'])
    if k not in seen:
        seen.add(k)
        unique_bridges.append(b)

result = {
    'fanout_producers': fanout,
    'loader_bridges': unique_bridges,
}

out = Path('output/phase1_fanout.json')
out.write_text(json.dumps(result, indent=2))
print(f'[written] {out}')
print(f'\n=== FAN-OUT PRODUCERS ({len(fanout)}) ===')
for f in fanout:
    print(f"  {f['name']}: -> {', '.join(f['consumers'])}")
print(f'\n=== LOADER BRIDGES ({len(unique_bridges)}) ===')
for b in unique_bridges:
    print(f"  {b['bridge']:35s} <- {b['producer_valid']}")
