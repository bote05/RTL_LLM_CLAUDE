#!/usr/bin/env python3
"""Phase 1 top-level wiring patcher.

Reads phase1_fanout.json (the inventory) and patches output/rtl/nn2rtl_top.v:

  1. For each loader bridge (u_ldr_node_conv_X), add `.in_ready(ldr_X_in_ready)`
     to its instantiation and declare the wire above it.
  2. For each producer needing A (fan-out or loader-feeding):
       - Compute `<prod>_ready_out` = AND of all its consumers' ready signals.
         Consumer ready signal mapping:
           skid_node_X       -> skid_node_X_ready
           skip_node_add_X   -> node_add_X_skip_in_ready
           u_ldr_node_conv_X -> ldr_node_conv_X_in_ready
       - Declare the wire BEFORE the producer's instantiation.
       - Add `.ready_out(<wire>)` to the producer's instantiation.
       - For each consumer skid_fifo or skip_fifo: change its `in_valid` line
         from `& <own_ready>` to `& <prod>_ready_out`, so the broadcast
         handshake captures every consumer on the same cycle.

  3. For loader-feeding (single consumer) producers: same pattern but only
     one consumer to AND with.

Backup: nn2rtl_top.v.phase1pre is created.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

TOP = Path('output/rtl/nn2rtl_top.v')
BACKUP = Path('output/rtl/nn2rtl_top.v.phase1pre')
INV = Path('output/phase1_fanout.json')

# Producers being patched in this phase (have ready_out port).
# relu_9 is already patched manually (we already added .ready_out there).
PATCHED_PRODUCERS = {
    'node_max_pool2d',
    'node_relu_3', 'node_relu_6', 'node_relu_9',
    'node_relu_12', 'node_relu_15', 'node_relu_18', 'node_relu_21',
    'node_relu_22', 'node_relu_24', 'node_relu_25', 'node_relu_27', 'node_relu_28',
    'node_relu_30', 'node_relu_31', 'node_relu_33', 'node_relu_34',
    'node_relu_36', 'node_relu_37', 'node_relu_39',
    'node_relu_41', 'node_relu_42', 'node_relu_44', 'node_relu_45', 'node_relu_47',
}


def consumer_ready_signal(consumer_inst: str) -> str:
    """Map a consumer instance name to its `ready` signal name."""
    # u_skid_node_<X> -> skid_node_<X>_ready
    if consumer_inst.startswith('u_skid_'):
        return consumer_inst[2:] + '_ready'
    # u_skip_node_add_N -> node_add_N_skip_in_ready
    if consumer_inst.startswith('u_skip_node_add'):
        return consumer_inst[len('u_skip_'):] + '_skip_in_ready'
    # u_ldr_node_conv_X -> ldr_node_conv_X_in_ready
    if consumer_inst.startswith('u_ldr_'):
        return consumer_inst[2:] + '_in_ready'
    raise ValueError(f'Unrecognized consumer: {consumer_inst}')


def consumer_invalid_pattern(consumer_inst: str) -> tuple[re.Pattern, str]:
    """Return (regex_to_find_in_valid_line, replacement_template).

    The replacement uses '\\g<1>{prod_ready}\\g<2>' style — caller fills in
    the producer's ready_out signal name.
    """
    # We want to replace `<prod>_valid_out & spatial_run & <X>_ready` with
    # `<prod>_valid_out & spatial_run & <prod_ready_out>`.
    # The producer's name varies; we use a generic pattern.
    # Find: `(\.in_valid\(\s*\w+_valid_out & spatial_run & )<X>_ready(.*?\))`
    # Replace: \g<1><prod_ready>\g<2>
    own_ready = consumer_ready_signal(consumer_inst)
    pat = re.compile(
        rf'(\.in_valid\(\s*\w+_valid_out\s*&\s*spatial_run\s*&\s*){re.escape(own_ready)}(\s*\))',
        re.DOTALL,
    )
    return pat, own_ready


def main() -> None:
    if not BACKUP.exists():
        shutil.copy2(TOP, BACKUP)
        print(f'[backup] {BACKUP.name}')

    inv = json.loads(INV.read_text())
    fanout_producers = inv['fanout_producers']
    loader_bridges = inv['loader_bridges']

    # Build producer -> list of consumers map (covering all PATCHED_PRODUCERS).
    prod_consumers: dict[str, list[str]] = {}
    for f in fanout_producers:
        prod_consumers[f['name']] = list(f['consumers'])
    # Also single-consumer loader producers: relu_22, 25, 28, 31, 34, 37, 41, 44, 47
    # We deduce their consumer (u_ldr_node_conv_X) from the loader_bridges list.
    for b in loader_bridges:
        prod_name = b['producer_valid'].replace('_valid_out', '')
        if prod_name in PATCHED_PRODUCERS and prod_name not in prod_consumers:
            prod_consumers[prod_name] = [b['bridge']]
        elif prod_name in prod_consumers and b['bridge'] not in prod_consumers[prod_name]:
            # add loader bridge as additional consumer for fan-out producers
            prod_consumers[prod_name].append(b['bridge'])

    txt = TOP.read_text()

    # ----- (1) Bridge in_ready: declare wires + add .in_ready() ports -----
    # Strategy: line-by-line. Find each bridge line `) u_ldr_node_conv_X (`,
    # then within the next ~30 lines find `.in_data(...)` and inject after,
    # and declare the wire before the `stream_to_act_bram_bridge #(` header.
    bridge_changes = 0
    for b in loader_bridges:
        bridge = b['bridge']
        wire_name = bridge[2:] + '_in_ready'  # ldr_node_conv_X_in_ready
        if f'.in_ready({wire_name})' in txt:
            print(f'  [skip] {bridge}: .in_ready already wired')
            continue

        # Find the line containing `) <bridge> (`
        bridge_line_re = re.compile(rf'\)\s*{re.escape(bridge)}\s*\(\s*\n')
        bm = bridge_line_re.search(txt)
        if not bm:
            print(f'  [WARN] could not find {bridge} instantiation line')
            continue
        # Walk back to find the preceding `stream_to_act_bram_bridge` header
        header_idx = txt.rfind('stream_to_act_bram_bridge', 0, bm.start())
        if header_idx < 0:
            print(f'  [WARN] no bridge header before {bridge}')
            continue
        header_line_start = txt.rfind('\n', 0, header_idx) + 1
        wire_decl = f'    wire {wire_name};\n'
        # Inject wire decl right before the bridge header
        txt = txt[:header_line_start] + wire_decl + txt[header_line_start:]
        # Re-find the bridge instantiation line (offsets shifted)
        bm = bridge_line_re.search(txt)
        # Find the `.in_data(...)` line within the next ~30 lines
        chunk = txt[bm.end():bm.end() + 1500]
        in_data_match = re.search(r'\.in_data\([^)]*\),\s*\n', chunk)
        if not in_data_match:
            print(f'  [WARN] no .in_data in {bridge} chunk')
            continue
        abs_pos = bm.end() + in_data_match.end()
        txt = txt[:abs_pos] + f'        .in_ready({wire_name}),\n' + txt[abs_pos:]
        bridge_changes += 1
    print(f'[bridges] patched {bridge_changes}/{len(loader_bridges)}')

    # ----- (2) For each producer: compute ready_out, patch instantiation, update consumers -----
    producer_changes = 0
    for prod, consumers in prod_consumers.items():
        if prod not in PATCHED_PRODUCERS:
            continue

        # Compute combined ready expression
        ready_sigs = [consumer_ready_signal(c) for c in sorted(consumers)]
        if len(ready_sigs) == 1:
            ready_expr = ready_sigs[0]
        else:
            ready_expr = ' & '.join(ready_sigs)
        ready_wire = f'{prod}_ready_out_combined'

        # Find the producer's instantiation.
        # Pattern: `<module_type> u_<name> (` where module_type == prod
        # and u_name is also predictable (typically u_<prod>).
        prod_inst_re = re.compile(
            rf'(    )?{re.escape(prod)}\s+(u_{re.escape(prod)})\s*\(',
        )
        m = prod_inst_re.search(txt)
        if not m:
            print(f'  [WARN] cannot find instantiation for {prod}')
            continue

        # Skip if already patched (.ready_out present in this instance).
        # Find the closing `);` of this instantiation.
        inst_start = m.start()
        inst_close_idx = txt.find(');', inst_start)
        if inst_close_idx < 0:
            print(f'  [WARN] cannot find instantiation end for {prod}')
            continue
        inst_body = txt[inst_start:inst_close_idx + 2]
        if '.ready_out(' in inst_body:
            print(f'  [skip] {prod}: .ready_out already present')
            # But maybe the wire declaration / consumer in_valid patches are
            # still missing. We re-do them idempotently below.
        else:
            # Insert .ready_out(<wire>) before .data_out(...) on the last lines.
            # Pattern: `\.valid_out\([^)]+\)` then we want to insert .ready_out before .data_out
            # We'll insert after `.valid_out(...)`.
            valid_out_re = re.compile(r'(\.valid_out\([^)]+\),)\s*\n', re.DOTALL)
            m2 = valid_out_re.search(txt, m.start(), inst_close_idx)
            if not m2:
                print(f'  [WARN] cannot find .valid_out in {prod} instance')
                continue
            # Insert wire declaration BEFORE the instantiation header.
            wire_decl = f'    wire {ready_wire} = {ready_expr};\n'
            # Find the start of the line containing the instantiation header.
            line_start = txt.rfind('\n', 0, m.start()) + 1
            txt = txt[:line_start] + wire_decl + txt[line_start:]
            # Re-find positions (shifted).
            m_new = prod_inst_re.search(txt)
            inst_start_new = m_new.start()
            inst_close_new = txt.find(');', inst_start_new)
            m2_new = valid_out_re.search(txt, inst_start_new, inst_close_new)
            if not m2_new:
                print(f'  [WARN] re-search valid_out failed for {prod}')
                continue
            repl = m2_new.group(1) + f'\n        .ready_out({ready_wire}),\n'
            txt = txt[:m2_new.start()] + repl + txt[m2_new.end():]
            producer_changes += 1

        # Update consumer in_valid lines to use the combined ready.
        # The producer's name in the in_valid expression is `<prod>_valid_out`.
        for cons in consumers:
            cons_pat, _ = consumer_invalid_pattern(cons)
            # Restrict to lines that mention this producer's valid_out
            # to avoid changing unrelated skid_fifos.
            # We need pattern that captures the right in_valid line for THIS consumer.
            # Strategy: find the consumer instantiation, then find its in_valid within.
            # Two acceptable upstream patterns inside this consumer's in_valid:
            #  A) skid/skip consumers:  prod_valid_out & spatial_run & <X>_ready
            #  B) loader bridges:       prod_valid_out & spatial_run
            # Both become:              prod_valid_out & spatial_run & <combined_ready>
            cons_inst_re_a = re.compile(
                rf'(\)\s*{re.escape(cons)}\s*\(.*?\.in_valid\(\s*){re.escape(prod)}_valid_out(\s*&\s*spatial_run\s*&\s*)\w+_(?:in_)?ready(\s*\))',
                re.DOTALL,
            )
            cons_inst_re_b = re.compile(
                rf'(\)\s*{re.escape(cons)}\s*\(.*?\.in_valid\(\s*){re.escape(prod)}_valid_out(\s*&\s*spatial_run)(\s*\))',
                re.DOTALL,
            )
            cons_inst_re = cons_inst_re_a
            m_cons = cons_inst_re_a.search(txt)
            if m_cons:
                repl = m_cons.group(1) + f'{prod}_valid_out' + m_cons.group(2) + ready_wire + m_cons.group(3)
                txt = txt[:m_cons.start()] + repl + txt[m_cons.end():]
                continue
            m_cons = cons_inst_re_b.search(txt)
            if not m_cons:
                print(f'  [WARN] cannot find in_valid for {cons} reading {prod}')
                continue
            # Pattern B: insert ' & ready_wire' before the closing paren.
            repl = m_cons.group(1) + f'{prod}_valid_out' + m_cons.group(2) + f' & {ready_wire}' + m_cons.group(3)
            txt = txt[:m_cons.start()] + repl + txt[m_cons.end():]

    print(f'[producers] patched {producer_changes}')
    TOP.write_text(txt)
    print(f'[written] {TOP}')


if __name__ == '__main__':
    main()
