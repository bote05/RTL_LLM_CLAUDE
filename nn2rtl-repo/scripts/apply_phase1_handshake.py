#!/usr/bin/env python3
"""Phase 1: Apply option-A held-valid handshake to ReLU modules.

For each node_relu_*.v in the target list, add:
  1. New input port `ready_out`
  2. Modify the sending-phase branch to gate on `ready_out`:
     - When ready_out=1: advance as before
     - When ready_out=0: hold valid_out=1, data_out and out_beat_count unchanged

The template is consistent across all relu modules. We use scoped regex
substitutions on the specific lines that need changing.

Backup files saved as <module>.v.prephase1 next to each file.

USAGE: python scripts/apply_phase1_handshake.py
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

# Producers needing A (from phase1 inventory):
# Fan-out relus + single-consumer loader-feeding relus.
# relu_9 already patched manually as prototype — skip it.
# COMPREHENSIVE-BACKPRESSURE pass (option 2): also patch the remaining
# intermediate single-consumer relus (0,1,2,4,5,7,8,10,11,13,14,16,17,19,20)
# and engine-region relus (23,26,29,32,35,38,40,43,46,48). The wrapper FSM
# is identical across all; the script's scoped regex handles the variants.
TARGETS = [
    0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20,
    23, 26, 29, 32, 35, 38, 40, 43, 46, 48,
]

RTL = Path('output/rtl')

def patch_relu(idx: int) -> bool:
    path = RTL / f'node_relu_{idx}.v'
    if not path.exists():
        # First relu has no _N suffix
        if idx == 0:
            path = RTL / 'node_relu.v'
        else:
            print(f'  [skip] {path.name}: file not found')
            return False
    backup = path.with_suffix(path.suffix + '.prephase1')
    if not backup.exists():
        shutil.copy2(path, backup)

    txt = path.read_text()
    if 'ready_out' in txt:
        print(f'  [skip] {path.name}: already patched (ready_out present)')
        return False

    # 1) Insert ready_out input port after valid_out.
    # The port list ends with `output reg  [255:0] data_out` (varying spacing).
    txt2 = re.sub(
        r'(output reg          valid_out,)\s*\n\s*(output reg  \[255:0\] data_out)',
        r'\1\n    input  wire         ready_out,\n    \2',
        txt,
    )
    if txt2 == txt:
        print(f'  [FAIL] {path.name}: port insert pattern did not match')
        return False

    # 2) Replace the sending else-branch to gate output on ready_out.
    # Original pattern (sending phase body):
    #   end else begin
    #       for (ch = ...) begin
    #           tmp_byte = ...
    #           data_out[ch*8 +: 8] <= ...
    #       end
    #       valid_out <= 1'b1;
    #       if (out_beat_count == BEATS_PER_PIXEL - 1) begin
    #           sending <= 1'b0;
    #           out_beat_count <= ...;
    #           ready_in <= 1'b1;
    #       end else begin
    #           out_beat_count <= out_beat_count + ...;
    #       end
    #   end
    #
    # We wrap the whole body in `if (ready_out) begin ... end else begin valid_out <= 1'b1; end`.
    sending_body_re = re.compile(
        r'(end else begin\s*\n'
        r'(?:\s*//[^\n]*\n)*'    # optional comment lines
        r'                for \(ch = 0; ch < CHANNEL_TILE.*?'
        r'                    out_beat_count <= out_beat_count \+ [^;]+;\s*\n'
        r'                end\s*\n)'
        r'(            end\s*\n        end\s*\n    end\s*\n+endmodule)',
        re.DOTALL,
    )
    # Reconstruct the new sending block. The existing else-branch body is
    # captured in group(1) (between "end else begin" and "end\n" before close).
    m = sending_body_re.search(txt2)
    if not m:
        print(f'  [FAIL] {path.name}: sending-body pattern did not match')
        return False

    old_body = m.group(1)
    tail = m.group(2)

    # Strip the leading "end else begin\n" and trailing "                end\n"
    inner_match = re.match(
        r'end else begin\s*\n(.*?)\n                end\s*\n$',
        old_body,
        re.DOTALL,
    )
    if not inner_match:
        print(f'  [FAIL] {path.name}: inner-body extract failed')
        return False
    inner = inner_match.group(1)

    # inner_match consumed the inner if-else's closing "                end"
    # but did NOT include it in `inner`. We must re-emit it before closing
    # the `if (ready_out)` wrapper, otherwise the resulting code has
    # `... + 4'd1;\nend else begin\n` which Verilog parses as a doubled
    # else attached to the inner if-else.
    new_sending = (
        'end else begin\n'
        '                if (ready_out) begin\n'
        f'{inner}\n'
        '                    end\n'                   # close inner if-else (re-added)
        '                end else begin\n'            # close if(ready_out), open hold-case
        '                    valid_out <= 1\'b1;\n'
        '                end\n'                        # close hold-case
        '            end\n'                            # close outer sending else
        '        end\n'                                # close always else
        '    end\n'                                    # close always
        'endmodule'
    )

    txt3 = txt2[:m.start()] + new_sending + txt2[m.end():]
    path.write_text(txt3)
    print(f'  [ok]   {path.name}')
    return True


def main() -> None:
    print(f'[targets] {len(TARGETS)} relu modules')
    ok = 0
    fail = 0
    for idx in TARGETS:
        if patch_relu(idx):
            ok += 1
        else:
            fail += 1
    print(f'\n[summary] patched: {ok}, skipped/failed: {fail}')


if __name__ == '__main__':
    main()
