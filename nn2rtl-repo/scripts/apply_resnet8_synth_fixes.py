#!/usr/bin/env python3
"""Idempotent synthesis-inference fixes for the ResNet-8 nn2rtl_top.v.

The 21 Foundry-generated ResNet-8 modules + the wrapper-local FIFO bodies were
only ever Verilator-verified (e2e gate 8/8, 2,541,691 cyc). Verilator ignores
RAM-inference rules, so some array-memory writes that simulate fine BLOCK Vivado
BRAM/distributed-RAM inference:

  ERROR [Synth 8-3391] Unable to infer a block/distributed RAM for 'mem_reg' ...
  Reason: RAM is sensitive to asynchronous reset signal.

ROOT CAUSE: the memory write `mem[wr_idx] <= in_data;` sits INSIDE an
`always @(posedge clk or negedge rst_n)` reset-sensitized block. A RAM write in
a reset block cannot infer as BRAM/LUTRAM.

BYTE-EXACT-SAFE FIX (the mem array is never reset, so this is timing-neutral):
move ONLY the `mem[wr_idx] <= in_data;` write into a SEPARATE reset-free block
`always @(posedge clk) if (push) mem[wr_idx] <= in_data;`, keep pointer/state
logic in the reset block, and keep the async read `assign out_data = mem[rd_idx];`
and the `(* ram_style = "distributed" *)` attribute untouched. This is identical
to the transform already applied to `skip_fifo` (see knowledge/patterns/
protected/08_common_bugs.md). Verified byte-exact via:
  NN2RTL_VALUE_THREADS=1 NN2RTL_VALUE_XINIT=0 npx tsx scripts/run_resnet8_top_value.ts 0..7

Targets in this script:
  - frame_gate_fifo (mem 2048x128 = 262144 bits, the [Synth 8-3391] blocker)

Idempotent: if the fix is already present, does nothing. Writes a .prefgfifo
backup the first time it changes the file.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOP = REPO / "output" / "resnet8" / "rtl" / "nn2rtl_top.v"

# The pristine (broken) line inside frame_gate_fifo's reset block.
BROKEN = "            if (push) begin mem[wr_idx] <= in_data; wr_ptr <= wr_ptr + 1'b1; end"
# After: pointer stays in the reset block, mem write removed.
FIXED  = "            if (push) begin wr_ptr <= wr_ptr + 1'b1; end"

# The reset-free write block appended just before frame_gate_fifo's endmodule.
# We locate the unique closing of the frame_gate_fifo always block followed by
# endmodule, and inject the split-out block between them.
RESET_BLOCK_TAIL = (
    "                if (push) fill_q <= fill_q + 1'b1;\n"
    "            end\n"
    "        end\n"
    "    end\n"
)
SPLIT_BLOCK = (
    "                if (push) fill_q <= fill_q + 1'b1;\n"
    "            end\n"
    "        end\n"
    "    end\n"
    "\n"
    "    // Array-memory write split out (reset-free) for BRAM/DRAM inference.\n"
    "    always @(posedge clk) begin\n"
    "        if (push) mem[wr_idx] <= in_data;\n"
    "    end\n"
)


# ---- Fix 2: node_conv2d_2 line_buf ram_style hint (synth-speed) -----------------
# node_conv2d_2 has an INLINE 1024x128 (131072-bit) line_buf with a 9-way
# COMBINATIONAL (async) window read. Without a ram_style hint Vivado tries to
# DISSOLVE it into individual registers + global constant-propagation, which makes
# synth_design's optimization pathologically slow (>30 min on this one small
# module). Mapping to distributed LUTRAM (native async read) is the correct
# primitive AND fast. (* ram_style *) is a SYNTHESIS-ONLY attribute -- Verilator
# ignores it, so byte-exact by construction. The other convs use the shared
# line_buf_window module (already ram_style-attributed); only this inline one lacks it.
CONV2 = REPO / "output" / "resnet8" / "rtl" / "node_conv2d_2.v"
CONV2_BROKEN = (
    "    // ---- Activation memory: full 32x32 frame for back-to-back vector safety ----\n"
    "    reg [127:0] line_buf [0:IN_PIXELS-1];\n"
)
CONV2_FIXED = (
    "    // ---- Activation memory: full 32x32 frame for back-to-back vector safety ----\n"
    "    // ram_style=\"distributed\": the 9-way combinational window read (window_sel,\n"
    "    // async) blocks BRAM inference; without a hint Vivado tries to DISSOLVE this\n"
    "    // 131072-bit array into individual registers + constant-propagate, which makes\n"
    "    // synth_design's global optimization pathologically slow. Mapping to LUTRAM\n"
    "    // (native async read) is the correct primitive AND fast. SYNTHESIS-ONLY\n"
    "    // attribute -- Verilator ignores it, so byte-exact by construction.\n"
    "    (* ram_style = \"distributed\" *)\n"
    "    reg [127:0] line_buf [0:IN_PIXELS-1];\n"
)

# ---- Fix 3: node_conv2d_2 prod_reg/sum8_reg ram_style + use_dsp (synth-speed) ----
# node_conv2d_2 is FULLY UNROLLED. prod_reg (2304x16b) and sum8_reg (288x19b) are
# written with compile-time-constant indices -- they are PIPELINE REGISTER arrays,
# not memories. Without a hint Vivado first attempts RAM inference on the indexed
# write, FAILS (multi-port), then "dissolves the memory into individual bits"
# (8-13159) with full constant-propagation -- a pass that STALLS synth_design (the
# stall point is exactly the prod_reg_reg dissolve). ram_style="registers" tells
# Vivado up front to skip that RAM-inference-then-dissolve analysis. Additionally
# use_dsp="no" on prod_reg forces the 2304 constant-coefficient 8x8 multiplies to LUT
# mapping, avoiding the per-multiply DSP-inference + cross-multiply area CSE that made
# synth_design's final optimization take >30 min. SYNTHESIS-ONLY attributes;
# flip-flop/LUT implementation is byte-identical -> Verilator ignores them -> byte-exact.
CONV2_W_BROKEN = (
    "    reg signed [7:0]  window_reg  [0:K_TOTAL-1];      // 144 INT8 window bytes\n"
    "    reg signed [15:0] prod_reg    [0:OC*K_TOTAL-1];   // 2304 signed products\n"
    "    reg signed [18:0] sum8_reg    [0:OC*18-1];        // 288 partial sums of 8 products\n"
)
CONV2_W_FIXED = (
    "    reg signed [7:0]  window_reg  [0:K_TOTAL-1];      // 144 INT8 window bytes\n"
    "    (* ram_style = \"registers\", use_dsp = \"no\" *)\n"
    "    reg signed [15:0] prod_reg    [0:OC*K_TOTAL-1];   // 2304 signed products\n"
    "    (* ram_style = \"registers\" *)\n"
    "    reg signed [18:0] sum8_reg    [0:OC*18-1];        // 288 partial sums of 8 products\n"
)


def fix_frame_gate_fifo() -> bool:
    """Returns True if a change was written."""
    if not TOP.exists():
        print(f"ERROR: {TOP} not found", file=sys.stderr)
        raise SystemExit(1)
    text = TOP.read_text()
    if (BROKEN not in text and FIXED in text
            and "Array-memory write split out (reset-free)" in text):
        print("apply_resnet8_synth_fixes: frame_gate_fifo fix already applied (no-op)")
        return False

    changed = False
    if BROKEN in text:
        text = text.replace(BROKEN, FIXED, 1)
        changed = True
    elif FIXED not in text:
        print("ERROR: could not find frame_gate_fifo reset-block mem-write line.",
              file=sys.stderr)
        raise SystemExit(2)

    if "Array-memory write split out (reset-free)" not in text:
        if RESET_BLOCK_TAIL not in text:
            print("ERROR: frame_gate_fifo reset-block tail anchor not found.",
                  file=sys.stderr)
            raise SystemExit(3)
        head, _, tail = text.rpartition(RESET_BLOCK_TAIL)
        text = head + SPLIT_BLOCK + tail
        changed = True

    if changed:
        backup = TOP.with_suffix(TOP.suffix + ".prefgfifo")
        if not backup.exists():
            backup.write_text(TOP.read_text())
            print(f"apply_resnet8_synth_fixes: wrote backup {backup.name}")
        TOP.write_text(text)
        print("apply_resnet8_synth_fixes: frame_gate_fifo RAM-inference fix APPLIED")
    return changed


def fix_conv2_line_buf() -> bool:
    if not CONV2.exists():
        print(f"ERROR: {CONV2} not found", file=sys.stderr)
        raise SystemExit(4)
    text = CONV2.read_text()
    if CONV2_FIXED.split("\n")[7] in text and 'ram_style = "distributed"' in text \
            and "reg [127:0] line_buf [0:IN_PIXELS-1];" in text:
        print("apply_resnet8_synth_fixes: node_conv2d_2 line_buf ram_style already applied (no-op)")
        return False
    if CONV2_BROKEN not in text:
        print("ERROR: node_conv2d_2 line_buf declaration not found (pristine form).",
              file=sys.stderr)
        raise SystemExit(5)
    backup = CONV2.with_suffix(CONV2.suffix + ".preramstyle")
    if not backup.exists():
        backup.write_text(text)
        print(f"apply_resnet8_synth_fixes: wrote backup {backup.name}")
    CONV2.write_text(text.replace(CONV2_BROKEN, CONV2_FIXED, 1))
    print("apply_resnet8_synth_fixes: node_conv2d_2 line_buf ram_style hint APPLIED")
    return True


def fix_conv2_pipe_regs() -> bool:
    text = CONV2.read_text()
    if ('(* ram_style = "registers", use_dsp = "no" *)\n    reg signed [15:0] prod_reg' in text
            and '(* ram_style = "registers" *)\n    reg signed [18:0] sum8_reg' in text):
        print("apply_resnet8_synth_fixes: node_conv2d_2 prod_reg/sum8_reg attrs already applied (no-op)")
        return False
    if CONV2_W_BROKEN not in text:
        print("ERROR: node_conv2d_2 prod_reg/sum8_reg declarations not found (pristine form).",
              file=sys.stderr)
        raise SystemExit(6)
    backup = CONV2.with_suffix(CONV2.suffix + ".prepiperegs")
    if not backup.exists():
        backup.write_text(text)
        print(f"apply_resnet8_synth_fixes: wrote backup {backup.name}")
    CONV2.write_text(text.replace(CONV2_W_BROKEN, CONV2_W_FIXED, 1))
    print("apply_resnet8_synth_fixes: node_conv2d_2 prod_reg/sum8_reg ram_style=registers APPLIED")
    return True


def main() -> int:
    fix_frame_gate_fifo()
    fix_conv2_line_buf()
    fix_conv2_pipe_regs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
