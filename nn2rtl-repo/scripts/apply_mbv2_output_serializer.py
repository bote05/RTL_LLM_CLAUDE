#!/usr/bin/env python3
"""FMAX/OOC-FIX 2026-06-08: insert output_serializer between node_linear and m_axis in
nn2rtl_top_engine.v so the top output is a ResNet-style 256-bit STREAM (32 beats) instead of
an 8000-bit parallel bus. Removes the OOC dependency (in-context placement) + the 8000-net bundle.

THREE surgical, anchored edits (the top is patched-not-regenerated -> edit on disk, never regen):
  A. m_axis_tdata port width  7999:0 -> 255:0
  B. node_linear .out_ready_in(m_axis_tready) -> .out_ready_in(ser_ready_out)  (it now feeds the serializer)
  C. replace the 1-beat m_axis assign+beatcount block with the output_serializer instance (drives
     m_axis_tvalid/tdata/tlast; tlast on beat 31).

node_linear is UNCHANGED (still emits its 8000b word) -> the verified MAC is intact; the serializer
is a pure byte-exact reslice. Idempotent + backs up the top.
"""
from __future__ import annotations
import shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOP = REPO / "output" / "mobilenet-v2" / "rtl" / "nn2rtl_top_engine.v"
BACKUP_DIR = REPO / "backups" / "mbv2_output_serializer_20260608"

A_OLD = "    output wire [7999:0]       m_axis_tdata,\n"
A_NEW = "    output wire [255:0]        m_axis_tdata,\n"

B_OLD = ("        .out_ready_in(m_axis_tready),\n"
         "        .valid_out(node_linear_valid_out),\n")
B_NEW = ("        .out_ready_in(ser_ready_out),\n"
         "        .valid_out(node_linear_valid_out),\n")

C_OLD = """    // ----- network output (Fix #4: tlast on final output beat) -----
    assign m_axis_tvalid = node_linear_valid_out;
    assign m_axis_tdata  = node_linear_data_out;
    // Output frame size: C=1000 H=1 W=1, busOut=8000b -> 1 beats
    reg [0:0] m_axis_beat_count;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axis_beat_count <= 1'd0;
        end else if (m_axis_tvalid & m_axis_tready) begin
            if (m_axis_beat_count == 1'd0)
                m_axis_beat_count <= 1'd0;
            else
                m_axis_beat_count <= m_axis_beat_count + 1'd1;
        end
    end
    assign m_axis_tlast = (m_axis_beat_count == 1'd0);
"""

C_NEW = """    // ----- network output: ResNet-style 256b STREAMING (FMAX/OOC-FIX 2026-06-08) -----
    // node_linear's 8000b logit word is serialized to 32x256b beats (byte-exact reslice,
    // tlast on beat 31). Narrows the m_axis pin bus 8000b->256b so the design places
    // IN-CONTEXT (no OOC) like ResNet and deletes the ~8000 parallel output nets.
    wire ser_ready_out;
    output_serializer #(.W_IN(8000), .BEATW(256)) u_output_serializer (
        .clk(clk), .rst_n(rst_n),
        .valid_in(node_linear_valid_out),
        .data_in(node_linear_data_out),
        .ready_out(ser_ready_out),
        .valid_out(m_axis_tvalid),
        .data_out(m_axis_tdata),
        .last_out(m_axis_tlast),
        .ready_in(m_axis_tready)
    );
"""


def main() -> int:
    src = TOP.read_text(encoding="utf-8")
    if "output_serializer" in src:
        print("[output-serializer] already applied; no-op."); return 0
    for name, anchor in (("A:port", A_OLD), ("B:out_ready_in", B_OLD), ("C:m_axis_block", C_OLD)):
        n = src.count(anchor)
        if n != 1:
            print(f"[output-serializer] FATAL: anchor {name} found {n} times (expected 1).", file=sys.stderr)
            return 2
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TOP, BACKUP_DIR / TOP.name)
    src = src.replace(A_OLD, A_NEW, 1).replace(B_OLD, B_NEW, 1).replace(C_OLD, C_NEW, 1)
    TOP.write_text(src, encoding="utf-8", newline="\n")
    print(f"[output-serializer] patched {TOP.relative_to(REPO)} (backup {BACKUP_DIR.relative_to(REPO)})")
    print("[output-serializer] m_axis is now 256b x 32 beats (tlast on beat 31). Update e2e harness compare.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
