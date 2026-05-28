#!/usr/bin/env python3
"""Generate an early-dump variant of engine_one_layer_tb.v.

The canonical TB only writes output after the FULL conv finishes (engine_done),
which under a gate-level funcsim takes ~454K cycles (~2h). For a fast gate-level
sanity check we instead dump the FIRST N output-pixel writes as they occur and
$finish early (~N*2304 cycles). Since the Verilator ±1 hit 193/196 pixels
(pervasive), the first handful of gate-computed pixels already reveal a bug.

Injects an always-block + early $finish just before engine_one_layer_tb's
endmodule. Writes the variant to build_engine_xsim/ — canonical TB untouched.
N is controlled by `define ED_N_EARLY (xvlog -d ED_N_EARLY=16); defaults to 16.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "tb/engine_one_layer_tb.v"
DST = ROOT / "build_engine_xsim/engine_one_layer_tb_earlydump.v"
N_EARLY = int(os.environ.get("ED_N_EARLY", "16"))

INJECT = r"""
    // ================= [earlydump] AUTO-INJECTED =================
    // Capture the first ED_N_EARLY engine output-pixel writes (pre act-mem),
    // one "pixel_index <512-hex-of-2048b>" line each, then finish early.
`define ED_N_EARLY __N_EARLY__
    integer ed_fh;
    integer ed_cnt;
    reg     ed_open;
    initial begin ed_cnt = 0; ed_open = 1'b0; ed_fh = 0; end
    always @(posedge clk) begin
        if (rst_n && engine_act_out_wr_en) begin
            if (!ed_open) begin
                ed_fh  = $fopen("output/engine_sweep/early_pixels.txt", "w");
                ed_open = 1'b1;
            end
            $fdisplay(ed_fh, "%0d %0512h",
                      (engine_act_out_wr_addr - `CFG_ACT_OUT_BASE),
                      engine_act_out_wr_data);
            ed_cnt = ed_cnt + 1;
            if (ed_cnt >= `ED_N_EARLY) begin
                $fclose(ed_fh);
                $display("[earlydump] captured %0d output pixels -> finishing early at cycle %0d",
                         ed_cnt, cycle_counter);
                $finish;
            end
        end
    end
    // ================= [earlydump] END INJECTED =================
"""


def main() -> None:
    text = SRC.read_text()
    # Inject before the FIRST `endmodule` (which closes engine_one_layer_tb).
    idx = text.find("\nendmodule")
    if idx < 0:
        sys.exit("FATAL: could not find engine_one_layer_tb endmodule")
    inject = INJECT.replace("__N_EARLY__", str(N_EARLY))
    out = text[:idx] + "\n" + inject + text[idx:]
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(out)
    print(f"[earlydump] wrote {DST.relative_to(ROOT)} (N_EARLY={N_EARLY}, before endmodule @ char {idx})")


if __name__ == "__main__":
    main()
