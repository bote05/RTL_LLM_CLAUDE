#!/usr/bin/env python3
"""TAIL_PIPE: pipeline the per-OC requant SCALE-MULTIPLY in the ResNet-8 stem
(node_conv2d.v) to raise Fmax.

WHY
---
The post-route critical path on the kpar design is, verbatim from the timing rpt:
    Source:      u_node_conv2d/A[4]__1/C            (a biased_s3 bit)
    Destination: u_node_conv2d/scaled_s4_reg[14]_psdsp/D
    Logic Levels: 57 (CARRY8=28 LUT2=26 ...)  Data Path Delay 13.785ns @14ns
i.e. the single-cycle SCALE MULTIPLY  scaled_s4 = biased_s3 * scale_mult_arr.
biased_s3 is BIASED_W=33 bits, scale is 16 bits -> a 33x16 signed multiply that
Vivado spreads across CARRY8/LUT fabric (it exceeds one DSP48's 27x18), giving a
57-level path. Splitting it across a register cuts the path ~in half -> Fmax up.

HOW (byte-exact)
----------------
Split biased into low-16 (unsigned) + high (arith-shifted, signed):
    biased = (bhi << 16) + blo,  blo = biased[15:0],  bhi = biased >>> 16
    biased * scale = ((bhi*scale) << 16) + (blo*scale)
Register the two partial products (NEW stage s4a), then combine (stage s4b, was
the old scaled_s4). Verified byte-exact for the full 33-bit x 16-bit signed range
(2,000,000 random vectors, 0 mismatches). Adds ONE pipeline cycle; the matching
valid + data delays are inserted so VALID_OUT_LATENCY stays aligned. The stem
already carries a LATENCY_PAD=32 data/valid shift-register downstream, so the
extra cycle just shifts the emit point by 1 -- handshake/value unchanged.

This edits ONLY output/resnet8/rtl/node_conv2d.v (a resnet8-LOCAL file). No shared
rtl_library/* or other-network file is touched.

Idempotent; writes a .pretailpipe backup.

VERIFY: NN2RTL_VALUE_THREADS=1 NN2RTL_VALUE_XINIT=0 \
        npx tsx scripts/run_resnet8_top_value.ts 0   -> result=PASS mismatch_bytes=0
"""
from __future__ import annotations
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
F = REPO / "output" / "resnet8" / "rtl" / "node_conv2d.v"
BACKUP = F.with_suffix(".v.pretailpipe")
MARK = "// TAIL_PIPE: split scale-multiply"

# --- 1) declarations: add the partial-product registers + the extra valid stage ---
DECL_OLD = """    reg signed [BIASED_W-1:0] biased_s3 [0:OC-1];
    reg signed [SCALED_W-1:0] scaled_s4 [0:OC-1];
    reg        [5:0]          shift_s4  [0:OC-1];"""
DECL_NEW = """    reg signed [BIASED_W-1:0] biased_s3 [0:OC-1];
    // TAIL_PIPE: split scale-multiply  scaled = biased*scale  into a pipelined
    // pair of partial products  (bhi*scale)<<16 + (blo*scale)  (byte-exact).
    // Stage s4a registers the two narrow products; the next stage combines.
    // bhi = biased>>>16 (signed, BIASED_W-16 bits); blo = biased[15:0] (unsigned
    // 16) -> the {1'b0,blo} feeds a 17-bit operand. Product widths bounded by
    // SCALED_W; declare wide enough then truncate into the SCALED_W combine.
    reg signed [SCALED_W:0]   pp_hi_s4a [0:OC-1];  // (biased>>>16)*scale
    reg signed [SCALED_W:0]   pp_lo_s4a [0:OC-1];  // biased[15:0]*scale (>=0 side)
    reg        [5:0]          shift_s4a [0:OC-1];
    reg signed [SCALED_W-1:0] scaled_s4 [0:OC-1];
    reg        [5:0]          shift_s4  [0:OC-1];"""

# --- 2) the valid pipe: insert pv_s4b between pv_s4 and pv_s5 (one extra cycle) ---
PV_DECL_OLD = "    reg pv_s1, pv_s2, pv_s3, pv_s4, pv_s5;"
PV_DECL_NEW = "    reg pv_s1, pv_s2, pv_s3, pv_s4, pv_s4a, pv_s5;"

PV_RST_OLD = "            pv_s1 <= 1'b0; pv_s2 <= 1'b0; pv_s3 <= 1'b0; pv_s4 <= 1'b0; pv_s5 <= 1'b0;"
PV_RST_NEW = "            pv_s1 <= 1'b0; pv_s2 <= 1'b0; pv_s3 <= 1'b0; pv_s4 <= 1'b0; pv_s4a <= 1'b0; pv_s5 <= 1'b0;"

# data gains ONE stage (biased_s3 -> pp_s4a -> scaled_s4 -> data_s5, was
# biased_s3 -> scaled_s4 -> data_s5), so the valid pipe gains ONE stage (pv_s4a).
PV_SEQ_OLD = "            pv_s2 <= pv_s1; pv_s3 <= pv_s2; pv_s4 <= pv_s3; pv_s5 <= pv_s4;"
PV_SEQ_NEW = "            pv_s2 <= pv_s1; pv_s3 <= pv_s2; pv_s4 <= pv_s3; pv_s4a <= pv_s4; pv_s5 <= pv_s4a;"

# --- 3) the multiply block: replace single-cycle multiply with the 2-stage split ---
MUL_OLD = """    always @(posedge clk) begin
        for (oc_s = 0; oc_s < OC; oc_s = oc_s + 1) begin
            scaled_s4[oc_s] <= $signed(biased_s3[oc_s]) *
                               $signed(scale_mult_arr[oc_s]);
            shift_s4[oc_s]  <= scale_shift_arr[oc_s];
        end
    end"""
MUL_NEW = """    // TAIL_PIPE stage s4a: the two partial products of the split multiply.
    //   biased = (biased>>>16)<<16 + biased[15:0]
    //   biased*scale = ((biased>>>16)*scale)<<16 + (biased[15:0]*scale)
    // Registering the two narrower products (vs one 33x16) halves the path.
    always @(posedge clk) begin
        for (oc_s = 0; oc_s < OC; oc_s = oc_s + 1) begin
            pp_hi_s4a[oc_s] <= $signed(biased_s3[oc_s] >>> 16) *
                               $signed(scale_mult_arr[oc_s]);
            pp_lo_s4a[oc_s] <= $signed({1'b0, biased_s3[oc_s][15:0]}) *
                               $signed(scale_mult_arr[oc_s]);
            shift_s4a[oc_s] <= scale_shift_arr[oc_s];
        end
    end
    // TAIL_PIPE combine (was the single-cycle scaled_s4 multiply):
    //   scaled = (pp_hi << 16) + pp_lo  == biased*scale  (byte-exact).
    always @(posedge clk) begin
        for (oc_s = 0; oc_s < OC; oc_s = oc_s + 1) begin
            scaled_s4[oc_s] <= ($signed(pp_hi_s4a[oc_s]) <<< 16) + $signed(pp_lo_s4a[oc_s]);
            shift_s4[oc_s]  <= shift_s4a[oc_s];
        end
    end"""


def main():
    txt = F.read_text()
    if MARK in txt:
        print("TAIL_PIPE already applied to node_conv2d.v; skip")
        return
    for old in (DECL_OLD, PV_DECL_OLD, PV_RST_OLD, PV_SEQ_OLD, MUL_OLD):
        if old not in txt:
            raise SystemExit(f"anchor not found:\n{old[:80]}...")
    if not BACKUP.exists():
        BACKUP.write_bytes(F.read_bytes())
    txt = txt.replace(DECL_OLD, DECL_NEW)
    txt = txt.replace(PV_DECL_OLD, PV_DECL_NEW)
    txt = txt.replace(PV_RST_OLD, PV_RST_NEW)
    txt = txt.replace(PV_SEQ_OLD, PV_SEQ_NEW)
    txt = txt.replace(MUL_OLD, MUL_NEW)
    F.write_text(txt)
    print("TAIL_PIPE applied to node_conv2d.v (stem scale-multiply pipelined +1 cycle)")


if __name__ == "__main__":
    main()
