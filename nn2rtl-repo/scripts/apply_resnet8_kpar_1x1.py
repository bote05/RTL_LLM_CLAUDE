#!/usr/bin/env python3
"""Re-parallelize the ResNet-8 1x1 stride-2 PROJECTION convs (conv_3, conv_6)
with K_PAR-tap + MP-lane parallelism.

WHY
---
After scripts/apply_resnet8_kpar_convs.py parallelized the six 3x3 convs and the
frame-gate->elastic-FIFO swap let layers overlap, the e2e (~146K cyc) was found
to be gated by the two 1x1 projection convs node_conv2d_3 (~143K) and
node_conv2d_6 (~137K) -- they were still the original SERIAL MP=4 single-multiply
FSM (one weight*tap multiply per cycle). These are pointwise 1x1 stride-2 convs:
per stride-aligned input pixel they MAC over IC taps for each of OC outputs.

This rewrite adds K_PAR parallel taps (over the IC contraction) and raises MP.
Per OC pass: (K_TOTAL/K_PAR)*MP + ~6 cycles (vs K_TOTAL*MP + 6 serial).

BYTE-EXACTNESS
--------------
Modeled EXACTLY on the existing 1x1 FSM (node_conv2d_3.v / _6.v):
  * same ST_STREAM stride-2 gating (in_row[0]==0 && in_col[0]==0), same
    out_count inter-frame reset, same ready_in backpressure,
  * same in_latch of the IC input bytes,
  * same PER-TENSOR SCALE_MULT / SCALE_SHIFT and the SAME sign-dependent round
    bias (scaled<0 ? HALF-1 : HALF) >>> SCALE_SHIFT, same saturation,
  * same flat weight values, just read K_PAR-at-a-time and tree-summed per lane
    in the SAME accumulation order.
Only the cycle count drops. Geometry + SCALE_MULT/SCALE_SHIFT are scraped from
the current .v so the constants are carried verbatim.

VERIFY: NN2RTL_VALUE_THREADS=1 NN2RTL_VALUE_XINIT=0 \
        npx tsx scripts/run_resnet8_top_value.ts 0   -> result=PASS mismatch_bytes=0
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts.repack_weights_wide import read_flat_weights, write_wide_weights  # noqa: E402

RTL_DIR = REPO / "output" / "resnet8" / "rtl"
WEIGHTS_DIR = REPO / "output" / "resnet8" / "weights"
BACKUP_TAG = ".prekpar1x1"

# Per-conv (MP, K_PAR). MP divides OC; K_PAR divides K_TOTAL (= IC for 1x1).
CONFIG: dict[str, tuple[int, int]] = {
    "node_conv2d_3": (16, 8),    # OC32 IC16 -> 128 mult, ~4K cyc (tiny pix=256)
    "node_conv2d_6": (16, 8),    # OC64 IC32 -> 128 mult, ~3K cyc (pix=64)
}
# Map the 1x1 MAC multipliers to LUT (not DSP). These projection convs are tiny
# (pix 256/64) and overlap-HIDDEN behind conv_1/conv_2, so they don't gate the
# e2e -- keeping their mults off the DSP array leaves DSP headroom for clean
# placement. Cycle-identical (LUT computes the same products).
USE_DSP_MAC = False


def scrape(mid: str) -> dict:
    txt = (RTL_DIR / f"{mid}.v").read_text()
    g = {}
    for key in ["IC", "OC", "IH", "IW", "OH", "OW", "SH", "SW", "KH", "KW",
                "SCALE_MULT", "SCALE_SHIFT"]:
        m = re.search(rf"localparam(?:\s+integer)?\s+{key}\s*=\s*(\d+);", txt)
        if not m:
            raise SystemExit(f"{mid}: localparam {key} not found")
        g[key] = int(m.group(1))
    return g


def backup(path: Path):
    bak = path.with_suffix(path.suffix + BACKUP_TAG)
    if not bak.exists():
        bak.write_bytes(path.read_bytes())
        print(f"  backup -> {bak.name}")


def build_wide_hex(mid: str, g: dict, mp: int, k_par: int) -> str:
    flat = WEIGHTS_DIR / f"{mid}_weights.hex"
    weights = read_flat_weights(flat)
    oc = g["OC"]
    k_total = g["IC"] * g["KH"] * g["KW"]
    out = WEIGHTS_DIR / f"{mid}_weights_wide1x1_mp{mp}_kp{k_par}.hex"
    entries, padded = write_wide_weights(out, weights, oc, k_total, mp, k_par, wgt_bits=8)
    print(f"  wide hex -> {out.name} ({entries} words, padded_zeros={padded})")
    return out.as_posix()


def emit(mid: str, g: dict, mp: int, k_par: int, wide_hex: str) -> str:
    IC, OC = g["IC"], g["OC"]
    IH, IW, OH, OW = g["IH"], g["IW"], g["OH"], g["OW"]
    SH, SW, KH, KW = g["SH"], g["SW"], g["KH"], g["KW"]
    SCALE_MULT, SCALE_SHIFT = g["SCALE_MULT"], g["SCALE_SHIFT"]
    K_TOTAL = IC * KH * KW
    if K_TOTAL % k_par:
        raise SystemExit(f"{mid}: K_TOTAL={K_TOTAL} % K_PAR={k_par} != 0")
    if OC % mp:
        raise SystemExit(f"{mid}: OC={OC} % MP={mp} != 0")
    OC_PASSES = OC // mp
    K_GROUPS = K_TOTAL // k_par
    OUT_W = OC * 8
    IN_W = IC * 8
    WIDE_W = mp * k_par * 8
    b_hex = (WEIGHTS_DIR / f"{mid}_bias.hex").as_posix()
    use_dsp_attr = '(* use_dsp = "yes" *) ' if USE_DSP_MAC else ""

    body = f"""// {mid} - pointwise 1x1 conv2d stride {SH}x{SW}, IC={IC}, OC={OC}, IH=IW={IH}, OH=OW={OH}.
// RE-PARALLELIZED 1x1: MP={mp} lanes x K_PAR={k_par} [{'DSP' if USE_DSP_MAC else 'LUT'}] taps = {mp*k_par} INT8 multiplies/cycle.
// ST_RUNNING = K_GROUPS({K_GROUPS}) * MP({mp}) cycles/pass; OC_PASSES={OC_PASSES} passes/pixel.
// Byte-exact vs the serial MP=4 1x1 FSM (same products, same accumulation order,
// same per-tensor SCALE_MULT={SCALE_MULT}/SCALE_SHIFT={SCALE_SHIFT}, same sign-dependent
// round bias + saturate, same stride-2 gating + inter-frame reset). Weights repacked
// WIDE (MP*K_PAR bytes/word) read one word/cycle.

module {mid} (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              valid_in,
    output reg               ready_in,
    input  wire [{IN_W-1}:0]      data_in,
    output reg               valid_out,
    output reg  [{OUT_W-1}:0]      data_out
);
    localparam IC        = {IC};
    localparam OC        = {OC};
    localparam IH        = {IH};
    localparam IW        = {IW};
    localparam OH        = {OH};
    localparam OW        = {OW};
    localparam OH_OW     = OH * OW;
    localparam SH        = {SH};
    localparam SW        = {SW};
    localparam KH        = {KH};
    localparam KW        = {KW};
    localparam K_TOTAL   = IC * KH * KW;       // {K_TOTAL}
    localparam MP        = {mp};
    localparam K_PAR     = {k_par};
    localparam K_GROUPS  = K_TOTAL / K_PAR;    // {K_GROUPS}
    localparam OC_PASSES = OC / MP;            // {OC_PASSES}
    localparam NUM_WIDE  = OC_PASSES * K_GROUPS; // {OC_PASSES*K_GROUPS}
    localparam WIDE_W    = MP * K_PAR * 8;     // {WIDE_W}

    localparam SCALE_MULT  = {SCALE_MULT};
    localparam SCALE_SHIFT = {SCALE_SHIFT};

    localparam integer PROD_W        = 16;
    localparam integer TREE_W        = PROD_W + $clog2(K_PAR);
    localparam integer ACC_W         = TREE_W + $clog2(K_GROUPS > 1 ? K_GROUPS : 2);
    localparam integer BIAS_W        = 32;
    localparam integer BIASED_W      = ((ACC_W > BIAS_W) ? ACC_W : BIAS_W) + 1;
    localparam integer SCALE_MAG_W   = $clog2(SCALE_MULT + 1);
    localparam integer SCALE_CONST_W = SCALE_MAG_W + 1;
    localparam integer SCALED_W      = BIASED_W + SCALE_CONST_W;
    localparam signed [SCALE_CONST_W-1:0] SCALE_MULT_CONST = SCALE_MULT;
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF =
        {{{{(SCALED_W-1){{1'b0}}}}, 1'b1}} <<< (SCALE_SHIFT - 1);

    localparam KGROUP_W   = (K_GROUPS <= 1) ? 1 : $clog2(K_GROUPS);
    localparam OC_GROUP_W = (OC_PASSES <= 1) ? 1 : $clog2(OC_PASSES);

    localparam ST_STREAM  = 3'd0;
    localparam ST_RUNNING = 3'd1;
    localparam ST_BIAS    = 3'd2;
    localparam ST_SCALE   = 3'd3;
    localparam ST_OUTPUT  = 3'd4;

    (* rom_style = "block", ram_style = "block" *) reg [WIDE_W-1:0]  weights_wide [0:NUM_WIDE-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [31:0] biases       [0:OC-1];
    initial begin
        $readmemh("{wide_hex}", weights_wide);
        $readmemh("{b_hex}", biases);
    end

    reg signed [7:0] in_latch [0:IC-1];

    reg signed [ACC_W-1:0]    acc    [0:MP-1];
    reg signed [BIASED_W-1:0] biased [0:MP-1];
    reg signed [SCALED_W-1:0] scaled [0:MP-1];
    reg signed [SCALED_W-1:0] v_tmp;

    reg [KGROUP_W-1:0]   k_group;
    reg [OC_GROUP_W-1:0] oc_group;
    reg [2:0] state;

    reg [$clog2(IH)-1:0] in_row;
    reg [$clog2(IW)-1:0] in_col;
    reg [$clog2(OH_OW+1)-1:0] out_count;

    wire [$clog2(NUM_WIDE+1)-1:0] weight_read_addr = oc_group * K_GROUPS + k_group;

    // Stage 1: register wide weight word + K_PAR taps (taps from in_latch).
    reg [WIDE_W-1:0] weight_word_q;
    reg signed [7:0] tap_q [0:K_PAR-1];
    integer ld_i;
    always @(posedge clk) begin
        weight_word_q <= weights_wide[weight_read_addr];
        for (ld_i = 0; ld_i < K_PAR; ld_i = ld_i + 1)
            tap_q[ld_i] <= in_latch[k_group * K_PAR + ld_i];
    end

    // Stage 2: MP*K_PAR multipliers, tree-sum per lane (combinational).
    {use_dsp_attr}reg signed [TREE_W-1:0] partial_q [0:MP-1];
    reg signed [TREE_W-1:0] sum_lane_w [0:MP-1];
    reg signed [PROD_W-1:0] prod_w;
    integer cs_lane, cs_kpos;
    always @* begin
        for (cs_lane = 0; cs_lane < MP; cs_lane = cs_lane + 1) begin
            sum_lane_w[cs_lane] = {{TREE_W{{1'b0}}}};
            for (cs_kpos = 0; cs_kpos < K_PAR; cs_kpos = cs_kpos + 1) begin
                prod_w = $signed(weight_word_q[(cs_lane * K_PAR + cs_kpos) * 8 +: 8]) *
                         $signed(tap_q[cs_kpos]);
                sum_lane_w[cs_lane] = sum_lane_w[cs_lane] + prod_w;
            end
        end
    end

    reg                  mac_valid_q1;
    reg                  mac_valid_q2;
    reg                  mac_done_issuing;
    integer i, lane, p_i;
    integer bias_oc, out_oc;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= ST_STREAM;
            ready_in         <= 1'b1;
            valid_out        <= 1'b0;
            k_group          <= 0;
            oc_group         <= 0;
            in_row           <= 0;
            in_col           <= 0;
            out_count        <= 0;
            data_out         <= {{(OC*8){{1'b0}}}};
            mac_valid_q1     <= 1'b0;
            mac_valid_q2     <= 1'b0;
            mac_done_issuing <= 1'b0;
            for (i = 0; i < IC; i = i + 1)
                in_latch[i] <= 8'sd0;
            for (lane = 0; lane < MP; lane = lane + 1) begin
                acc   [lane] <= 0;
                biased[lane] <= 0;
                scaled[lane] <= 0;
                partial_q[lane] <= 0;
            end
        end else begin
            // Stage 2 register + Stage 3 accumulate.
            for (p_i = 0; p_i < MP; p_i = p_i + 1)
                partial_q[p_i] <= sum_lane_w[p_i];
            mac_valid_q2 <= mac_valid_q1;
            if (mac_valid_q2) begin
                for (p_i = 0; p_i < MP; p_i = p_i + 1)
                    acc[p_i] <= acc[p_i] + $signed(partial_q[p_i]);
            end

            case (state)

            ST_STREAM: begin
                valid_out    <= 1'b0;
                mac_valid_q1 <= 1'b0;
                mac_valid_q2 <= 1'b0;
                if (valid_in) begin
                    if (in_col == IW - 1) begin
                        in_col <= 0;
                        if (in_row == IH - 1) in_row <= 0;
                        else                  in_row <= in_row + 1;
                    end else begin
                        in_col <= in_col + 1;
                    end

                    if ((in_row[0] == 1'b0) && (in_col[0] == 1'b0)) begin
                        for (i = 0; i < IC; i = i + 1)
                            in_latch[i] <= $signed(data_in[i*8 +: 8]);
                        ready_in         <= 1'b0;
                        k_group          <= 0;
                        oc_group         <= 0;
                        mac_done_issuing <= 1'b0;
                        for (lane = 0; lane < MP; lane = lane + 1)
                            acc[lane] <= 0;
                        state <= ST_RUNNING;
                    end
                end
            end

            ST_RUNNING: begin
                if (mac_done_issuing) begin
                    mac_valid_q1 <= 1'b0;
                    if (!mac_valid_q1 && !mac_valid_q2) begin
                        mac_done_issuing <= 1'b0;
                        state            <= ST_BIAS;
                    end
                end else begin
                    mac_valid_q1 <= 1'b1;
                    if (k_group == K_GROUPS - 1) begin
                        mac_done_issuing <= 1'b1;
                    end else begin
                        k_group <= k_group + 1;
                    end
                end
            end

            ST_BIAS: begin
                for (lane = 0; lane < MP; lane = lane + 1) begin
                    bias_oc = oc_group * MP + lane;
                    if (bias_oc < OC)
                        biased[lane] <= $signed(acc[lane]) + $signed(biases[bias_oc]);
                    else
                        biased[lane] <= 0;
                end
                state <= ST_SCALE;
            end

            ST_SCALE: begin
                for (lane = 0; lane < MP; lane = lane + 1)
                    scaled[lane] <= $signed(biased[lane]) * $signed(SCALE_MULT_CONST);
                state <= ST_OUTPUT;
            end

            ST_OUTPUT: begin
                for (lane = 0; lane < MP; lane = lane + 1) begin
                    out_oc = oc_group * MP + lane;
                    if (out_oc < OC) begin
                        v_tmp = (scaled[lane] +
                                 (scaled[lane][SCALED_W-1] ? (SCALE_ROUND_HALF - 1)
                                                           : SCALE_ROUND_HALF)
                                ) >>> SCALE_SHIFT;
                        data_out[out_oc*8 +: 8] <= (v_tmp >  127) ?  8'sd127 :
                                                   (v_tmp < -128) ? -8'sd128 :
                                                                    v_tmp[7:0];
                    end
                end

                if (oc_group < OC_PASSES - 1) begin
                    for (lane = 0; lane < MP; lane = lane + 1) acc[lane] <= 0;
                    k_group          <= 0;
                    oc_group         <= oc_group + 1;
                    mac_valid_q1     <= 1'b0;
                    mac_valid_q2     <= 1'b0;
                    mac_done_issuing <= 1'b0;
                    state            <= ST_RUNNING;
                end else begin
                    valid_out <= 1'b1;
                    ready_in  <= 1'b1;
                    oc_group  <= 0;
                    state     <= ST_STREAM;
                    if (out_count == OH_OW - 1) begin
                        out_count <= 0;
                        in_row    <= 0;
                        in_col    <= 0;
                    end else begin
                        out_count <= out_count + 1;
                    end
                end
            end

            default: state <= ST_STREAM;
            endcase
        end
    end
endmodule
"""
    return body


def apply_one(mid: str) -> bool:
    mp, k_par = CONFIG[mid]
    path = RTL_DIR / f"{mid}.v"
    txt = path.read_text()
    marker = f"RE-PARALLELIZED 1x1: MP={mp} lanes x K_PAR={k_par} [{'DSP' if USE_DSP_MAC else 'LUT'}]"
    if marker in txt:
        print(f"{mid}: already MP={mp} K_PAR={k_par}; skip")
        return False
    g = scrape(mid)
    backup(path)
    wide_hex = build_wide_hex(mid, g, mp, k_par)
    new = emit(mid, g, mp, k_par, wide_hex)
    path.write_text(new)
    K_TOTAL = g["IC"] * g["KH"] * g["KW"]
    print(f"{mid}: 1x1 re-parallelized -> MP={mp} K_PAR={k_par} "
          f"(IC={g['IC']} OC={g['OC']} K_TOTAL={K_TOTAL} "
          f"OC_PASSES={g['OC'] // mp} K_GROUPS={K_TOTAL // k_par} mult={mp * k_par})")
    return True


def main():
    print("Re-parallelizing ResNet-8 1x1 projection convs with MP + K_PAR:")
    changed = 0
    for mid in CONFIG:
        if apply_one(mid):
            changed += 1
    print(f"Done. ({changed} convs re-parallelized)")


if __name__ == "__main__":
    main()
