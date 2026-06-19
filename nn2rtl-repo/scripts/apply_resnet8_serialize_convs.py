#!/usr/bin/env python3
"""Serialize the oversized ResNet-8 convolutions to a low-MP (MP=4) FSM datapath
so the design FITS the ZCU104 (xczu7ev, 230,400 LUTs) and routes.

WHY
---
Four ResNet-8 convs were emitted by Foundry as a FULLY-PARALLEL-OC datapath: a
single combinational MAC stage that computes ALL OC*K_TOTAL INT8 multiplies per
pixel in one cycle. Because the weights are compile-time constants, every one of
those tens of thousands of multiplies maps to LUTs (not DSPs). The result is a
1,323,990-LUT netlist = 574% of the ZCU104. The post-synth hierarchy:

    node_conv2d_8 : 850,360 LUTs  (3x3 s1 p1, IC=OC=64,  576 taps -> 36864 // OC*tap MACs)
    node_conv2d_5 : 219,702 LUTs  (3x3 s1 p1, IC=OC=32,  288 taps)
    node_conv2d_2 :  88,529 LUTs  (3x3 s1 p1, IC=OC=16,  144 taps; 7-stage parallel pipe)
    node_conv2d_1 :  58,247 LUTs  (3x3 s1 p1, IC=OC=16,  144 taps)

Their SIBLINGS node_conv2d_4 / _7 instead use a SERIAL MP=4 FSM
(MAC -> BIAS -> SCALE -> OUTPUT, one weight*tap multiply per cycle, MP lanes per
OC-group) and cost only ~4K / ~10K LUTs each, using ~9 DSPs. The datapaths
compute the IDENTICAL result -- same per-OC requant (compute_scale_approx),
same MAC order, same saturation/rounding -- so swapping the parallel pipe for
the serial FSM is BYTE-EXACT on the logits. Only the cycle count rises (the conv
runs OC_PASSES * (MP*K_TOTAL+overhead) cycles per pixel instead of 1). The design
is handshake-elastic (valid/ready), so a slower conv just backpressures upstream;
the residual skip FIFOs already hold a full frame, so no operand-skew deadlock.

WHAT THIS DOES (idempotent; writes .preserialize backups)
---------------------------------------------------------
For each target conv it REWRITES output/resnet8/rtl/<mid>.v to the serial MP=4
FSM, modeled EXACTLY on the proven node_conv2d_7 inline FSM (see
apply_resnet8_perOC_convs.patch_conv7), parameterized by:
  * geometry (IC/OC/IH/IW/KH/KW/SH/SW/PH/PW) read from the current .v,
  * data_in / data_out bus widths = IC*8 / OC*8,
  * per-OC requant ROMs scale_mult_rom[oc]/scale_shift_rom[oc] =
    compute_scale_approx(scale_factor_per_oc[oc]) -- the exact golden constants,
  * the same flat weight/bias .hex ($readmemh weights[oc*K_TOTAL + k]).

MP=4 is the proven sibling parallelism. It bounds each conv to roughly
OC/4 * (4*K_TOTAL + 6) cycles/pixel (so e.g. conv_8: 16*(4*576+6)=36960 cyc/pix
* 64 pix), which the e2e value gate tolerates and which keeps each conv well
under ~10K LUTs.

The 1x1 projection convs node_conv2d_3 / _6 (already serial-ish, tiny) and the
already-serial node_conv2d_4 / _7 are NOT touched, nor is the stem node_conv2d
(only ~13K LUTs, IC=3). Run apply_resnet8_perOC_convs.py FIRST (it establishes
the per-OC requant correctness on every conv); this script then replaces the
PARALLEL ones with the serial FSM, carrying the same per-OC scale constants.

VERIFY: NN2RTL_VALUE_THREADS=1 NN2RTL_VALUE_XINIT=0 \
        npx tsx scripts/run_resnet8_top_value.ts 0   -> result=PASS mismatch_bytes=0
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts.golden_impl import compute_scale_approx  # noqa: E402

RTL_DIR = REPO / "output" / "resnet8" / "rtl"
IR_PATH = REPO / "output" / "resnet8" / "layer_ir.json"
WEIGHTS_DIR = REPO / "output" / "resnet8" / "weights"
BACKUP_TAG = ".preserialize"

# The PARALLEL-OC convs to serialize (all 3x3 stride-1 pad-1, IC==OC). The
# siblings conv_4/_7 are already serial; conv_3/_6 are tiny 1x1 projections; the
# stem (conv) is small (IC=3). MP=4 = the proven sibling parallelism.
TARGETS = ["node_conv2d_1", "node_conv2d_2", "node_conv2d_5", "node_conv2d_8"]
MP = 4


def read_geom(mid: str) -> dict:
    """Pull IC/OC/IH/IW/KH/KW/SH/SW/PH/PW from the current .v localparams.

    Some convs (e.g. node_conv2d_2, a custom pipeline) hardcode geometry and do
    NOT declare SH/SW/PH/PW localparams. For those we DERIVE them: stride from
    the IH/OH ratio, pad from the conv output formula OH = (IH+2P-KH)/S + 1.
    """
    txt = (RTL_DIR / f"{mid}.v").read_text()
    g = {}
    for key in ["IC", "OC", "IH", "IW", "OH", "OW", "KH", "KW"]:
        m = re.search(rf"localparam integer {key}\s*=\s*(\d+);", txt)
        if not m:
            raise SystemExit(f"{mid}: localparam {key} not found")
        g[key] = int(m.group(1))
    for key, num, den, kk in [("SH", "IH", "OH", "KH"), ("SW", "IW", "OW", "KW")]:
        m = re.search(rf"localparam integer {key}\s*=\s*(\d+);", txt)
        if m:
            g[key] = int(m.group(1))
        else:
            # derive stride: OH = (IH + 2P - KH)/S + 1; with the resnet8 same-pad
            # convs IH==OH => S=1. Use the integer S that makes OH consistent.
            s = max(1, round(g[num] / g[den]))
            g[key] = s
    for key, ih, oh, kk, sh in [("PH", "IH", "OH", "KH", "SH"), ("PW", "IW", "OW", "KW", "SW")]:
        m = re.search(rf"localparam integer {key}\s*=\s*(\d+);", txt)
        if m:
            g[key] = int(m.group(1))
        else:
            # OH = (IH + 2P - KH)/S + 1  =>  P = ((OH-1)*S - IH + KH) / 2
            p2 = (g[oh] - 1) * g[sh] - g[ih] + g[kk]
            if p2 % 2 != 0 or p2 < 0:
                raise SystemExit(f"{mid}: cannot derive {key} (p2={p2})")
            g[key] = p2 // 2
    return g


def per_oc_pairs(mid: str):
    ir = json.loads(IR_PATH.read_text())
    layer = next(l for l in ir["layers"]
                 if l.get("module_id") == mid and l.get("op_type") == "conv2d")
    sf = layer["scale_factor_per_oc"]
    pairs = [compute_scale_approx(float(s)) for s in sf]
    oc = layer["output_shape"][1]
    if len(pairs) != oc:
        raise SystemExit(f"{mid}: {len(pairs)} scales vs OC={oc}")
    return pairs, oc


def backup(path: Path):
    bak = path.with_suffix(path.suffix + BACKUP_TAG)
    if not bak.exists():
        bak.write_bytes(path.read_bytes())
        print(f"  backup -> {bak.name}")


def emit_serial_fsm(mid: str, g: dict, pairs) -> str:
    IC, OC = g["IC"], g["OC"]
    IH, IW, OH, OW = g["IH"], g["IW"], g["OH"], g["OW"]
    KH, KW, SH, SW, PH, PW = g["KH"], g["KW"], g["SH"], g["SW"], g["PH"], g["PW"]
    K_TOTAL = IC * KH * KW
    NUM_WEIGHTS = OC * K_TOTAL
    OC_PASSES = OC // MP
    if OC % MP != 0:
        raise SystemExit(f"{mid}: OC={OC} not divisible by MP={MP}")
    IN_W = IC * 8     # data_in bus width
    OUT_W = OC * 8    # data_out bus width

    w_hex = (WEIGHTS_DIR / f"{mid}_weights.hex").as_posix()
    b_hex = (WEIGHTS_DIR / f"{mid}_bias.hex").as_posix()
    if not (WEIGHTS_DIR / f"{mid}_weights.hex").exists():
        raise SystemExit(f"{mid}: weights hex not found: {w_hex}")
    if not (WEIGHTS_DIR / f"{mid}_bias.hex").exists():
        raise SystemExit(f"{mid}: bias hex not found: {b_hex}")

    mult_lines = "\n".join(
        f"        scale_mult_rom[{i}]  = 16'sd{m};" for i, (m, sh) in enumerate(pairs))
    shf_lines = "\n".join(
        f"        scale_shift_rom[{i}] = 6'd{sh};" for i, (m, sh) in enumerate(pairs))

    # This is an f-string, so EVERY literal Verilog brace must be doubled. To
    # avoid an unreadable thicket of {{{{ for the Verilog replication concats,
    # the few brace-heavy literals are emitted as sentinels and substituted
    # after the f-string renders (see the .replace() calls at the bottom).
    #   __OUT_W__       -> the integer OUT_W (used in a width prefix like 512'd0)
    #   __ROUND_BIAS__  -> {{(SCALED_W-1){1'b0}}, 1'b1}  (signed +0.5 round bias)
    body = f"""// {mid} -- {KH}x{KW} stride-{SH} pad-{PH} conv (IC={IC}, OC={OC}, IH=IW={IH}, OH=OW={OH}).
// SERIALIZED to a low-MP (MP={MP}) FSM datapath so the design fits the ZCU104.
//
// The Foundry-emitted version was a fully-parallel-OC combinational MAC (all
// {OC}*{K_TOTAL} = {OC*K_TOTAL} INT8 multiplies per pixel in one stage). With compile-time
// constant weights that maps to ~hundreds of thousands of LUTs. This rewrite
// uses the proven sibling serial FSM (modeled on node_conv2d_4 / _7): one
// weight*tap multiply per cycle through MP={MP} lanes, OC_PASSES={OC_PASSES} passes per
// output pixel. The compute is IDENTICAL (same MAC order, same per-OC requant
// compute_scale_approx, same round/saturate) -> byte-exact logits. Only the
// cycle count rises; the handshake (valid/ready) backpressures upstream.
//
// Per-OC requant: scale_mult_rom[oc] / scale_shift_rom[oc] =
//   compute_scale_approx(scale_factor_per_oc[oc]) -- the exact golden constants.

module {mid} (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       valid_in,
    output wire                       ready_in,
    input  wire [{IN_W-1}:0]               data_in,
    output wire                       valid_out,
    output wire [{OUT_W-1}:0]               data_out
);
    localparam integer IC          = {IC};
    localparam integer OC          = {OC};
    localparam integer IH          = {IH};
    localparam integer IW          = {IW};
    localparam integer OH          = {OH};
    localparam integer OW          = {OW};
    localparam integer KH          = {KH};
    localparam integer KW          = {KW};
    localparam integer SH          = {SH};
    localparam integer SW          = {SW};
    localparam integer PH          = {PH};
    localparam integer PW          = {PW};
    localparam integer K_TOTAL     = IC * KH * KW; // {K_TOTAL}
    localparam integer MP          = {MP};
    localparam integer OC_PASSES   = OC / MP;       // {OC_PASSES}
    localparam integer NUM_WEIGHTS = OC * K_TOTAL;  // {NUM_WEIGHTS}

    localparam integer SCALE_SHIFT_MAX = 23;
    localparam integer PROD_W       = 16;
    localparam integer ACC_W        = PROD_W + $clog2(K_TOTAL);
    localparam integer BIAS_W       = 32;
    localparam integer BIASED_W     = ((ACC_W > BIAS_W) ? ACC_W : BIAS_W) + 1;
    localparam integer SCALE_MULT_W = 16;
    localparam integer SCALED_W     = BIASED_W + SCALE_MULT_W;

    localparam integer WEIGHT_ADDR_W   = $clog2(NUM_WEIGHTS);
    localparam integer K_COUNTER_W     = $clog2(K_TOTAL);
    localparam integer LANE_COUNTER_W  = $clog2(MP);
    localparam integer OC_GROUP_W      = $clog2(OC_PASSES);
    localparam integer OC_INDEX_W      = $clog2(OC + MP);

    // ---- Per-OC requant ROMs: compute_scale_approx(scale_factor_per_oc[oc]) ----
    reg signed [SCALE_MULT_W-1:0] scale_mult_rom  [0:OC-1];
    reg        [5:0]              scale_shift_rom [0:OC-1];
    initial begin
{mult_lines}
{shf_lines}
    end

    reg started, start_pulse, pending_rearm;
    wire sched_out_frame_done;
    wire                              sched_needs_real_input;
    wire                              sched_ready_in;
    wire                              sched_output_fires;
    wire                              sched_advance;
    wire [$clog2(IH + PH + 1)-1:0]    sched_in_row;
    wire [$clog2(IW + PW + 1)-1:0]    sched_in_col;
    wire [$clog2(OH * OW + 1)-1:0]    sched_outputs_emitted;
    wire [KH*KW*IC*8-1:0]             window_flat;
    wire                              mac_busy_w;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            started       <= 1'b0;
            start_pulse   <= 1'b0;
            pending_rearm <= 1'b0;
        end else begin
            start_pulse <= 1'b0;
            if (sched_out_frame_done) begin
                pending_rearm <= 1'b1;
            end
            if (!started) begin
                started     <= 1'b1;
                start_pulse <= 1'b1;
            end else if (pending_rearm && !mac_busy_w) begin
                started       <= 1'b0;
                pending_rearm <= 1'b0;
            end
        end
    end

    wire stall_in = mac_busy_w;

    coord_scheduler #(
        .IH(IH), .IW(IW), .OH(OH), .OW(OW),
        .KH(KH), .KW(KW), .SH(SH), .SW(SW),
        .PH(PH), .PW(PW)
    ) scheduler (
        .clk(clk), .rst_n(rst_n),
        .start(start_pulse),
        .stall_in(stall_in),
        .valid_in(valid_in),
        .ready_in(sched_ready_in),
        .needs_real_input(sched_needs_real_input),
        .in_row(sched_in_row),
        .in_col(sched_in_col),
        .output_fires(sched_output_fires),
        .advance(sched_advance),
        .in_frame_done(),
        .out_frame_done(sched_out_frame_done),
        .outputs_emitted(sched_outputs_emitted)
    );

    line_buf_window #(
        .IC(IC), .IW(IW), .IH(IH),
        .KH(KH), .KW(KW), .PW(PW), .PH(PH)
    ) lbw (
        .clk(clk), .rst_n(rst_n),
        .frame_start(start_pulse),
        .sched_in_row(sched_in_row),
        .sched_in_col(sched_in_col),
        .sched_needs_real_input(sched_needs_real_input),
        .sched_advance(sched_advance),
        .sched_output_fires(sched_output_fires),
        .valid_in(valid_in),
        .data_in(data_in),
        .window_flat(window_flat)
    );

    localparam ST_IDLE   = 3'd0;
    localparam ST_MAC    = 3'd1;
    localparam ST_BIAS   = 3'd2;
    localparam ST_SCALE  = 3'd3;
    localparam ST_OUTPUT = 3'd4;

    reg [2:0]   state;
    reg         valid_out_r;
    reg [{OUT_W-1}:0] data_out_r;

    (* rom_style = "block", ram_style = "block" *) reg signed [7:0]  weights_mem [0:NUM_WEIGHTS-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [31:0] biases_mem  [0:OC-1];
    initial begin
        $readmemh("{w_hex}", weights_mem);
        $readmemh("{b_hex}",    biases_mem);
    end

    reg signed [ACC_W-1:0]    acc    [0:MP-1];
    reg signed [BIASED_W-1:0] biased [0:MP-1];
    reg signed [SCALED_W-1:0] scaled [0:MP-1];
    reg [5:0]                 shift_lane [0:MP-1];
    reg signed [SCALED_W-1:0] v_tmp;

    reg [K_COUNTER_W-1:0]    k_counter;
    reg [LANE_COUNTER_W-1:0] lane_counter;
    reg [OC_GROUP_W-1:0]     oc_group;

    integer i, lane_i;
    integer bias_oc;
    integer out_oc;

    assign mac_busy_w = (state != ST_IDLE);
    assign valid_out  = valid_out_r;     // [INVARIANT:VALID_OUT_LATENCY]
    assign data_out   = data_out_r;
    assign ready_in   = sched_ready_in;  // [INVARIANT:READY_IN_GATING]

    wire [OC_INDEX_W-1:0]    current_global_oc = oc_group * MP + lane_counter;
    wire [WEIGHT_ADDR_W-1:0] weight_read_addr  = current_global_oc * K_TOTAL + k_counter;

    function [7:0] tap_at;
        input [K_COUNTER_W-1:0] k;
        integer kh_idx, kw_idx, ic_idx, flat_idx;
        begin
            kh_idx   = (k % (KH * KW)) / KW;
            kw_idx   = k % KW;
            ic_idx   = k / (KH * KW);
            flat_idx = kh_idx * KW * IC + kw_idx * IC + ic_idx;
            tap_at   = window_flat[flat_idx*8 +: 8];
        end
    endfunction

    reg signed [7:0] weight_q;
    reg signed [7:0] tap_q;
    always @(posedge clk) begin
        weight_q <= weights_mem[weight_read_addr];
        tap_q    <= $signed(tap_at(k_counter));
    end

    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] mul_q;

    reg                       mac_valid_q1;
    reg [LANE_COUNTER_W-1:0]  mac_lane_q1;
    reg [OC_INDEX_W-1:0]      mac_global_oc_q1;
    reg                       mac_done_issuing;

    reg                       mac_valid_q2;
    reg [LANE_COUNTER_W-1:0]  mac_lane_q2;
    reg [OC_INDEX_W-1:0]      mac_global_oc_q2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= ST_IDLE;
            valid_out_r      <= 1'b0;
            data_out_r       <= {OUT_W}'d0;
            k_counter        <= 0;
            lane_counter     <= 0;
            oc_group         <= 0;
            mac_valid_q1     <= 1'b0;
            mac_lane_q1      <= 0;
            mac_global_oc_q1 <= 0;
            mac_valid_q2     <= 1'b0;
            mac_lane_q2      <= 0;
            mac_global_oc_q2 <= 0;
            mac_done_issuing <= 1'b0;
            mul_q            <= 0;
            for (i = 0; i < MP; i = i + 1) begin
                acc[i]        <= 0;
                biased[i]     <= 0;
                scaled[i]     <= 0;
                shift_lane[i] <= 0;
            end
        end else begin
            valid_out_r <= 1'b0;

            mul_q            <= $signed(weight_q) * $signed(tap_q);
            mac_valid_q2     <= mac_valid_q1;
            mac_lane_q2      <= mac_lane_q1;
            mac_global_oc_q2 <= mac_global_oc_q1;

            if (mac_valid_q2 && mac_global_oc_q2 < OC) begin
                acc[mac_lane_q2] <= acc[mac_lane_q2] + $signed(mul_q);
            end

            case (state)
                ST_IDLE: begin
                    if (sched_output_fires) begin
                        state            <= ST_MAC;
                        k_counter        <= 0;
                        lane_counter     <= 0;
                        oc_group         <= 0;
                        mac_valid_q1     <= 1'b0;
                        mac_valid_q2     <= 1'b0;
                        mac_done_issuing <= 1'b0;
                        for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1)
                            acc[lane_i] <= 0;
                    end
                end

                ST_MAC: begin
                    if (mac_done_issuing) begin
                        mac_valid_q1 <= 1'b0;
                        if (!mac_valid_q1 && !mac_valid_q2) begin
                            mac_done_issuing <= 1'b0;
                            state            <= ST_BIAS;
                        end
                    end else begin
                        mac_lane_q1      <= lane_counter;
                        mac_global_oc_q1 <= current_global_oc;
                        mac_valid_q1     <= 1'b1;

                        if (lane_counter == MP - 1) begin
                            lane_counter <= 0;
                            if (k_counter == K_TOTAL - 1) begin
                                mac_done_issuing <= 1'b1;
                            end else begin
                                k_counter <= k_counter + 1'b1;
                            end
                        end else begin
                            lane_counter <= lane_counter + 1'b1;
                        end
                    end
                end

                ST_BIAS: begin
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                        bias_oc = oc_group * MP + lane_i;
                        biased[lane_i] <= $signed(acc[lane_i]) + $signed(biases_mem[bias_oc]);
                    end
                    state <= ST_SCALE;
                end

                ST_SCALE: begin
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                        scaled[lane_i]     <= $signed(biased[lane_i]) *
                                              $signed(scale_mult_rom[oc_group * MP + lane_i]);
                        shift_lane[lane_i] <= scale_shift_rom[oc_group * MP + lane_i];
                    end
                    state <= ST_OUTPUT;
                end

                ST_OUTPUT: begin
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                        out_oc = oc_group * MP + lane_i;
                        // [INVARIANT:ROUNDING] single positive bias + arith >>> = golden floor.
                        v_tmp = (scaled[lane_i] +
                                 ($signed(__ROUND_BIAS__) <<< (shift_lane[lane_i] - 1))
                                ) >>> shift_lane[lane_i];
                        data_out_r[out_oc*8 +: 8] <=
                            (v_tmp >  127) ?  8'sd127 :
                            (v_tmp < -128) ? -8'sd128 : v_tmp[7:0];
                    end

                    if (oc_group == OC_PASSES - 1) begin
                        valid_out_r <= 1'b1;
                        state       <= ST_IDLE;
                    end else begin
                        oc_group     <= oc_group + 1'b1;
                        k_counter    <= 0;
                        lane_counter <= 0;
                        for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1)
                            acc[lane_i] <= 0;
                        state <= ST_MAC;
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
"""
    # Substitute the brace-heavy Verilog literal (kept out of the f-string to
    # avoid quadruple-brace escaping). This is the signed +0.5 round bias =
    # {{(SCALED_W-1){1'b0}}, 1'b1} : a concat of (SCALED_W-1) zero bits and a 1.
    round_bias = "{{(SCALED_W-1){1'b0}}, 1'b1}"
    return body.replace("__ROUND_BIAS__", round_bias)


def serialize(mid: str) -> bool:
    path = RTL_DIR / f"{mid}.v"
    txt = path.read_text()
    # Idempotency: the serial FSM contains the unique marker "ST_MAC" + the
    # "SERIALIZED to a low-MP" banner. The parallel pipe never has ST_MAC.
    if "SERIALIZED to a low-MP" in txt and "ST_MAC" in txt:
        print(f"{mid}: already serialized (MP={MP} FSM present); skip")
        return False
    g = read_geom(mid)
    pairs, oc = per_oc_pairs(mid)
    if oc != g["OC"]:
        raise SystemExit(f"{mid}: IR OC {oc} != .v OC {g['OC']}")
    backup(path)
    new = emit_serial_fsm(mid, g, pairs)
    path.write_text(new)
    K_TOTAL = g["IC"] * g["KH"] * g["KW"]
    print(f"{mid}: serialized parallel-OC pipe -> MP={MP} FSM "
          f"(IC={g['IC']} OC={g['OC']} K_TOTAL={K_TOTAL} OC_PASSES={oc // MP})")
    return True


def main():
    print("Serializing oversized ResNet-8 convs to MP=4 FSM:", ", ".join(TARGETS))
    changed = 0
    for mid in TARGETS:
        if serialize(mid):
            changed += 1
    print(f"Done. ({changed} convs rewritten)")


if __name__ == "__main__":
    main()
