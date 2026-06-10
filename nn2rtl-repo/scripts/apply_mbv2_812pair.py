#!/usr/bin/env python3
"""apply_mbv2_812pair.py -- MBV2 "812-PAIR": paired-channel MAC walk on node_conv_812.

DETERMINISTIC, anchor-asserted, idempotent patch script (same discipline as
scripts/apply_mpk9_depthwise.py / apply_k1_fdce_recode.py). Patches EXACTLY ONE
file -- output/mobilenet-v2/rtl/node_conv_812.v -- and touches NO shared
rtl_library file (the second per-channel window comes from line_buf_window's
EXPOSE_FULL_WINDOW(1) full-window FLATTEN, which is assign-only wiring inside
lbw; ResNet instantiations are untouched and provably unaffected).

THE LEVER (verified sweep 2026-06-10): node_conv_812 is the last spatial
depthwise conv post-DW-QUARTET and paces the entire FRONT zone:
  12544 px x ceil(C=32/MP=16)=2 passes x (16 lane-issues + 6) = 44 cyc/px
  -> 551,936 cycles with the engine 100% idle.
Paired walk: 2 channels (one even/odd lane pair) issue per ST_MAC cycle ->
  2 passes x (8 + 6) = 28 cyc/px -> front 351,232 (frame -200,704 expected).

BYTE-EXACT BY CONSTRUCTION: depthwise channel lanes are independent (disjoint
weights / window bytes / acc / per-OC requant slot per channel) and K_GROUPS=1
means each acc[] receives exactly ONE accumulate per pass -- no accumulation-
order change exists. The BIAS/SCALE/OUTPUT requant tail is ALREADY 16-lane
parallel per pass (per-OC scale_rom indexed by oc_group*MP+lane), so the
requant lane for the odd channel was always there; only the MAC issue walk
changes.

Usage:
  python scripts/apply_mbv2_812pair.py            # apply (no-op if already applied)
  python scripts/apply_mbv2_812pair.py --check    # verify live file == prepair + patch
  python scripts/apply_mbv2_812pair.py --revert   # restore the .prepair backup
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TARGET = REPO / "output" / "mobilenet-v2" / "rtl" / "node_conv_812.v"
BACKUP = TARGET.with_suffix(".v.prepair")

MARKER = "[812-PAIR"

# (old, new) anchored replacement pairs. Every `old` must occur EXACTLY ONCE
# in the pre-patch file; the patch asserts this before touching anything.
REPLACEMENTS: list[tuple[str, str]] = [
    # R1 -- module header comment
    (
        """// per-channel 9-tap dot product (no IC-axis reduction).
""",
        """// per-channel 9-tap dot product (no IC-axis reduction).
//
// [812-PAIR 2026-06-10] paired-channel MAC walk: the ST_MAC issue loop now
// processes TWO channels per cycle (an even/odd lane pair). Depthwise channel
// lanes are fully independent (disjoint weights / window bytes / acc / requant
// slot per channel) and each acc[] receives exactly ONE accumulate per pass
// (K_GROUPS=1 single-shot 9-tap dot product), so issuing two lanes per cycle
// is byte-exact by construction. Per-pixel MAC time: 2 passes x (16+6)=44 ->
// 2 passes x (8+6)=28 cycles. Lane B's 9-tap window comes from the lbw
// full-window FLATTEN (EXPOSE_FULL_WINDOW(1) -- pure assigns, zero logic
// change inside line_buf_window); lane A keeps the channel_select port path.
""",
    ),
    # R2 -- PAIR_STEPS localparam
    (
        """    localparam integer OC_PASSES = (C + MP - 1) / MP;
""",
        """    localparam integer OC_PASSES = (C + MP - 1) / MP;
    // [812-PAIR] 2 channels (one even/odd lane pair) issue per ST_MAC cycle.
    // PAIR_STEPS = MP/2 = 8 issue cycles per OC pass (was 16). Requires MP
    // even AND C a multiple of 2 (C=32, MP=16 here), so every pass covers
    // whole pairs and lane B's guard mirrors lane A's.
    localparam integer PAIR_STEPS = MP / 2;
""",
    ),
    # R3 -- window_flat_w wire declaration
    (
        """    wire [KH*KW*8-1:0]                chan_window_flat;
    wire                              mac_busy;
""",
        """    wire [KH*KW*8-1:0]                chan_window_flat;
    // [812-PAIR] full-window flatten from line_buf_window (EXPOSE_FULL_WINDOW(1)).
    // Inside lbw this is PURE WIRING (assigns of the same window / window_kwm1_wire
    // / bypass_reg sources the chan_window_flat mux reads) -- no extra logic or
    // behavior change. Lane B's 9 tap bytes are extracted from it below; for C=32
    // the flatten is only KH*KW*C*8 = 2304 wires.
    wire [KH*KW*C*8-1:0]              window_flat_w;
    wire                              mac_busy;
""",
    ),
    # R4 -- pair-step counter + lane A/B oc + weight bases
    (
        """    (* max_fanout = 256 *) reg [3:0] lane_counter;
    reg [2:0] oc_group;
    (* max_fanout = 256 *) wire [5:0]  current_global_oc = oc_group * MP + lane_counter;
    wire [15:0] weight_base_addr  = current_global_oc * K_TOTAL;  // contiguous K_TOTAL taps for this channel
""",
        """    // [812-PAIR] lane_counter is now the pair-STEP counter (0..PAIR_STEPS-1);
    // step s covers lanes {2s, 2s+1}. Lane A (even, 2s) keeps the legacy
    // current_global_oc name/role -- it still drives lbw.channel_select and
    // weight base A unchanged. Lane B (odd, 2s+1) gets its own oc/weight base.
    (* max_fanout = 256 *) reg [3:0] lane_counter;
    reg [2:0] oc_group;
    wire [3:0] pair_lane_a = {lane_counter[2:0], 1'b0};
    (* max_fanout = 256 *) wire [5:0]  current_global_oc   = oc_group * MP + pair_lane_a;
    wire [5:0]  current_global_oc_b = oc_group * MP + {lane_counter[2:0], 1'b1};
    wire [15:0] weight_base_addr   = current_global_oc   * K_TOTAL;  // contiguous K_TOTAL taps, lane A channel
    wire [15:0] weight_base_addr_b = current_global_oc_b * K_TOTAL;  // [812-PAIR] lane B channel
""",
    ),
    # R5a -- lbw instantiation: comment + EXPOSE_FULL_WINDOW(1)
    (
        """    // ----------------- line_buf_window (IC=C=32 packed) -----------------
    // Depthwise consumer: leave EXPOSE_FULL_WINDOW at default 0 (the wide
    // cross-channel window_flat mux is NOT instantiated -- routing-congestion
    // fix). Drive channel_select with current_global_oc (one channel per
    // cycle) and read the narrow chan_window_flat output instead.
    line_buf_window #(
        .IC(C), .IW(IW), .IH(IH),
        .KH(KH), .KW(KW), .PW(PW), .PH(PH),
        .EXPOSE_FULL_WINDOW(0),
""",
        """    // ----------------- line_buf_window (IC=C=32 packed) -----------------
    // Depthwise consumer. [812-PAIR] EXPOSE_FULL_WINDOW(1): the full-window
    // output is a pure FLATTEN (assign-only generate inside lbw -- no regs, no
    // behavioral change; for C=32 it is 2304 wires, not the C>=192 congestion
    // class the 0-setting was built for). Lane A still reads the narrow
    // chan_window_flat via channel_select (= current_global_oc, even lane);
    // lane B's 9 bytes are muxed from window_flat_w below using the documented
    // identity chan_window_flat[k] == window_flat[(k*IC + channel_select)*8 +: 8].
    line_buf_window #(
        .IC(C), .IW(IW), .IH(IH),
        .KH(KH), .KW(KW), .PW(PW), .PH(PH),
        .EXPOSE_FULL_WINDOW(1),
""",
    ),
    # R5b -- connect window_flat + lane-B extraction generate
    (
        """        .channel_select(current_global_oc),
        .chan_window_flat(chan_window_flat),
        .window_flat()
    );

    assign ready_in = sched_ready_in;
""",
        """        .channel_select(current_global_oc),
        .chan_window_flat(chan_window_flat),
        .window_flat(window_flat_w)
    );

    // [812-PAIR] lane-B per-channel window: extract the 9 tap bytes for the ODD
    // channel of the current pair from the full-window flatten, exactly the way
    // lbw's chan_window_flat mux does for channel_select (documented identity:
    //   chan_window_flat[(kh*KW+kw)*8 +: 8]
    //     == window_flat[((kh*KW+kw)*IC + channel_select)*8 +: 8]).
    // One C-way byte mux per tap (KH*KW muxes) -- the same logic a second
    // channel_select port would have instantiated, but with ZERO shared-file
    // (rtl_library/line_buf_window.v) changes.
    wire [KH*KW*8-1:0] chan_window_flat_b;
    genvar g_tap_b;
    generate
        for (g_tap_b = 0; g_tap_b < KH*KW; g_tap_b = g_tap_b + 1) begin : gen_tap_b
            assign chan_window_flat_b[g_tap_b*8 +: 8] =
                window_flat_w[(g_tap_b*C + current_global_oc_b)*8 +: 8];
        end
    endgenerate

    assign ready_in = sched_ready_in;
""",
    ),
    # R6 -- refresh stale datapath header math comment
    (
        """    // Identical FSM/pipeline to conv_datapath EXCEPT:
    //   - K_TOTAL = KH*KW (per-channel taps; no IC dim)
    //   - tap selector indexes window_flat at (kh, kw, current_channel)
    //   - one accumulator per LANE = one accumulator per output channel of
    //     the current OC pass; NO cross-channel reduction.
    // Per-pass cycle count = MP*K_TOTAL + 6 = 4*9 + 6 = 42 cycles.
    // OC_PASSES = 8. Total compute = 8*42 = 336. Spatial fill = 1*113 + 2
    // = 115. +1 for the registered output_fires => first valid_out at
    // exactly pipeline_latency_cycles = 452.
""",
        """    // Identical FSM/pipeline to conv_datapath EXCEPT:
    //   - K_TOTAL = KH*KW (per-channel taps; no IC dim)
    //   - tap selector indexes the per-channel window (lane A: chan_window_flat,
    //     lane B: window_flat_w extract) -- 9 taps read in parallel (MP_K=9)
    //   - one accumulator per LANE = one accumulator per output channel of
    //     the current OC pass; NO cross-channel reduction.
    // [812-PAIR] per-pass cycle count = PAIR_STEPS + 3 (q1/q2 drain) + 3
    // (BIAS/SCALE/OUTPUT) = 8 + 6 = 14 cycles (was 16 + 6 = 22).
    // OC_PASSES = 2. Total compute = 2*14 = 28 cycles per pixel (was 44).
""",
    ),
    # R7 -- lane-B tap/weight register stage
    (
        """    reg signed [7:0] weight_q [0:MP_K-1];
    reg signed [7:0] tap_q    [0:MP_K-1];
    integer kk;
    always @(posedge clk) begin
        for (kk = 0; kk < MP_K; kk = kk + 1) begin
            weight_q[kk] <= weights[weight_base_addr + kk];
            tap_q[kk]    <= $signed(chan_window_flat[kk*8 +: 8]);
        end
    end
""",
        """    reg signed [7:0] weight_q [0:MP_K-1];
    reg signed [7:0] tap_q    [0:MP_K-1];
    // [812-PAIR] lane-B copies of the tap/weight read stage (odd channel of the
    // pair). Same pipeline alignment, disjoint sources (weight_base_addr_b /
    // chan_window_flat_b), disjoint sinks (prod_qb -> sum_comb_b -> acc[odd]).
    reg signed [7:0] weight_qb [0:MP_K-1];
    reg signed [7:0] tap_qb    [0:MP_K-1];
    integer kk;
    always @(posedge clk) begin
        for (kk = 0; kk < MP_K; kk = kk + 1) begin
            weight_q[kk]  <= weights[weight_base_addr + kk];
            tap_q[kk]     <= $signed(chan_window_flat[kk*8 +: 8]);
            weight_qb[kk] <= weights[weight_base_addr_b + kk];
            tap_qb[kk]    <= $signed(chan_window_flat_b[kk*8 +: 8]);
        end
    end
""",
    ),
    # R8 -- lane-B product bank declaration
    (
        """    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] prod_q [0:MP_K-1];
""",
        """    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] prod_q [0:MP_K-1];
    // [812-PAIR] lane-B product bank (9 more DSP-class multipliers).
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] prod_qb [0:MP_K-1];
""",
    ),
    # R9 -- lane-B combinational tree-sum
    (
        """    reg signed [ACC_W-1:0] sum_comb;
    always @(*) begin
        sum_comb = {ACC_W{1'b0}};
        for (pp = 0; pp < MP_K; pp = pp + 1)
            sum_comb = sum_comb + $signed(prod_q[pp]);
    end
""",
        """    reg signed [ACC_W-1:0] sum_comb;
    always @(*) begin
        sum_comb = {ACC_W{1'b0}};
        for (pp = 0; pp < MP_K; pp = pp + 1)
            sum_comb = sum_comb + $signed(prod_q[pp]);
    end
    // [812-PAIR] lane-B tree-sum (identical form, disjoint operands).
    integer ppb;
    reg signed [ACC_W-1:0] sum_comb_b;
    always @(*) begin
        sum_comb_b = {ACC_W{1'b0}};
        for (ppb = 0; ppb < MP_K; ppb = ppb + 1)
            sum_comb_b = sum_comb_b + $signed(prod_qb[ppb]);
    end
""",
    ),
    # R10 -- K1 Block-A comment covers the lane-B twins
    (
        """    // [K1-MBV2] Block A: DATAPATH registers (sync-only, no reset) -- same
    // method as ResNet K1 P2 (apply_k1_fdce_recode.py). prod_q is rewritten
    // every cycle from the (no-reset) weight_q/tap_q stage and only reaches
    // acc under mac_valid_q2 (reset-kept); acc is sync-cleared on ST_IDLE&
""",
        """    // [K1-MBV2] Block A: DATAPATH registers (sync-only, no reset) -- same
    // method as ResNet K1 P2 (apply_k1_fdce_recode.py). prod_q (and its
    // [812-PAIR] lane-B twin prod_qb) is rewritten every cycle from the
    // (no-reset) weight_q/tap_q (weight_qb/tap_qb) stage and only reaches
    // acc under mac_valid_q2 (reset-kept); acc is sync-cleared on ST_IDLE&
""",
    ),
    # R11 -- Block A: lane-B products + lane-B accumulate
    (
        """    always @(posedge clk) begin
            for (i = 0; i < MP_K; i = i + 1)
                prod_q[i] <= $signed(weight_q[i]) * $signed(tap_q[i]);
            if (mac_valid_q2 && mac_global_oc_q2 < C[5:0]) begin
                acc[mac_lane_q2] <= acc[mac_lane_q2] + $signed(sum_comb);
            end
""",
        """    always @(posedge clk) begin
            for (i = 0; i < MP_K; i = i + 1) begin
                prod_q[i]  <= $signed(weight_q[i])  * $signed(tap_q[i]);
                // [812-PAIR] lane-B products: same stage, disjoint regs.
                prod_qb[i] <= $signed(weight_qb[i]) * $signed(tap_qb[i]);
            end
            if (mac_valid_q2 && mac_global_oc_q2 < C[5:0]) begin
                acc[mac_lane_q2] <= acc[mac_lane_q2] + $signed(sum_comb);
            end
            // [812-PAIR] lane-B accumulate: mac_lane_q2/mac_global_oc_q2 carry
            // the EVEN lane-A indices, so |1 is the paired odd lane (+1). The
            // two writes hit DISJOINT acc[] elements (even vs odd index) in the
            // same NBA block -- no ordering interaction. Guard mirrors lane A
            // (C=32 even => lane B is in-range exactly when lane A is, but keep
            // the explicit < C guard for form).
            if (mac_valid_q2 && (mac_global_oc_q2 | 6'd1) < C[5:0]) begin
                acc[mac_lane_q2 | 4'd1] <= acc[mac_lane_q2 | 4'd1] + $signed(sum_comb_b);
            end
""",
    ),
    # R12 -- ST_MAC issue walk: pair per cycle, PAIR_STEPS issues per pass
    (
        """                    end else begin
                        mac_lane_q1      <= lane_counter;
                        mac_global_oc_q1 <= current_global_oc;
                        mac_valid_q1     <= 1'b1;

                        if (lane_counter == (MP-1)) begin
""",
        """                    end else begin
                        // [812-PAIR] q1 carries the EVEN lane-A indices; the
                        // accumulate stage derives lane B as |1. One pair (2
                        // channels) issues per cycle -> PAIR_STEPS issues/pass.
                        mac_lane_q1      <= pair_lane_a;
                        mac_global_oc_q1 <= current_global_oc;
                        mac_valid_q1     <= 1'b1;

                        if (lane_counter == (PAIR_STEPS-1)) begin
""",
    ),
]


def apply_patch(text: str) -> str:
    for idx, (old, new) in enumerate(REPLACEMENTS, 1):
        n = text.count(old)
        if n != 1:
            raise SystemExit(
                f"[812pair] ANCHOR FAIL R{idx}: expected exactly 1 occurrence, found {n}. "
                f"File drifted -- refusing to patch."
            )
        text = text.replace(old, new, 1)
    return text


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "apply"
    live = TARGET.read_text(encoding="utf-8")

    if mode == "--revert":
        if not BACKUP.exists():
            raise SystemExit(f"[812pair] no backup at {BACKUP}")
        TARGET.write_text(BACKUP.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[812pair] reverted {TARGET} from {BACKUP.name}")
        return

    if mode == "--check":
        if not BACKUP.exists():
            raise SystemExit(f"[812pair] no backup at {BACKUP} -- cannot check")
        expect = apply_patch(BACKUP.read_text(encoding="utf-8"))
        if expect == live:
            print("[812pair] CHECK OK: live file == prepair + patch (byte-identical)")
        else:
            raise SystemExit("[812pair] CHECK FAIL: live file != prepair + patch")
        return

    # apply
    if MARKER in live:
        print("[812pair] already applied (marker found) -- no-op")
        return
    if not BACKUP.exists():
        BACKUP.write_text(live, encoding="utf-8")
        print(f"[812pair] backup written: {BACKUP.name}")
    TARGET.write_text(apply_patch(live), encoding="utf-8")
    print(f"[812pair] patched {TARGET} (12 anchored replacements)")


if __name__ == "__main__":
    main()
