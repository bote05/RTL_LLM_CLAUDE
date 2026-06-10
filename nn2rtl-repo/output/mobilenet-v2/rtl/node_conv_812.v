// node_conv_812 -- depthwise conv 3x3, stride 1, pad 1, C=32, IH=IW=112.
// Split-architecture (coord_scheduler + line_buf_window) + inline depthwise
// datapath that REPLACES conv_datapath's cross-channel adder tree with a
// per-channel 9-tap dot product (no IC-axis reduction).
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

`timescale 1ns/1ps
`default_nettype none

// PARAM-GATED ELASTIC BACKPRESSURE (ENABLE_BACKPRESSURE, default 0):
//   * ==0 (default): bit/cycle-IDENTICAL to the legacy module. out_ready_in is
//     IGNORED; skid_block is a constant 0 (scheduler/rearm never freeze); the
//     external valid_out/data_out come DIRECTLY from the datapath regs
//     (dp_valid_out/dp_data_out). The per-module verify TB (param=0) is
//     byte-exact with NO harness change.
//   * ==1: 1-deep output skid (out_full/out_data) captures the datapath's
//     1-cycle valid_out pulse; skid_block = out_full && !out_ready_in feeds
//     stall_in + blocks the frame rearm, freezing the scheduler/datapath while a
//     beat is parked and the downstream is not ready (per scratch/node_conv_812_bp.v).
module node_conv_812 #(
    parameter ENABLE_BACKPRESSURE = 0,
    parameter WEIGHTS_PATH = "output/mobilenet-v2/weights/node_conv_812_weights.hex",
    parameter BIAS_PATH    = "output/mobilenet-v2/weights/node_conv_812_bias.hex"
)(
    input  wire          clk,
    input  wire          rst_n,
    input  wire          valid_in,
    output wire          ready_in,
    input  wire [255:0]  data_in,
    input  wire          out_ready_in,   // NEW: downstream-ready (ignored when ENABLE_BACKPRESSURE==0)
    output wire          valid_out,
    output wire [255:0]  data_out
);

    // ----------------- Geometry -----------------
    localparam integer C         = 32;
    localparam integer IH        = 112;
    localparam integer IW        = 112;
    localparam integer OH        = 112;
    localparam integer OW        = 112;
    localparam integer KH        = 3;
    localparam integer KW        = 3;
    localparam integer SH        = 1;
    localparam integer SW        = 1;
    localparam integer PH        = 1;
    localparam integer PW        = 1;
    localparam integer K_TOTAL   = KH * KW;
    localparam integer MP        = 16;
    localparam integer MP_K      = 9;            // tap-parallel width (= K_TOTAL)
    localparam integer K_GROUPS  = K_TOTAL / MP_K; // = 1 (single-shot reduction)
    localparam integer OC_PASSES = (C + MP - 1) / MP;
    // [812-PAIR] 2 channels (one even/odd lane pair) issue per ST_MAC cycle.
    // PAIR_STEPS = MP/2 = 8 issue cycles per OC pass (was 16). Requires MP
    // even AND C a multiple of 2 (C=32, MP=16 here), so every pass covers
    // whole pairs and lane B's guard mirrors lane A's.
    localparam integer PAIR_STEPS = MP / 2;

    // ----------------- Quantization -----------------
    // compute_scale_approx(0.004891767275537665) picks MULT=10259, SHIFT=21.
    localparam integer SCALE_MULT  = 10259;
    localparam integer SCALE_SHIFT = 21;

    // ----------------- Weight / Bias ROMs -----------------
    // Canonical names required by structural preflight.
    (* rom_style = "block", ram_style = "block" *)
    reg signed [7:0]  weights [0:C*K_TOTAL-1];
    (* rom_style = "block", ram_style = "block" *)
    reg signed [31:0] biases  [0:C-1];
    // [PER-OC 2026-06-08][DW-CONSTSHIFT 2026-06-10] per-output-channel requant ROM. Slot is
    // the PRE-WIDENED multiplier mult' = mult << (DW_FIXED_SHIFT - shift), bits [30:0]
    // (< 2^31; the per-OC shift is folded OFFLINE -- scripts/apply_mbv2_dw_constshift.py /
    // build_spatial_scale_mems.py). RTL applies ONE compile-time >>> DW_FIXED_SHIFT with a
    // CONSTANT round, replacing the per-lane variable barrel shifter + round decode.
    (* rom_style = "block", ram_style = "block" *)
    reg [31:0]        scale_rom [0:C-1];

    initial begin
        $readmemh("output/mobilenet-v2/weights/node_conv_812_weights.hex", weights);
        $readmemh("output/mobilenet-v2/weights/node_conv_812_bias.hex", biases);
        $readmemh("output/mobilenet-v2/weights/node_conv_812_scale.mem", scale_rom);
    end

    // ----------------- Scheduler / window wires -----------------
    wire                              sched_needs_real_input;
    wire                              sched_ready_in;
    wire                              sched_output_fires;
    wire                              sched_advance;
    wire [$clog2(IH + PH + 1)-1:0]    sched_in_row;
    wire [$clog2(IW + PW + 1)-1:0]    sched_in_col;
    wire [$clog2(OH * OW + 1)-1:0]    sched_outputs_emitted;
    wire                              sched_out_frame_done;
    // Narrow per-channel window (one channel per cycle, selected by
    // current_global_oc). Replaces the wide cross-channel window_flat to
    // eliminate the KH*KW*IC-wide mux routing congestion. ZERO arithmetic
    // change: each byte is bit-identical to the old window_flat byte at
    // ((kh*KW+kw)*C + current_global_oc).
    wire [KH*KW*8-1:0]                chan_window_flat;
    // [812-PAIR] full-window flatten from line_buf_window (EXPOSE_FULL_WINDOW(1)).
    // Inside lbw this is PURE WIRING (assigns of the same window / window_kwm1_wire
    // / bypass_reg sources the chan_window_flat mux reads) -- no extra logic or
    // behavior change. Lane B's 9 tap bytes are extracted from it below; for C=32
    // the flatten is only KH*KW*C*8 = 2304 wires.
    wire [KH*KW*C*8-1:0]              window_flat_w;
    wire                              mac_busy;

    // current_global_oc drives line_buf_window.channel_select (below), so its
    // declaration -- and the counter regs it depends on -- are hoisted here
    // ABOVE its first use. Pure reordering moved from the datapath section;
    // zero logic change. Strict elaborators (Vivado / iverilog -g2012) reject
    // the forward reference that Verilator tolerated.
    // [812-PAIR] lane_counter is now the pair-STEP counter (0..PAIR_STEPS-1);
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

    // ---- datapath output regs (inlined) + 1-deep output skid ----
    // The datapath FSM (below) drives dp_valid_out/dp_data_out; the external
    // valid_out/data_out are routed from either the datapath directly (legacy)
    // or the skid (backpressure) via the generate block below.
    reg                  dp_valid_out;
    reg  [255:0]         dp_data_out;
    reg                  out_full;
    reg  [255:0]         out_data;
    // skid_block freezes the scheduler + rearm while a beat is parked and the
    // downstream cannot take it. Constant 0 when ENABLE_BACKPRESSURE==0 ->
    // legacy behaviour is exactly preserved.
    wire skid_block = (ENABLE_BACKPRESSURE != 0) && out_full && !out_ready_in;

    generate
    if (ENABLE_BACKPRESSURE == 0) begin : g_out_legacy
        assign valid_out = dp_valid_out;
        assign data_out  = dp_data_out;
    end else begin : g_out_bp
        assign valid_out = out_full;
        assign data_out  = out_data;
    end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_full <= 1'b0;
        end else begin
            if (out_full && out_ready_in)
                out_full <= 1'b0;
            if (dp_valid_out) begin
                out_full <= 1'b1;
            end
        end
    end
    // [K1-MBV2] out_data is skid DATA: sampled downstream only under
    // out_full (reset-kept); written only under dp_valid_out (reset-kept).
    always @(posedge clk) begin
        if (dp_valid_out) out_data <= dp_data_out;
    end

    // ----------------- start_pulse generator (mirrors conv3x3 ref) -----------------
    reg started, start_pulse, pending_rearm;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            started       <= 1'b0;
            start_pulse   <= 1'b0;
            pending_rearm <= 1'b0;
        end else begin
            start_pulse <= 1'b0;
            if (sched_out_frame_done) pending_rearm <= 1'b1;
            if (!started) begin
                started     <= 1'b1;
                start_pulse <= 1'b1;
            end else if (pending_rearm && !mac_busy && !skid_block) begin
                started       <= 1'b0;
                pending_rearm <= 1'b0;
            end
        end
    end

    wire stall_in = mac_busy || skid_block;

    // ----------------- coord_scheduler -----------------
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

    // ----------------- line_buf_window (IC=C=32 packed) -----------------
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
        // [FIT-FIX 2026-06-02] map the shallow-wide depthwise per-slot buffers to
        // RAMB36 (not width-binding URAM288); byte-exact, URAM reserved for engine.
        .LINE_BUF_USE_URAM(0)
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
        .channel_select(current_global_oc),
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

    // ====================================================================
    // DEPTHWISE DATAPATH (inlined fork of conv_datapath.v)
    // ====================================================================
    // Identical FSM/pipeline to conv_datapath EXCEPT:
    //   - K_TOTAL = KH*KW (per-channel taps; no IC dim)
    //   - tap selector indexes the per-channel window (lane A: chan_window_flat,
    //     lane B: window_flat_w extract) -- 9 taps read in parallel (MP_K=9)
    //   - one accumulator per LANE = one accumulator per output channel of
    //     the current OC pass; NO cross-channel reduction.
    // [812-PAIR] per-pass cycle count = PAIR_STEPS + 3 (q1/q2 drain) + 3
    // (BIAS/SCALE/OUTPUT) = 8 + 6 = 14 cycles (was 16 + 6 = 22).
    // OC_PASSES = 2. Total compute = 2*14 = 28 cycles per pixel (was 44).

    localparam integer PROD_W        = 16;
    localparam integer ACC_W         = 24;
    localparam integer BIAS_W        = 32;
    localparam integer BIASED_W      = 34;
    localparam integer SCALE_CONST_W = 16;
    // [DW-CONSTSHIFT 2026-06-10] constant-shift requant (FIT-FIX form proven on the ResNet
    // engine requant_pipeline.v 2026-06-07): the scale .mem now holds the pre-widened
    // mult' = mult << (DW_FIXED_SHIFT - shift) so the variable per-OC shift + variable
    // round decode collapse into ONE compile-time arithmetic shift + constant round.
    // Byte-exact identity (shift in [0,23], mult in [1,32767]):
    //   floor((x*mult + 2^(s-1))/2^s) == floor((x*(mult<<(23-s)) + 2^22)/2^23).
    localparam integer MULTP_W       = 32; // signed operand width for mult' ({1'b0, slot[30:0]})
    localparam integer SCALED_W      = BIASED_W + MULTP_W; // 66 (34b x 32b product, no truncation)
    localparam integer DW_FIXED_SHIFT = 23;
    localparam signed [SCALED_W-1:0] DW_ROUND_CONST =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (DW_FIXED_SHIFT - 1);

    localparam signed [SCALE_CONST_W-1:0] SCALE_MULT_CONST = SCALE_MULT[SCALE_CONST_W-1:0];
    localparam signed [SCALED_W-1:0]      SCALE_ROUND_BIAS =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);

    localparam [2:0] ST_IDLE   = 3'd0;
    localparam [2:0] ST_MAC    = 3'd1;
    localparam [2:0] ST_BIAS   = 3'd2;
    localparam [2:0] ST_SCALE  = 3'd3;
    localparam [2:0] ST_OUTPUT = 3'd4;

    reg [2:0] state;

    reg signed [ACC_W-1:0]    acc    [0:MP-1];
    reg signed [BIASED_W-1:0] biased [0:MP-1];
    (* use_dsp = "yes" *) reg signed [SCALED_W-1:0] scaled [0:MP-1];
    reg signed [SCALED_W-1:0] v_tmp;

    integer i, lane_i;
    integer bias_oc, out_oc, sc_oc;

    // (k_counter / lane_counter / oc_group regs and current_global_oc /
    // weight_read_addr wires hoisted above the line_buf_window instantiation
    // -- see top of module.)

    // Tap selector: chan_window_flat layout is (kh*KW + kw)*8 +: 8 for the
    // SINGLE channel that line_buf_window was asked to expose via
    // channel_select (= current_global_oc, wired at the instantiation). The
    // channel index is no longer part of the byte address -- it is selected
    // inside line_buf_window -- so the tap is just a 9-wide linear index
    // tap_k_lin = kh*KW + kw (0..8). ZERO arithmetic change: chan_window_flat
    // byte tap_k_lin is bit-identical to the old window_flat byte at
    // ((kh*KW+kw)*C + current_global_oc).
    // ---- Tap-parallel read: pull all KH*KW=9 weights + 9 window bytes for the
    // current channel at once. chan_window_flat byte kk (0..8) is the (kh*KW+kw)
    // tap for the channel line_buf_window exposes via channel_select
    // (= current_global_oc) -- bit-identical to the baseline's per-tap read.
    reg signed [7:0] weight_q [0:MP_K-1];
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

    // ---- 9 parallel products (one DSP per tap), registered at the SAME pipeline
    // stage the baseline registers its single `mul_q`. The tree-sum is done
    // COMBINATIONALLY in the accumulate stage so the q1->q2 valid pipeline depth is
    // BIT-FOR-BIT identical to the baseline (2 stages). Each product is an
    // independently-typed signed [PROD_W-1:0] reg so the multiply is PROD_W-wide
    // (NOT outer $signed(a*b), which self-determines to 8-bit and truncates).
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] prod_q [0:MP_K-1];
    // [812-PAIR] lane-B product bank (9 more DSP-class multipliers).
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] prod_qb [0:MP_K-1];

    reg                  mac_valid_q1;
    reg [3:0]            mac_lane_q1;
    reg [5:0]            mac_global_oc_q1;
    reg                  mac_done_issuing;

    reg                  mac_valid_q2;
    reg [3:0]            mac_lane_q2;
    reg [5:0]            mac_global_oc_q2;

    integer pp;
    // Combinational tree-sum of the 9 registered products into one ACC_W value.
    // Integer addition is associative -> this equals the baseline's serial
    // accumulation of the 9 per-tap products bit-for-bit.
    reg signed [ACC_W-1:0] sum_comb;
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

    assign mac_busy = (state != ST_IDLE);

    wire start_mac = sched_output_fires;

    // [K1-MBV2] Block A: DATAPATH registers (sync-only, no reset) -- same
    // method as ResNet K1 P2 (apply_k1_fdce_recode.py). prod_q (and its
    // [812-PAIR] lane-B twin prod_qb) is rewritten every cycle from the
    // (no-reset) weight_q/tap_q (weight_qb/tap_qb) stage and only reaches
    // acc under mac_valid_q2 (reset-kept); acc is sync-cleared on ST_IDLE&
    // start_mac / ST_OUTPUT oc-advance BEFORE the first gated accumulate of
    // every pass; biased/scaled/dp_data_out follow strict write(STn)->read(STn+1)
    // ordering and dp_data_out is only consumed under reset-kept valid/busy
    // control. acc clears are placed LAST (NBA last-write-wins parity with
    // the original single block). i/lane_i/bias_oc/sc_oc/out_oc/v_tmp
    // are referenced ONLY by this block after the move.
    always @(posedge clk) begin
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
            if (state == ST_BIAS) begin
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                        bias_oc = oc_group * MP + lane_i;
                        if (bias_oc < C)
                            biased[lane_i] <= $signed(acc[lane_i]) + $signed(biases[bias_oc]);
                        else
                            biased[lane_i] <= {BIASED_W{1'b0}};
                    end
            end
            if (state == ST_SCALE) begin
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                        sc_oc = oc_group * MP + lane_i;
                        if (sc_oc < C)
                            // [DW-CONSTSHIFT] slot = pre-widened mult' (bits [30:0], positive)
                            scaled[lane_i] <= $signed(biased[lane_i]) * $signed({1'b0, scale_rom[sc_oc][30:0]});
                        else
                            scaled[lane_i] <= {SCALED_W{1'b0}};
                    end
            end
            if (state == ST_OUTPUT) begin
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                        out_oc = oc_group * MP + lane_i;
                        if (out_oc < C) begin
                            // [INVARIANT:ROUNDING]
                            // [DW-CONSTSHIFT] per-OC shift folded offline into mult' ->
                            // constant round + compile-time shift (no barrel shifter)
                            v_tmp = (scaled[lane_i] + DW_ROUND_CONST) >>> DW_FIXED_SHIFT;
                            dp_data_out[out_oc*8 +: 8] <=
                                (v_tmp >  127) ?  8'sd127 :
                                (v_tmp < -128) ? -8'sd128 : v_tmp[7:0];
                        end
                    end
            end
            // Accumulator clears LAST: textual-order parity with the
            // original single block (clears overrode the accumulate).
            if (state == ST_IDLE && start_mac) begin
                for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1)
                    acc[lane_i] <= {ACC_W{1'b0}};
            end
            if (state == ST_OUTPUT && oc_group != OC_PASSES - 1) begin
                for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1)
                    acc[lane_i] <= {ACC_W{1'b0}};
            end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= ST_IDLE;
            dp_valid_out     <= 1'b0;
            lane_counter     <= 3'd0;
            oc_group         <= 3'd0;
            mac_valid_q1     <= 1'b0;
            mac_lane_q1      <= 3'd0;
            mac_global_oc_q1 <= 6'd0;
            mac_valid_q2     <= 1'b0;
            mac_lane_q2      <= 3'd0;
            mac_global_oc_q2 <= 6'd0;
            mac_done_issuing <= 1'b0;
        end else begin
            dp_valid_out <= 1'b0;

            mac_valid_q2     <= mac_valid_q1;
            mac_lane_q2      <= mac_lane_q1;
            mac_global_oc_q2 <= mac_global_oc_q1;

            case (state)
                ST_IDLE: begin
                    if (start_mac) begin
                        state            <= ST_MAC;
                        lane_counter     <= 3'd0;
                        oc_group         <= 3'd0;
                        mac_valid_q1     <= 1'b0;
                        mac_valid_q2     <= 1'b0;
                        mac_done_issuing <= 1'b0;
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
                        // [812-PAIR] q1 carries the EVEN lane-A indices; the
                        // accumulate stage derives lane B as |1. One pair (2
                        // channels) issues per cycle -> PAIR_STEPS issues/pass.
                        mac_lane_q1      <= pair_lane_a;
                        mac_global_oc_q1 <= current_global_oc;
                        mac_valid_q1     <= 1'b1;

                        if (lane_counter == (PAIR_STEPS-1)) begin
                            lane_counter     <= 3'd0;
                            mac_done_issuing <= 1'b1;
                        end else begin
                            lane_counter <= lane_counter + 3'd1;
                        end
                    end
                end

                ST_BIAS: begin
                    // [K1-MBV2] biased[] writes moved to Block A (sync-only).
                    state <= ST_SCALE;
                end

                ST_SCALE: begin
                    // [K1-MBV2] scaled[] writes moved to Block A (sync-only).
                    state <= ST_OUTPUT;
                end

                ST_OUTPUT: begin
                    // [K1-MBV2] dp_data_out[]/v_tmp writes moved to Block A (sync-only).
                    if (oc_group == (OC_PASSES - 1)) begin
                        // [INVARIANT:VALID_OUT_LATENCY]
                        // [THROUGHPUT A2 2026-06-03] was hardcoded 3'd7 (MP=4 -> OC_PASSES=8).
                        // With MP=8 -> OC_PASSES=4; deriving from OC_PASSES avoids 4 phantom
                        // passes (byte-exact either way, but the constant cost 2x the MAC time).
                        dp_valid_out <= 1'b1;
                        state     <= ST_IDLE;
                    end else begin
                        oc_group     <= oc_group + 3'd1;
                        lane_counter <= 3'd0;
                        state <= ST_MAC;
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
