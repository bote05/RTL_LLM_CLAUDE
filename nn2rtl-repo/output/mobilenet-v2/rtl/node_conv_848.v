// node_conv_848 -- MobileNet-v2 depthwise conv 3x3, STRIDE 2, pad 1, C=192,
//   IH=IW=28, OH=OW=14.  PORTED to the conv_812/conv_818 split-architecture
//   (coord_scheduler + rtl_library/line_buf_window.v) so the input window is
//   buffered in SYNC BRAM/URAM (KH=3 rows) instead of the prior 13-bank
//   ASYNC-read full-frame line_buf (which Vivado mapped to distributed
//   LUT-RAM).  Output is STREAMED per output pixel (no out_buf), exactly
//   like conv_812/conv_818.
//
//   PURE PARAMETER SUBSTITUTION of the conv_818 reference (C=96): only the
//   module name, data bus width (C*8 = 1536), geometry localparams
//   (C/IH/IW/OH/OW/SH/SW), OC_PASSES=ceil(C/4)=48, the SCALE_MULT/SCALE_SHIFT
//   constants (copied VERBATIM from the OLD node_conv_848 localparams for
//   byte-exact requant), the weight/bias $readmemh paths, and the C-dependent
//   field widths (oc_group=$clog2(OC_PASSES)=6 bits, current_global_oc/global_oc
//   regs widened to 8 bits to hold 0..191) differ.  The conv math, requant
//   (bias add, SCALE_MULT/SCALE_SHIFT, round-half-away-from-zero via
//   SCALE_ROUND_BIAS, INT8 saturate), channel/output ordering, and pipeline
//   cadence are IDENTICAL.
//
//   Pipeline latency (LayerIR-authoritative) = 2048 cycles =
//     fill_rows*(IW+PW) + fill_cols + OC_PASSES*(MP*K_TOTAL+6) + 1
//     = 1*29 + 2 + 48*42 + 1 = 31 + 2016 + 1 = 2048.
//
//   WIDE-CHANNEL: line_buf_window maps its per-slot mem to ram_style="ultra"
//   (URAM) which packs channels along DEPTH on U250.  The datapath taps
//   window_flat by channel index (no giant LUT-mux flatten), so the larger
//   C=192 inherits the URAM packing exactly like conv_818 (C=96).
//
//   Quantization (verbatim from the prior node_conv_848):
//     SCALE_MULT/2^SCALE_SHIFT = 19037 / 2^22 ~= 0.004538774

`timescale 1ns/1ps
`default_nettype none

// PARAM-GATED ELASTIC BACKPRESSURE (ENABLE_BACKPRESSURE, default 0):
//   * ==0 (default): bit/cycle-IDENTICAL to the legacy module. out_ready_in is
//     IGNORED; skid_block is a constant 0 (scheduler/rearm never freeze); the
//     external valid_out/data_out come DIRECTLY from the datapath regs
//     (dp_valid_out/dp_data_out). The per-module verify TB (param=0) is byte-exact.
//   * ==1: 1-deep output skid (out_full/out_data) captures the datapath's 1-cycle
//     valid_out pulse; skid_block = out_full && !out_ready_in feeds stall_in +
//     blocks the frame rearm, freezing the scheduler/datapath while a beat is
//     parked and the downstream is not ready. Arithmetic unchanged (== conv_812).
module node_conv_848 #(
    parameter ENABLE_BACKPRESSURE = 0,
    parameter WEIGHTS_PATH = "C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_848_weights.hex",
    parameter BIAS_PATH    = "C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_848_bias.hex"
)(
    input  wire           clk,
    input  wire           rst_n,
    input  wire           valid_in,
    output wire           ready_in,
    input  wire [1535:0]  data_in,
    input  wire           out_ready_in,   // NEW: downstream-ready (ignored when ENABLE_BACKPRESSURE==0)
    output wire           valid_out,
    output wire [1535:0]   data_out
);

    // ---- datapath output regs + 1-deep output skid (see header) ----
    reg                  dp_valid_out;
    reg  [1535:0]     dp_data_out;
    reg                  out_full;
    reg  [1535:0]     out_data;
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
            out_data <= 1536'd0;
        end else begin
            if (out_full && out_ready_in)
                out_full <= 1'b0;
            if (dp_valid_out) begin
                out_data <= dp_data_out;
                out_full <= 1'b1;
            end
        end
    end

    // ----------------- Geometry -----------------
    localparam integer C         = 192;
    localparam integer IH        = 28;
    localparam integer IW        = 28;
    localparam integer OH        = 14;
    localparam integer OW        = 14;
    localparam integer KH        = 3;
    localparam integer KW        = 3;
    localparam integer SH        = 2;
    localparam integer SW        = 2;
    localparam integer PH        = 1;
    localparam integer PW        = 1;
    localparam integer K_TOTAL   = KH * KW;          // 9
    localparam integer MP        = 16;
    localparam integer MP_K      = 9;            // tap-parallel width (= K_TOTAL)
    localparam integer K_GROUPS  = K_TOTAL / MP_K; // = 1 (single-shot reduction)
    localparam integer OC_PASSES = (C + MP - 1) / MP; // 48

    // ----------------- Quantization (verbatim from prior node) -----------------
    // compute_scale_approx(0.004538737) picks MULT=19037, SHIFT=22.
    localparam integer SCALE_MULT  = 19037;
    localparam integer SCALE_SHIFT = 22;

    // ----------------- Weight / Bias ROMs -----------------
    // Canonical names + layout identical to the prior node (oc*K_TOTAL + k).
    (* rom_style = "block", ram_style = "block" *)
    reg signed [7:0]  weights [0:C*K_TOTAL-1];
    (* rom_style = "block", ram_style = "block" *)
    reg signed [31:0] biases  [0:C-1];

    initial begin
        $readmemh("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_848_weights.hex", weights);
        $readmemh("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_848_bias.hex", biases);
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
    // current_global_oc). Replaces the wide KH*KW*C*8 window_flat to eliminate
    // the cross-channel wide-mux routing congestion (the ResNet-route failure
    // at conv_292). KH*KW bytes = 9 taps for the single selected channel.
    wire [KH*KW*8-1:0]                chan_window_flat;
    wire                              mac_busy;
    (* max_fanout = 256 *) reg [3:0] lane_counter;
    reg [5:0] oc_group;            // OC_PASSES=12 needs 4 bits (0..11); 6 bits is still ample
    (* max_fanout = 256 *) wire [7:0]  current_global_oc = oc_group * MP + lane_counter; // 0..191 -> 8 bits
    wire [15:0] weight_base_addr  = current_global_oc * K_TOTAL;  // contiguous K_TOTAL taps for this channel

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

    // [FIT-FIX 2026-06-06] line_buf_window tiled-storage burst stall (TILE_STORAGE>0).
    wire lbw_mem_busy;
    wire stall_in = mac_busy || skid_block || lbw_mem_busy;

    // ----------------- coord_scheduler (universal; SH=SW=2 handled internally) ---
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

    // ----------------- line_buf_window (IC=C=192 packed, SYNC BRAM/URAM) -----------
    line_buf_window #(
        .IC(C), .IW(IW), .IH(IH),
        .KH(KH), .KW(KW), .PW(PW), .PH(PH),
        .EXPOSE_FULL_WINDOW(0),
        // [FIT-FIX 2026-06-02] map the shallow-wide depthwise per-slot buffers to
        // RAMB36 (not width-binding URAM288); byte-exact, URAM reserved for engine.
        .LINE_BUF_USE_URAM(0),
        // [FIT-FIX 2026-06-06] deep-narrow tiled per-slot storage: 32 ch/tile.
        // Burst-serialized R/W stalls the scheduler via mem_busy -> atomic ->
        // byte-exact vs legacy (TILE_STORAGE=0). Verified by verify_lbw_c960/tb_equiv.
        .TILE_STORAGE(32)
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
        .window_flat(),
        .mem_busy(lbw_mem_busy)
    );

    assign ready_in = sched_ready_in;

    // ====================================================================
    // DEPTHWISE DATAPATH (inlined fork of conv_datapath.v, == conv_818)
    // ====================================================================
    //   - K_TOTAL = KH*KW (per-channel taps; no IC dim)
    //   - tap selector indexes window_flat at (kh, kw, current_channel)
    //   - one accumulator per LANE = one accumulator per output channel of
    //     the current OC pass; NO cross-channel reduction.
    // Per-pass cycle count = MP*K_TOTAL + 6 = 4*9 + 6 = 42 cycles.
    // OC_PASSES = 48. Total compute = 48*42 = 2016. Spatial fill = 1*29 + 2
    // = 31. +1 for the registered output_fires => first valid_out at
    // exactly pipeline_latency_cycles = 2048.

    localparam integer PROD_W        = 16;
    localparam integer ACC_W         = 24;
    localparam integer BIAS_W        = 32;
    localparam integer BIASED_W      = 34;
    localparam integer SCALE_CONST_W = 16;
    localparam integer SCALED_W      = BIASED_W + SCALE_CONST_W; // 50

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
    integer bias_oc, out_oc;


    // Tap selector: the line_buf_window now exposes the NARROW per-channel
    // window `chan_window_flat`, which packs the KH*KW taps for the SINGLE
    // channel `channel_select` (= current_global_oc, wired at the lbw
    // instance). The channel selection happens inside line_buf_window (a
    // single C-way mux per tap) instead of a giant cross-channel mux into the
    // wide window_flat -- this is the routing-congestion fix. The tap index is
    // therefore a 9-wide linear index tap_k_lin = kh*KW + kw (0..8) into
    // chan_window_flat. BYTE-EXACT: chan_window_flat[(kh*KW+kw)*8 +: 8] equals
    // the byte the old window_flat gave at ((kh*KW+kw)*IC + current_global_oc).
    // ---- Tap-parallel read: pull all KH*KW=9 weights + 9 window bytes for the
    // current channel at once. chan_window_flat byte kk (0..8) is the (kh*KW+kw)
    // tap for the channel line_buf_window exposes via channel_select
    // (= current_global_oc) -- bit-identical to the baseline's per-tap read.
    reg signed [7:0] weight_q [0:MP_K-1];
    reg signed [7:0] tap_q    [0:MP_K-1];
    integer kk;
    always @(posedge clk) begin
        for (kk = 0; kk < MP_K; kk = kk + 1) begin
            weight_q[kk] <= weights[weight_base_addr + kk];
            tap_q[kk]    <= $signed(chan_window_flat[kk*8 +: 8]);
        end
    end

    // ---- 9 parallel products (one DSP per tap), registered at the SAME pipeline
    // stage the baseline registers its single `mul_q`. The tree-sum is done
    // COMBINATIONALLY in the accumulate stage so the q1->q2 valid pipeline depth is
    // BIT-FOR-BIT identical to the baseline (2 stages). Each product is an
    // independently-typed signed [PROD_W-1:0] reg so the multiply is PROD_W-wide
    // (NOT outer $signed(a*b), which self-determines to 8-bit and truncates).
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] prod_q [0:MP_K-1];

    reg                  mac_valid_q1;
    reg [3:0]            mac_lane_q1;
    reg [7:0]            mac_global_oc_q1;
    reg                  mac_done_issuing;

    reg                  mac_valid_q2;
    reg [3:0]            mac_lane_q2;
    reg [7:0]            mac_global_oc_q2;

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

    assign mac_busy = (state != ST_IDLE);

    wire start_mac = sched_output_fires;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= ST_IDLE;
            dp_valid_out     <= 1'b0;
            dp_data_out      <= 1536'd0;
            lane_counter     <= 4'd0;
            oc_group         <= 6'd0;
            mac_valid_q1     <= 1'b0;
            mac_lane_q1      <= 4'd0;
            mac_global_oc_q1 <= 8'd0;
            mac_valid_q2     <= 1'b0;
            mac_lane_q2      <= 4'd0;
            mac_global_oc_q2 <= 8'd0;
            mac_done_issuing <= 1'b0;
            v_tmp            <= {SCALED_W{1'b0}};
            for (i = 0; i < MP_K; i = i + 1)
                prod_q[i] <= {PROD_W{1'b0}};
            for (i = 0; i < MP; i = i + 1) begin
                acc[i]    <= {ACC_W{1'b0}};
                biased[i] <= {BIASED_W{1'b0}};
                scaled[i] <= {SCALED_W{1'b0}};
            end
        end else begin
            dp_valid_out <= 1'b0;

            // Stage 2: registered parallel multiplies (one DSP per tap).
            for (i = 0; i < MP_K; i = i + 1)
                prod_q[i] <= $signed(weight_q[i]) * $signed(tap_q[i]);
            mac_valid_q2     <= mac_valid_q1;
            mac_lane_q2      <= mac_lane_q1;
            mac_global_oc_q2 <= mac_global_oc_q1;

            // Stage 3: accumulator add (gated by lane validity)
            if (mac_valid_q2 && mac_global_oc_q2 < C[7:0]) begin
                acc[mac_lane_q2] <= acc[mac_lane_q2] + $signed(sum_comb);
            end

            case (state)
                ST_IDLE: begin
                    if (start_mac) begin
                        state            <= ST_MAC;
                        lane_counter     <= 4'd0;
                        oc_group         <= 6'd0;
                        mac_valid_q1     <= 1'b0;
                        mac_valid_q2     <= 1'b0;
                        mac_done_issuing <= 1'b0;
                        for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1)
                            acc[lane_i] <= {ACC_W{1'b0}};
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

                        if (lane_counter == (MP-1)) begin
                            lane_counter     <= 4'd0;
                            mac_done_issuing <= 1'b1;
                        end else begin
                            lane_counter <= lane_counter + 4'd1;
                        end
                    end
                end

                ST_BIAS: begin
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                        bias_oc = oc_group * MP + lane_i;
                        if (bias_oc < C)
                            biased[lane_i] <= $signed(acc[lane_i]) + $signed(biases[bias_oc]);
                        else
                            biased[lane_i] <= {BIASED_W{1'b0}};
                    end
                    state <= ST_SCALE;
                end

                ST_SCALE: begin
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1)
                        scaled[lane_i] <= $signed(biased[lane_i]) * $signed(SCALE_MULT_CONST);
                    state <= ST_OUTPUT;
                end

                ST_OUTPUT: begin
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                        out_oc = oc_group * MP + lane_i;
                        if (out_oc < C) begin
                            // [INVARIANT:ROUNDING]
                            v_tmp = (scaled[lane_i] + SCALE_ROUND_BIAS) >>> SCALE_SHIFT;
                            dp_data_out[out_oc*8 +: 8] <=
                                (v_tmp >  127) ?  8'sd127 :
                                (v_tmp < -128) ? -8'sd128 : v_tmp[7:0];
                        end
                    end

                    if (oc_group == OC_PASSES - 1) begin
                        // [INVARIANT:VALID_OUT_LATENCY]
                        dp_valid_out <= 1'b1;
                        state     <= ST_IDLE;
                    end else begin
                        oc_group     <= oc_group + 6'd1;
                        lane_counter <= 4'd0;
                        for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1)
                            acc[lane_i] <= {ACC_W{1'b0}};
                        state <= ST_MAC;
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
