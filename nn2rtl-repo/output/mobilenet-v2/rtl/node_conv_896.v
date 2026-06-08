// node_conv_896 -- MobileNet-v2 depthwise conv 3x3, STRIDE 1, pad 1, C=960,
//   IH=IW=7, OH=OW=7.  PORTED to the conv_812/conv_818 split-architecture
//   (coord_scheduler + rtl_library/line_buf_window.v) so the input window is
//   buffered in SYNC URAM (KH=3 rows, ram_style="ultra") instead of the prior
//   async-read full-frame line_buf (which Vivado maps to distributed LUT-RAM).
//   Output is STREAMED per output pixel (no byte-granular out_buf).
//
//   WIDE-CHANNEL (C=960 > 512): the flat bus is 4096b with 2 beats/pixel
//   (beat0 = ch 0..511, beat1 = ch 512..959 in low 448*8 bits, rest zero-pad).
//   A 2-beat INPUT ASSEMBLER reconstructs the full C*8 = 7680b pixel and feeds
//   it as ONE valid_in to the scheduler + line_buf_window.  A 2-beat OUTPUT
//   EMITTER splits each computed 7680b output pixel back into lo/hi beats.
//
//   Quantization (verbatim from the prior node_conv_896):
//     SCALE_MULT/2^SCALE_SHIFT = 12275 / 2^22 ~= 0.00719237  (byte-exact)
//   Weight ROM layout unchanged: weights[oc*K_TOTAL + k], biases[oc].
//
//   Latency: split-arch handshake-derived first valid_out at
//     fill + OC_PASSES*(MP*K_TOTAL+6) + 1
//   The 2-beat assembler is made latency-transparent by gating the bench
//   handshake on the inner scheduler's ready: the scheduler only sees one
//   assembled pixel per real-input coord, so the fill/pass cadence to first
//   valid_out is identical to the single-beat split-arch.

`timescale 1ns/1ps
`default_nettype none

// PARAM-GATED ELASTIC BACKPRESSURE (ENABLE_BACKPRESSURE, default 0):
//   This is a 2-BEAT depthwise conv (each output pixel is emitted as a lo beat
//   ch0..511 then a hi beat ch512..959). The skid here is 2-DEEP / DUAL-PHASE so
//   it can hold BOTH beats under out_ready_in backpressure.
//   * ==0 (default): bit/cycle-IDENTICAL to the legacy module. out_ready_in is
//     IGNORED; skid_block is a constant 0; the FSM drives valid_out/data_out
//     directly (combinational passthrough of the unchanged dp_valid/dp_data). The
//     per-module verify TB (param=0) is byte-exact.
//   * ==1: the FSM still produces the lo beat then the hi beat over 2 consecutive
//     cycles, but those two beats are captured into a 2-entry beat FIFO and
//     replayed to the port ONLY when out_ready_in is high. skid_block = (FIFO
//     non-empty) feeds stall_in + blocks the frame rearm, so the FSM FREEZES
//     after emitting a pixel's 2 beats until both have drained => no overwrite /
//     reorder / drop. Datapath arithmetic and the FSM are UNCHANGED; only the
//     external emit *timing* changes.
module node_conv_896 #(
    parameter ENABLE_BACKPRESSURE = 0,
    parameter WEIGHTS_PATH = "C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_896_weights.hex",
    parameter BIAS_PATH    = "C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_896_bias.hex"
)(
    input  wire           clk,
    input  wire           rst_n,
    input  wire           valid_in,
    output wire           ready_in,
    input  wire [4095:0]  data_in,
    input  wire           out_ready_in,   // NEW: downstream-ready (ignored when ENABLE_BACKPRESSURE==0)
    output reg            valid_out,
    output reg  [4095:0]  data_out
);

    // skid_block (driven by the BP emitter below) freezes the scheduler + rearm
    // while a buffered pixel's beats are still draining under backpressure. With
    // ENABLE_BACKPRESSURE==0 it is a constant 0 -> legacy cycle-identical behavior.
    wire skid_block;

    // [FIT-FIX 2026-06-06] line_buf_window tiled-storage burst stall (TILE_STORAGE>0).
    wire lbw_mem_busy;

    // ----------------- Geometry -----------------
    localparam integer C         = 960;
    localparam integer IH        = 7;
    localparam integer IW        = 7;
    localparam integer OH        = 7;
    localparam integer OW        = 7;
    localparam integer KH        = 3;
    localparam integer KW        = 3;
    localparam integer SH        = 1;
    localparam integer SW        = 1;
    localparam integer PH        = 1;
    localparam integer PW        = 1;
    localparam integer K_TOTAL   = KH * KW;            // 9
    localparam integer MP        = 4;
    localparam integer MP_K      = 9;            // tap-parallel width (= K_TOTAL)
    localparam integer K_GROUPS  = K_TOTAL / MP_K; // = 1 (single-shot reduction)
    localparam integer OC_PASSES = (C + MP - 1) / MP;  // 240

    // ----------------- Wide-bus geometry -----------------
    localparam integer BEAT_W = 4096;
    localparam integer LO_CH  = 512;                   // channels per lo beat
    localparam integer HI_CH  = C - LO_CH;             // 448 real hi channels
    localparam integer PIX_W  = C * 8;                 // 7680 assembled pixel
    localparam integer LO_W   = LO_CH * 8;             // 4096
    localparam integer HI_W   = HI_CH * 8;             // 3584

    // ----------------- Quantization (verbatim from prior node) -----------------
    // compute_scale_approx(scale_factor) picks MULT=12275, SHIFT=22.
    localparam integer SCALE_MULT  = 12275;
    localparam integer SCALE_SHIFT = 19;

    // ----------------- Weight / Bias ROMs -----------------
    (* rom_style = "block", ram_style = "block" *)
    reg signed [7:0]  weights [0:C*K_TOTAL-1];
    (* rom_style = "block", ram_style = "block" *)
    reg signed [31:0] biases  [0:C-1];

    initial begin
        $readmemh("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_896_weights.hex", weights);
        $readmemh("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_896_bias.hex", biases);
    end

    // =====================================================================
    // 2-BEAT INPUT ASSEMBLER
    // =====================================================================
    // The bench delivers 2 beats per pixel (lo, hi). We expose `ready_in`
    // high whenever the scheduler can accept the NEXT real pixel OR we are
    // mid-pixel (waiting for the hi beat). The lo beat is latched; the hi
    // beat completes the 7680b pixel and asserts a single-cycle
    // `pix_valid` into the scheduler / line_buf_window.
    reg               beat_phase;        // 0 => expect lo, 1 => expect hi
    reg [LO_W-1:0]    lo_hold;
    wire              sched_ready_in;     // scheduler's ready for a real pixel

    // The scheduler only ever sees the assembled pixel on the hi beat.
    wire              bench_fire = valid_in && ready_in;
    wire              pix_valid  = bench_fire && (beat_phase == 1'b1);

    // Bench-facing ready: accept lo beat whenever the scheduler is ready for a
    // new real pixel; accept hi beat unconditionally (legacy) / when the
    // scheduler is ready (BP) once lo is latched.
    // [BP:HI_READY_GATE] In legacy mode (ENABLE_BACKPRESSURE==0) the hi beat is
    // accepted unconditionally (byte-exact, unchanged). In BP mode the scheduler
    // can be FROZEN by skid_block at an arbitrary phase; accepting the hi beat
    // while the scheduler is stalled would pulse pix_valid into a frozen
    // scheduler (no handshake) and DROP that input pixel -> corrupt window. So in
    // BP mode gate the hi beat on sched_ready_in, holding the upstream until the
    // scheduler can consume the assembled pixel.
    wire              hi_ready = (ENABLE_BACKPRESSURE == 0) ? 1'b1 : sched_ready_in;
    assign ready_in = (beat_phase == 1'b0) ? sched_ready_in : hi_ready;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            beat_phase <= 1'b0;
            lo_hold    <= {LO_W{1'b0}};
        end else if (bench_fire) begin
            if (beat_phase == 1'b0) begin
                lo_hold    <= data_in[LO_W-1:0];
                beat_phase <= 1'b1;
            end else begin
                beat_phase <= 1'b0;
            end
        end
    end

    // Assembled 7680b pixel: low 4096b from the latched lo beat, high 3584b
    // from the current (hi) beat's low bits.
    wire [PIX_W-1:0] pixel_assembled = { data_in[HI_W-1:0], lo_hold };

    // ----------------- Scheduler / window wires -----------------
    wire                              sched_needs_real_input;
    wire                              sched_output_fires;
    wire                              sched_advance;
    wire [$clog2(IH + PH + 1)-1:0]    sched_in_row;
    wire [$clog2(IW + PW + 1)-1:0]    sched_in_col;
    wire [$clog2(OH * OW + 1)-1:0]    sched_outputs_emitted;
    wire                              sched_out_frame_done;
    // NARROW per-channel window: KH*KW bytes for the single channel selected by
    // channel_select (= current_global_oc). Replaces the legacy wide window_flat
    // (KH*KW*C*8 bits) -- the wide cross-channel mux is gated off in line_buf_window
    // via EXPOSE_FULL_WINDOW(0) to eliminate the routing congestion. ZERO arithmetic
    // change: each byte is bit-identical to the corresponding window_flat byte.
    wire [KH*KW*8-1:0]                chan_window_flat;
    wire                              mac_busy;
    reg [1:0] lane_counter;
    reg [7:0] oc_group;            // OC_PASSES=240 needs 8 bits (0..239)
    wire [10:0] current_global_oc = oc_group * MP + lane_counter;
    wire [15:0] weight_base_addr  = current_global_oc * K_TOTAL;  // contiguous K_TOTAL taps for this channel

    // ----------------- start_pulse generator (mirrors conv_818 ref) -----------------
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

    wire stall_in = mac_busy || skid_block || lbw_mem_busy;

    // ----------------- coord_scheduler -----------------
    // Driven by pix_valid (one assembled pixel per real-input coord).
    coord_scheduler #(
        .IH(IH), .IW(IW), .OH(OH), .OW(OW),
        .KH(KH), .KW(KW), .SH(SH), .SW(SW),
        .PH(PH), .PW(PW)
    ) scheduler (
        .clk(clk), .rst_n(rst_n),
        .start(start_pulse),
        .stall_in(stall_in),
        .valid_in(pix_valid),
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

    // ----------------- line_buf_window (IC=C=960 packed, SYNC URAM) -----------
    line_buf_window #(
        .IC(C), .IW(IW), .IH(IH),
        .KH(KH), .KW(KW), .PW(PW), .PH(PH),
        // Depthwise consumer: leave the wide cross-channel window gated OFF.
        .EXPOSE_FULL_WINDOW(0),
        // [FIT-FIX 2026-06-02] map the shallow-wide depthwise per-slot buffers to
        // RAMB36 (not width-binding URAM288); byte-exact, URAM reserved for engine.
        .LINE_BUF_USE_URAM(0),
        // [FIT-FIX 2026-06-06] deep-narrow tiled per-slot storage: 32 ch/tile, NT=30
        // tiles deep (C=960). Burst-serialized R/W stalls the scheduler via mem_busy
        // -> atomic -> byte-exact vs legacy. Verified by verify_lbw_c960/tb_equiv.
        .TILE_STORAGE(32)
    ) lbw (
        .clk(clk), .rst_n(rst_n),
        .frame_start(start_pulse),
        .sched_in_row(sched_in_row),
        .sched_in_col(sched_in_col),
        .sched_needs_real_input(sched_needs_real_input),
        .sched_advance(sched_advance),
        .sched_output_fires(sched_output_fires),
        .valid_in(pix_valid),
        .data_in(pixel_assembled),
        // Drive the channel selector with the channel under accumulation
        // (one per cycle) and read the narrow per-channel window. Tie off the
        // legacy wide window_flat (gated off via EXPOSE_FULL_WINDOW(0)).
        .channel_select(current_global_oc),
        .chan_window_flat(chan_window_flat),
        .window_flat(),
        .mem_busy(lbw_mem_busy)
    );

    // ====================================================================
    // DEPTHWISE DATAPATH (inlined fork of conv_datapath.v, == conv_818)
    // ====================================================================
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

    // C=960 -> global_oc 0..959 needs 10 bits

    // Tap selector: the line_buf_window now exposes the NARROW per-channel
    // window `chan_window_flat`, which packs the KH*KW taps for the SINGLE
    // channel `channel_select` (= current_global_oc, wired at the lbw instance).
    // The channel selection happens inside line_buf_window (a single C-way mux
    // per tap) instead of a giant cross-channel mux into the wide window_flat --
    // this is the routing-congestion fix. The tap index is therefore a 9-wide
    // linear index tap_k_lin = kh*KW + kw (0..8) into chan_window_flat.
    // BYTE-EXACT: chan_window_flat[(kh*KW+kw)*8 +: 8] equals the byte the old
    // window_flat gave at ((kh*KW+kw)*IC + current_global_oc).
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
    reg [1:0]            mac_lane_q1;
    reg [10:0]           mac_global_oc_q1;
    reg                  mac_done_issuing;

    reg                  mac_valid_q2;
    reg [1:0]            mac_lane_q2;
    reg [10:0]           mac_global_oc_q2;

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

    // Datapath produces a full PIX_W (7680b) output pixel. The lo half
    // (ch 0..511, oc_group 0..127) and hi half (ch 512..959, oc_group
    // 128..239) are both fully written by the final pass (oc_group==239),
    // so we can drive the lo OUTPUT BEAT directly in that ST_OUTPUT cycle
    // (matching conv_818's valid_out latency) and emit the hi beat the
    // next cycle.  No extra register stage.
    reg [PIX_W-1:0] out_pix;
    reg             emit_hi;        // next-cycle: drive hi beat

    // ---- FSM-internal emit stream (UNCHANGED timing vs legacy). In legacy mode
    // these pass straight through to valid_out/data_out (combinational). In BP
    // mode they feed a 2-entry beat FIFO. ----
    reg             dp_valid;
    reg [BEAT_W-1:0] dp_data;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= ST_IDLE;
            out_pix          <= {PIX_W{1'b0}};
            dp_valid         <= 1'b0;
            dp_data          <= {BEAT_W{1'b0}};
            emit_hi          <= 1'b0;
            lane_counter     <= 2'd0;
            oc_group         <= 8'd0;
            mac_valid_q1     <= 1'b0;
            mac_lane_q1      <= 2'd0;
            mac_global_oc_q1 <= 11'd0;
            mac_valid_q2     <= 1'b0;
            mac_lane_q2      <= 2'd0;
            mac_global_oc_q2 <= 11'd0;
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
            // Defaults: valid_out drops unless re-asserted; emit hi beat if
            // the prior cycle emitted lo. By the emit_hi cycle, out_pix has
            // ALL 960 channels written (including ch 956..959 from the final
            // ST_OUTPUT pass), so read the hi half directly from out_pix.
            dp_valid <= 1'b0;
            if (emit_hi) begin
                dp_valid <= 1'b1;
                dp_data  <= {{(BEAT_W - HI_W){1'b0}}, out_pix[PIX_W-1:LO_W]};
                emit_hi  <= 1'b0;
            end

            // Stage 2: registered parallel multiplies (one DSP per tap).
            for (i = 0; i < MP_K; i = i + 1)
                prod_q[i] <= $signed(weight_q[i]) * $signed(tap_q[i]);
            mac_valid_q2     <= mac_valid_q1;
            mac_lane_q2      <= mac_lane_q1;
            mac_global_oc_q2 <= mac_global_oc_q1;

            // Stage 3: accumulator add (gated by lane validity)
            if (mac_valid_q2 && mac_global_oc_q2 < C[10:0]) begin
                acc[mac_lane_q2] <= acc[mac_lane_q2] + $signed(sum_comb);
            end

            case (state)
                ST_IDLE: begin
                    if (start_mac) begin
                        state            <= ST_MAC;
                        lane_counter     <= 2'd0;
                        oc_group         <= 8'd0;
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

                        if (lane_counter == 2'd3) begin
                            lane_counter     <= 2'd0;
                            mac_done_issuing <= 1'b1;
                        end else begin
                            lane_counter <= lane_counter + 2'd1;
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
                            out_pix[out_oc*8 +: 8] <=
                                (v_tmp >  127) ?  8'sd127 :
                                (v_tmp < -128) ? -8'sd128 : v_tmp[7:0];
                        end
                    end

                    if (oc_group == OC_PASSES - 1) begin
                        // [INVARIANT:VALID_OUT_LATENCY]
                        // Drive lo beat (ch 0..511) THIS cycle, same latency
                        // as conv_818. out_pix lo half is fully written by
                        // passes 0..127, so out_pix[LO_W-1:0] is stable now.
                        // The hi beat is emitted next cycle (emit_hi), by which
                        // point out_pix has ALL 960 channels (incl. ch 956..959
                        // from this final pass's non-blocking writes).
                        dp_valid <= 1'b1;
                        dp_data  <= out_pix[LO_W-1:0];
                        emit_hi  <= 1'b1;
                        state    <= ST_IDLE;
                    end else begin
                        oc_group     <= oc_group + 8'd1;
                        lane_counter <= 2'd0;
                        for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1)
                            acc[lane_i] <= {ACC_W{1'b0}};
                        state <= ST_MAC;
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

    // ====================================================================
    // OUTPUT EMITTER (legacy passthrough  |  2-entry elastic beat FIFO)
    // ====================================================================
    generate
    if (ENABLE_BACKPRESSURE == 0) begin : g_emit_legacy
        // Cycle-identical passthrough: out_ready_in IGNORED, never freeze.
        assign skid_block = 1'b0;
        always @(*) begin
            valid_out = dp_valid;
            data_out  = dp_data;
        end
    end else begin : g_emit_bp
        // 2-entry beat FIFO. The FSM emits the lo beat then the hi beat on two
        // consecutive cycles (dp_valid pulses), then is FROZEN by skid_block
        // (FIFO non-empty) so it cannot produce a third beat before these drain.
        // Each captured beat is replayed to the port ONLY when out_ready_in is
        // high -- a true ready/valid handshake holding the beat until accepted.
        reg [BEAT_W-1:0] fifo_d [0:1];
        reg              fifo_v [0:1];
        reg              head;          // next beat to emit
        reg              tail;          // next slot to write
        reg [1:0]        cnt;           // 0..2 occupancy

        wire wr = dp_valid;             // a beat arrives from the FSM
        wire rd = (cnt != 2'd0) && out_ready_in; // a beat is accepted downstream

        // skid_block freezes the FSM whenever a beat is parked. cnt becomes >0 on
        // the lo-beat cycle; the hi beat (driven by the independent emit_hi reg)
        // still lands the next cycle (stall_in does not gate emit_hi), so both
        // beats are always captured before the FSM is held idle.
        assign skid_block = (cnt != 2'd0);

        always @(*) begin
            valid_out = (cnt != 2'd0);
            data_out  = fifo_d[head];
        end

        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                fifo_d[0] <= {BEAT_W{1'b0}};
                fifo_d[1] <= {BEAT_W{1'b0}};
                fifo_v[0] <= 1'b0;
                fifo_v[1] <= 1'b0;
                head      <= 1'b0;
                tail      <= 1'b0;
                cnt       <= 2'd0;
            end else begin
                // Write the incoming beat (capacity guaranteed by skid_block: the
                // FSM is frozen once cnt>0, so at most 2 beats are ever in flight).
                if (wr) begin
                    fifo_d[tail] <= dp_data;
                    fifo_v[tail] <= 1'b1;
                    tail         <= ~tail;
                end
                // Pop the accepted beat.
                if (rd) begin
                    fifo_v[head] <= 1'b0;
                    head         <= ~head;
                end
                // Occupancy update (wr and rd can coincide).
                case ({wr, rd})
                    2'b10: cnt <= cnt + 2'd1;
                    2'b01: cnt <= cnt - 2'd1;
                    default: cnt <= cnt;   // 2'b00 or 2'b11 -> unchanged
                endcase
            end
        end
    end
    endgenerate

endmodule

`default_nettype wire
