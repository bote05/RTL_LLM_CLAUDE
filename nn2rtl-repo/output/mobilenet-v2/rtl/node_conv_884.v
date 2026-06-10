// node_conv_884 -- MobileNet-v2 depthwise conv 3x3, STRIDE 1, pad 1, C=576,
//   IH=IW=14, OH=OW=14.  PORTED to the conv_812/conv_818 split-architecture
//   (coord_scheduler + rtl_library/line_buf_window.v) so the input window is
//   buffered in SYNC BRAM/URAM (KH=3 rows) instead of the prior full-frame
//   line_buf + byte-granular out_buf.  Output is STREAMED per output pixel.
//
//   WIDE-CHANNEL (C=576) NOTE: the contract bus is 4096b but a packed pixel is
//   C*8 = 4608b, so the contract delivers/consumes TWO 4096b beats per pixel
//   (beat 0 = ch 0..511, beat 1 = ch 512..575 in the low 512b + z-pad).  The
//   proven split-arch core (scheduler + line_buf_window + MP=4 inline depthwise
//   datapath) is UNCHANGED and runs at the PIXEL level on a full 4608b internal
//   window bus; a thin 2-beat input ASSEMBLER and 2-beat output SPLITTER wrap it
//   so the external port stays 4096b/2-beat exactly as the golden expects.  This
//   is the SAME RTL as conv_818 (C=96) for everything past the beat adapters, so
//   line_buf_window inherits the ram_style="ultra" (URAM) channel-along-depth
//   packing on U250 -- the datapath taps window_flat by channel index, never a
//   giant LUT mux.
//
//   Quantization (verbatim from the prior node_conv_884 -- NOT recomputed):
//     SCALE_MULT/2^SCALE_SHIFT = 19655 / 2^20 ~= 0.018744468689
//   Weight ROM layout unchanged: weights[oc*K_TOTAL + k], biases[oc].

`timescale 1ns/1ps
`default_nettype none

// PARAM-GATED ELASTIC BACKPRESSURE (ENABLE_BACKPRESSURE, default 0):
//   This is a 2-BEAT depthwise conv (each output pixel is emitted as a lo beat
//   ch0..511 then a hi beat ch512..575). The skid here is therefore 2-DEEP /
//   DUAL-PHASE so it can hold BOTH beats under out_ready_in backpressure.
//   * ==0 (default): bit/cycle-IDENTICAL to the legacy module. out_ready_in is
//     IGNORED; skid_block is a constant 0 (scheduler/rearm never freeze); the
//     external valid_out/data_out come from the LEGACY 2-state emitter exactly
//     as before. The per-module verify TB (param=0) is byte-exact.
//   * ==1: a 2-entry (dual-phase) output skid captures the datapath's per-pixel
//     pix_out (pix_out_ready pulse) and emits the lo then hi beat, each ONLY when
//     out_ready_in is high. skid_block = bp_busy (a pixel still owes >=1 beat)
//     feeds stall_in + blocks the frame rearm, so the MAC FSM FREEZES while beats
//     are parked => the next pix_out_ready can never overwrite/reorder a buffered
//     pixel. Datapath arithmetic is unchanged; only emit *timing* changes.
// PARAM-GATED NATIVE-256b-TILED RE-ARCHITECTURE (NATIVE_TILED, default 0):
//   * ==0 (default): the LEGACY 4096b / 2-beat external contract, BYTE/CYCLE-
//     IDENTICAL to the prior module. The legacy ports (valid_in/ready_in/data_in
//     [4095:0], out_ready_in, valid_out/data_out[4095:0]) are the EXTERNAL
//     interface and the legacy 2-beat input assembler + 2-beat output splitter
//     wrap the unchanged split-arch core. Any caller that still wants the wide
//     2-beat port (or the per-module verify TB) gets bit-identical behavior.
//   * ==1: the BRIDGELESS NATIVE 256b TILED contract used by the engine top.
//     The retile gather/scatter bridges are deleted; node_conv_884 talks 18x256b
//     tiles/pixel DIRECTLY to its native-tiled producer (relu n4_25) and consumer
//     (relu n4_26). An internal 18-tile gather assembles the full 4608b pixel for
//     line_buf_window (which still re-tiles it for BRAM storage), and an internal
//     18-tile drain emits pix_out as 18 contiguous 256b slices. Byte layout is
//     contiguous (tile k = channels k*32..k*32+31 = wide[k*256+:256]) -> the
//     assembled pixel and the emitted tiles are LOGICAL-PIXEL-IDENTICAL to the old
//     gather->2beat->reassemble / split->scatter round-trip -> BYTE-EXACT.
//     Only the LEGACY 4096b ports are unused (tied off below); the native ports
//     valid_in_t/ready_in_t/data_in_t[255:0] and valid_out_t/out_ready_in_t/
//     data_out_t[255:0] carry the traffic.
module node_conv_884 #(
    parameter ENABLE_BACKPRESSURE = 0,
    parameter NATIVE_TILED = 0,
    parameter WEIGHTS_PATH = "C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_884_weights.hex",
    parameter BIAS_PATH    = "C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_884_bias.hex"
)(
    input  wire           clk,
    input  wire           rst_n,
    // ---- LEGACY 4096b / 2-beat ports (used when NATIVE_TILED==0) ----
    input  wire           valid_in,
    output wire           ready_in,
    input  wire [4095:0]  data_in,
    input  wire           out_ready_in,   // downstream-ready (ignored when ENABLE_BACKPRESSURE==0)
    output reg            valid_out,
    output reg  [4095:0]  data_out,
    // ---- NATIVE 256b TILED ports (used when NATIVE_TILED==1) ----
    input  wire           valid_in_t,
    output wire           ready_in_t,
    input  wire [255:0]   data_in_t,
    input  wire           out_ready_in_t,
    output wire           valid_out_t,
    output wire [255:0]   data_out_t
);

    // skid_block (driven by the BP emitter below) freezes the scheduler + rearm
    // while a buffered pixel still owes >=1 beat under backpressure. With
    // ENABLE_BACKPRESSURE==0 it is a constant 0 -> legacy cycle-identical behavior.
    wire skid_block;

    // In LEGACY mode (NATIVE_TILED==0) the native 256b ports are unused -> tie off
    // deterministically so the module elaborates cleanly regardless of caller.
    generate
    if (NATIVE_TILED == 0) begin : g_native_tieoff
        assign ready_in_t  = 1'b0;
        assign valid_out_t = 1'b0;
        assign data_out_t  = {256{1'b0}};
    end
    endgenerate

    // ----------------- Geometry -----------------
    localparam integer C         = 576;
    localparam integer IH        = 14;
    localparam integer IW        = 14;
    localparam integer OH        = 14;
    localparam integer OW        = 14;
    localparam integer KH        = 3;
    localparam integer KW        = 3;
    localparam integer SH        = 1;
    localparam integer SW        = 1;
    localparam integer PH        = 1;
    localparam integer PW        = 1;
    localparam integer K_TOTAL   = KH * KW;            // 9
    localparam integer MP        = 16;
    localparam integer MP_K      = 9;            // tap-parallel width (= K_TOTAL)
    localparam integer K_GROUPS  = K_TOTAL / MP_K; // = 1 (single-shot reduction)
    localparam integer OC_PASSES = (C + MP - 1) / MP;  // 144

    // ----------------- Bus / beat geometry -----------------
    localparam integer BUS_W   = 4096;                 // external beat width
    localparam integer PIX_W   = C * 8;                // 4608b packed pixel
    localparam integer LO_W    = 512 * 8;              // 4096b = ch 0..511
    localparam integer HI_W    = PIX_W - LO_W;         // (576-512)*8 = 512b

    // ----------------- Quantization (verbatim from prior node) -----------------
    localparam integer SCALE_MULT  = 19655;
    localparam integer SCALE_SHIFT = 20;

    // ----------------- Weight / Bias ROMs -----------------
    // Canonical names + layout identical to the prior node (oc*K_TOTAL + k).
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
        $readmemh("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_884_weights.hex", weights);
        $readmemh("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_884_bias.hex", biases);
        $readmemh("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_884_scale.mem", scale_rom);
    end

    // ====================================================================
    // INPUT ASSEMBLER (param-gated: legacy 2-beat OR native 18-tile gather)
    // ====================================================================
    // Both modes assemble the SAME full C*8 = 4608b packed pixel (pix_data_in)
    // and pulse the scheduler's valid_in (core_valid_in) for exactly ONE cycle,
    // so the split-arch scheduler/line_buf_window/datapath are UNCHANGED and the
    // assembled pixel is bit-identical -> byte-exact across modes.

    wire                sched_ready_in;

    // Signals presented to the split-arch core (scheduler + line_buf_window).
    // BOTH input modes (legacy 2-beat, native 18-tile) drive the SAME core via:
    //   pix_data_in   : the full 4608b packed pixel (assembled internally)
    //   core_valid_in : the ONE-cycle pixel handshake pulse into the scheduler
    // Everything downstream of these two signals is UNCHANGED between modes ->
    // byte-exact by construction (the assembled pixel is bit-identical).
    wire [PIX_W-1:0] pix_data_in;
    wire             core_valid_in;

    generate
    if (NATIVE_TILED == 0) begin : g_in_legacy
        // ================================================================
        // LEGACY INPUT BEAT ASSEMBLER (4096b x 2 beats -> 4608b pixel)
        // ================================================================
        // Bit/cycle-identical to the prior module.  See header.
        reg            in_phase;          // 0 = expecting low beat, 1 = expecting high beat
        reg [LO_W-1:0] lo_latch;

        // [TIMING:BEAT0_READY_GATE] Gate the beat-0 (lo) accept on sched_ready_in
        // so EVERY vector observes the same fill cadence -> uniform per-vector
        // latency. ZERO change to the assembled pixel -> byte-exact preserved.
        wire core_accept_beat0 = (in_phase == 1'b0) && valid_in && sched_ready_in;
        // [BP:BEAT1_READY_GATE] legacy: beat 1 accepted unconditionally; BP: gate
        // on sched_ready_in so core_valid_in never pulses into a frozen scheduler.
        wire beat1_ready = (ENABLE_BACKPRESSURE == 0) ? 1'b1 : sched_ready_in;
        wire core_accept_beat1 = (in_phase == 1'b1) && valid_in && beat1_ready;

        // Full assembled pixel: low half = lo_latch (beat 0), high 512b = beat-1 low.
        assign pix_data_in   = {data_in[HI_W-1:0], lo_latch};
        // Scheduler observes a pixel handshake only on the beat-1 accept cycle.
        assign core_valid_in = (in_phase == 1'b1) && valid_in && beat1_ready;
        // ready_in to the TB.
        assign ready_in = (in_phase == 1'b0) ? sched_ready_in : beat1_ready;

        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                in_phase <= 1'b0;
                lo_latch <= {LO_W{1'b0}};
            end else begin
                if (core_accept_beat0) begin
                    lo_latch <= data_in[LO_W-1:0];
                    in_phase <= 1'b1;
                end else if (core_accept_beat1) begin
                    in_phase <= 1'b0;
                end
            end
        end
    end else begin : g_in_native
        // ================================================================
        // NATIVE INPUT 18-TILE GATHER (18 x 256b tiles -> 4608b pixel)
        // ================================================================
        // The producer relu n4_25 emits 18 contiguous 256b tiles/pixel, one tile
        // per cycle, each held until accepted (it honors out_ready_in == ready_in_t
        // in BP mode -- it advances its emit beat iff out_ready_in, parking the
        // beat otherwise). We gather the 18 tiles into tile_acc[k*256+:256]=tile k.
        // On the 18th accepted tile we form the COMPLETE 4608b pixel
        // (tile_acc with [17*256+:256] = the just-arrived tile) and pulse
        // core_valid_in for EXACTLY one cycle -> bit-identical to the old
        // {data_in[511:0], lo_latch} assembled pixel (tiles 0..15 = ch0..511,
        // tiles 16..17 = ch512..575) and to retile_gather's contiguous packing.
        //
        // BACKPRESSURE: ready_in_t = sched_ready_in for ALL 18 tiles (the
        // scheduler is the only backpressure; stall_in=mac_busy|skid_block|
        // lbw_mem_busy holds sched_ready_in low during the MAC/burst, so the
        // producer correctly stalls). A tile is accepted iff (valid_in_t &
        // ready_in_t) -- the SAME boolean n4_25 sees as out_ready_in_t when it
        // advances -> advance-iff-latch, no lost tile (drain == latch by
        // construction; see retile_bridge.v THE INVARIANT, now enforced WITHOUT
        // a bridge because both ends share the ready_in_t boolean).
        localparam integer N_TILES = 18;       // C/32 = 576/32
        localparam integer TILE_W  = 256;
        reg [PIX_W-1:0]                tile_acc;
        reg [$clog2(N_TILES)-1:0]      in_tile;   // 0..17

        wire tile_ready  = sched_ready_in;
        wire accept_tile = valid_in_t && tile_ready;
        wire last_tile   = (in_tile == N_TILES[$clog2(N_TILES)-1:0] - 1'b1);

        // COMBINATIONAL complete-pixel: previously-gathered tiles 0..16 from
        // tile_acc plus the just-arrived tile 17 in its slot. Presented to the
        // core only on the last-tile accept cycle (core_valid_in pulse).
        wire [PIX_W-1:0] pix_complete;
        assign pix_complete = ({{(PIX_W-(N_TILES-1)*TILE_W){1'b0}},
                                tile_acc[(N_TILES-1)*TILE_W-1:0]})
                              | ({{(PIX_W-TILE_W){1'b0}}, data_in_t} << ((N_TILES-1)*TILE_W));

        assign pix_data_in   = pix_complete;
        assign core_valid_in = accept_tile && last_tile;   // ONE-cycle pulse on tile 17
        assign ready_in_t    = tile_ready;

        // [K1-MBV2] tile_acc is gather DATA: every consumed slice is
        // rewritten during the pixel's N_TILES-tile gather before the
        // last-tile core_valid_in pulse; writes are gated by accept_tile
        // (valid_in_t & sched_ready_in, both reset-held). Sync-only -> FDRE.
        always @(posedge clk) begin
            if (accept_tile) tile_acc[in_tile*TILE_W +: TILE_W] <= data_in_t;
        end
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                in_tile  <= {$clog2(N_TILES){1'b0}};
            end else begin
                if (accept_tile) begin
                    // ([K1-MBV2] tile_acc data write moved to Block A above.)
                    if (last_tile) in_tile <= {$clog2(N_TILES){1'b0}};
                    else           in_tile <= in_tile + 1'b1;
                end
            end
        end
    end
    endgenerate

    // ----------------- Scheduler / window wires -----------------
    wire                              sched_needs_real_input;
    wire                              sched_output_fires;
    wire                              sched_advance;
    wire [$clog2(IH + PH + 1)-1:0]    sched_in_row;
    wire [$clog2(IW + PW + 1)-1:0]    sched_in_col;
    wire [$clog2(OH * OW + 1)-1:0]    sched_outputs_emitted;
    wire                              sched_out_frame_done;
    // Narrow per-channel window from line_buf_window: KH*KW bytes for the single
    // channel selected by channel_select (= current_global_oc). Replaces the wide
    // KH*KW*C*8 window_flat to eliminate the cross-channel-mux routing congestion.
    wire [KH*KW*8-1:0]                chan_window_flat;
    wire                              mac_busy;
    (* max_fanout = 256 *) reg [3:0] lane_counter;
    reg [$clog2(OC_PASSES)-1:0] oc_group;   // OC_PASSES=144 -> 8 bits (0..143)
    wire [$clog2(C)-1:0] current_global_oc = oc_group * MP + lane_counter; // 0..575 -> 10 bits
    wire [15:0]          weight_base_addr  = current_global_oc * K_TOTAL;  // contiguous K_TOTAL taps for this channel

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

    // ----------------- coord_scheduler (universal; pixel-level) -----------------
    coord_scheduler #(
        .IH(IH), .IW(IW), .OH(OH), .OW(OW),
        .KH(KH), .KW(KW), .SH(SH), .SW(SW),
        .PH(PH), .PW(PW)
    ) scheduler (
        .clk(clk), .rst_n(rst_n),
        .start(start_pulse),
        .stall_in(stall_in),
        .valid_in(core_valid_in),
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

    // ----------------- line_buf_window (IC=C=576 packed, SYNC BRAM/URAM) --------
    // Depthwise consumer: leave EXPOSE_FULL_WINDOW at default 0 so the wide
    // cross-channel window_flat mux is NOT instantiated (routing-congestion fix).
    // Drive channel_select with current_global_oc (one channel per cycle) and read
    // the narrow chan_window_flat. window_flat is tied off (.window_flat()).
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
        .valid_in(core_valid_in),
        .data_in(pix_data_in),
        .channel_select(current_global_oc),
        .chan_window_flat(chan_window_flat),
        .window_flat(),
        .mem_busy(lbw_mem_busy)
    );

    // ====================================================================
    // DEPTHWISE DATAPATH (inlined fork of conv_datapath.v, == conv_818)
    // ====================================================================
    // Per-pass cycle count = MP*K_TOTAL + 6 = 4*9 + 6 = 42 cycles.
    // OC_PASSES = 144. Total compute = 144*42 = 6048. Spatial fill =
    // 1*(14+1) + 2 = 17. +1 for the registered output_fires => first
    // pixel-result ready at exactly pipeline_latency_cycles = 6066.

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

    // Pixel result is assembled into pix_out (4608b) as OC passes complete,
    // then streamed out as two 4096b beats by the output splitter below.
    reg [PIX_W-1:0] pix_out;
    reg             pix_out_ready;     // 1-cycle pulse: pix_out holds a fresh pixel


    integer i, lane_i;
    integer bias_oc, out_oc, sc_oc;


    // Tap selector: the per-channel window is now provided by line_buf_window via
    // the narrow chan_window_flat output, with channel_select = current_global_oc
    // (the channel mux moved INTO line_buf_window). chan_window_flat layout is
    //   chan_window_flat[(kh*KW + kw)*8 +: 8]
    // which is bit-identical to the old window_flat byte at
    //   window_flat[((kh*KW + kw)*C + current_global_oc)*8 +: 8].
    // So the tap is a 9-wide index (tap_k_lin = kh*KW + kw, 0..8); no C-way mux,
    // no current_global_oc term here. ZERO arithmetic change.
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
    reg [$clog2(C)-1:0]  mac_global_oc_q1;
    reg                  mac_done_issuing;

    reg                  mac_valid_q2;
    reg [3:0]            mac_lane_q2;
    reg [$clog2(C)-1:0]  mac_global_oc_q2;

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

    // [K1-MBV2] Block A: DATAPATH registers (sync-only, no reset) -- same
    // method as ResNet K1 P2 (apply_k1_fdce_recode.py). prod_q is rewritten
    // every cycle from the (no-reset) weight_q/tap_q stage and only reaches
    // acc under mac_valid_q2 (reset-kept); acc is sync-cleared on ST_IDLE&
    // start_mac / ST_OUTPUT oc-advance BEFORE the first gated accumulate of
    // every pass; biased/scaled/pix_out follow strict write(STn)->read(STn+1)
    // ordering and pix_out is only consumed under reset-kept valid/busy
    // control. acc clears are placed LAST (NBA last-write-wins parity with
    // the original single block). i/lane_i/bias_oc/sc_oc/out_oc/v_tmp
    // are referenced ONLY by this block after the move.
    always @(posedge clk) begin
            for (i = 0; i < MP_K; i = i + 1)
                prod_q[i] <= $signed(weight_q[i]) * $signed(tap_q[i]);
            if (mac_valid_q2 && mac_global_oc_q2 < C[$clog2(C)-1:0]) begin
                acc[mac_lane_q2] <= acc[mac_lane_q2] + $signed(sum_comb);
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
                            pix_out[out_oc*8 +: 8] <=
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
            pix_out_ready    <= 1'b0;
            lane_counter     <= 4'd0;
            oc_group         <= {$clog2(OC_PASSES){1'b0}};
            mac_valid_q1     <= 1'b0;
            mac_lane_q1      <= 4'd0;
            mac_global_oc_q1 <= {$clog2(C){1'b0}};
            mac_valid_q2     <= 1'b0;
            mac_lane_q2      <= 4'd0;
            mac_global_oc_q2 <= {$clog2(C){1'b0}};
            mac_done_issuing <= 1'b0;
        end else begin
            pix_out_ready <= 1'b0;

            mac_valid_q2     <= mac_valid_q1;
            mac_lane_q2      <= mac_lane_q1;
            mac_global_oc_q2 <= mac_global_oc_q1;

            case (state)
                ST_IDLE: begin
                    if (start_mac) begin
                        state            <= ST_MAC;
                        lane_counter     <= 4'd0;
                        oc_group         <= {$clog2(OC_PASSES){1'b0}};
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
                        mac_lane_q1      <= lane_counter;
                        mac_global_oc_q1 <= current_global_oc;
                        mac_valid_q1     <= 1'b1;

                        if (lane_counter == 4'd15) begin
                            lane_counter     <= 4'd0;
                            mac_done_issuing <= 1'b1;
                        end else begin
                            lane_counter <= lane_counter + 4'd1;
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
                    // [K1-MBV2] pix_out[]/v_tmp writes moved to Block A (sync-only).
                    if (oc_group == OC_PASSES - 1) begin
                        // Whole pixel complete: signal the output splitter.
                        pix_out_ready <= 1'b1;
                        state         <= ST_IDLE;
                    end else begin
                        oc_group     <= oc_group + 1'b1;
                        lane_counter <= 4'd0;
                        state <= ST_MAC;
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

    // ====================================================================
    // OUTPUT EMITTER (param-gated: legacy 2-beat splitter OR native 18-tile drain)
    // ====================================================================
    // The datapath assembles pix_out[4607:0] channel-by-channel over 144 OC
    // passes and pulses pix_out_ready once per pixel. The MAC per-pixel cadence
    // (42*144 = 6048 cycles) FAR exceeds the I/O drain length (2 beats legacy /
    // 18 tiles native), so the emitter always drains a pixel before the next one
    // completes (skid_block gates the rearm so it can never overwrite a draining
    // pixel). Both modes emit the SAME bytes in the SAME channel order:
    //   legacy  : beat0=pix_out[4095:0] (ch0..511), beat1={z, pix_out[4607:4096]}
    //   native  : tile k = pix_out[k*256+:256] (ch k*32..k*32+31), k=0..17
    // -> byte-exact (tiles 0..15 == lo beat, tiles 16..17 == hi beat low bits).
    //
    // [INVARIANT:VALID_OUT_LATENCY] The first beat/tile fires the cycle after the
    // last OC pass's ST_OUTPUT (pix_out_ready) -- the same edge the single-beat
    // reference asserts valid -- preserving the scheduler-to-output latency.

    reg [PIX_W-1:0] em_buf;
    reg             em_busy;
    reg             em_phase;          // 0 = drive beat 0 next, 1 = drive beat 1 next

    generate
    if (NATIVE_TILED == 1) begin : g_emit_native
        // ---- NATIVE 18-TILE OUTPUT DRAIN (pix_out[4607:0] -> 18 x 256b tiles) ----
        // On pix_out_ready latch the whole pixel and start an 18-tile drain. Tile k
        // = out_lat[k*256+:256] (channels k*32..k*32+31). valid_out_t is
        // COMBINATIONAL on out_busy (a held tile asserts valid continuously until
        // accepted = true ready/valid). Advance out_tile / clear out_busy ONLY on
        // (valid_out_t & out_ready_in_t) -- the SAME boolean n4_26 uses to latch
        // (n4_26.valid_in = node_conv_884_valid_out & n4_26.ready_in) -> advance ==
        // latch, no lost/duplicate tile (drain == latch by construction).
        //
        // skid_block = out_busy => the MAC FSM/rearm FREEZE while any of the 18
        // tiles is outstanding, so pix_out_ready can NEVER fire while out_busy is
        // set (no overwrite / reorder) -- IDENTICAL invariant to the legacy 2-beat
        // skid, just 18-deep counted instead of dual-phase.
        localparam integer ON_TILES = 18;      // C/32 = 576/32
        localparam integer OTILE_W  = 256;
        reg [PIX_W-1:0]            out_lat;     // latched pix_out being drained
        reg [$clog2(ON_TILES)-1:0] out_tile;    // 0..17
        reg                        out_busy;

        assign skid_block   = out_busy;
        assign valid_out_t  = out_busy;
        assign data_out_t   = out_lat[out_tile*OTILE_W +: OTILE_W];
        wire last_out_tile  = (out_tile == ON_TILES[$clog2(ON_TILES)-1:0] - 1'b1);

        // Legacy wide ports are unused in native mode -> hold at reset value
        // (never re-driven elsewhere in this branch) so they elaborate cleanly.
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                valid_out <= 1'b0;
                data_out  <= {BUS_W{1'b0}};
            end
        end

        // [K1-MBV2] out_lat is drain DATA: latched whole-pixel under
        // pix_out_ready (reset-kept pulse; skid_block guarantees !out_busy)
        // and consumed (data_out_t) only while out_busy (reset-kept).
        always @(posedge clk) begin
            if (pix_out_ready) out_lat <= pix_out;
        end
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                out_tile <= {$clog2(ON_TILES){1'b0}};
                out_busy <= 1'b0;
            end else begin
                if (pix_out_ready) begin
                    // ([K1-MBV2] out_lat data write moved to Block A above;
                    // skid_block guarantees !out_busy here.)
                    out_tile <= {$clog2(ON_TILES){1'b0}};
                    out_busy <= 1'b1;
                end else if (out_busy && out_ready_in_t) begin
                    // Current tile accepted -> advance; on the 18th tile the pixel
                    // is fully drained.
                    if (last_out_tile) begin
                        out_busy <= 1'b0;
                        out_tile <= {$clog2(ON_TILES){1'b0}};
                    end else begin
                        out_tile <= out_tile + 1'b1;
                    end
                end
            end
        end
    end else if (ENABLE_BACKPRESSURE == 0) begin : g_emit_legacy
        // ---- LEGACY 2-state emitter: cycle-identical to the original module. ----
        // out_ready_in is IGNORED; skid_block is a constant 0 (below).
        assign skid_block = 1'b0;
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                valid_out <= 1'b0;
                data_out  <= {BUS_W{1'b0}};
                em_buf    <= {PIX_W{1'b0}};
                em_busy   <= 1'b0;
                em_phase  <= 1'b0;
            end else begin
                valid_out <= 1'b0;
                if (pix_out_ready) begin
                    // Beat 0 right now; latch the pixel for beat 1 next cycle.
                    em_buf    <= pix_out;
                    valid_out <= 1'b1;
                    data_out  <= pix_out[BUS_W-1:0];
                    em_busy   <= 1'b1;
                    em_phase  <= 1'b1;
                end else if (em_busy && em_phase) begin
                    // Beat 1.
                    valid_out <= 1'b1;
                    data_out  <= {{(BUS_W - HI_W){1'b0}}, em_buf[PIX_W-1:LO_W]};
                    em_busy   <= 1'b0;
                    em_phase  <= 1'b0;
                end
            end
        end
    end else begin : g_emit_bp
        // ---- 2-DEEP (dual-phase) ELASTIC emitter. -----------------------------
        // Holds the WHOLE pixel and emits lo (bp_phase==0) then hi (bp_phase==1),
        // each only when out_ready_in is high. valid_out is COMBINATIONAL on
        // bp_busy so a held beat asserts valid continuously until accepted (true
        // ready/valid). data_out is registered (lo at capture, hi at lo-accept).
        //
        // skid_block = bp_busy => the MAC FSM freezes while a pixel owes >=1 beat,
        // so pix_out_ready can NEVER fire while bp_busy is set (no overwrite /
        // reorder). On the pix_out_ready edge bp_busy is asserted the same cycle
        // the MAC goes IDLE; skid_block then keeps the rearm gated until both
        // beats drain.
        reg bp_busy;
        reg bp_phase;                  // 0 = lo beat outstanding, 1 = hi beat outstanding
        reg [HI_W-1:0] bp_hi;          // latched hi half (ch 512..575)

        assign skid_block = bp_busy;

        // valid_out asserts whenever a beat is outstanding. data_out is the
        // registered lo half (captured at pix_out_ready) or hi half.
        always @(*) begin
            valid_out = bp_busy;
        end

        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                data_out <= {BUS_W{1'b0}};
                bp_busy  <= 1'b0;
                bp_phase <= 1'b0;
                bp_hi    <= {HI_W{1'b0}};
                em_buf   <= {PIX_W{1'b0}};
                em_busy  <= 1'b0;
                em_phase <= 1'b0;
            end else begin
                if (pix_out_ready) begin
                    // Capture the full pixel. (skid_block guarantees !bp_busy here.)
                    data_out <= pix_out[BUS_W-1:0];                 // lo beat ready now
                    bp_hi    <= pix_out[PIX_W-1:LO_W];             // hold hi for phase 1
                    bp_busy  <= 1'b1;
                    bp_phase <= 1'b0;                              // lo outstanding first
                end else if (bp_busy && out_ready_in) begin
                    if (bp_phase == 1'b0) begin
                        // lo accepted -> present hi beat next.
                        data_out <= {{(BUS_W - HI_W){1'b0}}, bp_hi};
                        bp_phase <= 1'b1;
                    end else begin
                        // hi accepted -> pixel fully drained.
                        bp_busy  <= 1'b0;
                        bp_phase <= 1'b0;
                    end
                end
            end
        end
    end
    endgenerate

endmodule

`default_nettype wire
