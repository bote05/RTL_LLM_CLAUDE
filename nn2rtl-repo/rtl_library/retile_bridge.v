`default_nettype none

// ============================================================================
// retile_bridge.v  --  MobileNet-v2 final-stage wave-2 RETILE BRIDGES
//                      (ALWAYS-ACCEPT PING-PONG DOUBLE-BUFFER edition)
//
// The MobileNet-v2 final stage ALTERNATES streaming contracts because the
// channel counts (576 / 960 / 1280) exceed the 4096-bit flat-bus cap:
//
//   * POINTWISE convs + RELUs  -> tiled-streaming  (256b bus, channel_tile=32)
//   * DEPTHWISE convs          -> depthwise-conv    (4096b bus, 2 beats/pixel)
//   * residual ADDs / GAP/mean -> flat-bus          (full channel pack/beat)
//
// Two adjacent modules whose contracts differ cannot wire directly: a tiled
// producer emits N 256-bit beats per pixel, while a full-width consumer wants
// one (or two) wide beats per pixel (and vice-versa).  These two bridges
// perform the WIDTH/RATE adaptation -- a pure gather/scatter of bytes.  They
// do NOT reorder channels.
//
// ============================================================================
// WHY PING-PONG  (the deadlock root-cause and its fix)
// ============================================================================
// The PRIOR single-buffer FSM (ST_GATHER with ready_out=1, ST_EMIT with
// ready_out=0) DEADLOCKED.  The MobileNet spatial/depthwise producers in this
// design DO NOT sample ready_out -- they FREE-RUN their output:
//
//   * relu (n4_*) modules: valid_out free-runs for 18/30 beats per pixel and
//     hold their *input* ready_in LOW during that send window (they do not look
//     at any downstream ready).
//   * depthwise convs: emit beat0(out_lo) then beat1(out_hi) on consecutive
//     cycles purely off cyc_cnt / pix_done -- they never look at a ready_down.
//
// A free-running producer that meets ready_out=0 (the old ST_EMIT phase)
// SILENTLY DROPS its beat, and the resulting beat-misalignment cascades into a
// chain deadlock.  Verified facts driving this rewrite:
//   (a) MobileNet modules FREE-RUN: ready_in is an OUTPUT status flag, not a
//       term sampled by upstream producers.
//   (b) spatial_run gates NEW valid_in ONLY; it does NOT freeze in-flight
//       pipelines (a depthwise mid-frame keeps emitting).
//   (c) depthwise emitters free-run their OUTPUT on cyc_cnt.
//
// FIX: a TWO-BUFFER PING-PONG.  ready_out = !(full0 & full1) -- the bridge can
// ALWAYS accept into whichever write-buffer is free WHILE the other buffer
// drains.  The intake is wired with the producer's RAW valid_out (NO
// spatial_run): the producers free-run and never re-present a dropped beat, so
// gating intake by spatial_run would itself silently lose a beat whenever
// spatial_run dropped for ANY other bridge.  The bridge already only WRITES
// when the selected write buffer is free (do_write = valid_in & wsel_empty), so
// RAW valid_in is genuinely always-accept and never drops a free-buffer beat.
// Backpressure is the per-bridge stall throttling the PRODUCER's OWN new-pixel
// start (the producer instance keeps "& spatial_run" on its valid_in) plus the
// one-pixel ping-pong slack so the bridge does not fill mid-send.
//
// ----------------------------------------------------------------------------
// THE INVARIANT  (drain advances  <=>  consumer latches, same cycle, same gate)
// ----------------------------------------------------------------------------
// A bridge's read buffer MUST advance (free full{0,1}, flip rsel/e_idx) on
// EXACTLY the cycle its single downstream consumer LATCHES the beat -- never a
// cycle earlier (would free a beat the frozen consumer ignored = SILENT LOST
// BEAT = byte misalignment = deadlock) and never a cycle later (would re-drive a
// beat the consumer already took = duplicate).  Both sides are therefore gated
// by the IDENTICAL per-bridge signal supplied as drain_en:
//
//     drain xfer       = valid_out & ready_down & drain_en
//     consumer latches = bridge_valid_out & ready_down & drain_en
//
// where the wrapper passes  drain_en = spatial_run_drain_<this_bridge>  AND
// gates this bridge's consumer's valid_in spatial term with the SAME
// spatial_run_drain_<this_bridge>.  ready_down is the consumer's RAW accept
// term (its registered ready_in, + skip_valid for adds) -- an independent status
// flag, NEVER a function of any bridge stall.
//
// ----------------------------------------------------------------------------
// WHY drain_en IS PER-BRIDGE (the deadlock that this revision fixes)
// ----------------------------------------------------------------------------
// stall_out feeds the wrapper's spatial_throttle via any_retile_stall, so when
// THIS bridge raises stall_out it drives the GLOBAL spatial_run LOW.  The naive
// fixes are mutually exclusive for >1 bridge:
//   * If drain_en INCLUDED any_retile_stall, this bridge's OWN stall would
//     freeze its OWN drain -> buffers never empty -> stall latches (SELF-FREEZE
//     DEADLOCK).
//   * If drain_en EXCLUDED any_retile_stall GLOBALLY (one shared spatial_run_drain
//     = ~(engine_busy|sched_spatial_stall)) but the consumer's valid_in still
//     used the GLOBAL spatial_run (which includes any_retile_stall), then while
//     ANY OTHER bridge X is full, THIS bridge's drain fires (drain_en=1) while
//     THIS bridge's consumer does NOT latch (valid_in has spatial_run=0) ->
//     SILENT LOST BEAT at every concurrently-draining bridge -> deadlock.
// Resolution: drain_en (and the matching consumer valid_in gate) is the global
// spatial_run with ONLY THIS bridge's OWN stall term removed:
//     spatial_run_drain_i = ~(engine_busy | sched_spatial_stall
//                             | (any_retile_stall & ~this_bridge_stall_out))
// Then: this bridge's own full never blocks its own drain (no self-freeze), and
// ANY OTHER bridge's full freezes this bridge's drain AND its consumer's latch
// TOGETHER (no lost beat).  See apply_mbv2_wave2_bridges.py for the wiring.
//
// -------------------- BYTE LAYOUT (the load-bearing invariant) --------------
// Both sides pack channels CONTIGUOUSLY, INT8, LSB = lowest channel:
//
//   tiled  : beat k carries channels [k*32 .. k*32+31];  byte i of beat k
//            (data[i*8 +: 8]) is channel (k*32 + i).
//   full   : data[c*8 +: 8] is channel c, for the whole packed beat.
//
// Therefore concatenating the tiled beats in tile order produces EXACTLY the
// contiguous full-width packing:  full[ k*256 +: 256 ] == tiled_beat_k.
// Gather = concatenate tiles in order; Scatter = slice the wide word in
// 256-bit chunks in order.  No permutation anywhere.  This is PIXEL-TILED
// (tiles-outer); channel c lives at bits c*8 +: 8 of the gathered word.
//
// The depthwise 4096b 2-beat case: gather emits beat0 = wide[0 +: 4096]
// (channels 0..511) then beat1 = wide[4096 +: 4096] which, because the real
// word is only FULL_W bits, carries the HI channels (512..) in its LOW bits and
// zero-extends above -- exactly what the depthwise module reads as
// data_in[HI_W-1:0] on its odd beat.  Scatter reverses it: input beat j lands
// at wide[j*IN_W +: min(IN_W, FULL_W-j*IN_W)], so the hi-channel beat's low
// bits restore channels 512.. .  lo-then-hi ordering is preserved end to end.
// ============================================================================


// ----------------------------------------------------------------------------
// retile_gather  --  tiled-streaming  ->  full-width   (PING-PONG)
//
//   Gather phase : write N_TILES input beats of TILE_W bits into the active
//                  WRITE buffer, then flip wsel and mark that buffer full.
//   Emit  phase  : drain the READ buffer for OUT_BEATS beats of OUT_W bits,
//                  chunk j = wide[j*OUT_W +: OUT_W] (zero-extended past FULL_W
//                  on the final partial chunk), under (valid_out & ready_down);
//                  on the last beat clear that buffer's full flag and flip rsel.
//
//   ready_out = !(full0 & full1)   (accept whenever a write buffer is free)
//   valid_out = read buffer full
//   stall_out = full0 & full1      (ORs into spatial_throttle upstream)
//
//   Typical instantiations:
//     - tiled -> flat-bus add LHS / mean : OUT_W = FULL_W, OUT_BEATS = 1.
//     - tiled -> depthwise conv          : OUT_W = 4096, OUT_BEATS = 2.
// ----------------------------------------------------------------------------
module retile_gather #(
    parameter integer TILE_W    = 256,   // tiled beat width (32ch * 8b)
    parameter integer N_TILES   = 18,    // tiled beats per pixel (= C/32)
    parameter integer OUT_W     = 4096,  // consumer (full) bus width
    parameter integer OUT_BEATS = 2,     // beats the consumer reads per pixel
    parameter integer SPATIAL   = 196,   // pixels per frame (documents contract)
    // SYNTH_FIXED_MUX: 0 (default) = original variable barrel-shift (bit-identical,
    // legacy callers unaffected).  1 = a FIXED mux of constant part-selects with a
    // partial-last-beat CLAMP (see emit block below) -- BYTE-EXACT to the shift's
    // zero-fill, and synthesizes with NO out-of-range part-select (avoids the
    // [Synth 8-524] part-select-out-of-range that a naive fixed mux triggers when
    // FULL_W is not a clean multiple of OUT_W, i.e. the depthwise partial last beat).
    parameter integer SYNTH_FIXED_MUX = 0
) (
    input  wire                 clk,
    input  wire                 rst_n,
    // upstream (tiled producer) side
    input  wire                 valid_in,
    output wire                 ready_out,   // toward producer
    input  wire [TILE_W-1:0]    data_in,
    // downstream (full-width consumer) side
    output wire                 valid_out,
    input  wire                 ready_down,  // consumer's raw ready_in (NO spatial_run)
    // drain_en = the EXACT per-bridge gate (spatial_run_drain_<this_bridge>) that
    // ALSO gates this bridge's consumer's valid_in spatial term, so drain and
    // latch are bit-identical.  It is the global spatial_run with ONLY this
    // bridge's OWN stall removed.  See THE INVARIANT block above.
    input  wire                 drain_en,
    output wire [OUT_W-1:0]     data_out,
    // backpressure status toward the wrapper's spatial_throttle
    output wire                 wr_accept,   // per-beat intake accept (= wsel_empty)
    output wire                 stall_out
);
    localparam integer FULL_W   = N_TILES * TILE_W;
    localparam integer GIDX_W   = (N_TILES   > 1) ? $clog2(N_TILES)   : 1;
    localparam integer EIDX_W   = (OUT_BEATS > 1) ? $clog2(OUT_BEATS) : 1;

    // Two ping-pong buffers.  buf[ k*TILE_W +: TILE_W ] holds tile k.
    reg [FULL_W-1:0] buf0, buf1;
    reg              full0, full1;   // buffer holds a complete pixel ready to drain
    reg              wsel;           // write-buffer select (gather target)
    reg              rsel;           // read-buffer  select (emit source)
    reg [GIDX_W-1:0] g_idx;          // tile write index into the WRITE buffer
    reg [EIDX_W-1:0] e_idx;          // chunk read  index into the READ  buffer

    // ALWAYS-ACCEPT: a write buffer is free unless BOTH are full.
    wire write_free = ~(full0 & full1);
    assign ready_out = write_free;
    assign stall_out = (full0 & full1);

    // The write buffer is the one selected by wsel; it is acceptable to write
    // only if THAT buffer is currently empty.  (When wsel's buffer is full but
    // the other is draining, write_free is still 0 for wsel -> we wait one beat
    // for the flip; in practice the producer is throttled by stall_out first.)
    wire wsel_empty = wsel ? ~full1 : ~full0;
    assign wr_accept = wsel_empty;
    wire do_write   = valid_in & wsel_empty;

    // EMIT: drain the read buffer (the one selected by rsel) when it is full.
    wire rsel_full  = rsel ? full1 : full0;
    assign valid_out = rsel_full;

    wire [FULL_W-1:0] rbuf = rsel ? buf1 : buf0;

    // Emit chunk selector: extract OUT_W bits starting at e_idx*OUT_W, padding
    // with zeros for any bits beyond FULL_W (final partial beat -- the depthwise
    // hi-channel beat).  CRITICAL: hi channels land in the LOW bits of beat 1.
    wire [OUT_W-1:0] emit_chunk;
    generate
    if (SYNTH_FIXED_MUX == 0) begin : g_emit_shift
        // [SIM-SPEED 2026-06-03] BIT-IDENTICAL to the prior per-bit loop
        //   for b in 0..OUT_W-1: emit_chunk[b] = (e_idx*OUT_W+b < FULL_W) ? rbuf[e_idx*OUT_W+b] : 0
        // i.e. select the OUT_W-bit chunk e_idx from rbuf, zero-filling past FULL_W.
        // A wide logical shift gives EXACTLY that (>> zero-fills the high bits; the
        // OUT_W-wide LHS truncates), and synthesizes to the same chunk-select mux,
        // but Verilator evaluates ONE wide op instead of OUT_W (up to 10240) bit-ops
        // every cycle -> ~order-of-magnitude faster sim, identical hardware.
        reg [OUT_W-1:0] emit_chunk_r;
        always @(*) emit_chunk_r = rbuf >> (e_idx * OUT_W);
        assign emit_chunk = emit_chunk_r;
    end else begin : g_emit_mux
        // FIXED MUX with PARTIAL-LAST-BEAT CLAMP.  One constant part-select arm per
        // emit beat j in [0, OUT_BEATS).  Arm j reads OUT_W bits at offset j*OUT_W,
        // EXCEPT when j*OUT_W + OUT_W > FULL_W (the partial last beat): then it reads
        // only the (FULL_W - j*OUT_W) in-range bits and zero-pads the top, EXACTLY as
        // the >> shift zero-fills past FULL_W.  No arm's range ever exceeds FULL_W
        // (the declared rbuf width), so NO [Synth 8-524] out-of-range part-select.
        wire [OUT_W-1:0] arm [0:OUT_BEATS-1];
        genvar j;
        for (j = 0; j < OUT_BEATS; j = j + 1) begin : g_arm
            localparam integer OFF   = j * OUT_W;
            // CHUNK = in-range bits available at this offset (clamped to >= 0).
            localparam integer CHUNK = (OFF >= FULL_W) ? 0 :
                                       ((OFF + OUT_W > FULL_W) ? (FULL_W - OFF) : OUT_W);
            localparam integer PAD   = OUT_W - CHUNK;
            if (CHUNK <= 0) begin : g_zero
                assign arm[j] = {OUT_W{1'b0}};
            end else if (PAD == 0) begin : g_full
                assign arm[j] = rbuf[OFF +: CHUNK];
            end else begin : g_clamp
                assign arm[j] = { {PAD{1'b0}}, rbuf[OFF +: CHUNK] };
            end
        end
        assign emit_chunk = arm[e_idx];
    end
    endgenerate
    assign data_out = emit_chunk;

    // DRAIN xfer.
    //   xfer = valid_out & ready_down & drain_en
    // ready_down is the consumer's RAW ready_in (and add skip/operand terms),
    // NEVER spatial_run.  drain_en is the PER-BRIDGE gate spatial_run_drain_i
    // that the wrapper ALSO puts on this bridge's consumer's valid_in spatial
    // term, so the two booleans are identical:
    //     drain xfer       = valid_out       & ready_down & drain_en
    //     consumer latches = bridge_valid_out & ready_down & drain_en
    // The bridge ADVANCES its buffer (frees full{0,1}, flips rsel/e_idx) ONLY on
    // a real xfer, which therefore coincides bit-for-bit with the consumer latch.
    // See THE INVARIANT and WHY drain_en IS PER-BRIDGE in the file header.  No
    // self-freeze: spatial_run_drain_i masks out this bridge's OWN stall, so a
    // full buffer keeps draining and self-clears.  No lost beat: any OTHER
    // bridge's stall freezes drain and latch together via the same drain_en.
    wire xfer = valid_out & ready_down & drain_en;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            buf0 <= {FULL_W{1'b0}};
            buf1 <= {FULL_W{1'b0}};
            full0 <= 1'b0; full1 <= 1'b0;
            wsel  <= 1'b0; rsel  <= 1'b0;
            g_idx <= {GIDX_W{1'b0}};
            e_idx <= {EIDX_W{1'b0}};
        end else begin
            // ---- GATHER / write side (always-accept into the free buffer) ----
            if (do_write) begin
                if (wsel == 1'b0) buf0[g_idx*TILE_W +: TILE_W] <= data_in;
                else              buf1[g_idx*TILE_W +: TILE_W] <= data_in;
                if (g_idx == N_TILES[GIDX_W-1:0] - 1'b1) begin
                    g_idx <= {GIDX_W{1'b0}};
                    if (wsel == 1'b0) full0 <= 1'b1; else full1 <= 1'b1;
                    wsel  <= ~wsel;                 // flip to the other buffer
                end else begin
                    g_idx <= g_idx + 1'b1;
                end
            end

            // ---- EMIT / drain side (keeps draining even under stall_out) ----
            if (xfer) begin
                if (e_idx == OUT_BEATS[EIDX_W-1:0] - 1'b1) begin
                    e_idx <= {EIDX_W{1'b0}};
                    if (rsel == 1'b0) full0 <= 1'b0; else full1 <= 1'b0;
                    rsel  <= ~rsel;                 // flip to the other buffer
                end else begin
                    e_idx <= e_idx + 1'b1;
                end
            end
        end
    end
endmodule


// ----------------------------------------------------------------------------
// retile_scatter  --  full-width  ->  tiled-streaming   (PING-PONG)
//
//   Gather phase : write IN_BEATS input beats of IN_W bits into the active
//                  WRITE buffer (beat j at wide[j*IN_W +: min(IN_W,FULL_W-j*IN_W)]),
//                  then flip wsel and mark that buffer full.
//   Emit  phase  : drain the READ buffer for N_TILES tiled beats of TILE_W bits,
//                  beat k = wide[k*TILE_W +: TILE_W], under (valid_out &
//                  ready_down); on the last beat clear full and flip rsel.
//
//   ready_out = !(full0 & full1);  valid_out = read buffer full;
//   stall_out = full0 & full1.
//
//   Typical instantiations:
//     - flat-bus add/mean -> tiled : IN_W = FULL_W, IN_BEATS = 1.
//     - depthwise conv    -> tiled : IN_W = 4096,   IN_BEATS = 2.
//       (beat 1's real channels live in its LOW (FULL_W-4096) bits.)
// ----------------------------------------------------------------------------
module retile_scatter #(
    parameter integer TILE_W   = 256,    // tiled beat width (32ch * 8b)
    parameter integer N_TILES  = 18,     // tiled beats per pixel (= C/32)
    parameter integer IN_W     = 4096,   // producer (full) bus width
    parameter integer IN_BEATS = 2,      // beats the producer emits per pixel
    parameter integer SPATIAL  = 196,    // pixels per frame (documentation)
    // SYNTH_FIXED_MUX: 0 (default) = original variable barrel ops (bit-identical).
    // 1 = FIXED muxes of constant part-selects with partial-last-beat CLAMP on the
    // insert side (the depthwise hi-channel beat where g_idx*IN_W+IN_W > FULL_W).
    // Byte-exact to the variable ops; no out-of-range part-select (no Synth 8-524).
    parameter integer SYNTH_FIXED_MUX = 0
) (
    input  wire                 clk,
    input  wire                 rst_n,
    // upstream (full-width producer) side
    input  wire                 valid_in,
    output wire                 ready_out,   // toward producer
    input  wire [IN_W-1:0]      data_in,
    // downstream (tiled consumer) side
    output wire                 valid_out,
    input  wire                 ready_down,  // consumer's raw ready_in (NO spatial_run)
    // drain_en = the per-bridge gate spatial_run_drain_<this_bridge> that ALSO
    // gates this bridge's consumer's valid_in spatial term, so drain and latch
    // are bit-identical.  See THE INVARIANT / WHY drain_en IS PER-BRIDGE in the
    // file header -- identical reasoning to retile_gather.
    input  wire                 drain_en,
    output wire [TILE_W-1:0]    data_out,
    // backpressure status toward the wrapper's spatial_throttle
    output wire                 wr_accept,   // per-beat intake accept (= wsel_empty)
    output wire                 stall_out
);
    localparam integer FULL_W = N_TILES * TILE_W;
    localparam integer GIDX_W = (IN_BEATS > 1) ? $clog2(IN_BEATS) : 1;
    localparam integer EIDX_W = (N_TILES  > 1) ? $clog2(N_TILES)  : 1;

    reg [FULL_W-1:0] buf0, buf1;
    reg              full0, full1;
    reg              wsel;
    reg              rsel;
    reg [GIDX_W-1:0] g_idx;          // beat write index into the WRITE buffer
    reg [EIDX_W-1:0] e_idx;          // tile read  index into the READ  buffer

    wire write_free = ~(full0 & full1);
    assign ready_out = write_free;
    assign stall_out = (full0 & full1);

    wire wsel_empty = wsel ? ~full1 : ~full0;
    assign wr_accept = wsel_empty;
    wire do_write   = valid_in & wsel_empty;

    wire rsel_full  = rsel ? full1 : full0;
    assign valid_out = rsel_full;

    wire [FULL_W-1:0] rbuf = rsel ? buf1 : buf0;

    // EMIT read selector: tile e_idx = rbuf[e_idx*TILE_W +: TILE_W].  Here
    // FULL_W == N_TILES*TILE_W exactly, so every arm is fully in range (no clamp
    // strictly needed), but the fixed-mux path still applies the same clamp guard
    // so the construction is uniform and provably never out-of-range.
    wire [TILE_W-1:0] read_chunk;
    generate
    if (SYNTH_FIXED_MUX == 0) begin : g_read_var
        assign read_chunk = rbuf[e_idx*TILE_W +: TILE_W];
    end else begin : g_read_mux
        wire [TILE_W-1:0] tarm [0:N_TILES-1];
        genvar k;
        for (k = 0; k < N_TILES; k = k + 1) begin : g_tarm
            localparam integer OFF   = k * TILE_W;
            localparam integer CHUNK = (OFF >= FULL_W) ? 0 :
                                       ((OFF + TILE_W > FULL_W) ? (FULL_W - OFF) : TILE_W);
            localparam integer PAD   = TILE_W - CHUNK;
            if (CHUNK <= 0) begin : g_zero
                assign tarm[k] = {TILE_W{1'b0}};
            end else if (PAD == 0) begin : g_full
                assign tarm[k] = rbuf[OFF +: CHUNK];
            end else begin : g_clamp
                assign tarm[k] = { {PAD{1'b0}}, rbuf[OFF +: CHUNK] };
            end
        end
        assign read_chunk = tarm[e_idx];
    end
    endgenerate
    assign data_out = read_chunk;

    // Place input beat g_idx into the active write buffer at offset g_idx*IN_W,
    // keeping only the bits inside FULL_W (the depthwise hi-channel beat is
    // partial: its real channels land in the LOW bits of the upper half).
    wire [FULL_W-1:0] wbuf_cur = wsel ? buf1 : buf0;
    wire [FULL_W-1:0] wbuf_next;
    generate
    if (SYNTH_FIXED_MUX == 0) begin : g_ins_var
        // REFERENCE (sim) path = the per-bit GUARDED insert loop, the canonical
        // documented behavior:
        //   for b in 0..IN_W-1: if (base+b < FULL_W) wbuf_next[base+b] = data_in[b]
        // where base = g_idx*IN_W.  Out-of-range bits are left = wbuf_cur (the
        // depthwise hi-channel partial beat).  NOTE: the prior [SIM-SPEED] form
        //   wbuf_next[g_idx*IN_W +: IN_W] = data_in
        // is NOT a faithful reference for the PARTIAL beat (IN_W*IN_BEATS != FULL_W):
        // a variable indexed part-select WRITE whose base is partly out of range is
        // undefined and (verified) corrupts in-range bits -- the same hazard that
        // makes a naive fixed mux hit [Synth 8-524].  This guarded loop is sim-exact
        // to the documented spec for BOTH exact and partial params, so the
        // SYNTH_FIXED_MUX(1) clamp-mux below is provably byte-exact against it.
        reg [FULL_W-1:0] wbuf_next_r;
        integer ins_base, ins_b;
        always @(*) begin
            wbuf_next_r = wbuf_cur;
            ins_base = g_idx * IN_W;
            for (ins_b = 0; ins_b < IN_W; ins_b = ins_b + 1)
                if (ins_base + ins_b < FULL_W)
                    wbuf_next_r[ins_base + ins_b] = data_in[ins_b];
        end
        assign wbuf_next = wbuf_next_r;
    end else begin : g_ins_mux
        // FIXED MUX with PARTIAL-LAST-BEAT CLAMP.  Candidate ins_arm[j] = wbuf_cur with
        // the in-range CHUNK bits at constant offset j*IN_W overwritten by data_in's low
        // CHUNK bits, where CHUNK = min(IN_W, FULL_W - j*IN_W).  The partial last beat
        // (j*IN_W+IN_W > FULL_W) writes ONLY its in-range bits -- the out-of-range portion
        // of data_in is dropped, EXACTLY as the variable indexed part-select write does.
        // No arm's destination range exceeds FULL_W, so NO [Synth 8-524].
        wire [FULL_W-1:0] ins_arm [0:IN_BEATS-1];
        genvar j;
        for (j = 0; j < IN_BEATS; j = j + 1) begin : g_ins_arm
            localparam integer OFF   = j * IN_W;
            localparam integer CHUNK = (OFF >= FULL_W) ? 0 :
                                       ((OFF + IN_W > FULL_W) ? (FULL_W - OFF) : IN_W);
            if (CHUNK <= 0) begin : g_none
                // entire beat out of range -> buffer unchanged (matches drop semantics)
                assign ins_arm[j] = wbuf_cur;
            end else if (OFF == 0 && CHUNK == FULL_W) begin : g_whole
                assign ins_arm[j] = data_in[CHUNK-1:0];
            end else if (OFF == 0) begin : g_low
                assign ins_arm[j] = { wbuf_cur[FULL_W-1 : CHUNK], data_in[CHUNK-1:0] };
            end else if (OFF + CHUNK == FULL_W) begin : g_high
                assign ins_arm[j] = { data_in[CHUNK-1:0], wbuf_cur[OFF-1:0] };
            end else begin : g_mid
                assign ins_arm[j] = { wbuf_cur[FULL_W-1 : OFF+CHUNK],
                                      data_in[CHUNK-1:0],
                                      wbuf_cur[OFF-1:0] };
            end
        end
        assign wbuf_next = ins_arm[g_idx];
    end
    endgenerate

    // DRAIN xfer.  xfer = valid_out & ready_down & drain_en.  ready_down is the
    // consumer's RAW ready_in (no spatial_run); drain_en is the PER-BRIDGE gate
    // spatial_run_drain_i that ALSO gates this bridge's consumer's valid_in, so
    //     drain xfer       = valid_out       & ready_down & drain_en
    //     consumer latches = bridge_valid_out & ready_down & drain_en
    // are bit-identical -> drain advances exactly when the consumer latches.
    // spatial_run_drain_i excludes ONLY this bridge's own stall: no self-freeze
    // (own full keeps draining), no lost beat (any other bridge's stall freezes
    // drain and latch together).  See THE INVARIANT block in the file header.
    wire xfer = valid_out & ready_down & drain_en;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            buf0 <= {FULL_W{1'b0}};
            buf1 <= {FULL_W{1'b0}};
            full0 <= 1'b0; full1 <= 1'b0;
            wsel  <= 1'b0; rsel  <= 1'b0;
            g_idx <= {GIDX_W{1'b0}};
            e_idx <= {EIDX_W{1'b0}};
        end else begin
            // ---- GATHER / write side (always-accept into the free buffer) ----
            if (do_write) begin
                if (wsel == 1'b0) buf0 <= wbuf_next;
                else              buf1 <= wbuf_next;
                if (g_idx == IN_BEATS[GIDX_W-1:0] - 1'b1) begin
                    g_idx <= {GIDX_W{1'b0}};
                    if (wsel == 1'b0) full0 <= 1'b1; else full1 <= 1'b1;
                    wsel  <= ~wsel;
                end else begin
                    g_idx <= g_idx + 1'b1;
                end
            end

            // ---- EMIT / drain side (keeps draining even under stall_out) ----
            if (xfer) begin
                if (e_idx == N_TILES[EIDX_W-1:0] - 1'b1) begin
                    e_idx <= {EIDX_W{1'b0}};
                    if (rsel == 1'b0) full0 <= 1'b0; else full1 <= 1'b0;
                    rsel  <= ~rsel;
                end else begin
                    e_idx <= e_idx + 1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
