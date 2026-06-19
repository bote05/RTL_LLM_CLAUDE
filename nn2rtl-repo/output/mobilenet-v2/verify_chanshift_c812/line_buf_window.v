// line_buf_window -- KH-row line buffer (BRAM-inferred via per-slot
// single-port memories with a rotating row pointer) + KH x KW x IC
// registered shift-register window.
//
// Replaces the prior flop-based bulk-vertical-shift implementation:
// the legacy "line_buf[i] <= line_buf[i+1]" copy is now a single
// counter increment on `oldest_slot`. Per-slot BRAMs hold the actual
// pixel data; `row_valid[KH-1:0]` masks reads from slots whose data
// is stale relative to the current frame.
//
// Vivado/Artix-7 BRAM inference targets:
//   line_buf : KH separate BRAM18/BRAM36 instances. Each holds
//              IW+PW words of (IC*8) bits with sync read+write at
//              the same address (standard single-port template).
//   window   : KH x KW x IC * 8-bit shift register in fabric flops
//              (small: 9*64*8 = 4.6 kbit for 3x3 conv2 = trivial).
//
// Multi-frame correctness (this is exactly where the prior bug class
// lived, so it is reasoned about explicitly here):
//
//   At frame_start:
//     oldest_slot <= 0
//     row_valid   <= 0     (all KH bits cleared)
//     window      <= 0     (shift reg cleared)
//     bypass_reg  <= 0     (data_in bypass FF cleared)
//
//   The BRAM cells themselves are NOT cleared on frame_start. They
//   retain whatever data was written during the prior frame. The
//   `row_valid` mask is the sole guarantee that top-pad reads (from
//   slots whose row hasn't been written yet this frame) return zero
//   instead of stale frame-N-1 pixels:
//
//       window_kwm1_wire[i] = row_valid[slot_for_rf[i]]
//                              ? q_array[slot_for_rf[i]]
//                              : 0
//
//   row_valid is updated at row_wrap_this_cycle:
//     - the slot that just finished filling becomes "history" and
//       has its bit set
//     - the slot that becomes the new "currently writing" has its
//       bit cleared (it will be overwritten with this frame's
//       next input row, so its prior contents are stale)
//
//   Right-pad columns: writes are gated by `!right_padded`, so the
//   BRAM cell at addr=MAX_IN_COL is never touched. Vivado initialises
//   BRAMs to zero at FPGA configuration, so reads at MAX_IN_COL
//   return zero -- no explicit right_padded mask is needed in the
//   read path.
//
// Latency: this rewrite preserves the legacy 1-cycle read pipeline
// EXACTLY. The legacy design had async-read of line_buf followed by
// a window[i][KW-1] FF (1 cycle). The new design replaces that with
// a per-slot BRAM whose output register q_array[s] IS the window's
// rightmost column (no extra FF stage). compute_conv2d_latency_cycles
// in scripts/golden_impl.py is unchanged.

module line_buf_window #(
    parameter integer IC = 64,
    parameter integer IW = 112,
    parameter integer IH = 112,
    parameter integer KH = 3,
    parameter integer KW = 3,
    parameter integer PW = 1,
    parameter integer PH = 1,
    // When 1, drive the legacy wide `window_flat` output (KH*KW*IC*8 bits).
    // Default 1 = BACKWARD-COMPATIBLE: legacy (cross-channel) consumers that
    // instantiate WITHOUT this param (all ResNet spatial convs: node_conv_196..,
    // and conv_datapath_mp_k) keep driving the full window_flat unchanged.
    // The routing-congestion fix consumers (MobileNet depthwise) EXPLICITLY pass
    // EXPOSE_FULL_WINDOW(0) + read the narrow `chan_window_flat` (one channel per
    // cycle via `channel_select`), so the wide cross-channel mux is NOT built for
    // them. (Flipping default 0->1 prevents a silent break of unmodified ResNet
    // instantiations on a fresh rebuild; mbv2 sets it to 0 explicitly.)
    parameter integer EXPOSE_FULL_WINDOW = 1,

    // [FIT-FIX 2026-06-02] Selects the synthesis ram_style attribute placed on the
    // per-slot line-buffer memories (`mem`). This is a MAPPING-ONLY knob: it only
    // tells Vivado which primitive to infer; the RTL behaviour (values, latency,
    // control) is BIT-IDENTICAL for either setting (Verilator ignores ram_style),
    // and the right-pad read is masked explicitly so correctness does NOT depend
    // on the chosen primitive's power-up state in either case.
    //   1 (DEFAULT) = "ultra"  -> URAM288. BACKWARD-COMPATIBLE: ResNet spatial
    //                 convs (node_conv_196.., conv_datapath_mp_k) instantiate
    //                 WITHOUT this param and keep the intentional URAM mapping
    //                 alongside EXPOSE_FULL_WINDOW=1.
    //   0           = "block"  -> RAMB36. MobileNetV2 depthwise convs pass 0 so
    //                 their shallow-but-very-wide per-slot buffers (e.g. C=960 ->
    //                 8 deep x 7680 bit) reshape into block RAM instead of
    //                 width-binding URAM288 (4096x72 fixed geometry => ~2394
    //                 URAM288 = 187% OVER on U250). The same 2.87 Mbit packs into
    //                 ~78 RAMB36. URAM is reserved for the engine weight banks.
    parameter integer LINE_BUF_USE_URAM = 1,

    // [FIT-FIX 2026-06-06] TILE_STORAGE: 0 (DEFAULT) = legacy shallow-wide per-slot mem
    // (IC*8 wide x MEM_DEPTH deep), BIT/CYCLE-IDENTICAL to the prior design -> ResNet and any
    // caller that omits this param elaborates UNCHANGED. >0 = deep-narrow TILED storage: each
    // slot's (IC*8)-bit word is serialized into NT=ceil(IC/TILE_STORAGE) tiles of (TILE_STORAGE*8)
    // bits, stored MEM_DEPTH*NT deep x (TILE_STORAGE*8) wide so Vivado packs RAMB36 by DEPTH
    // instead of width-binding (e.g. C=960: 4 RAMB36/slot vs 107). The per-slot R/W burst is made
    // ATOMIC by raising mem_busy -> the node ORs it into stall_in (exactly like mac_busy), so the
    // scheduler/window/datapath freeze for the burst => byte-exact by elasticity, downstream logic
    // (window shift / window_kwm1_wire / chan_window_flat / row_valid) UNCHANGED. Cost: +1.5-8% cyc.
    parameter integer TILE_STORAGE = 0,

    // [FIT-FIX 2026-06-07] CHAN_SHIFT: 0 (DEFAULT) = legacy per-tap C:1 channel-select MUX feeds
    // chan_window_flat (window[kh][kw][csel] / window_kwm1_wire[kh][csel] / bypass_reg[csel]).
    // BIT-IDENTICAL to the prior design -> ResNet (which omits this param, EXPOSE_FULL_WINDOW=1,
    // never reads chan_window_flat) and any caller that omits it elaborate UNCHANGED.
    //
    // 1 = ROTATION SHIFT-REGISTER (depthwise only). The C:1 byte mux per tap (the ~450K-LUT block:
    // 9 taps x up to 960:1 x 17 convs = device-dominant F7/F8 muxes) is REPLACED by a per-tap
    // C-deep byte rotation register. The depthwise datapath drives channel_select = current_global_oc
    // which, AT THE ISSUING CYCLES the datapath actually consumes chan_window_flat, is a CLEAN
    // SEQUENTIAL +1 walk 0,1,2,...,C-1 (oc_group*MP + lane_counter; verified in node_conv_896/848).
    // A value picked by a monotonic +1 index is a SHIFT, not a random-access mux. So:
    //   * LOAD the bank in parallel from the (frozen) window/window_kwm1_wire/bypass at
    //     sched_output_fires (one cycle before the datapath's first issue) -- the SAME source bytes
    //     the mux would select, captured for ALL C channels in channel-natural order (ch0 at head).
    //   * ROTATE the bank by 1 each `chan_advance` (the datapath's per-channel issuing strobe), so
    //     the head walks ch0,ch1,...,ch(C-1) exactly in step with current_global_oc at issuing cycles.
    //   * chan_window_flat = the bank HEAD (position 0) -- COMBINATIONAL, same-cycle as the legacy
    //     mux read => ZERO added latency. Bank is FFs => NO BRAM change. Byte-IDENTICAL: head at the
    //     i-th issue == the byte the C:1 mux gave for csel==i (same source, same order).
    // The window storage, horizontal shift, window_kwm1_wire, bypass, row_valid and the BRAM/URAM/
    // tiled per-slot mems are ALL UNCHANGED -- the bank is a purely ADDITIVE shadow on top of them.
    parameter integer CHAN_SHIFT = 0
) (
    input  wire                               clk,
    input  wire                               rst_n,

    // One-cycle pulse wired from top-level start_pulse. Clears
    // the rotating pointer, row_valid bits, window shift register,
    // and the data_in bypass FF. Does NOT clear BRAM cells (they
    // retain prior-frame data; row_valid is the multi-frame
    // safety net).
    input  wire                               frame_start,

    // Scheduler observers.
    input  wire [$clog2(IH + PH + 1)-1:0]     sched_in_row,
    input  wire [$clog2(IW + PW + 1)-1:0]     sched_in_col,
    input  wire                               sched_needs_real_input,
    input  wire                               sched_advance,
    input  wire                               sched_output_fires,

    // Upstream pixel stream.
    input  wire                               valid_in,
    input  wire [IC*8-1:0]                    data_in,

    // Channel selector for the narrow `chan_window_flat` output below.
    // Depthwise datapaths drive this with current_global_oc (one channel
    // per cycle). Width is wide enough to index any of the IC channels.
    // (IC==1 would make $clog2 zero-width, so floor the width at 1.)
    input  wire [((IC > 1) ? $clog2(IC) : 1)-1:0] channel_select,

    // [FIT-FIX 2026-06-07] Per-channel issuing strobe for CHAN_SHIFT==1: pulse HIGH for exactly one
    // cycle each time the datapath CONSUMES one channel's chan_window_flat (the depthwise MAC's
    // `(state==ST_MAC) && !mac_done_issuing` issue cycle). Advances the rotation bank by 1 so its head
    // tracks current_global_oc at the consumed cycles. IGNORED when CHAN_SHIFT==0 (legacy mux); legacy
    // callers leave it unconnected (defaults to 0 -> bank inert even if elaborated).
    input  wire                               chan_advance,

    // Narrow per-channel window for depthwise datapaths. KH*KW bytes, one
    // per receptive-field tap, all for the SINGLE channel `channel_select`:
    //   chan_window_flat[(kh*KW + kw)*8 +: 8]
    //     == window_flat[((kh*KW + kw)*IC + channel_select)*8 +: 8]
    // Read from the SAME three source regions as window_flat (window shift
    // regs / window_kwm1_wire BRAM mux / bypass_reg). ZERO arithmetic change.
    output wire [KH*KW*8-1:0]                 chan_window_flat,

    // Flat packed window for the datapath. Layout matches
    // conv_datapath's tap_at():
    //   window_flat[(kh*KW*IC + kw*IC + ic)*8 +: 8]
    // Only driven when EXPOSE_FULL_WINDOW==1 (see generate at end). When 0
    // it is held at 0 so the wide cross-channel mux is not instantiated.
    output wire [KH*KW*IC*8-1:0]              window_flat,

    // [FIT-FIX 2026-06-06] High during an NT-cycle tiled-storage burst (TILE_STORAGE>0); the node
    // ORs this into stall_in so the burst is atomic. Tied to 0 when TILE_STORAGE==0 (ResNet/legacy
    // callers leave this output unconnected -> harmless).
    output wire                               mem_busy
);

    // ---------------- Derived constants ------------------------------
    localparam integer MAX_IN_COL = IW + PW - 1;
    localparam integer MEM_DEPTH  = IW + PW;
    localparam integer SLOT_W     = (KH > 1) ? $clog2(KH) : 1;
    localparam integer SCHED_COL_W = $clog2(IW + PW + 1);
    localparam integer SCHED_ROW_W = $clog2(IH + PH + 1);
    // Width of channel_select (matches the port-declaration expression).
    localparam integer CSEL_W      = (IC > 1) ? $clog2(IC) : 1;

    // [FIT-FIX 2026-06-06] tiled-storage derived constants (active when TILE_STORAGE>0).
    localparam integer TILE    = (TILE_STORAGE > 0) ? TILE_STORAGE : IC;
    localparam integer NT      = (IC + TILE - 1) / TILE;            // tiles per (IC*8) word
    localparam integer TILE_W  = TILE * 8;
    localparam integer NT_W    = (NT > 1) ? $clog2(NT) : 1;
    localparam integer TADDR_W = $clog2(MEM_DEPTH * NT);            // mem_t address width

    // KH is a 32-bit `parameter integer`; we perform pointer arithmetic
    // and modulo-KH reductions in (SLOT_W+1)-bit space. The width
    // truncation / expansion in the conditionals below is intentional
    // -- KH always fits in SLOT_W+1 bits by definition of SLOT_W. Wrap
    // the affected expressions in lint_off pragmas to silence Verilator's
    // strict-lint warnings without disabling them globally.
    /* verilator lint_off WIDTH */

    // ---------------- Edge classifiers + handshake -------------------
    wire right_padded  = (sched_in_col >= IW[SCHED_COL_W-1:0]);
    wire bottom_padded = (sched_in_row >= IH[SCHED_ROW_W-1:0]);
    wire handshake_real =
        sched_advance && sched_needs_real_input && valid_in;
    wire row_wrap_this_cycle =
        sched_advance && (sched_in_col == MAX_IN_COL[SCHED_COL_W-1:0]);

    // ---------------- Rotating slot pointer + row_valid --------------
    reg [SLOT_W-1:0] oldest_slot;
    reg [KH-1:0]     row_valid;

    // current_write_slot = (oldest_slot + KH - 1) mod KH.
    // For non-power-of-2 KH (e.g. 3 or 7) we need explicit mod logic.
    wire [SLOT_W:0]   cws_sum    = {1'b0, oldest_slot} + (KH - 1);
    wire [SLOT_W-1:0] current_write_slot =
        (cws_sum >= KH) ? (cws_sum - KH) : cws_sum[SLOT_W-1:0];

    // After row_wrap the new "currently writing" slot is the one
    // that was previously the oldest (= oldest_slot pre-increment).
    // That slot's row_valid must be cleared because its prior
    // contents will be overwritten by the new input row's writes.

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            oldest_slot <= {SLOT_W{1'b0}};
            row_valid   <= {KH{1'b0}};
        end else if (frame_start) begin
            oldest_slot <= {SLOT_W{1'b0}};
            row_valid   <= {KH{1'b0}};
        end else if (row_wrap_this_cycle) begin
            oldest_slot <= (oldest_slot == (KH - 1)) ?
                            {SLOT_W{1'b0}} : (oldest_slot + 1'b1);
            // The slot that just finished filling is only "valid history"
            // if the row that just ended was a REAL input row. Bottom-pad
            // rows (sched_in_row >= IH) emit row_wrap events too -- the
            // scheduler walks all in_row positions including the bottom-
            // pad fringe -- but no writes have actually landed in the
            // current_write_slot during that row. Marking the slot valid
            // would cause subsequent reads to return stale data from KH
            // rows ago (the previous occupant), breaking bottom-pad
            // correctness. Gate on !bottom_padded so bottom-pad row_wraps
            // leave row_valid[current_write_slot] cleared.
            row_valid[current_write_slot] <= !bottom_padded;
            row_valid[oldest_slot]        <= 1'b0;
        end
    end

    // ---------------- Per-slot BRAM/URAM storage ---------------------
    // KH separate single-port memories. Vivado infers each as a
    // BRAM/URAM instance from the standard "if (we) write; q <=
    // mem[addr];" template + the (* ram_style = ... *) attribute.
    // The address path is unconditional (sched_in_col) so synth
    // sees a clean inference target -- no muxes wrapped around the
    // address.
    //
    // [FIT-FIX 2026-06-02] The ram_style attribute is selected by the
    // LINE_BUF_USE_URAM parameter via a generate-if. The two branches are
    // BIT-IDENTICAL apart from the single (* ram_style *) attribute line:
    //   LINE_BUF_USE_URAM==1 -> "ultra" (URAM288) -- ResNet default, unchanged.
    //   LINE_BUF_USE_URAM==0 -> "block" (RAMB36)  -- MobileNet depthwise; the
    //     shallow-wide buffers reshape into block RAM instead of width-binding
    //     URAM (see the parameter doc above). ram_style is a SYNTHESIS attribute
    //     only -- Verilator ignores it, so simulation output is identical for
    //     either branch (this is what the byte-exact verification confirms).
    //
    // Shared correctness reasoning (holds for BOTH primitives):
    // [FIT-FIX 2026-05-30] mem is NOT content-initialized (URAM cannot be, and we
    // keep the no-init template uniform so the two branches stay bit-identical).
    // Correctness does NOT depend on the mem power-up state: the only read path
    // that previously relied on a never-written cell reading 0 is the RIGHT-PAD
    // column (addr >= IW, e.g. MAX_IN_COL); that is masked explicitly at the read
    // (`right_padded ? 0` below). Top-/bottom-pad + cross-frame stale reads are
    // already masked by `row_valid` (window_kwm1_wire), so no undefined mem cell
    // is ever consumed. q_reg is a fabric reg (not BRAM/URAM) so its zero-init is
    // kept.

    wire [IC*8-1:0] q_array [0:KH-1];

    // [FIT-FIX 2026-06-06] Tiled-storage burst sequencer (shared by all KH slots; they share the
    // sched_advance cadence). On each sched_advance run NT clocks of per-tile R/W; mem_busy stalls
    // the scheduler so the burst is ATOMIC (no WAW at the gap=1 row-fill). TILE_STORAGE==0 or NT<=1
    // => mem_busy const 0, sequencer inert. mem_busy is held high for the full NT cycles after an
    // advance (tcnt 0..NT-1). q_reg is INCREMENTALLY folded one tile per burst cycle (NOT all-at-once
    // at the end): the depthwise MAC consumes channels sequentially, and since the burst writes a
    // whole 32-channel tile every cycle while the MAC drains <1 channel/cycle, each tile is in q_reg
    // long before the MAC reaches its channels -- tile 0 (the first channels) lands on burst-cycle 0
    // (the cycle the MAC enters ST_MAC). q_reg therefore reads exactly read(col@advance) for every
    // channel when consumed, mirroring the legacy 1-advance q_reg latency. VERIFIED byte-exact
    // (mismatch=0) by output/mobilenet-v2/verify_lbw_c960/tb_equiv.sv on node_conv_896 (C=960, NT=30,
    // 2 frames). The equiv-TB is the authority on this timing.
    reg  [NT_W-1:0] tcnt;
    reg             burst_active;
    generate
        if (TILE_STORAGE == 0 || NT <= 1) begin : gen_no_burst
            assign mem_busy = 1'b0;
        end else begin : gen_burst
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n)             begin burst_active <= 1'b0; tcnt <= {NT_W{1'b0}}; end
                else if (frame_start)   begin burst_active <= 1'b0; tcnt <= {NT_W{1'b0}}; end
                else if (sched_advance) begin burst_active <= 1'b1; tcnt <= {NT_W{1'b0}}; end
                else if (burst_active) begin
                    if (tcnt == NT-1) burst_active <= 1'b0;
                    tcnt <= tcnt + 1'b1;
                end
            end
            assign mem_busy = burst_active;   // high for NT cycles after the advance
        end
    endgenerate

    genvar g_slot;
    generate
        for (g_slot = 0; g_slot < KH; g_slot = g_slot + 1) begin : gen_slot
            reg [IC*8-1:0] q_reg;
            initial q_reg = {(IC*8){1'b0}};

            wire is_writing = (current_write_slot == g_slot[SLOT_W-1:0]);
            wire write_en   =
                handshake_real && !right_padded && !bottom_padded && is_writing;

            if (TILE_STORAGE == 0) begin : gen_legacy_storage
            // ===== legacy shallow-wide per-slot mem -- BIT/CYCLE-IDENTICAL to the prior design;
            // ===== ResNet and any caller with TILE_STORAGE==0 (the default) elaborate this branch.
            if (LINE_BUF_USE_URAM != 0) begin : gen_mem_ultra
                (* ram_style = "ultra" *)
                reg [IC*8-1:0] mem [0:MEM_DEPTH-1];

                always @(posedge clk) begin
                    if (write_en) begin
                        mem[sched_in_col] <= data_in;
                    end
                    // q_reg must be FROZEN whenever the scheduler is stalled.
                    // The scheduler asserts output_fires the cycle AFTER it
                    // advances past a firing coordinate, with sched_in_col
                    // already pointing at the NEXT coord. If q_reg free-ran
                    // off live sched_in_col, the rightmost window column would
                    // change mid-MAC -- the MAC's tap_at() reads window_flat
                    // at every k_counter step, so a moving q would corrupt
                    // every k>=1 contribution. Gating on sched_advance keeps
                    // q stable from output_fires through the entire MAC pass
                    // (stall_in = mac_busy holds sched_advance low for that
                    // window, by design of the split-architecture contract).
                    // [FIT-FIX 2026-05-30] right-pad columns (addr >= IW) read 0,
                    // masked explicitly (mem has no zero-init). Byte-exact vs the
                    // prior BRAM-zero-init design (which also yielded 0 here, and
                    // which Verilator --x-initial 0 reproduced).
                    if (sched_advance) begin
                        q_reg <= right_padded ? {(IC*8){1'b0}} : mem[sched_in_col];
                    end
                end
            end else begin : gen_mem_block
                (* ram_style = "block" *)
                reg [IC*8-1:0] mem [0:MEM_DEPTH-1];

                // Identical template to gen_mem_ultra; differs ONLY in the
                // (* ram_style = "block" *) attribute above so the shallow-wide
                // buffers reshape into RAMB36 rather than width-binding URAM.
                always @(posedge clk) begin
                    if (write_en) begin
                        mem[sched_in_col] <= data_in;
                    end
                    if (sched_advance) begin
                        q_reg <= right_padded ? {(IC*8){1'b0}} : mem[sched_in_col];
                    end
                end
            end
            end else begin : gen_tiled_storage
                // [FIT-FIX 2026-06-06] deep-narrow tiled storage: split the (IC*8)-bit column into
                // NT tiles of TILE_W bits, stored MEM_DEPTH*NT deep (depth-packed => ~4 RAMB36/slot
                // vs ~107 width-bound). Write + read-reassembly are serialized over NT cycles under
                // burst_active; the scheduler is stalled (mem_busy) for the burst. q_reg is folded
                // INCREMENTALLY (one tile/cycle) so each tile reaches the channel-sequential MAC just
                // in time -> byte-exact vs the legacy 1-advance q_reg latency (see sequencer comment
                // above). Downstream window/chan_window logic is UNCHANGED.
                (* ram_style = "block" *)
                reg [TILE_W-1:0]       mem_t [0:MEM_DEPTH*NT-1];
                reg  [SCHED_COL_W-1:0] col_l;     // latched column at the advance
                reg                    rpad_l;    // latched right_padded at the advance
                reg  [IC*8-1:0]        wdat_l;    // latched data_in at the advance
                reg                    we_l;      // latched write_en at the advance
                wire [TADDR_W-1:0]     taddr = col_l * NT + tcnt;
                always @(posedge clk) begin
                    if (sched_advance) begin
                        col_l  <= sched_in_col;
                        rpad_l <= right_padded;
                        wdat_l <= data_in;
                        we_l   <= write_en;
                    end else if (burst_active) begin
                        if (we_l) mem_t[taddr] <= wdat_l[tcnt*TILE_W +: TILE_W];   // write tile tcnt
                        // [FIT-FIX 2026-06-06] INCREMENTAL fold: update q_reg's tile-tcnt slice the
                        // cycle that tile is read, so each tile becomes available to the
                        // (channel-sequential) MAC as soon as it is read -- NOT all-at-once at
                        // tcnt==NT-1. The burst writes 1 tile/cyc (32 ch/cyc) while the MAC consumes
                        // <1 ch/cyc, so the burst stays far ahead: tile 0 (ch 0..31) is folded by the
                        // first burst cycle (A+1, visible A+2) exactly when the MAC enters ST_MAC and
                        // samples channel 0; every later tile is folded long before the MAC reaches
                        // its channels. q_reg therefore equals read(col@advance) for every channel by
                        // the time it is consumed -> byte-exact vs the legacy 1-advance q_reg latency.
                        // Read returns the OLD mem_t value at a written address (NBA), mirroring the
                        // legacy read-during-write on the current-write slot (whose q_reg the window
                        // mux never consumes -- it is the bypass row). Right-pad reads 0 (masked).
                        q_reg[tcnt*TILE_W +: TILE_W] <= rpad_l ? {TILE_W{1'b0}} : mem_t[taddr];
                    end
                end
            end

            assign q_array[g_slot] = q_reg;
        end
    endgenerate

    // ---------------- Slot mux for window's rightmost column ----------
    // For each receptive-field row i in 0..KH-2, look up the slot
    // currently holding that row's data and route q_array[slot] to
    // window_kwm1_wire[i], gated by row_valid for top-pad / cross-
    // frame stale data.
    //
    // slot_for_rf[i] = (oldest_slot + i) mod KH.

    wire [IC*8-1:0] window_kwm1_wire [0:KH-2];

    genvar g_rf;
    generate
        for (g_rf = 0; g_rf < KH - 1; g_rf = g_rf + 1) begin : gen_rf_mux
            wire [SLOT_W:0]   sum_g    = {1'b0, oldest_slot} + g_rf[SLOT_W:0];
            wire [SLOT_W-1:0] slot_idx =
                (sum_g >= KH) ? (sum_g - KH) : sum_g[SLOT_W-1:0];

            assign window_kwm1_wire[g_rf] =
                row_valid[slot_idx] ? q_array[slot_idx]
                                    : {(IC*8){1'b0}};
        end
    endgenerate
    /* verilator lint_on WIDTH */

    // ---------------- Window shift register --------------------------
    // window[i][j] for i in 0..KH-1, j in 0..KW-2 are the shift-reg
    // stages. The KW-1 column is reconstructed from window_kwm1_wire
    // (for i < KH-1) or from bypass_reg (for i = KH-1, the row
    // currently being received).
    //
    // The shift gate matches the legacy semantics: shift fires on
    // every scheduler advance EXCEPT when sched_output_fires is high
    // (during MAC pipeline the window must hold the receptive field
    // steady).

    reg [7:0] window [0:KH-1][0:KW-2][0:IC-1];
    reg [7:0] bypass_reg [0:IC-1];

    integer i, j, c_ch;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < KH; i = i + 1)
                for (j = 0; j < KW - 1; j = j + 1)
                    for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1)
                        window[i][j][c_ch] <= 8'sd0;
            for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1)
                bypass_reg[c_ch] <= 8'sd0;
        end else if (frame_start) begin
            for (i = 0; i < KH; i = i + 1)
                for (j = 0; j < KW - 1; j = j + 1)
                    for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1)
                        window[i][j][c_ch] <= 8'sd0;
            for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1)
                bypass_reg[c_ch] <= 8'sd0;
        end else if (sched_advance && !sched_output_fires) begin
            // Shift columns left by one. The new KW-2 column samples
            // the (combinational) KW-1 column wire / bypass_reg.
            for (i = 0; i < KH; i = i + 1) begin
                for (j = 0; j < KW - 2; j = j + 1)
                    for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1)
                        window[i][j][c_ch] <= window[i][j+1][c_ch];
            end

            // Last shift stage -- column KW-2 latches column KW-1.
            // Top KH-1 rows: from BRAM mux. Bottom row: from bypass.
            for (i = 0; i < KH - 1; i = i + 1) begin
                for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1)
                    window[i][KW-2][c_ch] <=
                        $signed(window_kwm1_wire[i][c_ch*8 +: 8]);
            end
            for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1)
                window[KH-1][KW-2][c_ch] <= $signed(bypass_reg[c_ch]);

            // bypass_reg update (data_in for the row currently being
            // received, masked by right_padded/bottom_padded).
            for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1) begin
                if (right_padded || bottom_padded) begin
                    bypass_reg[c_ch] <= 8'sd0;
                end else if (handshake_real) begin
                    bypass_reg[c_ch] <= $signed(data_in[c_ch*8 +: 8]);
                end else begin
                    bypass_reg[c_ch] <= 8'sd0;
                end
            end
        end
    end

    // ---------------- Flatten window output --------------------------
    // Layout matches conv_datapath::tap_at():
    //   window_flat[(kh*KW*IC + kw*IC + ic)*8 +: 8]
    //
    // Columns 0..KW-2 come from the shift-reg flops. Column KW-1
    // is mixed: rows 0..KH-2 come from window_kwm1_wire (combinational
    // BRAM mux); row KH-1 comes from bypass_reg.

    genvar g_kh, g_kw, g_ic;
    generate
        if (EXPOSE_FULL_WINDOW != 0) begin : gen_full_window
            for (g_kh = 0; g_kh < KH; g_kh = g_kh + 1) begin : gen_win_kh
                for (g_kw = 0; g_kw < KW; g_kw = g_kw + 1) begin : gen_win_kw
                    for (g_ic = 0; g_ic < IC; g_ic = g_ic + 1) begin : gen_win_ic
                        if (g_kw < KW - 1) begin : gen_shift_col
                            assign window_flat[(g_kh*KW*IC + g_kw*IC + g_ic)*8 +: 8] =
                                window[g_kh][g_kw][g_ic];
                        end else if (g_kh < KH - 1) begin : gen_bram_col
                            assign window_flat[(g_kh*KW*IC + g_kw*IC + g_ic)*8 +: 8] =
                                window_kwm1_wire[g_kh][g_ic*8 +: 8];
                        end else begin : gen_bypass_col
                            assign window_flat[(g_kh*KW*IC + g_kw*IC + g_ic)*8 +: 8] =
                                bypass_reg[g_ic];
                        end
                    end
                end
            end
        end else begin : gen_no_full_window
            // Wide window not exposed -- tie off so the cross-channel mux is
            // never instantiated (eliminates the window_flat routing
            // congestion). Depthwise consumers use chan_window_flat only.
            assign window_flat = {(KH*KW*IC*8){1'b0}};
        end
    endgenerate

    // ---------------- Narrow per-channel window output ----------------
    // chan_window_flat[(kh*KW + kw)*8 +: 8] is the byte that window_flat
    // would have produced at index ((kh*KW + kw)*IC + channel_select).
    // It reads the SAME three source regions, indexed by channel_select:
    //   - columns 0..KW-2          : window[kh][kw][channel_select]
    //   - column KW-1, rows < KH-1 : window_kwm1_wire[kh][channel_select*8 +: 8]
    //   - column KW-1, row  KH-1   : bypass_reg[channel_select]
    // channel_select is a runtime byte-index into the existing packed
    // arrays -- a single C-way mux per tap (KH*KW muxes) instead of the
    // full KH*KW*IC-wide window_flat. ZERO arithmetic change: each output
    // byte is bit-identical to the corresponding window_flat byte.
    /* verilator lint_off WIDTH */
    wire [CSEL_W-1:0] csel = channel_select;
    /* verilator lint_on WIDTH */

    genvar c_kh, c_kw;
    generate
    if (CHAN_SHIFT == 0) begin : gen_chan_mux
        // ===== LEGACY per-tap C:1 channel-select MUX (DEFAULT). BIT-IDENTICAL to the prior design.
        // ResNet (EXPOSE_FULL_WINDOW=1, never reads chan_window_flat) and any caller omitting
        // CHAN_SHIFT elaborate THIS branch unchanged.
        for (c_kh = 0; c_kh < KH; c_kh = c_kh + 1) begin : gen_chan_kh
            for (c_kw = 0; c_kw < KW; c_kw = c_kw + 1) begin : gen_chan_kw
                if (c_kw < KW - 1) begin : gen_chan_shift_col
                    assign chan_window_flat[(c_kh*KW + c_kw)*8 +: 8] =
                        window[c_kh][c_kw][csel];
                end else if (c_kh < KH - 1) begin : gen_chan_bram_col
                    assign chan_window_flat[(c_kh*KW + c_kw)*8 +: 8] =
                        window_kwm1_wire[c_kh][csel*8 +: 8];
                end else begin : gen_chan_bypass_col
                    assign chan_window_flat[(c_kh*KW + c_kw)*8 +: 8] =
                        bypass_reg[csel];
                end
            end
        end
    end else begin : gen_chan_shift
        // ===== ROTATION SHIFT-REGISTER (CHAN_SHIFT==1). Removes the wide per-tap C:1 channel-select
        // mux for every FF-RESIDENT tap (see param doc) -- i.e. the 6 shift-column taps (window[][])
        // and the 1 bypass-row tap (bypass_reg). One C-deep byte rotation bank per such tap:
        //   * LOAD (parallel, all IC channels, ch0 at head) from the SAME source bytes the mux would
        //     select, captured at sched_output_fires (one cycle before the datapath issues channel 0).
        //     These sources are FFs FROZEN during the MAC, so they are fully settled at output_fires.
        //   * ROTATE by 1 on each chan_advance so head walks ch0,ch1,...,ch(IC-1) in lockstep with
        //     current_global_oc at the issuing cycles. Reloaded every output pixel (so the wrap value
        //     is never relied on).
        //   * chan_window_flat = head (position 0), COMBINATIONAL -> same-cycle, ZERO added latency,
        //     FFs only -> NO BRAM. Byte-IDENTICAL: head at issue i == mux(csel==i).
        //
        // The 2 KW-1-column / row<KH-1 taps are sourced from `window_kwm1_wire`, which is COMBINATIONAL
        // off the per-slot BRAM/URAM output register q_reg. Under TILE_STORAGE>0 q_reg is folded
        // INCREMENTALLY during the NT-cycle burst, and that burst is STILL ACTIVE at sched_output_fires
        // (mem_busy overlaps output_fires) -- q_reg settles only at the FIRST issue cycle, the SAME
        // cycle the datapath reads channel 0. There is therefore NO pre-issue cycle at which a
        // REGISTERED load could capture the settled value, so these 2 taps CANNOT be pre-loaded into a
        // rotation bank without adding a latency cycle. They KEEP the legacy C:1 mux (cheap relative
        // to the 7 removed; 2/9 of the wide muxes remain). PROVEN by the per-tap equiv dump: the 7
        // FF-resident taps are byte-exact via rotation; only the 2 window_kwm1_wire taps lag at T0.
        for (c_kh = 0; c_kh < KH; c_kh = c_kh + 1) begin : gen_rot_kh
            for (c_kw = 0; c_kw < KW; c_kw = c_kw + 1) begin : gen_rot_kw
                if (c_kw < KW - 1 || c_kh == KH - 1) begin : gen_tap_rot
                    // FF-resident tap (shift-column OR bypass row): ROTATION BANK (no wide mux).
                    wire [IC*8-1:0] tap_load_src;
                    genvar g_c;
                    for (g_c = 0; g_c < IC; g_c = g_c + 1) begin : gen_rot_src
                        if (c_kw < KW - 1) begin : gen_src_shift_col
                            assign tap_load_src[g_c*8 +: 8] = window[c_kh][c_kw][g_c];
                        end else begin : gen_src_bypass_col
                            assign tap_load_src[g_c*8 +: 8] = bypass_reg[g_c];
                        end
                    end

                    // C-deep byte rotation bank. bank[0] is the head (presented to the datapath).
                    reg [7:0] bank [0:IC-1];
                    integer rb;
                    always @(posedge clk or negedge rst_n) begin
                        if (!rst_n) begin
                            for (rb = 0; rb < IC; rb = rb + 1) bank[rb] <= 8'd0;
                        end else if (frame_start) begin
                            for (rb = 0; rb < IC; rb = rb + 1) bank[rb] <= 8'd0;
                        end else if (sched_output_fires) begin
                            // Parallel load: channel g_c -> bank position g_c (ch0 at head). One cycle
                            // before the datapath's first issue (it consumes head==ch0 next cycle).
                            for (rb = 0; rb < IC; rb = rb + 1)
                                bank[rb] <= tap_load_src[rb*8 +: 8];
                        end else if (chan_advance) begin
                            // Rotate up by 1: head <- bank[1], ..., bank[IC-1] <- bank[0] (circular).
                            // One rotate per consumed channel so head tracks current_global_oc.
                            for (rb = 0; rb < IC - 1; rb = rb + 1) bank[rb] <= bank[rb+1];
                            bank[IC-1] <= bank[0];
                        end
                    end

                    assign chan_window_flat[(c_kh*KW + c_kw)*8 +: 8] = bank[0];
                end else begin : gen_tap_bram_mux
                    // KW-1-column, row<KH-1 tap: window_kwm1_wire is not pre-issue-settled under tiled
                    // storage (see block comment) -> keep the legacy C:1 channel-select mux. Bit-
                    // identical to the CHAN_SHIFT==0 path for this tap.
                    assign chan_window_flat[(c_kh*KW + c_kw)*8 +: 8] =
                        window_kwm1_wire[c_kh][csel*8 +: 8];
                end
            end
        end
    end
    endgenerate

endmodule
