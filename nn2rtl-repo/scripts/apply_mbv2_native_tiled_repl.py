#!/usr/bin/env python3
r"""
apply_mbv2_native_tiled_repl.py

REPLICATE the PROVEN node_conv_878 NATIVE-256b-TILED re-architecture onto the
STRUCTURALLY-IDENTICAL siblings node_conv_884 and node_conv_890 (and, by --conv,
any other C=576 / BUS_W=4096 / N_TILES=18 depthwise sharing the exact
lo_latch / in_phase 2-beat input assembler + 2-beat output splitter).

This script has TWO parts that mirror apply_mbv2_native_tiled_878.py exactly:

  (A) MODULE edits  -> output/mobilenet-v2/rtl/node_conv_<id>.v
      Insert the param-gated NATIVE_TILED path (parameter, native ports, native
      tie-off, native 18-tile input gather, native 18-tile output drain) WITHOUT
      touching any per-conv constant (SCALE_MULT/SCALE_SHIFT/OH/OW/SH/SW/conv-id).
      The inserted logic is written purely in terms of localparams (C, PIX_W,
      LO_W, HI_W, BUS_W, N_TILES=18, TILE_W=256) that are IDENTICAL across these
      siblings, so the inserted text is verbatim-identical to 878's -- only two
      comment-level conv-id / neighbour-relu references are parametrized.

  (B) TOP edits     -> output/mobilenet-v2/rtl/nn2rtl_top_engine.v
      The same delete-bridges + narrow-data_out + rewire-native transform
      apply_mbv2_native_tiled_878.py did, generalized over (conv id, gather inst,
      scatter inst, producer relu, consumer relu). Bridge/relu names are
      DISCOVERED from the top (asserted) so a wrong assumption fails loudly.

Both parts are idempotent + atomic (assert every anchor BEFORE writing; on any
mismatch the file is left untouched) + back up first to backups/native_tiled_repl/.

Default targets: 884, 890. (878 is already applied; --conv overrides.)
"""
import argparse
import os
import re
import shutil
import sys

REPO = r"D:/RTL_LLM_CLAUDE/nn2rtl-repo"
RTL_DIR = os.path.join(REPO, "output", "mobilenet-v2", "rtl")
DEFAULT_TOP = os.path.join(RTL_DIR, "nn2rtl_top_engine.v")
BACKUP_DIR = os.path.join(REPO, "backups", "native_tiled_repl")

# ---------------------------------------------------------------------------
# MODULE-LEVEL before/after hunks (verbatim from node_conv_878.v; constant-free)
# ---------------------------------------------------------------------------

# (1) HEADER: the NATIVE_TILED-mode comment block is INSERTED right before the
#     `module node_conv_<id> #(` line. We anchor on the line that precedes the
#     module decl in the original (the BP-emitter comment tail) + the module line.
HEADER_BEFORE = """//     pixel. Datapath arithmetic is unchanged; only emit *timing* changes.
module node_conv_{ID} #("""

HEADER_AFTER = """//     pixel. Datapath arithmetic is unchanged; only emit *timing* changes.
// PARAM-GATED NATIVE-256b-TILED RE-ARCHITECTURE (NATIVE_TILED, default 0):
//   * ==0 (default): the LEGACY 4096b / 2-beat external contract, BYTE/CYCLE-
//     IDENTICAL to the prior module. The legacy ports (valid_in/ready_in/data_in
//     [4095:0], out_ready_in, valid_out/data_out[4095:0]) are the EXTERNAL
//     interface and the legacy 2-beat input assembler + 2-beat output splitter
//     wrap the unchanged split-arch core. Any caller that still wants the wide
//     2-beat port (or the per-module verify TB) gets bit-identical behavior.
//   * ==1: the BRIDGELESS NATIVE 256b TILED contract used by the engine top.
//     The retile gather/scatter bridges are deleted; node_conv_{ID} talks 18x256b
//     tiles/pixel DIRECTLY to its native-tiled producer (relu {PROD}) and consumer
//     (relu {CONS}). An internal 18-tile gather assembles the full 4608b pixel for
//     line_buf_window (which still re-tiles it for BRAM storage), and an internal
//     18-tile drain emits pix_out as 18 contiguous 256b slices. Byte layout is
//     contiguous (tile k = channels k*32..k*32+31 = wide[k*256+:256]) -> the
//     assembled pixel and the emitted tiles are LOGICAL-PIXEL-IDENTICAL to the old
//     gather->2beat->reassemble / split->scatter round-trip -> BYTE-EXACT.
//     Only the LEGACY 4096b ports are unused (tied off below); the native ports
//     valid_in_t/ready_in_t/data_in_t[255:0] and valid_out_t/out_ready_in_t/
//     data_out_t[255:0] carry the traffic.
module node_conv_{ID} #("""

# (2) MODULE PORT LIST: add NATIVE_TILED parameter, the LEGACY-ports comment, the
#     out_ready_in comment de-"NEW", and the NATIVE 256b ports.
PORTS_BEFORE = """    parameter ENABLE_BACKPRESSURE = 0,
    parameter WEIGHTS_PATH = "{WPATH}",
    parameter BIAS_PATH    = "{BPATH}"
)(
    input  wire           clk,
    input  wire           rst_n,
    input  wire           valid_in,
    output wire           ready_in,
    input  wire [4095:0]  data_in,
    input  wire           out_ready_in,   // NEW: downstream-ready (ignored when ENABLE_BACKPRESSURE==0)
    output reg            valid_out,
    output reg  [4095:0]  data_out
);"""

PORTS_AFTER = """    parameter ENABLE_BACKPRESSURE = 0,
    parameter NATIVE_TILED = 0,
    parameter WEIGHTS_PATH = "{WPATH}",
    parameter BIAS_PATH    = "{BPATH}"
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
);"""

# (3) NATIVE TIE-OFF: inserted after the `wire skid_block;` decl block.
TIEOFF_BEFORE = """    wire skid_block;

    // ----------------- Geometry -----------------"""

TIEOFF_AFTER = """    wire skid_block;

    // In LEGACY mode (NATIVE_TILED==0) the native 256b ports are unused -> tie off
    // deterministically so the module elaborates cleanly regardless of caller.
    generate
    if (NATIVE_TILED == 0) begin : g_native_tieoff
        assign ready_in_t  = 1'b0;
        assign valid_out_t = 1'b0;
        assign data_out_t  = {256{1'b0}};
    end
    endgenerate

    // ----------------- Geometry -----------------"""

# (4) INPUT ASSEMBLER: the whole legacy block (comment header through the in_phase
#     FSM always) is replaced with the param-gated generate (legacy + native).
INPUT_BEFORE = """    // ====================================================================
    // INPUT BEAT ASSEMBLER (4096b x 2 beats  ->  4608b pixel)
    // ====================================================================
    // The contract delivers each input pixel as two beats:
    //   beat 0 : ch 0..511   (full 4096b)
    //   beat 1 : ch 512..575 in the low 512b, high 3584b are z-pad
    // The split-arch scheduler/line_buf_window expect ONE pixel handshake with
    // the full C*8 = 4608b packed pixel. We latch beat 0 into lo_latch and, on
    // beat 1, present {beat1_low_512b, lo_latch} = the whole pixel to the core
    // and pulse the scheduler's valid_in so it advances exactly ONE coord.
    //
    // ready_in (to TB) is high whenever the core can accept the *current* beat.
    //   beat-phase 0 : accept beat 0 (latch low half); the scheduler does NOT
    //                  advance this cycle.
    //   beat-phase 1 : accept beat 1 only when the core's scheduler is ready
    //                  (sched_ready_in); on accept the scheduler advances.
    // During the MAC the scheduler de-asserts sched_ready_in (stall_in=mac_busy),
    // so ready_in falls and the TB pauses -- correct backpressure.

    reg            in_phase;          // 0 = expecting low beat, 1 = expecting high beat
    reg [LO_W-1:0] lo_latch;

    wire                sched_ready_in;
    // [TIMING:BEAT0_READY_GATE] Mirror the conv_908 assembler exactly: gate the
    // beat-0 (lo) accept on sched_ready_in too. Previously beat 0 was accepted
    // unconditionally (ready_in=1 even before the scheduler was running), which
    // let the very first lo beat through one phase early and desync'd vector 0
    // (vec0 latency 6083 vs steady-state 6082). Gating beat 0 on sched_ready_in
    // makes EVERY vector observe the same fill cadence -> uniform per-vector
    // latency, exactly like the (already-uniform) conv_908 sibling. ZERO change
    // to the assembled pixel contents -> byte-exact preserved.
    wire                core_accept_beat0 = (in_phase == 1'b0) && valid_in && sched_ready_in;
    // [BP:BEAT1_READY_GATE] In legacy mode (ENABLE_BACKPRESSURE==0) beat 1 is
    // accepted unconditionally (byte-exact, unchanged). In BP mode the scheduler
    // can be FROZEN by skid_block at an arbitrary phase; if beat 1 were accepted
    // while the scheduler is stalled, core_valid_in would pulse into a frozen
    // scheduler (no handshake) and that input pixel would be DROPPED -> corrupt
    // window -> wrong outputs. So in BP mode gate beat 1 on sched_ready_in too,
    // holding the upstream until the scheduler can actually consume the pixel.
    wire                beat1_ready = (ENABLE_BACKPRESSURE == 0) ? 1'b1 : sched_ready_in;
    wire                core_accept_beat1 = (in_phase == 1'b1) && valid_in && beat1_ready;

    // Full assembled pixel presented to line_buf_window. On the beat-1 accept
    // cycle the low half is lo_latch (from beat 0) and the high 512b are the
    // low 512b of the current beat-1 data.
    wire [PIX_W-1:0] pix_data_in = {data_in[HI_W-1:0], lo_latch};

    // Scheduler observes a pixel handshake only on the beat-1 accept cycle.
    // (beat1_ready is 1 in legacy mode => unchanged; in BP mode it gates on
    //  sched_ready_in so the pulse never lands in a frozen scheduler.)
    wire core_valid_in = (in_phase == 1'b1) && valid_in && beat1_ready;

    // ready_in to the TB.  Beat 0 accepted only when the scheduler can take a
    // new real pixel; beat 1 accepted unconditionally (legacy) / when the
    // scheduler is ready (BP) once lo is latched.
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
    end"""

INPUT_AFTER = """    // ====================================================================
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
        // The producer relu {PROD} emits 18 contiguous 256b tiles/pixel, one tile
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
        // ready_in_t) -- the SAME boolean {PROD} sees as out_ready_in_t when it
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

        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                tile_acc <= {PIX_W{1'b0}};
                in_tile  <= {$clog2(N_TILES){1'b0}};
            end else begin
                if (accept_tile) begin
                    // Write the just-arrived tile into its slot.
                    tile_acc[in_tile*TILE_W +: TILE_W] <= data_in_t;
                    if (last_tile) in_tile <= {$clog2(N_TILES){1'b0}};
                    else           in_tile <= in_tile + 1'b1;
                end
            end
        end
    end
    endgenerate"""

# (5) OUTPUT EMITTER HEADER: comment block + (no body change here) -- the comment
#     header is rewritten, then the native generate branch is PREPENDED before
#     the legacy `if (ENABLE_BACKPRESSURE == 0) begin : g_emit_legacy`.
OUTPUT_BEFORE = """    // ====================================================================
    // OUTPUT BEAT SPLITTER (4608b pixel  ->  4096b x 2 beats)
    // ====================================================================
    // When the datapath completes a pixel (pix_out_ready), emit it as two
    // beats over two consecutive cycles:
    //   beat 0 : pix_out[4095:0]              (ch 0..511)
    //   beat 1 : {z-pad, pix_out[4607:4096]}  (ch 512..575)
    // valid_out is asserted on BOTH beat cycles. The MAC's per-pixel cadence
    // (42*144 = 6048 cycles >> 2) guarantees the two beats always drain before
    // the next pixel completes, so a simple 2-state emitter suffices.
    //
    // [INVARIANT:VALID_OUT_LATENCY] The first beat-0 fires the cycle after the
    // last OC pass's ST_OUTPUT (pix_out_ready) -- i.e. exactly at the same
    // edge the single-beat reference would assert valid_out -- preserving the
    // 6066-cycle scheduler-to-output latency the golden expects.

    reg [PIX_W-1:0] em_buf;
    reg             em_busy;
    reg             em_phase;          // 0 = drive beat 0 next, 1 = drive beat 1 next

    generate
    if (ENABLE_BACKPRESSURE == 0) begin : g_emit_legacy"""

OUTPUT_AFTER = """    // ====================================================================
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
        // (valid_out_t & out_ready_in_t) -- the SAME boolean {CONS} uses to latch
        // ({CONS}.valid_in = node_conv_{ID}_valid_out & {CONS}.ready_in) -> advance ==
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

        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                out_lat  <= {PIX_W{1'b0}};
                out_tile <= {$clog2(ON_TILES){1'b0}};
                out_busy <= 1'b0;
            end else begin
                if (pix_out_ready) begin
                    // Capture the full pixel. (skid_block guarantees !out_busy here.)
                    out_lat  <= pix_out;
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
    end else if (ENABLE_BACKPRESSURE == 0) begin : g_emit_legacy"""

MODULE_MARK = "// PARAM-GATED NATIVE-256b-TILED RE-ARCHITECTURE (NATIVE_TILED, default 0):"


def wpath(cid):
    return (f"D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/"
            f"mobilenet-v2/weights/node_conv_{cid}_weights.hex")


def bpath(cid):
    return (f"D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/"
            f"mobilenet-v2/weights/node_conv_{cid}_bias.hex")


def apply_one_module(cid, prod, cons):
    path = os.path.join(RTL_DIR, f"node_conv_{cid}.v")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # NOTE: the hunk text contains Verilog concatenations ({...}) so we MUST NOT
    # use str.format here (it would treat them as format fields). Substitute only
    # the explicit {ID}/{PROD}/{CONS}/{WPATH}/{BPATH} placeholders via .replace.
    subs = {"{ID}": cid, "{PROD}": prod, "{CONS}": cons,
            "{WPATH}": wpath(cid), "{BPATH}": bpath(cid)}

    def fill(s):
        for k, v in subs.items():
            s = s.replace(k, v)
        return s

    if MODULE_MARK in text:
        print(f"  module node_conv_{cid}: already native-tiled (no-op)")
        return False

    edits = [
        ("HEADER", fill(HEADER_BEFORE), fill(HEADER_AFTER)),
        ("PORTS", fill(PORTS_BEFORE), fill(PORTS_AFTER)),
        ("TIEOFF", TIEOFF_BEFORE, TIEOFF_AFTER),
        ("INPUT", INPUT_BEFORE, fill(INPUT_AFTER)),
        ("OUTPUT", OUTPUT_BEFORE, fill(OUTPUT_AFTER)),
    ]
    # Assert every BEFORE matches EXACTLY once, then apply.
    for name, before, _after in edits:
        n = text.count(before)
        if n != 1:
            raise RuntimeError(
                f"node_conv_{cid}: edit '{name}' anchor matched {n} times "
                f"(expected 1). File NOT modified.")
    for _name, before, after in edits:
        text = text.replace(before, after, 1)

    # Post-conditions.
    if "parameter NATIVE_TILED = 0," not in text:
        raise RuntimeError(f"node_conv_{cid}: NATIVE_TILED parameter not inserted")
    for tok in ["valid_in_t", "ready_in_t", "data_in_t", "valid_out_t",
                "out_ready_in_t", "data_out_t", "g_in_native", "g_emit_native",
                "g_native_tieoff", "N_TILES = 18", "ON_TILES = 18"]:
        if tok not in text:
            raise RuntimeError(f"node_conv_{cid}: post-condition token '{tok}' missing")

    os.makedirs(BACKUP_DIR, exist_ok=True)
    bak = os.path.join(BACKUP_DIR, f"node_conv_{cid}.v.pre_native_tiled")
    if not os.path.exists(bak):
        shutil.copyfile(path, bak)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  module node_conv_{cid}: native-tiled re-arch applied (backup {bak})")
    return True


# ---------------------------------------------------------------------------
# TOP-LEVEL transform (generalized apply_mbv2_native_tiled_878.py)
# ---------------------------------------------------------------------------

def find_inst(text, mid):
    m = re.search(r"(?:^|\n)\s*" + re.escape(mid)
                  + r"\s+(?:#\([^\n]*\)\s+)?u_" + re.escape(mid)
                  + r"\s*\((?:.|\n)*?\n\s*\);", text)
    if not m:
        raise RuntimeError(f"instance u_{mid} not found")
    return m


def find_retile_inst(text, kind, inst):
    m = re.search(r"retile_" + re.escape(kind) + r"\s+#\([^\n]*\)\s+u_" + re.escape(inst)
                  + r"\s*\((?:.|\n)*?\n\s*\);\n", text)
    if not m:
        raise RuntimeError(f"retile_{kind} u_{inst} not found")
    return m


def delete_wire_group(text, base):
    pat = (r"[ \t]*wire " + re.escape(base) + r"_valid_out;\n"
           r"[ \t]*wire \[\d+:0\] " + re.escape(base) + r"_data_out;\n"
           r"[ \t]*wire " + re.escape(base) + r"_ready_out;[^\n]*\n"
           r"[ \t]*wire " + re.escape(base) + r"_stall_out;\n"
           r"[ \t]*wire " + re.escape(base) + r"_wr_accept;[^\n]*\n")
    new, n = re.subn(pat, "", text, count=1)
    if n != 1:
        raise RuntimeError(f"wire group for {base} not matched (n={n})")
    return new


def delete_drain_wire(text, base):
    pat = r"[ \t]*wire spatial_run_drain_" + re.escape(base) + r" = [^\n;]*;\n"
    new, n = re.subn(pat, "", text, count=1)
    if n != 1:
        raise RuntimeError(f"spatial_run_drain_{base} wire not matched (n={n})")
    return new


def remove_stall_terms(text, terms):
    m = re.search(r"wire any_retile_stall = ([^\n;]*);", text)
    if not m:
        raise RuntimeError("any_retile_stall assignment not found")
    expr = m.group(1)
    parts = [p.strip() for p in expr.split("|")]
    before = len(parts)
    parts = [p for p in parts if p not in terms]
    removed = before - len(parts)
    if removed != len(terms):
        if all(t not in expr for t in terms):
            return text
        raise RuntimeError(
            f"expected to remove {len(terms)} stall terms, removed {removed} "
            f"(terms={terms}, expr={expr!r})")
    new_expr = " | ".join(parts)
    return text[:m.start(1)] + new_expr + text[m.end(1):]


def patch_producer_relu(text, prod, gather, cid):
    """producer.out_ready_in: <gather>_wr_accept -> node_conv_<cid>_ready_in."""
    m = find_inst(text, prod)
    blk = m.group(0)
    nb, n = re.subn(r"\.out_ready_in\(" + re.escape(gather) + r"_wr_accept\)",
                    f".out_ready_in(node_conv_{cid}_ready_in)", blk, count=1)
    if n != 1:
        if f"node_conv_{cid}_ready_in" in blk:
            return text
        raise RuntimeError(
            f".out_ready_in({gather}_wr_accept) not found in u_{prod}")
    return text[:m.start()] + nb + text[m.end():]


CONV_NATIVE_TMPL = """node_conv_{ID} #(.ENABLE_BACKPRESSURE(1), .NATIVE_TILED(1)) u_node_conv_{ID} (
.clk(clk), .rst_n(rst_n),
        // [NATIVE_TILED_{ID}] bridgeless native 256b tiled ports wired DIRECTLY
        // {PROD} -> {ID} -> {CONS} (legacy wide ports unconnected -> tied off inside).
        // RAW valid (no & spatial_run): advance-iff-latch via the shared ready
        // boolean (node_conv_{ID}_ready_in on the input edge, {CONS}_ready_in on the
        // output edge); see retile_bridge.v THE INVARIANT and apply_mbv2_native_tiled_repl.py.
        .valid_in_t({PROD}_valid_out),
        .ready_in_t(node_conv_{ID}_ready_in),
        .data_in_t({PROD}_data_out),
        .out_ready_in_t({CONS}_ready_in),
        .valid_out_t(node_conv_{ID}_valid_out),
        .data_out_t(node_conv_{ID}_data_out)
    );"""


def patch_conv(text, cid, prod, cons):
    m = find_inst(text, f"node_conv_{cid}")
    blk = m.group(0)
    if ".NATIVE_TILED(1)" in blk and ".valid_in_t(" in blk:
        return text
    repl = CONV_NATIVE_TMPL.format(ID=cid, PROD=prod, CONS=cons)
    return text[:m.start()] + repl + text[m.end():]


def patch_consumer_relu(text, cons, cid):
    """consumer relu: valid_in/data_in re-sourced from node_conv_<cid> (RAW valid)."""
    m = find_inst(text, cons)
    blk = m.group(0)
    if f"node_conv_{cid}_valid_out" in blk:
        return text
    nb, n1 = re.subn(r"\.valid_in\([^\n]*?\)(?=[,\s])",
                     f".valid_in(node_conv_{cid}_valid_out)", blk, count=1)
    if n1 != 1:
        raise RuntimeError(f".valid_in not matched in u_{cons}")
    nb, n2 = re.subn(r"\.data_in\([^\n]*?\)(?=[,\s])",
                     f".data_in(node_conv_{cid}_data_out)", nb, count=1)
    if n2 != 1:
        raise RuntimeError(f".data_in not matched in u_{cons}")
    return text[:m.start()] + nb + text[m.end():]


def narrow_data_out_wire(text, cid):
    pat = r"wire \[4095:0\] node_conv_" + re.escape(cid) + r"_data_out;[^\n]*"
    repl = (f"wire [255:0] node_conv_{cid}_data_out;  "
            f"// [NATIVE_TILED_{cid}] narrowed: native 256b tile bus")
    new, n = re.subn(pat, repl, text, count=1)
    if n != 1:
        if re.search(r"wire \[255:0\] node_conv_" + re.escape(cid) + r"_data_out;", text):
            return text
        raise RuntimeError(f"node_conv_{cid}_data_out [4095:0] decl not found")
    return new


def discover_neighbours(text, cid, gather, scatter):
    """Deterministically read the bridges to confirm producer/consumer relus:
       * gather.data_in(<prod>_data_out)  names the PRODUCER relu.
       * the unique non-retile module instance whose BODY reads
         <scatter>_valid_out (its .valid_in source) is the CONSUMER relu.
       Returns (prod, cons). Idempotent: after the patch the bridges are gone, so
       we instead read the already-native conv instance (data_in_t / out_ready_in_t)."""
    # --- Producer ---
    if re.search(r"u_" + re.escape(gather) + r"\s*\(", text):
        gb = find_retile_inst(text, "gather", gather).group(0)
        mp = re.search(r"\.data_in\((\w+?)_data_out\)", gb)
        if not mp:
            raise RuntimeError(f"could not find producer in u_{gather} .data_in")
        prod = mp.group(1)
    else:
        # Already patched: read the native conv .data_in_t(<prod>_data_out).
        cb = find_inst(text, f"node_conv_{cid}").group(0)
        mp = re.search(r"\.data_in_t\((\w+?)_data_out\)", cb)
        prod = mp.group(1) if mp else "n4_prod"

    # --- Consumer ---
    cons = None
    if re.search(r"u_" + re.escape(scatter) + r"\s*\(", text):
        # Scan every module instance; pick the one whose BODY references
        # <scatter>_valid_out (the consumer relu's .valid_in source).
        for inst in re.finditer(
                r"(\w+)\s+(?:#\([^\n]*\)\s+)?u_(\w+)\s*\((?:[^()]|\([^()]*\))*?\n\s*\);",
                text):
            body = inst.group(0)
            mod, uname = inst.group(1), inst.group(2)
            if mod.startswith("retile_"):
                continue
            if re.search(r"\.valid_in\([^)]*\b" + re.escape(scatter) + r"_valid_out\b", body):
                cons = uname
                break
        if cons is None:
            raise RuntimeError(
                f"could not find consumer relu reading {scatter}_valid_out")
    else:
        # Already patched: read the native conv .out_ready_in_t(<cons>_ready_in).
        cb = find_inst(text, f"node_conv_{cid}").group(0)
        mc = re.search(r"\.out_ready_in_t\((\w+?)_ready_in\)", cb)
        cons = mc.group(1) if mc else "n4_cons"
    return prod, cons


def apply_one_top(text, cid, gather, scatter):
    mark = f"// [NATIVE_TILED_{cid}] node_conv_{cid} bridgeless native-256b re-arch applied"
    prod, cons = discover_neighbours(text, cid, gather, scatter)
    print(f"  top node_conv_{cid}: producer={prod} consumer={cons} "
          f"gather=u_{gather} scatter=u_{scatter}")

    # 1) Delete the two retile instances.
    if re.search(r"u_" + re.escape(gather) + r"\s*\(", text):
        m = find_retile_inst(text, "gather", gather)
        text = text[:m.start()] + text[m.end():]
    if re.search(r"u_" + re.escape(scatter) + r"\s*\(", text):
        m = find_retile_inst(text, "scatter", scatter)
        text = text[:m.start()] + text[m.end():]

    # 2) Delete the dead per-bridge drain wires.
    if re.search(r"wire spatial_run_drain_" + re.escape(gather) + r" ", text):
        text = delete_drain_wire(text, gather)
    if re.search(r"wire spatial_run_drain_" + re.escape(scatter) + r" ", text):
        text = delete_drain_wire(text, scatter)

    # 3) Delete the dead bridge wire-decl groups.
    if re.search(r"wire " + re.escape(gather) + r"_valid_out;", text):
        text = delete_wire_group(text, gather)
    if re.search(r"wire " + re.escape(scatter) + r"_valid_out;", text):
        text = delete_wire_group(text, scatter)

    # 4) Remove the two stall terms from any_retile_stall.
    text = remove_stall_terms(text, [f"{gather}_stall_out", f"{scatter}_stall_out"])

    # 5) Narrow the data_out wire.
    text = narrow_data_out_wire(text, cid)

    # 6) Re-wire the three instances.
    text = patch_producer_relu(text, prod, gather, cid)
    text = patch_conv(text, cid, prod, cons)
    text = patch_consumer_relu(text, cons, cid)

    # 7) Sanity: no dangling references to the deleted nets/insts remain.
    for base in (gather, scatter):
        for suff in ["_valid_out", "_data_out", "_ready_out", "_stall_out",
                     "_wr_accept"]:
            dead = base + suff
            if re.search(r"\b" + re.escape(dead) + r"\b", text):
                raise RuntimeError(
                    f"node_conv_{cid}: dangling reference '{dead}' remains")
        if re.search(r"\bspatial_run_drain_" + re.escape(base) + r"\b", text):
            raise RuntimeError(
                f"node_conv_{cid}: dangling spatial_run_drain_{base} remains")
        if re.search(r"\bu_" + re.escape(base) + r"\b", text):
            raise RuntimeError(f"node_conv_{cid}: dangling inst u_{base} remains")
    if not re.search(r"wire \[255:0\] node_conv_" + re.escape(cid) + r"_data_out;", text):
        raise RuntimeError(f"node_conv_{cid}_data_out not narrowed to [255:0]")
    if f".NATIVE_TILED(1)) u_node_conv_{cid}" not in text:
        raise RuntimeError(f"node_conv_{cid} not instantiated with .NATIVE_TILED(1)")

    if mark not in text:
        # Stamp near the wave-2 bridges banner (idempotency detector).
        if "// ===== WAVE-2 RETILE BRIDGES" in text:
            text = text.replace(
                "    // ===== WAVE-2 RETILE BRIDGES",
                "    " + mark + "\n    // ===== WAVE-2 RETILE BRIDGES", 1)
        else:
            text = mark + "\n" + text
    return text


# Default conv -> (gather inst, scatter inst). producer/consumer relus are
# DISCOVERED from the top so we never hard-code a wrong neighbour.
DEFAULT_CONVS = {
    "884": ("br_884", "br_n4_26"),
    "890": ("br_890", "br_n4_28"),
}


def main():
    ap = argparse.ArgumentParser(
        description="Replicate the node_conv_878 NATIVE-256b-tiled re-arch onto siblings.")
    ap.add_argument("--conv", action="append", default=None,
                    help="conv id to apply (repeatable). Default: 884, 890.")
    ap.add_argument("--gather", default=None, help="override gather inst name (single --conv only)")
    ap.add_argument("--scatter", default=None, help="override scatter inst name (single --conv only)")
    ap.add_argument("--top", default=os.environ.get("NN2RTL_TOP", DEFAULT_TOP))
    ap.add_argument("--module-only", action="store_true")
    ap.add_argument("--top-only", action="store_true")
    args = ap.parse_args()

    if args.conv:
        convs = {}
        for c in args.conv:
            if args.gather and args.scatter and len(args.conv) == 1:
                convs[c] = (args.gather, args.scatter)
            else:
                convs[c] = DEFAULT_CONVS[c]
    else:
        convs = dict(DEFAULT_CONVS)

    # ---- Phase A: module edits ----
    if not args.top_only:
        print("== MODULE edits ==")
        # Discover producer/consumer for the module-header comment from the TOP.
        with open(args.top, "r", encoding="utf-8") as f:
            toptxt = f.read()
        for cid, (gather, scatter) in convs.items():
            try:
                prod, cons = discover_neighbours(toptxt, cid, gather, scatter)
            except Exception:
                prod, cons = ("n4_prod", "n4_cons")
            apply_one_module(cid, prod, cons)

    # ---- Phase B: top edits ----
    if not args.module_only:
        print("== TOP edits ==")
        with open(args.top, "r", encoding="utf-8") as f:
            text = f.read()
        os.makedirs(BACKUP_DIR, exist_ok=True)
        bak = os.path.join(BACKUP_DIR, "nn2rtl_top_engine.v.pre_native_tiled_repl")
        if not os.path.exists(bak):
            shutil.copyfile(args.top, bak)
        for cid, (gather, scatter) in convs.items():
            text = apply_one_top(text, cid, gather, scatter)
        with open(args.top, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  top written: {args.top} (backup {bak})")

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
