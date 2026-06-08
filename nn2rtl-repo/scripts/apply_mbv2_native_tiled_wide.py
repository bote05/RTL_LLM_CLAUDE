#!/usr/bin/env python3
r"""
apply_mbv2_native_tiled_wide.py

Replicate the PROVEN node_conv_878 NATIVE-256b-TILED re-architecture onto the
THREE WIDE depthwise convs node_conv_896 / 902 / 908 (C=960, N_TILES=30,
PIX_W=7680).  These convs DIFFER structurally from 878: their input assembler
is the (beat_phase / lo_hold) 2-beat assembler with LO_W=4096 / HI_W=3584 that
exposes `pixel_assembled` + `pix_valid` to the core (NOT 878's pix_data_in /
core_valid_in abstraction), and their output side is an FSM that drives
dp_valid/dp_data + emit_hi directly into a legacy passthrough OR a 2-entry beat
FIFO (NOT 878's pix_out / pix_out_ready + 2-beat splitter).  So the NATIVE_TILED
insertion is ADAPTED to the wide assembler's anchors here, written purely in
terms of the wide localparams (C=960, N_TILES=30, PIX_W=7680, TILE_W=256).

MODULE edits  -> output/mobilenet-v2/rtl/node_conv_<id>.v
  (1) HEADER  : NATIVE_TILED comment block + parameter NATIVE_TILED=0
  (2) PORTS   : native 256b ports (valid_in_t/ready_in_t/data_in_t[255:0],
                out_ready_in_t/valid_out_t/data_out_t[255:0])
  (3) TIEOFF  : legacy-mode native tie-off generate
  (4) INPUT   : param-gated input assembler -> legacy 2-beat OR native 30-tile
                gather; both expose the SAME `pixel_assembled`(7680b) + 1-cyc
                `pix_valid`. The native gather's ready/handshake mirrors 878.
  (5) FSM PIX_DONE : add a `pix_out_ready` 1-cyc pulse in the final ST_OUTPUT
                pass (NON-arithmetic; out_pix already settles by the next edge,
                identical pattern to 878). Legacy dp_valid/emit_hi UNCHANGED.
  (6) OUTPUT  : prepend a NATIVE 30-tile output drain generate branch; native
                drives valid_out_t/data_out_t/skid_block=out_busy and reads
                out_pix directly (byte-exact: out_pix tile k = ch k*32..+31).

TOP edits     -> output/mobilenet-v2/rtl/nn2rtl_top_engine.v
  The SAME generic delete-bridges + narrow-data_out + rewire-native transform
  (reused VERBATIM from apply_mbv2_native_tiled_repl.py: it is N_TILES-agnostic
  and only keys off conv id / gather inst / scatter inst / discovered relus).

Idempotent + atomic (assert every anchor BEFORE writing). Backs up to
backups/native_tiled_wide/.

Default targets: 896, 902, 908.
"""
import argparse
import os
import shutil
import sys

REPO = r"C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo"
RTL_DIR = os.path.join(REPO, "output", "mobilenet-v2", "rtl")
DEFAULT_TOP = os.path.join(RTL_DIR, "nn2rtl_top_engine.v")
BACKUP_DIR = os.path.join(REPO, "backups", "native_tiled_wide")
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)

# Reuse the PROVEN, N_TILES-agnostic TOP-level transform from the repl script.
import apply_mbv2_native_tiled_repl as repl  # noqa: E402

MODULE_MARK = "// PARAM-GATED NATIVE-256b-TILED RE-ARCHITECTURE (NATIVE_TILED, default 0):"

# ---------------------------------------------------------------------------
# (1) HEADER: insert NATIVE_TILED comment block before the module decl.
# ---------------------------------------------------------------------------
HEADER_BEFORE = """//     after emitting a pixel's 2 beats until both have drained => no overwrite /
//     reorder / drop. Datapath arithmetic and the FSM are UNCHANGED; only the
//     external emit *timing* changes.
module node_conv_{ID} #("""

HEADER_AFTER = """//     after emitting a pixel's 2 beats until both have drained => no overwrite /
//     reorder / drop. Datapath arithmetic and the FSM are UNCHANGED; only the
//     external emit *timing* changes.
// PARAM-GATED NATIVE-256b-TILED RE-ARCHITECTURE (NATIVE_TILED, default 0):
//   * ==0 (default): the LEGACY 4096b / 2-beat external contract, BYTE/CYCLE-
//     IDENTICAL to the prior module. The legacy 2-beat input assembler + the
//     dp_valid/dp_data (legacy passthrough | 2-entry beat FIFO) output emitter
//     wrap the unchanged split-arch core. The per-module verify TB gets
//     bit-identical behavior.
//   * ==1: the BRIDGELESS NATIVE 256b TILED contract used by the engine top.
//     The retile gather/scatter bridges are deleted; node_conv_{ID} talks 30x256b
//     tiles/pixel DIRECTLY to its native-tiled producer (relu {PROD}) and consumer
//     (relu {CONS}). An internal 30-tile gather assembles the full C*8=7680b pixel
//     (the SAME `pixel_assembled` the legacy 2-beat assembler builds) for the
//     scheduler + line_buf_window, and an internal 30-tile drain emits the
//     completed `out_pix` as 30 contiguous 256b slices. Byte layout is contiguous
//     (tile k = channels k*32..k*32+31 = wide[k*256+:256]) -> LOGICAL-PIXEL-
//     IDENTICAL to the old gather->2beat->reassemble / split->scatter round-trip
//     -> BYTE-EXACT. The legacy 4096b ports are unused (tied off below); the
//     native ports valid_in_t/ready_in_t/data_in_t[255:0] and valid_out_t/
//     out_ready_in_t/data_out_t[255:0] carry the traffic.
module node_conv_{ID} #("""

# ---------------------------------------------------------------------------
# (2) PORTS: add NATIVE_TILED parameter + native 256b ports.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# (3) TIEOFF: insert after the `wire skid_block;` decl block.
#     896 has an EXTRA `wire lbw_mem_busy;` block right after skid_block; 902/908
#     don't. Anchor only on the skid_block decl comment+wire (present in all 3),
#     inserting the tie-off generate immediately after the `wire skid_block;`.
# ---------------------------------------------------------------------------
TIEOFF_BEFORE = """    // ENABLE_BACKPRESSURE==0 it is a constant 0 -> legacy cycle-identical behavior.
    wire skid_block;"""

TIEOFF_AFTER = """    // ENABLE_BACKPRESSURE==0 it is a constant 0 -> legacy cycle-identical behavior.
    wire skid_block;

    // In LEGACY mode (NATIVE_TILED==0) the native 256b ports are unused -> tie off
    // deterministically so the module elaborates cleanly regardless of caller.
    generate
    if (NATIVE_TILED == 0) begin : g_native_tieoff
        assign ready_in_t  = 1'b0;
        assign valid_out_t = 1'b0;
        assign data_out_t  = {256{1'b0}};
    end
    endgenerate"""

# ---------------------------------------------------------------------------
# (4) INPUT ASSEMBLER: replace the whole legacy 2-beat assembler block with a
#     param-gated generate (legacy 2-beat OR native 30-tile gather). Both expose
#     the SAME module-scope wires `pixel_assembled`(7680b) + `pix_valid`(1-cyc).
# ---------------------------------------------------------------------------
INPUT_BEFORE = """    // =====================================================================
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
    wire [PIX_W-1:0] pixel_assembled = { data_in[HI_W-1:0], lo_hold };"""

INPUT_AFTER = """    // =====================================================================
    // INPUT ASSEMBLER (param-gated: legacy 2-beat  |  native 30-tile gather)
    // =====================================================================
    // BOTH modes assemble the SAME full C*8 = 7680b packed pixel
    // (`pixel_assembled`) and pulse `pix_valid` for EXACTLY one cycle into the
    // scheduler / line_buf_window, so the split-arch core is UNCHANGED and the
    // assembled pixel is bit-identical -> byte-exact across modes.
    wire              sched_ready_in;     // scheduler's ready for a real pixel
    wire [PIX_W-1:0]  pixel_assembled;    // full 7680b packed pixel (assembled below)
    wire              pix_valid;          // ONE-cycle pixel handshake into the core

    generate
    if (NATIVE_TILED == 0) begin : g_in_legacy
        // ----------------------------------------------------------------
        // LEGACY 2-BEAT INPUT ASSEMBLER (4096b x 2 beats -> 7680b pixel)
        // ----------------------------------------------------------------
        // Bit/cycle-identical to the prior module.  See header.
        reg               beat_phase;        // 0 => expect lo, 1 => expect hi
        reg [LO_W-1:0]    lo_hold;

        wire bench_fire = valid_in && ready_in;
        // [BP:HI_READY_GATE] legacy: hi beat accepted unconditionally; BP: gate on
        // sched_ready_in so pix_valid never pulses into a frozen scheduler.
        wire hi_ready = (ENABLE_BACKPRESSURE == 0) ? 1'b1 : sched_ready_in;

        assign pix_valid       = bench_fire && (beat_phase == 1'b1);
        // Assembled 7680b pixel: low 4096b = latched lo beat, high 3584b = hi-beat low.
        assign pixel_assembled = { data_in[HI_W-1:0], lo_hold };
        assign ready_in        = (beat_phase == 1'b0) ? sched_ready_in : hi_ready;

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
    end else begin : g_in_native
        // ----------------------------------------------------------------
        // NATIVE INPUT 30-TILE GATHER (30 x 256b tiles -> 7680b pixel)
        // ----------------------------------------------------------------
        // The producer relu {PROD} emits 30 contiguous 256b tiles/pixel, one tile
        // per cycle, each held until accepted (it advances its emit beat iff
        // out_ready_in == ready_in_t, parking the beat otherwise). We gather the
        // 30 tiles into tile_acc[k*256+:256]=tile k. On the 30th accepted tile we
        // form the COMPLETE 7680b pixel (tile_acc with [29*256+:256] = the
        // just-arrived tile) and pulse pix_valid for EXACTLY one cycle -> bit-
        // identical to the legacy {data_in[3583:0], lo_hold} pixel (tiles 0..15 =
        // ch0..511, tiles 16..29 = ch512..959) and to retile_gather's contiguous
        // packing (tile k = channels k*32..k*32+31).
        //
        // BACKPRESSURE: ready_in_t = sched_ready_in for ALL 30 tiles (the scheduler
        // is the only backpressure; stall_in=mac_busy|skid_block|lbw_mem_busy holds
        // sched_ready_in low during the MAC/burst, so the producer stalls). A tile
        // is accepted iff (valid_in_t & ready_in_t) -- the SAME boolean {PROD} sees
        // as out_ready_in_t when it advances -> advance-iff-latch, no lost tile
        // (drain == latch by construction; retile_bridge.v THE INVARIANT, enforced
        // WITHOUT a bridge because both ends share the ready_in_t boolean).
        localparam integer N_TILES = 30;       // C/32 = 960/32
        localparam integer TILE_W  = 256;
        reg [PIX_W-1:0]                tile_acc;
        reg [$clog2(N_TILES)-1:0]      in_tile;   // 0..29

        wire tile_ready  = sched_ready_in;
        wire accept_tile = valid_in_t && tile_ready;
        wire last_tile   = (in_tile == N_TILES[$clog2(N_TILES)-1:0] - 1'b1);

        // COMBINATIONAL complete-pixel: previously-gathered tiles 0..28 from
        // tile_acc plus the just-arrived tile 29 in its slot. Presented to the core
        // only on the last-tile accept cycle (pix_valid pulse).
        wire [PIX_W-1:0] pix_complete;
        assign pix_complete = ({{(PIX_W-(N_TILES-1)*TILE_W){1'b0}},
                                tile_acc[(N_TILES-1)*TILE_W-1:0]})
                              | ({{(PIX_W-TILE_W){1'b0}}, data_in_t} << ((N_TILES-1)*TILE_W));

        assign pixel_assembled = pix_complete;
        assign pix_valid       = accept_tile && last_tile;   // ONE-cycle pulse on tile 29
        assign ready_in_t      = tile_ready;

        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                tile_acc <= {PIX_W{1'b0}};
                in_tile  <= {$clog2(N_TILES){1'b0}};
            end else begin
                if (accept_tile) begin
                    tile_acc[in_tile*TILE_W +: TILE_W] <= data_in_t;
                    if (last_tile) in_tile <= {$clog2(N_TILES){1'b0}};
                    else           in_tile <= in_tile + 1'b1;
                end
            end
        end
    end
    endgenerate"""

# ---------------------------------------------------------------------------
# (5) FSM PIX_DONE pulse: declare a `pix_out_ready` reg next to the FSM emit
#     regs, reset it, default-clear it each cycle, and set it in the final
#     ST_OUTPUT pass. NON-arithmetic; out_pix settles by the same edge the
#     native drain samples pix_out_ready (identical to 878's pix_out_ready).
# ---------------------------------------------------------------------------
# 5a: declare next to dp_valid/dp_data.
FSM_DECL_BEFORE = """    reg             dp_valid;
    reg [BEAT_W-1:0] dp_data;"""
FSM_DECL_AFTER = """    reg             dp_valid;
    reg [BEAT_W-1:0] dp_data;
    // [NATIVE_TILED] 1-cycle whole-pixel-complete pulse for the native 30-tile
    // drain. Set in the final ST_OUTPUT pass; out_pix is fully written (all 960
    // channels) by the edge this is sampled high -> the drain latches the settled
    // out_pix. Unused (never read) in legacy/BP modes -> ZERO behavior change.
    reg             pix_out_ready;"""

# 5b: reset.
FSM_RST_BEFORE = """            dp_valid         <= 1'b0;
            dp_data          <= {BEAT_W{1'b0}};
            emit_hi          <= 1'b0;"""
FSM_RST_AFTER = """            dp_valid         <= 1'b0;
            dp_data          <= {BEAT_W{1'b0}};
            pix_out_ready    <= 1'b0;
            emit_hi          <= 1'b0;"""

# 5c: default-clear each cycle (alongside dp_valid default).
FSM_DEFCLR_BEFORE = """            // ST_OUTPUT pass), so read the hi half directly from out_pix.
            dp_valid <= 1'b0;"""
FSM_DEFCLR_AFTER = """            // ST_OUTPUT pass), so read the hi half directly from out_pix.
            dp_valid <= 1'b0;
            pix_out_ready <= 1'b0;"""

# 5d: set in the final ST_OUTPUT pass (alongside dp_valid/emit_hi).
FSM_SET_BEFORE = """                        dp_valid <= 1'b1;
                        dp_data  <= out_pix[LO_W-1:0];
                        emit_hi  <= 1'b1;
                        state    <= ST_IDLE;"""
FSM_SET_AFTER = """                        dp_valid <= 1'b1;
                        dp_data  <= out_pix[LO_W-1:0];
                        emit_hi  <= 1'b1;
                        pix_out_ready <= 1'b1;   // [NATIVE_TILED] whole-pixel done
                        state    <= ST_IDLE;"""

# ---------------------------------------------------------------------------
# (6) OUTPUT EMITTER: prepend the NATIVE 30-tile drain generate branch before
#     the legacy/BP branches; turn the `if (ENABLE_BACKPRESSURE==0)` into an
#     `else if`. Native drives valid_out_t/data_out_t and skid_block=out_busy.
# ---------------------------------------------------------------------------
OUTPUT_BEFORE = """    // ====================================================================
    // OUTPUT EMITTER (legacy passthrough  |  2-entry elastic beat FIFO)
    // ====================================================================
    generate
    if (ENABLE_BACKPRESSURE == 0) begin : g_emit_legacy"""

OUTPUT_AFTER = """    // ====================================================================
    // OUTPUT EMITTER (native 30-tile drain | legacy passthrough | 2-entry FIFO)
    // ====================================================================
    // NATIVE: on pix_out_ready latch out_pix[7679:0] and drain 30 x 256b tiles
    //   (tile k = out_lat[k*256+:256] = channels k*32..k*32+31). valid_out_t is
    //   COMBINATIONAL on out_busy; advance/clear ONLY on (valid_out_t &
    //   out_ready_in_t) -- the SAME boolean {CONS} uses to latch -> advance ==
    //   latch (drain == latch by construction), no lost/dup tile. skid_block =
    //   out_busy freezes the MAC FSM/rearm while any tile is outstanding, so
    //   pix_out_ready can NEVER fire while out_busy is set (no overwrite/reorder)
    //   -- IDENTICAL invariant to the legacy 2-beat skid, 30-deep counted.
    //   Byte-exact: the 30 emitted tiles == tiles 0..15 (lo beat ch0..511) +
    //   16..29 (hi beat ch512..959) of the old split->scatter round-trip.
    generate
    if (NATIVE_TILED == 1) begin : g_emit_native
        localparam integer ON_TILES = 30;      // C/32 = 960/32
        localparam integer OTILE_W  = 256;
        reg [PIX_W-1:0]            out_lat;     // latched out_pix being drained
        reg [$clog2(ON_TILES)-1:0] out_tile;    // 0..29
        reg                        out_busy;

        assign skid_block   = out_busy;
        assign valid_out_t  = out_busy;
        assign data_out_t   = out_lat[out_tile*OTILE_W +: OTILE_W];
        wire last_out_tile  = (out_tile == ON_TILES[$clog2(ON_TILES)-1:0] - 1'b1);

        // Legacy wide ports unused in native mode -> hold at reset value.
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                valid_out <= 1'b0;
                data_out  <= {BEAT_W{1'b0}};
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
                    out_lat  <= out_pix;
                    out_tile <= {$clog2(ON_TILES){1'b0}};
                    out_busy <= 1'b1;
                end else if (out_busy && out_ready_in_t) begin
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


def apply_one_module(cid, prod, cons):
    path = os.path.join(RTL_DIR, f"node_conv_{cid}.v")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    subs = {"{ID}": cid, "{PROD}": prod, "{CONS}": cons,
            "{WPATH}": repl.wpath(cid), "{BPATH}": repl.bpath(cid)}

    def fill(s):
        for k, v in subs.items():
            s = s.replace(k, v)
        return s

    if MODULE_MARK in text:
        print(f"  module node_conv_{cid}: already native-tiled (no-op)")
        return False

    edits = [
        ("HEADER",      fill(HEADER_BEFORE), fill(HEADER_AFTER)),
        ("PORTS",       fill(PORTS_BEFORE),  fill(PORTS_AFTER)),
        ("TIEOFF",      TIEOFF_BEFORE,       TIEOFF_AFTER),
        ("INPUT",       INPUT_BEFORE,        fill(INPUT_AFTER)),
        ("FSM_DECL",    FSM_DECL_BEFORE,     FSM_DECL_AFTER),
        ("FSM_RST",     FSM_RST_BEFORE,      FSM_RST_AFTER),
        ("FSM_DEFCLR",  FSM_DEFCLR_BEFORE,   FSM_DEFCLR_AFTER),
        ("FSM_SET",     FSM_SET_BEFORE,      FSM_SET_AFTER),
        ("OUTPUT",      OUTPUT_BEFORE,       fill(OUTPUT_AFTER)),
    ]
    for name, before, _after in edits:
        n = text.count(before)
        if n != 1:
            raise RuntimeError(
                f"node_conv_{cid}: edit '{name}' anchor matched {n} times "
                f"(expected 1). File NOT modified.")
    for _name, before, after in edits:
        text = text.replace(before, after, 1)

    if "parameter NATIVE_TILED = 0," not in text:
        raise RuntimeError(f"node_conv_{cid}: NATIVE_TILED parameter not inserted")
    for tok in ["valid_in_t", "ready_in_t", "data_in_t", "valid_out_t",
                "out_ready_in_t", "data_out_t", "g_in_native", "g_emit_native",
                "g_native_tieoff", "N_TILES = 30", "ON_TILES = 30",
                "pix_out_ready"]:
        if tok not in text:
            raise RuntimeError(f"node_conv_{cid}: post-condition token '{tok}' missing")

    os.makedirs(BACKUP_DIR, exist_ok=True)
    bak = os.path.join(BACKUP_DIR, f"node_conv_{cid}.v.pre_native_tiled")
    if not os.path.exists(bak):
        shutil.copyfile(path, bak)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  module node_conv_{cid}: WIDE native-tiled re-arch applied (backup {bak})")
    return True


DEFAULT_CONVS = {
    "896": ("br_896", "br_n4_30"),
    "902": ("br_902", "br_n4_32"),
    "908": ("br_908", "br_n4_34"),
}


def main():
    ap = argparse.ArgumentParser(
        description="Replicate the node_conv_878 NATIVE-256b-tiled re-arch onto the WIDE C=960 convs.")
    ap.add_argument("--conv", action="append", default=None,
                    help="conv id to apply (repeatable). Default: 896, 902, 908.")
    ap.add_argument("--top", default=os.environ.get("NN2RTL_TOP", DEFAULT_TOP))
    ap.add_argument("--module-only", action="store_true")
    ap.add_argument("--top-only", action="store_true")
    args = ap.parse_args()

    convs = {c: DEFAULT_CONVS[c] for c in args.conv} if args.conv else dict(DEFAULT_CONVS)

    # ---- Phase A: module edits ----
    if not args.top_only:
        print("== MODULE edits (WIDE C=960, N_TILES=30) ==")
        with open(args.top, "r", encoding="utf-8") as f:
            toptxt = f.read()
        for cid, (gather, scatter) in convs.items():
            try:
                prod, cons = repl.discover_neighbours(toptxt, cid, gather, scatter)
            except Exception:
                prod, cons = ("n4_prod", "n4_cons")
            apply_one_module(cid, prod, cons)

    # ---- Phase B: top edits (reuse the proven generic transform) ----
    if not args.module_only:
        print("== TOP edits ==")
        with open(args.top, "r", encoding="utf-8") as f:
            text = f.read()
        os.makedirs(BACKUP_DIR, exist_ok=True)
        bak = os.path.join(BACKUP_DIR, "nn2rtl_top_engine.v.pre_native_tiled_wide")
        if not os.path.exists(bak):
            shutil.copyfile(args.top, bak)
        for cid, (gather, scatter) in convs.items():
            text = repl.apply_one_top(text, cid, gather, scatter)
        with open(args.top, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  top written: {args.top} (backup {bak})")

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
