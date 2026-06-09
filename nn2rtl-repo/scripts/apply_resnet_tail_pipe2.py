#!/usr/bin/env python3
"""apply_resnet_tail_pipe2.py (2026-06-09) -- TAIL_PIPE, re-engineered for the K1 era.

Fmax: pipeline the requant TAIL of rtl_library/conv_datapath_mp_k.v. The binding
combinational stages after DSP_INPUT_PIPE+max_fanout are ST_OUTPUT (round-bias +
add + arithmetic barrel shift + clip in ONE cycle, ~12-16ns) and ST_SCALE
(scale_rom read feeding the 33x16 multiply, ~8-11ns). This splits them:

  ST_BIAS   : + prefetch/register the per-OC scale word (sc_mult_q/sc_shift_q)
  ST_SCALE  : multiply biased (reg) x sc_mult_q (REG)   -> pure reg-to-reg mult
  ST_OUT_ROUND : out_round_q <= (shift==0) ? 0 : 1 <<< (shift-1)   [barrel #1]
  ST_OUT_SHIFT : v_tmp_q     <= (scaled + out_round_q) >>> shift   [add+barrel #2]
  ST_OUT_SAT   : data_out    <= clip(v_tmp_q); oc advance / valid_out

+2 cycles per oc_pass when ON. Byte-exact: identical operator chain on identical
values -- only register boundaries move. DEFAULT-OFF param TAIL_PIPE(=0):
MobileNetV2 + every unpatched instantiation elaborate byte- AND latency-IDENTICAL
(all new code is behind elaboration-constant TAIL_PIPE conditions; the new FSM
arms recover to ST_IDLE for TAIL_PIPE=0, exactly the old `default:` behavior).

DIFFERENCES vs the FAILED 2026-06-05 apply_resnet_tail_pipe.py (B23 deadlock --
see docs/agent_tasks/TAIL_PIPE2_ANALYSIS.md for the root cause):
  1. K1-aware: the 06-05 hunks predate the [K1-FDCE] recode that moved every
     datapath write (biased/scaled/data_out/acc) out of the async-reset FSM
     block into sync-only "Block A". This applier extends BOTH blocks
     coherently: Block A gets the new write conditions (state==ST_OUT_*),
     keeping the acc clears textually LAST (NBA last-write-wins parity); the
     FSM block gets ONLY control (state, valid_out, oc_group, k_group).
  2. Enablement set: ONLY the live spatial mp_k convs *instantiated in
     nn2rtl_top.v*. NOT node_conv_196 (stem: special fixed-timing 2-beat
     output splitter + it heads every downstream phase) and NOT the 9
     engine-dispatched files (246/254/260/266/272/278 + K5's 284/292/298).
     The 06-05 applier keyed off `.DSP_INPUT_PIPE(1)` which matched ALL 45,
     including the stem.
  3. New tail registers are written ONLY in Block A (sync, NO reset) --
     K1/FDRE-consistent. Every read is preceded by a same-pass write
     (sc_*_q in ST_BIAS; out_round_q in ROUND; v_tmp_q in SHIFT), so no
     reset value is ever observable.
  4. Fork-receiver FIFO margin (TOP_EDITS below): e2e forensics proved the
     06-05 wedge class is actually a LATENT lossy handshake -- the narrow-relu
     streamer presents each pixel's LAST beat for exactly one cycle, and a
     cadence change can land that cycle on a not-ready fork => beat silently
     lost => lockstep residual join wedges forever. The tail's +2/oc_pass
     trips it at relu_9's fork; deepening that fork's two receivers restores
     the alignment margin (byte-exact, URAM-cheap). The DURABLE fix (hold the
     last beat until accepted, elastic-producer retrofit of the relu
     template) is recommended follow-up -- see TAIL_PIPE2_ANALYSIS.md.

Anchor-asserted (every hunk must match EXACTLY once), idempotent (marker
[TAIL-PIPE2] / .TAIL_PIPE( skips), --dry-run (report only, no writes),
timestamped backups of every touched file.

Usage:
  python scripts/apply_resnet_tail_pipe2.py [--dry-run]

Verify after applying:
  Verilator lint  (see TAIL_PIPE2_ANALYSIS.md)
  NN2RTL_VALUE_THREADS=1 npx tsx scripts/run_nn2rtl_top_value.ts 0
    -> expect PASS mismatch=0; cycle count slightly above the 12,670,107
       baseline (+2 cycles x OC_PASSES per output pixel on the 35 convs).
"""
import sys, os, re, glob, time, shutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DP   = os.path.join(REPO, "rtl_library", "conv_datapath_mp_k.v")
TOP  = os.path.join(REPO, "output", "rtl", "nn2rtl_top.v")
NODE_GLOB = os.path.join(REPO, "output", "rtl", "node_conv_*.v")

DRY_RUN = "--dry-run" in sys.argv

# The stem: fixed-timing 2-beat output splitter wrapper; also do-not-touch per task.
STEM = "node_conv_196"

# --------------------------------------------------------------------------
# conv_datapath_mp_k.v hunks (against the K1-era file, commit be16f61)
# --------------------------------------------------------------------------
DP_EDITS = [

# HUNK A -- TAIL_PIPE parameter -----------------------------------------------
("""    parameter integer DSP_INPUT_PIPE = 0
) (""",
 """    parameter integer DSP_INPUT_PIPE = 0,
    // [TAIL-PIPE2] 1 => pipeline the requant tail: prefetch/register the per-OC
    // scale word during ST_BIAS (ST_SCALE then multiplies a REGISTER, not a
    // ROM read) and split ST_OUTPUT's round+add+barrel-shift+clip into three
    // register-bounded states ST_OUT_ROUND/ST_OUT_SHIFT/ST_OUT_SAT. +2 cycles
    // per oc_pass, byte-identical values. 0 = legacy: byte- AND latency-
    // identical elaboration (MobileNetV2 + unpatched instantiations unaffected).
    parameter integer TAIL_PIPE = 0
) ("""),

# HUNK B -- tail sub-state encodings ------------------------------------------
("""    localparam ST_OUTPUT = 3'd4;

    reg [2:0] state;""",
 """    localparam ST_OUTPUT = 3'd4;
    // [TAIL-PIPE2] requant-tail sub-states (reachable only when TAIL_PIPE!=0;
    // for TAIL_PIPE==0 their FSM arms recover to ST_IDLE = old default: arm).
    localparam ST_OUT_ROUND = 3'd5;
    localparam ST_OUT_SHIFT = 3'd6;
    localparam ST_OUT_SAT   = 3'd7;

    reg [2:0] state;"""),

# HUNK C -- tail pipeline registers (datapath regs: Block-A-written, NO reset,
# K1/FDRE-consistent; every read is preceded by a same-pass write) -------------
("""    reg        [5:0]          out_shift;   // per-OC shift (OUTPUT stage)
    reg signed [SCALED_W-1:0] out_round;   // per-OC round bias (OUTPUT stage)""",
 """    reg        [5:0]          out_shift;   // per-OC shift (OUTPUT stage)
    reg signed [SCALED_W-1:0] out_round;   // per-OC round bias (OUTPUT stage)
    // [TAIL-PIPE2] requant-tail pipeline registers (read iff TAIL_PIPE!=0).
    // Sync-only Block A writes, no reset (K1/FDRE rule): sc_*_q are written in
    // ST_BIAS before any ST_SCALE/ST_OUT_* read of the same oc_pass;
    // out_round_q in ST_OUT_ROUND before its ST_OUT_SHIFT read; v_tmp_q in
    // ST_OUT_SHIFT before its ST_OUT_SAT read.
    reg        [15:0]         sc_mult_q   [0:MP-1];
    reg        [5:0]          sc_shift_q  [0:MP-1];
    reg signed [SCALED_W-1:0] out_round_q [0:MP-1];
    reg signed [SCALED_W-1:0] v_tmp_q     [0:MP-1];"""),

# HUNK D -- Block A: ST_BIAS arm also prefetches the per-OC scale word ---------
("""      // ST_BIAS: bias-add per lane.
      if (state == ST_BIAS) begin
        for (fsm_lane_i = 0; fsm_lane_i < MP; fsm_lane_i = fsm_lane_i + 1) begin
          bias_oc = oc_group * MP + fsm_lane_i;
          if (bias_oc < OC)
            biased[fsm_lane_i] <= $signed(acc[fsm_lane_i]) + $signed(biases[bias_oc]);
          else
            biased[fsm_lane_i] <= 0;
        end
      end""",
 """      // ST_BIAS: bias-add per lane.
      if (state == ST_BIAS) begin
        for (fsm_lane_i = 0; fsm_lane_i < MP; fsm_lane_i = fsm_lane_i + 1) begin
          bias_oc = oc_group * MP + fsm_lane_i;
          if (bias_oc < OC)
            biased[fsm_lane_i] <= $signed(acc[fsm_lane_i]) + $signed(biases[bias_oc]);
          else
            biased[fsm_lane_i] <= 0;
          // [TAIL-PIPE2] prefetch+register the per-OC scale word so ST_SCALE
          // multiplies a REGISTER (kills the scale_rom-read->multiply path) and
          // the tail states shift by a REGISTER. Same scale_rom word the legacy
          // path reads in ST_SCALE/ST_OUTPUT (ROM is read-only after init).
          if (TAIL_PIPE != 0 && bias_oc < OC) begin
            sc_mult_q[fsm_lane_i]  <= scale_rom[bias_oc][15:0];
            sc_shift_q[fsm_lane_i] <= scale_rom[bias_oc][21:16];
          end
        end
      end"""),

# HUNK E -- Block A: ST_SCALE multiplies the registered scale when ON ----------
("""          if (sc_oc < OC)
            scaled[fsm_lane_i] <= $signed(biased[fsm_lane_i]) *
                                  $signed(scale_rom[sc_oc][15:0]);
          else
            scaled[fsm_lane_i] <= 0;""",
 """          if (sc_oc < OC)
            // [TAIL-PIPE2] ON: registered operand (identical VALUE -- sc_mult_q
            // was loaded from scale_rom[bias_oc==sc_oc][15:0] in ST_BIAS); both
            // branches are 16-bit $signed, so the multiply context is unchanged.
            // The (TAIL_PIPE != 0) select is an elaboration constant (no mux).
            scaled[fsm_lane_i] <= $signed(biased[fsm_lane_i]) *
                                  ((TAIL_PIPE != 0) ? $signed(sc_mult_q[fsm_lane_i])
                                                    : $signed(scale_rom[sc_oc][15:0]));
          else
            scaled[fsm_lane_i] <= 0;"""),

# HUNK F -- Block A: write conditions for the three tail states. Inserted
# BEFORE the accumulator clears so the clears stay textually LAST (the K1
# NBA last-write-wins contract). data_out's ST_OUT_SAT writes mirror the
# ST_OUTPUT writes byte-for-byte, two register stages later. ------------------
("""      // Accumulator clears LAST: textual-order parity with the original
      // single block (the case-statement clears overrode the accumulate).
      if (state == ST_IDLE && start_mac) begin
        for (fsm_lane_i = 0; fsm_lane_i < MP; fsm_lane_i = fsm_lane_i + 1)
          acc[fsm_lane_i] <= 0;
      end
      if (state == ST_OUTPUT && oc_group != OC_PASSES - 1) begin
        for (fsm_lane_i = 0; fsm_lane_i < MP; fsm_lane_i = fsm_lane_i + 1)
          acc[fsm_lane_i] <= 0;
      end
    end""",
 """      // [TAIL-PIPE2] ST_OUT_ROUND: per-OC round bias from the REGISTERED shift.
      // Identical expression to the legacy ST_OUTPUT out_round computation.
      if (TAIL_PIPE != 0 && state == ST_OUT_ROUND) begin
        for (fsm_lane_i = 0; fsm_lane_i < MP; fsm_lane_i = fsm_lane_i + 1) begin
          out_oc = oc_group * MP + fsm_lane_i;
          if (out_oc < OC)
            out_round_q[fsm_lane_i] <=
                (sc_shift_q[fsm_lane_i] == 6'd0) ? {SCALED_W{1'b0}}
              : ({{(SCALED_W-1){1'b0}}, 1'b1} <<< (sc_shift_q[fsm_lane_i] - 6'd1));
        end
      end

      // [TAIL-PIPE2] ST_OUT_SHIFT: add + arithmetic right barrel shift.
      // Identical expression to legacy v_tmp, on identical operand values.
      if (TAIL_PIPE != 0 && state == ST_OUT_SHIFT) begin
        for (fsm_lane_i = 0; fsm_lane_i < MP; fsm_lane_i = fsm_lane_i + 1) begin
          out_oc = oc_group * MP + fsm_lane_i;
          if (out_oc < OC)
            v_tmp_q[fsm_lane_i] <= (scaled[fsm_lane_i] + out_round_q[fsm_lane_i])
                                   >>> sc_shift_q[fsm_lane_i];
        end
      end

      // [TAIL-PIPE2] ST_OUT_SAT: saturate into the staged output pixel.
      // Identical clip to legacy ST_OUTPUT; data_out is only sampled
      // downstream under valid_out, which ST_OUT_SAT raises the same edge
      // the last bytes land (exact ST_OUTPUT contract).
      if (TAIL_PIPE != 0 && state == ST_OUT_SAT) begin
        for (fsm_lane_i = 0; fsm_lane_i < MP; fsm_lane_i = fsm_lane_i + 1) begin
          out_oc = oc_group * MP + fsm_lane_i;
          if (out_oc < OC)
            data_out[out_oc*8 +: 8] <=
                (v_tmp_q[fsm_lane_i] >  127) ?  8'sd127 :
                (v_tmp_q[fsm_lane_i] < -128) ? -8'sd128 : v_tmp_q[fsm_lane_i][7:0];
        end
      end

      // Accumulator clears LAST: textual-order parity with the original
      // single block (the case-statement clears overrode the accumulate).
      if (state == ST_IDLE && start_mac) begin
        for (fsm_lane_i = 0; fsm_lane_i < MP; fsm_lane_i = fsm_lane_i + 1)
          acc[fsm_lane_i] <= 0;
      end
      if (state == ST_OUTPUT && oc_group != OC_PASSES - 1) begin
        for (fsm_lane_i = 0; fsm_lane_i < MP; fsm_lane_i = fsm_lane_i + 1)
          acc[fsm_lane_i] <= 0;
      end
      // [TAIL-PIPE2] same per-oc_pass clear, fired from ST_OUT_SAT when the
      // tail is ON (ST_OUTPUT is unreachable then). Mutually exclusive with
      // the arm above; stays textually after the gated accumulate.
      if (TAIL_PIPE != 0 && state == ST_OUT_SAT && oc_group != OC_PASSES - 1) begin
        for (fsm_lane_i = 0; fsm_lane_i < MP; fsm_lane_i = fsm_lane_i + 1)
          acc[fsm_lane_i] <= 0;
      end
    end"""),

# HUNK G -- FSM block: ST_SCALE next-state select (elaboration constant) -------
("""                ST_SCALE: begin
                    // [K1-FDCE] scaled[] writes moved to Block A (sync-only).
                    state <= ST_OUTPUT;
                end""",
 """                ST_SCALE: begin
                    // [K1-FDCE] scaled[] writes moved to Block A (sync-only).
                    // [TAIL-PIPE2] elaboration-constant select: ON -> split tail.
                    state <= (TAIL_PIPE != 0) ? ST_OUT_ROUND : ST_OUTPUT;
                end"""),

# HUNK H -- FSM block: the three tail arms, control only. For TAIL_PIPE==0
# every arm collapses to `state <= ST_IDLE` == the pre-patch `default:`
# recovery for encodings 5/6/7 (byte- and netlist-identical after pruning). ----
("""                default: state <= ST_IDLE;
            endcase""",
 """                // [TAIL-PIPE2] split requant tail -- control only (all data
                // writes live in Block A, gated on these same states).
                // ST_OUT_SAT carries the EXACT advance/finish control the
                // legacy ST_OUTPUT arm performs (oc_group/k_group/valid_out);
                // acc clears are Block A's (sync), as in K1 ST_OUTPUT.
                ST_OUT_ROUND: state <= (TAIL_PIPE != 0) ? ST_OUT_SHIFT : ST_IDLE;

                ST_OUT_SHIFT: state <= (TAIL_PIPE != 0) ? ST_OUT_SAT : ST_IDLE;

                ST_OUT_SAT: begin
                    if (TAIL_PIPE == 0) begin
                        state <= ST_IDLE;   // unreachable; old default: parity
                    end else if (oc_group == OC_PASSES - 1) begin
                        valid_out <= 1'b1;
                        state     <= ST_IDLE;
                    end else begin
                        oc_group <= oc_group + 1'b1;
                        k_group  <= 0;
                        state    <= ST_MAC;
                    end
                end

                default: state <= ST_IDLE;
            endcase"""),
]

NODE_OLD = "conv_datapath_mp_k #(.DSP_INPUT_PIPE(1),"
NODE_NEW = "conv_datapath_mp_k #(.DSP_INPUT_PIPE(1),.TAIL_PIPE(1),"

# --------------------------------------------------------------------------
# nn2rtl_top.v: skip-FIFO margin restoration required by the +2/oc_pass tail.
# EMPIRICAL (e2e forensics 2026-06-09/10, see TAIL_PIPE2_ANALYSIS.md §5): the
# tail-piped chain wedged because relu_9's LAST output beat of the frame was
# silently LOST at its dual-ready fork {skid218, skid224}: the narrow-relu
# streamer presents each pixel's FINAL beat for exactly ONE cycle (in the
# !sending state valid_out is dropped unconditionally the next cycle), so any
# fork-not-ready on that precise cycle discards the beat (B20 bug class --
# the LATENT defect that also underlies the B22/B23 "rate change wedges the
# joins" findings). The +2/oc_pass tail shifted the cadence so skid218/224
# were momentarily full on relu_9's frame-final-beat cycle. Deepening both
# fork receivers so they are never full when a last beat is offered restores
# the alignment margin. Depth is value-neutral (lossless order-preserving
# FIFO -> byte-exact); fit-safe (DEPTH>=512 maps to URAM, ~16% used;
# +~1.5 URAM total).
# --------------------------------------------------------------------------
TOP_EDITS = [
("""    skip_fifo #(.WIDTH(256), .DEPTH(512)) u_skid_node_conv_224 (""",
 """    // [TAIL-PIPE2-FIFO] 512->1024: fork-receiver margin for relu_9's one-cycle
    // last-beat offers (B20-class loss; see TAIL_PIPE2_ANALYSIS.md #1/#5).
    skip_fifo #(.WIDTH(256), .DEPTH(1024)) u_skid_node_conv_224 ("""),

("""    skip_fifo #(.WIDTH(256), .DEPTH(128)) u_skid_node_conv_218 (""",
 """    // [TAIL-PIPE2-FIFO] 128->1024: relu_9's dual-ready fork {skid218, skid224} must
    // never be un-ready on a cycle relu_9 presents a pixel's LAST beat (the narrow-
    // relu streamer offers it for exactly ONE cycle -- a not-ready fork silently
    // LOSES it; e2e forensics 2026-06-10 caught beat 25088/25088 vanish here).
    // The end-of-frame burst through add_2 is bounded by skip_node_add_2's depth
    // (512) + in-flight, so 1024 makes skid218 always-ready by construction.
    skip_fifo #(.WIDTH(256), .DEPTH(1024)) u_skid_node_conv_218 ("""),
]

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def read_norm(path):
    """bytes -> latin-1 text with LF endings; returns (text, had_crlf)."""
    with open(path, "rb") as f:
        raw = f.read()
    crlf = b"\r\n" in raw
    text = raw.decode("latin-1")
    if crlf:
        text = text.replace("\r\n", "\n")
    return text, crlf


def write_norm(path, text, crlf):
    out = text.replace("\n", "\r\n") if crlf else text
    with open(path, "wb") as f:
        f.write(out.encode("latin-1"))


def patch(path, edits, bk, marker):
    name = os.path.basename(path)
    text, crlf = read_norm(path)
    if marker in text:
        print(f"SKIP {name}: already patched ({marker} present)")
        return True
    work = text
    for i, (old, new) in enumerate(edits):
        c = work.count(old)
        if c != 1:
            print(f"FAIL {name} hunk#{i}: anchor count={c} (need exactly 1): {old[:64]!r}")
            return False
    if DRY_RUN:
        print(f"DRY  {name}: all {len(edits)} anchor(s) match; would patch")
        return True
    for old, new in edits:
        work = work.replace(old, new, 1)
    os.makedirs(bk, exist_ok=True)   # lazy: no empty backup dir on all-SKIP runs
    shutil.copy2(path, os.path.join(bk, name))
    write_norm(path, work, crlf)
    print(f"OK   {name}: applied {len(edits)} hunk(s)")
    return True


def live_spatial_mpk_convs():
    """node_conv files that (a) instantiate conv_datapath_mp_k and (b) are
    instantiated in nn2rtl_top.v. Returns (enable_list, dead_list)."""
    top_text, _ = read_norm(TOP)
    enable, dead = [], []
    for f in sorted(glob.glob(NODE_GLOB)):
        text, _ = read_norm(f)
        if "conv_datapath_mp_k" not in text:
            continue
        mod = os.path.splitext(os.path.basename(f))[0]
        if re.search(r"\b%s\s+u_%s\b" % (re.escape(mod), re.escape(mod)), top_text):
            enable.append(f)
        else:
            dead.append(mod)
    return enable, dead


def main():
    # ---- compute + assert the enablement set -------------------------------
    enable, dead = live_spatial_mpk_convs()
    names = [os.path.splitext(os.path.basename(f))[0] for f in enable]

    if STEM not in names:
        print(f"FAIL: expected stem {STEM} to be live in nn2rtl_top.v (anchor drift?)")
        sys.exit(1)
    enable = [f for f in enable if STEM not in os.path.basename(f)]
    names.remove(STEM)

    # Hard asserts: the engine-dispatched files must NOT be in the live set,
    # and the live set must be exactly the 35 known spatial convs (anchor-
    # asserted against top drift; update BOTH lists deliberately if the top
    # legitimately changes).
    EXPECTED = ["node_conv_%d" % n for n in
                (198, 200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 220,
                 222, 224, 226, 228, 230, 232, 234, 236, 238, 240, 242, 244,
                 248, 252, 256, 258, 262, 268, 270, 274, 276, 280, 288)]
    EXPECTED_DEAD = ["node_conv_%d" % n for n in
                     (246, 254, 260, 266, 272, 278, 284, 292, 298)]
    if names != EXPECTED:
        print("FAIL: live spatial mp_k conv set != expected 35.")
        print("  got     :", names)
        print("  expected:", EXPECTED)
        sys.exit(1)
    if sorted(dead) != sorted(EXPECTED_DEAD):
        print("FAIL: engine-dispatched (dead) mp_k file set != expected 9.")
        print("  got     :", sorted(dead))
        print("  expected:", sorted(EXPECTED_DEAD))
        sys.exit(1)
    print(f"live spatial mp_k convs to enable: {len(names)} "
          f"(stem {STEM} excluded; {len(dead)} engine-dispatched files untouched)")

    # ---- patch --------------------------------------------------------------
    bk = os.path.join(REPO, "backups", f"resnet_tail_pipe2_{time.strftime('%Y%m%d_%H%M%S')}")

    ok = patch(DP, DP_EDITS, bk, "[TAIL-PIPE2]")
    if not ok:
        print("-" * 60)
        print("FAILED in conv_datapath_mp_k.v -- aborting (no node edits made)")
        sys.exit(1)

    n_nodes = 0
    for f in enable:
        n_nodes += 1
        ok &= patch(f, [(NODE_OLD, NODE_NEW)], bk, ".TAIL_PIPE(")

    # nn2rtl_top.v skip-FIFO margin bump (see TOP_EDITS rationale above).
    ok &= patch(TOP, TOP_EDITS, bk, "[TAIL-PIPE2-FIFO]")

    print("-" * 60)
    print(f"node_conv wrappers enabled (.TAIL_PIPE(1)): {n_nodes}")
    if not DRY_RUN:
        print(f"backups -> {bk}")
    print(("DRY-RUN " if DRY_RUN else "") + ("ALL OK" if ok else "FAILED (see above)"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
