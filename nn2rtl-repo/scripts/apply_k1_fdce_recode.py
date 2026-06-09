#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_k1_fdce_recode.py -- K1: async-reset -> no-reset recode of DATAPATH-ONLY
registers in the ResNet-50 nn2rtl design (FDCE -> FDRE on UltraScale+).

WHY (Fmax/congestion): ~73.5% of the design's ~1.3M FFs are async-reset FDCE
whose reset value is provably DEAD (written-before-read every frame, or only
consumed under reset-gated valid/control bits). On UltraScale+ this fragments
control sets, blocks SRL/RAM packing, and fans rst_n out to ~960K loads -- a
major contributor to the 96.7% CLB-packing congestion at only ~67% LUT.

WHAT: converts the SAFE datapath register classes from the async-reset always
block to a separate no-reset "Block A" (the established repo pattern -- see
node_relu.v / node_add_1.v / node_add_5.v "Block A: array/data writes
(sync-only)" vs "Block B: control FSM"). ALL control/FSM/handshake/valid/ptr
registers KEEP their async resets, untouched.

BYTE-EXACTNESS ARGUMENT (per class, details in
docs/agent_tasks/K1_FDCE_RECODE_ANALYSIS.md):
  * e2e sim runs Verilator --x-initial 0 (= FPGA power-on zeros): every recoded
    register starts at 0 in BOTH sim and HW (Vivado INIT default 0), exactly the
    old reset value.
  * During the (single, t=0) rst_n assertion window no recoded register can be
    written, because every write-enable traces to control registers that are
    still async-reset-held (valid_in chains, FSM states, sched_advance,
    start_pulse, skid_valid, mac_valid_q*, ...).
  * Therefore the machine state at reset release is bit-identical to the
    pre-K1 design; by induction every subsequent cycle is identical.
  * Defense in depth: each recoded register is also written-before-read every
    frame/pixel/OC-pass, or only sampled under a reset-kept valid bit.

MECHANICS:
  * anchor-asserted: every edit is an exact-string match that must occur
    EXACTLY ONCE at its point in the edit sequence; any drift aborts the whole
    run BEFORE any file is written (two-phase validate-then-commit).
  * idempotent: files already containing the [K1-FDCE] marker are skipped.
  * backups: <file>.prek1 written once (never overwritten on re-runs).
  * --dry-run: full validation + per-file register listing, no writes.

USAGE:
  python scripts/apply_k1_fdce_recode.py --dry-run
  python scripts/apply_k1_fdce_recode.py
  python scripts/apply_k1_fdce_recode.py --repo-root <path>   # e.g. a sandbox copy

Scope deliberately EXCLUDED (see analysis doc): node_conv_196 wrapper (special
fixed shift-reg streamer), relu/add data_out output regs, add MAC pipes
(s1/s2/s3, lhs_term/...), node_max_pool2d pixel regs, coord_scheduler,
nn2rtl_scheduler, address_generator/config_register_block, all valid/ready/
state/counter/pointer bits everywhere.
"""

import argparse
import sys
from pathlib import Path

MARKER = "[K1-FDCE]"

# ---------------------------------------------------------------------------
# Edit primitives
# ---------------------------------------------------------------------------

class Edit:
    """One exact-string replacement. old must occur exactly once when applied."""
    def __init__(self, desc, old, new):
        self.desc = desc
        self.old = old
        self.new = new

class FilePatch:
    def __init__(self, relpath, regs, ff_estimate, edits):
        self.relpath = relpath          # repo-relative, forward slashes
        self.regs = regs                # registers moved out of async reset
        self.ff_estimate = ff_estimate  # FFs moved off rst_n (per file/class)
        self.edits = edits

# ---------------------------------------------------------------------------
# P1: rtl_library/line_buf_window.v  -- window[][][] + bypass_reg[]
#     (38 instances: 35 spatial conv wrappers + conv_196 stem + maxpool;
#      window 37,872 FF + bypass 90,648 FF = 128,520 FF total)
# Safety: node wrappers pulse frame_start (start_pulse) on the FIRST cycle
# after reset release (ST_ARM), which sync-clears window/bypass before any
# pixel is shifted in; during reset, frame_start/sched_advance are held 0 by
# the (still async-reset) wrapper FSM + coord_scheduler, so no write fires.
# ---------------------------------------------------------------------------

P1 = FilePatch(
    "rtl_library/line_buf_window.v",
    regs=["window[0:KH-1][0:KW-2][0:IC-1] (8b cells)", "bypass_reg[0:IC-1] (8b cells)"],
    ff_estimate=128520,
    edits=[Edit(
        "window/bypass_reg: async reset clause removed (frame_start sync-clear kept)",
        """    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < KH; i = i + 1)
                for (j = 0; j < KW - 1; j = j + 1)
                    for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1)
                        window[i][j][c_ch] <= 8'sd0;
            for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1)
                bypass_reg[c_ch] <= 8'sd0;
        end else if (frame_start) begin
""",
        """    // [K1-FDCE] window/bypass_reg are DATAPATH-only: the node wrapper pulses
    // frame_start (start_pulse) on the first post-reset cycle (ST_ARM), which
    // sync-clears them below before any pixel is shifted in, and during reset
    // both enables (frame_start, sched_advance) are held 0 by the still-async-
    // reset wrapper FSM + coord_scheduler -> the dropped reset clause is dead.
    // FDCE -> FDRE: removes the largest per-conv rst_n fanout + control sets.
    always @(posedge clk) begin
        if (frame_start) begin
""",
    )],
)

# ---------------------------------------------------------------------------
# P2: rtl_library/conv_datapath_mp_k.v -- 36 instances (35 wrappers + conv_196)
#     partial_q/acc/biased/scaled (71,400 FF) + data_out (116,224 FF)
# Safety: acc is sync-cleared (ST_IDLE&start_mac / ST_OUTPUT oc-advance) BEFORE
# the first mac_valid_q2-gated accumulate of every pass; biased/scaled/data_out
# follow strict write(STn) -> read(STn+1) ordering; data_out is only sampled
# downstream under valid_out (kept async-reset). All write gates (state,
# mac_valid_q2, start_mac) are reset-held control, so no write during reset.
# The clears are placed AFTER the accumulate in Block A to preserve the
# original last-write-wins NBA ordering of the single block.
# Loop vars fsm_i/fsm_lane_i and temps bias_oc/sc_oc/out_oc/out_shift/
# out_round/v_tmp become Block-A-ONLY after the move (the FSM block no longer
# references them) -> no cross-always shared-variable race (the race class
# documented at the ld_i/cs_lane_i declarations).
# ---------------------------------------------------------------------------

_MPK_BLOCK_A = """    // [K1-FDCE] Block A: DATAPATH registers (sync-only, no reset). Moved out of
    // the async-reset FSM block below. Safety: acc is sync-cleared on
    // ST_IDLE&start_mac / ST_OUTPUT oc-advance BEFORE the first gated
    // accumulate of every pass; biased/scaled/data_out are written in
    // ST_BIAS/ST_SCALE/ST_OUTPUT strictly before their readers; data_out is
    // only sampled downstream under valid_out (still async-reset). All write
    // gates (state, mac_valid_q2, mac_oc_group_q2, start_mac) remain
    // async-reset control in the FSM block, so no write fires during reset.
    // fsm_i/fsm_lane_i/bias_oc/sc_oc/out_oc/out_shift/out_round/v_tmp are
    // referenced ONLY by this block after the move (no shared-var race).
    always @(posedge clk) begin
      // Stage 2: register the MP partial sums.
      for (fsm_i = 0; fsm_i < MP; fsm_i = fsm_i + 1)
        partial_q[fsm_i] <= sum_lane_w[fsm_i];

      // Stage 3: accumulate partial sums into MP lanes.
      if (mac_valid_q2) begin
        for (fsm_i = 0; fsm_i < MP; fsm_i = fsm_i + 1) begin
          if (mac_oc_group_q2 * MP + fsm_i < OC)
            acc[fsm_i] <= acc[fsm_i] + $signed(partial_q[fsm_i]);
        end
      end

      // ST_BIAS: bias-add per lane.
      if (state == ST_BIAS) begin
        for (fsm_lane_i = 0; fsm_lane_i < MP; fsm_lane_i = fsm_lane_i + 1) begin
          bias_oc = oc_group * MP + fsm_lane_i;
          if (bias_oc < OC)
            biased[fsm_lane_i] <= $signed(acc[fsm_lane_i]) + $signed(biases[bias_oc]);
          else
            biased[fsm_lane_i] <= 0;
        end
      end

      // ST_SCALE: per-OC scale multiply.
      if (state == ST_SCALE) begin
        for (fsm_lane_i = 0; fsm_lane_i < MP; fsm_lane_i = fsm_lane_i + 1) begin
          sc_oc = oc_group * MP + fsm_lane_i;
          if (sc_oc < OC)
            scaled[fsm_lane_i] <= $signed(biased[fsm_lane_i]) *
                                  $signed(scale_rom[sc_oc][15:0]);
          else
            scaled[fsm_lane_i] <= 0;
        end
      end

      // ST_OUTPUT: per-OC round/shift/saturate into the staged output pixel.
      if (state == ST_OUTPUT) begin
        for (fsm_lane_i = 0; fsm_lane_i < MP; fsm_lane_i = fsm_lane_i + 1) begin
          out_oc = oc_group * MP + fsm_lane_i;
          if (out_oc < OC) begin
            out_shift = scale_rom[out_oc][21:16];
            out_round = (out_shift == 6'd0) ? {SCALED_W{1'b0}}
                      : ({{(SCALED_W-1){1'b0}}, 1'b1} <<< (out_shift - 6'd1));
            v_tmp = (scaled[fsm_lane_i] + out_round) >>> out_shift;
            data_out[out_oc*8 +: 8] <=
                (v_tmp >  127) ?  8'sd127 :
                (v_tmp < -128) ? -8'sd128 : v_tmp[7:0];
`ifdef DBG_SCALE
            if ((out_oc == 0 || out_oc == 32) && dbg_n < 8) begin
                $display("[DBG_SCALE] oc=%0d biased=%0d scale_rom=%h mult=%0d shift=%0d out_round=%0d scaled=%0d v_tmp=%0d -> out=%0d",
                    out_oc, $signed(biased[fsm_lane_i]), scale_rom[out_oc],
                    $signed({1'b0,scale_rom[out_oc][15:0]}), out_shift, $signed(out_round),
                    $signed(scaled[fsm_lane_i]), $signed(v_tmp),
                    $signed((v_tmp>127)?8'sd127:(v_tmp<-128)?-8'sd128:v_tmp[7:0]));
                dbg_n = dbg_n + 1;
            end
`endif
          end
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
    end

"""

P2 = FilePatch(
    "rtl_library/conv_datapath_mp_k.v",
    regs=["data_out[OC*8]", "acc[0:MP-1]", "biased[0:MP-1]", "scaled[0:MP-1]",
          "partial_q[0:MP-1]"],
    ff_estimate=187624,
    edits=[
        Edit("reset clause: drop data_out",
             """        if (!rst_n) begin
            state            <= ST_IDLE;
            valid_out        <= 1'b0;
            data_out         <= {OC*8{1'b0}};
            k_group          <= 0;
""",
             """        if (!rst_n) begin
            state            <= ST_IDLE;
            valid_out        <= 1'b0;
            k_group          <= 0;
"""),
        Edit("reset clause: drop acc/biased/scaled/partial_q lane-clear loop",
             """            mac_done_issuing <= 1'b0;
            for (fsm_i = 0; fsm_i < MP; fsm_i = fsm_i + 1) begin
                acc[fsm_i]      <= 0;
                biased[fsm_i]   <= 0;
                scaled[fsm_i]   <= 0;
                partial_q[fsm_i] <= 0;
            end
        end else begin
""",
             """            mac_done_issuing <= 1'b0;
        end else begin
"""),
        Edit("FSM body: remove stage-2 partial_q register (moved to Block A)",
             """            valid_out <= 1'b0;

            // Stage 2: register the MP partial sums.
            for (fsm_i = 0; fsm_i < MP; fsm_i = fsm_i + 1)
                partial_q[fsm_i] <= sum_lane_w[fsm_i];
""",
             """            valid_out <= 1'b0;
"""),
        Edit("FSM body: remove stage-3 accumulate (moved to Block A)",
             """            // Stage 3: accumulate partial sums into MP lanes.
            if (mac_valid_q2) begin
                for (fsm_i = 0; fsm_i < MP; fsm_i = fsm_i + 1) begin
                    if (mac_oc_group_q2 * MP + fsm_i < OC)
                        acc[fsm_i] <= acc[fsm_i] + $signed(partial_q[fsm_i]);
                end
            end

""",
             ""),
        Edit("ST_IDLE: remove acc clear (moved to Block A)",
             """                        mac_done_issuing <= 1'b0;
                        for (fsm_lane_i = 0; fsm_lane_i < MP; fsm_lane_i = fsm_lane_i + 1)
                            acc[fsm_lane_i] <= 0;
                    end
""",
             """                        mac_done_issuing <= 1'b0;
                    end
"""),
        Edit("ST_BIAS: remove biased writes (moved to Block A)",
             """                ST_BIAS: begin
                    for (fsm_lane_i = 0; fsm_lane_i < MP; fsm_lane_i = fsm_lane_i + 1) begin
                        bias_oc = oc_group * MP + fsm_lane_i;
                        if (bias_oc < OC)
                            biased[fsm_lane_i] <= $signed(acc[fsm_lane_i]) + $signed(biases[bias_oc]);
                        else
                            biased[fsm_lane_i] <= 0;
                    end
                    state <= ST_SCALE;
                end
""",
             """                ST_BIAS: begin
                    // [K1-FDCE] biased[] writes moved to Block A (sync-only).
                    state <= ST_SCALE;
                end
"""),
        Edit("ST_SCALE: remove scaled writes (moved to Block A)",
             """                ST_SCALE: begin
                    for (fsm_lane_i = 0; fsm_lane_i < MP; fsm_lane_i = fsm_lane_i + 1) begin
                        sc_oc = oc_group * MP + fsm_lane_i;
                        // Per-OC mult (positive 15-bit in a 16-bit slot -> signed
                        // positive). out-of-range lanes don't matter (OUTPUT guards).
                        if (sc_oc < OC)
                            scaled[fsm_lane_i] <= $signed(biased[fsm_lane_i]) *
                                                  $signed(scale_rom[sc_oc][15:0]);
                        else
                            scaled[fsm_lane_i] <= 0;
                    end
                    state <= ST_OUTPUT;
                end
""",
             """                ST_SCALE: begin
                    // [K1-FDCE] scaled[] writes moved to Block A (sync-only).
                    state <= ST_OUTPUT;
                end
"""),
        Edit("ST_OUTPUT: remove data_out/acc writes (moved to Block A)",
             """                ST_OUTPUT: begin
                    for (fsm_lane_i = 0; fsm_lane_i < MP; fsm_lane_i = fsm_lane_i + 1) begin
                        out_oc = oc_group * MP + fsm_lane_i;
                        if (out_oc < OC) begin
                            // Per-OC shift + round bias (shift==0 -> no rounding).
                            out_shift = scale_rom[out_oc][21:16];
                            out_round = (out_shift == 6'd0) ? {SCALED_W{1'b0}}
                                      : ({{(SCALED_W-1){1'b0}}, 1'b1} <<< (out_shift - 6'd1));
                            v_tmp = (scaled[fsm_lane_i] + out_round) >>> out_shift;
                            data_out[out_oc*8 +: 8] <=
                                (v_tmp >  127) ?  8'sd127 :
                                (v_tmp < -128) ? -8'sd128 : v_tmp[7:0];
`ifdef DBG_SCALE
                            if ((out_oc == 0 || out_oc == 32) && dbg_n < 8) begin
                                $display("[DBG_SCALE] oc=%0d biased=%0d scale_rom=%h mult=%0d shift=%0d out_round=%0d scaled=%0d v_tmp=%0d -> out=%0d",
                                    out_oc, $signed(biased[fsm_lane_i]), scale_rom[out_oc],
                                    $signed({1'b0,scale_rom[out_oc][15:0]}), out_shift, $signed(out_round),
                                    $signed(scaled[fsm_lane_i]), $signed(v_tmp),
                                    $signed((v_tmp>127)?8'sd127:(v_tmp<-128)?-8'sd128:v_tmp[7:0]));
                                dbg_n = dbg_n + 1;
                            end
`endif
                        end
                    end

                    if (oc_group == OC_PASSES - 1) begin
                        valid_out <= 1'b1;
                        state     <= ST_IDLE;
                    end else begin
                        oc_group     <= oc_group + 1'b1;
                        k_group      <= 0;
                        for (fsm_lane_i = 0; fsm_lane_i < MP; fsm_lane_i = fsm_lane_i + 1)
                            acc[fsm_lane_i] <= 0;
                        state <= ST_MAC;
                    end
                end
""",
             """                ST_OUTPUT: begin
                    // [K1-FDCE] data_out[]/acc[] writes moved to Block A (sync-only).
                    if (oc_group == OC_PASSES - 1) begin
                        valid_out <= 1'b1;
                        state     <= ST_IDLE;
                    end else begin
                        oc_group     <= oc_group + 1'b1;
                        k_group      <= 0;
                        state <= ST_MAC;
                    end
                end
"""),
        Edit("insert Block A before the FSM block",
             """    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= ST_IDLE;
""",
             _MPK_BLOCK_A + """    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= ST_IDLE;
"""),
    ],
)

# ---------------------------------------------------------------------------
# P3: output/rtl/engine/requant_pipeline.v -- scale_q1/q2 (16,384 FF) +
#     256 lanes x 116 FF pipes (29,696 FF) = 46,080 FF
# Safety: pure feed-forward pipes; data_out is sampled by the engine only
# under valid_out (4-deep valid chain KEPT async-reset). scale_q1/q2 are
# written every cycle (reset value dead after 2 clocks).
# ---------------------------------------------------------------------------

P3 = FilePatch(
    "output/rtl/engine/requant_pipeline.v",
    regs=["scale_q1[8191:0]", "scale_q2[8191:0]",
          "per-lane: biased_q1, scaled_q2, sat_hi_q3a, sat_lo_q3a, v_low_q3a, data_out_q4 (x256)"],
    ff_estimate=46080,
    edits=[
        Edit("split scale_q1/q2 out of the valid-chain block (valids keep reset)",
             """    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_q1       <= 1'b0;
            valid_q2       <= 1'b0;
            valid_q3       <= 1'b0;
            valid_out      <= 1'b0;
            scale_q1       <= {8192{1'b0}};
            scale_q2       <= {8192{1'b0}};
        end else begin
            valid_q1       <= valid_in;
            valid_q2       <= valid_q1;
            valid_q3       <= valid_q2;    // Lever 2: extra stage
            valid_out      <= valid_q3;    // Lever 2: shifted from valid_q2
            scale_q1       <= scale_in;
            scale_q2       <= scale_q1;
        end
    end
""",
             """    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_q1       <= 1'b0;
            valid_q2       <= 1'b0;
            valid_q3       <= 1'b0;
            valid_out      <= 1'b0;
        end else begin
            valid_q1       <= valid_in;
            valid_q2       <= valid_q1;
            valid_q3       <= valid_q2;    // Lever 2: extra stage
            valid_out      <= valid_q3;    // Lever 2: shifted from valid_q2
        end
    end

    // [K1-FDCE] scale_q1/scale_q2 are DATAPATH pipes (2 x 8192 FF): their
    // values reach data_out only through the per-lane pipeline, which the
    // engine samples strictly under valid_out (reset-gated above), and they
    // are rewritten every cycle -> the reset value is dead. No-reset => FDRE.
    always @(posedge clk) begin
        scale_q1       <= scale_in;
        scale_q2       <= scale_q1;
    end
"""),
        Edit("per-lane pipe: drop reset clause",
             """            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    biased_q1   <= {BIASED_W{1'b0}};
                    scaled_q2   <= {SCALED_W{1'b0}};
                    sat_hi_q3a  <= 1'b0;
                    sat_lo_q3a  <= 1'b0;
                    v_low_q3a   <= 8'sd0;
                    data_out_q4 <= 8'sd0;
                end else begin
""",
             """            // [K1-FDCE] per-lane requant pipe (116 FF x 256 lanes): pure
            // feed-forward datapath; data_out is sampled only under valid_out
            // (reset-gated valid chain above). Reset clause removed -> FDRE.
            always @(posedge clk) begin
                begin
"""),
    ],
)

# ---------------------------------------------------------------------------
# P4: output/rtl/engine/mac_array.v -- 256 x acc[31:0] = 8,192 FF
# Safety: the engine FSM pulses mac_clear on EVERY ST_RUN entry (run_entered
# in shared_engine_skeleton.v), so acc is sync-cleared before the first
# mac_valid_q1-gated accumulate of every dot product, including the first
# after power-on (mac_valid_q1 is reset-held 0 until then). Reset value dead.
# ---------------------------------------------------------------------------

P4 = FilePatch(
    "output/rtl/engine/mac_array.v",
    regs=["acc[31:0] (x256 lanes)"],
    ff_estimate=8192,
    edits=[Edit(
        "acc: async reset dropped (mac_clear sync-clear is the live initializer)",
        """            always @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    acc <= 32'sd0;
                else if (mac_clear)
                    acc <= 32'sd0;
                else if (mac_valid_q1)
                    acc <= acc + $signed(mul_q1);
            end
""",
        """            // [K1-FDCE] acc's async reset is dead: the engine FSM pulses
            // mac_clear on EVERY ST_RUN entry (run_entered), so acc is sync-
            // cleared before the first gated accumulate of every dot product
            // (incl. the first after power-on; mac_valid_q1 is reset-held 0
            // until then). FDCE -> FDRE on 256 x 32 accumulator bits.
            always @(posedge clk) begin
                if (mac_clear)
                    acc <= 32'sd0;
                else if (mac_valid_q1)
                    acc <= acc + $signed(mul_q1);
            end
""",
    )],
)

# ---------------------------------------------------------------------------
# P5: output/rtl/shared_engine_skeleton.v -- act_in_rd_data_d (2,048 FF)
# Safety: pure hold register; consumed (mac_act_byte) only when
# ag_act_in_rd_en_d2 / mac_valid_in are high (reset-held 0); rewritten every
# cycle. Reset value dead.
# ---------------------------------------------------------------------------

P5 = FilePatch(
    "output/rtl/shared_engine_skeleton.v",
    regs=["act_in_rd_data_d[ACT_BUS_W-1:0]"],
    ff_estimate=2048,
    edits=[Edit(
        "act_in_rd_data_d: split into a no-reset block (control delays keep reset)",
        """    reg [ACT_BUS_W-1:0] act_in_rd_data_d;   // activation word held one extra cycle
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ag_weight_rd_en_d        <= 1'b0;
            ag_weight_rd_en_d2       <= 1'b0;
            ag_act_in_rd_en_d        <= 1'b0;
            ag_act_in_rd_en_d2       <= 1'b0;
            ag_act_in_ic_byte_idx_d  <= 8'd0;
            ag_act_in_ic_byte_idx_d2 <= 8'd0;
            act_in_rd_data_d         <= {ACT_BUS_W{1'b0}};
        end else begin
            ag_weight_rd_en_d        <= ag_weight_rd_en;
            ag_weight_rd_en_d2       <= ag_weight_rd_en_d;
            ag_act_in_rd_en_d        <= ag_act_in_rd_en;
            ag_act_in_rd_en_d2       <= ag_act_in_rd_en_d;
            ag_act_in_ic_byte_idx_d  <= ag_act_in_ic_byte_idx;
            ag_act_in_ic_byte_idx_d2 <= ag_act_in_ic_byte_idx_d;
            act_in_rd_data_d         <= act_in_rd_data;  // read-N act (valid N+1) -> held at N+2
        end
    end
""",
        """    reg [ACT_BUS_W-1:0] act_in_rd_data_d;   // activation word held one extra cycle
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ag_weight_rd_en_d        <= 1'b0;
            ag_weight_rd_en_d2       <= 1'b0;
            ag_act_in_rd_en_d        <= 1'b0;
            ag_act_in_rd_en_d2       <= 1'b0;
            ag_act_in_ic_byte_idx_d  <= 8'd0;
            ag_act_in_ic_byte_idx_d2 <= 8'd0;
        end else begin
            ag_weight_rd_en_d        <= ag_weight_rd_en;
            ag_weight_rd_en_d2       <= ag_weight_rd_en_d;
            ag_act_in_rd_en_d        <= ag_act_in_rd_en;
            ag_act_in_rd_en_d2       <= ag_act_in_rd_en_d;
            ag_act_in_ic_byte_idx_d  <= ag_act_in_ic_byte_idx;
            ag_act_in_ic_byte_idx_d2 <= ag_act_in_ic_byte_idx_d;
        end
    end
    // [K1-FDCE] act_in_rd_data_d (2048b) is a DATAPATH hold register: it only
    // reaches the MAC when the (reset-held) ..._rd_en_d2 gates are high, and
    // it is rewritten every cycle -> reset value dead. No-reset => FDRE.
    always @(posedge clk) begin
        act_in_rd_data_d         <= act_in_rd_data;  // read-N act (valid N+1) -> held at N+2
    end
""",
    )],
)

# ---------------------------------------------------------------------------
# P6: output/rtl/nn2rtl_top.v helper modules
#   skip_fifo URAM out_data_r (33 deep insts x 256 = 8,448 FF)
#   engine_output_fifo out_data (2 x 2048 = 4,096 FF)
#   stream_to_act_bram_bridge wr_data/skid_data/accumulator/beat_buf
#     (17 insts, ~74K FF)
#   engine_output_bridge beat_buf/data_out (17 insts, ~39K FF)
# Safety: all are stream-data registers consumed ONLY under reset-kept
# control/valid bits (out_valid_r, out_valid, wr_req, skid_valid, buf_valid,
# valid_out); writes are gated by the same reset-held controls.
# ---------------------------------------------------------------------------

P6 = FilePatch(
    "output/rtl/nn2rtl_top.v",
    regs=["skip_fifo.g_uram_fifo.out_data_r (x33 deep insts)",
          "engine_output_fifo.out_data (x2)",
          "stream_to_act_bram_bridge.{wr_data, skid_data, accumulator, beat_buf} (x17)",
          "engine_output_bridge.{beat_buf, data_out} (x17)"],
    ff_estimate=125696,
    edits=[
        # --- skip_fifo g_uram_fifo ---
        Edit("skip_fifo(URAM): out_data_r -> sync-only block",
             """        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                wr_ptr <= {(ADDR_W+1){1'b0}};
                rd_ptr <= {(ADDR_W+1){1'b0}};
                peak_occ <= {(ADDR_W+1){1'b0}};
                out_valid_r <= 1'b0;
                out_data_r  <= {WIDTH{1'b0}};
            end else begin
                if (in_valid && ~full) wr_ptr <= wr_ptr + 1'b1;
                if (do_rd) begin
                    out_data_r  <= mem[rd_idx];
                    out_valid_r <= 1'b1;
                    rd_ptr      <= rd_ptr + 1'b1;
                end else if (out_valid_r && out_ready) begin
                    out_valid_r <= 1'b0;
                end
                if (occ_now > peak_occ) peak_occ <= occ_now;
            end
        end
""",
             """        // [K1-FDCE] out_data_r is FIFO DATA: sampled downstream only under
        // out_valid_r (kept async-reset); written only under do_rd (gated by
        // reset-held pointers). Sync-only write, reset dropped -> FDRE.
        always @(posedge clk) begin
            if (do_rd) out_data_r <= mem[rd_idx];
        end
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                wr_ptr <= {(ADDR_W+1){1'b0}};
                rd_ptr <= {(ADDR_W+1){1'b0}};
                peak_occ <= {(ADDR_W+1){1'b0}};
                out_valid_r <= 1'b0;
            end else begin
                if (in_valid && ~full) wr_ptr <= wr_ptr + 1'b1;
                if (do_rd) begin
                    out_valid_r <= 1'b1;
                    rd_ptr      <= rd_ptr + 1'b1;
                end else if (out_valid_r && out_ready) begin
                    out_valid_r <= 1'b0;
                end
                if (occ_now > peak_occ) peak_occ <= occ_now;
            end
        end
"""),
        # --- engine_output_fifo ---
        Edit("engine_output_fifo: out_data -> sync-only block",
             """    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr    <= {(ADDR_W+1){1'b0}};
            rd_ptr    <= {(ADDR_W+1){1'b0}};
            out_valid <= 1'b0;
            out_data  <= {DATA_W{1'b0}};
        end else begin
            if (wr_fire) wr_ptr <= wr_ptr + 1'b1;
            // Output handshake: drop valid when consumer accepts.
            if (out_valid && out_ready) begin
                out_valid <= 1'b0;
            end
            // Refill output skid when it is empty (or being consumed
            // this cycle) and the FIFO has data. rd_ptr advances on the
            // same edge so the next refill reads the next entry.
            if (load_skid) begin
                out_data  <= mem[rd_ptr[ADDR_W-1:0]];
                out_valid <= 1'b1;
                rd_ptr    <= rd_ptr + 1'b1;
            end
        end
    end
""",
             """    // [K1-FDCE] out_data is FIFO DATA: sampled only under out_valid (kept
    // async-reset); written only under load_skid. Sync-only write -> FDRE.
    always @(posedge clk) begin
        if (load_skid) out_data <= mem[rd_ptr[ADDR_W-1:0]];
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr    <= {(ADDR_W+1){1'b0}};
            rd_ptr    <= {(ADDR_W+1){1'b0}};
            out_valid <= 1'b0;
        end else begin
            if (wr_fire) wr_ptr <= wr_ptr + 1'b1;
            // Output handshake: drop valid when consumer accepts.
            if (out_valid && out_ready) begin
                out_valid <= 1'b0;
            end
            // Refill output skid when it is empty (or being consumed
            // this cycle) and the FIFO has data. rd_ptr advances on the
            // same edge so the next refill reads the next entry.
            if (load_skid) begin
                out_valid <= 1'b1;
                rd_ptr    <= rd_ptr + 1'b1;
            end
        end
    end
"""),
        # --- stream_to_act_bram_bridge g_w_eq ---
        Edit("stream bridge g_w_eq: wr_data/skid_data -> Block A",
             """        assign in_ready = !loaded && (!skid_valid || drain_skid);
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                wr_req     <= 1'b0;
                wr_addr    <= 15'd0;
                wr_data    <= 2048'd0;
                word_count <= 16'd0;
                loaded     <= 1'b0;
                skid_valid <= 1'b0;
                skid_data  <= 2048'd0;
            end else begin
                // (1) Grant retires wr_req and advances count.
                if (wr_req && wr_grant) begin
                    wr_req <= 1'b0;
                    word_count <= next_word_count;
                    if (next_word_count == TOTAL_BRAM_WORDS) loaded <= 1'b1;
                end
                // (2) Drain skid into wr_req when bridge is free.
                if (drain_skid) begin
                    wr_req  <= 1'b1;
                    wr_addr <= next_wr_addr;
                    wr_data <= skid_data;
                end
                // (3) Capture new beat into skid; clear skid if drained and no new.
                if (in_valid && !loaded && (!skid_valid || drain_skid)) begin
                    skid_valid <= 1'b1;
                    skid_data  <= in_data;
                end else if (drain_skid) begin
                    skid_valid <= 1'b0;
                end
            end
        end
""",
             """        assign in_ready = !loaded && (!skid_valid || drain_skid);
        // [K1-FDCE] Block A: stream DATA regs (sync-only, no reset). wr_data is
        // consumed only while wr_req is pending; skid_data only while
        // skid_valid -- both controls stay async-reset below. Reset values dead.
        always @(posedge clk) begin
            if (drain_skid) wr_data <= skid_data;
            if (in_valid && !loaded && (!skid_valid || drain_skid))
                skid_data <= in_data;
        end
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                wr_req     <= 1'b0;
                wr_addr    <= 15'd0;
                word_count <= 16'd0;
                loaded     <= 1'b0;
                skid_valid <= 1'b0;
            end else begin
                // (1) Grant retires wr_req and advances count.
                if (wr_req && wr_grant) begin
                    wr_req <= 1'b0;
                    word_count <= next_word_count;
                    if (next_word_count == TOTAL_BRAM_WORDS) loaded <= 1'b1;
                end
                // (2) Drain skid into wr_req when bridge is free.
                if (drain_skid) begin
                    wr_req  <= 1'b1;
                    wr_addr <= next_wr_addr;
                end
                // (3) Capture new beat into skid; clear skid if drained and no new.
                if (in_valid && !loaded && (!skid_valid || drain_skid)) begin
                    skid_valid <= 1'b1;
                end else if (drain_skid) begin
                    skid_valid <= 1'b0;
                end
            end
        end
"""),
        # --- stream_to_act_bram_bridge g_w_lt ---
        Edit("stream bridge g_w_lt: accumulator/wr_data/skid_data -> Block A",
             """        assign in_ready = !loaded && (!skid_valid || drain_skid);
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                wr_req      <= 1'b0;
                wr_addr     <= 15'd0;
                wr_data     <= 2048'd0;
                word_count  <= 16'd0;
                accumulator <= 2048'd0;
                beat_idx    <= {(BEAT_W+1){1'b0}};
                loaded      <= 1'b0;
                skid_valid  <= 1'b0;
                skid_data   <= {BUS_W{1'b0}};
            end else begin
                // (1) Grant retires wr_req and advances count.
                if (wr_req && wr_grant) begin
                    wr_req <= 1'b0;
                    word_count <= next_word_count;
                    if (next_word_count == TOTAL_BRAM_WORDS) loaded <= 1'b1;
                end
                // (2) Drain skid into accumulator (and possibly emit word).
                if (drain_skid) begin
                    accumulator[beat_idx*BUS_W +: BUS_W] <= skid_data;
                    if (would_complete) begin
                        beat_idx <= {(BEAT_W+1){1'b0}};
                        wr_req   <= 1'b1;
                        wr_addr  <= next_wr_addr;
                        wr_data  <= { skid_data,
                                      accumulator[2048-BUS_W-1:0] };
                    end else begin
                        beat_idx <= beat_idx + 1'b1;
                    end
                end
                // (3) Capture new beat into skid.
                if (in_valid && !loaded && (!skid_valid || drain_skid)) begin
                    skid_valid <= 1'b1;
                    skid_data  <= in_data;
                end else if (drain_skid) begin
                    skid_valid <= 1'b0;
                end
            end
        end
""",
             """        assign in_ready = !loaded && (!skid_valid || drain_skid);
        // [K1-FDCE] Block A: stream DATA regs (sync-only, no reset). The
        // accumulator's consumed slices are all (re)written each word before
        // wr_data is formed; wr_data/skid_data are consumed only under
        // wr_req/skid_valid (async-reset control below). Reset values dead.
        always @(posedge clk) begin
            if (drain_skid) begin
                accumulator[beat_idx*BUS_W +: BUS_W] <= skid_data;
                if (would_complete)
                    wr_data <= { skid_data,
                                 accumulator[2048-BUS_W-1:0] };
            end
            if (in_valid && !loaded && (!skid_valid || drain_skid))
                skid_data <= in_data;
        end
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                wr_req      <= 1'b0;
                wr_addr     <= 15'd0;
                word_count  <= 16'd0;
                beat_idx    <= {(BEAT_W+1){1'b0}};
                loaded      <= 1'b0;
                skid_valid  <= 1'b0;
            end else begin
                // (1) Grant retires wr_req and advances count.
                if (wr_req && wr_grant) begin
                    wr_req <= 1'b0;
                    word_count <= next_word_count;
                    if (next_word_count == TOTAL_BRAM_WORDS) loaded <= 1'b1;
                end
                // (2) Drain skid (data writes moved to Block A above).
                if (drain_skid) begin
                    if (would_complete) begin
                        beat_idx <= {(BEAT_W+1){1'b0}};
                        wr_req   <= 1'b1;
                        wr_addr  <= next_wr_addr;
                    end else begin
                        beat_idx <= beat_idx + 1'b1;
                    end
                end
                // (3) Capture new beat into skid.
                if (in_valid && !loaded && (!skid_valid || drain_skid)) begin
                    skid_valid <= 1'b1;
                end else if (drain_skid) begin
                    skid_valid <= 1'b0;
                end
            end
        end
"""),
        # --- stream_to_act_bram_bridge g_w_gt ---
        Edit("stream bridge g_w_gt: beat_buf/wr_data/skid_data -> Block A",
             """        assign in_ready = !loaded && (!skid_valid || drain_skid);
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                wr_req     <= 1'b0;
                wr_addr    <= 15'd0;
                wr_data    <= 2048'd0;
                word_count <= 16'd0;
                beat_buf   <= {BUS_W{1'b0}};
                slice_idx  <= {(SLICE_W+1){1'b0}};
                buf_active <= 1'b0;
                loaded     <= 1'b0;
                skid_valid <= 1'b0;
                skid_data  <= {BUS_W{1'b0}};
            end else begin
                // (1) Grant retires wr_req and advances count.
                if (wr_req && wr_grant) begin
                    wr_req <= 1'b0;
                    word_count <= next_word_count;
                    if (next_word_count == TOTAL_BRAM_WORDS) loaded <= 1'b1;
                    if (slice_idx == WORDS_PER_BEAT - 1) begin
                        slice_idx  <= {(SLICE_W+1){1'b0}};
                        buf_active <= 1'b0;
                    end else if (next_word_count != TOTAL_BRAM_WORDS) begin
                        // Continue slicing the current beat (defensive guard
                        // on next_word_count so we don't overrun TOTAL_BRAM_WORDS
                        // when a frame ends mid-beat with a non-divisible total).
                        slice_idx <= slice_idx + 1'b1;
                        wr_req    <= 1'b1;
                        wr_addr   <= BRAM_BASE_ADDR[14:0] + next_word_count[14:0];
                        wr_data   <= beat_buf[(slice_idx+1)*2048 +: 2048];
                    end
                end
                // (2) Load new beat from skid when buf is free.
                if (drain_skid) begin
                    beat_buf   <= skid_data;
                    buf_active <= 1'b1;
                    slice_idx  <= {(SLICE_W+1){1'b0}};
                    wr_req     <= 1'b1;
                    wr_addr    <= next_wr_addr;
                    wr_data    <= skid_data[2047:0];
                end
                // (3) Capture new beat into skid.
                if (in_valid && !loaded && (!skid_valid || drain_skid)) begin
                    skid_valid <= 1'b1;
                    skid_data  <= in_data;
                end else if (drain_skid) begin
                    skid_valid <= 1'b0;
                end
            end
        end
""",
             """        assign in_ready = !loaded && (!skid_valid || drain_skid);
        // [K1-FDCE] Block A: stream DATA regs (sync-only, no reset), consumed
        // only under wr_req/buf_active/skid_valid (async-reset control below).
        // Textual order preserved: a drain_skid wr_data write overrides the
        // continue-slice write, exactly as in the original single block.
        always @(posedge clk) begin
            if (wr_req && wr_grant && (slice_idx != WORDS_PER_BEAT - 1)
                && (next_word_count != TOTAL_BRAM_WORDS))
                wr_data <= beat_buf[(slice_idx+1)*2048 +: 2048];
            if (drain_skid) begin
                beat_buf <= skid_data;
                wr_data  <= skid_data[2047:0];
            end
            if (in_valid && !loaded && (!skid_valid || drain_skid))
                skid_data <= in_data;
        end
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                wr_req     <= 1'b0;
                wr_addr    <= 15'd0;
                word_count <= 16'd0;
                slice_idx  <= {(SLICE_W+1){1'b0}};
                buf_active <= 1'b0;
                loaded     <= 1'b0;
                skid_valid <= 1'b0;
            end else begin
                // (1) Grant retires wr_req and advances count.
                if (wr_req && wr_grant) begin
                    wr_req <= 1'b0;
                    word_count <= next_word_count;
                    if (next_word_count == TOTAL_BRAM_WORDS) loaded <= 1'b1;
                    if (slice_idx == WORDS_PER_BEAT - 1) begin
                        slice_idx  <= {(SLICE_W+1){1'b0}};
                        buf_active <= 1'b0;
                    end else if (next_word_count != TOTAL_BRAM_WORDS) begin
                        // Continue slicing the current beat (defensive guard
                        // on next_word_count so we don't overrun TOTAL_BRAM_WORDS
                        // when a frame ends mid-beat with a non-divisible total).
                        slice_idx <= slice_idx + 1'b1;
                        wr_req    <= 1'b1;
                        wr_addr   <= BRAM_BASE_ADDR[14:0] + next_word_count[14:0];
                    end
                end
                // (2) Load new beat from skid when buf is free.
                if (drain_skid) begin
                    buf_active <= 1'b1;
                    slice_idx  <= {(SLICE_W+1){1'b0}};
                    wr_req     <= 1'b1;
                    wr_addr    <= next_wr_addr;
                end
                // (3) Capture new beat into skid.
                if (in_valid && !loaded && (!skid_valid || drain_skid)) begin
                    skid_valid <= 1'b1;
                end else if (drain_skid) begin
                    skid_valid <= 1'b0;
                end
            end
        end
"""),
        # --- engine_output_bridge ---
        Edit("engine_output_bridge: beat_buf/data_out -> Block A",
             """    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out      <= 1'b0;
            data_out       <= {DATA_W{1'b0}};
            beat_buf       <= {ACT_W{1'b0}};
            buf_valid      <= 1'b0;
            tile_idx       <= {(TILE_IDX_W+1){1'b0}};
            tiles_emitted  <= 32'd0;
            drain_complete <= 1'b0;
        end else begin
            // (1) Consumer accepted current tile — drop valid_out.
            if (valid_out && ready_out) valid_out <= 1'b0;
            // (2) Emit a tile this cycle.
            if (emit_ready) begin
                valid_out     <= 1'b1;
                data_out      <= current_tile;
                tiles_emitted <= tiles_emitted + 32'd1;
""",
             """    // [K1-FDCE] Block A: beat_buf/data_out are stream DATA (sync-only, no
    // reset): beat_buf is consumed (current_tile) only while buf_valid;
    // data_out is sampled downstream only under valid_out -- both controls
    // stay async-reset below. Reset values dead.
    always @(posedge clk) begin
        if (emit_ready) data_out <= current_tile;
        if (fifo_out_ready && fifo_out_valid) beat_buf <= fifo_out_data;
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out      <= 1'b0;
            buf_valid      <= 1'b0;
            tile_idx       <= {(TILE_IDX_W+1){1'b0}};
            tiles_emitted  <= 32'd0;
            drain_complete <= 1'b0;
        end else begin
            // (1) Consumer accepted current tile — drop valid_out.
            if (valid_out && ready_out) valid_out <= 1'b0;
            // (2) Emit a tile this cycle.
            if (emit_ready) begin
                valid_out     <= 1'b1;
                tiles_emitted <= tiles_emitted + 32'd1;
"""),
        Edit("engine_output_bridge: remove beat_buf write from control block",
             """            // (3) Pull next beat from FIFO. Placed AFTER the emit block so
            // a simultaneous "emit last tile + load new beat" lands with
            // buf_valid=1 (FIFO load wins the NBA race) and tile_idx=0.
            if (fifo_out_ready && fifo_out_valid) begin
                beat_buf  <= fifo_out_data;
                buf_valid <= 1'b1;
                tile_idx  <= {(TILE_IDX_W+1){1'b0}};
            end
""",
             """            // (3) Pull next beat from FIFO. Placed AFTER the emit block so
            // a simultaneous "emit last tile + load new beat" lands with
            // buf_valid=1 (FIFO load wins the NBA race) and tile_idx=0.
            // ([K1-FDCE] beat_buf data write moved to Block A above.)
            if (fifo_out_ready && fifo_out_valid) begin
                buf_valid <= 1'b1;
                tile_idx  <= {(TILE_IDX_W+1){1'b0}};
            end
"""),
    ],
)

# ---------------------------------------------------------------------------
# P7: spatial conv wrappers (35 files) -- in_lo (81,152 FF) + out_pix
#     (115,712 FF). node_conv_196 deliberately EXCLUDED (special stem wrapper).
# Safety: every in_lo slice [0..IN_BEATS-2] is written during each pixel's
# gather BEFORE the last-beat read of {data_in, in_lo} (in_beat_idx resets to
# 0 and walks up); out_pix is written the cycle out_busy is set, and data_out
# is only sampled downstream while valid_out (= out_busy, async-reset) is
# high. During reset, beat_fire/lib_valid_out_w trace to reset-held valids.
# ---------------------------------------------------------------------------

_WRAP_BLOCK_A = """    // [K1-FDCE] Block A: gather/stream DATAPATH regs (sync-only, no reset).
    // in_lo slices are all rewritten during each pixel's gather before the
    // last-beat read of {data_in, in_lo}; out_pix is written before out_busy
    // raises valid_out. Reset values dead; control stays async-reset below.
    always @(posedge clk) begin
        if (beat_fire && !is_last_in_beat)
            in_lo[in_beat_idx*TILE_BITS +: TILE_BITS] <= data_in;
        if (lib_valid_out_w && !out_busy)
            out_pix <= lib_data_out_w;
    end

"""

_WRAP_OUTPIX_OLD = """            if (lib_valid_out_w && !out_busy) begin
                out_pix  <= lib_data_out_w;
                out_idx  <= 0;
                out_busy <= 1'b1;
            end else if (out_busy && ready_out) begin
"""
_WRAP_OUTPIX_NEW = """            if (lib_valid_out_w && !out_busy) begin
                out_idx  <= 0;
                out_busy <= 1'b1;
            end else if (out_busy && ready_out) begin
"""

_WRAP_INSERT_ANCHOR = """    always @(posedge clk or negedge rst_n) begin
"""

def _wrapper_patch_variant_a(name):
    return FilePatch(
        "output/rtl/%s.v" % name,
        regs=["in_lo[IC*8-256-1:0]", "out_pix[OC*8-1:0]"],
        ff_estimate=None,  # aggregated in the class summary
        edits=[
            Edit("reset clause: drop in_lo / out_pix",
                 """            frame_state<=ST_ARM; start_pulse<=1'b0; in_beat_idx<=0; in_lo<=0;
            out_pix<=0; out_idx<=0; out_busy<=1'b0;
""",
                 """            frame_state<=ST_ARM; start_pulse<=1'b0; in_beat_idx<=0;
            out_idx<=0; out_busy<=1'b0;
"""),
            Edit("gather: remove in_lo write (moved to Block A)",
                 """            if (beat_fire) begin
                if (!is_last_in_beat)
                    in_lo[in_beat_idx*TILE_BITS +: TILE_BITS] <= data_in;
                in_beat_idx <= is_last_in_beat ? 0 : in_beat_idx + 1'b1;
            end
""",
                 """            if (beat_fire) begin
                in_beat_idx <= is_last_in_beat ? 0 : in_beat_idx + 1'b1;
            end
"""),
            Edit("streamer: remove out_pix write (moved to Block A)",
                 _WRAP_OUTPIX_OLD, _WRAP_OUTPIX_NEW),
            Edit("insert Block A before the control block",
                 _WRAP_INSERT_ANCHOR, _WRAP_BLOCK_A + _WRAP_INSERT_ANCHOR),
        ],
    )

def _wrapper_patch_variant_b(name):
    return FilePatch(
        "output/rtl/%s.v" % name,
        regs=["in_lo[IC*8-256-1:0]", "out_pix[OC*8-1:0]"],
        ff_estimate=None,
        edits=[
            Edit("reset clause: drop in_lo / out_pix",
                 """            in_beat_idx<=0; in_lo<=0; irow<=0; icol<=0;
            out_pix<=0; out_idx<=0; out_busy<=1'b0;
""",
                 """            in_beat_idx<=0; irow<=0; icol<=0;
            out_idx<=0; out_busy<=1'b0;
"""),
            Edit("gather: remove in_lo write (moved to Block A)",
                 """            if (beat_fire) begin
                if (!is_last_in_beat) begin
                    in_lo[in_beat_idx*TILE_BITS +: TILE_BITS] <= data_in;
                    in_beat_idx <= in_beat_idx + 1'b1;
                end else begin
""",
                 """            if (beat_fire) begin
                if (!is_last_in_beat) begin
                    in_beat_idx <= in_beat_idx + 1'b1;
                end else begin
"""),
            Edit("streamer: remove out_pix write (moved to Block A)",
                 _WRAP_OUTPIX_OLD, _WRAP_OUTPIX_NEW),
            Edit("insert Block A before the control block",
                 _WRAP_INSERT_ANCHOR, _WRAP_BLOCK_A + _WRAP_INSERT_ANCHOR),
        ],
    )

# The 33 variant-A (frame_state streamer) + 2 variant-B (irow/icol decimator)
# instantiated spatial conv wrappers. conv_196 is EXCLUDED (special wrapper).
_CONV_VARIANT_A = [
    "node_conv_198", "node_conv_200", "node_conv_202", "node_conv_204",
    "node_conv_206", "node_conv_208", "node_conv_210", "node_conv_212",
    "node_conv_214", "node_conv_216", "node_conv_218", "node_conv_220",
    "node_conv_222", "node_conv_226", "node_conv_228", "node_conv_230",
    "node_conv_232", "node_conv_234", "node_conv_236", "node_conv_238",
    "node_conv_240", "node_conv_242", "node_conv_244", "node_conv_248",
    "node_conv_252", "node_conv_256", "node_conv_258", "node_conv_262",
    "node_conv_268", "node_conv_270", "node_conv_274", "node_conv_276",
    "node_conv_280",
]
_CONV_VARIANT_B = ["node_conv_224", "node_conv_288"]

# ---------------------------------------------------------------------------
# P8: node_relu_* (48 files; node_relu.v ALREADY uses the sync-only pattern
#     and is skipped automatically by its missing anchors -> excluded here).
#     beat_buf = IC*8 bits/instance, 181,248 FF total.
# Safety: identical to the node_relu.v precedent -- beat_buf is gather DATA,
# fully rewritten each pixel (beats 0..N-1) before the sending phase reads it;
# write gate (!sending && valid_in && ready_in) replicates the original nested
# condition exactly; during reset valid_in traces to reset-held upstream valids.
# ---------------------------------------------------------------------------

_RELU_BLOCK_A = """    // [K1-FDCE] sync-only memory write -- no reset clause (same pattern as
    // node_relu.v): beat_buf is gather DATA, fully rewritten each pixel
    // before the sending phase reads it. Also unblocks LUTRAM inference.
    always @(posedge clk) begin
        if (!sending && valid_in && ready_in) begin
            beat_buf[in_beat_count] <= data_in;
        end
    end

"""

def _relu_patch(name):
    return FilePatch(
        "output/rtl/%s.v" % name,
        regs=["beat_buf[0:BEATS_PER_PIXEL-1] (256b words)"],
        ff_estimate=None,
        edits=[
            Edit("reset clause: drop beat_buf clear loop",
                 """            for (i = 0; i < BEATS_PER_PIXEL; i = i + 1)
                beat_buf[i] <= {BEAT_WIDTH_BITS{1'b0}};
""",
                 ""),
            Edit("gather: remove beat_buf write (moved to sync-only block)",
                 """                if (valid_in && ready_in) begin
                    beat_buf[in_beat_count] <= data_in;
""",
                 """                if (valid_in && ready_in) begin
"""),
            Edit("insert sync-only write block before the control block",
                 _WRAP_INSERT_ANCHOR, _RELU_BLOCK_A + _WRAP_INSERT_ANCHOR),
        ],
    )

_RELU_FILES = ["node_relu_%d" % i for i in range(1, 49)]  # node_relu.v already done

# ---------------------------------------------------------------------------
# P9/P10: node_add_14 / node_add_15 -- the two OC=2048 adds whose
#     lhs_buf/rhs_buf/out_beats array writes still live INSIDE the async-reset
#     block (the known activation_memory_in_async_reset_block pattern;
#     node_add_1/_5/etc. already use Block A). 49,152 FF-equivalent each.
# Safety: lhs/rhs are fully rewritten during each pixel's 64-beat gather
# before ST_COMPUTE reads them; every out_beats byte is written by the
# 3-stage pipe before ST_STREAM presents it under valid_out.
# ---------------------------------------------------------------------------

P9 = FilePatch(
    "output/rtl/node_add_14.v",
    regs=["lhs_buf[0:2047]", "rhs_buf[0:2047]", "out_beats[0:63] (256b words)"],
    ff_estimate=49152,
    edits=[
        Edit("reset clause: drop out_beats clear loop",
             """            sum_term        <= {TERM_W{1'b0}};
            for (gi = 0; gi < BEATS_PER_PIXEL; gi = gi + 1) begin
                out_beats[gi] <= 256'd0;
            end
""",
             """            sum_term        <= {TERM_W{1'b0}};
"""),
        Edit("remove out_beats pipe write (moved to Block A)",
             """            if (stage3_valid) begin
                out_beats[beat_idx][lane_idx*8 +: 8] <= sat_w;
            end

""",
             ""),
        Edit("ST_IDLE: remove lhs/rhs gather writes (moved to Block A)",
             """                ST_IDLE: begin
                    ready_in <= 1'b1;
                    if (valid_in) begin
                        in_beat_count <= 7'd1;
                        state         <= ST_GATHER;
                        for (gi = 0; gi < CHANNEL_TILE; gi = gi + 1) begin
                            lhs_buf[gi] <= $signed(data_in[gi*8 +: 8]);
                            rhs_buf[gi] <= $signed(data_in[256 + gi*8 +: 8]);
                        end
                    end
                end
""",
             """                ST_IDLE: begin
                    ready_in <= 1'b1;
                    if (valid_in) begin
                        in_beat_count <= 7'd1;
                        state         <= ST_GATHER;
                    end
                end
"""),
        Edit("ST_GATHER: remove lhs/rhs gather writes (moved to Block A)",
             """                ST_GATHER: begin
                    if (valid_in) begin
                        for (gi = 0; gi < CHANNEL_TILE; gi = gi + 1) begin
                            lhs_buf[in_beat_count*CHANNEL_TILE + gi] <= $signed(data_in[gi*8 +: 8]);
                            rhs_buf[in_beat_count*CHANNEL_TILE + gi] <= $signed(data_in[256 + gi*8 +: 8]);
                        end
                        if (in_beat_count == BEATS_PER_PIXEL-1) begin
""",
             """                ST_GATHER: begin
                    if (valid_in) begin
                        if (in_beat_count == BEATS_PER_PIXEL-1) begin
"""),
        Edit("insert Block A before the control block",
             _WRAP_INSERT_ANCHOR,
             """    // [K1-FDCE] Block A: array/data writes (sync-only) -- node_add_1
    // precedent. lhs_buf/rhs_buf are fully rewritten during each pixel's
    // gather before ST_COMPUTE reads them; every out_beats byte is written
    // by the 3-stage pipe before ST_STREAM presents it under valid_out.
    // Sync-only writes also unblock RAM inference for lhs/rhs.
    always @(posedge clk) begin
        if (state == ST_IDLE && valid_in) begin
            for (gi = 0; gi < CHANNEL_TILE; gi = gi + 1) begin
                lhs_buf[gi] <= $signed(data_in[gi*8 +: 8]);
                rhs_buf[gi] <= $signed(data_in[256 + gi*8 +: 8]);
            end
        end
        if (state == ST_GATHER && valid_in) begin
            for (gi = 0; gi < CHANNEL_TILE; gi = gi + 1) begin
                lhs_buf[in_beat_count*CHANNEL_TILE + gi] <= $signed(data_in[gi*8 +: 8]);
                rhs_buf[in_beat_count*CHANNEL_TILE + gi] <= $signed(data_in[256 + gi*8 +: 8]);
            end
        end
        if (stage3_valid) begin
            out_beats[beat_idx][lane_idx*8 +: 8] <= sat_w;
        end
    end

""" + _WRAP_INSERT_ANCHOR),
    ],
)

P10 = FilePatch(
    "output/rtl/node_add_15.v",
    regs=["lhs_buf[0:2047]", "rhs_buf[0:2047]", "out_beats[0:63] (256b words)",
          "v_tmp (blocking temp, moves with out_beats)"],
    ff_estimate=49152,
    edits=[
        Edit("reset clause: drop v_tmp (becomes Block-A-only blocking temp)",
             """            sum_term       <= {SUM_W{1'b0}};
            v_tmp          <= {SUM_W{1'b0}};
""",
             """            sum_term       <= {SUM_W{1'b0}};
"""),
        Edit("remove out_beats pipe write (moved to Block A)",
             """            if (stage3_valid) begin
                v_tmp = sum_term >>> FUSED_SHIFT;
                out_beats[ch_s3 / CHANNEL_TILE][(ch_s3 % CHANNEL_TILE)*8 +: 8] <=
                    (v_tmp > SAT_HI) ? 8'sd127 :
                    (v_tmp < SAT_LO) ? 8'h80   : v_tmp[7:0];
            end

""",
             ""),
        Edit("ST_IDLE: remove lhs/rhs gather writes (moved to Block A)",
             """                ST_IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in && ready_in) begin
                        for (i = 0; i < CHANNEL_TILE; i = i + 1) begin
                            lhs_buf[i] <= $signed(data_in[i*8 +: 8]);
                            rhs_buf[i] <= $signed(data_in[256 + i*8 +: 8]);
                        end
                        in_beat_count <= 7'd1;
                        state         <= ST_GATHER;
                    end
                end
""",
             """                ST_IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in && ready_in) begin
                        in_beat_count <= 7'd1;
                        state         <= ST_GATHER;
                    end
                end
"""),
        Edit("ST_GATHER: remove lhs/rhs gather writes (moved to Block A)",
             """                ST_GATHER: begin
                    if (valid_in && ready_in) begin
                        for (i = 0; i < CHANNEL_TILE; i = i + 1) begin
                            lhs_buf[in_beat_count*CHANNEL_TILE + i] <= $signed(data_in[i*8 +: 8]);
                            rhs_buf[in_beat_count*CHANNEL_TILE + i] <= $signed(data_in[256 + i*8 +: 8]);
                        end
                        if (in_beat_count == 7'd63) begin
""",
             """                ST_GATHER: begin
                    if (valid_in && ready_in) begin
                        if (in_beat_count == 7'd63) begin
"""),
        Edit("insert Block A before the control block",
             _WRAP_INSERT_ANCHOR,
             """    // [K1-FDCE] Block A: array/data writes (sync-only) -- node_add_1
    // precedent. lhs_buf/rhs_buf are fully rewritten during each pixel's
    // gather before ST_COMPUTE reads them; every out_beats byte is written
    // by the 3-stage pipe before ST_STREAM presents it under valid_out.
    // v_tmp is a blocking temp referenced ONLY by this block after the move.
    always @(posedge clk) begin
        if (state == ST_IDLE && valid_in && ready_in) begin
            for (i = 0; i < CHANNEL_TILE; i = i + 1) begin
                lhs_buf[i] <= $signed(data_in[i*8 +: 8]);
                rhs_buf[i] <= $signed(data_in[256 + i*8 +: 8]);
            end
        end
        if (state == ST_GATHER && valid_in && ready_in) begin
            for (i = 0; i < CHANNEL_TILE; i = i + 1) begin
                lhs_buf[in_beat_count*CHANNEL_TILE + i] <= $signed(data_in[i*8 +: 8]);
                rhs_buf[in_beat_count*CHANNEL_TILE + i] <= $signed(data_in[256 + i*8 +: 8]);
            end
        end
        if (stage3_valid) begin
            v_tmp = sum_term >>> FUSED_SHIFT;
            out_beats[ch_s3 / CHANNEL_TILE][(ch_s3 % CHANNEL_TILE)*8 +: 8] <=
                (v_tmp > SAT_HI) ? 8'sd127 :
                (v_tmp < SAT_LO) ? 8'h80   : v_tmp[7:0];
        end
    end

""" + _WRAP_INSERT_ANCHOR),
    ],
)

# ---------------------------------------------------------------------------
# Assemble the full patch list
# ---------------------------------------------------------------------------

def build_patches():
    patches = [P1, P2, P3, P4, P5, P6]
    for n in _CONV_VARIANT_A:
        patches.append(_wrapper_patch_variant_a(n))
    for n in _CONV_VARIANT_B:
        patches.append(_wrapper_patch_variant_b(n))
    for n in _RELU_FILES:
        patches.append(_relu_patch(n))
    patches.append(P9)
    patches.append(P10)
    return patches

# Class-level FF summary (computed from the live RTL parameters, see
# docs/agent_tasks/K1_FDCE_RECODE_ANALYSIS.md for the derivation):
CLASS_FF_SUMMARY = [
    ("line_buf_window window+bypass (38 insts)",            128520),
    ("conv_datapath_mp_k pipes+data_out (36 insts)",        187624),
    ("conv wrapper in_lo+out_pix (35 files)",               196864),
    ("node_relu beat_buf (48 files)",                       181248),
    ("requant_pipeline lanes+scale pipes",                   46080),
    ("mac_array acc (256 lanes)",                             8192),
    ("shared_engine act_in_rd_data_d",                        2048),
    ("top: skip_fifo/eng-fifo/bridges (69 insts)",          125696),
    ("node_add_14/15 lhs/rhs/out_beats (FF-equivalent)",     98304),
]

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def read_text_smart(path):
    """Decode utf-8 first, fall back to cp1252 (several generated .v files
    carry Windows-1252 em-dashes), and normalize CRLF -> LF for anchor
    matching. Returns (lf_text, encoding, eol) so the file is written back
    with the SAME encoding + EOL -> byte-identical outside the edits.
    (Each target file is internally EOL-consistent; verified 2026-06-09.)"""
    raw = path.read_bytes()
    try:
        text, enc = raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        text, enc = raw.decode("cp1252"), "cp1252"
    eol = "\r\n" if "\r\n" in text else "\n"
    return text.replace("\r\n", "\n"), enc, eol


def main():
    ap = argparse.ArgumentParser(description="K1 FDCE->FDRE datapath recode")
    ap.add_argument("--repo-root", default=None,
                    help="repo root (default: parent of this script's dir)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate anchors + list registers; write nothing")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve() if args.repo_root else \
        Path(__file__).resolve().parent.parent
    print("[K1] repo root: %s" % root)

    patches = build_patches()

    # -------- Phase 1: validate everything (no writes) --------
    plan = []     # (patch, new_text, encoding)
    skipped = []  # already-applied
    errors = []
    for p in patches:
        fp = root / p.relpath
        if not fp.is_file():
            errors.append("MISSING FILE: %s" % p.relpath)
            continue
        text, enc, eol = read_text_smart(fp)
        if MARKER in text:
            skipped.append(p.relpath)
            continue
        t = text
        ok = True
        for e in p.edits:
            n = t.count(e.old)
            if n != 1:
                errors.append("ANCHOR DRIFT in %s -- edit '%s' matched %d times (need exactly 1)"
                              % (p.relpath, e.desc, n))
                ok = False
                break
            t = t.replace(e.old, e.new, 1)
        if ok:
            plan.append((p, t, enc, eol))

    if errors:
        print("\n[K1] ABORT -- anchor validation failed; NO files were modified:")
        for e in errors:
            print("  " + e)
        sys.exit(1)

    # -------- Report --------
    print("\n[K1] plan: %d files to patch, %d already applied (marker found)"
          % (len(plan), len(skipped)))
    for p, _, _, _ in plan:
        print("  %-44s -> moves: %s" % (p.relpath, "; ".join(p.regs)))
    if skipped:
        print("[K1] skipped (idempotent): %s" % ", ".join(skipped))

    print("\n[K1] estimated FFs moved off rst_n (async-reset FDCE -> no-reset FDRE):")
    tot = 0
    for name, ff in CLASS_FF_SUMMARY:
        print("  %-52s %8d" % (name, ff))
        tot += ff
    print("  %-52s %8d" % ("TOTAL (~74% of the ~1.3M-FF design)", tot))

    if args.dry_run:
        print("\n[K1] dry-run: no files written.")
        return

    # -------- Phase 2: commit (backups first) --------
    for p, new_text, enc, eol in plan:
        fp = root / p.relpath
        bak = fp.with_name(fp.name + ".prek1")
        if not bak.exists():
            bak.write_bytes(fp.read_bytes())
        if eol != "\n":
            new_text = new_text.replace("\n", eol)
        fp.write_bytes(new_text.encode(enc))
        print("[K1] patched %s (backup: %s)" % (p.relpath, bak.name))

    print("\n[K1] done: %d files patched, %d skipped." % (len(plan), len(skipped)))
    print("[K1] next: lint (verilator --lint-only), then e2e byte-exact gate vs")
    print("     the FRESH golden before any Vivado run (HARD RULE).")

if __name__ == "__main__":
    main()
