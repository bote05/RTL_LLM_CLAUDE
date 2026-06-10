#!/usr/bin/env python3
"""apply_mbv2_engpipe.py — ENG-PIPE: pipelined (pixel, oc_pass) issue for the
MobileNetV2 shared-engine top. Anchor-asserted + idempotent; writes .preengpipe
backups before first mutation of each file.

DESIGN (full schedule table + hazard proofs: docs/agent_tasks/ENG_PIPE_ANALYSIS.md)
------------------------------------------------------------------------------
The legacy engine FSM round-trips ST_RUN -> ST_REQUANT -> ST_DRAIN -> ST_RUN
per (pixel, oc_pass) with the address generator PARKED (run_active =
state==ST_RUN only): ~10 idle issue cycles per intermediate oc_pass and ~12
per pixel boundary. ENG-PIPE restarts the next walk 3 cycles after mac_done
while the previous pass's requant/drain retires IN PARALLEL off per-pass
capture registers:

  * NEW param `ENG_PIPE` (default 0). Every change is inside
    `generate if (ENG_PIPE == 0)` (legacy text VERBATIM) /
    `generate if (ENG_PIPE != 0)` (new g_ep block) — ResNet (which never
    sets ENG_PIPE) elaborates bit- AND cycle-identically
    (gate: ResNet vec0 PASS 0/100352 @ EXACTLY 5,664,715).
  * ISSUE side (ENG_PIPE=1): ST_RUN -> ST_GAP (2 cycles) -> ST_RUN. Outer
    counters advance at mac_done_d1; the AG's existing run_active
    rising-edge init (counter reset + bias read) re-fires per restart.
    mac_clear moves from run-entry to run-entry+3 (strictly after the
    previous pass's acc capture at mac_done_d5, strictly before the new
    walk's first accumulate at run-entry+4).
  * RETIRE side: per-pass captures addr_cap (mac_done_d1) /
    bias_cap+scale_cap (d4: one cycle before the restarted walk's bias read
    lands on the live bus at D+5) / acc_cap (d5: acc final since D+4). The
    requant pipe is fed FROM THE CAPTURES by a retire FIRE pulse gated on
    (requant pipe empty && bridge write empty), so a downstream stall
    (out_ready low, ENABLE_OUTPUT_BACKPRESSURE) can never clobber or drop a
    beat: the bridge holds exactly one beat, the in-flight walk completes
    into the (uncleared) accumulators, at most TWO passes pend, and the gap
    holds while pend==2 (freezing every live source). A pass finishing
    behind an unfired head skips its d-chain captures and is staged from
    the frozen live sources at the head's fire (fire_recap); its own d5
    refreshes acc_cap if the recapture ran before its accumulator was final.
  * Per-pass issue bubble: 12 (pixel boundary) / 10 (intermediate) -> 3.
    Retire latency +1 cycle per pass (fire = d6 vs legacy d5) — latency
    only, never throughput, except dense IC=16 layers (walk N=2 < retire
    spacing 6 -> effective period 6 vs 5).
  * MODES: dense fast-walk, depthwise serial, FC — the retire machinery is
    keyed only to ag_mac_done and FSM-owned counters, so ALL dispatch modes
    take the same pipelined path (no per-mode gating needed).
  * ATOMIC RULE: no scripts/ Python formula models the shared-engine FSM
    cycles (compute_conv2d_latency_cycles in golden_impl.py models the
    per-module SPATIAL datapath only); goldens are value-streams. Nothing
    else to update.

Files touched:
  output/rtl/shared_engine_skeleton.v        (param + generate-if recode)
  output/mobilenet-v2/rtl/nn2rtl_top_engine.v (.ENG_PIPE(1))
  tb/engine_iso_wrap_mbv2.v                  (`ifdef ENG_PIPE / `ifdef THROTTLE)

Usage: python scripts/apply_mbv2_engpipe.py [--check]
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKEL = REPO / "output" / "rtl" / "shared_engine_skeleton.v"
TOP = REPO / "output" / "mobilenet-v2" / "rtl" / "nn2rtl_top_engine.v"
ISO = REPO / "tb" / "engine_iso_wrap_mbv2.v"

MARK = "[ENG_PIPE 2026-06-10]"

_backed_up: set[Path] = set()


def patch(path: Path, old: str, new: str, tag: str, count: int = 1) -> None:
    text = path.read_text(encoding="utf-8")
    # Idempotency: `new` always carries [ENG_PIPE] marker text that the
    # pre-change files cannot contain, so its presence == already applied.
    if new in text:
        print(f"  [skip] {path.name}: {tag} already applied")
        return
    n = text.count(old)
    if n != count:
        raise SystemExit(f"ANCHOR FAIL {path.name} / {tag}: found {n}, want {count}")
    if path not in _backed_up:
        bak = path.with_name(path.name + ".preengpipe")
        if not bak.exists():
            bak.write_text(text, encoding="utf-8", newline="\n")
        _backed_up.add(path)
    path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")
    print(f"  [ok]   {path.name}: {tag}")


# ============================================================================
# 1. shared_engine_skeleton.v
# ============================================================================
def patch_skel() -> None:
    # ---- A: ENG_PIPE parameter ----
    patch(SKEL, """    parameter integer K_PAR = 1
) (
    // ---- Clock + reset ----
""", f"""    parameter integer K_PAR = 1,

    // ---- Pipelined (pixel, oc_pass) issue (default OFF = verbatim legacy) ----
    // {MARK} When 0 (DEFAULT) every ENG_PIPE generate-if in this file
    // elaborates the ORIGINAL stop-and-wait FSM, outer counters and requant
    // hookups VERBATIM — every ResNet instance (which never sets ENG_PIPE)
    // is bit- and cycle-identical. When 1 (MBV2 engine top + the -DENG_PIPE
    // iso build) the engine restarts the next (pixel, oc_pass) walk 3
    // cycles after mac_done while the previous pass's requant/drain retires
    // in parallel off per-pass capture registers (acc/bias/scale/wr-addr):
    // per-pass issue bubble 12 (pixel) / 10 (intermediate) -> 3. Schedule
    // table + hazard proofs: docs/agent_tasks/ENG_PIPE_ANALYSIS.md.
    parameter integer ENG_PIPE = 0
) (
    // ---- Clock + reset ----
""", "ENG_PIPE parameter")

    # ---- B: ST_GAP state encoding ----
    patch(SKEL, """    localparam ST_DONE         = 3'd5;
""", f"""    localparam ST_DONE         = 3'd5;
    // {MARK} fixed 2-cycle issue gap between walks (extends while a retire
    // backlog or the layer end holds it). Unreachable when ENG_PIPE==0 (no
    // arc targets it in the legacy FSM).
    localparam ST_GAP          = 3'd6;
""", "ST_GAP localparam")

    # ---- C: wrap req_done_pending + the FSM comb block (legacy VERBATIM) ----
    patch(SKEL, """    reg req_done_pending;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            req_done_pending <= 1'b0;
        end else if (state != ST_REQUANT) begin
            req_done_pending <= 1'b0;
        end else if (requant_valid_out && !eff_out_ready) begin
            req_done_pending <= 1'b1;   // beat done, stalled on downstream
        end else if (eff_out_ready) begin
            req_done_pending <= 1'b0;   // free to advance -> clear
        end
    end
    always @* begin
        next_state = state;
        case (state)
            ST_IDLE:        if (engine_start_pulse) next_state = ST_LOAD_CONFIG;
            ST_LOAD_CONFIG: next_state = ST_RUN;
            ST_RUN:         if (ag_mac_done)        next_state = ST_REQUANT;
""", f"""    // {MARK} legacy stop-and-wait FSM kept VERBATIM inside generate-if;
    // the ENG_PIPE FSM (ST_GAP issue pipelining + end-of-layer ST_DRAIN
    // flush) lives in g_ep at the bottom of the module.
    generate if (ENG_PIPE == 0) begin : g_fsm_legacy
    reg req_done_pending;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            req_done_pending <= 1'b0;
        end else if (state != ST_REQUANT) begin
            req_done_pending <= 1'b0;
        end else if (requant_valid_out && !eff_out_ready) begin
            req_done_pending <= 1'b1;   // beat done, stalled on downstream
        end else if (eff_out_ready) begin
            req_done_pending <= 1'b0;   // free to advance -> clear
        end
    end
    always @* begin
        next_state = state;
        case (state)
            ST_IDLE:        if (engine_start_pulse) next_state = ST_LOAD_CONFIG;
            ST_LOAD_CONFIG: next_state = ST_RUN;
            ST_RUN:         if (ag_mac_done)        next_state = ST_REQUANT;
""", "FSM legacy wrap (open)")

    patch(SKEL, """            ST_DRAIN:       if (!bridge_busy)       next_state = ag_pixel_done ? ST_DONE : ST_RUN;
            ST_DONE:        if (!engine_start)      next_state = ST_IDLE;
            default:        next_state = ST_IDLE;
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= ST_IDLE;
        else        state <= next_state;
    end
""", f"""            ST_DRAIN:       if (!bridge_busy)       next_state = ag_pixel_done ? ST_DONE : ST_RUN;
            ST_DONE:        if (!engine_start)      next_state = ST_IDLE;
            default:        next_state = ST_IDLE;
        endcase
    end
    end endgenerate   // {MARK} g_fsm_legacy

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= ST_IDLE;
        else        state <= next_state;
    end
""", "FSM legacy wrap (close)")

    # ---- C2 (RIDER, pre-existing B-class bug): legacy backpressure-hold
    #      pass-skip. At the requant_valid_out cycle oc_pass_idx ALREADY
    #      advances, so the held re-evaluation (req_done_pending) re-checked
    #      the last-pass compare with the NEXT pass's index: a hold landing
    #      on the second-to-last pass jumped to ST_DRAIN and SKIPPED the
    #      final oc_pass (stale output channels). Dormant in deployment
    #      (MBV2 e2e never stalled at the vulnerable cycle; ResNet has
    #      backpressure disabled => req_done_pending identically 0 => the
    #      changed arm is DEAD => provably bit/cycle-identical there).
    #      Found by this task's LFSR-throttled iso (legacy DW 896:
    #      mismatch=14114, REPRODUCED on the unpatched .preengpipe skeleton
    #      => pre-existing). A held LAST pass never re-enters this arm (it
    #      went to ST_DRAIN at valid_out), so the pending re-evaluation can
    #      only ever resume ST_RUN. ----
    patch(SKEL, """            ST_REQUANT:     if (requant_valid_out || req_done_pending) begin
                                if (oc_pass_idx == oc_pass_total_m1[2:0])
                                    next_state = ST_DRAIN;
                                else
                                    next_state = eff_out_ready ? ST_RUN : ST_REQUANT;
                            end
""", f"""            ST_REQUANT:     if (requant_valid_out) begin
                                if (oc_pass_idx == oc_pass_total_m1[2:0])
                                    next_state = ST_DRAIN;
                                else
                                    next_state = eff_out_ready ? ST_RUN : ST_REQUANT;
                            end else if (req_done_pending) begin
                                // {MARK}[B-fix] held INTERMEDIATE pass: oc_pass_idx
                                // already advanced at the valid_out cycle, so the old
                                // shared arm re-checked last-pass with the NEXT pass's
                                // index — a hold landing on the second-to-last pass
                                // jumped to ST_DRAIN and SKIPPED the final pass. A held
                                // LAST pass never re-enters here (it went to ST_DRAIN
                                // at valid_out), so resuming ST_RUN is the only arc.
                                next_state = eff_out_ready ? ST_RUN : ST_REQUANT;
                            end
""", "legacy hold pass-skip B-fix")

    # ---- D: act_out_wr_addr hookup ----
    patch(SKEL, """    assign weight_rd_en    = ag_weight_rd_en;
    assign act_in_rd_addr  = ag_act_in_rd_addr;
    assign act_in_rd_en    = ag_act_in_rd_en;
    assign act_out_wr_addr = ag_act_out_wr_addr;
""", f"""    assign weight_rd_en    = ag_weight_rd_en;
    assign act_in_rd_addr  = ag_act_in_rd_addr;
    assign act_in_rd_en    = ag_act_in_rd_en;
    // {MARK} legacy: live AG write address (stable through REQUANT/DRAIN
    // because the FSM stop-and-waits per pass). ENG_PIPE: the restarted walk
    // overwrites the AG register long before the bridge write fires, so the
    // port is driven from the per-beat in-flight capture instead (see g_ep).
    generate if (ENG_PIPE == 0) begin : g_waddr_out_legacy
    assign act_out_wr_addr = ag_act_out_wr_addr;
    end endgenerate
""", "act_out_wr_addr generate-if")

    # ---- E: mac_clear hookup ----
    patch(SKEL, """    wire run_entered = (state == ST_RUN) && !state_run_d;
    assign mac_clear = run_entered;
""", f"""    wire run_entered = (state == ST_RUN) && !state_run_d;
    // {MARK} legacy: clear accumulators on ST_RUN entry (the previous
    // pass's acc was already captured by the requant pipe in ST_REQUANT).
    // ENG_PIPE: clear 3 cycles AFTER run-entry (see g_ep) — strictly after
    // the previous pass's acc_cap capture at mac_done_d5 and strictly
    // before the restarted walk's first accumulate at run-entry+4.
    generate if (ENG_PIPE == 0) begin : g_clear_legacy
    assign mac_clear = run_entered;
    end endgenerate
""", "mac_clear generate-if")

    # ---- F: wrap the outer-counter block (legacy VERBATIM) ----
    patch(SKEL, """    // Outer counters tick on the FSM transitions documented in
    // 00_engine_skeleton_spec_FSM.md.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            oc_pass_idx_r <= 3'd0;
            pixel_h_r     <= 8'd0;
            pixel_w_r     <= 8'd0;
        end else begin
""", f"""    // Outer counters tick on the FSM transitions documented in
    // 00_engine_skeleton_spec_FSM.md.
    // {MARK} legacy counter block kept VERBATIM inside generate-if; the
    // ENG_PIPE counters advance at mac_done_d1 instead (see g_ep).
    generate if (ENG_PIPE == 0) begin : g_counters_legacy
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            oc_pass_idx_r <= 3'd0;
            pixel_h_r     <= 8'd0;
            pixel_w_r     <= 8'd0;
        end else begin
""", "counters legacy wrap (open)")

    patch(SKEL, """            if (state == ST_DRAIN && !bridge_busy && !ag_pixel_done) begin
                if (pixel_w_r == pixel_w_m1) begin
                    pixel_w_r <= 8'd0;
                    pixel_h_r <= (pixel_h_r == pixel_h_m1) ? 8'd0 : (pixel_h_r + 8'd1);
                end else begin
                    pixel_w_r <= pixel_w_r + 8'd1;
                end
            end
        end
    end
""", f"""            if (state == ST_DRAIN && !bridge_busy && !ag_pixel_done) begin
                if (pixel_w_r == pixel_w_m1) begin
                    pixel_w_r <= 8'd0;
                    pixel_h_r <= (pixel_h_r == pixel_h_m1) ? 8'd0 : (pixel_h_r + 8'd1);
                end else begin
                    pixel_w_r <= pixel_w_r + 8'd1;
                end
            end
        end
    end
    end endgenerate   // {MARK} g_counters_legacy
""", "counters legacy wrap (close)")

    # ---- G: requant input hookups ----
    patch(SKEL, """    assign requant_bias_in  = bias_rd_data;
    // Per-OC scale: read at bias's address/enable (base_words identical), so the
    // scale word for the current oc_pass arrives aligned with bias_rd_data.
    assign scale_rd_addr    = ag_bias_rd_addr;
    assign scale_rd_en      = ag_bias_rd_en;
    wire [MAC_COUNT*32-1:0] requant_scale_in = scale_rd_data;
    assign requant_valid_in = ag_mac_done_d5;
""", f"""    // Per-OC scale: read at bias's address/enable (base_words identical), so the
    // scale word for the current oc_pass arrives aligned with bias_rd_data.
    assign scale_rd_addr    = ag_bias_rd_addr;
    assign scale_rd_en      = ag_bias_rd_en;
    // {MARK} requant input sources. Legacy (VERBATIM hookups): live
    // bias/scale buses + live MAC accumulators, captured by the pipe at
    // mac_done_d5 while the FSM stop-and-waits. ENG_PIPE: per-pass capture
    // registers + the retire FIRE pulse (see g_ep) — the restarted walk
    // (and its bias read) is already in flight at capture time, so the live
    // sources can no longer be sampled at the capture cycle.
    wire [MAC_COUNT*32-1:0]    requant_scale_in;
    wire [MAC_COUNT*ACC_W-1:0] requant_acc_in;
    generate if (ENG_PIPE == 0) begin : g_requant_in_legacy
    assign requant_bias_in  = bias_rd_data;
    assign requant_scale_in = scale_rd_data;
    assign requant_acc_in   = mac_acc_out;
    assign requant_valid_in = ag_mac_done_d5;
    end endgenerate
""", "requant input generate-if")

    # ---- H: requant_pipeline acc_in port ----
    patch(SKEL, """        .valid_in     (requant_valid_in),
        .acc_in       (mac_acc_out),
""", f"""        .valid_in     (requant_valid_in),
        .acc_in       (requant_acc_in),   // {MARK} == mac_acc_out verbatim when ENG_PIPE==0
""", "requant acc_in port")

    # ---- I: the g_ep block (all ENG_PIPE machinery) ----
    patch(SKEL, """        .valid_out    (requant_valid_out),
        .data_out     (requant_data_out)
    );

endmodule
""", f"""        .valid_out    (requant_valid_out),
        .data_out     (requant_data_out)
    );

    // ====================================================================
    // {MARK} Pipelined (pixel, oc_pass) issue — ENG_PIPE != 0 only.
    //
    // ISSUE side: ST_RUN -> (mac_done) -> ST_GAP (2 cycles min) -> ST_RUN
    // restarts the NEXT (pixel, oc_pass) walk 3 cycles after mac_done; the
    // outer counters advance at mac_done_d1 so the restarted walk (weight
    // pass_offset, DW act chunk, bias read at the run_active rising edge)
    // sees the new pass. The per-pass ST_REQUANT/ST_DRAIN round trip is
    // GONE; ST_DRAIN remains only as the end-of-layer flush.
    //
    // RETIRE side (event-driven, decoupled from the FSM): each finished
    // pass is a "pend". Its write address / bias / scale / accumulator are
    // captured into single-slot capture registers on the mac_done d-chain
    // (d1 / d4 / d4 / d5) while the live sources still hold that pass's
    // values; the requant pipe is FED FROM THE CAPTURES by a retire FIRE
    // pulse (requant_valid_in) gated on (requant pipe empty && bridge write
    // empty), so a downstream stall (out_ready low) can never clobber or
    // drop a beat — the bridge holds exactly one beat, the in-flight walk
    // completes into the (not-yet-cleared) accumulators, at most TWO passes
    // pend, and the gap HOLDS while pend==2, freezing every live source.
    // That freeze is what makes the single-slot captures sufficient: a pass
    // finishing behind an unfired head skips its d-chain captures (it is
    // not the head) and is staged from the FROZEN live sources at the
    // head's fire (fire_recap); its own d5 then refreshes acc_cap with the
    // final accumulator if the recapture ran before D+4 (acc-final).
    //
    // Schedule (D = mac_done cycle of pass P; unstalled; proofs in
    // docs/agent_tasks/ENG_PIPE_ANALYSIS.md):
    //   D    : last weight issue of P; FSM -> ST_GAP next cycle
    //   D+1  : d1: oc_pass/pixel advance + addr_cap <= live AG write addr
    //   D+2  : gap exit decision (pixel_done -> DRAIN; pend<=1 -> RUN)
    //   D+3  : ST_RUN: AG rising edge (counter reset + bias read issue)
    //   D+4  : first new weight issue; d4: bias_cap/scale_cap <= live bus
    //          (the restarted walk's bias word lands one cycle LATER, D+5)
    //   D+5  : d5: acc_cap <= acc (final since D+4); head-ready tick
    //   D+6  : mac_clear (run_entered_d3); earliest retire FIRE
    //   D+7  : first accumulate of the new pass (strictly after the clear)
    //   D+10 : requant_valid_out;  D+11: bridge write presented
    // Per-pass issue bubble: 12 (pixel) / 10 (intermediate) -> 3.
    // ====================================================================
    generate if (ENG_PIPE != 0) begin : g_ep

        // ---- retire bookkeeping ----
        // pend_cnt: passes finished but not yet fired into the requant pipe.
        // Max 2 BY CONSTRUCTION: the gap only releases a new walk when
        // pend<=1 (at most one walking + one pending behind the head).
        reg [1:0] pend_cnt;
        reg       rdy_head;   // head pend's acc_cap is valid (its d5 ran)
        reg       rdy_tail;   // tail pend's d5 ran while the head was unfired
        reg       in_pipe;    // a beat occupies the requant pipe (fire..valid_out)

        // FIRE: feed the head pend into the requant pipe. Gated on the pipe
        // AND the bridge write register being empty so a stalled (held)
        // write can never be clobbered. Unstalled this gating costs nothing
        // for N>=3-issue walks (the previous beat clears the bridge before
        // this beat's earliest fire); N==2 walks retire at period 6.
        wire fire = (pend_cnt != 2'd0) && rdy_head && !in_pipe && !act_out_wr_en;
        // head-fire recapture: a successor pend exists (gap held, live
        // sources frozen) -> stage its live values into the caps this fire
        // frees. Reads of the caps at this same edge get the OLD (head's)
        // values (non-blocking semantics).
        wire fire_recap = fire && (pend_cnt == 2'd2);
        // d-chain capture eligibility: the event's pass (the most recent
        // mac_done) is the queue head <=> it is the ONLY pend.
        wire ev_is_head = (pend_cnt == 2'd1);

        // queue update: fire-shift, then ready tick (d5), then push.
        wire       nh0 = fire ? rdy_tail : rdy_head;
        wire       nt0 = fire ? 1'b0     : rdy_tail;
        wire [1:0] np0 = pend_cnt - (fire ? 2'd1 : 2'd0);
        wire       nh1 = (ag_mac_done_d5 && !nh0 && (np0 != 2'd0)) ? 1'b1 : nh0;
        wire       nt1 = (ag_mac_done_d5 &&  nh0 && (np0 == 2'd2)) ? 1'b1 : nt0;
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                pend_cnt <= 2'd0;
                rdy_head <= 1'b0;
                rdy_tail <= 1'b0;
                in_pipe  <= 1'b0;
            end else begin
                pend_cnt <= np0 + (ag_mac_done ? 2'd1 : 2'd0);
                rdy_head <= nh1;
                rdy_tail <= nt1;
                if (fire)                    in_pipe <= 1'b1;
                else if (requant_valid_out)  in_pipe <= 1'b0;
            end
        end

        // ---- per-pass capture registers (datapath: no reset => FDRE) ----
        reg [ACT_BRAM_ADDR_W-1:0]  addr_cap;
        reg [MAC_COUNT*BIAS_W-1:0] bias_cap;
        reg [MAC_COUNT*32-1:0]     scale_cap;
        reg [MAC_COUNT*ACC_W-1:0]  acc_cap;
        always @(posedge clk) begin
            // wr-addr: the live AG register holds pass P's address until the
            // restarted walk's first cycle (D+3) overwrites it (visible D+4).
            if ((ag_mac_done_d1 && ev_is_head) || fire_recap)
                addr_cap <= ag_act_out_wr_addr;
            // bias/scale: live buses hold pass P's words until the restarted
            // walk's bias read lands at D+5; captured at D+4.
            if ((ag_mac_done_d4 && ev_is_head) || fire_recap) begin
                bias_cap  <= bias_rd_data;
                scale_cap <= scale_rd_data;
            end
            // acc: final from D+4 (last accumulate edge ends D+3); captured
            // at D+5, strictly before mac_clear at D+6. A fire_recap earlier
            // than the successor's D+4 stages a not-yet-final acc — its own
            // d5 (it is the head by then) refreshes it with the final value.
            if ((ag_mac_done_d5 && ev_is_head) || fire_recap)
                acc_cap <= mac_acc_out;
        end

        // beat-in-flight write address: latched at fire, stable until the
        // NEXT fire — which is gated on this beat's bridge acceptance, so it
        // covers the entire (possibly held) write window.
        reg [ACT_BRAM_ADDR_W-1:0] addr_inflight;
        always @(posedge clk) begin
            if (fire) addr_inflight <= addr_cap;
        end
        assign act_out_wr_addr = addr_inflight;

        // ---- FSM (replaces the per-pass REQUANT/DRAIN round trip) ----
        reg gap_eligible;  // 0 on the first ST_GAP cycle -> minimum 2-cycle gap
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n)               gap_eligible <= 1'b0;
            else if (state != ST_GAP) gap_eligible <= 1'b0;
            else                      gap_eligible <= 1'b1;
        end
        always @* begin
            next_state = state;
            case (state)
                ST_IDLE:        if (engine_start_pulse) next_state = ST_LOAD_CONFIG;
                ST_LOAD_CONFIG: next_state = ST_RUN;
                ST_RUN:         if (ag_mac_done)        next_state = ST_GAP;
                // gap exit: layer done -> final flush; else restart the next
                // walk only when at most the just-finished pass is pending
                // (an older unfired pend HOLDS the gap -> live sources stay
                // frozen for its fire_recap staging).
                ST_GAP:         if (gap_eligible) begin
                                    if (ag_pixel_done)
                                        next_state = ST_DRAIN;
                                    else if (pend_cnt <= 2'd1)
                                        next_state = ST_RUN;
                                end
                // end-of-layer flush: every pend fired, pipe empty, bridge
                // write accepted (bridge_busy covers requant_valid + wr_en).
                ST_DRAIN:       if ((pend_cnt == 2'd0) && !in_pipe && !bridge_busy)
                                    next_state = ST_DONE;
                ST_DONE:        if (!engine_start)      next_state = ST_IDLE;
                default:        next_state = ST_IDLE;
            endcase
        end

        // ---- outer counters: advance at mac_done_d1 (was REQUANT/DRAIN).
        //      The restarted walk at D+3 then reads the new pass/pixel for
        //      its weight pass_offset, DW act chunk and bias read address.
        //      Gated on !ag_pixel_done so the layer's final pass parks the
        //      counters (they reset at the next ST_LOAD_CONFIG). ----
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                oc_pass_idx_r <= 3'd0;
                pixel_h_r     <= 8'd0;
                pixel_w_r     <= 8'd0;
            end else begin
                if (state == ST_LOAD_CONFIG) begin
                    oc_pass_idx_r <= 3'd0;
                    pixel_h_r     <= 8'd0;
                    pixel_w_r     <= 8'd0;
                end
                if (ag_mac_done_d1 && !ag_pixel_done) begin
                    if (oc_pass_idx_r == oc_pass_total_m1[2:0]) begin
                        oc_pass_idx_r <= 3'd0;
                        if (pixel_w_r == pixel_w_m1) begin
                            pixel_w_r <= 8'd0;
                            pixel_h_r <= (pixel_h_r == pixel_h_m1) ? 8'd0
                                                                   : (pixel_h_r + 8'd1);
                        end else begin
                            pixel_w_r <= pixel_w_r + 8'd1;
                        end
                    end else begin
                        oc_pass_idx_r <= oc_pass_idx_r + 3'd1;
                    end
                end
            end
        end

        // ---- mac_clear: 3 cycles after run-entry — strictly between the
        //      previous pass's acc_cap capture (D+5) and the restarted
        //      walk's first accumulate (run-entry+4). ----
        reg run_entered_d1, run_entered_d2, run_entered_d3;
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                run_entered_d1 <= 1'b0;
                run_entered_d2 <= 1'b0;
                run_entered_d3 <= 1'b0;
            end else begin
                run_entered_d1 <= run_entered;
                run_entered_d2 <= run_entered_d1;
                run_entered_d3 <= run_entered_d2;
            end
        end
        assign mac_clear = run_entered_d3;

        // ---- retire FIRE feeds the requant pipe from the captures ----
        assign requant_valid_in = fire;
        assign requant_acc_in   = acc_cap;
        assign requant_bias_in  = bias_cap;
        assign requant_scale_in = scale_cap;
    end endgenerate

endmodule
""", "g_ep block")


# ============================================================================
# 2. nn2rtl_top_engine.v — arm ENG_PIPE on the MBV2 engine top
# ============================================================================
def patch_top() -> None:
    patch(TOP, """        .K_PAR(8),              // [KPAR8 2026-06-10] 8 taps/cycle/lane (dense 1x1 + FC fast walk; DW serial fallback)
""", f"""        .K_PAR(8),              // [KPAR8 2026-06-10] 8 taps/cycle/lane (dense 1x1 + FC fast walk; DW serial fallback)
        .ENG_PIPE(1),           // {MARK} pipelined (pixel, oc_pass) issue: bubble 12/10 -> 3
""", "top .ENG_PIPE(1)")


def patch_top_arb_commit() -> None:
    """[ARB-COMMIT] The engine's act-BRAM write port occupancy must be ONE
    cycle per beat (the FIFO-ACCEPT cycle), not the whole held presentation.

    Root cause (found via per-dispatch beat checksums + input-region
    checksums, see ENG_PIPE_ANALYSIS.md §"e2e loader starvation"): the
    bram_to_stream_bridge holds act_out_wr_en HIGH while the engine_output_
    fifo is full (out_ready low). The act-BRAM write arbiter gives the
    engine absolute priority, so during drain-limited stretches the engine
    hogged the write port ~16/17 cycles and the input-loader bridges
    (1-deep skid, documented drop-on-multi-cycle-denial limitation,
    push-only in_valid) DROPPED beats -> corrupted engine input regions
    (first seen: dispatch 6's region, e2e ±1 logit mismatches). ENG_PIPE
    made this regime common (production rate ~6 cyc/beat >> drain rate);
    in legacy the FIFO rarely filled. The held writes are REDUNDANT for
    the BRAM (same addr/data every cycle): committing only on the accept
    cycle (wr_en && eofifo_in_ready) is value-identical for the engine
    (every beat still lands exactly once, before engine_done — ST_DRAIN
    waits for the accept) and frees the port for the loaders. The FIFO's
    in_valid stays the raw wr_en (push semantics unchanged)."""
    patch(TOP, """    assign ldr0_wr_grant = ldr0_wr_req & ~(engine_act_out_wr_en);
""", f"""    // {MARK}[ARB-COMMIT] engine occupies the act-BRAM write port ONLY on
    // the FIFO-accept cycle of each beat; a held presentation (out_ready
    // low) no longer starves the input-loader bridges (1-deep-skid drop
    // hazard). See scripts/apply_mbv2_engpipe.py + ENG_PIPE_ANALYSIS.md.
    wire engine_act_wr_commit = engine_act_out_wr_en & eofifo_in_ready;
    assign ldr0_wr_grant = ldr0_wr_req & ~(engine_act_wr_commit);
""", "ARB-COMMIT wire + ldr0 grant")

    # remaining grant lines + the three final muxes: targeted line rewrite
    text = TOP.read_text(encoding="utf-8")
    out_lines = []
    n_rew = 0
    for line in text.split("\n"):
        ls = line.lstrip()
        if (ls.startswith("assign ldr") and "_wr_grant = " in line
                and "engine_act_out_wr_en" in line) or \
           (ls.startswith("assign act_wr_en_final")
                and "engine_act_out_wr_en" in line) or \
           (ls.startswith("assign act_wr_addr_final")
                and "engine_act_out_wr_en" in line) or \
           (ls.startswith("assign act_wr_data_final")
                and "engine_act_out_wr_en" in line):
            line = line.replace("engine_act_out_wr_en", "engine_act_wr_commit")
            n_rew += 1
        out_lines.append(line)
    if n_rew:
        TOP.write_text("\n".join(out_lines), encoding="utf-8", newline="\n")
        print(f"  [ok]   {TOP.name}: ARB-COMMIT rewrote {n_rew} arbiter lines")
    else:
        print(f"  [skip] {TOP.name}: ARB-COMMIT arbiter lines already rewritten")


# ============================================================================
# 2b. node_add_*.v (MBV2 residual adds) — [ADD-JOIN FIX]
# ============================================================================
ADD_FILES = sorted((REPO / "output" / "mobilenet-v2" / "rtl").glob("node_add_*.v"))


def patch_mbv2_adds() -> None:
    """[ADD-JOIN FIX] accept-vs-pop desync in the MBV2 residual-add elastic
    handshake (B-class, latent; exposed by ENG_PIPE's denser act-BRAM
    write-port pulses toggling the downstream loader's in_ready).

    The top wires each residual join as: lhs/skip skid-FIFOs pop on
    (add.ready_in && other.valid && spatial_run); the add ACCEPTS on
    (valid_in && !skid_block) [some variants && ready_in too]. ready_in is
    REGISTERED (1-cycle stale) while skid_block is combinational, so a
    downstream-ready toggle in the wrong cycle makes the add accept a pair
    the FIFOs did not pop (duplicate/stale pair, counts preserved, every
    later pixel shifted one position -> tiny final-logit deltas because the
    GAP head averages out the spatial shift; localized via lhs/skip/out
    stream checksums: inputs identical, accepted-output stream diverged).

    Fix: ready_in becomes the COMBINATIONAL truth
        assign ready_in = (state == ST_IDLE) && !skid_block;
    so the pop predicate and the accept predicate are the SAME expression
    by construction (pop = ready_in && bothValid && run == accept). The old
    registered writes are retargeted to a dead shadow reg (ready_in_r) so
    every FSM shape across the 10 generated files is handled uniformly.
    With ENABLE_BACKPRESSURE==0 the combinational form is cycle-identical
    to the old register (1 in ST_IDLE, 0 in ST_RUN, same edges); with ==1
    it additionally drops ready exactly on the parked-and-stalled cycles —
    which is precisely what the old stale register failed to do. The
    accept's !skid_block gate is unchanged, so the parked-beat overwrite
    protection is preserved (a new frame is never accepted while a beat is
    parked and stalled)."""
    import re

    for f in ADD_FILES:
        text = f.read_text(encoding="utf-8")
        if "[ADD-JOIN FIX]" in text:
            print(f"  [skip] {f.name}: ADD-JOIN FIX already applied")
            continue
        orig = text
        # (a) port: output reg -> output wire
        text, n_port = re.subn(r"output reg(\s+)ready_in,",
                               r"output wire\1ready_in,", text, count=1)
        # (b) retarget every registered write to the shadow reg
        text, n_wr = re.subn(r"\bready_in(\s*)<=", r"ready_in_r\1<=", text)
        # (c) shadow decl + the combinational truth, inserted at endmodule
        idle_name = "ST_IDLE" if "ST_IDLE" in text else "S_IDLE"
        fix = (
            "    // [ENG_PIPE 2026-06-10][ADD-JOIN FIX] ready_in is the SAME signal the\n"
            "    // two input skid-FIFOs pop on, so it must be the COMBINATIONAL truth of\n"
            "    // the accept predicate (the old registered ready_in was 1 cycle stale\n"
            "    // vs the combinational skid_block -> accept/pop desync = duplicate or\n"
            "    // stale pair processing when the downstream ready toggled). ready_in_r\n"
            "    // keeps the legacy register writes (now shadow/dead) so every generated\n"
            "    // FSM shape is patched uniformly. Cycle-identical when\n"
            "    // ENABLE_BACKPRESSURE==0 (1 in IDLE, 0 in RUN, same edges).\n"
            "    reg ready_in_r;\n"
            "    /* verilator lint_off UNUSED */\n"
            "    wire _unused_ready_in_r = ready_in_r;\n"
            "    /* verilator lint_on UNUSED */\n"
            f"    assign ready_in = (state == {idle_name}) && !skid_block;\n\n"
            "endmodule"
        )
        n_end = text.count("\nendmodule")
        if n_end != 1:
            raise SystemExit(f"ANCHOR FAIL {f.name}: endmodule count {n_end}")
        text = text.replace("\nendmodule", "\n" + fix, 1)
        if n_port != 1 or n_wr < 3:
            raise SystemExit(
                f"ANCHOR FAIL {f.name}: port={n_port} writes={n_wr} (want 1, >=3)")
        bak = f.with_name(f.name + ".preengpipe")
        if not bak.exists():
            bak.write_text(orig, encoding="utf-8", newline="\n")
        f.write_text(text, encoding="utf-8", newline="\n")
        print(f"  [ok]   {f.name}: ADD-JOIN FIX (port={n_port}, writes={n_wr})")


# ============================================================================
# 3. engine_iso_wrap_mbv2.v — -DENG_PIPE + -DTHROTTLE build hooks
# ============================================================================
def patch_iso() -> None:
    patch(ISO, """        // [DW-ENGINE P1] mirror the MBV2 engine top: depthwise mode armed
        // (inert for dense dispatches — cfg 0x3C resets to 0).
        .ENABLE_DEPTHWISE(1)
    ) u_engine(
""", f"""        // [DW-ENGINE P1] mirror the MBV2 engine top: depthwise mode armed
        // (inert for dense dispatches — cfg 0x3C resets to 0).
        .ENABLE_DEPTHWISE(1),
`ifdef ENG_PIPE
        // {MARK} pipelined (pixel, oc_pass) issue under test (-DENG_PIPE).
        .ENG_PIPE(1),
`endif
`ifdef THROTTLE
        // {MARK} engine-output backpressure armed so the LFSR throttle below
        // exercises the stall/hold path (-DTHROTTLE; mirrors the MBV2 top's
        // ENABLE_OUTPUT_BACKPRESSURE(1) + eofifo_in_ready wiring).
        .ENABLE_OUTPUT_BACKPRESSURE(1),
`endif
        .MAC_COUNT(256)
    ) u_engine(
`ifdef THROTTLE
        .out_ready(thr_out_ready),
`endif
""", "iso ENG_PIPE/THROTTLE params")

    patch(ISO, """    // ---- the REAL engine, parameterised for mbv2 INT8 weight slots ----
    shared_engine #(
""", f"""`ifdef THROTTLE
    // {MARK} deterministic ~50% duty out_ready throttle (16b Fibonacci
    // LFSR, taps 16/14/13/11). Runs of 0s up to ~10 cycles exercise the
    // bridge hold + gap-hold (pend==2) + fire_recap paths.
    reg [15:0] thr_lfsr = 16'hACE1;
    always @(posedge clk) thr_lfsr <= {{thr_lfsr[14:0], thr_lfsr[15] ^ thr_lfsr[13] ^ thr_lfsr[12] ^ thr_lfsr[10]}};
    wire thr_out_ready = thr_lfsr[0];
`endif
    // ---- the REAL engine, parameterised for mbv2 INT8 weight slots ----
    shared_engine #(
""", "iso throttle LFSR")


def main() -> None:
    check = "--check" in sys.argv
    if check:
        # presence-of-marker check on all three files
        ok = all(MARK in p.read_text(encoding="utf-8") for p in (SKEL, TOP, ISO))
        print(f"[engpipe] check: {'APPLIED' if ok else 'NOT APPLIED'}")
        sys.exit(0 if ok else 1)
    print("[engpipe] patching shared_engine_skeleton.v ...")
    patch_skel()
    print("[engpipe] patching nn2rtl_top_engine.v ...")
    patch_top()
    patch_top_arb_commit()
    print("[engpipe] patching node_add_*.v (ADD-JOIN FIX) ...")
    patch_mbv2_adds()
    print("[engpipe] patching engine_iso_wrap_mbv2.v ...")
    patch_iso()
    print("[engpipe] done.")


if __name__ == "__main__":
    main()
