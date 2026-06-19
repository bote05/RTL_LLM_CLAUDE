#!/usr/bin/env python3
"""
apply_resnet8_fsm_pipeline.py -- pixel-pipeline the ResNet-8 spatial conv FSM.

Transforms a conv (node_conv2d_*.v emitted by apply_resnet8_kpar_convs.py) from
the serial per-pixel FSM (ST_IDLE->ST_MAC[issue+drain]->ST_BIAS->ST_SCALE->
ST_OUTPUT->ST_IDLE, II~18 for conv_1) into a BANKED PIPELINE:

  * Work-item = (pixel, oc_group). A pixel = OC_PASSES work-items, all sharing one
    HELD window (pixel_active). Each work-item issues its K_GROUPS into a bank.
  * Two acc banks (acc_b0/acc_b1) alternate per work-item: the next work-item issues
    into the idle bank while the prior one drains + requants in the background.
  * Valid chain (depth n_valid=6, UNCHANGED) also carries per-partial tags:
    bank (mac_bank_q*), last-k_group (mac_last_q*), and oc_group (mac_oc_q*).
  * Decoupled 3-stage requant pipeline (BIAS->SCALE->OUTPUT) reads a completed bank,
    indexed by the carried oc; valid_out fires only on the last oc_group.
  * mac_busy_w = pixel_active||pending so the scheduler releases the window as soon
    as a pixel's work-items are issued (II -> ~OC_PASSES*K_GROUPS + small overhead).

WHY byte-exact: the DATA PATH (stage1 weight/tap load, stage2 multiply + reduction
tree) is UNTOUCHED, and the valid-chain DEPTH (n_valid=6) is UNCHANGED -- each
k_group's partial still emerges exactly n_valid cycles after issue and accumulates
EXACTLY once, now into its work-item's bank. A bank is reallocated only after its
requant has read it (bank_busy*), so reset never races an in-flight accumulate.
=> avoids the #1 silent-corruption risk (lengthening data_latency). The OC_PASSES==1
path reduces to the validated conv_1/conv_2 pipeline (regression check).

SCOPE: ResNet-8 ONLY (these node_conv2d_*.v are resnet8-specific; resnet50/mbv2 use
the shared engine). Gate: run_resnet8_top_value.ts must stay 8/8 mismatch=0.
Idempotent (skips if already pipelined); backs up to .prefsmpipe.

Requires n_valid==6 (mac_valid_q1..q6) -- true for all current ResNet-8 convs.
"""
import re
import sys
from pathlib import Path

RTL_DIR = Path(__file__).resolve().parent.parent / "output" / "resnet8" / "rtl"

PIPELINE_MIDS = ["node_conv2d_1", "node_conv2d_2",
                 "node_conv2d_4", "node_conv2d_5",
                 "node_conv2d_7", "node_conv2d_8"]

MARKER = "// [PIPELINE] Banked pipeline (OC_PASSES>=1)."

# ---- EDIT 1: acc declaration -> two banks + pipeline control regs ----
E1_OLD = """    reg signed [ACC_W-1:0]    acc    [0:MP-1];
    reg signed [BIASED_W-1:0] biased [0:MP-1];"""
E1_NEW = """    // [PIPELINE] Two accumulator banks (acc_b0/acc_b1) alternate per work-item.
    reg signed [ACC_W-1:0]    acc_b0 [0:MP-1];
    reg signed [ACC_W-1:0]    acc_b1 [0:MP-1];
    reg                       issuing;             // issuing K_GROUPS of a work-item
    reg                       pending;             // fired pixel awaiting first work-item
    reg                       pixel_active;        // window held across the pixel's OC_PASSES work-items
    reg                       ib;                  // issue bank (0/1)
    reg                       bank_busy0, bank_busy1;
    reg                       rq_v1, rq_bank1, rq_v2, rq_v3;        // 3-stage requant pipe
    reg [OC_GROUP_W-1:0]      rq_oc1, rq_oc2, rq_oc3;               // oc_group carried through requant
    reg signed [BIASED_W-1:0] biased [0:MP-1];"""

# ---- EDIT 2: mac_busy_w gating ----
E2_OLD = """    assign mac_busy_w = (state != ST_IDLE);"""
E2_NEW = """    assign mac_busy_w = pixel_active || pending;  // [PIPELINE] hold window across pixel's work-items"""

# ---- EDIT 3: valid chain decls -> valid + bank + last + oc tags ----
E3_OLD = """    reg                       mac_valid_q1;
    reg [OC_GROUP_W-1:0]      mac_oc_group_q1;
    reg                       mac_valid_q2;
    reg [OC_GROUP_W-1:0]      mac_oc_group_q2;
    reg                       mac_valid_q3;
    reg [OC_GROUP_W-1:0]      mac_oc_group_q3;
    reg                       mac_valid_q4;
    reg [OC_GROUP_W-1:0]      mac_oc_group_q4;
    reg                       mac_valid_q5;
    reg [OC_GROUP_W-1:0]      mac_oc_group_q5;
    reg                       mac_valid_q6;
    reg [OC_GROUP_W-1:0]      mac_oc_group_q6;
    reg                       mac_done_issuing;
    integer p_i;"""
E3_NEW = """    // [PIPELINE] valid chain (depth n_valid=6) carries valid + bank + last + oc tags.
    reg mac_valid_q1; reg mac_bank_q1; reg mac_last_q1; reg [OC_GROUP_W-1:0] mac_oc_q1;
    reg mac_valid_q2; reg mac_bank_q2; reg mac_last_q2; reg [OC_GROUP_W-1:0] mac_oc_q2;
    reg mac_valid_q3; reg mac_bank_q3; reg mac_last_q3; reg [OC_GROUP_W-1:0] mac_oc_q3;
    reg mac_valid_q4; reg mac_bank_q4; reg mac_last_q4; reg [OC_GROUP_W-1:0] mac_oc_q4;
    reg mac_valid_q5; reg mac_bank_q5; reg mac_last_q5; reg [OC_GROUP_W-1:0] mac_oc_q5;
    reg mac_valid_q6; reg mac_bank_q6; reg mac_last_q6; reg [OC_GROUP_W-1:0] mac_oc_q6;
    integer p_i;"""

# ---- EDIT 4: the whole FSM always block -> banked pipeline. {W} = data_out width. ----
E4_OLD = """    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= ST_IDLE;
            valid_out_r      <= 1'b0;
            data_out_r       <= {W}'d0;
            k_group          <= 0;
            oc_group         <= 0;
            mac_valid_q1     <= 1'b0;
            mac_oc_group_q1  <= 0;
            mac_valid_q2     <= 1'b0;
            mac_oc_group_q2  <= 0;
            mac_valid_q3     <= 1'b0;
            mac_oc_group_q3  <= 0;
            mac_valid_q4     <= 1'b0;
            mac_oc_group_q4  <= 0;
            mac_valid_q5     <= 1'b0;
            mac_oc_group_q5  <= 0;
            mac_valid_q6     <= 1'b0;
            mac_oc_group_q6  <= 0;
            mac_done_issuing <= 1'b0;
            for (i = 0; i < MP; i = i + 1) begin
                acc[i]        <= 0;
                biased[i]     <= 0;
                scaled[i]     <= 0;
                shift_lane[i] <= 0;
            end
        end else begin
            valid_out_r <= 1'b0;

            // Stage 3: valid-chain propagation + gated accumulate into MP lanes.
            // The valid chain is deepened to match the data path's reduction
            // latency so each k_group's partial is accumulated EXACTLY once.
            mac_valid_q2    <= mac_valid_q1;
            mac_oc_group_q2 <= mac_oc_group_q1;
            mac_valid_q3    <= mac_valid_q2;
            mac_oc_group_q3 <= mac_oc_group_q2;
            mac_valid_q4    <= mac_valid_q3;
            mac_oc_group_q4 <= mac_oc_group_q3;
            mac_valid_q5    <= mac_valid_q4;
            mac_oc_group_q5 <= mac_oc_group_q4;
            mac_valid_q6    <= mac_valid_q5;
            mac_oc_group_q6 <= mac_oc_group_q5;
            if (mac_valid_q6) begin
                for (p_i = 0; p_i < MP; p_i = p_i + 1)
                    acc[p_i] <= acc[p_i] + $signed(lane_partial[p_i]);
            end

            case (state)
                ST_IDLE: begin
                    if (sched_output_fires) begin
                        state            <= ST_MAC;
                        k_group          <= 0;
                        oc_group         <= 0;
                        mac_valid_q1     <= 1'b0;
                        mac_valid_q2     <= 1'b0;
                        mac_valid_q3     <= 1'b0;
                        mac_valid_q4     <= 1'b0;
                        mac_valid_q5     <= 1'b0;
                        mac_valid_q6     <= 1'b0;
                        mac_done_issuing <= 1'b0;
                        for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1)
                            acc[lane_i] <= 0;
                    end
                end

                ST_MAC: begin
                    if (mac_done_issuing) begin
                        mac_valid_q1 <= 1'b0;
                        if (!mac_valid_q1 && !mac_valid_q2 && !mac_valid_q3 && !mac_valid_q4 && !mac_valid_q5 && !mac_valid_q6) begin
                            mac_done_issuing <= 1'b0;
                            state            <= ST_BIAS;
                        end
                    end else begin
                        mac_oc_group_q1 <= oc_group;
                        mac_valid_q1    <= 1'b1;
                        if (k_group == K_GROUPS - 1) begin
                            mac_done_issuing <= 1'b1;
                        end else begin
                            k_group <= k_group + 1'b1;
                        end
                    end
                end

                ST_BIAS: begin
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                        bias_oc = oc_group * MP + lane_i;
                        biased[lane_i] <= $signed(acc[lane_i]) + $signed(biases_mem[bias_oc]);
                    end
                    state <= ST_SCALE;
                end

                ST_SCALE: begin
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                        scaled[lane_i]     <= $signed(biased[lane_i]) *
                                              $signed(scale_mult_rom[oc_group * MP + lane_i]);
                        shift_lane[lane_i] <= scale_shift_rom[oc_group * MP + lane_i];
                    end
                    state <= ST_OUTPUT;
                end

                ST_OUTPUT: begin
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                        out_oc = oc_group * MP + lane_i;
                        // [INVARIANT:ROUNDING] single positive bias + arith >>> = golden floor.
                        v_tmp = (scaled[lane_i] +
                                 ($signed({{(SCALED_W-1){1'b0}}, 1'b1}) <<< (shift_lane[lane_i] - 1))
                                ) >>> shift_lane[lane_i];
                        data_out_r[out_oc*8 +: 8] <=
                            (v_tmp >  127) ?  8'sd127 :
                            (v_tmp < -128) ? -8'sd128 : v_tmp[7:0];
                    end

                    if (oc_group == OC_PASSES - 1) begin
                        valid_out_r <= 1'b1;
                        state       <= ST_IDLE;
                    end else begin
                        oc_group         <= oc_group + 1'b1;
                        k_group          <= 0;
                        mac_valid_q1     <= 1'b0;
                        mac_valid_q2     <= 1'b0;
                        mac_valid_q3     <= 1'b0;
                        mac_valid_q4     <= 1'b0;
                        mac_valid_q5     <= 1'b0;
                        mac_valid_q6     <= 1'b0;
                        mac_done_issuing <= 1'b0;
                        for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1)
                            acc[lane_i] <= 0;
                        state <= ST_MAC;
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end"""
E4_NEW = """    // [PIPELINE] Banked pipeline (OC_PASSES>=1). A pixel = OC_PASSES work-items
    // (one per oc_group), all sharing one held window. Each work-item issues its
    // K_GROUPS into a bank; the next work-item issues into the idle bank while this
    // one drains + requants in the background. II -> ~OC_PASSES*K_GROUPS + overhead
    // (vs the serial OC_PASSES*(K_GROUPS + drain 5 + requant 3 + idle/sched)).
    // Data path + valid-chain depth (n_valid=6) UNCHANGED -> byte-exact.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out_r  <= 1'b0;
            data_out_r   <= {W}'d0;
            k_group      <= 0;
            oc_group     <= 0;
            issuing      <= 1'b0;
            pending      <= 1'b0;
            pixel_active <= 1'b0;
            ib           <= 1'b0;
            bank_busy0   <= 1'b0;
            bank_busy1   <= 1'b0;
            rq_v1<=1'b0; rq_bank1<=1'b0; rq_oc1<=0;
            rq_v2<=1'b0; rq_oc2<=0;
            rq_v3<=1'b0; rq_oc3<=0;
            mac_valid_q1<=1'b0; mac_bank_q1<=1'b0; mac_last_q1<=1'b0; mac_oc_q1<=0;
            mac_valid_q2<=1'b0; mac_bank_q2<=1'b0; mac_last_q2<=1'b0; mac_oc_q2<=0;
            mac_valid_q3<=1'b0; mac_bank_q3<=1'b0; mac_last_q3<=1'b0; mac_oc_q3<=0;
            mac_valid_q4<=1'b0; mac_bank_q4<=1'b0; mac_last_q4<=1'b0; mac_oc_q4<=0;
            mac_valid_q5<=1'b0; mac_bank_q5<=1'b0; mac_last_q5<=1'b0; mac_oc_q5<=0;
            mac_valid_q6<=1'b0; mac_bank_q6<=1'b0; mac_last_q6<=1'b0; mac_oc_q6<=0;
            for (i = 0; i < MP; i = i + 1) begin
                acc_b0[i]     <= 0;
                acc_b1[i]     <= 0;
                biased[i]     <= 0;
                scaled[i]     <= 0;
                shift_lane[i] <= 0;
            end
        end else begin
            valid_out_r <= 1'b0;
            rq_v1       <= 1'b0;

            // latch a newly-fired output pixel (scheduler holds the window while busy).
            if (sched_output_fires) pending <= 1'b1;

            // ---- valid/bank/last/oc chain shift (depth = n_valid = 6) ----
            mac_valid_q2<=mac_valid_q1; mac_bank_q2<=mac_bank_q1; mac_last_q2<=mac_last_q1; mac_oc_q2<=mac_oc_q1;
            mac_valid_q3<=mac_valid_q2; mac_bank_q3<=mac_bank_q2; mac_last_q3<=mac_last_q2; mac_oc_q3<=mac_oc_q2;
            mac_valid_q4<=mac_valid_q3; mac_bank_q4<=mac_bank_q3; mac_last_q4<=mac_last_q3; mac_oc_q4<=mac_oc_q3;
            mac_valid_q5<=mac_valid_q4; mac_bank_q5<=mac_bank_q4; mac_last_q5<=mac_last_q4; mac_oc_q5<=mac_oc_q4;
            mac_valid_q6<=mac_valid_q5; mac_bank_q6<=mac_bank_q5; mac_last_q6<=mac_last_q5; mac_oc_q6<=mac_oc_q5;
            mac_valid_q1 <= 1'b0;   // default; issue re-asserts below

            // ---- banked accumulate on the last valid stage (routed by bank tag) ----
            if (mac_valid_q6) begin
                if (!mac_bank_q6) begin
                    for (p_i = 0; p_i < MP; p_i = p_i + 1)
                        acc_b0[p_i] <= acc_b0[p_i] + $signed(lane_partial[p_i]);
                end else begin
                    for (p_i = 0; p_i < MP; p_i = p_i + 1)
                        acc_b1[p_i] <= acc_b1[p_i] + $signed(lane_partial[p_i]);
                end
                if (mac_last_q6) begin   // last k_group accumulated -> bank complete
                    rq_v1    <= 1'b1;
                    rq_bank1 <= mac_bank_q6;
                    rq_oc1   <= mac_oc_q6;
                end
            end

            // ---- decoupled requant pipeline: BIAS -> SCALE -> OUTPUT (oc-indexed) ----
            if (rq_v1) begin
                for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1)
                    biased[lane_i] <= $signed(rq_bank1 ? acc_b1[lane_i] : acc_b0[lane_i])
                                      + $signed(biases_mem[rq_oc1 * MP + lane_i]);
                if (!rq_bank1) bank_busy0 <= 1'b0; else bank_busy1 <= 1'b0;  // acc consumed -> free
            end
            rq_v2 <= rq_v1; rq_oc2 <= rq_oc1;
            if (rq_v2) begin
                for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                    scaled[lane_i]     <= $signed(biased[lane_i]) *
                                          $signed(scale_mult_rom[rq_oc2 * MP + lane_i]);
                    shift_lane[lane_i] <= scale_shift_rom[rq_oc2 * MP + lane_i];
                end
            end
            rq_v3 <= rq_v2; rq_oc3 <= rq_oc2;
            if (rq_v3) begin
                for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                    out_oc = rq_oc3 * MP + lane_i;
                    // [INVARIANT:ROUNDING] single positive bias + arith >>> = golden floor.
                    v_tmp = (scaled[lane_i] +
                             ($signed({{(SCALED_W-1){1'b0}}, 1'b1}) <<< (shift_lane[lane_i] - 1))
                            ) >>> shift_lane[lane_i];
                    data_out_r[out_oc*8 +: 8] <=
                        (v_tmp >  127) ?  8'sd127 :
                        (v_tmp < -128) ? -8'sd128 : v_tmp[7:0];
                end
                if (rq_oc3 == OC_PASSES - 1) valid_out_r <= 1'b1;  // fire only after last oc_group
            end

            // ---- issue engine: one k_group/cycle; work-items chained via pixel_active ----
            if (issuing) begin
                mac_valid_q1 <= 1'b1;
                mac_bank_q1  <= ib;
                mac_oc_q1    <= oc_group;
                mac_last_q1  <= (k_group == K_GROUPS - 1);
                if (k_group == K_GROUPS - 1) begin
                    if (oc_group == OC_PASSES - 1) begin
                        issuing <= 1'b0; ib <= ~ib; pixel_active <= 1'b0;  // pixel done -> release window
                    end else if ((ib && !bank_busy0) || (!ib && !bank_busy1)) begin
                        // [PIPELINE] continue DIRECTLY into the other bank -- no inter-work-item bubble
                        oc_group <= oc_group + 1'b1;
                        ib       <= ~ib;
                        k_group  <= 0;
                        if (ib) begin   // next bank = ~ib = 0
                            bank_busy0 <= 1'b1;
                            for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) acc_b0[lane_i] <= 0;
                        end else begin  // next bank = ~ib = 1
                            bank_busy1 <= 1'b1;
                            for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) acc_b1[lane_i] <= 0;
                        end
                    end else begin
                        issuing <= 1'b0; ib <= ~ib;  // next bank busy -> fall back to pixel_active stall
                    end
                end else begin
                    k_group <= k_group + 1'b1;
                end
            end else if (pixel_active) begin
                // start the next work-item of the SAME pixel (oc_group+1) into bank ib.
                if ((!ib && !bank_busy0) || (ib && !bank_busy1)) begin
                    issuing  <= 1'b1;
                    oc_group <= oc_group + 1'b1;
                    k_group  <= 0;
                    if (!ib) begin
                        bank_busy0 <= 1'b1;
                        for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) acc_b0[lane_i] <= 0;
                    end else begin
                        bank_busy1 <= 1'b1;
                        for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) acc_b1[lane_i] <= 0;
                    end
                end
            end else if (pending) begin
                // start a new pixel: oc_group 0 into bank ib, when free.
                if ((!ib && !bank_busy0) || (ib && !bank_busy1)) begin
                    issuing      <= 1'b1;
                    pending      <= 1'b0;
                    pixel_active <= 1'b1;
                    oc_group     <= 0;
                    k_group      <= 0;
                    if (!ib) begin
                        bank_busy0 <= 1'b1;
                        for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) acc_b0[lane_i] <= 0;
                    end else begin
                        bank_busy1 <= 1'b1;
                        for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) acc_b1[lane_i] <= 0;
                    end
                end
            end
        end
    end"""


def apply_one(mid: str) -> bool:
    path = RTL_DIR / f"{mid}.v"
    txt = path.read_text()
    if MARKER in txt:
        print(f"{mid}: already pipelined; skip")
        return False
    if "mac_valid_q6" not in txt or "mac_valid_q7" in txt:
        raise SystemExit(f"{mid}: expected n_valid==6 (mac_valid_q1..q6) -- ABORT")
    m = re.search(r"data_out_r\s+<=\s+(\d+)'d0;", txt)
    if not m:
        raise SystemExit(f"{mid}: could not find data_out_r reset width -- ABORT")
    w = m.group(1)
    edits = [(E1_OLD, E1_NEW), (E2_OLD, E2_NEW), (E3_OLD, E3_NEW),
             (E4_OLD.replace("{W}", w), E4_NEW.replace("{W}", w))]
    for i, (old, new) in enumerate(edits):
        n = txt.count(old)
        if n != 1:
            raise SystemExit(f"{mid}: edit #{i} matched {n} times (expected 1) -- structure drift, ABORT")
        txt = txt.replace(old, new)
    bak = path.with_suffix(".v.prefsmpipe")
    if not bak.exists():
        bak.write_text(path.read_text())
    path.write_text(txt)
    print(f"{mid}: pixel-pipelined (banked, data_out={w}b)")
    return True


if __name__ == "__main__":
    mids = sys.argv[1:] or PIPELINE_MIDS
    changed = sum(apply_one(m) for m in mids)
    print(f"done: {changed} conv(s) pipelined")
