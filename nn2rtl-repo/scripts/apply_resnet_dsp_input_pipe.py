#!/usr/bin/env python3
"""apply_resnet_dsp_input_pipe.py (2026-06-05)

Fmax fix: the routed ResNet-50 critical path is conv_298 k_group_reg -> lbw chan_window
mux / weights ROM -> DSP MAC input, ONE pipeline stage that is ~98% ROUTE (~27ns) because
the line-buffer memory and the DSP column are physically far apart (90%+ BRAM/URAM density).
Splitting that route needs a pipeline register => +1 MAC latency.

Scope workflow (wh6kq7v17) proved a UNIFORM +1 is SAFE: residual-add joins are
handshake-elastic with occupancy-sized FIFOs; a uniform slow-down preserves all relative
arm phases (the MP-increase deadlock was the opposite sign — a relative speed-UP). e2e
value golden stays valid (values are timing-independent).

This script (idempotent, backed up):
  PART 1 — rtl_library/conv_datapath_mp_k.v: add default-OFF param DSP_INPUT_PIPE and a
    1-deep retiming stage (weight_word_q->weight_word_q2, tap_q->tap_q2) before the
    combinational multiply, plus a matching valid/oc-group delay (q1->q1b->q2) so the
    accumulate gate stays aligned. OFF (default) => q2 aliases q1 combinationally =>
    byte- AND latency-identical (MobileNet + all non-piped instances unaffected).
  PART 2 — enable .DSP_INPUT_PIPE(1) on every ResNet spatial node_conv (those that
    instantiate conv_datapath_mp_k), uniformly.

Byte-exact: same products / tree-sum order / acc / scale; only +1 cycle start_mac->valid_out.
Verify: npx tsx scripts/run_nn2rtl_top_value.ts 0  (expect PASS mismatch=0). Then bump
golden_impl CONV_PIPELINE_STAGES + regen sidecars (atomic-change hygiene), then synth.
"""
import sys, os, glob, time, shutil

REPO = r"C:\Users\User\Desktop\RTL_LLM_CLAUDE\nn2rtl-repo"
DP = os.path.join(REPO, "rtl_library", "conv_datapath_mp_k.v")
NODE_GLOB = os.path.join(REPO, "output", "rtl", "node_conv_*.v")

# ---------------- PART 1: conv_datapath_mp_k.v edits ----------------
DP_EDITS = [
# HUNK A — new default-off parameter (make USE_CHAN_WINDOW non-last with a comma)
("    parameter integer USE_CHAN_WINDOW = 0\n) (",
 "    parameter integer USE_CHAN_WINDOW = 0,\n"
 "    // [DSP-INPUT-PIPE] 1 => insert ONE extra register stage on the multiplier inputs\n"
 "    // (weight_word_q->q2, tap_q->q2) + a matching valid delay. Breaks the long\n"
 "    // lbw/ROM->DSP route (Fmax). +1 latency. 0 = exact current behavior + latency.\n"
 "    parameter integer DSP_INPUT_PIPE = 0\n) ("),

# HUNK B — declare the optional 2nd-stage regs + companion valid/oc-group delay
("    reg [WIDE_W-1:0]    weight_word_q;\n    reg signed [7:0]    tap_q [0:MP_K-1];",
 "    reg [WIDE_W-1:0]    weight_word_q;\n    reg signed [7:0]    tap_q [0:MP_K-1];\n"
 "    // [DSP-INPUT-PIPE] optional 2nd input-register stage. DSP_INPUT_PIPE=0 => these\n"
 "    // alias q1 combinationally (zero-cost, exact behavior + latency).\n"
 "    reg [WIDE_W-1:0]    weight_word_q2;\n"
 "    reg signed [7:0]    tap_q2 [0:MP_K-1];\n"
 "    reg                    mac_valid_q1b;\n"
 "    reg [OC_GROUP_W-1:0]   mac_oc_group_q1b;\n"
 "    integer p2_i;"),

# HUNK C — drive the 2nd stage (registered when ON, combinational alias when OFF)
("    endgenerate\n\n    // ---- Stage 2: MP × MP_K parallel multipliers, tree-sum per lane ----",
 "    endgenerate\n"
 "    // [DSP-INPUT-PIPE] 2nd input-register stage (q2). ON: real reg (+1 latency, breaks\n"
 "    // the lbw/ROM->DSP route). OFF: combinational alias of q1 (identical behavior+latency).\n"
 "    generate\n"
 "    if (DSP_INPUT_PIPE != 0) begin : g_dsp_pipe\n"
 "        always @(posedge clk) begin\n"
 "            weight_word_q2 <= weight_word_q;\n"
 "            for (p2_i = 0; p2_i < MP_K; p2_i = p2_i + 1)\n"
 "                tap_q2[p2_i] <= tap_q[p2_i];\n"
 "        end\n"
 "    end else begin : g_dsp_pipe_off\n"
 "        always @* begin\n"
 "            weight_word_q2 = weight_word_q;\n"
 "            for (p2_i = 0; p2_i < MP_K; p2_i = p2_i + 1)\n"
 "                tap_q2[p2_i] = tap_q[p2_i];\n"
 "        end\n"
 "    end\n"
 "    endgenerate\n\n"
 "    // ---- Stage 2: MP × MP_K parallel multipliers, tree-sum per lane ----"),

# HUNK D — combinational multiply reads the _q2 regs
("                prod_w = $signed(weight_word_q[(cs_lane_i * MP_K + cs_kpos) * WGT_BITS +: WGT_BITS]) *\n                         $signed(tap_q[cs_kpos]);",
 "                prod_w = $signed(weight_word_q2[(cs_lane_i * MP_K + cs_kpos) * WGT_BITS +: WGT_BITS]) *\n                         $signed(tap_q2[cs_kpos]);"),

# HUNK E — matching valid/oc-group delay (3-deep when piped, 2-deep otherwise)
("            mac_valid_q2     <= mac_valid_q1;\n            mac_oc_group_q2  <= mac_oc_group_q1;",
 "            // [DSP-INPUT-PIPE] extra valid/oc-group delay when piped so the accumulate\n"
 "            // gate stays aligned with the deeper q2 data path.\n"
 "            if (DSP_INPUT_PIPE != 0) begin\n"
 "                mac_valid_q1b    <= mac_valid_q1;\n"
 "                mac_oc_group_q1b <= mac_oc_group_q1;\n"
 "                mac_valid_q2     <= mac_valid_q1b;\n"
 "                mac_oc_group_q2  <= mac_oc_group_q1b;\n"
 "            end else begin\n"
 "                mac_valid_q2     <= mac_valid_q1;\n"
 "                mac_oc_group_q2  <= mac_oc_group_q1;\n"
 "            end"),

# HUNK F1 — reset the new valid/oc-group regs
("            mac_oc_group_q1  <= 0;\n            mac_oc_group_q2  <= 0;",
 "            mac_oc_group_q1  <= 0;\n            mac_oc_group_q2  <= 0;\n"
 "            mac_valid_q1b    <= 1'b0;\n            mac_oc_group_q1b <= 0;"),

# HUNK F2 — drain guard waits for the deeper valid pipe (q1b term is const-0 when OFF)
("if (!mac_valid_q1 && !mac_valid_q2) begin",
 "if (!mac_valid_q1 && !mac_valid_q1b && !mac_valid_q2) begin"),

# HUNK F3 — clear q1b on ST_IDLE start (tidy flush; no-op when OFF)
("                        mac_valid_q1     <= 1'b0;\n                        mac_valid_q2     <= 1'b0;\n                        mac_done_issuing <= 1'b0;",
 "                        mac_valid_q1     <= 1'b0;\n                        mac_valid_q1b    <= 1'b0;\n                        mac_valid_q2     <= 1'b0;\n                        mac_done_issuing <= 1'b0;"),
]

NODE_OLD = "conv_datapath_mp_k #("
NODE_NEW = "conv_datapath_mp_k #(.DSP_INPUT_PIPE(1),"


def patch(path, edits, bk, marker):
    with open(path, "rb") as f:
        raw = f.read()
    crlf = b"\r\n" in raw
    # latin-1 round-trips any byte 0..255 losslessly; anchors are pure ASCII so
    # they still match. Avoids UnicodeDecodeError on cp1252 em-dashes in some files.
    text = raw.decode("latin-1")
    if crlf:
        text = text.replace("\r\n", "\n")
    if marker in text:
        print(f"SKIP {os.path.basename(path)}: already has {marker}")
        return True
    work = text
    for i, (old, new) in enumerate(edits):
        c = work.count(old)
        if c != 1:
            print(f"FAIL {os.path.basename(path)} edit#{i}: anchor count={c} (need 1): {old[:55]!r}")
            return False
        work = work.replace(old, new, 1)
    shutil.copy2(path, os.path.join(bk, os.path.basename(path)))
    out = work.replace("\n", "\r\n") if crlf else work
    with open(path, "wb") as f:
        f.write(out.encode("latin-1"))
    print(f"OK   {os.path.basename(path)}: applied {len(edits)} edit(s)")
    return True


def main():
    bk = os.path.join(REPO, "backups", f"resnet_dsp_input_pipe_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(bk, exist_ok=True)
    ok = True
    # PART 1: shared datapath module
    ok &= patch(DP, DP_EDITS, bk, "[DSP-INPUT-PIPE]")
    if not ok:
        print("-" * 50); print("FAILED in conv_datapath_mp_k.v — aborting (no node edits)"); sys.exit(1)
    # PART 2: enable on every spatial node_conv (those that instantiate conv_datapath_mp_k)
    n_nodes = 0
    for f in sorted(glob.glob(NODE_GLOB)):
        with open(f, "rb") as fh:
            raw = fh.read()
        if b"conv_datapath_mp_k" not in raw:
            continue  # engine-only wrapper, no spatial datapath
        n_nodes += 1
        ok &= patch(f, [(NODE_OLD, NODE_NEW)], bk, ".DSP_INPUT_PIPE(")
    print("-" * 50)
    print(f"spatial node_conv files enabled: {n_nodes}")
    print(f"backups -> {bk}")
    print("ALL OK" if ok else "FAILED (see above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
