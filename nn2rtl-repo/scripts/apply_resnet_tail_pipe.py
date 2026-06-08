#!/usr/bin/env python3
"""apply_resnet_tail_pipe.py (2026-06-05)

Fmax STEP 2: pipeline the requant TAIL of conv_datapath_mp_k. After the DSP-input
pipe + max_fanout fixes, the binding combinational stages become ST_OUTPUT (two
cascaded 49-bit barrel shifters, ~12-16ns) and ST_SCALE (scale_rom-read-into-33x16
multiply, ~8-11ns). Both > 10ns -> cap Fmax ~62-83 MHz. This splits them, byte-exact.

Adds default-OFF param TAIL_PIPE (consistent with DSP_INPUT_PIPE/USE_CHAN_WINDOW):
  - ST_BIAS prefetches+registers the per-OC scale (sc_mult_q/sc_shift_q) -> removes the
    combinational scale_rom-read-into-arith hazard.
  - ST_SCALE multiplies with the REGISTERED scale (byte-identical value).
  - ST_OUTPUT (round+add+shift+clip in one cycle) splits into ST_OUT_ROUND (round bias),
    ST_OUT_SHIFT (add + arithmetic right barrel shift), ST_OUT_SAT (saturate + advance).
  +2 latency when ON (BIAS,SCALE,OUTPUT=3 -> BIAS,SCALE,ROUND,SHIFT,SAT=5). OFF =
  byte- AND latency-identical (new states pruned; ST_SCALE->ST_OUTPUT unchanged).

Byte-exact: identical operators on identical operand values, only register boundaries
move. Verify: npx tsx scripts/run_nn2rtl_top_value.ts 0 (expect PASS mismatch=0, no
deadlock — elastic joins absorb uniform +latency). FIT preserved: only FFs added
(~480/conv x 45 ~= 22k FF, budget 3.456M), NO BRAM/URAM change. Idempotent + backed up.
"""
import sys, os, glob, time, shutil

REPO = r"C:\Users\User\Desktop\RTL_LLM_CLAUDE\nn2rtl-repo"
DP = os.path.join(REPO, "rtl_library", "conv_datapath_mp_k.v")
NODE_GLOB = os.path.join(REPO, "output", "rtl", "node_conv_*.v")

DP_EDITS = [
# HUNK A — TAIL_PIPE parameter
("    parameter integer DSP_INPUT_PIPE = 0\n) (",
 "    parameter integer DSP_INPUT_PIPE = 0,\n"
 "    // [TAIL-PIPE] 1 => pipeline the requant tail (prefetch scale_rom in ST_BIAS,\n"
 "    // register the multiply operands, split ST_OUTPUT into ROUND/SHIFT/SAT). +2\n"
 "    // latency, byte-identical. 0 = legacy (MobileNet + non-piped stay identical).\n"
 "    parameter integer TAIL_PIPE = 0\n) ("),

# HUNK B — new tail sub-states
("    localparam ST_OUTPUT = 3'd4;\n\n    reg [2:0] state;",
 "    localparam ST_OUTPUT = 3'd4;\n"
 "    // [TAIL-PIPE] extra requant-tail sub-stages (entered only when TAIL_PIPE!=0)\n"
 "    localparam ST_OUT_ROUND = 3'd5;\n"
 "    localparam ST_OUT_SHIFT = 3'd6;\n"
 "    localparam ST_OUT_SAT   = 3'd7;\n\n"
 "    reg [2:0] state;"),

# HUNK C — tail pipeline registers
("    reg signed [SCALED_W-1:0] scaled [0:MP-1];\n    reg signed [SCALED_W-1:0] v_tmp;",
 "    reg signed [SCALED_W-1:0] scaled [0:MP-1];\n    reg signed [SCALED_W-1:0] v_tmp;\n"
 "    // [TAIL-PIPE] requant-tail pipeline registers (used iff TAIL_PIPE!=0).\n"
 "    reg        [15:0]         sc_mult_q   [0:MP-1];\n"
 "    reg        [5:0]          sc_shift_q  [0:MP-1];\n"
 "    reg signed [SCALED_W-1:0] out_round_q [0:MP-1];\n"
 "    reg signed [SCALED_W-1:0] v_tmp_q     [0:MP-1];"),

# HUNK D — ST_BIAS prefetch+register the per-OC scale
("                        if (bias_oc < OC)\n"
 "                            biased[fsm_lane_i] <= $signed(acc[fsm_lane_i]) + $signed(biases[bias_oc]);\n"
 "                        else\n"
 "                            biased[fsm_lane_i] <= 0;\n"
 "                    end\n"
 "                    state <= ST_SCALE;",
 "                        if (bias_oc < OC)\n"
 "                            biased[fsm_lane_i] <= $signed(acc[fsm_lane_i]) + $signed(biases[bias_oc]);\n"
 "                        else\n"
 "                            biased[fsm_lane_i] <= 0;\n"
 "                        // [TAIL-PIPE] prefetch+register per-OC scale (kills the\n"
 "                        // combinational scale_rom-read-into-mult/shift hazard).\n"
 "                        if (TAIL_PIPE != 0 && bias_oc < OC) begin\n"
 "                            sc_mult_q[fsm_lane_i]  <= scale_rom[bias_oc][15:0];\n"
 "                            sc_shift_q[fsm_lane_i] <= scale_rom[bias_oc][21:16];\n"
 "                        end\n"
 "                    end\n"
 "                    state <= ST_SCALE;"),

# HUNK E — ST_SCALE: registered scale operand (ON) + conditional next-state
("                        if (sc_oc < OC)\n"
 "                            scaled[fsm_lane_i] <= $signed(biased[fsm_lane_i]) *\n"
 "                                                  $signed(scale_rom[sc_oc][15:0]);\n"
 "                        else\n"
 "                            scaled[fsm_lane_i] <= 0;\n"
 "                    end\n"
 "                    state <= ST_OUTPUT;",
 "                        if (sc_oc < OC)\n"
 "                            scaled[fsm_lane_i] <= $signed(biased[fsm_lane_i]) *\n"
 "                                                  (TAIL_PIPE != 0 ? $signed(sc_mult_q[fsm_lane_i])\n"
 "                                                                  : $signed(scale_rom[sc_oc][15:0]));\n"
 "                        else\n"
 "                            scaled[fsm_lane_i] <= 0;\n"
 "                    end\n"
 "                    state <= (TAIL_PIPE != 0) ? ST_OUT_ROUND : ST_OUTPUT;"),

# HUNK F — insert the 3 split tail states before `default:`
("                        state <= ST_MAC;\n"
 "                    end\n"
 "                end\n\n"
 "                default: state <= ST_IDLE;",
 "                        state <= ST_MAC;\n"
 "                    end\n"
 "                end\n\n"
 "                // [TAIL-PIPE] split requant tail (entered only when TAIL_PIPE!=0); byte-\n"
 "                // identical to ST_OUTPUT round/shift/clip, across 3 register-bounded cycles.\n"
 "                ST_OUT_ROUND: begin\n"
 "                    for (fsm_lane_i = 0; fsm_lane_i < MP; fsm_lane_i = fsm_lane_i + 1) begin\n"
 "                        out_oc = oc_group * MP + fsm_lane_i;\n"
 "                        if (out_oc < OC)\n"
 "                            out_round_q[fsm_lane_i] <= (sc_shift_q[fsm_lane_i] == 6'd0) ? {SCALED_W{1'b0}}\n"
 "                                : ({{(SCALED_W-1){1'b0}}, 1'b1} <<< (sc_shift_q[fsm_lane_i] - 6'd1));\n"
 "                    end\n"
 "                    state <= ST_OUT_SHIFT;\n"
 "                end\n\n"
 "                ST_OUT_SHIFT: begin\n"
 "                    for (fsm_lane_i = 0; fsm_lane_i < MP; fsm_lane_i = fsm_lane_i + 1) begin\n"
 "                        out_oc = oc_group * MP + fsm_lane_i;\n"
 "                        if (out_oc < OC)\n"
 "                            v_tmp_q[fsm_lane_i] <= (scaled[fsm_lane_i] + out_round_q[fsm_lane_i]) >>> sc_shift_q[fsm_lane_i];\n"
 "                    end\n"
 "                    state <= ST_OUT_SAT;\n"
 "                end\n\n"
 "                ST_OUT_SAT: begin\n"
 "                    for (fsm_lane_i = 0; fsm_lane_i < MP; fsm_lane_i = fsm_lane_i + 1) begin\n"
 "                        out_oc = oc_group * MP + fsm_lane_i;\n"
 "                        if (out_oc < OC)\n"
 "                            data_out[out_oc*8 +: 8] <=\n"
 "                                (v_tmp_q[fsm_lane_i] >  127) ?  8'sd127 :\n"
 "                                (v_tmp_q[fsm_lane_i] < -128) ? -8'sd128 : v_tmp_q[fsm_lane_i][7:0];\n"
 "                    end\n"
 "                    if (oc_group == OC_PASSES - 1) begin\n"
 "                        valid_out <= 1'b1;\n"
 "                        state     <= ST_IDLE;\n"
 "                    end else begin\n"
 "                        oc_group     <= oc_group + 1'b1;\n"
 "                        k_group      <= 0;\n"
 "                        for (fsm_lane_i = 0; fsm_lane_i < MP; fsm_lane_i = fsm_lane_i + 1)\n"
 "                            acc[fsm_lane_i] <= 0;\n"
 "                        state <= ST_MAC;\n"
 "                    end\n"
 "                end\n\n"
 "                default: state <= ST_IDLE;"),

# HUNK G — reset the new tail regs
("                partial_q[fsm_i] <= 0;\n            end",
 "                partial_q[fsm_i] <= 0;\n"
 "                sc_mult_q[fsm_i]   <= 16'd0;\n"
 "                sc_shift_q[fsm_i]  <= 6'd0;\n"
 "                out_round_q[fsm_i] <= 0;\n"
 "                v_tmp_q[fsm_i]     <= 0;\n            end"),
]

NODE_OLD = "conv_datapath_mp_k #(.DSP_INPUT_PIPE(1),"
NODE_NEW = "conv_datapath_mp_k #(.DSP_INPUT_PIPE(1),.TAIL_PIPE(1),"


def patch(path, edits, bk, marker):
    with open(path, "rb") as f:
        raw = f.read()
    crlf = b"\r\n" in raw
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
    bk = os.path.join(REPO, "backups", f"resnet_tail_pipe_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(bk, exist_ok=True)
    ok = patch(DP, DP_EDITS, bk, "[TAIL-PIPE]")
    if not ok:
        print("-" * 50); print("FAILED in conv_datapath_mp_k.v — aborting (no node edits)"); sys.exit(1)
    n_nodes = 0
    for f in sorted(glob.glob(NODE_GLOB)):
        with open(f, "rb") as fh:
            raw = fh.read()
        if b".DSP_INPUT_PIPE(1)" not in raw:
            continue  # only the spatial convs we already piped
        n_nodes += 1
        ok &= patch(f, [(NODE_OLD, NODE_NEW)], bk, ".TAIL_PIPE(")
    print("-" * 50)
    print(f"spatial node_conv files enabled: {n_nodes}")
    print(f"backups -> {bk}")
    print("ALL OK" if ok else "FAILED (see above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
