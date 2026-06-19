#!/usr/bin/env python3
"""
apply_resnet8_parallel_adds.py

Rewrite the three ResNet-8 residual-add modules (node_add_25/56/87) from the
generated per-channel SERIAL FSM into a fully-PIPELINED, FREE-RUNNING form
(all OC channels per cycle, fixed latency, ready_in tied high), preserving the
EXACT fused-scale arithmetic byte-for-byte.

WHY
---
The generated add is a per-channel serial FSM: it latches one beat in ST_IDLE,
then walks the OC channels one-per-cycle through a 3-stage MAC/round/sat pipe
(~OC+a few cycles per beat), dropping ready_in throughout. In ISOLATION (its
per-module TB respects ready_in) it is byte-exact. But the assembled ResNet-8
spatial chain is FREE-RUNNING: every conv/relu hard-wires `ready_in = 1'b1` and
emits one 128/256/512-bit beat PER CYCLE for the whole OUT_PIXELS frame, ignoring
any downstream ready_in. The serial add can only accept ~1 beat / (OC+k) cycles,
so the fast producers overrun it: the directly-wired main operand (conv2d_2/5/8)
beats fly past while the add is busy and are LOST, and the skip FIFO overflows.
Result: the add never assembles a full output frame -> the whole top DEADLOCKS
(all 1024 inputs consumed, zero output beats). MBV2 dodged this because its conv
modules are themselves per-pixel FSMs that drop ready_in (rate-matched), and its
skip FIFOs feed `out_ready = add.ready_in`. ResNet-8's convs are fully-pipelined
systolic (un-throttleable), so the ONLY rate-compatible add is a free-running
one.

ARITHMETIC (UNCHANGED — verified uniform across channels in every add):
    For channel c each cycle:
      lhs_term[c] = lhs[c] * FUSED_LHS_MULT      (signed 8 x MULT_W -> PROD_W)
      rhs_term[c] = rhs[c] * FUSED_RHS_MULT
      sum[c]      = lhs_term[c] + rhs_term[c] + FUSED_ROUND_BIAS   (SUM_W)
      v[c]        = sum[c] >>> FUSED_SHIFT                          (arith)
      out[c]      = (v > 127) ? 127 : (v < -128) ? -128 : v[7:0]
    Constants (FUSED_LHS_MULT, FUSED_RHS_MULT, FUSED_SHIFT, FUSED_ROUND_BIAS,
    SAT_HI/LO, OC, widths) are copied verbatim from the original module, so the
    per-beat result is bit-identical; only the SCHEDULE changes (all channels in
    parallel, one beat in -> one beat out, latency 3, throughput 1/cycle).

PIPELINE (3 stages, matches the original stage1/2/3 structure):
    s1: register lhs_term[c], rhs_term[c]      (valid v1)
    s2: register sum_term[c] = lhs+rhs+round    (valid v2)
    s3: register data_out[c] = saturate(sum>>>S) (valid_out)
  ready_in = 1'b1 always (free-running). data_in layout preserved:
    data_in[ i*8 +: 8]            = lhs (LOW half)
    data_in[ HALF + i*8 +: 8]     = rhs (HIGH half)
  where HALF = OUTPUT_WIDTH = OC*8.

This is a surgical whole-module replacement (idempotent: re-running detects the
free-running marker and skips). Per-module .preparallel backups are made by the
caller. Module name / port names / port widths are IDENTICAL to the original so
the top instantiation is untouched.
"""
import argparse
import os
import re
import sys

RTL_DIR = r"D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/resnet8/rtl"
ADDS = ["node_add_25", "node_add_56", "node_add_87"]
MARKER = "// [FREE-RUNNING PARALLEL ADD] rewritten by apply_resnet8_parallel_adds.py"


def _grab_int(text, name):
    m = re.search(r"localparam\s+integer\s+" + name + r"\s*=\s*([0-9]+)\s*;", text)
    if m:
        return int(m.group(1))
    return None


def _grab_signed_const(text, name):
    # localparam signed [W-1:0] NAME = [-]W'sdNNN;  (capture incl. optional sign)
    m = re.search(r"localparam\s+signed\s*\[[^\]]*\]\s*" + name + r"\s*=\s*(-?[0-9]+'s[dh][0-9A-Fa-f]+)\s*;", text)
    if m:
        return m.group(1)
    return None


def parse_params(text):
    p = {}
    p["OC"] = _grab_int(text, "OC")
    p["FUSED_SHIFT"] = _grab_int(text, "FUSED_SHIFT")
    p["MULT_W"] = _grab_int(text, "MULT_W")
    # PROD_W / SUM_W may be written as "8 + MULT_W" / "PROD_W + 2" — derive them.
    p["PROD_W"] = 8 + p["MULT_W"]
    p["SUM_W"] = p["PROD_W"] + 2
    p["FUSED_LHS_MULT"] = _grab_signed_const(text, "FUSED_LHS_MULT")
    p["FUSED_RHS_MULT"] = _grab_signed_const(text, "FUSED_RHS_MULT")
    p["FUSED_ROUND_BIAS"] = _grab_signed_const(text, "FUSED_ROUND_BIAS")
    p["SAT_HI"] = _grab_signed_const(text, "SAT_HI")
    p["SAT_LO"] = _grab_signed_const(text, "SAT_LO")
    missing = [k for k, v in p.items() if v is None]
    if missing:
        raise RuntimeError(f"could not parse params {missing}")
    return p


def gen_module(mod, p):
    OC = p["OC"]
    SUM_W = p["SUM_W"]
    PROD_W = p["PROD_W"]
    MULT_W = p["MULT_W"]
    SHIFT = p["FUSED_SHIFT"]
    IN_W = OC * 8 * 2
    OUT_W = OC * 8
    return f"""{MARKER}
// {mod} -- INT8 residual add, flat-bus, OC={OC}.
// FREE-RUNNING fully-parallel rewrite (byte-identical arithmetic to the
// generated serial FSM; see scripts/apply_resnet8_parallel_adds.py).
//   data_in[{OUT_W-1}:0]      = lhs ({OC} ch * 8b)
//   data_in[{IN_W-1}:{OUT_W}]  = rhs ({OC} ch * 8b)
//   data_out[{OUT_W-1}:0]     = saturated INT8 sum, {OC} channels packed
//   ready_in = 1; latency 3 cycles; throughput 1 beat/cycle.

module {mod} (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 valid_in,
    output wire                 ready_in,
    input  wire [{IN_W-1}:0] data_in,
    output reg                  valid_out,
    output reg  [{OUT_W-1}:0] data_out
);

    localparam integer OC          = {OC};
    localparam integer FUSED_SHIFT = {SHIFT};
    localparam integer MULT_W      = {MULT_W};
    localparam integer PROD_W      = {PROD_W};  // 8 + MULT_W
    localparam integer SUM_W       = {SUM_W};  // PROD_W + 2

    localparam signed [MULT_W-1:0] FUSED_LHS_MULT   = {p["FUSED_LHS_MULT"]};
    localparam signed [MULT_W-1:0] FUSED_RHS_MULT   = {p["FUSED_RHS_MULT"]};
    localparam signed [SUM_W-1:0]  FUSED_ROUND_BIAS = {p["FUSED_ROUND_BIAS"]};
    localparam signed [SUM_W-1:0]  SAT_HI           = {p["SAT_HI"]};
    localparam signed [SUM_W-1:0]  SAT_LO           = {p["SAT_LO"]};

    // free-running: never stall an un-throttleable systolic producer
    assign ready_in = 1'b1;  // [INVARIANT:READY_IN_GATING]

    // ---- stage 1: per-channel products (all OC in parallel) ----
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] lhs_term [0:OC-1];
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] rhs_term [0:OC-1];
    reg v1;
    // ---- stage 2: per-channel rounded sums ----
    reg signed [SUM_W-1:0] sum_term [0:OC-1];
    reg v2;

    integer i;
    reg signed [SUM_W-1:0] v_tmp;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1        <= 1'b0;
            v2        <= 1'b0;
            valid_out <= 1'b0;
            data_out  <= {OUT_W}'d0;
            for (i = 0; i < OC; i = i + 1) begin
                lhs_term[i] <= {{PROD_W{{1'b0}}}};
                rhs_term[i] <= {{PROD_W{{1'b0}}}};
                sum_term[i] <= {{SUM_W{{1'b0}}}};
            end
        end else begin
            // stage 1
            for (i = 0; i < OC; i = i + 1) begin
                lhs_term[i] <= $signed(data_in[i*8 +: 8])           * FUSED_LHS_MULT;
                rhs_term[i] <= $signed(data_in[{OUT_W} + i*8 +: 8]) * FUSED_RHS_MULT;
            end
            v1 <= valid_in;

            // stage 2: lhs+rhs+round  [INVARIANT:ROUNDING]
            for (i = 0; i < OC; i = i + 1)
                sum_term[i] <= $signed(lhs_term[i]) + $signed(rhs_term[i]) + FUSED_ROUND_BIAS;
            v2 <= v1;

            // stage 3: arithmetic shift + saturate
            for (i = 0; i < OC; i = i + 1) begin
                v_tmp = sum_term[i] >>> FUSED_SHIFT;
                data_out[i*8 +: 8] <= (v_tmp > SAT_HI) ? 8'sd127 :
                                      (v_tmp < SAT_LO) ? 8'h80   : v_tmp[7:0];
            end
            valid_out <= v2;  // [INVARIANT:VALID_OUT_LATENCY]
        end
    end

endmodule
"""


def main():
    ap = argparse.ArgumentParser(description="Rewrite ResNet-8 adds to free-running parallel form.")
    ap.add_argument("--rtl-dir", default=os.environ.get("NN2RTL_RTL_DIR") or RTL_DIR)
    args = ap.parse_args()

    report = []
    for mod in ADDS:
        path = os.path.join(args.rtl_dir, mod + ".v")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if MARKER in text:
            report.append((mod, "already-parallel"))
            continue
        p = parse_params(text)
        new_text = gen_module(mod, p)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_text)
        report.append((mod, f"rewritten OC={p['OC']} SHIFT={p['FUSED_SHIFT']} "
                            f"LHS={p['FUSED_LHS_MULT']} RHS={p['FUSED_RHS_MULT']}"))

    print("ResNet-8 parallel free-running adds:")
    for mod, state in report:
        print(f"  {mod:14s} [{state}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
