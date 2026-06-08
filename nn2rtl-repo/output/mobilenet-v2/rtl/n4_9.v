`timescale 1ns / 1ps

// n4_9 — ReLU6 requantize (clip_max=6, scale_factor=1.963112195332845)
// Per-channel: out = clamp( round( max(0,in) * (input_scale/output_scale) ), -128, 127 )
// Composite scale ratio = 1.96311... → SCALE_MULT=8041, SCALE_SHIFT=12.
// After ReLU the value is non-negative, so the clamp at +127 implements the
// clip-at-6 (output_scale = input_scale/1.963, so 128 LSBs == 6.0 float).
//
// DSP-ELIMINATION REWRITE:
// The post-ReLU input domain is strictly 0..127 (a 7-bit non-negative value,
// since max(0, signed8) >= 0). The requant is a multiply by a compile-time
// constant SCALE_MULT followed by round/shift/saturate. Because the input
// domain has only 128 distinct values and SCALE_MULT/SCALE_SHIFT are
// compile-time constants, the entire per-channel multiply datapath is
// replaced by a 128-entry x 8-bit ROM (rom[x] == exact requant(x)).
// The ROM is initialized at elaboration using the IDENTICAL requant
// expression, so the result is byte-exact. The multiply is gone → 0 DSP.
// FSM / latency are unchanged (1-cycle valid_out latency, ready_in held high).

module n4_9 #(
    parameter ENABLE_BACKPRESSURE = 0
)(
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  valid_in,
    output wire                   ready_in,
    input  wire [1535:0]         data_in,
    input  wire          out_ready_in,   // NEW: downstream-ready (ignored when ENABLE_BACKPRESSURE==0)
    output wire                   valid_out,
    output wire  [1535:0]         data_out
);

    localparam integer OC          = 192;
    localparam integer SCALE_MULT  = 8041;
    localparam integer SCALE_SHIFT = 12;

    localparam integer SCALED_W = 32;

    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};
    localparam signed [SCALED_W-1:0] SCALE_MULT_CONST = SCALE_MULT;

    // 128-entry requant ROM: rom[x] = exact requant(x) for relu_byte x in 0..127.
    // Distributed/LUT ROM (or BRAM) — no DSP, can be read in parallel per channel.
    (* rom_style = "distributed" *)
    reg signed [7:0] rom [0:127];

    integer ri;
    reg signed [SCALED_W-1:0] rscaled;
    reg signed [SCALED_W-1:0] rv_tmp;
    initial begin
        for (ri = 0; ri < 128; ri = ri + 1) begin
            // relu_byte == ri (0..127); replicate the original requant expression.
            rscaled = $signed({{(SCALED_W-8){1'b0}}, ri[7:0]}) * SCALE_MULT_CONST;
            // [INVARIANT:ROUNDING]
            rv_tmp = (rscaled +
                      (rscaled[SCALED_W-1] ? SCALE_ROUND_HALF_M1
                                           : SCALE_ROUND_HALF)
                     ) >>> SCALE_SHIFT;
            rom[ri] = (rv_tmp >  127) ?  8'sd127 :
                      (rv_tmp < -128) ? -8'sd128 : rv_tmp[7:0];
        end
    end

    integer i;
    reg signed [7:0] in_byte;
    reg        [6:0] relu_idx;

    // Shared combinational requant of the incoming beat (identical arithmetic
    // in both modes). Lifted verbatim from the legacy per-channel datapath.
    reg  [1535:0] requant_comb;
    always @(*) begin
        requant_comb = 1536'd0;
        in_byte   = '0;   // [LATCH-FIX 2026-06-08] unconditional default (no inferred latch)
        relu_idx  = '0;   // [LATCH-FIX2 2026-06-08] unconditional default (no inferred latch)
        if (valid_in) begin
        for (i = 0; i < OC; i = i + 1) begin
            in_byte  = $signed(data_in[i*8 +: 8]);
            // post-ReLU index: max(0,in) clamped to 0..127 (in is =127).
            relu_idx = (in_byte > 0) ? in_byte[6:0] : 7'd0;
            requant_comb[i*8 +: 8] = rom[relu_idx];
        end
        end
    end

    generate
    if (ENABLE_BACKPRESSURE == 0) begin : g_legacy
        // ---- LEGACY: bit/cycle-identical to the pre-backpressure module ----
        reg          valid_out_r;
        reg          ready_in_r;
        reg  [1535:0] data_out_r;
        assign valid_out = valid_out_r;
        assign ready_in  = ready_in_r;
        assign data_out  = data_out_r;
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                valid_out_r <= 1'b0;
                ready_in_r  <= 1'b1; // [INVARIANT:READY_IN_GATING]
                data_out_r  <= 1536'd0;
            end else begin
                valid_out_r <= valid_in; // [INVARIANT:VALID_OUT_LATENCY]
                ready_in_r  <= 1'b1;     // [INVARIANT:READY_IN_GATING]
                if (valid_in) begin
                    data_out_r <= requant_comb;
                end
            end
        end
    end else begin : g_bp
        // ---- ELASTIC: 1-deep output skid (per scratch/elastic_relu.v) ----
        reg          out_full;
        reg  [1535:0] out_data;
        wire accept = (!out_full || out_ready_in);
        assign ready_in  = accept;
        assign valid_out = out_full;
        assign data_out  = out_data;
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                out_full <= 1'b0;
                out_data <= 1536'd0;
            end else begin
                if (out_full && out_ready_in)
                    out_full <= 1'b0;
                if (accept && valid_in) begin
                    out_data <= requant_comb;
                    out_full <= 1'b1;
                end
            end
        end
    end
    endgenerate

endmodule
