`timescale 1ns / 1ps

// n4_9 — ReLU6 requantize (clip_max=6, scale_factor=1.963112195332845)
// Per-channel: out = clamp( round( max(0,in) * (input_scale/output_scale) ), -128, 127 )
// Composite scale ratio = 1.96311... → SCALE_MULT=32'd23109, SCALE_SHIFT=5'd13.
// After ReLU the value is non-negative, so the clamp at +127 implements the
// clip-at-6 (output_scale = input_scale/1.963, so 128 LSBs == 6.0 float).

module n4_10 #(
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
    localparam integer SCALE_MULT  = 32'd23109;
    localparam integer SCALE_SHIFT = 5'd13;

    localparam integer SCALED_W = 32;

    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};
    localparam signed [SCALED_W-1:0] SCALE_MULT_CONST = SCALE_MULT;

    // ----------------------------------------------------------------------
    // DSP-FREE requantize: 128-entry lookup ROM.
    //
    // After ReLU the operand is strictly in 0..127 (max(in,0) on a signed
    // INT8 input). The per-tensor scale (SCALE_MULT / SCALE_SHIFT) is a
    // compile-time constant, so the entire multiply+round+shift+saturate
    // datapath is a pure function of the 7-bit relu_byte. We precompute the
    // EXACT result for every input x in 0..127 into requant_rom[x] and replace
    // the per-channel multiply with a parallel ROM lookup — 0 DSP.
    //
    // The ROM is populated by the identical expression used by the original
    // multiply datapath (incl. sign-aware rounding); for non-negative operands
    // the sign bit of `scaled` is 0, so the round term resolves to
    // SCALE_ROUND_HALF — matching the original bit-for-bit.
    // ----------------------------------------------------------------------
    (* rom_style = "distributed" *)
    reg signed [7:0] requant_rom [0:127];

    integer g;
    reg signed [SCALED_W-1:0] g_scaled;
    reg signed [SCALED_W-1:0] g_vtmp;
    initial begin
        for (g = 0; g < 128; g = g + 1) begin
            // relu_byte == g (already non-negative, 0..127)
            g_scaled = $signed({{(SCALED_W-8){1'b0}}, g[7:0]}) * SCALE_MULT_CONST;
            // [INVARIANT:ROUNDING]
            g_vtmp = (g_scaled +
                      (g_scaled[SCALED_W-1] ? SCALE_ROUND_HALF_M1
                                            : SCALE_ROUND_HALF)
                     ) >>> SCALE_SHIFT;
            requant_rom[g] = (g_vtmp >  127) ?  8'sd127 :
                             (g_vtmp < -128) ? -8'sd128 : g_vtmp[7:0];
        end
    end

    integer i;
    reg signed [7:0] in_byte;
    reg        [6:0] relu_byte;   // post-ReLU operand index, 0..127

    // Shared combinational requant of the incoming beat (identical arithmetic
    // in both modes). Lifted verbatim from the legacy per-channel datapath.
    reg  [1535:0] requant_comb;
    always @(*) begin
        requant_comb = 1536'd0;
        in_byte   = '0;   // [LATCH-FIX 2026-06-08] unconditional default (no inferred latch)
        relu_byte = '0;   // [LATCH-FIX2 2026-06-08] unconditional default (no inferred latch)
        if (valid_in) begin
        for (i = 0; i < OC; i = i + 1) begin
            in_byte   = $signed(data_in[i*8 +: 8]);
            // max(in,0): negative -> 0, else the 7-bit magnitude (in=127)
            relu_byte = (in_byte > 0) ? in_byte[6:0] : 7'd0;
            // parallel, DSP-free ROM lookup replaces multiply+shift+sat
            requant_comb[i*8 +: 8] = requant_rom[relu_byte];
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
        // [K1-MBV2] data_out_r is DATAPATH: consumed downstream only under
        // valid_out_r (reset-kept); written under valid_in (upstream valid
        // chain is reset-held at t=0). Sync-only write -> FDRE.
        always @(posedge clk) begin
            if (valid_in) data_out_r <= requant_comb;
        end
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                valid_out_r <= 1'b0;
                ready_in_r  <= 1'b1; // [INVARIANT:READY_IN_GATING]
            end else begin
                valid_out_r <= valid_in; // [INVARIANT:VALID_OUT_LATENCY]
                ready_in_r  <= 1'b1;     // [INVARIANT:READY_IN_GATING]
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
        // [K1-MBV2] out_data is skid DATA: consumed only under out_full
        // (reset-kept); written under accept && valid_in (control). -> FDRE.
        always @(posedge clk) begin
            if (accept && valid_in) out_data <= requant_comb;
        end
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                out_full <= 1'b0;
            end else begin
                if (out_full && out_ready_in)
                    out_full <= 1'b0;
                if (accept && valid_in) begin
                    out_full <= 1'b1;
                end
            end
        end
    end
    endgenerate

endmodule
