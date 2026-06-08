`timescale 1ns / 1ps

// n4_13 — ReLU6 requantize (clip_max=6, scale_factor=1.963112195332845)
// Per-channel: out = clamp( round( max(0,in) * (input_scale/output_scale) ), -128, 127 )
// Composite scale ratio = 1.96311... → SCALE_MULT=32'd9263, SCALE_SHIFT=5'd12.
// After ReLU the value is non-negative, so the clamp at +127 implements the
// clip-at-6 (output_scale = input_scale/1.963, so 128 LSBs == 6.0 float).
//
// DSP-FREE VARIANT: the requant input domain is the post-ReLU byte, which is
// strictly 0..127 (max(0,in) of a signed INT8). With SCALE_MULT/SCALE_SHIFT a
// compile-time constant, the entire multiply+round+shift+saturate is a pure
// function of that 7-bit value, so it is precomputed into a 128-entry ROM.
// rom[x] == EXACT result of the original requant expression for input x.
// Per-channel lookups are parallel (LUT/BRAM ROM), 0 DSP. FSM/latency unchanged.

module n4_13 #(
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
    localparam integer SCALE_MULT  = 32'd9263;
    localparam integer SCALE_SHIFT = 5'd12;

    localparam integer SCALED_W = 32;

    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};
    localparam signed [SCALED_W-1:0] SCALE_MULT_CONST = SCALE_MULT;

    // ---- 128-entry requant ROM (0 DSP) -------------------------------------
    // rom[x] = clamp( round( x * SCALE_MULT >> SCALE_SHIFT ), -128, 127 ) for
    // x in 0..127, computed with the IDENTICAL arithmetic as the multiply path.
    (* rom_style = "distributed" *) reg signed [7:0] rom [0:127];

    integer k;
    reg signed [SCALED_W-1:0] rom_scaled;
    reg signed [SCALED_W-1:0] rom_vtmp;
    initial begin
        for (k = 0; k < 128; k = k + 1) begin
            // x is always non-negative here (post-ReLU domain), so sign-extend
            // of the 8-bit value reduces to the value itself.
            rom_scaled = $signed(k[SCALED_W-1:0]) * SCALE_MULT_CONST;
            rom_vtmp   = (rom_scaled +
                          (rom_scaled[SCALED_W-1] ? SCALE_ROUND_HALF_M1
                                                  : SCALE_ROUND_HALF)
                         ) >>> SCALE_SHIFT;
            rom[k] = (rom_vtmp >  127) ?  8'sd127 :
                     (rom_vtmp < -128) ? -8'sd128 : rom_vtmp[7:0];
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
        if (valid_in) begin
        for (i = 0; i < OC; i = i + 1) begin
            in_byte  = $signed(data_in[i*8 +: 8]);
            // ReLU: negatives -> index 0 (== rom[0] == 0); else the byte.
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
