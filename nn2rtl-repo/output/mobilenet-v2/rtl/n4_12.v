`timescale 1ns / 1ps

// n4_12 — ReLU6 requantize (clip_max=6, scale_factor=1.963112195332845)
// Per-channel: out = clamp( round( max(0,in) * (input_scale/output_scale) ), -128, 127 )
// Composite scale ratio = 1.96311... → SCALE_MULT=32'd25289, SCALE_SHIFT=5'd13.
// After ReLU the value is non-negative, so the clamp at +127 implements the
// clip-at-6 (output_scale = input_scale/1.963, so 128 LSBs == 6.0 float).
//
// DSP-FREE ROM VARIANT:
//   The per-channel datapath multiplied each post-ReLU byte by the compile-time
//   constant SCALE_MULT. Because the post-ReLU input domain is only 0..127
//   (non-negative INT8) and SCALE_MULT/SCALE_SHIFT are constants, the entire
//   requant expression is precomputed offline into a 128-entry x 8-bit ROM.
//   rom[x] = clamp( round( x * SCALE_MULT ) >>> SCALE_SHIFT, -128, 127 ) for
//   x in 0..127, using the IDENTICAL arithmetic (same widths, sign-aware round,
//   arithmetic shift, saturation). The runtime path becomes a pure table lookup
//   rom[relu_byte], which maps to LUT/BRAM and uses ZERO DSP. The FSM and the
//   1-cycle valid_out latency are unchanged. Byte-exact with the multiply form.

module n4_12 #(
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
    localparam integer SCALE_MULT  = 32'd25289;
    localparam integer SCALE_SHIFT = 5'd13;

    localparam integer SCALED_W = 32;

    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};
    localparam signed [SCALED_W-1:0] SCALE_MULT_CONST = SCALE_MULT;

    // ---------------------------------------------------------------------
    // Precomputed requant ROM: rom[x] = exact requant(x) for x in 0..127.
    // Built once at elaboration using the SAME expression as the original
    // multiply datapath, so the table is byte-identical. No DSP — the runtime
    // path is a lookup. Distributed/block ROM (synth-tool choice).
    // ---------------------------------------------------------------------
    (* rom_style = "distributed" *)
    reg signed [7:0] rom [0:127];

    integer ridx;
    reg signed [7:0]          rom_relu_byte;
    reg signed [SCALED_W-1:0] rom_scaled;
    reg signed [SCALED_W-1:0] rom_v_tmp;
    initial begin
        for (ridx = 0; ridx < 128; ridx = ridx + 1) begin
            // ridx is already the post-ReLU byte (always >= 0 here).
            rom_relu_byte = ridx[7:0];
            rom_scaled    = $signed({{(SCALED_W-8){rom_relu_byte[7]}}, rom_relu_byte})
                            * SCALE_MULT_CONST;
            // [INVARIANT:ROUNDING]
            rom_v_tmp = (rom_scaled +
                         (rom_scaled[SCALED_W-1] ? SCALE_ROUND_HALF_M1
                                                 : SCALE_ROUND_HALF)
                        ) >>> SCALE_SHIFT;
            rom[ridx] = (rom_v_tmp >  127) ?  8'sd127 :
                        (rom_v_tmp < -128) ? -8'sd128 : rom_v_tmp[7:0];
        end
    end

    integer i;
    reg signed [7:0]          in_byte;
    reg signed [7:0]          relu_byte;

    // Shared combinational requant of the incoming beat (identical arithmetic
    // in both modes). Lifted verbatim from the legacy per-channel datapath.
    reg  [1535:0] requant_comb;
    always @(*) begin
        requant_comb = 1536'd0;
        in_byte   = '0;   // [LATCH-FIX 2026-06-08] unconditional default (no inferred latch)
        relu_byte = '0;   // [LATCH-FIX 2026-06-08] unconditional default (no inferred latch)
        if (valid_in) begin
        for (i = 0; i < OC; i = i + 1) begin
            in_byte   = $signed(data_in[i*8 +: 8]);
            relu_byte = (in_byte > 0) ? in_byte : 8'sd0;
            // Pure ROM lookup — no per-channel multiply/shift, 0 DSP.
            requant_comb[i*8 +: 8] = rom[relu_byte[6:0]];
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
