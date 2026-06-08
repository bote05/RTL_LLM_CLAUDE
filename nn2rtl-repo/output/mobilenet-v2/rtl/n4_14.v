// n4_14 — ReLU6 (relu + requantize) for [1,192,14,14], flat-bus
// scale_factor = 2.762976964314779, clip_max = 6
// SCALE_MULT/SCALE_SHIFT chosen with MULT in [1,32767]:
//   SHIFT=12, MULT=11317  ->  11317/4096 = 2.76293945... (err ~ 1.36e-5)
//
// DSP-FREE REQUANT: the post-ReLU byte is strictly in 0..127, and the requant
// is a compile-time-constant affine map (multiply + rounding + arithmetic shift
// + saturation). We precompute the EXACT result for every possible input 0..127
// into a 128-entry x 8-bit ROM at elaboration time, then replace the per-channel
// multiply datapath with parallel ROM lookups. This is byte-exact with the
// original multiply path and eliminates all DSPs. FSM/latency are unchanged.
module n4_14 #(
    parameter ENABLE_BACKPRESSURE = 0
)(
    input  wire            clk,
    input  wire            rst_n,
    input  wire            valid_in,
    output wire             ready_in,
    input  wire [1535:0]   data_in,
    input  wire          out_ready_in,   // NEW: downstream-ready (ignored when ENABLE_BACKPRESSURE==0)
    output wire             valid_out,
    output wire  [1535:0]   data_out
);

    localparam integer OC          = 192;
    localparam integer SCALE_MULT  = 11317;
    localparam integer SCALE_SHIFT = 12;
    localparam integer SCALED_W    = 32;

    localparam signed [SCALED_W-1:0] SCALE_MULT_CONST = SCALE_MULT;
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};

    // 128-entry requant ROM: rom[x] = requant(x) for x in 0..127, byte-exact.
    (* rom_style = "distributed" *)
    reg [7:0] requant_rom [0:127];

    integer j;
    reg signed [SCALED_W-1:0] rom_scaled;
    reg signed [SCALED_W-1:0] rom_rounded;
    reg signed [SCALED_W-1:0] rom_clamped;
    initial begin
        for (j = 0; j < 128; j = j + 1) begin
            // x = j is the post-ReLU byte (always non-negative here).
            rom_scaled  = $signed(j[7:0]) * SCALE_MULT_CONST;
            // [INVARIANT:ROUNDING]
            rom_rounded = (rom_scaled +
                           (rom_scaled[SCALED_W-1] ? SCALE_ROUND_HALF_M1
                                                   : SCALE_ROUND_HALF)
                          ) >>> SCALE_SHIFT;
            rom_clamped = (rom_rounded >  32'sd127) ?  32'sd127 :
                          (rom_rounded < -32'sd128) ? -32'sd128 : rom_rounded;
            requant_rom[j] = rom_clamped[7:0];
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
            // ReLU: clamp negatives to 0, then index the requant ROM.
            relu_idx = (in_byte > 8'sd0) ? in_byte[6:0] : 7'd0;
            requant_comb[i*8 +: 8] = requant_rom[relu_idx];
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
