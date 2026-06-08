// n4_19: ReLU6 with requantize tail (clip_max=6.0, scale_factor=1.384429136912028).
// Pattern reference: knowledge/patterns/protected/06_relu.md (ReLU6 / clipped activations).
// Upstream conv emits INT8 at a coarser scale; this module applies the ReLU
// non-linearity (negatives -> 0) then requantises from the conv's output_scale
// to ReLU6's tighter output_scale (= 6/128), saturating to INT8.
//
// DSP-elimination variant: the per-channel requant MULTIPLY+SHIFT+SATURATE has
// been replaced by a 128-entry lookup ROM. The post-ReLU byte is strictly in
// 0..127 (relu_byte = max(in,0) for an 8-bit signed input), and the scale is a
// compile-time constant, so rom[x] precomputes the EXACT result of the original
// requant expression for every legal input x. This is byte-exact and uses 0 DSP.

module n4_19 #(
    parameter ENABLE_BACKPRESSURE = 0
)(
    input  wire                  clk,
    input  wire                  rst_n,        // active-low reset
    input  wire                  valid_in,
    output wire                   ready_in,
    input  wire [3071:0]         data_in,
    input  wire          out_ready_in,   // NEW: downstream-ready (ignored when ENABLE_BACKPRESSURE==0)
    output wire                   valid_out,
    output wire  [3071:0]         data_out
);

    localparam integer OC          = 384;
    localparam integer SCALE_MULT  = 32'd8999;     // round(1.384429136912028 * 2^14)
    localparam integer SCALE_SHIFT = 5'd13;
    localparam integer SCALED_W    = 32;

    localparam signed [SCALED_W-1:0] SCALE_MULT_CONST    = SCALE_MULT;
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF    = 32'sd1 <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};

    integer                       i;

    // ---- 128-entry requant ROM (0 DSP) -------------------------------------
    // rom[x] = clamp( ( x*SCALE_MULT + round ) >>> SCALE_SHIFT , -128, 127 )
    // for x in 0..127 (the only values relu_byte can take). Computed offline at
    // elaboration with the SAME expression as the original multiply datapath.
    (* rom_style = "distributed" *)
    reg signed [7:0]              rom [0:127];

    integer                       g;
    reg signed [7:0]              g_relu;
    reg signed [SCALED_W-1:0]     g_scaled;
    reg signed [SCALED_W-1:0]     g_vtmp;
    initial begin
        for (g = 0; g < 128; g = g + 1) begin
            g_relu   = g[7:0];                              // 0..127, always >= 0
            g_scaled = g_relu * SCALE_MULT_CONST;
            // [INVARIANT:ROUNDING] (identical to original)
            g_vtmp   = (g_scaled +
                        (g_scaled[SCALED_W-1] ? SCALE_ROUND_HALF_M1
                                              : SCALE_ROUND_HALF)
                       ) >>> SCALE_SHIFT;
            rom[g]   = (g_vtmp > 127)  ?  8'sd127 :
                       (g_vtmp < -128) ? -8'sd128 : g_vtmp[7:0];
        end
    end

    // ---- per-channel ReLU + ROM lookup -------------------------------------
    reg [6:0]                     relu_idx;   // clamped post-ReLU byte index 0..127

    // Shared combinational requant of the incoming beat (identical arithmetic
    // in both modes). Lifted verbatim from the legacy per-channel datapath.
    reg  [3071:0] requant_comb;
    always @(*) begin
        requant_comb = 3072'd0;
        relu_idx  = '0;   // [LATCH-FIX 2026-06-08] unconditional default (no inferred latch)
        if (valid_in) begin
        for (i = 0; i < OC; i = i + 1) begin
            relu_idx = ($signed(data_in[i*8 +: 8]) > 0)
                         ? data_in[i*8 +: 7]   // positive -> low 7 bits == value (0..127)
                         : 7'd0;
            requant_comb[i*8 +: 8] = rom[relu_idx];
        end
        end
    end

    generate
    if (ENABLE_BACKPRESSURE == 0) begin : g_legacy
        // ---- LEGACY: bit/cycle-identical to the pre-backpressure module ----
        reg          valid_out_r;
        reg          ready_in_r;
        reg  [3071:0] data_out_r;
        assign valid_out = valid_out_r;
        assign ready_in  = ready_in_r;
        assign data_out  = data_out_r;
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                valid_out_r <= 1'b0;
                ready_in_r  <= 1'b1; // [INVARIANT:READY_IN_GATING]
                data_out_r  <= 3072'd0;
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
        reg  [3071:0] out_data;
        wire accept = (!out_full || out_ready_in);
        assign ready_in  = accept;
        assign valid_out = out_full;
        assign data_out  = out_data;
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                out_full <= 1'b0;
                out_data <= 3072'd0;
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
