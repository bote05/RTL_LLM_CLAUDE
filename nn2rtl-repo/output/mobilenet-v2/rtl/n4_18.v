// n4_18: ReLU6 with requantize tail (clip_max=6.0, scale_factor=1.384429136912028).
// Pattern reference: knowledge/patterns/protected/06_relu.md (ReLU6 / clipped activations).
// Upstream conv emits INT8 at a coarser scale; this module applies the ReLU
// non-linearity (negatives -> 0) then requantises from the conv's output_scale
// to ReLU6's tighter output_scale (= 6/128), saturating to INT8.
//
// DSP-ELIMINATION VARIANT: the per-channel requant MULTIPLY is replaced by a
// 128-entry ROM. The post-ReLU byte domain is strictly 0..127 (negatives clamp
// to 0, the upstream INT8 max is 127), and SCALE_MULT/SCALE_SHIFT are
// compile-time constants, so the entire (multiply + round + arithmetic-shift +
// saturate) datapath is precomputed at elaboration into requant_rom[0:127].
// rom[x] holds the EXACT result of the original expression for input x, so the
// lookup is byte-exact while using 0 DSP (LUT/BRAM ROM, parallel across OC).

module n4_18 #(
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
    localparam integer SCALE_MULT  = 32'd12751;     // round(1.384429136912028 * 2^14)
    localparam integer SCALE_SHIFT = 5'd12;
    localparam integer SCALED_W    = 32;

    localparam signed [SCALED_W-1:0] SCALE_MULT_CONST    = SCALE_MULT;
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF    = 32'sd1 <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};

    integer                       i;

    // ---- Precomputed requant ROM (0 DSP). One 8-bit entry per post-ReLU byte. ----
    // Input domain is 0..127 (post-ReLU, non-negative INT8). Each entry is the
    // EXACT output of the original multiply/round/shift/saturate expression.
    (* rom_style = "distributed" *)
    reg signed [7:0] requant_rom [0:127];

    integer                       g;
    reg signed [7:0]              rom_relu_byte;
    reg signed [SCALED_W-1:0]     rom_scaled_tmp;
    reg signed [SCALED_W-1:0]     rom_v_tmp;
    initial begin
        for (g = 0; g < 128; g = g + 1) begin
            // relu_byte for a non-negative input x is just x (x>0 -> x, x==0 -> 0)
            rom_relu_byte  = g[7:0];
            rom_scaled_tmp = rom_relu_byte * SCALE_MULT_CONST;
            // [INVARIANT:ROUNDING] -- identical arithmetic to the original datapath
            rom_v_tmp = (rom_scaled_tmp +
                         (rom_scaled_tmp[SCALED_W-1] ? SCALE_ROUND_HALF_M1
                                                     : SCALE_ROUND_HALF)
                        ) >>> SCALE_SHIFT;
            requant_rom[g] = (rom_v_tmp > 127)  ?  8'sd127 :
                             (rom_v_tmp < -128) ? -8'sd128 : rom_v_tmp[7:0];
        end
    end

    reg signed [7:0]              relu_byte;

    // Shared combinational requant of the incoming beat (identical arithmetic
    // in both modes). Lifted verbatim from the legacy per-channel datapath.
    reg  [3071:0] requant_comb;
    always @(*) begin
        requant_comb = 3072'd0;
        relu_byte = '0;   // [LATCH-FIX 2026-06-08] unconditional default (no inferred latch)
        if (valid_in) begin
        for (i = 0; i < OC; i = i + 1) begin
            // ReLU: clamp negatives to 0 -> index in [0,127]
            relu_byte = ($signed(data_in[i*8 +: 8]) > 0)
                         ? $signed(data_in[i*8 +: 8])
                         : 8'sd0;
            // Byte-exact requant via precomputed ROM (0 DSP, parallel).
            requant_comb[i*8 +: 8] = requant_rom[relu_byte[6:0]];
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
        reg  [3071:0] out_data;
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
