// n4_6 - ReLU6 (quantized requantize tail).
// op_type=relu, IC=OC=144, spatial=56x56, bus=1152b, pipeline_latency=1.
// scale_factor = 5.733251571655273 -> SCALE_MULT=16'sd23483, SCALE_SHIFT=5'd12.
//
// DSP-elimination: the per-channel requant MULTIPLY is replaced by a 128-entry
// ROM. Post-ReLU bytes are strictly in [0,127] (max(in,0) on a signed INT8),
// and SCALE_MULT/SCALE_SHIFT are compile-time constants, so the entire
// multiply -> round -> shift -> saturate chain has a fixed 128-entry domain.
// REQUANT_ROM[x] is precomputed (in an initial loop using the SAME expression)
// to the exact INT8 output, making this byte-exact with the multiplier version
// while using 0 DSP. The ROM is read combinationally per channel (parallel,
// LUT/distributed) and the result is registered exactly as before, so the
// FSM and 1-cycle pipeline latency are unchanged.

module n4_6 #(
    parameter ENABLE_BACKPRESSURE = 0
)(
    input  wire           clk,
    input  wire           rst_n,
    input  wire           valid_in,
    output wire            ready_in,
    input  wire [1151:0]  data_in,
    input  wire          out_ready_in,   // NEW: downstream-ready (ignored when ENABLE_BACKPRESSURE==0)
    output wire            valid_out,
    output wire  [1151:0]  data_out
);

    localparam integer OC          = 144;
    localparam integer SCALE_SHIFT = 5'd12;
    localparam integer MULT_W      = 16;
    localparam signed [MULT_W-1:0]  SCALE_MULT_CONST = 16'sd23483;
    localparam integer PROD_W      = 8 + MULT_W; // 24
    localparam signed [PROD_W-1:0]  SCALE_ROUND_HALF =
        {{(PROD_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [PROD_W-1:0]  SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(PROD_W-1){1'b0}}, 1'b1};
    localparam signed [PROD_W-1:0]  SAT_HI =  24'sd127;
    localparam signed [PROD_W-1:0]  SAT_LO = -24'sd128;

    // 128-entry requant ROM: REQUANT_ROM[x] = exact INT8 requant of post-ReLU x.
    // Stored as 8-bit (two's-complement) bytes; held in LUT/distributed RAM.
    (* rom_style = "distributed" *)
    reg [7:0] REQUANT_ROM [0:127];

    integer j;
    reg signed [7:0]        rb;
    reg signed [PROD_W-1:0] sc;
    reg signed [PROD_W-1:0] vt;
    initial begin
        for (j = 0; j < 128; j = j + 1) begin
            rb = j[7:0]; // post-ReLU byte, domain 0..127 (always non-negative)
            sc = $signed(rb) * SCALE_MULT_CONST;
            // [INVARIANT:ROUNDING] -- identical sign-aware rounding
            vt = (sc +
                   (sc[PROD_W-1] ? SCALE_ROUND_HALF_M1
                                 : SCALE_ROUND_HALF)) >>> SCALE_SHIFT;
            REQUANT_ROM[j] = (vt > SAT_HI) ?  8'sd127 :
                             (vt < SAT_LO) ? -8'sd128 :
                                              vt[7:0];
        end
    end

    integer i;
    reg signed [7:0] relu_byte;

    // Shared combinational requant of the incoming beat (identical arithmetic
    // in both modes). Lifted verbatim from the legacy per-channel datapath.
    reg  [1151:0] requant_comb;
    always @(*) begin
        requant_comb = 1152'd0;
        relu_byte = '0;   // [LATCH-FIX 2026-06-08] unconditional default (no inferred latch)
        if (valid_in) begin
        for (i = 0; i < OC; i = i + 1) begin
            // ReLU: clamp signed INT8 to [0,127] -> 7-bit ROM index.
            relu_byte = ($signed(data_in[i*8 +: 8]) > 8'sd0)
                          ? $signed(data_in[i*8 +: 8])
                          : 8'sd0;
            // Requant via ROM lookup (0 DSP), registered as before.
            requant_comb[i*8 +: 8] = REQUANT_ROM[relu_byte[6:0]];
        end
        end
    end

    generate
    if (ENABLE_BACKPRESSURE == 0) begin : g_legacy
        // ---- LEGACY: bit/cycle-identical to the pre-backpressure module ----
        reg          valid_out_r;
        reg          ready_in_r;
        reg  [1151:0] data_out_r;
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
        reg  [1151:0] out_data;
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
