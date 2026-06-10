// n4_3 - ReLU6 (quantized requantize tail).
// op_type=relu, IC=OC=96, spatial=112x112, bus=768b, pipeline_latency=1.
// scale_factor = 13.995984395345053 -> SCALE_MULT=28664, SCALE_SHIFT=11
// (28664/2048 = 13.99609375, rel_err ~7.8e-6).
//
// DSP-ELIMINATION: the per-channel requant multiply has been replaced by a
// 128-entry ROM. Post-ReLU bytes are strictly in [0,127] (negatives clamp to
// 0 -> index 0), so rom[relu_byte] equals the EXACT requant expression for
// every reachable input. The ROM is populated in an `initial` loop using the
// identical multiply/round/shift/saturate expression, so it is byte-exact by
// construction. Multiplies by the compile-time SCALE_MULT constant fold away
// at elaboration -> 0 DSP. FSM/latency unchanged (1-cycle pipeline).

module n4_3 #(
    parameter ENABLE_BACKPRESSURE = 0
)(
    input  wire          clk,
    input  wire          rst_n,
    input  wire          valid_in,
    output wire           ready_in,
    input  wire [767:0]  data_in,
    input  wire          out_ready_in,   // NEW: downstream-ready (ignored when ENABLE_BACKPRESSURE==0)
    output wire           valid_out,
    output wire  [767:0]  data_out
);

    localparam integer OC            = 96;
    localparam integer SCALE_SHIFT   = 11;
    localparam integer MULT_W        = 17;
    localparam signed [MULT_W-1:0] SCALE_MULT = 17'sd28664;
    localparam integer PROD_W        = 8 + MULT_W + 1;  // 26
    localparam signed [PROD_W-1:0] SCALE_ROUND_HALF =
        {{(PROD_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [PROD_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(PROD_W-1){1'b0}}, 1'b1};
    localparam signed [PROD_W-1:0] SAT_HI =  26'sd127;
    localparam signed [PROD_W-1:0] SAT_LO = -26'sd128;

    integer i;

    // ---- precomputed requant ROM: rom[x] = requant(x) for x in 0..127 ----
    // Distributed-LUT ROM (small, parallel reads across all OC lanes).
    (* rom_style = "distributed" *)
    reg signed [7:0] rom [0:127];

    integer                 k;
    reg signed [7:0]        rb;
    reg signed [PROD_W-1:0] s_init;
    reg signed [PROD_W-1:0] v_init;
    initial begin
        for (k = 0; k < 128; k = k + 1) begin
            rb     = k[7:0];                 // relu_byte, always >= 0 here
            s_init = $signed(rb) * SCALE_MULT;
            // [INVARIANT:ROUNDING] - identical to original datapath
            v_init = (s_init +
                       (s_init[PROD_W-1] ? SCALE_ROUND_HALF_M1
                                         : SCALE_ROUND_HALF)) >>> SCALE_SHIFT;
            rom[k] = (v_init > SAT_HI) ?  8'sd127 :
                     (v_init < SAT_LO) ? -8'sd128 :
                                          v_init[7:0];
        end
    end

    reg signed [7:0] in_byte;
    reg signed [7:0] relu_byte;

    // Shared combinational requant of the incoming beat (identical arithmetic
    // in both modes). Lifted verbatim from the legacy per-channel datapath.
    reg  [767:0] requant_comb;
    always @(*) begin
        requant_comb = 768'd0;
        in_byte   = '0;   // [LATCH-FIX 2026-06-08] unconditional default (no inferred latch)
        relu_byte = '0;   // [LATCH-FIX 2026-06-08] unconditional default (no inferred latch)
        if (valid_in) begin
        for (i = 0; i < OC; i = i + 1) begin
            in_byte   = $signed(data_in[i*8 +: 8]);
            relu_byte = (in_byte > 8'sd0) ? in_byte : 8'sd0;
            // ROM lookup replaces the multiply/round/shift/saturate.
            requant_comb[i*8 +: 8] = rom[relu_byte];
        end
        end
    end

    generate
    if (ENABLE_BACKPRESSURE == 0) begin : g_legacy
        // ---- LEGACY: bit/cycle-identical to the pre-backpressure module ----
        reg          valid_out_r;
        reg          ready_in_r;
        reg  [767:0] data_out_r;
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
        reg  [767:0] out_data;
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
