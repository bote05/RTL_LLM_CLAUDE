// n4_4 - ReLU6 (quantized requantize tail).
// op_type=relu, IC=OC=96, spatial=56x56, bus=768b, pipeline_latency=1.
// scale_factor = 4.232076327006022 -> SCALE_MULT=8667, SCALE_SHIFT=11
// (8667/2048 = 4.23193359375, rel_err ~3.4e-5).
//
// DSP-FREE VARIANT: the per-channel requant multiply is replaced by a
// 128-entry 8-bit ROM. The post-ReLU input is always in 0..127 (max(x,0)
// of a signed INT8), and SCALE_MULT/SCALE_SHIFT are compile-time constants,
// so rom[x] is precomputed to the EXACT result of the original requant
// expression for every legal input x. Lookups are pure LUT/BRAM (0 DSP) and
// run in parallel across the 96 channels. FSM and latency are unchanged
// (output registered 1 cycle after valid_in).

module n4_4 #(
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
    localparam integer MULT_W        = 15;
    localparam signed [MULT_W-1:0] SCALE_MULT = 15'sd8667;
    localparam integer PROD_W        = 8 + MULT_W + 1;  // 24
    localparam signed [PROD_W-1:0] SCALE_ROUND_HALF =
        {{(PROD_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [PROD_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(PROD_W-1){1'b0}}, 1'b1};
    localparam signed [PROD_W-1:0] SAT_HI =  24'sd127;
    localparam signed [PROD_W-1:0] SAT_LO = -24'sd128;

    // 128-entry requant ROM: rom[x] = exact INT8 requant of post-ReLU byte x.
    // Index is the post-ReLU value (0..127); 0 DSP, mapped to LUT/BRAM.
    (* rom_style = "distributed" *)
    reg signed [7:0] rom [0:127];

    integer ri;
    reg signed [PROD_W-1:0] rom_scaled;
    reg signed [PROD_W-1:0] rom_vtmp;
    initial begin
        for (ri = 0; ri < 128; ri = ri + 1) begin
            // ri is the post-ReLU byte (always >= 0), so use it directly.
            rom_scaled = $signed(ri[7:0]) * SCALE_MULT;
            // [INVARIANT:ROUNDING] - identical arithmetic to the multiply path.
            rom_vtmp   = (rom_scaled +
                           (rom_scaled[PROD_W-1] ? SCALE_ROUND_HALF_M1
                                                 : SCALE_ROUND_HALF)) >>> SCALE_SHIFT;
            rom[ri]    = (rom_vtmp > SAT_HI) ?  8'sd127 :
                         (rom_vtmp < SAT_LO) ? -8'sd128 :
                                                rom_vtmp[7:0];
        end
    end

    integer i;
    reg signed [7:0] in_byte;
    reg        [6:0] relu_idx;

    // Shared combinational requant of the incoming beat (identical arithmetic
    // in both modes). Lifted verbatim from the legacy per-channel datapath.
    reg  [767:0] requant_comb;
    always @(*) begin
        requant_comb = 768'd0;
        in_byte   = '0;   // [LATCH-FIX 2026-06-08] unconditional default (no inferred latch)
        if (valid_in) begin
        for (i = 0; i < OC; i = i + 1) begin
            in_byte  = $signed(data_in[i*8 +: 8]);
            // ReLU clamp -> 0..127, used as the 7-bit ROM index.
            relu_idx = (in_byte > 8'sd0) ? in_byte[6:0] : 7'd0;
            requant_comb[i*8 +: 8] = rom[relu_idx];
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
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                valid_out_r <= 1'b0;
                ready_in_r  <= 1'b1; // [INVARIANT:READY_IN_GATING]
                data_out_r  <= 768'd0;
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
        reg  [767:0] out_data;
        wire accept = (!out_full || out_ready_in);
        assign ready_in  = accept;
        assign valid_out = out_full;
        assign data_out  = out_data;
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                out_full <= 1'b0;
                out_data <= 768'd0;
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
