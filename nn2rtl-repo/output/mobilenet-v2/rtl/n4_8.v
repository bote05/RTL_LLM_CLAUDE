// n4_8 - ReLU6 (quantized requantize tail).
// op_type=relu, IC=OC=144, spatial=28x28, bus=1152b, pipeline_latency=1.
// scale_factor = 2.604058583577474 -> SCALE_MULT=5333, SCALE_SHIFT=11
//   ratio = 5333/2048 = 2.60400390625 (relative err ~2.1e-5).
//
// ROM VARIANT: the per-channel requant MULTIPLY+shift+round+saturate is
// replaced by a 128-entry x 8-bit lookup. The post-ReLU input domain is
// strictly 0..127 (negatives clamped to 0), and SCALE_MULT/SCALE_SHIFT are
// compile-time constants, so the entire requant is a pure function of the
// 7-bit relu_byte. rom[x] is precomputed with the IDENTICAL requant
// expression as the original multiply datapath -> byte-exact, 0 DSP.
// FSM/latency unchanged (pipeline_latency=1, valid_out <= valid_in).

module n4_8 #(
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
    localparam integer SCALE_SHIFT = 11;
    localparam integer MULT_W      = 16;
    localparam signed [MULT_W-1:0] SCALE_MULT_CONST = 16'sd5333;
    localparam integer PROD_W      = 8 + MULT_W; // 24
    localparam signed [PROD_W-1:0] SCALE_ROUND_HALF =
        {{(PROD_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [PROD_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(PROD_W-1){1'b0}}, 1'b1};
    localparam signed [PROD_W-1:0] SAT_HI =  24'sd127;
    localparam signed [PROD_W-1:0] SAT_LO = -24'sd128;

    // ---- Precomputed requant ROM (replaces the per-channel multiply) ----
    // rom[x] = exact original requant output for relu_byte == x, x in 0..127.
    (* rom_style = "distributed" *)
    reg [7:0] rom [0:127];

    integer ri;
    reg signed [7:0]        rb_init;
    reg signed [PROD_W-1:0] scaled_init;
    reg signed [PROD_W-1:0] vtmp_init;
    initial begin
        for (ri = 0; ri < 128; ri = ri + 1) begin
            // relu_byte domain is 0..127; index ri IS the relu_byte value.
            rb_init     = ri[7:0];
            scaled_init = $signed(rb_init) * SCALE_MULT_CONST;
            // [INVARIANT:ROUNDING] - identical expression to original datapath.
            vtmp_init   = (scaled_init +
                            (scaled_init[PROD_W-1] ? SCALE_ROUND_HALF_M1
                                                   : SCALE_ROUND_HALF)) >>> SCALE_SHIFT;
            rom[ri]     = (vtmp_init > SAT_HI) ?  8'sd127 :
                          (vtmp_init < SAT_LO) ? -8'sd128 :
                                                  vtmp_init[7:0];
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
            // ReLU clamp: negatives -> 0, leaving a 7-bit (0..127) index.
            relu_byte = ($signed(data_in[i*8 +: 8]) > 8'sd0)
                          ? $signed(data_in[i*8 +: 8])
                          : 8'sd0;
            // Parallel ROM lookup (LUT/BRAM, 0 DSP) replaces multiply+shift.
            requant_comb[i*8 +: 8] = rom[relu_byte[6:0]];
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
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                valid_out_r <= 1'b0;
                ready_in_r  <= 1'b1; // [INVARIANT:READY_IN_GATING]
                data_out_r  <= 1152'd0;
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
        reg  [1151:0] out_data;
        wire accept = (!out_full || out_ready_in);
        assign ready_in  = accept;
        assign valid_out = out_full;
        assign data_out  = out_data;
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                out_full <= 1'b0;
                out_data <= 1152'd0;
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
