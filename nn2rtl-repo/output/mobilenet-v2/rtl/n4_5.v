// n4_5 - ReLU6 (quantized requantize tail).
// op_type=relu, IC=OC=144, spatial=56x56, bus=1152b, pipeline_latency=1.
// scale_factor = 5.327365557352701 -> SCALE_MULT=21821, SCALE_SHIFT=12.
//
// DSP-elimination rewrite: the per-channel requant multiply+shift+saturate is a
// pure function of the post-ReLU byte, whose domain is only 0..127 (7-bit
// non-negative). It is precomputed into a 128-entry 8-bit ROM at elaboration
// time using the EXACT original expression, so the lookup is byte-exact while
// using 0 DSP. FSM/latency are unchanged (1-cycle registered output).

module n4_5 #(
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
    localparam integer SCALE_SHIFT = 12;
    localparam integer MULT_W      = 16;
    localparam signed [MULT_W-1:0]  SCALE_MULT_CONST = 16'sd21821;
    localparam integer PROD_W      = 8 + MULT_W; // 24
    localparam signed [PROD_W-1:0]  SCALE_ROUND_HALF =
        {{(PROD_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [PROD_W-1:0]  SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(PROD_W-1){1'b0}}, 1'b1};
    localparam signed [PROD_W-1:0]  SAT_HI =  24'sd127;
    localparam signed [PROD_W-1:0]  SAT_LO = -24'sd128;

    // ----------------------------------------------------------------------
    // Precomputed requant ROM: rom[x] = exact requant of post-ReLU byte x.
    // Domain of x is 0..127 (post-ReLU of an 8-bit signed input). The ROM is
    // built with the IDENTICAL arithmetic the multiply datapath used, so the
    // result is byte-exact by construction with 0 DSP.
    // ----------------------------------------------------------------------
    (* rom_style = "distributed" *) reg [7:0] requant_rom [0:127];

    integer                 g;
    reg signed [7:0]        g_byte;
    reg signed [PROD_W-1:0] g_scaled;
    reg signed [PROD_W-1:0] g_vtmp;
    initial begin
        for (g = 0; g < 128; g = g + 1) begin
            g_byte   = $signed(g[7:0]); // 0..127, always >= 0 (post-ReLU)
            g_scaled = $signed(g_byte) * SCALE_MULT_CONST;
            // [INVARIANT:ROUNDING]
            g_vtmp   = (g_scaled +
                          (g_scaled[PROD_W-1] ? SCALE_ROUND_HALF_M1
                                              : SCALE_ROUND_HALF)) >>> SCALE_SHIFT;
            requant_rom[g] = (g_vtmp > SAT_HI) ?  8'sd127 :
                             (g_vtmp < SAT_LO) ? -8'sd128 :
                                                  g_vtmp[7:0];
        end
    end

    integer i;
    reg signed [7:0] relu_byte;

    // Shared combinational requant of the incoming beat (identical arithmetic
    // in both modes). Lifted verbatim from the legacy per-channel datapath.
    reg  [1151:0] requant_comb;
    always @(*) begin
        requant_comb = 1152'd0;
        if (valid_in) begin
        for (i = 0; i < OC; i = i + 1) begin
            // clamp to [0, max] (ReLU); domain becomes 0..127
            relu_byte = ($signed(data_in[i*8 +: 8]) > 8'sd0)
                          ? $signed(data_in[i*8 +: 8])
                          : 8'sd0;
            // parallel ROM lookup replaces the per-channel multiply
            requant_comb[i*8 +: 8] = requant_rom[relu_byte[6:0]];
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
