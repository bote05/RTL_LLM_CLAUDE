// n4_7 - ReLU6 (quantized requantize tail).
// op_type=relu, IC=OC=144, spatial=56x56, bus=1152b, pipeline_latency=1.
// scale_factor = 5.086697578430176 -> SCALE_MULT=16'sd20835, SCALE_SHIFT=5'd12.
//   20835 / 2^12 = 5.086669921875 (rel.err ~ 5.4e-6)
//
// DSP-FREE ROM VARIANT:
//   The per-channel requantize input is the post-ReLU byte, which is strictly
//   in 0..127 (8-bit signed clamped to >=0). SCALE_MULT/SCALE_SHIFT are
//   compile-time constants, so the entire requant expression is a pure function
//   of a 7-bit index. We precompute a 128-entry x 8-bit ROM whose entry rom[x]
//   equals the EXACT result of the original multiply/round/shift/saturate
//   expression for input x. Each channel then performs rom[relu_byte] instead of
//   a signed multiply -> 0 DSP, byte-exact, identical FSM/latency.

module n4_7 #(
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
    localparam signed [MULT_W-1:0]  SCALE_MULT_CONST = 16'sd20835;
    localparam integer PROD_W      = 8 + MULT_W; // 24
    localparam signed [PROD_W-1:0]  SCALE_ROUND_HALF =
        {{(PROD_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [PROD_W-1:0]  SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(PROD_W-1){1'b0}}, 1'b1};
    localparam signed [PROD_W-1:0]  SAT_HI =  24'sd127;
    localparam signed [PROD_W-1:0]  SAT_LO = -24'sd128;

    // 128-entry requant ROM: rom[x] = requant(x) for x in 0..127, 0 DSP.
    (* rom_style = "distributed" *)
    reg [7:0] rom [0:127];

    integer j;
    reg signed [7:0]        rb;
    reg signed [PROD_W-1:0] sc;
    reg signed [PROD_W-1:0] vt;
    initial begin
        for (j = 0; j < 128; j = j + 1) begin
            // index j is the post-ReLU byte (always >= 0).
            rb = j[7:0];
            sc = $signed(rb) * SCALE_MULT_CONST;
            // [INVARIANT:ROUNDING] (identical expression to the multiply path)
            vt = (sc +
                   (sc[PROD_W-1] ? SCALE_ROUND_HALF_M1
                                 : SCALE_ROUND_HALF)) >>> SCALE_SHIFT;
            rom[j] = (vt > SAT_HI) ?  8'sd127 :
                     (vt < SAT_LO) ? -8'sd128 :
                                      vt[7:0];
        end
    end

    integer i;
    reg signed [7:0]        relu_byte;

    // Shared combinational requant of the incoming beat (identical arithmetic
    // in both modes). Lifted verbatim from the legacy per-channel datapath.
    reg  [1151:0] requant_comb;
    always @(*) begin
        requant_comb = 1152'd0;
        if (valid_in) begin
        for (i = 0; i < OC; i = i + 1) begin
            relu_byte = ($signed(data_in[i*8 +: 8]) > 8'sd0)
                          ? $signed(data_in[i*8 +: 8])
                          : 8'sd0;
            // ROM lookup replaces the per-channel multiply/round/shift/sat.
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
