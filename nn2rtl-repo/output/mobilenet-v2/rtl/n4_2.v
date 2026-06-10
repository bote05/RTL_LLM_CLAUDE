// n4_2 - ReLU6 (quantized requantize tail). ROM-BASED (0 DSP).
// op_type=relu, IC=OC=32, spatial=112x112, bus=256b, pipeline_latency=1.
// scale_factor = 12.842647552490234 -> SCALE_MULT=13151, SCALE_SHIFT=10
//   ratio = 13151/1024 = 12.8427734375 (relative err ~7.7e-7)
//
// The per-channel requant multiply has been replaced by a 128-entry ROM.
// Input bytes are post-ReLU (max(x,0)) so the relevant input domain is 0..127.
// rom[x] holds the EXACT result of the original requant expression for input x,
// computed at elaboration time with the identical arithmetic. This removes the
// 32-bit multiply+shift (and its DSP) while staying byte-exact. FSM/latency
// (pipeline_latency=1) is unchanged.
//
// PARAM-GATED ELASTIC BACKPRESSURE (ENABLE_BACKPRESSURE, default 0):
//   * ==0 (default): bit/cycle-IDENTICAL to the legacy module; out_ready_in is
//     IGNORED. The per-module verify TB (param=0) is byte-exact, no harness change.
//   * ==1: TRUE 1-deep elastic output skid (per scratch/elastic_relu.v); the
//     requant beat is HELD until the downstream takes it, ready_in drops while a
//     beat is parked and out_ready_in is low. Only valid_out *timing* changes.

module n4_2 #(
    parameter ENABLE_BACKPRESSURE = 0
)(
    input  wire          clk,
    input  wire          rst_n,
    input  wire          valid_in,
    output wire          ready_in,
    input  wire [255:0]  data_in,
    input  wire          out_ready_in,   // NEW: downstream-ready (ignored when ENABLE_BACKPRESSURE==0)
    output wire          valid_out,
    output wire [255:0]  data_out
);

    localparam integer OC          = 32;
    localparam integer SCALE_SHIFT = 10;
    localparam integer MULT_W      = 16;
    localparam signed [MULT_W-1:0] SCALE_MULT = 16'sd13151;
    localparam integer PROD_W      = 8 + MULT_W; // 24
    localparam signed [PROD_W-1:0] SCALE_ROUND_HALF =
        {{(PROD_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [PROD_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(PROD_W-1){1'b0}}, 1'b1};
    localparam signed [PROD_W-1:0] SAT_HI =  24'sd127;
    localparam signed [PROD_W-1:0] SAT_LO = -24'sd128;

    integer i;

    // --- Precomputed requant ROM: rom[x] = requant(x) for x in 0..127 ---
    // Indexed by the post-ReLU byte (always non-negative, 0..127), so a
    // 7-bit address (128 entries) fully covers the input domain. No DSP.
    (* rom_style = "distributed" *) reg [7:0] rom [0:127];

    integer                 jinit;
    reg signed [PROD_W-1:0] scaled_init;
    reg signed [PROD_W-1:0] vtmp_init;
    initial begin
        for (jinit = 0; jinit < 128; jinit = jinit + 1) begin
            // jinit is the post-ReLU byte value (0..127); identical arithmetic to
            // the original per-channel multiply datapath.
            scaled_init = $signed(jinit[7:0]) * SCALE_MULT;
            vtmp_init   = (scaled_init +
                            (scaled_init[PROD_W-1] ? SCALE_ROUND_HALF_M1
                                                   : SCALE_ROUND_HALF)) >>> SCALE_SHIFT;
            rom[jinit] = (vtmp_init > SAT_HI) ?  8'sd127 :
                         (vtmp_init < SAT_LO) ? -8'sd128 :
                                                 vtmp_init[7:0];
        end
    end

    // Shared combinational requant of the incoming beat (identical arithmetic
    // in both modes). rom[relu_byte] matches the legacy data_out expression.
    reg signed [7:0] relu_byte;
    reg  [255:0] requant_comb;
    always @(*) begin
        requant_comb = 256'd0;
        for (i = 0; i < OC; i = i + 1) begin
            // ReLU then ROM lookup (parallel across channels, all LUT-ROM).
            relu_byte = ($signed(data_in[i*8 +: 8]) > 8'sd0)
                          ? $signed(data_in[i*8 +: 8])
                          : 8'sd0;
            requant_comb[i*8 +: 8] = rom[relu_byte[6:0]];
        end
    end

    generate
    if (ENABLE_BACKPRESSURE == 0) begin : g_legacy
        // ---- LEGACY: bit/cycle-identical to the pre-backpressure module ----
        reg          valid_out_r;
        reg          ready_in_r;
        reg  [255:0] data_out_r;
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
        reg  [255:0] out_data;
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
