// n4 - ReLU6 (quantized requantize tail).
// op_type=relu, IC=OC=32, spatial=112x112, bus=256b, pipeline_latency=1.
// scale_factor = 441.6328531901041 -> SCALE_MULT/2^SCALE_SHIFT.
//
// DSP-FREE VARIANT: per-channel requant MULTIPLY replaced by a 128-entry
// precomputed ROM. Input domain is post-ReLU bytes (0..127), and the scale
// is a compile-time constant, so rom[x] is the EXACT result of the original
// requant expression for input x. 0 DSP; FSM/latency identical (1 cycle).
//
// PARAM-GATED ELASTIC BACKPRESSURE (ENABLE_BACKPRESSURE, default 0):
//   * ENABLE_BACKPRESSURE==0 (default): bit/cycle-IDENTICAL to the legacy
//     module. out_ready_in is IGNORED. ready_in is always 1, valid_out is a
//     1-cycle echo of valid_in (pipeline_latency=1). The per-module verify TB
//     does not drive the param (=0) -> byte-exact, NO harness change.
//   * ENABLE_BACKPRESSURE==1: TRUE 1-deep elastic skid per scratch/
//     elastic_relu.v. The computed beat is HELD in out_data until the
//     downstream takes it (out_full && out_ready_in). ready_in drops when a
//     beat is parked and the downstream is not ready, so the producer stalls
//     instead of dropping a beat. The requant value (rom[relu_byte]) is the
//     SAME expression as the legacy path, computed on the SAME admitted beat;
//     only the *timing* of valid_out changes, never the bytes.

module n4 #(
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
    localparam integer SCALE_SHIFT = 16;
    localparam integer MULT_W      = 26;
    localparam signed [MULT_W-1:0] SCALE_MULT = 26'sd28942851;
    localparam integer PROD_W      = 8 + MULT_W; // 34
    localparam signed [PROD_W-1:0] SCALE_ROUND_HALF =
        {{(PROD_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [PROD_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(PROD_W-1){1'b0}}, 1'b1};
    localparam signed [PROD_W-1:0] SAT_HI = 34'sd127;
    localparam signed [PROD_W-1:0] SAT_LO = -34'sd128;

    // 128-entry requant ROM: rom[x] = exact requant(x) for x in 0..127.
    // No DSP: constant-folded at elaboration (initial loop). Distributed LUT-RAM.
    (* rom_style = "distributed" *)
    reg signed [7:0] rom [0:127];

    integer k;
    reg signed [PROD_W-1:0] rom_scaled;
    reg signed [PROD_W-1:0] rom_vtmp;
    initial begin
        for (k = 0; k < 128; k = k + 1) begin
            rom_scaled = $signed(k[7:0]) * SCALE_MULT;
            // [INVARIANT:ROUNDING]
            rom_vtmp   = (rom_scaled +
                          (rom_scaled[PROD_W-1] ? SCALE_ROUND_HALF_M1
                                                : SCALE_ROUND_HALF)) >>> SCALE_SHIFT;
            rom[k]     = (rom_vtmp > SAT_HI) ?  8'sd127 :
                         (rom_vtmp < SAT_LO) ? -8'sd128 :
                                                rom_vtmp[7:0];
        end
    end

    // Shared combinational requant of the incoming beat (identical arithmetic
    // in both modes). rom[relu_byte] matches the legacy data_out expression.
    integer i;
    reg signed [7:0] relu_byte;
    reg  [255:0] requant_comb;
    always @(*) begin
        requant_comb = 256'd0;
        for (i = 0; i < OC; i = i + 1) begin
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
        // accept a new beat when the skid is empty, OR full but the downstream
        // is taking the parked beat this same cycle.
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
