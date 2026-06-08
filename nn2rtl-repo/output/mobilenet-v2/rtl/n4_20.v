// n4_15: ReLU6 with requantize tail (clip_max=6.0, scale_factor=1.384429136912028).
// Pattern reference: knowledge/patterns/protected/06_relu.md (ReLU6 / clipped activations).
// Upstream conv emits INT8 at a coarser scale; this module applies the ReLU
// non-linearity (negatives -> 0) then requantises from the conv's output_scale
// to ReLU6's tighter output_scale (= 6/128), saturating to INT8.

module n4_20 #(
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
    localparam integer SCALE_MULT  = 32'd13525;     // round(1.384429136912028 * 2^14)
    localparam integer SCALE_SHIFT = 5'd12;
    localparam integer SCALED_W    = 32;

    localparam signed [SCALED_W-1:0] SCALE_MULT_CONST    = SCALE_MULT;
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF    = 32'sd1 <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};

    // ----------------------------------------------------------------------
    // DSP-ELIMINATION: relu_byte is strictly in [0,127] (post-ReLU 7-bit), and
    // the requant constants are compile-time. The entire multiply+round+shift+
    // clamp chain is therefore a pure function of relu_byte over a 128-entry
    // domain. We precompute REQUANT_ROM[relu_byte] = the EXACT final INT8
    // result of the original expression, in an initial loop that evaluates the
    // SAME arithmetic. The per-channel multiply (the DSPs) is replaced by a
    // parallel ROM lookup. FSM, handshake, and latency unchanged.
    // ----------------------------------------------------------------------
    (* rom_style = "distributed" *)
    reg signed [7:0]              REQUANT_ROM [0:127];

    integer                       r;
    reg signed [SCALED_W-1:0]     rom_scaled;
    reg signed [SCALED_W-1:0]     rom_v_tmp;
    initial begin
        for (r = 0; r < 128; r = r + 1) begin
            // relu_byte == r (always >= 0 over this domain).
            rom_scaled = $signed(r[7:0]) * SCALE_MULT_CONST;
            // [INVARIANT:ROUNDING]
            rom_v_tmp  = (rom_scaled +
                          (rom_scaled[SCALED_W-1] ? SCALE_ROUND_HALF_M1
                                                  : SCALE_ROUND_HALF)
                         ) >>> SCALE_SHIFT;
            REQUANT_ROM[r] = (rom_v_tmp > 127)  ?  8'sd127 :
                             (rom_v_tmp < -128) ? -8'sd128 : rom_v_tmp[7:0];
        end
    end

    integer                       i;
    reg        [6:0]              relu_byte;

    // Shared combinational requant of the incoming beat (identical arithmetic
    // in both modes). Lifted verbatim from the legacy per-channel datapath.
    reg  [3071:0] requant_comb;
    always @(*) begin
        requant_comb = 3072'd0;
        if (valid_in) begin
        for (i = 0; i < OC; i = i + 1) begin
            relu_byte = ($signed(data_in[i*8 +: 8]) > 8'sd0)
                         ? data_in[i*8 +: 7]
                         : 7'd0;
            // [INVARIANT:ROUNDING] requant via precomputed ROM
            requant_comb[i*8 +: 8] = REQUANT_ROM[relu_byte];
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
