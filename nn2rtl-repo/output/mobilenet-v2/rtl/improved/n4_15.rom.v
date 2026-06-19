// n4_15: ReLU6 with requantize tail (clip_max=6.0, scale_factor=1.384429136912028).
// Pattern reference: knowledge/patterns/protected/06_relu.md (ReLU6 / clipped activations).
// Upstream conv emits INT8 at a coarser scale; this module applies the ReLU
// non-linearity (negatives -> 0) then requantises from the conv's output_scale
// to ReLU6's tighter output_scale (= 6/128), saturating to INT8.
//
// ROM-COMPRESSED VARIANT (0 DSP):
//   The post-ReLU input domain is strictly 0..127 (negatives clamp to 0, and a
//   signed INT8 maxes at 127). The requant expression
//       v = clamp( ((x*SCALE_MULT) + round) >>> SCALE_SHIFT , -128, 127 )
//   is a compile-time-constant function of x alone. We precompute a 128-entry
//   8-bit ROM (REQ_ROM[x] = exact requant output for input x) in an `initial`
//   loop using the SAME expression, then replace the per-channel multiply/shift
//   datapath with a parallel ROM lookup REQ_ROM[relu_byte]. This eliminates the
//   multiplier (DSP) entirely; the ROM is pure LUT/BRAM. FSM/latency unchanged
//   (1-cycle valid_out latency, ready_in held high).

module n4_15 (
    input  wire                  clk,
    input  wire                  rst_n,        // active-low reset
    input  wire                  valid_in,
    output reg                   ready_in,
    input  wire [3071:0]         data_in,
    output reg                   valid_out,
    output reg  [3071:0]         data_out
);

    localparam integer OC          = 384;
    localparam integer SCALE_MULT  = 22682;     // round(1.384429136912028 * 2^14)
    localparam integer SCALE_SHIFT = 14;
    localparam integer SCALED_W    = 32;

    localparam signed [SCALED_W-1:0] SCALE_MULT_CONST    = SCALE_MULT;
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF    = 32'sd1 <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};

    // ---- Precomputed requant ROM: REQ_ROM[x] = requant(x) for x in 0..127 ----
    // 0 DSP: filled at elaboration with the identical expression. Distributed
    // ROM by default; can be retargeted with rom_style if BRAM packing desired.
    (* rom_style = "distributed" *)
    reg  [7:0]                    REQ_ROM [0:127];

    integer                       k;
    reg signed [7:0]              relu_byte_init;
    reg signed [SCALED_W-1:0]     scaled_init;
    reg signed [SCALED_W-1:0]     v_init;
    initial begin
        for (k = 0; k < 128; k = k + 1) begin
            relu_byte_init = k[7:0];                       // 0..127, always non-negative
            scaled_init    = relu_byte_init * SCALE_MULT_CONST;
            // [INVARIANT:ROUNDING]
            v_init = (scaled_init +
                      (scaled_init[SCALED_W-1] ? SCALE_ROUND_HALF_M1
                                               : SCALE_ROUND_HALF)
                     ) >>> SCALE_SHIFT;
            REQ_ROM[k] = (v_init > 127)  ?  8'sd127 :
                         (v_init < -128) ? -8'sd128 : v_init[7:0];
        end
    end

    integer                       i;
    reg signed [7:0]              relu_byte;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            ready_in  <= 1'b1;                  // [INVARIANT:READY_IN_GATING]
            data_out  <= {3072{1'b0}};
        end else begin
            valid_out <= valid_in;              // [INVARIANT:VALID_OUT_LATENCY]
            ready_in  <= 1'b1;                  // [INVARIANT:READY_IN_GATING]
            if (valid_in) begin
                for (i = 0; i < OC; i = i + 1) begin
                    relu_byte = ($signed(data_in[i*8 +: 8]) > 0)
                                 ? $signed(data_in[i*8 +: 8])
                                 : 8'sd0;
                    // ROM lookup replaces multiply+round+shift+saturate.
                    data_out[i*8 +: 8] <= REQ_ROM[relu_byte[6:0]];
                end
            end
        end
    end
endmodule
