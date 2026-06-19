// node_relu_5 — quantized ReLU with activation rescale (flat-bus, 64 INT8 ch)
// Input scale  (from node_conv2d_7 output) = node_relu_4.scale_factor / node_conv2d_7.scale_factor
//                                          = 0.09856558972456324 / 1.2727433159114343
//                                          = 0.07744431...
// Output scale                              = 0.05712437817430872
// Requantize ratio                          = 0.07744431 / 0.05712437817 = 1.35569
// SCALE_MULT=11105, SCALE_SHIFT=13 -> 11105/8192 = 1.355591 (err ~1.6e-5)
// pipeline_latency_cycles = 1
`timescale 1ns / 1ps

module node_relu_5 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    output reg          ready_in,
    input  wire [511:0] data_in,
    output reg          valid_out,
    output reg  [511:0] data_out
);

    localparam integer       OC            = 64;
    localparam integer       SCALE_SHIFT   = 13;
    localparam integer       SCALE_CONST_W = 15;
    localparam signed [SCALE_CONST_W-1:0] SCALE_MULT_CONST = 15'sd11105;
    localparam integer       SCALED_W      = 8 + SCALE_CONST_W; // 23

    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};

    reg signed [7:0]          in_byte;
    reg signed [7:0]          relu_byte;
    reg signed [SCALED_W-1:0] scaled_val;
    reg signed [SCALED_W-1:0] v_tmp;
    integer                   i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            ready_in  <= 1'b1;          // [INVARIANT:READY_IN_GATING]
            data_out  <= 512'd0;
        end else begin
            valid_out <= valid_in;      // [INVARIANT:VALID_OUT_LATENCY]
            ready_in  <= 1'b1;          // [INVARIANT:READY_IN_GATING]
            if (valid_in) begin
                for (i = 0; i < OC; i = i + 1) begin
                    in_byte    = $signed(data_in[i*8 +: 8]);
                    relu_byte  = (in_byte > 8'sd0) ? in_byte : 8'sd0;
                    scaled_val = $signed(relu_byte) * SCALE_MULT_CONST;
                    // [INVARIANT:ROUNDING]
                    v_tmp      = (scaled_val +
                                  (scaled_val[SCALED_W-1] ? SCALE_ROUND_HALF_M1
                                                          : SCALE_ROUND_HALF)
                                 ) >>> SCALE_SHIFT;
                    data_out[i*8 +: 8] <= (v_tmp > 23'sd127)  ?  8'sd127 :
                                          (v_tmp < -23'sd128) ? -8'sd128 : v_tmp[7:0];
                end
            end
        end
    end

endmodule
