// n4_5 - ReLU6 (quantized requantize tail).
// op_type=relu, IC=OC=144, spatial=56x56, bus=1152b, pipeline_latency=1.
// scale_factor = 5.327365557352701 -> SCALE_MULT=32'd23483, SCALE_SHIFT=5'd12.

module n4_6 (
    input  wire           clk,
    input  wire           rst_n,
    input  wire           valid_in,
    output reg            ready_in,
    input  wire [1151:0]  data_in,
    output reg            valid_out,
    output reg  [1151:0]  data_out
);

    localparam integer OC          = 144;
    localparam integer SCALE_SHIFT = 5'd12;
    localparam integer MULT_W      = 16;
    localparam signed [MULT_W-1:0]  SCALE_MULT_CONST = 16'sd21821;
    localparam integer PROD_W      = 8 + MULT_W; // 24
    localparam signed [PROD_W-1:0]  SCALE_ROUND_HALF =
        {{(PROD_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [PROD_W-1:0]  SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(PROD_W-1){1'b0}}, 1'b1};
    localparam signed [PROD_W-1:0]  SAT_HI =  24'sd127;
    localparam signed [PROD_W-1:0]  SAT_LO = -24'sd128;

    integer i;
    reg signed [7:0]        relu_byte;
    reg signed [PROD_W-1:0] scaled;
    reg signed [PROD_W-1:0] v_tmp;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            ready_in  <= 1'b1; // [INVARIANT:READY_IN_GATING]
            data_out  <= 1152'd0;
        end else begin
            valid_out <= valid_in; // [INVARIANT:VALID_OUT_LATENCY]
            ready_in  <= 1'b1;     // [INVARIANT:READY_IN_GATING]
            if (valid_in) begin
                for (i = 0; i < OC; i = i + 1) begin
                    relu_byte = ($signed(data_in[i*8 +: 8]) > 8'sd0)
                                  ? $signed(data_in[i*8 +: 8])
                                  : 8'sd0;
                    scaled    = $signed(relu_byte) * SCALE_MULT_CONST;
                    // [INVARIANT:ROUNDING]
                    v_tmp     = (scaled +
                                  (scaled[PROD_W-1] ? SCALE_ROUND_HALF_M1
                                                    : SCALE_ROUND_HALF)) >>> SCALE_SHIFT;
                    data_out[i*8 +: 8] <= (v_tmp > SAT_HI) ?  8'sd127 :
                                          (v_tmp < SAT_LO) ? -8'sd128 :
                                                              v_tmp[7:0];
                end
            end
        end
    end

endmodule
