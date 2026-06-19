// n4 - ReLU6 (quantized requantize tail).
// op_type=relu, IC=OC=32, spatial=112x112, bus=256b, pipeline_latency=1.
// scale_factor = 441.6328531901041 -> SCALE_MULT/2^SCALE_SHIFT.

module n4_2 (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          valid_in,
    output reg           ready_in,
    input  wire [255:0]  data_in,
    output reg           valid_out,
    output reg  [255:0]  data_out
);

    localparam integer OC          = 32;
    localparam integer SCALE_SHIFT = 5'd10;
    localparam integer MULT_W      = 26;
    localparam signed [MULT_W-1:0] SCALE_MULT = 32'd13151'sd28942851;
    localparam integer PROD_W      = 8 + MULT_W; // 34
    localparam signed [PROD_W-1:0] SCALE_ROUND_HALF =
        {{(PROD_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [PROD_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(PROD_W-1){1'b0}}, 1'b1};
    localparam signed [PROD_W-1:0] SAT_HI = 34'sd127;
    localparam signed [PROD_W-1:0] SAT_LO = -34'sd128;

    integer i;
    reg signed [7:0]        relu_byte;
    reg signed [PROD_W-1:0] scaled;
    reg signed [PROD_W-1:0] v_tmp;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            ready_in  <= 1'b1; // [INVARIANT:READY_IN_GATING]
            data_out  <= 256'd0;
        end else begin
            valid_out <= valid_in; // [INVARIANT:VALID_OUT_LATENCY]
            ready_in  <= 1'b1;     // [INVARIANT:READY_IN_GATING]
            if (valid_in) begin
                for (i = 0; i < OC; i = i + 1) begin
                    relu_byte = ($signed(data_in[i*8 +: 8]) > 8'sd0)
                                  ? $signed(data_in[i*8 +: 8])
                                  : 8'sd0;
                    scaled    = $signed(relu_byte) * SCALE_MULT;
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
