`timescale 1ns / 1ps

module node_relu_1 (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          valid_in,
    output reg           ready_in,
    input  wire [127:0]  data_in,
    output reg           valid_out,
    output reg  [127:0]  data_out
);

    localparam integer OC            = 16;
    localparam integer SCALE_SHIFT   = 2;
    localparam integer SCALE_CONST_W = 4;
    localparam signed [SCALE_CONST_W-1:0] SCALE_MULT_CONST = 4'sd5;
    localparam integer SCALED_W = 8 + SCALE_CONST_W;

    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};

    integer i;
    reg signed [7:0]          in_byte;
    reg signed [7:0]          relu_out;
    reg signed [SCALED_W-1:0] scaled;
    reg signed [SCALED_W-1:0] v_tmp;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            ready_in  <= 1'b1; // [INVARIANT:READY_IN_GATING]
            data_out  <= 128'd0;
        end else begin
            valid_out <= valid_in; // [INVARIANT:VALID_OUT_LATENCY]
            ready_in  <= 1'b1;     // [INVARIANT:READY_IN_GATING]
            if (valid_in) begin
                for (i = 0; i < OC; i = i + 1) begin
                    in_byte  = $signed(data_in[i*8 +: 8]);
                    relu_out = (in_byte > 0) ? in_byte : 8'sd0;
                    scaled   = $signed(relu_out) * SCALE_MULT_CONST;
                    // [INVARIANT:ROUNDING]
                    v_tmp = (scaled +
                             (scaled[SCALED_W-1] ? SCALE_ROUND_HALF_M1 : SCALE_ROUND_HALF)
                            ) >>> SCALE_SHIFT;
                    data_out[i*8 +: 8] <= (v_tmp > 127)  ?  8'sd127 :
                                          (v_tmp < -128) ? -8'sd128 : v_tmp[7:0];
                end
            end
        end
    end

endmodule
