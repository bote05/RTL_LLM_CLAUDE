// node_relu_2 — quantized ReLU (flat-bus, 16 INT8 channels)
// scale_factor = 0.09533731205256905 -> SCALE_MULT=24993, SCALE_SHIFT=18
// pipeline_latency_cycles = 1

module node_relu_2 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    output reg          ready_in,
    input  wire [127:0] data_in,
    output reg          valid_out,
    output reg  [127:0] data_out
);

    localparam integer       OC          = 16;
    localparam integer       SCALE_SHIFT = 18;
    localparam signed [15:0] SCALE_MULT  = 16'sd24993;
    localparam integer       SCALED_W    = 26;

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
            data_out  <= 128'd0;
        end else begin
            valid_out <= valid_in;      // [INVARIANT:VALID_OUT_LATENCY]
            ready_in  <= 1'b1;          // [INVARIANT:READY_IN_GATING]
            if (valid_in) begin
                for (i = 0; i < OC; i = i + 1) begin
                    in_byte    = $signed(data_in[i*8 +: 8]);
                    relu_byte  = (in_byte > 8'sd0) ? in_byte : 8'sd0;
                    scaled_val = $signed(relu_byte) * $signed(SCALE_MULT);
                    v_tmp      = (scaled_val +
                                  (scaled_val[SCALED_W-1] ? SCALE_ROUND_HALF_M1
                                                          : SCALE_ROUND_HALF)
                                 ) >>> SCALE_SHIFT;   // [INVARIANT:ROUNDING]
                    data_out[i*8 +: 8] <= (v_tmp > 26'sd127) ? 8'sd127 : v_tmp[7:0];
                end
            end
        end
    end

endmodule
