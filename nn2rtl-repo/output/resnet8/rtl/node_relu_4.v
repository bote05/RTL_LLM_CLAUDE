// node_relu_4 - INT8 elementwise ReLU + requantize, flat-bus, 32 channels, 256-bit packed.
// pipeline_latency_cycles = 1.
// Requant ratio input_scale/output_scale = 1.394043 -> compute_scale_approx =
// (MULT=2855, SHIFT=11), the EXACT golden contract: reproduces node_relu_4.goldout
// byte-exact on all 8 vectors. The prior coarse fit (1428/1024 = 1.3945) diverged
// from the golden by +-1 on 16/65536 bytes. ROUND = 1<<(SHIFT-1) = 1024.
// Applies output = clamp((max(0,x) * MULT + ROUND) >>> SHIFT, 0, 127).

module node_relu_4 (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                valid_in,
    output reg                 ready_in,
    input  wire [255:0]        data_in,
    output reg                 valid_out,
    output reg  [255:0]        data_out
);

    localparam integer OC       = 32;
    localparam integer RS_MULT  = 2855;
    localparam integer RS_SHIFT = 11;
    localparam integer RS_ROUND = 1024;

    integer i;
    reg signed [7:0]  tmp_byte;
    reg signed [31:0] rs_in;
    reg signed [31:0] rs_out;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;        // [INVARIANT:VALID_OUT_LATENCY]
            ready_in  <= 1'b1;        // [INVARIANT:READY_IN_GATING]
            data_out  <= 256'b0;
        end else begin
            ready_in  <= 1'b1;        // [INVARIANT:READY_IN_GATING]
            valid_out <= valid_in;    // [INVARIANT:VALID_OUT_LATENCY]
            if (valid_in) begin
                for (i = 0; i < OC; i = i + 1) begin
                    tmp_byte = $signed(data_in[i*8 +: 8]);
                    rs_in    = (tmp_byte > 8'sd0) ? $signed({{24{1'b0}}, tmp_byte}) : 32'sd0;
                    rs_out   = (rs_in * RS_MULT + RS_ROUND) >>> RS_SHIFT;
                    data_out[i*8 +: 8] <= (rs_out > 32'sd127) ? 8'sd127 : rs_out[7:0];
                end
            end
        end
    end

endmodule
