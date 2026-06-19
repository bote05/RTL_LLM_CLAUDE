// node_relu -- INT8 element-wise ReLU.
// 16 channels packed into 128-bit bus, no clip_max -> passthrough max(0,x).
// pipeline_latency_cycles = 1.

module node_relu (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    output reg          ready_in,
    input  wire [127:0] data_in,
    output reg          valid_out,
    output reg  [127:0] data_out
);

    localparam integer OC = 16;

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            ready_in  <= 1'b1; // [INVARIANT:READY_IN_GATING]
            data_out  <= 128'd0;
        end else begin
            ready_in  <= 1'b1; // [INVARIANT:READY_IN_GATING]
            valid_out <= valid_in; // [INVARIANT:VALID_OUT_LATENCY]
            if (valid_in) begin
                for (i = 0; i < OC; i = i + 1) begin
                    // [INVARIANT:ROUNDING] no scale -- passthrough on positive, 0 on negative
                    data_out[i*8 +: 8] <= ($signed(data_in[i*8 +: 8]) > 0)
                                           ? data_in[i*8 +: 8]
                                           : 8'sd0;
                end
            end
        end
    end

endmodule
