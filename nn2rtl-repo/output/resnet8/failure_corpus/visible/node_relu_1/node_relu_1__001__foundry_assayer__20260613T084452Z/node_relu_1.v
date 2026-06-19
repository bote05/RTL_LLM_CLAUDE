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

    localparam integer OC = 16;

    integer i;

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
                    data_out[i*8 +: 8] <= ($signed(data_in[i*8 +: 8]) > 0)
                                           ? data_in[i*8 +: 8]
                                           : 8'sd0;
                end
            end
        end
    end

endmodule
