`timescale 1ns/1ps
`default_nettype none
module dbg9_width_tb;
    reg clk=0; always #5 clk=~clk;
    // Test variable indexed part-select write at several widths.
    localparam FW=4608, IW=4096;
    reg [FW-1:0] buf_a, next_a;
    reg [IW-1:0] din_a;
    reg [31:0]   gi;            // WIDE g_idx
    always @(*) begin
        next_a = buf_a;
        next_a[gi*IW +: IW] = din_a;
    end
    // narrow control: gi as exact-fit index expression
    reg [FW-1:0] next_b;
    integer off;
    always @(*) begin
        next_b = buf_a;
        next_b[off +: IW] = din_a;
    end
    initial begin
        repeat(2) @(negedge clk);
        buf_a = {FW{1'b0}}; buf_a[63:0]=64'hAAAAAAAAAAAAAAAA;
        din_a = {IW{1'b0}}; din_a[63:0]=64'h5555555555555555;
        gi = 0; off = 0;
        @(negedge clk);
        $display("gi=0 widevar: next_a[63:0]=%h (spec 5555)", next_a[63:0]);
        $display("off=0 intvar: next_b[63:0]=%h (spec 5555)", next_b[63:0]);
        gi = 1; off = 4096;
        @(negedge clk);
        $display("gi=1 widevar: next_a[4607:4544]=%h (spec 5555) lo[63:0]=%h (spec AAAA)", next_a[4607:4544], next_a[63:0]);
        $display("off=4096 intvar: next_b[4607:4544]=%h (spec 5555) lo[63:0]=%h (spec AAAA)", next_b[4607:4544], next_b[63:0]);
        $finish;
    end
endmodule
`default_nettype wire
