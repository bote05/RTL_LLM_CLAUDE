`timescale 1ns/1ps
`default_nettype none
// Replicate the ORIGINAL shipped structure: wbuf_cur + wbuf_next both reg in ONE always.
module dbg8_orig_tb;
    reg clk=0; always #5 clk=~clk;
    localparam FW=4608, IW=4096;
    reg [FW-1:0] buf0, buf1, wbuf_cur, wbuf_next;
    reg wsel;
    reg [IW-1:0] data_in;
    reg [0:0] g_idx;
    always @(*) begin
        wbuf_cur  = wsel ? buf1 : buf0;
        wbuf_next = wbuf_cur;
        wbuf_next[g_idx*IW +: IW] = data_in;
    end
    initial begin
        repeat(2) @(negedge clk);
        buf0 = {FW{1'b0}}; buf0[63:0] = 64'hAAAAAAAAAAAAAAAA; buf1={FW{1'b0}};
        wsel = 0;
        data_in  = {IW{1'b0}}; data_in[63:0] = 64'h5555555555555555;
        g_idx = 1;
        @(negedge clk);
        $display("ORIG g_idx=1: wbuf_next[63:0]=%h (spec AAAA)  wbuf_next[4607:4544]=%h (spec 5555)",
            wbuf_next[63:0], wbuf_next[4607:4544]);
        g_idx = 0;
        @(negedge clk);
        $display("ORIG g_idx=0: wbuf_next[63:0]=%h (spec 5555)", wbuf_next[63:0]);
        $finish;
    end
endmodule
`default_nettype wire
