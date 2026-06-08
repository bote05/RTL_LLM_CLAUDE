`timescale 1ns/1ps
`default_nettype none
module dbg6_var_tb;
    reg clk=0; always #5 clk=~clk;
    localparam FW=4608, IW=4096;
    reg [FW-1:0] wbuf_cur;
    reg [IW-1:0] data_in;
    reg [0:0]    g_idx;
    reg [FW-1:0] var_next;
    integer prod;
    always @(*) begin
        var_next = wbuf_cur;
        var_next[g_idx*IW +: IW] = data_in;
    end
    initial begin
        wbuf_cur = {FW{1'b0}};
        wbuf_cur[3:0] = 4'hA;       // low marker
        wbuf_cur[4095:4092] = 4'h5; // near top-of-low marker
        data_in = {IW{1'b1}};       // all ones
        g_idx = 1;
        prod = g_idx*IW;
        @(negedge clk);
        $display("g_idx*IW = %0d (decl FW=%0d)", prod, FW);
        $display("var_next[3:0]      = %h (expect A: low unchanged)", var_next[3:0]);
        $display("var_next[4095:4092]= %h (expect 5: low unchanged)", var_next[4095:4092]);
        $display("var_next[4607:4096]= %h (expect all-1: data_in[511:0])", var_next[4607:4096]);
        $finish;
    end
endmodule
`default_nettype wire
