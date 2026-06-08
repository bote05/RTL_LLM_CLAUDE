`timescale 1ns/1ps
`default_nettype none
module dbg7_var_tb;
    reg clk=0; always #5 clk=~clk;
    localparam FW=4608, IW=4096;
    reg [FW-1:0] wbuf_cur, var_next;
    reg [IW-1:0] data_in;
    reg [0:0] g_idx;
    always @(*) begin
        var_next = wbuf_cur;
        var_next[g_idx*IW +: IW] = data_in;
    end
    integer i;
    initial begin
        repeat(2) @(negedge clk);
        // marker pattern: wbuf_cur low = AAAA..., data_in = 5555...
        wbuf_cur = {FW{1'b0}};
        wbuf_cur[63:0] = 64'hAAAAAAAAAAAAAAAA;
        data_in  = {IW{1'b0}};
        data_in[63:0] = 64'h5555555555555555;
        g_idx = 1;
        @(negedge clk);
        $display("g_idx=1 (offset=4096, buf width=4608):");
        $display("  var_next[63:0]      = %h  (spec: AAAA.. low untouched)", var_next[63:0]);
        $display("  var_next[4607:4544] = %h  (spec: 5555.. = data_in[63:0] at offset 4096)", var_next[4607:4544]);
        g_idx = 0;
        @(negedge clk);
        $display("g_idx=0 (offset=0):");
        $display("  var_next[63:0]      = %h  (spec: 5555.. = data_in[63:0])", var_next[63:0]);
        $finish;
    end
endmodule
`default_nettype wire
