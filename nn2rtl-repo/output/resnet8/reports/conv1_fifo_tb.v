// Elastic-buffer test: free-running producer (1 beat/cycle) -> skip_fifo ->
// node_conv2d_1 draining on the conv's REAL ready_in. Expect 1024 valid_out.
`timescale 1ns/1ps
module conv1_fifo_tb;
    reg clk=0, rst_n=0; always #5 clk=~clk;
    // producer
    reg  [127:0] prod_data; reg prod_valid; integer pushed;
    wire fifo_in_ready;
    // fifo -> conv
    wire fifo_out_valid; wire [127:0] fifo_out_data;
    wire conv_ready_in, conv_valid_out; wire [127:0] conv_data_out;
    integer outc, i;

    skip_fifo #(.WIDTH(128), .DEPTH(2048)) u_fifo (
        .clk(clk), .rst_n(rst_n),
        .in_valid(prod_valid), .in_data(prod_data), .in_ready(fifo_in_ready),
        .out_valid(fifo_out_valid), .out_data(fifo_out_data), .out_ready(conv_ready_in));

    node_conv2d_1 dut (.clk(clk), .rst_n(rst_n),
        .valid_in(fifo_out_valid), .ready_in(conv_ready_in), .data_in(fifo_out_data),
        .valid_out(conv_valid_out), .data_out(conv_data_out));

    always @(posedge clk) if(rst_n && conv_valid_out) outc=outc+1;

    initial begin
        outc=0; pushed=0; prod_valid=0; prod_data=0;
        repeat(4) @(posedge clk); rst_n=1; @(posedge clk);
        // free-running producer: push 1024 beats one/cycle (gated only by fifo not full)
        for(i=0;i<3000;i=i+1) begin
            @(negedge clk);
            prod_valid = (pushed<1024);
            prod_data  = pushed;
            if(prod_valid && fifo_in_ready) pushed=pushed+1;
        end
        prod_valid=0; repeat(400) @(posedge clk);
        $display("[fifo-iso] pushed=%0d valid_out_count=%0d", pushed, outc);
        $finish;
    end
endmodule
