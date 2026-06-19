`timescale 1ns/1ps
module conv2_iso_tb;
    reg clk=0,rst_n=0; always #5 clk=~clk;
    reg [127:0] din; reg vin; wire rdy,vout; wire [127:0] dout;
    integer outc,sent,i;
    node_conv2d_2 dut(.clk(clk),.rst_n(rst_n),.valid_in(vin),.ready_in(rdy),.data_in(din),.valid_out(vout),.data_out(dout));
    always @(posedge clk) if(rst_n&&vout) outc=outc+1;
    initial begin
        outc=0; sent=0; vin=0; din=0;
        repeat(4)@(posedge clk); rst_n=1; @(posedge clk);
        // contiguous 1024 valid beats, 1/cycle (clean stream as from a FIFO drained 1/cyc)
        for(i=0;i<1024;i=i+1) begin @(negedge clk); vin=1; din=i; sent=sent+1; end
        @(negedge clk); vin=0;
        repeat(1500)@(posedge clk);
        $display("[conv2-iso clean1024] sent=%0d valid_out=%0d", sent, outc);
        $finish;
    end
endmodule
