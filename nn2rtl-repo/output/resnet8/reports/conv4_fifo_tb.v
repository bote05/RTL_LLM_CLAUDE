`timescale 1ns/1ps
module conv4_fifo_tb;
    reg clk=0,rst_n=0; always #5 clk=~clk;
    reg [127:0] pdin; reg pvin; wire fir;
    wire fov; wire [127:0] fod; wire crdy,cvout; wire [255:0] cdout;
    integer outc,pushed,i;
    skip_fifo #(.WIDTH(128),.DEPTH(1024)) u_f(.clk(clk),.rst_n(rst_n),.in_valid(pvin),.in_data(pdin),.in_ready(fir),.out_valid(fov),.out_data(fod),.out_ready(crdy));
    node_conv2d_4 dut(.clk(clk),.rst_n(rst_n),.valid_in(fov),.ready_in(crdy),.data_in(fod),.valid_out(cvout),.data_out(cdout));
    always @(posedge clk) if(rst_n&&cvout) outc=outc+1;
    initial begin
        outc=0; pushed=0; pvin=0; pdin=0; i=0;
        repeat(4)@(posedge clk); rst_n=1; @(posedge clk);
        // gappy producer (every 2nd cycle), 1024 beats
        while(pushed<1024) begin @(negedge clk);
            if(i%2==0) pvin=1; else pvin=0;
            pdin=pushed; if(pvin&&fir) pushed=pushed+1; i=i+1;
        end
        @(negedge clk); pvin=0; repeat(2000)@(posedge clk);
        $display("[conv4-fifo] pushed=%0d valid_out=%0d (expect 256)",pushed,outc);
        $finish;
    end
endmodule
