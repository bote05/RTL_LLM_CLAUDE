`timescale 1ns/1ps
module conv2_fg_tb;
    reg clk=0,rst_n=0; always #5 clk=~clk;
    reg [127:0] pdin; reg pvin; wire fir;
    wire fov; wire [127:0] fod; wire crdy,cvout; wire [127:0] cdout;
    integer outc,pushed,i;
    frame_gate_fifo #(.WIDTH(128),.DEPTH(2048),.FRAME(1024)) u_fg(
        .clk(clk),.rst_n(rst_n),.in_valid(pvin),.in_data(pdin),.in_ready(fir),
        .out_valid(fov),.out_data(fod),.out_ready(crdy));
    node_conv2d_2 dut(.clk(clk),.rst_n(rst_n),.valid_in(fov),.ready_in(crdy),.data_in(fod),.valid_out(cvout),.data_out(cdout));
    always @(posedge clk) if(rst_n&&cvout) outc=outc+1;
    initial begin
        outc=0; pushed=0; pvin=0; pdin=0; i=0;
        repeat(4)@(posedge clk); rst_n=1; @(posedge clk);
        // gappy producer: push every 3rd cycle until 1024 pushed
        while(pushed<1024) begin
            @(negedge clk);
            if(i%3==0) begin pvin=1; pdin=pushed; end else pvin=0;
            if(pvin && fir) pushed=pushed+1;
            i=i+1;
        end
        @(negedge clk); pvin=0;
        repeat(2500)@(posedge clk);
        $display("[conv2-framegate] pushed=%0d valid_out=%0d",pushed,outc);
        $finish;
    end
endmodule
