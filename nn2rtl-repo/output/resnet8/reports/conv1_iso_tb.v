// Isolated node_conv2d_1: compare two feed modes.
//   Mode A: gate valid_in on ready_in (per-module contract). Expect 1024 out.
//   Mode B: drive valid_in high for 1024 consecutive cycles ignoring ready_in
//           (the free-running chain condition). Reveals dropped-beat desync.
`timescale 1ns/1ps
module conv1_iso_tb;
    reg clk=0, rst_n=0; always #5 clk=~clk;
    reg  [127:0] data_in; reg valid_in;
    wire ready_in, valid_out; wire [127:0] data_out;
    integer outc, sent, i, mode;
    node_conv2d_1 dut(.clk(clk),.rst_n(rst_n),.valid_in(valid_in),
        .ready_in(ready_in),.data_in(data_in),.valid_out(valid_out),.data_out(data_out));
    always @(posedge clk) if(rst_n && valid_out) outc=outc+1;

    task run_mode(input integer m);
    begin
        outc=0; sent=0; valid_in=0; data_in=0;
        rst_n=0; repeat(4) @(posedge clk); rst_n=1; @(posedge clk);
        for(i=0;i<4000;i=i+1) begin
            @(negedge clk);
            if(m==0) begin // Mode A: respect ready_in
                valid_in = (sent<1024);
                if(valid_in && ready_in) sent=sent+1;
            end else begin // Mode B: 1024 consecutive valid cycles
                valid_in = (sent<1024);
                if(valid_in) sent=sent+1;   // advance regardless of ready_in
            end
            data_in = sent;
        end
        valid_in=0; repeat(300) @(posedge clk);
        $display("[iso mode%0d] sent=%0d valid_out_count=%0d", m, sent, outc);
    end
    endtask

    initial begin
        run_mode(0);
        run_mode(1);
        $finish;
    end
endmodule
