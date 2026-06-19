`timescale 1ns/1ps
module add_equiv_tb;
    reg clk=0, rst_n=0; always #5 clk=~clk;
    reg  [1023:0] data_in;
    reg  ser_valid_in, par_valid_in;
    wire ser_ready_in, par_ready_in, ser_valid_out, par_valid_out;
    wire [511:0] ser_data_out, par_data_out;
    node_add_87_old u_ser(.clk(clk),.rst_n(rst_n),.valid_in(ser_valid_in),.ready_in(ser_ready_in),.data_in(data_in),.valid_out(ser_valid_out),.data_out(ser_data_out));
    node_add_87     u_par(.clk(clk),.rst_n(rst_n),.valid_in(par_valid_in),.ready_in(par_ready_in),.data_in(data_in),.valid_out(par_valid_out),.data_out(par_data_out));
    integer n,mism,sc,pc,k,idx,seed;
    reg [511:0] sq[0:4095]; reg [511:0] pq[0:4095]; reg [1023:0] stim[0:1023];
    always @(posedge clk) if(rst_n&&ser_valid_out) begin sq[sc]=ser_data_out; sc=sc+1; end
    always @(posedge clk) if(rst_n&&par_valid_out) begin pq[pc]=par_data_out; pc=pc+1; end
    initial begin
        seed=987; mism=0; sc=0; pc=0; ser_valid_in=0; par_valid_in=0; data_in=0;
        for(n=0;n<1024;n=n+1) for(k=0;k<32;k=k+1) stim[n][k*32 +:32]=$random(seed);
        stim[0]=0; stim[1]={1024{1'b0}}; stim[1][7:0]=8'h7f; stim[2]={1024{1'b1}};
        repeat(4) @(posedge clk); rst_n=1; @(posedge clk);
        idx=0;
        while(idx<1024) begin @(negedge clk);
          if(ser_ready_in) begin data_in=stim[idx]; ser_valid_in=1; @(negedge clk); ser_valid_in=0; idx=idx+1; while(!ser_ready_in) @(negedge clk); end
          else @(negedge clk);
        end
        repeat(80) @(posedge clk);
        for(idx=0;idx<1024;idx=idx+1) begin @(negedge clk); data_in=stim[idx]; par_valid_in=1; end
        @(negedge clk); par_valid_in=0; repeat(20) @(posedge clk);
        $display("[equiv] sc=%0d pc=%0d",sc,pc);
        if(sc!=1024||pc!=1024) begin $display("[equiv][summary] result=FAIL count sc=%0d pc=%0d",sc,pc); $finish; end
        for(n=0;n<1024;n=n+1) if(sq[n]!==pq[n]) begin if(mism<8) $display("[equiv] MM beat %0d",n); mism=mism+1; end
        if(mism==0) $display("[equiv][summary] result=PASS mismatches=0 beats=1024");
        else $display("[equiv][summary] result=FAIL mismatches=%0d",mism);
        $finish;
    end
endmodule
