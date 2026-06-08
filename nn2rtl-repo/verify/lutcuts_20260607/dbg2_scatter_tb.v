`timescale 1ns/1ps
`default_nettype none
module dbg2_scatter_tb;
    reg clk = 0; reg rst_n = 0;
    always #5 clk = ~clk;

    localparam P_TILE=256, P_NT=18, P_IW=4096, P_IB=2;
    wire r0,r1,v0,v1,st0,st1,wa0,wa1;
    wire [P_TILE-1:0] d0, d1;
    reg vin=0; reg [P_IW-1:0] din=0; reg rdwn=0; reg den=0;

    retile_scatter #(.TILE_W(P_TILE),.N_TILES(P_NT),.IN_W(P_IW),.IN_BEATS(P_IB),.SPATIAL(196),.SYNTH_FIXED_MUX(0)) u0(
      .clk(clk),.rst_n(rst_n),.valid_in(vin),.ready_out(r0),.data_in(din),
      .valid_out(v0),.ready_down(rdwn),.drain_en(den),.data_out(d0),.wr_accept(wa0),.stall_out(st0));
    retile_scatter #(.TILE_W(P_TILE),.N_TILES(P_NT),.IN_W(P_IW),.IN_BEATS(P_IB),.SPATIAL(196),.SYNTH_FIXED_MUX(1)) u1(
      .clk(clk),.rst_n(rst_n),.valid_in(vin),.ready_out(r1),.data_in(din),
      .valid_out(v1),.ready_down(rdwn),.drain_en(den),.data_out(d1),.wr_accept(wa1),.stall_out(st1));

    integer i, mism=0;
    always @(negedge clk) if (rst_n) begin
      if (v1 && d0!==d1) begin
        if (mism<3) $display("MISM @%0t e0buf u0=%h u1=%h  rsel0=%b", $time, d0, d1, u0.rsel);
        mism=mism+1;
      end
    end
    initial begin
      rst_n=0; repeat(4) @(negedge clk); rst_n=1;
      for (i=0;i<20000;i=i+1) begin
        @(negedge clk);
        din = {$random,$random,$random,$random,$random,$random,$random,$random,
               $random,$random,$random,$random,$random,$random,$random,$random,
               $random,$random,$random,$random,$random,$random,$random,$random,
               $random,$random,$random,$random,$random,$random,$random,$random,
               $random,$random,$random,$random,$random,$random,$random,$random,
               $random,$random,$random,$random,$random,$random,$random,$random,
               $random,$random,$random,$random,$random,$random,$random,$random,
               $random,$random,$random,$random,$random,$random,$random,$random,
               $random,$random,$random,$random,$random,$random,$random,$random,
               $random,$random,$random,$random,$random,$random,$random,$random,
               $random,$random,$random,$random,$random,$random,$random,$random,
               $random,$random,$random,$random,$random,$random,$random,$random,
               $random,$random,$random,$random,$random,$random,$random,$random,
               $random,$random,$random,$random,$random,$random,$random,$random,
               $random,$random,$random,$random,$random,$random,$random,$random,
               $random,$random,$random,$random,$random,$random,$random,$random};
        vin=$random; rdwn=$random; den=$random;
      end
      $display("dbg2 mism=%0d", mism);
      $finish;
    end
endmodule
`default_nettype wire
