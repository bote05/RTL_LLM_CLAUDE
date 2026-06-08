`timescale 1ns/1ps
`default_nettype none
module dbg4_scatter_tb;
    reg clk = 0; reg rst_n = 0;
    always #5 clk = ~clk;
    localparam P_TILE=256, P_NT=18, P_IW=4096, P_IB=2, FW=4608;
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
    reg [FW-1:0] x;
    always @(negedge clk) if (rst_n) begin
      // compare BOTH buffers each cycle to catch first divergence
      if (u0.buf0 !== u1.buf0) begin
        if (mism<2) begin
          x = u0.buf0 ^ u1.buf0;
          $display("BUF0 DIVERGE @%0t  any_hi512=%b any_lo4096=%b  u0lo64=%h u1lo64=%h",
            $time, |x[FW-1:FW-512], |x[FW-513:0], u0.buf0[63:0], u1.buf0[63:0]);
        end
        mism=mism+1;
      end
    end
    initial begin
      rst_n=0; repeat(4) @(negedge clk); rst_n=1;
      for (i=0;i<2000;i=i+1) begin
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
      $display("dbg4 buf0 diverge cycles=%0d", mism);
      $finish;
    end
endmodule
`default_nettype wire
