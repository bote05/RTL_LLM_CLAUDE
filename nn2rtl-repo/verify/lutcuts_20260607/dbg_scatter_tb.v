`timescale 1ns/1ps
`default_nettype none
module dbg_scatter_tb;
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

    integer k; integer mism=0;
    initial begin
      rst_n=0; repeat(4) @(negedge clk); rst_n=1;
      // write 2 input beats (full pixel), known patterns
      @(negedge clk);
      // beat0
      din = {3584'b0, 512'h0, 4096'h0}; // placeholder, set below per-bit
      din = 4096'h0; din[0]=1; din[100]=1; din[4095]=1; // low beat markers
      vin=1; rdwn=0; den=0;
      @(negedge clk);          // g_idx 0 written (do_write since wsel buffer empty)
      din = 4096'h0; din[0]=1; din[1]=1; din[511]=1; din[512]=1; din[4095]=1; // hi beat markers
      @(negedge clk);          // g_idx 1 written -> buffer full
      vin=0;
      // now drain all 18 tiles
      rdwn=1; den=1;
      for (k=0;k<20;k=k+1) begin
        @(negedge clk);
        if (v1 && d0!==d1) begin mism=mism+1; $display("DRAIN MISMATCH tile-cyc %0d d0=%h d1=%h", k, d0, d1); end
        else if (v1) $display("ok tile-cyc %0d d=%h", k, d0);
      end
      $display("dbg mism=%0d", mism);
      $finish;
    end
endmodule
`default_nettype wire
