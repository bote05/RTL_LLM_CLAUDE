`timescale 1ns/1ps
`default_nettype none
// Find the EXACT first cycle + bit-range where u0.buf0/buf1 diverge from u1, with
// full context (do_write, g_idx, wsel, data_in low/mid/hi). Random, real modules.
module dbg13_bufdiff_tb;
    reg clk=0; always #5 clk=~clk;
    reg rst_n=0;
    localparam FW=4608, IW=4096, TW=256, NT=18;
    wire r0,r1,v0,v1,st0,st1,wa0,wa1;
    wire [TW-1:0] d0,d1;
    reg vin=0; reg [IW-1:0] din=0; reg rdwn=0; reg den=0;

    retile_scatter #(.TILE_W(TW),.N_TILES(NT),.IN_W(IW),.IN_BEATS(2),.SPATIAL(196),.SYNTH_FIXED_MUX(0)) u0(
      .clk(clk),.rst_n(rst_n),.valid_in(vin),.ready_out(r0),.data_in(din),
      .valid_out(v0),.ready_down(rdwn),.drain_en(den),.data_out(d0),.wr_accept(wa0),.stall_out(st0));
    retile_scatter #(.TILE_W(TW),.N_TILES(NT),.IN_W(IW),.IN_BEATS(2),.SPATIAL(196),.SYNTH_FIXED_MUX(1)) u1(
      .clk(clk),.rst_n(rst_n),.valid_in(vin),.ready_out(r1),.data_in(din),
      .valid_out(v1),.ready_down(rdwn),.drain_en(den),.data_out(d1),.wr_accept(wa1),.stall_out(st1));

    reg [63:0] rng=64'hace1ace1ace1ace1;
    task nr; begin rng=rng^(rng<<13); rng=rng^(rng>>7); rng=rng^(rng<<17); end endtask
    integer i,k; reg done; reg [FW-1:0] x;
    initial begin
      done=0;
      rst_n=0; repeat(4) @(negedge clk); rst_n=1;
      for (i=0;i<3000 && !done;i=i+1) begin
        @(negedge clk);
        // sample BEFORE next edge: compare buffers
        if (u0.buf0 !== u1.buf0 && !done) begin
          x=u0.buf0^u1.buf0;
          $display("buf0 DIVERGE @cyc%0d hi512=%b mid[4095:512]=%b lo512=%b  prev g_idx0=%0d wsel0=%b dowrite0=%b",
             i, |x[4607:4096], |x[4095:512], |x[511:0], u0.g_idx, u0.wsel, u0.do_write);
          done=1;
        end
        if (u0.buf1 !== u1.buf1 && !done) begin
          x=u0.buf1^u1.buf1;
          $display("buf1 DIVERGE @cyc%0d hi512=%b mid[4095:512]=%b lo512=%b  prev g_idx0=%0d wsel0=%b dowrite0=%b",
             i, |x[4607:4096], |x[4095:512], |x[511:0], u0.g_idx, u0.wsel, u0.do_write);
          done=1;
        end
        // drive next
        for (k=0;k<IW;k=k+32) begin nr; din[k +: 32]=rng[31:0]; end
        nr; vin=rng[0]; nr; rdwn=rng[0]; nr; den=rng[0];
      end
      if (!done) $display("no divergence in 3000 cyc");
      $finish;
    end
endmodule
`default_nettype wire
