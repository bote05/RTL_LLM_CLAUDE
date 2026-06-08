`timescale 1ns/1ps
`default_nettype none
// Module-level: drive S1 (scatter 18/4096/2) FIXED_MUX 0 and 1 with the SAME
// fully-deterministic single pixel, drain all 18 tiles, and for each tile show
// whether the bytes match a SPEC golden computed in the TB (constant offsets).
module dbg12_modcmp_tb;
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

    // SPEC golden buffer (constant-offset construction)
    reg [FW-1:0] gold;
    reg [IW-1:0] beat0, beat1;
    integer b, t;
    integer mism0=0, mism1=0;

    initial begin
      rst_n=0; repeat(4) @(negedge clk); rst_n=1;
      // distinctive beats: beat0 = 0x11*..., beat1 low512 = 0x22*..., beat1 hi = garbage 0xFF (should be DROPPED)
      beat0 = {IW{1'b0}}; for (b=0;b<IW;b=b+8) beat0[b +: 8] = 8'h11;
      beat1 = {IW{1'b0}};
      for (b=0;b<512;b=b+8)   beat1[b +: 8] = 8'h22;   // real hi channels (low 512 bits)
      for (b=512;b<IW;b=b+8)  beat1[b +: 8] = 8'hFF;   // out-of-range garbage -> must be dropped

      // build SPEC gold: beat0 -> [0+:4096], beat1 low512 -> [4096+:512]
      gold = {FW{1'b0}};
      for (b=0;b<4096;b=b+1) gold[b]       = beat0[b];
      for (b=0;b<512; b=b+1) gold[4096+b]  = beat1[b];

      // feed beat0 then beat1
      @(negedge clk); din=beat0; vin=1; rdwn=0; den=0;
      @(negedge clk); din=beat1;
      @(negedge clk); vin=0;
      // drain 18 tiles
      rdwn=1; den=1;
      for (t=0;t<NT;t=t+1) begin
        @(negedge clk);
        // expected tile t = gold[t*256 +: 256]
        if (v0 && d0 !== gold[t*256 +: 256]) begin mism0=mism0+1; $display("U0(var) tile %0d MISMATCH-vs-SPEC d0=%h exp=%h", t, d0, gold[t*256 +: 256]); end
        if (v1 && d1 !== gold[t*256 +: 256]) begin mism1=mism1+1; $display("U1(mux) tile %0d MISMATCH-vs-SPEC d1=%h exp=%h", t, d1, gold[t*256 +: 256]); end
      end
      $display("=== U0(var) vs SPEC mism=%0d ; U1(mux) vs SPEC mism=%0d ===", mism0, mism1);
      $finish;
    end
endmodule
`default_nettype wire
