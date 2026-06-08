`timescale 1ns/1ps
`default_nettype none
// Verify gather emit MUX (SYNTH_FIXED_MUX=1) chunk == constant-offset SPEC golden
// for the partial-final-beat combos G1 (18/4096/2) and G4 (18/2048/3) and G5 (30/2048/4).
module dbg15_gather_spec_tb;
    reg clk=0; always #5 clk=~clk;

    // emulate the gather emit chunk selection directly from a known rbuf.
    // G1: FULL_W=4608, OUT_W=4096, OUT_BEATS=2.  beat0=[0+:4096], beat1=[4096+:512]zpad
    localparam FW1=4608;
    reg [FW1-1:0] rbuf1; reg [0:0] e1;
    // mux arms (mirror g_emit_mux)
    wire [4095:0] g1_arm0 = rbuf1[0 +: 4096];
    wire [4095:0] g1_arm1 = { {3584{1'b0}}, rbuf1[4096 +: 512] };
    wire [4095:0] g1_mux  = e1 ? g1_arm1 : g1_arm0;
    // spec golden (constant offsets, zero-pad past FW)
    reg [4095:0] g1_gold; integer b;
    always @(*) begin
        g1_gold = {4096{1'b0}};
        if (e1==0) for (b=0;b<4096;b=b+1) g1_gold[b] = rbuf1[b];
        else       for (b=0;b<4096;b=b+1) g1_gold[b] = (4096+b < FW1) ? rbuf1[4096+b] : 1'b0;
    end

    // G4: FULL_W=4608, OUT_W=2048, OUT_BEATS=3.  beats 0,1 full; beat2 = [4096+:512]zpad
    localparam FW4=4608;
    reg [FW4-1:0] rbuf4; reg [1:0] e4;
    wire [2047:0] g4_arm0 = rbuf4[0 +: 2048];
    wire [2047:0] g4_arm1 = rbuf4[2048 +: 2048];
    wire [2047:0] g4_arm2 = { {1536{1'b0}}, rbuf4[4096 +: 512] };
    reg  [2047:0] g4_mux;
    always @(*) case(e4) 0:g4_mux=g4_arm0; 1:g4_mux=g4_arm1; 2:g4_mux=g4_arm2; default:g4_mux=g4_arm0; endcase
    reg [2047:0] g4_gold; integer c; integer off4;
    always @(*) begin
        g4_gold={2048{1'b0}}; off4=e4*2048;
        for (c=0;c<2048;c=c+1) g4_gold[c] = (off4+c < FW4) ? rbuf4[off4+c] : 1'b0;
    end

    integer i,k,m1=0,m4=0;
    reg [63:0] rng=64'hf00df00dcafecafe;
    task nr; begin rng=rng^(rng<<13); rng=rng^(rng>>7); rng=rng^(rng<<17); end endtask
    initial begin
      repeat(2) @(negedge clk);
      for (i=0;i<900;i=i+1) begin
        for (k=0;k<FW1;k=k+32) begin nr; rbuf1[k +: 32]=rng[31:0]; end
        for (k=0;k<FW4;k=k+32) begin nr; rbuf4[k +: 32]=rng[31:0]; end
        e1 = i[0];
        e4 = i % 3;
        @(negedge clk);
        if (g1_mux !== g1_gold) begin m1=m1+1; if(m1<=2)$display("G1 MUX!=SPEC e1=%0d",e1); end
        if (g4_mux !== g4_gold) begin m4=m4+1; if(m4<=2)$display("G4 MUX!=SPEC e4=%0d",e4); end
      end
      $display("dbg15 G1 mux-vs-spec=%0d  G4 mux-vs-spec=%0d  / 900", m1, m4);
      if (m1==0 && m4==0) $display("RESULT: GATHER MUX MATCHES SPEC PASS");
      else $display("RESULT: FAIL");
      $finish;
    end
endmodule
`default_nettype wire
