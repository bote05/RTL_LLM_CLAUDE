`timescale 1ns/1ps
`default_nettype none
// Test whether a per-bit GUARDED loop insert (integer offset) is sim-correct
// for the partial-beat case, matching the constant-offset SPEC golden.
module dbg14_loopvar_tb;
    reg clk=0; always #5 clk=~clk;
    localparam FW=4608, IW=4096;
    reg [FW-1:0] wbuf_cur;
    reg [IW-1:0] data_in;
    reg [0:0]    g_idx;

    // CANDIDATE reference: per-bit guarded loop with INTEGER base offset
    reg [FW-1:0] loopn; integer base, b;
    always @(*) begin
        loopn = wbuf_cur;
        base = g_idx * IW;
        for (b=0;b<IW;b=b+1)
            if (base + b < FW) loopn[base + b] = data_in[b];
    end

    // SPEC golden (constant offsets)
    reg [FW-1:0] gold; integer c;
    always @(*) begin
        gold = wbuf_cur;
        if (g_idx==1'b0) for (c=0;c<4096;c=c+1) gold[c] = data_in[c];
        else             for (c=0;c<512; c=c+1) gold[4096+c] = data_in[c];
    end

    integer i,k,mism=0;
    reg [63:0] rng=64'hbeefbeefbeefbeef;
    task nr; begin rng=rng^(rng<<13); rng=rng^(rng>>7); rng=rng^(rng<<17); end endtask
    initial begin
      repeat(2) @(negedge clk);
      for (i=0;i<600;i=i+1) begin
        for (k=0;k<FW;k=k+32) begin nr; wbuf_cur[k +: 32]=rng[31:0]; end
        for (k=0;k<IW;k=k+32) begin nr; data_in[k +: 32]=rng[31:0]; end
        g_idx=i[0];
        @(negedge clk);
        if (loopn !== gold) begin mism=mism+1; if(mism<=3)$display("LOOP vs SPEC MISM i=%0d g_idx=%0d",i,g_idx); end
      end
      $display("dbg14 LOOP vs SPEC mism=%0d / 600", mism);
      if (mism==0) $display("RESULT: GUARDED-LOOP MATCHES SPEC PASS");
      else         $display("RESULT: FAIL");
      $finish;
    end
endmodule
`default_nettype wire
