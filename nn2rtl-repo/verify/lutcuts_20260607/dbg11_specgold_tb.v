`timescale 1ns/1ps
`default_nettype none
// GOLDEN built with CONSTANT offsets only (no variable wide indexing -> no Verilator
// wide-index bug).  Spec: scatter insert beat j puts data_in's low CHUNK bits at
// wide[j*IW +: CHUNK], CHUNK=min(IW, FW-j*IW), rest = wbuf_cur.  S1 params.
module dbg11_specgold_tb;
    reg clk=0; always #5 clk=~clk;
    localparam FW=4608, IW=4096;
    reg [FW-1:0] wbuf_cur;
    reg [IW-1:0] data_in;
    reg [0:0]    g_idx;

    // GOLDEN (constant-offset per-bit, branch on g_idx with literal offsets)
    reg [FW-1:0] gold; integer b;
    always @(*) begin
        gold = wbuf_cur;
        if (g_idx == 1'b0) begin
            // OFF=0, CHUNK=4096
            for (b=0;b<4096;b=b+1) gold[0+b] = data_in[b];
        end else begin
            // OFF=4096, CHUNK = FW-4096 = 512
            for (b=0;b<512;b=b+1) gold[4096+b] = data_in[b];
        end
    end

    // MUX candidate (mirror retile_scatter g_ins_mux arms for S1)
    wire [FW-1:0] arm0 = { wbuf_cur[FW-1:IW], data_in[IW-1:0] };
    wire [FW-1:0] arm1 = { data_in[511:0], wbuf_cur[IW-1:0] };
    wire [FW-1:0] mux  = g_idx ? arm1 : arm0;

    integer i, mism=0;
    reg [63:0] rng = 64'h1234567890abcdef;
    task nr; begin rng=rng^(rng<<13); rng=rng^(rng>>7); rng=rng^(rng<<17); end endtask
    integer k;
    initial begin
      repeat(2) @(negedge clk);
      for (i=0;i<800;i=i+1) begin
        for (k=0;k<FW;k=k+32) begin nr; wbuf_cur[k +: 32] = rng[31:0]; end
        for (k=0;k<IW;k=k+32) begin nr; data_in[k +: 32]  = rng[31:0]; end
        g_idx = i[0];
        @(negedge clk);
        if (mux !== gold) begin
          mism=mism+1;
          if (mism<=3) $display("SPEC MISM i=%0d g_idx=%0d", i, g_idx);
        end
      end
      $display("dbg11 MUX vs SPEC-GOLD mism=%0d / 800", mism);
      if (mism==0) $display("RESULT: MUX MATCHES SPEC PASS");
      else         $display("RESULT: FAIL");
      $finish;
    end
endmodule
`default_nettype wire
