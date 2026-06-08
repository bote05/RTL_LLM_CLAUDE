`timescale 1ns/1ps
`default_nettype none
// Definitive: compare per-bit-loop GOLDEN (the documented spec) vs the variable
// wide part-select vs the fixed mux, for the scatter INSERT, FULL_W=4608/IW=4096.
module dbg10_golden_tb;
    reg clk=0; always #5 clk=~clk;
    localparam FW=4608, IW=4096;
    reg [FW-1:0] wbuf_cur;
    reg [IW-1:0] data_in;
    reg [0:0]    g_idx;

    // GOLDEN: per-bit loop = exactly the spec ("insert IN_W bits at offset g_idx*IN_W, in-range only")
    reg [FW-1:0] gold; integer b;
    always @(*) begin
        gold = wbuf_cur;
        for (b=0;b<IW;b=b+1)
            if (g_idx*IW + b < FW) gold[g_idx*IW + b] = data_in[b];
    end

    // VARIABLE (the shipped path)
    reg [FW-1:0] varn;
    always @(*) begin
        varn = wbuf_cur;
        varn[g_idx*IW +: IW] = data_in;
    end

    // MUX (my fixed path)
    // beat0: OFF=0 CHUNK=4096 -> {wbuf_cur[4607:4096], data_in[4095:0]}
    // beat1: OFF=4096 CHUNK=512 -> {data_in[511:0], wbuf_cur[4095:0]}
    wire [FW-1:0] arm0 = { wbuf_cur[FW-1:IW], data_in[IW-1:0] };
    wire [FW-1:0] arm1 = { data_in[511:0], wbuf_cur[IW-1:0] };
    wire [FW-1:0] mux  = g_idx ? arm1 : arm0;

    integer i, mux_vs_gold=0, var_vs_gold=0;
    initial begin
      repeat(2) @(negedge clk);
      for (i=0;i<400;i=i+1) begin
        wbuf_cur = {$random,$random,$random,$random,$random,$random,$random,$random,
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
                    $random,$random,$random,$random,$random,$random,$random,$random,
                    $random,$random,$random,$random,$random,$random,$random,$random,
                    $random,$random,$random,$random,$random,$random};
        data_in  = {$random,$random,$random,$random,$random,$random,$random,$random,
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
        g_idx = i[0];
        @(negedge clk);
        if (mux !== gold) mux_vs_gold = mux_vs_gold + 1;
        if (varn !== gold) var_vs_gold = var_vs_gold + 1;
      end
      $display("MUX_vs_GOLD mism=%0d   VAR_vs_GOLD mism=%0d   (400 trials, both g_idx)", mux_vs_gold, var_vs_gold);
      if (mux_vs_gold==0) $display("MUX MATCHES SPEC: PASS");
      else                $display("MUX DEVIATES FROM SPEC: FAIL");
      $finish;
    end
endmodule
`default_nettype wire
