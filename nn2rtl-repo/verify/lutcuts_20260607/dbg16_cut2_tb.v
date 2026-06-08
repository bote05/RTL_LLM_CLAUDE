`timescale 1ns/1ps
`default_nettype none
// Cut 2 equiv: fixed-mux cont_slice == variable beat_buf[(slice_idx+1)*2048 +: 2048]
// for every REACHABLE slice_idx (0 .. WORDS_PER_BEAT-2), for the deployed WPB values.
module dbg16_cut2_tb #(parameter integer BUS_W=4096);
    reg clk=0; always #5 clk=~clk;
    localparam integer WORDS_PER_BEAT = BUS_W/2048;
    reg [BUS_W-1:0] beat_buf;
    reg [31:0] slice_idx;

    // fixed mux (mirror the patched cont_slice)
    reg [2047:0] cont_slice; integer cs_i;
    always @(*) begin
        cont_slice = beat_buf[1*2048 +: 2048];
        for (cs_i=0; cs_i<WORDS_PER_BEAT-1; cs_i=cs_i+1)
            if (slice_idx==cs_i) cont_slice = beat_buf[(cs_i+1)*2048 +: 2048];
    end
    // variable reference (only valid/in-range for slice_idx 0..WPB-2)
    wire [2047:0] var_slice = beat_buf[(slice_idx+1)*2048 +: 2048];

    integer i,k,si,mism=0;
    reg [63:0] rng=64'h13371337c0dec0de;
    task nr; begin rng=rng^(rng<<13); rng=rng^(rng>>7); rng=rng^(rng<<17); end endtask
    initial begin
      repeat(2) @(negedge clk);
      for (i=0;i<500;i=i+1) begin
        for (k=0;k<BUS_W;k=k+32) begin nr; beat_buf[k +: 32]=rng[31:0]; end
        for (si=0; si<WORDS_PER_BEAT-1; si=si+1) begin
          slice_idx=si;
          @(negedge clk);
          if (cont_slice !== var_slice) begin mism=mism+1; if(mism<=3)$display("CUT2 MISM BUS_W=%0d si=%0d",BUS_W,si); end
        end
      end
      $display("dbg16 BUS_W=%0d cont vs var mism=%0d", BUS_W, mism);
      if (mism==0) $display("RESULT: CUT2 BUS_W=%0d PASS", BUS_W);
      else $display("RESULT: CUT2 BUS_W=%0d FAIL", BUS_W);
      $finish;
    end
endmodule

module dbg16_top;
    dbg16_cut2_tb #(.BUS_W(4096)) u_4096();  // deployed case (WPB=2)
endmodule
`default_nettype wire
