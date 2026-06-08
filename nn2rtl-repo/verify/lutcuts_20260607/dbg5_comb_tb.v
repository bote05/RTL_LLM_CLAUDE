`timescale 1ns/1ps
`default_nettype none
// Pure combinational check of the insert expression for FULL_W=4608, IN_W=4096.
module dbg5_comb_tb;
    localparam FW=4608, IW=4096;
    reg [FW-1:0] wbuf_cur;
    reg [IW-1:0] data_in;
    reg [0:0]    g_idx;

    // variable reference
    reg [FW-1:0] var_next;
    always @(*) begin
        var_next = wbuf_cur;
        var_next[g_idx*IW +: IW] = data_in;
    end

    // mux candidate (replicate g_low / g_high logic)
    // beat0: OFF=0 CHUNK=4096 -> {wbuf_cur[4607:4096], data_in[4095:0]}
    // beat1: OFF=4096 CHUNK=512 OFF+CHUNK=4608=FW -> {data_in[511:0], wbuf_cur[4095:0]}
    wire [FW-1:0] arm0 = { wbuf_cur[FW-1:IW], data_in[IW-1:0] };
    wire [FW-1:0] arm1 = { data_in[511:0], wbuf_cur[IW-1:0] };
    wire [FW-1:0] mux_next = g_idx ? arm1 : arm0;

    integer i, mism=0;
    initial begin
      for (i=0;i<200;i=i+1) begin
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
                    $random,$random,$random,$random,$random,$random};  // 144 -> trim
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
        #1;
        if (var_next !== mux_next) begin
          mism=mism+1;
          if (mism<=3) $display("COMB MISM i=%0d g_idx=%0d  var_lo64=%h mux_lo64=%h  var_hi64=%h mux_hi64=%h",
             i, g_idx, var_next[63:0], mux_next[63:0], var_next[FW-1:FW-64], mux_next[FW-1:FW-64]);
        end
      end
      $display("dbg5 comb mism=%0d", mism);
      $finish;
    end
endmodule
`default_nettype wire
