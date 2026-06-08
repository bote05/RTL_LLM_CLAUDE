`timescale 1ns/1ps
`default_nettype none
// Clean random equiv test for retile_scatter S1.  Inputs are REGISTERED on the
// posedge (TB drives next-state in an always_ff), so they are stable for the
// whole cycle the DUT samples them, and buffers are compared in the SAME
// clocked block (no mid-cycle combinational sampling).
module scatter_rand_tb;
    reg clk=0, rst_n=0;
    always #5 clk=~clk;

    reg v_in, rd_down;
    reg [4095:0] din;

    wire a_vo,a_ro; wire [255:0] a_do;
    wire b_vo,b_ro; wire [255:0] b_do;
    retile_scatter #(.TILE_W(256),.N_TILES(18),.IN_W(4096),.IN_BEATS(2),.SPATIAL(196),.SYNTH_FIXED_MUX(0)) a (
        .clk(clk),.rst_n(rst_n),.valid_in(v_in),.ready_out(a_ro),.data_in(din),
        .valid_out(a_vo),.ready_down(rd_down),.drain_en(1'b1),.data_out(a_do),.wr_accept(),.stall_out());
    retile_scatter #(.TILE_W(256),.N_TILES(18),.IN_W(4096),.IN_BEATS(2),.SPATIAL(196),.SYNTH_FIXED_MUX(1)) b (
        .clk(clk),.rst_n(rst_n),.valid_in(v_in),.ready_out(b_ro),.data_in(din),
        .valid_out(b_vo),.ready_down(rd_down),.drain_en(1'b1),.data_out(b_do),.wr_accept(),.stall_out());

    integer mism=0; integer cyc=0; integer bb;
    reg [63:0] rng=64'hfeedfacecafef00d;
    task step; begin rng=rng^(rng<<13); rng=rng^(rng>>7); rng=rng^(rng<<17); end endtask

    // Drive inputs registered on posedge: stable for the whole next cycle.
    always @(posedge clk) begin
        if (!rst_n) begin v_in<=0; rd_down<=0; din<=0; end
        else begin
            step; v_in <= rng[0]|rng[3];
            step; rd_down <= rng[1]|rng[7];
            for (bb=0; bb<4096; bb=bb+32) begin step; din[bb +: 32] <= rng[31:0]; end
        end
    end

    // Compare registered DUT state + outputs at posedge (after the same edge
    // that updated buffers and the inputs DUT consumed last cycle).
    always @(posedge clk) begin
        if (rst_n) begin
            cyc = cyc + 1;
            if (a_vo !== b_vo) begin mism=mism+1; if(mism<=5)$display("cyc=%0d VO diff",cyc); end
            if (a_ro !== b_ro) begin mism=mism+1; if(mism<=5)$display("cyc=%0d RO diff",cyc); end
            if (a.buf0 !== b.buf0) begin mism=mism+1; if(mism<=5)$display("cyc=%0d BUF0 diff",cyc); end
            if (a.buf1 !== b.buf1) begin mism=mism+1; if(mism<=5)$display("cyc=%0d BUF1 diff",cyc); end
            if (a_vo && (a_do !== b_do)) begin mism=mism+1; if(mism<=5)$display("cyc=%0d DATA diff",cyc); end
        end
    end

    initial begin
        rst_n=0;
        repeat(4) @(posedge clk); rst_n=1;
        repeat(40000) @(posedge clk);
        if (mism==0) $display("SCATTER-RAND PASS mismatch=0 cyc=%0d", cyc);
        else         $display("SCATTER-RAND FAIL mismatch=%0d cyc=%0d", mism, cyc);
        $finish;
    end
endmodule
`default_nettype wire
