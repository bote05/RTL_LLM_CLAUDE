`timescale 1ns/1ps
// Verify node_add_4 (after BP-FIX) is LOSSLESS under output backpressure.
// ready_out driven by a FREE-RUNNING block (never freezes). Capture data_out at the
// handshake (valid_out & ready_out). Correct (handshaked) module: the captured SEQUENCE
// is identical between mode0 (always-ready) and mode1 (gappy ready_out), 0 mismatch.
// node_add_4: OC=512, CHANNEL_TILE=32 -> beats_per_pixel = OC/32 = 16. 3 px -> 48 beats.
module node_add_4_bp_test;
    localparam integer BPP = 16;       // OC/32 = 512/32
    localparam integer NPIX = 3;
    localparam integer EXP = BPP*NPIX; // 48

    reg clk=0, rst_n=0;
    reg valid_in=0;
    wire ready_in;
    reg [511:0] data_in=0;
    wire valid_out;
    wire [255:0] data_out;
    reg  ready_out_sig=1;
    reg  gappy=0;            // 0 => always ready, 1 => gappy
    integer bpc=0;

    node_add_4 dut(.clk(clk),.rst_n(rst_n),.valid_in(valid_in),.ready_in(ready_in),
                   .data_in(data_in),.valid_out(valid_out),.ready_out(ready_out_sig),.data_out(data_out));

    always #5 clk=~clk;
    // free-running ready_out (stable at posedge), never freezes
    always @(negedge clk) begin
        bpc <= bpc + 1;
        ready_out_sig <= gappy ? (((bpc) % 3) != 0) : 1'b1;
    end

    integer mode, i;
    reg [255:0] capbuf[0:1][0:511];
    integer capn[0:1];

    task feed_pixel;
        integer b, c; reg [511:0] d;
        begin
            for (b=0;b<BPP;b=b+1) begin
                d=0;
                for (c=0;c<32;c=c+1) begin
                    d[c*8 +: 8]     = (b*32+c) % 50 - 25;
                    d[256+c*8 +: 8] = (b*32+c*3) % 60 - 30;
                end
                // wait until the DUT can accept a gather beat, then present exactly one
                @(negedge clk); while(!ready_in) @(negedge clk);
                valid_in=1; data_in=d;
                @(negedge clk); valid_in=0;   // one-beat pulse; no over-feed
            end
        end
    endtask

    // capture (always-on, both modes) at the handshake
    always @(posedge clk) begin
        if (rst_n && valid_out && ready_out_sig && capn[mode] < 512) begin
            capbuf[mode][capn[mode]] = data_out; capn[mode] = capn[mode] + 1;
        end
    end

    integer p;
    initial begin
        for (mode=0; mode<2; mode=mode+1) begin
            rst_n=0; valid_in=0; capn[mode]=0; bpc=0; gappy=(mode==1);
            repeat(4) @(negedge clk); rst_n=1; @(negedge clk);
            for (p=0;p<NPIX;p=p+1) feed_pixel;
            repeat(800) @(negedge clk);   // drain (compute ~515 cyc + stream + gappy slack)
        end
        $display("mode0 (always-ready) captured %0d beats", capn[0]);
        $display("mode1 (gappy-ready)  captured %0d beats", capn[1]);
        begin integer mism; mism=0;
            for (i=0; i<capn[0] && i<capn[1]; i=i+1)
                if (capbuf[0][i] !== capbuf[1][i]) mism=mism+1;
            $display("seq mismatch = %0d ; count_diff = %0d", mism, capn[0]-capn[1]);
            if (capn[0]==EXP && capn[1]==EXP && mism==0)
                $display("RESULT: LOSSLESS + complete (%0d beats) -- BP-FIX WORKS", EXP);
            else
                $display("RESULT: LOSSY/incomplete -- mode0=%0d mode1=%0d mism=%0d", capn[0], capn[1], mism);
        end
        $finish;
    end
endmodule
