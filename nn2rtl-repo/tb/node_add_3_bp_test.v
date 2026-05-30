`timescale 1ns/1ps
// Verify node_add_3 (after BP-FIX) is LOSSLESS under output backpressure.
// ready_out driven by a FREE-RUNNING block (never freezes). Capture data_out at the
// handshake (valid_out & ready_out). Correct module: mode0(always-ready)==mode1(gappy),
// both capture 48 beats (3 px x 16). Pre-fix: mode1 dropped beats.
// node_add_3: OC=512, channel_tile=32 => beats_per_pixel = 16.
module node_add_3_bp_test;
    reg clk=0, rst_n=0;
    reg valid_in=0;
    wire ready_in;
    reg [511:0] data_in=0;
    wire valid_out;
    wire [255:0] data_out;
    reg  ready_out_sig=1;
    reg  gappy=0;            // 0 => always ready, 1 => gappy
    integer bpc=0;

    node_add_3 dut(.clk(clk),.rst_n(rst_n),.valid_in(valid_in),.ready_in(ready_in),
                   .data_in(data_in),.valid_out(valid_out),.ready_out(ready_out_sig),.data_out(data_out));

    always #5 clk=~clk;
    // free-running ready_out (stable at posedge)
    always @(negedge clk) begin
        bpc <= bpc + 1;
        ready_out_sig <= gappy ? (((bpc) % 3) != 0) : 1'b1;
    end

    integer mode, cyc, i;
    reg [255:0] capbuf[0:1][0:511];
    integer capn[0:1];

    task feed_pixel;
        integer b, c; reg [511:0] d;
        begin
            for (b=0;b<16;b=b+1) begin
                d=0;
                for (c=0;c<32;c=c+1) begin
                    d[c*8 +: 8]     = (b*32+c) % 50 - 25;
                    d[256+c*8 +: 8] = (b*32+c*3) % 60 - 30;
                end
                @(negedge clk); while(!ready_in) @(negedge clk);
                valid_in=1; data_in=d;
                @(negedge clk); valid_in=0;
            end
            valid_in=0;
        end
    endtask

    // capture (always-on, both modes)
    always @(posedge clk) begin
        if (rst_n && valid_out && ready_out_sig && capn[mode] < 512) begin
            capbuf[mode][capn[mode]] = data_out; capn[mode] = capn[mode] + 1;
        end
    end

    initial begin
        for (mode=0; mode<2; mode=mode+1) begin
            rst_n=0; valid_in=0; capn[mode]=0; bpc=0; gappy=(mode==1);
            repeat(4) @(negedge clk); rst_n=1; @(negedge clk);
            feed_pixel; feed_pixel; feed_pixel;
            repeat(2000) @(negedge clk);   // drain (compute serializes OC=512)
        end
        $display("mode0 (always-ready) captured %0d beats", capn[0]);
        $display("mode1 (gappy-ready)  captured %0d beats", capn[1]);
        begin integer mism; mism=0;
            for (i=0; i<capn[0] && i<capn[1]; i=i+1)
                if (capbuf[0][i] !== capbuf[1][i]) mism=mism+1;
            $display("seq mismatch = %0d ; count_diff = %0d", mism, capn[0]-capn[1]);
            if (capn[0]==48 && capn[1]==48 && mism==0)
                $display("RESULT: LOSSLESS + complete (48 beats) -- BP-FIX WORKS");
            else
                $display("RESULT: LOSSY/incomplete -- mode0=%0d mode1=%0d mism=%0d", capn[0], capn[1], mism);
        end
        $finish;
    end
endmodule
