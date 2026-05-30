`timescale 1ns/1ps
// conv200_realvalue_tb -- feed the REAL in-chain input (relu_1 output =
// node_conv_200.goldin frame 0) to the standalone node_conv_200 wrapper,
// CLEAN (no input throttle, no output backpressure), capture data_out, and
// dump it. Compared offline to node_conv_200.goldout. If clean output ==
// goldout, the standalone module is byte-exact on real data and the in-chain
// compression is an INTEGRATION/handshake effect. If it differs even clean,
// the bug is data-dependent inside the datapath/scale path.
//
// Each input pixel is 64 int8 channels fed over IN_BEATS=2 beats of 32 ch.
// goldin hex: one line per pixel = 64 bytes MSB-first (byte63..byte0).

module conv200_realvalue_tb;
    localparam integer IC=64, OC=64, IH=56, IW=56, OH=56, OW=56;
    localparam integer IN_BEATS=2, OUT_BEATS=2;
    localparam integer NPIX = IH*IW;       // 3136

    reg clk=0, rst_n=0;
    reg valid_in=0;
    wire ready_in;
    reg [255:0] data_in=0;
    wire valid_out;
    reg  ready_out=1;
    wire [255:0] data_out;

    node_conv_200 dut (
        .clk(clk), .rst_n(rst_n),
        .valid_in(valid_in), .ready_in(ready_in), .data_in(data_in),
        .valid_out(valid_out), .ready_out(ready_out), .data_out(data_out)
    );
    always #5 clk = ~clk;

    // input pixels: [pixel][512-bit] (64 bytes)
    reg [IC*8-1:0] gin [0:NPIX-1];
    // output capture: 2 beats/pixel * NPIX = 6272 beats of 256 bits
    reg [255:0] ocap [0:OUT_BEATS*NPIX-1];
    integer ocap_idx;
    integer fd, p, b;

    integer feed_pix, feed_beat;

    initial begin
        $readmemh("output/conv200_goldin_f0.hex", gin);
        ocap_idx=0; feed_pix=0; feed_beat=0;
        rst_n=0; valid_in=0; data_in=0;
        repeat (8) @(posedge clk);
        rst_n=1; @(posedge clk);

        // feed all pixels, 2 beats each, clean (valid held until accepted)
        while (feed_pix < NPIX) begin
            // beat 0 = channels 0..31 (low), beat 1 = channels 32..63 (high)
            data_in = gin[feed_pix][feed_beat*256 +: 256];
            valid_in = 1;
            #1;
            if (ready_in) begin
                if (feed_beat==IN_BEATS-1) begin feed_beat=0; feed_pix=feed_pix+1; end
                else feed_beat=feed_beat+1;
            end
            @(posedge clk);
        end
        valid_in=0;

        // drain
        repeat (200000) begin
            if (ocap_idx >= OUT_BEATS*NPIX) ; @(posedge clk);
        end

        // dump capture
        fd = $fopen("output/conv200_realvalue_out.hex","w");
        for (p=0; p<ocap_idx; p=p+1)
            $fwrite(fd, "%h\n", ocap[p]);
        $fclose(fd);
        $display("=== captured %0d output beats (expect %0d) ===", ocap_idx, OUT_BEATS*NPIX);
        $finish;
    end

    always @(posedge clk) begin
        if (rst_n && valid_out && ready_out) begin
            if (ocap_idx < OUT_BEATS*NPIX) ocap[ocap_idx] <= data_out;
            ocap_idx <= ocap_idx + 1;
        end
    end

    initial begin
        #4_000_000_000;
        $display("TIMEOUT ocap_idx=%0d feed_pix=%0d", ocap_idx, feed_pix);
        $finish;
    end
endmodule
