`timescale 1ns/1ps
// conv200_bp_probe_tb -- drive node_conv_200 with deterministic non-uniform
// input and observe data_out under TWO backpressure regimes:
//   * regime A: ready_out tied high (no backpressure) -- this is the equiv
//     baseline that is known byte-exact.
//   * regime B: ready_out toggled (multi-cycle output stalls) -- the in-chain
//     condition.
// We capture both output streams and diff them. We also tap window_flat and
// the datapath accumulator at the moment start_mac fires to see whether the
// window delivered under backpressure differs.
//
// The DUT (node_conv_200) has a ready_out port wired in by
// scripts/wire_conv3x3_ready_out.py. If the on-disk node_conv_200.v does NOT
// have a ready_out port, this TB instantiates the backpressured-port variant
// produced by apply_3x3_backpressure.py (no --equiv).

module conv200_bp_probe_tb;
    localparam integer IC=64, OC=64, IH=56, IW=56, OH=56, OW=56;
    localparam integer IN_BEATS=2, OUT_BEATS=2;
    localparam integer TOTAL_IN_PIX = IH*IW;     // 3136
    localparam integer TOTAL_OUT_PIX = OH*OW;    // 3136
    localparam integer STOP_BEATS = 400;         // bounded run for speed (200 outpix)

    reg clk=0, rst_n=0;
    reg valid_in=0;
    wire ready_in;
    reg [255:0] data_in=0;
    wire valid_out;
    reg  ready_out=1;
    wire [255:0] data_out;

    // mode 0 = no backpressure (ready_out always 1)
    // mode 1 = periodic backpressure
    integer mode;

    node_conv_200 dut (
        .clk(clk), .rst_n(rst_n),
        .valid_in(valid_in), .ready_in(ready_in), .data_in(data_in),
        .valid_out(valid_out), .ready_out(ready_out), .data_out(data_out)
    );

    always #5 clk = ~clk;

    // ---- deterministic non-uniform input pixel generator ----
    // Each input pixel (IH*IW of them) has IC=64 int8 channels. We make them
    // vary by (row,col,channel) so missing taps change the accumulator.
    // 2 beats per pixel (32 channels each).
    function [7:0] pix_byte;
        input integer r;
        input integer c;
        input integer ch;
        integer v;
        begin
            v = (r*7 + c*13 + ch*3 + 5);
            // fold into signed-ish [-100,100] range, non-uniform
            v = (v % 201) - 100;
            pix_byte = v[7:0];
        end
    endfunction

    // capture output streams for both modes
    reg [255:0] cap [0:1][0:STOP_BEATS-1];
    integer cap_idx;

    // capture the datapath's TRUE computed pixel (lib_data_out_w) at every
    // lib_valid_out_w pulse -- before the streamer can drop it. Indexed by
    // the datapath's own pixel counter (dp_pix). If these match between modes
    // the accumulator/MAC is correct and the bug is purely streamer DELIVERY.
    reg [255:0] dpcap [0:1][0:STOP_BEATS];   // OC=64 -> 1 entry/pixel (512 bits won't fit; store low 256)
    reg [511:0] dpfull [0:1][0:STOP_BEATS];
    integer dp_pix;

    integer feed_pix;     // which input pixel we are on
    integer feed_beat;    // which beat of that pixel (0..IN_BEATS-1)
    integer bp_ctr;

    // drop detector: count cycles where the datapath asserted a fresh output
    // (lib_valid_out_w) while the streamer was still busy (out_busy) -> the
    // computed pixel cannot be latched and is LOST.
    integer drop_count;

    // drive ready_out
    // mode 1 mimics in-chain `spatial_run`-style LONG contiguous stalls:
    // ready_out low for a long burst, then high to drain. Period 600:
    // low for 550, high for 50. A long low burst lets a gap-advance MAC
    // complete while out_busy still holds the prior pixel -> drop.
    always @(*) begin
        if (mode==0) ready_out = 1'b1;
        else         ready_out = (bp_ctr[2:0] != 3'd0);  // MILD: high 7 of every 8 (realistic skid bp)
    end

    // capture data_out on accepted beats + detect datapath-output drops
    always @(posedge clk) begin
        if (rst_n && valid_out && ready_out) begin
            if (cap_idx < STOP_BEATS) cap[mode][cap_idx] <= data_out;
            cap_idx <= cap_idx + 1;
        end
        if (rst_n && dut.lib_valid_out_w) begin
            if (dut.out_busy) drop_count <= drop_count + 1;
            if (dp_pix <= STOP_BEATS) dpfull[mode][dp_pix] <= dut.lib_data_out_w;
            dp_pix <= dp_pix + 1;
        end
        if (rst_n) bp_ctr <= (bp_ctr==599) ? 0 : bp_ctr+1;
    end

    // feed inputs: hold a beat until accepted (ready_in)
    task drive_beat;
        integer ch, basech;
        reg [255:0] d;
        begin
            basech = feed_beat*32;
            d = 0;
            for (ch=0; ch<32; ch=ch+1)
                d[ch*8 +: 8] = pix_byte(feed_pix/IW, feed_pix%IW, basech+ch);
            data_in = d;
        end
    endtask

    integer i, m;
    integer out_count;
    integer mismatches;
    integer settle;

    initial begin
        // ---------- run both modes ----------
        for (m=0; m<2; m=m+1) begin
            mode = m;
            // reset
            rst_n=0; valid_in=0; data_in=0; feed_pix=0; feed_beat=0;
            cap_idx=0; bp_ctr=0; out_count=0; drop_count=0; dp_pix=0;
            repeat (8) @(posedge clk);
            rst_n=1;
            @(posedge clk);

            // feed all input pixels, beat by beat.
            while (feed_pix < TOTAL_IN_PIX && cap_idx < STOP_BEATS) begin
                drive_beat;
                valid_in = 1;
                #1;                 // let ready_in settle for this cycle
                if (ready_in) begin
                    if (feed_beat == IN_BEATS-1) begin
                        feed_beat = 0;
                        feed_pix  = feed_pix + 1;
                    end else begin
                        feed_beat = feed_beat + 1;
                    end
                end
                @(posedge clk);
            end
            valid_in = 0;
            // let output drain
            settle = 0;
            while (cap_idx < STOP_BEATS && settle < 5000000) begin
                @(posedge clk);
                settle = settle + 1;
            end
            $display("[mode %0d] captured %0d output beats, drop_count=%0d, feed_pix=%0d",
                     m, cap_idx, drop_count, feed_pix);
        end

        // ---------- diff ----------
        mismatches = 0;
        for (i=0; i<STOP_BEATS; i=i+1) begin
            if (cap[0][i] !== cap[1][i]) begin
                if (mismatches < 30)
                    $display("MISMATCH beat %0d (pix %0d, tile %0d): no_bp=%h  bp=%h",
                             i, i/2, i%2, cap[0][i], cap[1][i]);
                mismatches = mismatches + 1;
            end
        end
        $display("=== TOTAL OUTPUT-BEAT MISMATCHES (bp vs no_bp): %0d / %0d ===",
                 mismatches, STOP_BEATS);

        // ---- compare the datapath's TRUE COMPUTED pixels (pre-streamer) ----
        // Indexed by the datapath's own production order. If these match, the
        // MAC/accumulator is identical between modes -> bug is purely DELIVERY
        // (streamer drop), NOT accumulator compression.
        mismatches = 0;
        for (i=0; i<150; i=i+1) begin
            if (dpfull[0][i] !== dpfull[1][i]) begin
                if (mismatches < 8)
                    $display("DP-COMPUTED MISMATCH pix %0d:\n  no_bp=%h\n     bp=%h",
                             i, dpfull[0][i], dpfull[1][i]);
                mismatches = mismatches + 1;
            end
        end
        $display("=== DATAPATH-COMPUTED-PIXEL MISMATCHES (first 150): %0d ===", mismatches);
        $finish;
    end

    // safety timeout
    initial begin
        #2_000_000_000;
        $display("TIMEOUT");
        $finish;
    end
endmodule
