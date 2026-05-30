`timescale 1ns/1ps
// conv200_multiframe_probe_tb -- test the MULTI-FRAME regime under throttled
// valid_in + throttled ready_out. The line_buf BRAM is NOT cleared on
// frame_start; only row_valid masks stale prior-frame data. In-chain conv_200
// processes back-to-back frames continuously. This TB feeds N_FRAMES full
// frames; the per-frame input is identical (pix_byte depends only on r,c,ch),
// so EVERY frame's windows must be byte-identical to frame 0. Any divergence
// frame>0 vs frame 0 (esp. under throttling) is a multi-frame / row_valid bug.
//
// We compare, per (frame, output-pixel) the window tapsum to the frame-0
// clean-mode reference, all keyed by output-pixel index within the frame.

module conv200_multiframe_probe_tb;
    localparam integer IC=64, OC=64, IH=56, IW=56, OH=56, OW=56;
    localparam integer KH=3, KW=3;
    localparam integer IN_BEATS=2;
    localparam integer PIX_PER_FRAME = IH*IW;     // 3136
    localparam integer OUT_PER_FRAME = OH*OW;      // 3136
    localparam integer N_FRAMES = 3;

    reg clk=0, rst_n=0;
    reg valid_in=0;
    wire ready_in;
    reg [255:0] data_in=0;
    wire valid_out;
    reg  ready_out=1;
    wire [255:0] data_out;

    integer mode;   // 0 = clean, 1 = throttled

    node_conv_200 dut (
        .clk(clk), .rst_n(rst_n),
        .valid_in(valid_in), .ready_in(ready_in), .data_in(data_in),
        .valid_out(valid_out), .ready_out(ready_out), .data_out(data_out)
    );

    always #5 clk = ~clk;

    function [7:0] pix_byte;
        input integer r;
        input integer c;
        input integer ch;
        integer v;
        begin
            v = (r*7 + c*13 + ch*3 + 5);
            v = (v % 201) - 100;
            pix_byte = v[7:0];
        end
    endfunction

    integer feed_pix, feed_beat, bp_ctr, vin_ctr;
    reg vin_gate;
    always @(*) begin
        if (mode==0) ready_out = 1'b1;
        else         ready_out = (bp_ctr < 2);
    end
    always @(*) begin
        if (mode==0) vin_gate = 1'b1;
        else         vin_gate = (vin_ctr < 3);
    end
    always @(posedge clk) if (rst_n) begin
        bp_ctr  <= (bp_ctr ==6) ? 0 : bp_ctr +1;
        vin_ctr <= (vin_ctr==4) ? 0 : vin_ctr+1;
    end

    task drive_beat;
        integer ch, basech, pr;
        reg [255:0] d;
        begin
            pr = feed_pix % PIX_PER_FRAME;   // per-frame pixel index (input repeats each frame)
            basech = feed_beat*32;
            d = 0;
            for (ch=0; ch<32; ch=ch+1)
                d[ch*8 +: 8] = pix_byte(pr/IW, pr%IW, basech+ch);
            data_in = d;
        end
    endtask

    // frame-0 clean reference, indexed by per-frame output pixel index.
    integer ref_sum [0:OUT_PER_FRAME-1];
    integer of_count;       // global output_fires count across frames

    function integer win_tapsum;
        integer t; reg signed [7:0] b; integer s;
        begin
            s=0;
            for (t=0; t<KH*KW*IC; t=t+1) begin b=dut.window_flat[t*8 +: 8]; s=s+b; end
            win_tapsum=s;
        end
    endfunction

    integer cur_sum, ofp, fr;
    integer mism_total, mism_frame [0:N_FRAMES-1];
    integer net_bias_frame [0:N_FRAMES-1];
    integer first_mism_ofp;

    always @(posedge clk) begin
        if (rst_n && dut.sched_output_fires) begin
            cur_sum = win_tapsum();
            ofp = of_count % OUT_PER_FRAME;     // per-frame output index
            fr  = of_count / OUT_PER_FRAME;
            if (mode==0 && fr==0) begin
                ref_sum[ofp] = cur_sum;
            end else begin
                if (fr < N_FRAMES && cur_sum !== ref_sum[ofp]) begin
                    if (first_mism_ofp < 0) first_mism_ofp = ofp;
                    mism_total = mism_total + 1;
                    mism_frame[fr] = mism_frame[fr] + 1;
                    net_bias_frame[fr] = net_bias_frame[fr] + (cur_sum - ref_sum[ofp]);
                    if (mism_total <= 25)
                        $display("MULTIFRAME MISMATCH mode=%0d frame=%0d ofp=%0d: ref(f0clean)=%0d cur=%0d (delta=%0d)",
                            mode, fr, ofp, ref_sum[ofp], cur_sum, cur_sum - ref_sum[ofp]);
                end
            end
            of_count = of_count + 1;
        end
    end

    integer m, settle, k;

    initial begin
        mism_total=0; first_mism_ofp=-1;
        for (k=0;k<N_FRAMES;k=k+1) begin mism_frame[k]=0; net_bias_frame[k]=0; end

        // ---- PASS 1: mode 0 clean, capture frame-0 reference ----
        mode=0;
        rst_n=0; valid_in=0; data_in=0; feed_pix=0; feed_beat=0; bp_ctr=0; vin_ctr=0; of_count=0;
        repeat (8) @(posedge clk); rst_n=1; @(posedge clk);
        // feed exactly 1 frame of input (we only need frame-0 ref)
        while (of_count < OUT_PER_FRAME) begin
            drive_beat; valid_in=vin_gate; #1;
            if (valid_in && ready_in) begin
                if (feed_beat==IN_BEATS-1) begin feed_beat=0; feed_pix=feed_pix+1; end
                else feed_beat=feed_beat+1;
            end
            @(posedge clk);
            if (feed_pix >= PIX_PER_FRAME) valid_in=0;   // stop feeding after 1 frame
        end
        valid_in=0;
        $display("[clean f0] captured %0d ref windows", of_count);

        // ---- PASS 2: mode 1 throttled, N_FRAMES frames continuous ----
        mode=1;
        rst_n=0; valid_in=0; data_in=0; feed_pix=0; feed_beat=0; bp_ctr=0; vin_ctr=0; of_count=0;
        repeat (8) @(posedge clk); rst_n=1; @(posedge clk);
        while (of_count < N_FRAMES*OUT_PER_FRAME && feed_pix < N_FRAMES*PIX_PER_FRAME + 200) begin
            drive_beat; valid_in=vin_gate; #1;
            if (valid_in && ready_in) begin
                if (feed_beat==IN_BEATS-1) begin feed_beat=0; feed_pix=feed_pix+1; end
                else feed_beat=feed_beat+1;
            end
            @(posedge clk);
        end
        valid_in=0;
        settle=0;
        while (of_count < N_FRAMES*OUT_PER_FRAME && settle < 4000000) begin
            @(posedge clk); settle=settle+1;
        end
        $display("[throttled] of_count=%0d (target=%0d) feed_pix=%0d",
                 of_count, N_FRAMES*OUT_PER_FRAME, feed_pix);

        $display("=== MULTIFRAME MISMATCH TOTAL: %0d ; first ofp=%0d ===", mism_total, first_mism_ofp);
        for (k=0;k<N_FRAMES;k=k+1)
            $display("   frame %0d: mismatches=%0d  net_tapsum_bias=%0d", k, mism_frame[k], net_bias_frame[k]);
        $finish;
    end

    initial begin
        #6_000_000_000;
        $display("TIMEOUT of_count=%0d mode=%0d feed_pix=%0d", of_count, mode, feed_pix);
        $finish;
    end
endmodule
