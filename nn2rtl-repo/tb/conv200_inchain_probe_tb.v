`timescale 1ns/1ps
// conv200_inchain_probe_tb -- THE decisive probe that prior probes missed.
//
// Prior probes (conv200_bp_probe_tb, conv200_win_probe_tb) held valid_in HIGH
// every cycle and only toggled ready_out. That is NOT the in-chain condition:
// in the real chain conv_200's PRODUCER (conv_198+skid) presents valid_in
// INTERMITTENTLY, and that intermittency is correlated with downstream
// backpressure (spatial_run). This TB reproduces that: in mode 1 BOTH valid_in
// and ready_out are throttled.
//
// We key the per-output-pixel window signature by the OUTPUT COORDINATE
// (sched_in_row,sched_in_col at output_fires) so that reordering/dropping
// cannot mask a window-content mismatch. We compare tapsum at the SAME coord
// between the clean run (mode 0) and the throttled run (mode 1). A
// SYSTEMATICALLY SMALLER tapsum under throttling == the compression mechanism.

module conv200_inchain_probe_tb;
    localparam integer IC=64, OC=64, IH=56, IW=56, OH=56, OW=56;
    localparam integer KH=3, KW=3;
    localparam integer IN_BEATS=2;
    localparam integer TOTAL_IN_PIX = IH*IW;     // 3136
    localparam integer STOP_OUTPIX = 1200;       // bounded run

    reg clk=0, rst_n=0;
    reg valid_in=0;
    wire ready_in;
    reg [255:0] data_in=0;
    wire valid_out;
    reg  ready_out=1;
    wire [255:0] data_out;

    integer mode;

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

    // mode 0: never backpressure, always-valid producer.
    // mode 1: throttle ready_out (downstream) AND throttle valid_in (producer),
    //         on DIFFERENT periods so their phases drift and the producer is
    //         frequently not-valid exactly while downstream is draining.
    reg vin_gate;
    always @(*) begin
        if (mode==0) ready_out = 1'b1;
        else         ready_out = (bp_ctr < 2);    // ready 2 of every 7 cycles
    end
    always @(*) begin
        if (mode==0) vin_gate = 1'b1;
        else         vin_gate = (vin_ctr < 3);     // valid 3 of every 5 cycles
    end
    always @(posedge clk) if (rst_n) begin
        bp_ctr  <= (bp_ctr ==6) ? 0 : bp_ctr +1;
        vin_ctr <= (vin_ctr==4) ? 0 : vin_ctr+1;
    end

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

    // signature keyed by output COORDINATE index = of_count (production order is
    // deterministic in the scheduler since coords walk a fixed sequence). We ALSO
    // record (row,col) and assert they match between modes; if they ever differ
    // we know we are comparing different coords (a drop) and we flag it.
    integer sig_sum [0:OH*OW-1];
    integer sig_row [0:OH*OW-1];
    integer sig_col [0:OH*OW-1];
    integer of_count;

    function integer win_tapsum;
        integer t;
        reg signed [7:0] b;
        integer s;
        begin
            s = 0;
            for (t=0; t<KH*KW*IC; t=t+1) begin
                b = dut.window_flat[t*8 +: 8];
                s = s + b;
            end
            win_tapsum = s;
        end
    endfunction

    // count zero-taps in the bottom row's rightmost column (bypass_reg path) and
    // the history rows' rightmost column (q_reg path) so we can localize WHICH
    // taps go missing.
    function integer count_zero_col;
        input integer rowsel;   // 0..KH-1
        integer ic, base;
        reg signed [7:0] b;
        integer z;
        begin
            z = 0;
            base = (rowsel*KW*IC + (KW-1)*IC)*8;   // rightmost column (kw=KW-1)
            for (ic=0; ic<IC; ic=ic+1) begin
                b = dut.window_flat[base + ic*8 +: 8];
                if (b == 0) z = z + 1;
            end
            count_zero_col = z;
        end
    endfunction

    integer cur_sum, cur_row, cur_col;
    integer cur_z0, cur_z1, cur_z2;
    integer mism_win, mism_first, coord_mism;
    integer net_bias, neg_bias_pix, pos_bias_pix;
    integer tot_z2_m0, tot_z2_m1;

    always @(posedge clk) begin
        if (rst_n && dut.sched_output_fires) begin
            cur_sum = win_tapsum();
            cur_row = dut.sched_in_row;
            cur_col = dut.sched_in_col;
            cur_z0  = count_zero_col(0);
            cur_z1  = count_zero_col(1);
            cur_z2  = count_zero_col(2);
            if (mode==0) begin
                if (of_count < OH*OW) begin
                    sig_sum[of_count] = cur_sum;
                    sig_row[of_count] = cur_row;
                    sig_col[of_count] = cur_col;
                    tot_z2_m0 = tot_z2_m0 + cur_z2;
                end
            end else begin
                if (of_count < OH*OW) begin
                    tot_z2_m1 = tot_z2_m1 + cur_z2;
                    if (cur_row !== sig_row[of_count] || cur_col !== sig_col[of_count])
                        coord_mism = coord_mism + 1;
                    if (cur_sum !== sig_sum[of_count]) begin
                        if (mism_first < 0) mism_first = of_count;
                        net_bias = net_bias + (cur_sum - sig_sum[of_count]);
                        if (cur_sum - sig_sum[of_count] < 0) neg_bias_pix = neg_bias_pix + 1;
                        else pos_bias_pix = pos_bias_pix + 1;
                        if (mism_win < 30)
                            $display("WIN MISMATCH outpix %0d (r=%0d c=%0d): tapsum m0=%0d m1=%0d (m1-m0=%0d) | zerocol row0=%0d row1=%0d row2=%0d",
                                of_count, cur_row, cur_col,
                                sig_sum[of_count], cur_sum, cur_sum - sig_sum[of_count],
                                cur_z0, cur_z1, cur_z2);
                        mism_win = mism_win + 1;
                    end
                end
            end
            of_count = of_count + 1;
        end
    end

    integer m, settle;

    initial begin
        mism_win=0; mism_first=-1; coord_mism=0; net_bias=0;
        neg_bias_pix=0; pos_bias_pix=0; tot_z2_m0=0; tot_z2_m1=0;
        for (m=0; m<2; m=m+1) begin
            mode = m;
            rst_n=0; valid_in=0; data_in=0; feed_pix=0; feed_beat=0;
            bp_ctr=0; vin_ctr=0; of_count=0;
            repeat (8) @(posedge clk);
            rst_n=1;
            @(posedge clk);
            while (feed_pix < TOTAL_IN_PIX && of_count < STOP_OUTPIX) begin
                drive_beat;
                valid_in = vin_gate;
                #1;
                if (valid_in && ready_in) begin
                    if (feed_beat == IN_BEATS-1) begin feed_beat=0; feed_pix=feed_pix+1; end
                    else feed_beat = feed_beat + 1;
                end
                @(posedge clk);
            end
            valid_in = 0;
            settle = 0;
            while (of_count < STOP_OUTPIX && settle < 4000000) begin
                @(posedge clk);
                settle = settle + 1;
            end
            $display("[mode %0d] output_fires seen = %0d (feed_pix=%0d)", m, of_count, feed_pix);
        end
        $display("=== WINDOW MISMATCHES: %0d ; first at outpix %0d ; coord_mism=%0d ===",
                 mism_win, mism_first, coord_mism);
        $display("=== NET tapsum bias (m1-m0) over mismatched pix: %0d  (neg=%0d pos=%0d) ===",
                 net_bias, neg_bias_pix, pos_bias_pix);
        $display("=== total zero-taps in bottom-row rightmost col: m0=%0d m1=%0d ===",
                 tot_z2_m0, tot_z2_m1);
        $finish;
    end

    initial begin
        #3_000_000_000;
        $display("TIMEOUT of_count=%0d mode=%0d", of_count, mode);
        $finish;
    end
endmodule
