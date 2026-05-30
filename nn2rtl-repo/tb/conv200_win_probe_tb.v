`timescale 1ns/1ps
// conv200_win_probe_tb -- directly probe the WINDOW the MAC consumes at each
// start_mac (output_fires) pulse, under two backpressure regimes, and emit a
// compact signature per output pixel. Divergence => the window delivered to
// the MAC differs under backpressure. The signed tap-sum tells us whether the
// accumulator is SMALLER (compression) or merely shifted.
//
// We hierarchically reference window_flat (the datapath input) and the
// scheduler/line_buf internal state at the cycle start_mac pulses.

module conv200_win_probe_tb;
    localparam integer IC=64, OC=64, IH=56, IW=56, OH=56, OW=56;
    localparam integer KH=3, KW=3;
    localparam integer IN_BEATS=2;
    localparam integer TOTAL_IN_PIX = IH*IW;     // 3136
    localparam integer STOP_OUTPIX = 900;        // bounded run for speed

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

    integer feed_pix, feed_beat, bp_ctr;

    always @(*) begin
        if (mode==0) ready_out = 1'b1;
        else         ready_out = (bp_ctr == 0);   // ready 1 of every 3 cycles
    end
    always @(posedge clk) if (rst_n) bp_ctr <= (bp_ctr==2) ? 0 : bp_ctr+1;

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

    // ---- per-output-pixel window signature ----
    // store signature for mode 0, compare in mode 1.
    integer sig_sum [0:OH*OW-1];   // signed tap sum
    integer sig_row [0:OH*OW-1];
    integer sig_col [0:OH*OW-1];
    integer sig_os  [0:OH*OW-1];
    integer sig_rv  [0:OH*OW-1];
    integer of_count;

    // compute signed sum of all KH*KW*IC taps from dut.window_flat
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

    // sample at the cycle sched_output_fires is high (the start_mac cycle):
    // the window is the receptive field the MAC will consume.
    integer cur_sum, cur_row, cur_col, cur_os, cur_rv;
    integer mism_win, mism_first;

    always @(posedge clk) begin
        if (rst_n && dut.sched_output_fires) begin
            cur_sum = win_tapsum();
            cur_row = dut.sched_in_row;
            cur_col = dut.sched_in_col;
            cur_os  = dut.lbw.oldest_slot;
            cur_rv  = dut.lbw.row_valid;
            if (mode==0) begin
                if (of_count < OH*OW) begin
                    sig_sum[of_count] = cur_sum;
                    sig_row[of_count] = cur_row;
                    sig_col[of_count] = cur_col;
                    sig_os[of_count]  = cur_os;
                    sig_rv[of_count]  = cur_rv;
                end
            end else begin
                if (of_count < OH*OW) begin
                    if (cur_sum !== sig_sum[of_count]) begin
                        if (mism_first < 0) mism_first = of_count;
                        if (mism_win < 40)
                            $display("WIN MISMATCH outpix %0d (r=%0d c=%0d | nobp r=%0d c=%0d): tapsum nobp=%0d bp=%0d  (bp-nobp=%0d) | os nobp=%0d bp=%0d rv nobp=%b bp=%b",
                                of_count, cur_row, cur_col, sig_row[of_count], sig_col[of_count],
                                sig_sum[of_count], cur_sum, cur_sum - sig_sum[of_count],
                                sig_os[of_count], cur_os, sig_rv[of_count][2:0], cur_rv[2:0]);
                        mism_win = mism_win + 1;
                    end
                end
            end
            of_count = of_count + 1;
        end
    end

    integer m;
    integer settle;
    integer net_bias;

    initial begin
        mism_win=0; mism_first=-1; net_bias=0;
        for (m=0; m<2; m=m+1) begin
            mode = m;
            rst_n=0; valid_in=0; data_in=0; feed_pix=0; feed_beat=0; bp_ctr=0; of_count=0;
            repeat (8) @(posedge clk);
            rst_n=1;
            @(posedge clk);
            while (feed_pix < TOTAL_IN_PIX && of_count < STOP_OUTPIX) begin
                drive_beat;
                valid_in = 1;
                #1;
                if (ready_in) begin
                    if (feed_beat == IN_BEATS-1) begin feed_beat=0; feed_pix=feed_pix+1; end
                    else feed_beat = feed_beat + 1;
                end
                @(posedge clk);
            end
            valid_in = 0;
            settle = 0;
            while (of_count < STOP_OUTPIX && settle < 2000000) begin
                @(posedge clk);
                settle = settle + 1;
            end
            $display("[mode %0d] output_fires seen = %0d (bounded stop=%0d, feed_pix=%0d)", m, of_count, STOP_OUTPIX, feed_pix);
        end
        // net bias of bp vs nobp tap sums (over matched pixels)
        $display("=== WINDOW MISMATCHES: %0d ; first at outpix %0d ===", mism_win, mism_first);
        $finish;
    end

    initial begin
        #3_000_000_000;
        $display("TIMEOUT of_count=%0d mode=%0d", of_count, mode);
        $finish;
    end
endmodule
