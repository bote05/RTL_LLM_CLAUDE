// Self-checking TB for output_serializer: feeds N deterministic 8000-bit words, reassembles the
// 32x256b beats, and asserts the concatenation equals the input word (byte-exact). Watchdog
// guarantees termination. PASS prints "SER_RESULT PASS mismatch=0".
`timescale 1ns/1ps
module tb_output_serializer;
    localparam integer W_IN   = 8000;
    localparam integer BEATW  = 256;
    localparam integer NBEATS = (W_IN + BEATW - 1) / BEATW; // 32
    localparam integer NWORDS = 6;

    reg clk = 0, rst_n = 0;
    reg valid_in = 0;
    reg [W_IN-1:0] data_in = 0;
    wire ready_out, valid_out, last_out;
    wire [BEATW-1:0] data_out;
    reg ready_in = 1;

    output_serializer #(.W_IN(W_IN), .BEATW(BEATW)) dut (
        .clk(clk), .rst_n(rst_n),
        .valid_in(valid_in), .data_in(data_in), .ready_out(ready_out),
        .valid_out(valid_out), .data_out(data_out), .last_out(last_out), .ready_in(ready_in)
    );

    always #5 clk = ~clk;

    integer w, j, beat, mism, total_mism;
    reg [W_IN-1:0]  golden;
    reg [NBEATS*BEATW-1:0] reasm;

    function [7:0] gbyte(input integer ww, input integer jj);
        gbyte = (ww*7 + jj*13 + 5) & 8'hFF;
    endfunction

    // watchdog
    initial begin
        #2000000;
        $display("SER_RESULT FAIL TIMEOUT");
        $finish;
    end

    initial begin
        total_mism = 0;
        ready_in = 1'b1;
        rst_n = 0; repeat (4) @(posedge clk); rst_n = 1; @(posedge clk);

        for (w = 0; w < NWORDS; w = w + 1) begin
            golden = {W_IN{1'b0}};
            for (j = 0; j < W_IN/8; j = j + 1) golden[j*8 +: 8] = gbyte(w, j);

            // present until accepted: an accept happens on the posedge where ready_out==1 & valid_in
            @(negedge clk); valid_in = 1; data_in = golden;
            @(posedge clk);                 // ready_out==1 here (idle) -> word latched THIS edge
            @(negedge clk); valid_in = 0;

            // collect NBEATS beats (ready_in held high)
            reasm = {(NBEATS*BEATW){1'b0}};
            beat = 0;
            while (beat < NBEATS) begin
                @(posedge clk);
                if (valid_out && ready_in) begin
                    reasm[beat*BEATW +: BEATW] = data_out;
                    beat = beat + 1;
                end
            end

            mism = 0;
            for (j = 0; j < W_IN/8; j = j + 1)
                if (reasm[j*8 +: 8] !== golden[j*8 +: 8]) mism = mism + 1;
            total_mism = total_mism + mism;
            $display("word %0d: mismatch=%0d", w, mism);
        end

        if (total_mism == 0) $display("SER_RESULT PASS mismatch=0 (%0d words x %0d beats byte-exact)", NWORDS, NBEATS);
        else                 $display("SER_RESULT FAIL mismatch=%0d", total_mism);
        $finish;
    end
endmodule
