`timescale 1ns/1ps
// Equivalence: bram_fifo (sync-read BRAM, FWFT) must deliver the SAME ordered
// data stream as skip_fifo (async-read LUTRAM, FWFT) for the same push/pop
// randomized handshake. The ONLY allowed difference is fill latency (bram_fifo
// may lag), absorbed by the elastic handshake. We compare the SEQUENCE of beats
// popped, not the per-cycle timing.
module fifo_equiv_tb;
    localparam WIDTH = 16;
    localparam DEPTH = 64;
    localparam NBEATS = 4000;

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    // producer feeds both FIFOs with the SAME in_data when BOTH are ready.
    reg  [WIDTH-1:0] feed_data;
    reg              feed_valid;
    wire s_in_ready, b_in_ready;
    // each FIFO accepts independently; producer holds the same beat until BOTH
    // have accepted it, so both store the identical ordered stream.
    reg  s_accepted, b_accepted;

    wire s_out_valid, b_out_valid;
    wire [WIDTH-1:0] s_out_data, b_out_data;
    reg  s_out_ready, b_out_ready;

    skip_fifo #(.WIDTH(WIDTH), .DEPTH(DEPTH)) u_skip (
        .clk(clk), .rst_n(rst_n),
        .in_valid(feed_valid & ~s_accepted), .in_data(feed_data), .in_ready(s_in_ready),
        .out_valid(s_out_valid), .out_data(s_out_data), .out_ready(s_out_ready)
    );
    bram_fifo #(.WIDTH(WIDTH), .DEPTH(DEPTH)) u_bram (
        .clk(clk), .rst_n(rst_n),
        .in_valid(feed_valid & ~b_accepted), .in_data(feed_data), .in_ready(b_in_ready),
        .out_valid(b_out_valid), .out_data(b_out_data), .out_ready(b_out_ready)
    );

    // pop streams captured separately, compared in order.
    reg [WIDTH-1:0] s_q [0:NBEATS-1];
    reg [WIDTH-1:0] b_q [0:NBEATS-1];
    integer s_n = 0, b_n = 0;
    integer fed = 0;
    integer i;
    reg [31:0] lfsr_in = 32'h1234_5678;
    reg [31:0] lfsr_or = 32'h9abc_def0;

    function [31:0] nxt; input [31:0] s; begin
        nxt = (s >> 1) ^ (-(s & 1) & 32'hD0000001);
    end endfunction

    initial begin
        feed_valid = 0; feed_data = 0; s_accepted=0; b_accepted=0;
        s_out_ready=0; b_out_ready=0;
        repeat (4) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // drive for many cycles with random backpressure on both ends.
        for (i = 0; i < 60000; i = i + 1) begin
            // --- produce: keep a beat presented until both FIFOs accepted it ---
            if (!feed_valid && fed < NBEATS) begin
                feed_data  = fed[WIDTH-1:0];   // deterministic increasing payload
                feed_valid = 1; s_accepted=0; b_accepted=0;
            end
            // random output-ready backpressure
            s_out_ready = lfsr_or[0];
            b_out_ready = lfsr_or[3];

            @(posedge clk);
            // sample acceptance (push happened this edge if in_valid&in_ready)
            if (feed_valid && ~s_accepted && s_in_ready) s_accepted = 1;
            if (feed_valid && ~b_accepted && b_in_ready) b_accepted = 1;
            if (feed_valid && s_accepted && b_accepted) begin
                feed_valid = 0; fed = fed + 1;
            end
            // capture pops (out_valid&out_ready committed on this edge)
            if (s_out_valid && s_out_ready) begin s_q[s_n]=s_out_data; s_n=s_n+1; end
            if (b_out_valid && b_out_ready) begin b_q[b_n]=b_out_data; b_n=b_n+1; end
            lfsr_in = nxt(lfsr_in); lfsr_or = nxt(lfsr_or);
            if (s_n >= NBEATS && b_n >= NBEATS) i = 60000;
        end

        // drain both fully
        s_out_ready=1; b_out_ready=1;
        for (i=0;i<8000;i=i+1) begin
            @(posedge clk);
            if (s_out_valid && s_out_ready && s_n<NBEATS) begin s_q[s_n]=s_out_data; s_n=s_n+1; end
            if (b_out_valid && b_out_ready && b_n<NBEATS) begin b_q[b_n]=b_out_data; b_n=b_n+1; end
        end

        $display("[fifo_equiv] popped skip=%0d bram=%0d (fed=%0d)", s_n, b_n, fed);
        if (s_n != b_n) begin
            $display("[fifo_equiv] FAIL count mismatch s_n=%0d b_n=%0d", s_n, b_n);
            $finish;
        end
        for (i = 0; i < s_n; i = i + 1) begin
            if (s_q[i] !== b_q[i]) begin
                $display("[fifo_equiv] FAIL at beat %0d: skip=%0d bram=%0d", i, s_q[i], b_q[i]);
                $finish;
            end
        end
        // also check the SKIP stream itself is the increasing payload (sanity)
        for (i = 0; i < s_n; i = i + 1)
            if (s_q[i] !== i[WIDTH-1:0]) begin
                $display("[fifo_equiv] FAIL skip stream not increasing at %0d: %0d", i, s_q[i]);
                $finish;
            end
        $display("[fifo_equiv] PASS  %0d beats identical ordering", s_n);
        $finish;
    end
endmodule
