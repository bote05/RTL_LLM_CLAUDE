// Equivalence TB: new (BRAM-fix) node_mean vs backed-up known-good node_mean_ref.
// Drives BOTH with the SAME deterministic int8 input stream (covers accumulate / scale /
// round-half on +/- / clamp), collects each output-beat stream via its own valid/ready
// handshake (so the +158-cycle latency difference is irrelevant), and compares the ordered
// output beats. PASS = bit-identical => the restructure is byte-exact vs the prior version.
`timescale 1ns/1ps
module tb_equiv;
    localparam integer N_FRAMES = 3;
    localparam integer HW       = 49;
    localparam integer N_TILES  = 5;
    localparam integer TOTAL_IN  = N_FRAMES*HW*N_TILES;   // 735 input beats
    localparam integer TOTAL_OUT = N_FRAMES*N_TILES;      // 15 output beats

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #5 clk = ~clk;

    reg          valid_in_a, valid_in_b;
    reg  [2047:0] data_in_a, data_in_b;
    wire         ready_in_a, ready_in_b, valid_out_a, valid_out_b;
    wire [2047:0] data_out_a, data_out_b;

    reg  [2047:0] in_mem [0:TOTAL_IN-1];
    reg  [2047:0] out_a  [0:TOTAL_OUT-1];
    reg  [2047:0] out_b  [0:TOTAL_OUT-1];
    integer in_idx_a, in_idx_b, out_idx_a, out_idx_b;
    integer i, k, b, mism;
    reg [31:0] lcg;
    reg done;

    node_mean #(.ENABLE_BACKPRESSURE(0)) u_new (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in_a), .ready_in(ready_in_a),
        .data_in(data_in_a), .out_ready_in(1'b1), .valid_out(valid_out_a), .data_out(data_out_a));
    node_mean_ref #(.ENABLE_BACKPRESSURE(0)) u_ref (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in_b), .ready_in(ready_in_b),
        .data_in(data_in_b), .out_ready_in(1'b1), .valid_out(valid_out_b), .data_out(data_out_b));

    // present inputs combinationally (gated on rst_n so nothing accumulates during reset)
    always @(*) begin
        valid_in_a = rst_n && (in_idx_a < TOTAL_IN);
        valid_in_b = rst_n && (in_idx_b < TOTAL_IN);
        data_in_a  = (in_idx_a < TOTAL_IN) ? in_mem[in_idx_a] : {2048{1'b0}};
        data_in_b  = (in_idx_b < TOTAL_IN) ? in_mem[in_idx_b] : {2048{1'b0}};
    end

    // advance input on accept; collect output on valid
    always @(posedge clk) begin
        if (rst_n) begin
            if (valid_in_a && ready_in_a) in_idx_a <= in_idx_a + 1;
            if (valid_in_b && ready_in_b) in_idx_b <= in_idx_b + 1;
            if (valid_out_a && (out_idx_a < TOTAL_OUT)) begin out_a[out_idx_a] <= data_out_a; out_idx_a <= out_idx_a + 1; end
            if (valid_out_b && (out_idx_b < TOTAL_OUT)) begin out_b[out_idx_b] <= data_out_b; out_idx_b <= out_idx_b + 1; end
            if (!done && (out_idx_a == TOTAL_OUT) && (out_idx_b == TOTAL_OUT)) begin
                done <= 1'b1;
                mism = 0;
                for (i = 0; i < TOTAL_OUT; i = i + 1)
                    if (out_a[i] !== out_b[i]) mism = mism + 1;
                if (mism == 0)
                    $display("EQUIV_RESULT PASS out_beats=%0d mismatch=0", TOTAL_OUT);
                else
                    $display("EQUIV_RESULT FAIL mismatch=%0d / %0d beats", mism, TOTAL_OUT);
                $finish;
            end
        end
    end

    initial begin
        lcg = 32'h1234_5678;
        for (k = 0; k < TOTAL_IN; k = k + 1)
            for (b = 0; b < 256; b = b + 1) begin
                lcg = lcg * 32'd1664525 + 32'd1013904223;
                in_mem[k][b*8 +: 8] = lcg[15:8];   // pseudo-random int8 byte
            end
        in_idx_a = 0; in_idx_b = 0; out_idx_a = 0; out_idx_b = 0;
        done = 1'b0;
        valid_in_a = 1'b0; valid_in_b = 1'b0; data_in_a = {2048{1'b0}}; data_in_b = {2048{1'b0}};
        rst_n = 1'b0;
        repeat (6) @(posedge clk);
        rst_n = 1'b1;
        #4000000;  // 400k-cycle timeout
        $display("EQUIV_RESULT TIMEOUT out_a=%0d/%0d out_b=%0d/%0d in_a=%0d in_b=%0d",
                 out_idx_a, TOTAL_OUT, out_idx_b, TOTAL_OUT, in_idx_a, in_idx_b);
        $finish;
    end
endmodule
