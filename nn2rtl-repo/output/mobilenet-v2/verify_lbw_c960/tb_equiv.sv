// Equivalence TB: node_conv_896 (line_buf_window TILE_STORAGE=32, deep-narrow tiled per-slot
// storage + mem_busy burst stall) vs node_conv_896_ref (line_buf_window default TILE_STORAGE=0,
// legacy shallow-wide). C=960 depthwise 3x3, stride1, pad1, IH=IW=7 -> 49 output pixels/frame.
//
// Both modules get the SAME deterministic pseudo-random int8 pixel-beat stream (2 beats/pixel:
// lo = ch 0..511, hi = ch 512..959 in low 448*8 bits). Each module's own valid/ready handshake
// drives input consumption; output beats are collected via each module's own valid_out (so the
// per-module latency difference -- the tiled path stalls NT cycles/advance -- is irrelevant).
// PASS = the ordered output-beat streams are bit-identical => the tiled storage is byte-exact.
`timescale 1ns/1ps
module tb_equiv;
    // 1 frame is enough to exercise: top-pad row, real-input row-fill (writes to all 3 slots),
    // the gap=1 row-fill cross-slot rotation, several output rows, and bottom-pad. 49 out pixels.
    localparam integer C        = 960;
    localparam integer IH       = 7;
    localparam integer IW       = 7;
    localparam integer OH       = 7;
    localparam integer OW       = 7;
    localparam integer N_FRAMES = 2;
    localparam integer PIX_PER_FRAME = IH*IW;               // 49 input pixels
    localparam integer BEATS_PER_PIX = 2;
    localparam integer TOTAL_IN  = N_FRAMES*PIX_PER_FRAME*BEATS_PER_PIX;  // 98 input beats
    localparam integer OUT_PIX   = OH*OW;                    // 49 output pixels
    localparam integer TOTAL_OUT = N_FRAMES*OUT_PIX*BEATS_PER_PIX;        // 98 output beats

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #5 clk = ~clk;

    reg          valid_in_a, valid_in_b;
    reg  [4095:0] data_in_a, data_in_b;
    wire          ready_in_a, ready_in_b, valid_out_a, valid_out_b;
    wire [4095:0] data_out_a, data_out_b;

    reg  [4095:0] in_mem [0:TOTAL_IN-1];
    reg  [4095:0] out_a  [0:TOTAL_OUT-1];
    reg  [4095:0] out_b  [0:TOTAL_OUT-1];
    integer in_idx_a, in_idx_b, out_idx_a, out_idx_b;
    integer i, k, bb, mism;
    reg [31:0] lcg;
    reg done;

    // u_new = tiled storage (TILE_STORAGE=32); u_ref = legacy (TILE_STORAGE=0).
    node_conv_896 #(.ENABLE_BACKPRESSURE(0)) u_new (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in_a), .ready_in(ready_in_a),
        .data_in(data_in_a), .out_ready_in(1'b1), .valid_out(valid_out_a), .data_out(data_out_a));
    node_conv_896_ref #(.ENABLE_BACKPRESSURE(0)) u_ref (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in_b), .ready_in(ready_in_b),
        .data_in(data_in_b), .out_ready_in(1'b1), .valid_out(valid_out_b), .data_out(data_out_b));

    // present inputs combinationally (gated on rst_n so nothing accumulates during reset)
    always @(*) begin
        valid_in_a = rst_n && (in_idx_a < TOTAL_IN);
        valid_in_b = rst_n && (in_idx_b < TOTAL_IN);
        data_in_a  = (in_idx_a < TOTAL_IN) ? in_mem[in_idx_a] : {4096{1'b0}};
        data_in_b  = (in_idx_b < TOTAL_IN) ? in_mem[in_idx_b] : {4096{1'b0}};
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
                    if (out_a[i] !== out_b[i]) begin
                        mism = mism + 1;
                        if (mism <= 3) begin
                            // print which channel bytes differ in this beat (lo beat = ch 0..511)
                            for (k = 0; k < 512; k = k + 1)
                                if (out_a[i][k*8 +: 8] !== out_b[i][k*8 +: 8])
                                    $display("DBG beat=%0d ch=%0d (tile=%0d off=%0d) a=%02x b=%02x",
                                             i, k, k/32, k%32, out_a[i][k*8 +: 8], out_b[i][k*8 +: 8]);
                        end
                    end
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
        // Each input beat carries 512 int8 bytes (4096 bits). For lo beats all 512 are real
        // channels; for hi beats only the low 448 bytes are real (the assembler reads
        // data_in[HI_W-1:0] = low 3584 bits), the rest are don't-care. Fill all 512 anyway
        // (identical to both DUTs, so comparison is fair).
        for (k = 0; k < TOTAL_IN; k = k + 1)
            for (bb = 0; bb < 512; bb = bb + 1) begin
                lcg = lcg * 32'd1664525 + 32'd1013904223;
                in_mem[k][bb*8 +: 8] = lcg[23:16];   // pseudo-random int8 byte
            end
        in_idx_a = 0; in_idx_b = 0; out_idx_a = 0; out_idx_b = 0;
        done = 1'b0;
        valid_in_a = 1'b0; valid_in_b = 1'b0; data_in_a = {4096{1'b0}}; data_in_b = {4096{1'b0}};
        rst_n = 1'b0;
        repeat (6) @(posedge clk);
        rst_n = 1'b1;
        // generous timeout: per output pixel ~ OC_PASSES*(MP*K+...) ~ 240*~15 ~ 3600 cyc;
        // 49 pixels ~ 180k cyc + tiled NT=30 stalls per advance. 80M ns @ 10ns/cyc = 8M cyc.
        #80000000;
        $display("EQUIV_RESULT TIMEOUT out_a=%0d/%0d out_b=%0d/%0d in_a=%0d in_b=%0d",
                 out_idx_a, TOTAL_OUT, out_idx_b, TOTAL_OUT, in_idx_a, in_idx_b);
        $finish;
    end
endmodule
