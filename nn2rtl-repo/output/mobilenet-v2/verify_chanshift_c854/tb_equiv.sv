// Equivalence TB: node_conv_854 (line_buf_window CHAN_SHIFT=1 rotation banks + TILE_STORAGE=32)
// vs node_conv_854_ref (line_buf_window CHAN_SHIFT=0 legacy per-tap channel-select mux).
// C=384 depthwise 3x3, STRIDE 1, pad1, IH=IW=14 -> OH=OW=14 = 196 out px/frame. MP=16.
//
// SINGLE BEAT per pixel (C=384 <= 512 -> one full C*8=3072b pixel per beat in AND out).
// MP=16 means the per-channel issuing strobe (chan_issue) walks current_global_oc 0..C-1 over a
// DIFFERENT cadence than the MP=4 nodes; this TB proves the CHAN_SHIFT rotation stays a clean +1
// csel walk under MP=16. Both modules get the SAME deterministic pseudo-random int8 pixel stream;
// each drives its own valid/ready handshake; output beats are collected via each module's own
// valid_out (per-module latency difference is irrelevant). PASS = ordered output streams
// bit-identical => CHAN_SHIFT rotation is byte-exact at MP=16.
`timescale 1ns/1ps
module tb_equiv;
    localparam integer C        = 384;
    localparam integer BUS      = C*8;                      // 3072
    localparam integer IH       = 14;
    localparam integer IW       = 14;
    localparam integer OH       = 14;
    localparam integer OW       = 14;
    localparam integer N_FRAMES = 2;
    localparam integer PIX_PER_FRAME = IH*IW;               // 196 input pixels
    localparam integer TOTAL_IN  = N_FRAMES*PIX_PER_FRAME;  // 392 input beats
    localparam integer OUT_PIX   = OH*OW;                   // 196 output pixels
    localparam integer TOTAL_OUT = N_FRAMES*OUT_PIX;        // 392 output beats

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #5 clk = ~clk;

    reg           valid_in_a, valid_in_b;
    reg  [BUS-1:0] data_in_a, data_in_b;
    wire          ready_in_a, ready_in_b, valid_out_a, valid_out_b;
    wire [BUS-1:0] data_out_a, data_out_b;

    reg  [BUS-1:0] in_mem [0:TOTAL_IN-1];
    reg  [BUS-1:0] out_a  [0:TOTAL_OUT-1];
    reg  [BUS-1:0] out_b  [0:TOTAL_OUT-1];
    integer in_idx_a, in_idx_b, out_idx_a, out_idx_b;
    integer i, k, bb, mism;
    reg [31:0] lcg;
    reg done;

    // u_new = CHAN_SHIFT=1 rotation banks; u_ref = CHAN_SHIFT=0 legacy mux.
    node_conv_854 #(.ENABLE_BACKPRESSURE(0)) u_new (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in_a), .ready_in(ready_in_a),
        .data_in(data_in_a), .out_ready_in(1'b1), .valid_out(valid_out_a), .data_out(data_out_a));
    node_conv_854_ref #(.ENABLE_BACKPRESSURE(0)) u_ref (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in_b), .ready_in(ready_in_b),
        .data_in(data_in_b), .out_ready_in(1'b1), .valid_out(valid_out_b), .data_out(data_out_b));

    always @(*) begin
        valid_in_a = rst_n && (in_idx_a < TOTAL_IN);
        valid_in_b = rst_n && (in_idx_b < TOTAL_IN);
        data_in_a  = (in_idx_a < TOTAL_IN) ? in_mem[in_idx_a] : {BUS{1'b0}};
        data_in_b  = (in_idx_b < TOTAL_IN) ? in_mem[in_idx_b] : {BUS{1'b0}};
    end

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
                            for (k = 0; k < C; k = k + 1)
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
        for (k = 0; k < TOTAL_IN; k = k + 1)
            for (bb = 0; bb < C; bb = bb + 1) begin
                lcg = lcg * 32'd1664525 + 32'd1013904223;
                in_mem[k][bb*8 +: 8] = lcg[23:16];   // pseudo-random int8 byte
            end
        in_idx_a = 0; in_idx_b = 0; out_idx_a = 0; out_idx_b = 0;
        done = 1'b0;
        valid_in_a = 1'b0; valid_in_b = 1'b0; data_in_a = {BUS{1'b0}}; data_in_b = {BUS{1'b0}};
        rst_n = 1'b0;
        repeat (6) @(posedge clk);
        rst_n = 1'b1;
        #80000000;
        $display("EQUIV_RESULT TIMEOUT out_a=%0d/%0d out_b=%0d/%0d in_a=%0d in_b=%0d",
                 out_idx_a, TOTAL_OUT, out_idx_b, TOTAL_OUT, in_idx_a, in_idx_b);
        $finish;
    end
endmodule
