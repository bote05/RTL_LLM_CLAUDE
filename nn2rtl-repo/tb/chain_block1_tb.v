// First-block chain probe: conv_196 -> relu -> maxpool -> conv_198 -> relu_1 -> conv_200 -> relu_2 -> conv_202.
// This is ResNet-50's stem + first bottleneck block start.
// Drives 50176 input beats, prints per-stage valid_out counts, finds where data dies.
`timescale 1ns / 1ps

module chain_block1_tb;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0;

    reg          valid_in = 0;
    wire         c196_ready_in;
    reg  [23:0]  data_in = 24'h010203;

    wire         c196_valid_out;
    wire [255:0] c196_data_out;

    wire         relu_valid_out;
    wire [255:0] relu_data_out;
    wire         relu_ready_in;

    wire         mp_valid_out;
    wire [255:0] mp_data_out;
    wire         mp_ready_in;

    wire         c198_valid_out;
    wire [255:0] c198_data_out;
    wire         c198_ready_in;

    wire         r1_valid_out;
    wire [255:0] r1_data_out;
    wire         r1_ready_in;

    wire         c200_valid_out;
    wire [255:0] c200_data_out;
    wire         c200_ready_in;

    wire         r2_valid_out;
    wire [255:0] r2_data_out;
    wire         r2_ready_in;

    wire         c202_valid_out;
    wire [255:0] c202_data_out;
    wire         c202_ready_in;

    node_conv_196 u_c196 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(valid_in), .ready_in(c196_ready_in), .data_in(data_in),
        .valid_out(c196_valid_out), .data_out(c196_data_out)
    );
    node_relu u_relu (
        .clk(clk), .rst_n(rst_n),
        .valid_in(c196_valid_out), .ready_in(relu_ready_in), .data_in(c196_data_out),
        .valid_out(relu_valid_out), .data_out(relu_data_out)
    );
    node_max_pool2d u_mp (
        .clk(clk), .rst_n(rst_n),
        .valid_in(relu_valid_out), .ready_in(mp_ready_in), .data_in(relu_data_out),
        .valid_out(mp_valid_out), .data_out(mp_data_out)
    );
    node_conv_198 u_c198 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(mp_valid_out), .ready_in(c198_ready_in), .data_in(mp_data_out),
        .valid_out(c198_valid_out), .data_out(c198_data_out)
    );
    node_relu_1 u_r1 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(c198_valid_out), .ready_in(r1_ready_in), .data_in(c198_data_out),
        .valid_out(r1_valid_out), .data_out(r1_data_out)
    );
    node_conv_200 u_c200 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(r1_valid_out), .ready_in(c200_ready_in), .data_in(r1_data_out),
        .valid_out(c200_valid_out), .data_out(c200_data_out)
    );
    node_relu_2 u_r2 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(c200_valid_out), .ready_in(r2_ready_in), .data_in(c200_data_out),
        .valid_out(r2_valid_out), .data_out(r2_data_out)
    );
    node_conv_202 u_c202 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(r2_valid_out), .ready_in(c202_ready_in), .data_in(r2_data_out),
        .valid_out(c202_valid_out), .data_out(c202_data_out)
    );

    integer cyc = 0;
    integer in_count = 0;
    integer c196_cnt = 0, relu_cnt = 0, mp_cnt = 0;
    integer c198_cnt = 0, r1_cnt = 0, c200_cnt = 0, r2_cnt = 0, c202_cnt = 0;

    always @(posedge clk) begin
        cyc <= cyc + 1;
        if (cyc == 5) rst_n <= 1;
        if (rst_n && in_count < 50176) valid_in <= 1'b1;
        else                           valid_in <= 1'b0;
        if (valid_in && c196_ready_in && in_count < 50176) in_count <= in_count + 1;

        if (c196_valid_out) c196_cnt <= c196_cnt + 1;
        if (relu_valid_out) relu_cnt <= relu_cnt + 1;
        if (mp_valid_out)   mp_cnt   <= mp_cnt   + 1;
        if (c198_valid_out) c198_cnt <= c198_cnt + 1;
        if (r1_valid_out)   r1_cnt   <= r1_cnt   + 1;
        if (c200_valid_out) c200_cnt <= c200_cnt + 1;
        if (r2_valid_out)   r2_cnt   <= r2_cnt   + 1;
        if (c202_valid_out) c202_cnt <= c202_cnt + 1;

        if (cyc > 0 && cyc % 100000 == 0) begin
            $display("[%0d] in=%0d c196=%0d relu=%0d mp=%0d c198=%0d r1=%0d c200=%0d r2=%0d c202=%0d rdy:c196=%b r1=%b c200=%b c202=%b",
                cyc, in_count, c196_cnt, relu_cnt, mp_cnt, c198_cnt, r1_cnt, c200_cnt, r2_cnt, c202_cnt,
                c196_ready_in, r1_ready_in, c200_ready_in, c202_ready_in);
        end
        if (cyc > 5000000) begin
            $display("[FINAL] cyc=%0d in=%0d c196=%0d relu=%0d mp=%0d c198=%0d r1=%0d c200=%0d r2=%0d c202=%0d",
                cyc, in_count, c196_cnt, relu_cnt, mp_cnt, c198_cnt, r1_cnt, c200_cnt, r2_cnt, c202_cnt);
            $finish;
        end
    end
endmodule
