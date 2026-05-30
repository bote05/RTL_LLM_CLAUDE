// Chain head probe: conv_196 -> relu -> maxpool only.
// Drives 50176 input beats, prints per-stage valid_out counts every 100K cycles.
// Discovers which stage stalls.
`timescale 1ns / 1ps

module chain_head_probe_tb;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0;

    reg          valid_in = 0;
    wire         conv196_ready_in;
    reg  [23:0]  data_in = 24'h010203;

    wire         conv196_valid_out;
    wire [255:0] conv196_data_out;

    wire         relu_valid_out;
    wire [255:0] relu_data_out;
    wire         relu_ready_in;

    wire         maxpool_valid_out;
    wire [255:0] maxpool_data_out;
    wire         maxpool_ready_in;

    node_conv_196 u_conv196 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(valid_in),
        .ready_in(conv196_ready_in),
        .data_in(data_in),
        .valid_out(conv196_valid_out),
        .data_out(conv196_data_out)
    );

    node_relu u_relu (
        .clk(clk), .rst_n(rst_n),
        .valid_in(conv196_valid_out),
        .ready_in(relu_ready_in),
        .data_in(conv196_data_out),
        .valid_out(relu_valid_out),
        .data_out(relu_data_out)
    );

    node_max_pool2d u_maxpool (
        .clk(clk), .rst_n(rst_n),
        .valid_in(relu_valid_out),
        .ready_in(maxpool_ready_in),
        .data_in(relu_data_out),
        .valid_out(maxpool_valid_out),
        .data_out(maxpool_data_out)
    );

    integer cycle_count = 0;
    integer input_count = 0;
    integer conv196_count = 0;
    integer relu_count = 0;
    integer maxpool_count = 0;

    always @(posedge clk) begin
        cycle_count <= cycle_count + 1;
        if (cycle_count == 5) rst_n <= 1;
        if (rst_n && input_count < 50176) valid_in <= 1'b1;
        else                              valid_in <= 1'b0;
        if (valid_in && conv196_ready_in && input_count < 50176)
            input_count <= input_count + 1;

        if (conv196_valid_out) conv196_count <= conv196_count + 1;
        if (relu_valid_out)    relu_count    <= relu_count    + 1;
        if (maxpool_valid_out) maxpool_count <= maxpool_count + 1;

        if (cycle_count > 0 && cycle_count % 50000 == 0) begin
            $display("[probe] cyc=%0d in=%0d/50176 conv196=%0d relu=%0d maxpool=%0d c_rdy=%b r_rdy=%b m_rdy=%b",
                     cycle_count, input_count, conv196_count, relu_count, maxpool_count,
                     conv196_ready_in, relu_ready_in, maxpool_ready_in);
        end
        if (cycle_count > 5000000) begin
            $display("[probe] FINAL cyc=%0d in=%0d conv196=%0d relu=%0d maxpool=%0d",
                     cycle_count, input_count, conv196_count, relu_count, maxpool_count);
            $finish;
        end
    end
endmodule
