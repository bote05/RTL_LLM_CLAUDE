// Full-chain probe: instantiates nn2rtl_top, drives input, probes
// valid_out count from each spatial chain module via hierarchical
// references. Finds where data dies in the chain.
`timescale 1ns / 1ps

module chain_probe_full_tb;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0;

    reg          s_axis_tvalid = 0;
    reg  [255:0] s_axis_tdata  = 256'd0;
    reg          s_axis_tlast  = 0;
    wire         s_axis_tready;
    reg          m_axis_tready = 1;
    wire         m_axis_tvalid;
    wire [255:0] m_axis_tdata;
    wire         m_axis_tlast;
    // Tie all AXI-Lite off
    reg          s_axil_awvalid = 0;
    wire         s_axil_awready;
    reg  [7:0]   s_axil_awaddr  = 0;
    reg          s_axil_wvalid  = 0;
    wire         s_axil_wready;
    reg  [31:0]  s_axil_wdata   = 0;
    reg  [3:0]   s_axil_wstrb   = 0;
    wire         s_axil_bvalid;
    reg          s_axil_bready  = 0;
    wire [1:0]   s_axil_bresp;
    reg          s_axil_arvalid = 0;
    wire         s_axil_arready;
    reg  [7:0]   s_axil_araddr  = 0;
    wire         s_axil_rvalid;
    reg          s_axil_rready  = 0;
    wire [31:0]  s_axil_rdata;
    wire [1:0]   s_axil_rresp;

    nn2rtl_top dut (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tlast(s_axis_tlast),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tlast(m_axis_tlast),
        .s_axil_awvalid(s_axil_awvalid), .s_axil_awready(s_axil_awready), .s_axil_awaddr(s_axil_awaddr),
        .s_axil_wvalid(s_axil_wvalid),   .s_axil_wready(s_axil_wready),
        .s_axil_wdata(s_axil_wdata),     .s_axil_wstrb(s_axil_wstrb),
        .s_axil_bvalid(s_axil_bvalid),   .s_axil_bready(s_axil_bready), .s_axil_bresp(s_axil_bresp),
        .s_axil_arvalid(s_axil_arvalid), .s_axil_arready(s_axil_arready), .s_axil_araddr(s_axil_araddr),
        .s_axil_rvalid(s_axil_rvalid),   .s_axil_rready(s_axil_rready),
        .s_axil_rdata(s_axil_rdata),     .s_axil_rresp(s_axil_rresp)
    );

    integer cyc = 0;
    integer in_count = 0;
    integer out_count = 0;
    // valid_out counters for each spatial chain stage (sampled by name)
    integer cnt_conv196 = 0, cnt_relu = 0, cnt_maxpool = 0;
    integer cnt_conv198 = 0, cnt_relu_1 = 0, cnt_conv200 = 0, cnt_relu_2 = 0;
    integer cnt_conv202 = 0, cnt_add = 0, cnt_relu_3 = 0;
    integer cnt_conv206 = 0, cnt_relu_4 = 0, cnt_conv208 = 0, cnt_relu_5 = 0;
    integer cnt_conv210 = 0, cnt_add_1 = 0, cnt_relu_6 = 0;
    integer cnt_conv212 = 0, cnt_relu_7 = 0, cnt_conv214 = 0, cnt_relu_8 = 0;
    integer cnt_conv216 = 0, cnt_add_2 = 0, cnt_relu_9 = 0;
    integer cnt_conv218 = 0, cnt_relu_10 = 0, cnt_conv220 = 0, cnt_relu_11 = 0;
    integer cnt_conv222 = 0, cnt_conv244 = 0, cnt_relu_22 = 0;
    // engine loader fill signal
    // wire ldr0_loaded = dut.ldr0_loaded;  // hierarchical reference

    always @(posedge clk) begin
        cyc <= cyc + 1;
        if (cyc == 5) rst_n <= 1;
        if (rst_n && in_count < 50176) s_axis_tvalid <= 1'b1;
        else                           s_axis_tvalid <= 1'b0;
        if (s_axis_tvalid && s_axis_tready && in_count < 50176) in_count <= in_count + 1;
        if (m_axis_tvalid && m_axis_tready) out_count <= out_count + 1;

        // Count valid_out from each module via hierarchical refs
        if (dut.node_conv_196_valid_out) cnt_conv196 <= cnt_conv196 + 1;
        if (dut.node_relu_valid_out)     cnt_relu    <= cnt_relu    + 1;
        if (dut.node_max_pool2d_valid_out) cnt_maxpool <= cnt_maxpool + 1;
        if (dut.node_conv_198_valid_out) cnt_conv198 <= cnt_conv198 + 1;
        if (dut.node_relu_1_valid_out)   cnt_relu_1  <= cnt_relu_1  + 1;
        if (dut.node_conv_200_valid_out) cnt_conv200 <= cnt_conv200 + 1;
        if (dut.node_relu_2_valid_out)   cnt_relu_2  <= cnt_relu_2  + 1;
        if (dut.node_conv_202_valid_out) cnt_conv202 <= cnt_conv202 + 1;
        if (dut.node_add_valid_out)      cnt_add     <= cnt_add     + 1;
        if (dut.node_relu_3_valid_out)   cnt_relu_3  <= cnt_relu_3  + 1;
        if (dut.node_conv_206_valid_out) cnt_conv206 <= cnt_conv206 + 1;
        if (dut.node_relu_4_valid_out)   cnt_relu_4  <= cnt_relu_4  + 1;
        if (dut.node_conv_208_valid_out) cnt_conv208 <= cnt_conv208 + 1;
        if (dut.node_relu_5_valid_out)   cnt_relu_5  <= cnt_relu_5  + 1;
        if (dut.node_conv_210_valid_out) cnt_conv210 <= cnt_conv210 + 1;
        if (dut.node_add_1_valid_out)    cnt_add_1   <= cnt_add_1   + 1;
        if (dut.node_relu_6_valid_out)   cnt_relu_6  <= cnt_relu_6  + 1;
        if (dut.node_conv_212_valid_out) cnt_conv212 <= cnt_conv212 + 1;
        if (dut.node_relu_7_valid_out)   cnt_relu_7  <= cnt_relu_7  + 1;
        if (dut.node_conv_214_valid_out) cnt_conv214 <= cnt_conv214 + 1;
        if (dut.node_relu_8_valid_out)   cnt_relu_8  <= cnt_relu_8  + 1;
        if (dut.node_conv_216_valid_out) cnt_conv216 <= cnt_conv216 + 1;
        if (dut.node_add_2_valid_out)    cnt_add_2   <= cnt_add_2   + 1;
        if (dut.node_relu_9_valid_out)   cnt_relu_9  <= cnt_relu_9  + 1;
        if (dut.node_conv_218_valid_out) cnt_conv218 <= cnt_conv218 + 1;
        if (dut.node_relu_10_valid_out)  cnt_relu_10 <= cnt_relu_10 + 1;
        if (dut.node_conv_220_valid_out) cnt_conv220 <= cnt_conv220 + 1;
        if (dut.node_relu_11_valid_out)  cnt_relu_11 <= cnt_relu_11 + 1;
        if (dut.node_conv_222_valid_out) cnt_conv222 <= cnt_conv222 + 1;
        if (dut.node_relu_22_valid_out)  cnt_relu_22 <= cnt_relu_22 + 1;

        if (cyc > 0 && cyc % 200000 == 0) begin
            $display("[cyc=%0d] in=%0d out=%0d c196=%0d relu=%0d mp=%0d c198=%0d r1=%0d c200=%0d r2=%0d c202=%0d add=%0d r3=%0d c208=%0d r5=%0d c210=%0d a1=%0d r6=%0d c212=%0d r8=%0d c216=%0d a2=%0d c220=%0d c222=%0d r22=%0d ldr0=%b",
                cyc, in_count, out_count, cnt_conv196, cnt_relu, cnt_maxpool,
                cnt_conv198, cnt_relu_1, cnt_conv200, cnt_relu_2,
                cnt_conv202, cnt_add, cnt_relu_3,
                cnt_conv208, cnt_relu_5, cnt_conv210, cnt_add_1, cnt_relu_6,
                cnt_conv212, cnt_relu_8, cnt_conv216, cnt_add_2,
                cnt_conv220, cnt_conv222, cnt_relu_22,
                dut.ldr0_loaded);
        end
        if (cyc > 5000000) begin
            $display("[FINAL] cyc=%0d in=%0d out=%0d c196=%0d r=%0d mp=%0d c198=%0d r1=%0d c200=%0d r2=%0d c202=%0d add=%0d r3=%0d c208=%0d c210=%0d a1=%0d c216=%0d a2=%0d c220=%0d c222=%0d r22=%0d ldr0=%b",
                cyc, in_count, out_count, cnt_conv196, cnt_relu, cnt_maxpool,
                cnt_conv198, cnt_relu_1, cnt_conv200, cnt_relu_2,
                cnt_conv202, cnt_add, cnt_relu_3,
                cnt_conv208, cnt_conv210, cnt_add_1, cnt_conv216, cnt_add_2,
                cnt_conv220, cnt_conv222, cnt_relu_22, dut.ldr0_loaded);
            $finish;
        end
    end
endmodule
