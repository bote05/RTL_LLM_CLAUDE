// DECISIVE tap-method test. Captures conv_200's output TWO ways in ONE run:
//   cap_mod  = node_conv_200_data_out  @ (valid_out && skid_relu_2_ready && spatial_run)
//              -- the OLD probe (combinational module output at skid-INPUT accept)
//   cap_skid = skid_node_relu_2_data    @ (skid_relu_2_valid && relu_2_ready_in && spatial_run)
//              -- the REGISTERED value actually DELIVERED to relu_2 (clean).
// The skid is a pure FIFO (no compute) so cap_skid is conv_200's golden output, in order.
// If cap_mod != cap_skid, the old tap was skewed (capturing wrong-cycle data) => the 94%
// was a TAP artifact, and conv_200 is fine. Compared offline to node_conv_200.goldout.
`timescale 1ns/1ps
module chain_iv_conv200_skidtap_tb;
    localparam integer NIN  = 50176;
    localparam integer NCAP = 384;

    reg clk = 0; always #5 clk = ~clk;
    reg rst_n = 0;
    reg          s_axis_tvalid = 0;
    reg  [255:0] s_axis_tdata  = 256'd0;
    reg          s_axis_tlast  = 0;
    wire         s_axis_tready;
    reg          m_axis_tready = 1;
    wire         m_axis_tvalid; wire [255:0] m_axis_tdata; wire m_axis_tlast;
    reg s_axil_awvalid=0; wire s_axil_awready; reg [7:0] s_axil_awaddr=0;
    reg s_axil_wvalid=0; wire s_axil_wready; reg [31:0] s_axil_wdata=0; reg [3:0] s_axil_wstrb=0;
    wire s_axil_bvalid; reg s_axil_bready=0; wire [1:0] s_axil_bresp;
    reg s_axil_arvalid=0; wire s_axil_arready; reg [7:0] s_axil_araddr=0;
    wire s_axil_rvalid; reg s_axil_rready=0; wire [31:0] s_axil_rdata; wire [1:0] s_axil_rresp;

    nn2rtl_top dut (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tvalid(s_axis_tvalid), .s_axis_tready(s_axis_tready),
        .s_axis_tdata(s_axis_tdata), .s_axis_tlast(s_axis_tlast),
        .m_axis_tvalid(m_axis_tvalid), .m_axis_tready(m_axis_tready),
        .m_axis_tdata(m_axis_tdata), .m_axis_tlast(m_axis_tlast),
        .s_axil_awvalid(s_axil_awvalid), .s_axil_awready(s_axil_awready), .s_axil_awaddr(s_axil_awaddr),
        .s_axil_wvalid(s_axil_wvalid), .s_axil_wready(s_axil_wready),
        .s_axil_wdata(s_axil_wdata), .s_axil_wstrb(s_axil_wstrb),
        .s_axil_bvalid(s_axil_bvalid), .s_axil_bready(s_axil_bready), .s_axil_bresp(s_axil_bresp),
        .s_axil_arvalid(s_axil_arvalid), .s_axil_arready(s_axil_arready), .s_axil_araddr(s_axil_araddr),
        .s_axil_rvalid(s_axil_rvalid), .s_axil_rready(s_axil_rready),
        .s_axil_rdata(s_axil_rdata), .s_axil_rresp(s_axil_rresp)
    );

    reg [255:0] gin [0:NIN-1];
    reg [255:0] cap_mod  [0:NCAP-1];
    reg [255:0] cap_skid [0:NCAP-1];
    integer in_idx=0, mi=0, si=0, cyc=0, i, fd;
    initial $readmemh("output/conv196_saxis_f0.hex", gin);

    always @(posedge clk) begin
        cyc <= cyc + 1;
        if (cyc == 5) rst_n <= 1;
        if (rst_n) begin
            s_axis_tvalid <= (in_idx < NIN);
            s_axis_tdata  <= gin[in_idx];
            if (s_axis_tvalid && s_axis_tready && in_idx < NIN) in_idx <= in_idx + 1;
        end
    end

    always @(posedge clk) begin
        // OLD tap: combinational module output at skid-INPUT accept
        if (rst_n && dut.node_conv_200_valid_out && dut.skid_node_relu_2_ready
                  && dut.spatial_run && mi < NCAP) begin
            cap_mod[mi] <= dut.node_conv_200_data_out; mi <= mi + 1;
        end
        // NEW tap: registered skid output actually delivered to relu_2
        if (rst_n && dut.skid_node_relu_2_valid && dut.node_relu_2_ready_in
                  && dut.spatial_run && si < NCAP) begin
            cap_skid[si] <= dut.skid_node_relu_2_data; si <= si + 1;
        end
        if (cyc>0 && cyc%50000==0) $display("[skidtap cyc=%0d] mod=%0d skid=%0d /%0d", cyc, mi, si, NCAP);
        if ((mi>=NCAP && si>=NCAP) || cyc>500000) begin
            $display("[skidtap DONE] cyc=%0d mod=%0d skid=%0d", cyc, mi, si);
            fd=$fopen("output/reports_integrated/conv200_cap_MOD.hex","w");
            for (i=0;i<mi;i=i+1) $fwrite(fd,"%064x\n",cap_mod[i]); $fclose(fd);
            fd=$fopen("output/reports_integrated/conv200_cap_SKID.hex","w");
            for (i=0;i<si;i=i+1) $fwrite(fd,"%064x\n",cap_skid[i]); $fclose(fd);
            $finish;
        end
    end
endmodule
