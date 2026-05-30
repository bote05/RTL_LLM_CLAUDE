// FAST variant of chain_iv_conv200_tb.v: identical real-chain iverilog test, but
// captures only the FIRST NCAP conv_200 beats (a few output rows) instead of the
// whole frame. The conv_200 deficit is DETERMINISTIC + spatial, so it shows in the
// first rows -> same artifact-vs-real verdict in ~1/15th the runtime. Writes to a
// SEPARATE capture file + is a SEPARATE module + compiles to a SEPARATE vvp, so it
// never touches the long run's files (chain_iv_c200.vvp / conv200_iverilog_cap.hex).
`timescale 1ns/1ps
module chain_iv_conv200_fast_tb;
    localparam integer NIN  = 50176;   // conv_196 goldin beats (frame 0)
    localparam integer NCAP = 384;     // FAST: ~3 output rows (enough for a deterministic deficit)

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
    reg [255:0] cap [0:NCAP-1];
    integer in_idx = 0, cap_idx = 0, cyc = 0, i, fd;

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
        if (rst_n && dut.node_conv_200_valid_out && dut.skid_node_relu_2_ready
                  && dut.spatial_run && cap_idx < NCAP) begin
            cap[cap_idx] <= dut.node_conv_200_data_out;
            cap_idx <= cap_idx + 1;
        end
        if (cyc > 0 && cyc % 50000 == 0)
            $display("[FAST cyc=%0d] in=%0d/%0d conv200_cap=%0d/%0d", cyc, in_idx, NIN, cap_idx, NCAP);
        if (cap_idx >= NCAP || cyc > 500000) begin
            $display("[FAST DONE] cyc=%0d conv200_cap=%0d/%0d -> dumping", cyc, cap_idx, NCAP);
            fd = $fopen("output/reports_integrated/conv200_iverilog_cap_FAST.hex","w");
            for (i = 0; i < cap_idx; i = i + 1) $fwrite(fd, "%064x\n", cap[i]);
            $fclose(fd);
            $finish;
        end
    end
endmodule
