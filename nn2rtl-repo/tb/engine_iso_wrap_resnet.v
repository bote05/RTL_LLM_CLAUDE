`timescale 1ns/1ps
// engine_iso_wrap_resnet.v -- ResNet-50 engine-isolation wrapper ([KPAR4-RN]).
//
// Sibling of tb/engine_iso_wrap_mbv2.v with the ResNet INT3 geometry.
// Instantiates the REAL shared_engine + REAL memory shapes wired EXACTLY as
// output/rtl/nn2rtl_top.v wires them for the engine path:
//   * 8 uram_weight_bank-shaped ROMs (INT3: 96-bit line, ALL bits real ->
//     768-bit weight bus; READ_LATENCY=2 -- the deployment BRAM/URAM read)
//   * bias_mem / scale_mem (8192-bit = 256 INT32 per oc_pass, READ_LATENCY=1)
//   * act mem (2048-bit = 256 INT8 channels per pixel word, READ_LATENCY=1)
//     with a TB write/read port for preload + result capture.
//
// ResNet lane k = 3-bit slot: bank(k/32) bits [(k%32)*3 +: 3]. The engine is
// parameterised WGT_W=3, URAM_DATA_W=768 (legacy) -- mirroring the deployed
// top's ENGINE_WGT_W=3 / ENGINE_WBUS_W. Define KPAR4 for the K_PAR=4 build:
// URAM_DATA_W=3072, repacked _kp4 banks (384-bit tap-major lines, group
// addressing handled INSIDE the engine via the old>>2 export).
//
// PORT LIST IS IDENTICAL to engine_iso_wrap_mbv2.v: the shared C++ driver
// (tb/engine_iso_wrap_mbv2_tb.cpp) is reused via verilator
//   --top-module engine_iso_wrap_resnet --prefix Vengine_iso_wrap_mbv2
// `WLAT macro: default 2-cycle URAM read; define WLAT1 for the (FALSE-
// confidence) 1-cycle read.

module engine_iso_wrap_resnet (
    input  wire         clk,
    input  wire         rst_n,
    // AXI-Lite
    input  wire         s_axil_awvalid, output wire s_axil_awready,
    input  wire [7:0]   s_axil_awaddr,
    input  wire         s_axil_wvalid,  output wire s_axil_wready,
    input  wire [31:0]  s_axil_wdata,   input  wire [3:0] s_axil_wstrb,
    output wire         s_axil_bvalid,  input  wire s_axil_bready,
    output wire [1:0]   s_axil_bresp,
    input  wire         s_axil_arvalid, output wire s_axil_arready,
    input  wire [7:0]   s_axil_araddr,
    output wire         s_axil_rvalid,  input  wire s_axil_rready,
    output wire [31:0]  s_axil_rdata,   output wire [1:0] s_axil_rresp,
    // handshake
    input  wire         engine_start,
    output wire         engine_busy,
    output wire         engine_done,
    // TB activation write port (preload + result readback)
    input  wire         tb_act_wr_en,
    input  wire [15:0]  tb_act_wr_addr,
    input  wire [2047:0] tb_act_wr_data,
    input  wire [15:0]  tb_act_rd_addr,
    output wire [2047:0] tb_act_rd_data
);
    // ---- engine <-> memory wires ----
    wire [15:0]   eng_act_in_rd_addr;
    wire          eng_act_in_rd_en;
    wire [2047:0] eng_act_in_rd_data;
    wire [15:0]   eng_act_out_wr_addr;
    wire          eng_act_out_wr_en;
    wire [2047:0] eng_act_out_wr_data;
    wire [21:0]   eng_weight_rd_addr;
    wire          eng_weight_rd_en;
`ifdef KPAR8
    wire [6143:0] eng_weight_rd_data;     // [KPAR8-RN] 8 tap-major 768b words
`elsif KPAR4
    wire [3071:0] eng_weight_rd_data;     // [KPAR4-RN] 4 tap-major 768b words
`else
    wire [767:0]  eng_weight_rd_data;     // 768b = 256 INT3 weights (resnet)
`endif
    wire [21:0]   eng_bias_rd_addr;
    wire          eng_bias_rd_en;
    wire [8191:0] eng_bias_rd_data;
    wire [21:0]   eng_scale_rd_addr;
    wire          eng_scale_rd_en;
    wire [8191:0] eng_scale_rd_data;

    // ---- 8 weight banks (INT3 96-bit lines, all bits real), 2-cycle read ----
    // Mirrors output/rtl/nn2rtl_top.v: weight bus = concat of each bank's
    // ENGINE_LANE_B=96 real bits, bank0 lowest.
`ifdef KPAR8
    // [KPAR8-RN] repacked 8-taps-per-line banks; the ENGINE exports the
    // GROUP address (old>>3), so the bank addr is just the low bits.
    wire [13:0]  wbank_addr = eng_weight_rd_addr[13:0];
    wire [767:0] b0,b1,b2,b3,b4,b5,b6,b7;
    genvar kp_tap;
    generate for (kp_tap = 0; kp_tap < 8; kp_tap = kp_tap + 1) begin : g_kpar_wbus
        assign eng_weight_rd_data[kp_tap*768 +: 768] = {
            b7[kp_tap*96 +: 96], b6[kp_tap*96 +: 96],
            b5[kp_tap*96 +: 96], b4[kp_tap*96 +: 96],
            b3[kp_tap*96 +: 96], b2[kp_tap*96 +: 96],
            b1[kp_tap*96 +: 96], b0[kp_tap*96 +: 96]};
    end endgenerate
    iso_uram_bank_rn #(.WORD_W(768), .MEM_INIT_FILE("output/weights/uram_weights_bank0_kp8.mem")) u0(.clk(clk),.rd_addr({3'b0,wbank_addr}),.rd_data(b0),.rd_en(eng_weight_rd_en));
    iso_uram_bank_rn #(.WORD_W(768), .MEM_INIT_FILE("output/weights/uram_weights_bank1_kp8.mem")) u1(.clk(clk),.rd_addr({3'b0,wbank_addr}),.rd_data(b1),.rd_en(eng_weight_rd_en));
    iso_uram_bank_rn #(.WORD_W(768), .MEM_INIT_FILE("output/weights/uram_weights_bank2_kp8.mem")) u2(.clk(clk),.rd_addr({3'b0,wbank_addr}),.rd_data(b2),.rd_en(eng_weight_rd_en));
    iso_uram_bank_rn #(.WORD_W(768), .MEM_INIT_FILE("output/weights/uram_weights_bank3_kp8.mem")) u3(.clk(clk),.rd_addr({3'b0,wbank_addr}),.rd_data(b3),.rd_en(eng_weight_rd_en));
    iso_uram_bank_rn #(.WORD_W(768), .MEM_INIT_FILE("output/weights/uram_weights_bank4_kp8.mem")) u4(.clk(clk),.rd_addr({3'b0,wbank_addr}),.rd_data(b4),.rd_en(eng_weight_rd_en));
    iso_uram_bank_rn #(.WORD_W(768), .MEM_INIT_FILE("output/weights/uram_weights_bank5_kp8.mem")) u5(.clk(clk),.rd_addr({3'b0,wbank_addr}),.rd_data(b5),.rd_en(eng_weight_rd_en));
    iso_uram_bank_rn #(.WORD_W(768), .MEM_INIT_FILE("output/weights/uram_weights_bank6_kp8.mem")) u6(.clk(clk),.rd_addr({3'b0,wbank_addr}),.rd_data(b6),.rd_en(eng_weight_rd_en));
    iso_uram_bank_rn #(.WORD_W(768), .MEM_INIT_FILE("output/weights/uram_weights_bank7_kp8.mem")) u7(.clk(clk),.rd_addr({3'b0,wbank_addr}),.rd_data(b7),.rd_en(eng_weight_rd_en));
`elsif KPAR4
    // [KPAR4-RN] repacked 4-taps-per-line banks; the ENGINE exports the
    // GROUP address (old>>2), so the bank addr is just the low bits.
    wire [14:0]  wbank_addr = eng_weight_rd_addr[14:0];
    wire [383:0] b0,b1,b2,b3,b4,b5,b6,b7;
    genvar kp_tap;
    generate for (kp_tap = 0; kp_tap < 4; kp_tap = kp_tap + 1) begin : g_kpar_wbus
        assign eng_weight_rd_data[kp_tap*768 +: 768] = {
            b7[kp_tap*96 +: 96], b6[kp_tap*96 +: 96],
            b5[kp_tap*96 +: 96], b4[kp_tap*96 +: 96],
            b3[kp_tap*96 +: 96], b2[kp_tap*96 +: 96],
            b1[kp_tap*96 +: 96], b0[kp_tap*96 +: 96]};
    end endgenerate
    iso_uram_bank_rn #(.WORD_W(384), .MEM_INIT_FILE("output/weights/uram_weights_bank0_kp4.mem")) u0(.clk(clk),.rd_addr({2'b0,wbank_addr}),.rd_data(b0),.rd_en(eng_weight_rd_en));
    iso_uram_bank_rn #(.WORD_W(384), .MEM_INIT_FILE("output/weights/uram_weights_bank1_kp4.mem")) u1(.clk(clk),.rd_addr({2'b0,wbank_addr}),.rd_data(b1),.rd_en(eng_weight_rd_en));
    iso_uram_bank_rn #(.WORD_W(384), .MEM_INIT_FILE("output/weights/uram_weights_bank2_kp4.mem")) u2(.clk(clk),.rd_addr({2'b0,wbank_addr}),.rd_data(b2),.rd_en(eng_weight_rd_en));
    iso_uram_bank_rn #(.WORD_W(384), .MEM_INIT_FILE("output/weights/uram_weights_bank3_kp4.mem")) u3(.clk(clk),.rd_addr({2'b0,wbank_addr}),.rd_data(b3),.rd_en(eng_weight_rd_en));
    iso_uram_bank_rn #(.WORD_W(384), .MEM_INIT_FILE("output/weights/uram_weights_bank4_kp4.mem")) u4(.clk(clk),.rd_addr({2'b0,wbank_addr}),.rd_data(b4),.rd_en(eng_weight_rd_en));
    iso_uram_bank_rn #(.WORD_W(384), .MEM_INIT_FILE("output/weights/uram_weights_bank5_kp4.mem")) u5(.clk(clk),.rd_addr({2'b0,wbank_addr}),.rd_data(b5),.rd_en(eng_weight_rd_en));
    iso_uram_bank_rn #(.WORD_W(384), .MEM_INIT_FILE("output/weights/uram_weights_bank6_kp4.mem")) u6(.clk(clk),.rd_addr({2'b0,wbank_addr}),.rd_data(b6),.rd_en(eng_weight_rd_en));
    iso_uram_bank_rn #(.WORD_W(384), .MEM_INIT_FILE("output/weights/uram_weights_bank7_kp4.mem")) u7(.clk(clk),.rd_addr({2'b0,wbank_addr}),.rd_data(b7),.rd_en(eng_weight_rd_en));
`else
    wire [16:0] wbank_addr = eng_weight_rd_addr[16:0];
    wire [95:0] b0,b1,b2,b3,b4,b5,b6,b7;
    assign eng_weight_rd_data = {b7,b6,b5,b4,b3,b2,b1,b0};
    iso_uram_bank_rn #(.MEM_INIT_FILE("output/weights/uram_weights_bank0.mem")) u0(.clk(clk),.rd_addr(wbank_addr),.rd_data(b0),.rd_en(eng_weight_rd_en));
    iso_uram_bank_rn #(.MEM_INIT_FILE("output/weights/uram_weights_bank1.mem")) u1(.clk(clk),.rd_addr(wbank_addr),.rd_data(b1),.rd_en(eng_weight_rd_en));
    iso_uram_bank_rn #(.MEM_INIT_FILE("output/weights/uram_weights_bank2.mem")) u2(.clk(clk),.rd_addr(wbank_addr),.rd_data(b2),.rd_en(eng_weight_rd_en));
    iso_uram_bank_rn #(.MEM_INIT_FILE("output/weights/uram_weights_bank3.mem")) u3(.clk(clk),.rd_addr(wbank_addr),.rd_data(b3),.rd_en(eng_weight_rd_en));
    iso_uram_bank_rn #(.MEM_INIT_FILE("output/weights/uram_weights_bank4.mem")) u4(.clk(clk),.rd_addr(wbank_addr),.rd_data(b4),.rd_en(eng_weight_rd_en));
    iso_uram_bank_rn #(.MEM_INIT_FILE("output/weights/uram_weights_bank5.mem")) u5(.clk(clk),.rd_addr(wbank_addr),.rd_data(b5),.rd_en(eng_weight_rd_en));
    iso_uram_bank_rn #(.MEM_INIT_FILE("output/weights/uram_weights_bank6.mem")) u6(.clk(clk),.rd_addr(wbank_addr),.rd_data(b6),.rd_en(eng_weight_rd_en));
    iso_uram_bank_rn #(.MEM_INIT_FILE("output/weights/uram_weights_bank7.mem")) u7(.clk(clk),.rd_addr(wbank_addr),.rd_data(b7),.rd_en(eng_weight_rd_en));
`endif

    // ---- bias + scale ROMs (8192-bit, 1-cycle) ----
    iso_bias_mem_rn #(.MEM_INIT_FILE("output/weights/bias.mem"))
      u_bias(.clk(clk),.rd_addr(eng_bias_rd_addr[7:0]),.rd_data(eng_bias_rd_data),.rd_en(eng_bias_rd_en));
    iso_bias_mem_rn #(.MEM_INIT_FILE("output/weights/scale.mem"))
      u_scale(.clk(clk),.rd_addr(eng_scale_rd_addr[7:0]),.rd_data(eng_scale_rd_data),.rd_en(eng_scale_rd_en));

    // ---- activation unified mem (2048-bit, 1-cycle), engine + TB write mux ----
    wire        act_wr_en   = eng_act_out_wr_en | tb_act_wr_en;
    wire [15:0] act_wr_addr = eng_act_out_wr_en ? eng_act_out_wr_addr : tb_act_wr_addr;
    wire [2047:0] act_wr_data = eng_act_out_wr_en ? eng_act_out_wr_data : tb_act_wr_data;
    iso_act_mem_rn u_act(.clk(clk),
        .rd_addr(eng_act_in_rd_addr), .rd_en(eng_act_in_rd_en), .rd_data(eng_act_in_rd_data),
        .wr_addr(act_wr_addr), .wr_en(act_wr_en), .wr_data(act_wr_data),
        .tb_rd_addr(tb_act_rd_addr), .tb_rd_data(tb_act_rd_data));

    // ---- the REAL engine, parameterised EXACTLY like the ResNet top ----
    // (nn2rtl_top.v passes only WGT_W/URAM_DATA_W [+K_PAR]; ENABLE_DEPTHWISE
    // and all MAX_* keep their defaults there too.)
    shared_engine #(
        .WGT_W(3),
`ifdef KPAR8
        .URAM_DATA_W(6144),     // [KPAR8-RN] 8 tap-major 768b words per line
        .K_PAR(8)
`elsif KPAR4
        .URAM_DATA_W(3072),     // [KPAR4-RN] 4 tap-major 768b words per line
        .K_PAR(4)
`else
        .URAM_DATA_W(768)
`endif
`ifdef ENG_PIPE
        // [ENG-PIPE-RN] pipelined (pixel, oc_pass) issue — mirrors the
        // deployed ResNet top's .ENG_PIPE(1) (apply_resnet_engpipe.py).
        , .ENG_PIPE(1)
`endif
    ) u_engine(
        .clk(clk), .rst_n(rst_n),
        .s_axil_awvalid(s_axil_awvalid), .s_axil_awready(s_axil_awready), .s_axil_awaddr(s_axil_awaddr),
        .s_axil_wvalid(s_axil_wvalid), .s_axil_wready(s_axil_wready), .s_axil_wdata(s_axil_wdata), .s_axil_wstrb(s_axil_wstrb),
        .s_axil_bvalid(s_axil_bvalid), .s_axil_bready(s_axil_bready), .s_axil_bresp(s_axil_bresp),
        .s_axil_arvalid(s_axil_arvalid), .s_axil_arready(s_axil_arready), .s_axil_araddr(s_axil_araddr),
        .s_axil_rvalid(s_axil_rvalid), .s_axil_rready(s_axil_rready), .s_axil_rdata(s_axil_rdata), .s_axil_rresp(s_axil_rresp),
        .engine_start(engine_start), .engine_busy(engine_busy), .engine_done(engine_done),
        .act_in_rd_addr(eng_act_in_rd_addr), .act_in_rd_en(eng_act_in_rd_en), .act_in_rd_data(eng_act_in_rd_data),
        .act_out_wr_addr(eng_act_out_wr_addr), .act_out_wr_en(eng_act_out_wr_en), .act_out_wr_data(eng_act_out_wr_data),
        .weight_rd_addr(eng_weight_rd_addr), .weight_rd_en(eng_weight_rd_en), .weight_rd_data(eng_weight_rd_data),
        .bias_rd_addr(eng_bias_rd_addr), .bias_rd_en(eng_bias_rd_en), .bias_rd_data(eng_bias_rd_data),
        .scale_rd_addr(eng_scale_rd_addr), .scale_rd_en(eng_scale_rd_en), .scale_rd_data(eng_scale_rd_data)
    );
endmodule

// 96/384-bit weight bank, READ_LATENCY via `WLAT (1 or 2). Mirrors
// nn2rtl_top.v uram_weight_bank rd_data_r1 -> rd_data_r2 (READ_LATENCY=2).
module iso_uram_bank_rn #(parameter MEM_INIT_FILE="", parameter integer WORD_W=96) (
    input wire clk, input wire [16:0] rd_addr,
    output wire [WORD_W-1:0] rd_data, input wire rd_en);
    reg [WORD_W-1:0] mem [0:131071];
    initial if (MEM_INIT_FILE!="") $readmemh(MEM_INIT_FILE, mem);
    reg [WORD_W-1:0] r1, r2;
    always @(posedge clk) begin
        if (rd_en) r1 <= mem[rd_addr];
        r2 <= r1;
    end
`ifdef WLAT1
    assign rd_data = r1;   // 1-cycle (FALSE-confidence)
`else
    assign rd_data = r2;   // 2-cycle (deployment, READ_LATENCY=2)
`endif
endmodule

module iso_bias_mem_rn #(parameter MEM_INIT_FILE="") (
    input wire clk, input wire [7:0] rd_addr,
    output reg [8191:0] rd_data, input wire rd_en);
    reg [8191:0] mem [0:255];
    initial if (MEM_INIT_FILE!="") $readmemh(MEM_INIT_FILE, mem);
    always @(posedge clk) if (rd_en) rd_data <= mem[rd_addr];
endmodule

module iso_act_mem_rn (
    input wire clk,
    input wire [15:0] rd_addr, input wire rd_en, output reg [2047:0] rd_data,
    input wire [15:0] wr_addr, input wire wr_en, input wire [2047:0] wr_data,
    input wire [15:0] tb_rd_addr, output wire [2047:0] tb_rd_data);
    reg [2047:0] mem [0:65535];
    always @(posedge clk) if (rd_en) rd_data <= mem[rd_addr];
    always @(posedge clk) if (wr_en) mem[wr_addr] <= wr_data;
    assign tb_rd_data = mem[tb_rd_addr];   // combinational TB readback
endmodule
