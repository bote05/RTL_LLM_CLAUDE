`timescale 1ns/1ps
// engine_iso_wrap_mbv2.v -- MobileNetV2 engine-isolation wrapper.
//
// Instantiates the REAL shared_engine + REAL memory modules wired EXACTLY as
// output/mobilenet-v2/rtl/nn2rtl_top.v wires them for the engine path:
//   * 8 uram_weight_bank (288-bit line, low 256 used -> 2048-bit weight bus,
//     READ_LATENCY=2 -- the DEPLOYMENT 2-cycle URAM read)
//   * bias_mem / scale_mem (8192-bit = 256 INT32 per oc_pass, READ_LATENCY=1)
//   * act_unified_mem (2048-bit = 256 INT8 channels per pixel, READ_LATENCY=1)
//     with a TB write/read port for preload + result capture.
//
// The mbv2 weight banks store INT8 byte weights (32 OC per bank): OC k lives at
// byte k of the 2048-bit bus = bank(k/32) byte-slot(k%32). bias/scale lane = OC.
// PROVEN against node_conv_816.goldout pixel0 OC0..11 in Python before build.
// Therefore the engine is parameterised WGT_W=8, URAM_DATA_W=2048 (NOT the
// ResNet INT4 nibble defaults WGT_W=4/URAM_DATA_W=1024).
//
// The C++ driver does AXI config + engine_start + act preload + compare only;
// no C++ memory model. `WLAT macro selects URAM read latency (default 2 =
// deployment). Define WLAT1 to collapse to a (FALSE-confidence) 1-cycle read.

module engine_iso_wrap_mbv2 (
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
    wire [2047:0] eng_weight_rd_data;     // 2048b = 256 INT8 weights (mbv2)
    wire [21:0]   eng_bias_rd_addr;
    wire          eng_bias_rd_en;
    wire [8191:0] eng_bias_rd_data;
    wire [21:0]   eng_scale_rd_addr;
    wire          eng_scale_rd_en;
    wire [8191:0] eng_scale_rd_data;

    // ---- 8 URAM weight banks (288-bit line, low 256 used), 2-cycle read ----
    // Mirrors nn2rtl_top.v: weight bus = concat of low 256 bits of each bank,
    // bank0 lowest. The engine consumes the full 2048-bit bus (WGT_W=8).
    wire [16:0]  wbank_addr = eng_weight_rd_addr[16:0];
    wire [287:0] b0,b1,b2,b3,b4,b5,b6,b7;
    assign eng_weight_rd_data = {b7[255:0],b6[255:0],b5[255:0],b4[255:0],
                                 b3[255:0],b2[255:0],b1[255:0],b0[255:0]};
    iso_uram_bank #(.MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank0.mem")) u0(.clk(clk),.rd_addr(wbank_addr),.rd_data(b0),.rd_en(eng_weight_rd_en));
    iso_uram_bank #(.MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank1.mem")) u1(.clk(clk),.rd_addr(wbank_addr),.rd_data(b1),.rd_en(eng_weight_rd_en));
    iso_uram_bank #(.MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank2.mem")) u2(.clk(clk),.rd_addr(wbank_addr),.rd_data(b2),.rd_en(eng_weight_rd_en));
    iso_uram_bank #(.MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank3.mem")) u3(.clk(clk),.rd_addr(wbank_addr),.rd_data(b3),.rd_en(eng_weight_rd_en));
    iso_uram_bank #(.MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank4.mem")) u4(.clk(clk),.rd_addr(wbank_addr),.rd_data(b4),.rd_en(eng_weight_rd_en));
    iso_uram_bank #(.MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank5.mem")) u5(.clk(clk),.rd_addr(wbank_addr),.rd_data(b5),.rd_en(eng_weight_rd_en));
    iso_uram_bank #(.MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank6.mem")) u6(.clk(clk),.rd_addr(wbank_addr),.rd_data(b6),.rd_en(eng_weight_rd_en));
    iso_uram_bank #(.MEM_INIT_FILE("output/mobilenet-v2/weights/uram_weights_bank7.mem")) u7(.clk(clk),.rd_addr(wbank_addr),.rd_data(b7),.rd_en(eng_weight_rd_en));

    // ---- bias + scale ROMs (8192-bit, 1-cycle) ----
    iso_bias_mem #(.MEM_INIT_FILE("output/mobilenet-v2/weights/bias.mem"))
      u_bias(.clk(clk),.rd_addr(eng_bias_rd_addr[7:0]),.rd_data(eng_bias_rd_data),.rd_en(eng_bias_rd_en));
    iso_bias_mem #(.MEM_INIT_FILE("output/mobilenet-v2/weights/scale.mem"))
      u_scale(.clk(clk),.rd_addr(eng_scale_rd_addr[7:0]),.rd_data(eng_scale_rd_data),.rd_en(eng_scale_rd_en));

    // ---- activation unified mem (2048-bit, 1-cycle), engine + TB write mux ----
    wire        act_wr_en   = eng_act_out_wr_en | tb_act_wr_en;
    wire [15:0] act_wr_addr = eng_act_out_wr_en ? eng_act_out_wr_addr : tb_act_wr_addr;
    wire [2047:0] act_wr_data = eng_act_out_wr_en ? eng_act_out_wr_data : tb_act_wr_data;
    iso_act_mem u_act(.clk(clk),
        .rd_addr(eng_act_in_rd_addr), .rd_en(eng_act_in_rd_en), .rd_data(eng_act_in_rd_data),
        .wr_addr(act_wr_addr), .wr_en(act_wr_en), .wr_data(act_wr_data),
        .tb_rd_addr(tb_act_rd_addr), .tb_rd_data(tb_act_rd_data));

    // ---- the REAL engine, parameterised for mbv2 INT8 weight slots ----
    shared_engine #(
        .WGT_W(8),
        .URAM_DATA_W(2048),
        .MAX_IC(2048),
        .MAX_OC(2048),
        .MAX_IH(112),
        .MAX_IW(112),
        .MAX_OH(112),
        .MAX_OW(112),
        // [DW-ENGINE P1] mirror the MBV2 engine top: depthwise mode armed
        // (inert for dense dispatches — cfg 0x3C resets to 0).
        .ENABLE_DEPTHWISE(1)
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

// 288-bit URAM bank, READ_LATENCY via `WLAT (1 or 2). Mirrors nn2rtl_top.v
// uram_weight_bank XPM READ_LATENCY_A=2 (2-stage). For WLAT1 collapse to 1.
module iso_uram_bank #(parameter MEM_INIT_FILE="") (
    input wire clk, input wire [16:0] rd_addr,
    output wire [287:0] rd_data, input wire rd_en);
    reg [287:0] mem [0:131071];
    initial if (MEM_INIT_FILE!="") $readmemh(MEM_INIT_FILE, mem);
    reg [287:0] r1, r2;
    always @(posedge clk) begin
        if (rd_en) r1 <= mem[rd_addr];
        r2 <= r1;
    end
`ifdef WLAT1
    assign rd_data = r1;   // 1-cycle (FALSE-confidence)
`else
    assign rd_data = r2;   // 2-cycle (deployment, READ_LATENCY_A=2)
`endif
endmodule

module iso_bias_mem #(parameter MEM_INIT_FILE="") (
    input wire clk, input wire [7:0] rd_addr,
    output reg [8191:0] rd_data, input wire rd_en);
    reg [8191:0] mem [0:255];
    initial if (MEM_INIT_FILE!="") $readmemh(MEM_INIT_FILE, mem);
    always @(posedge clk) if (rd_en) rd_data <= mem[rd_addr];
endmodule

module iso_act_mem (
    input wire clk,
    input wire [15:0] rd_addr, input wire rd_en, output reg [2047:0] rd_data,
    input wire [15:0] wr_addr, input wire wr_en, input wire [2047:0] wr_data,
    input wire [15:0] tb_rd_addr, output wire [2047:0] tb_rd_data);
    reg [2047:0] mem [0:65535];
    always @(posedge clk) if (rd_en) rd_data <= mem[rd_addr];
    always @(posedge clk) if (wr_en) mem[wr_addr] <= wr_data;
    assign tb_rd_data = mem[tb_rd_addr];   // combinational TB readback
endmodule
