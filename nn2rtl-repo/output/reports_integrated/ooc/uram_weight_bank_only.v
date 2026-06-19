module uram_weight_bank #(
    parameter integer DEPTH         = 1024,
    parameter integer ADDR_W        = 17,
    parameter         MEM_INIT_FILE = ""
) (
    input  wire                    clk,
    input  wire [ADDR_W-1:0]       rd_addr,
    output wire [143:0]            rd_data,   // INT4 nibble-packed (low 128b = 32 nibbles)
    input  wire                    rd_en
);
    // [ENGINE BRAM 2026-05-30] Inferred block-RAM ROM with $readmemh init — ONE path
    // for sim (Verilator/iverilog) AND Vivado synth. Replaces the old XPM-URAM (ultra)
    // branch: URAM CANNOT be bitstream-initialized on this device (proven), and the
    // XPM-URAM + $readmemh fallback blew up BRAM/LUTs. ram_style="block" + $readmemh is
    // bitstream-initializable (Vivado packs the init into the BRAM INIT_xx strings, same
    // mechanism as bias_mem/scale_mem). The TWO output register stages (rd_data_r1 ->
    // rd_data_r2) preserve the READ_LATENCY=2 contract (the hard-won 8677bc0 weight-read-
    // latency alignment) — do NOT collapse to one stage. INT4 nibble-packed: 144-bit lines.
    (* ram_style = "block" *) reg [143:0] mem [0:DEPTH-1];
    initial begin
        if (MEM_INIT_FILE != "") $readmemh(MEM_INIT_FILE, mem);
    end
    reg [143:0] rd_data_r1, rd_data_r2;
    always @(posedge clk) begin
        if (rd_en) rd_data_r1 <= mem[rd_addr];
        rd_data_r2 <= rd_data_r1;
    end
    assign rd_data = rd_data_r2;
endmodule
