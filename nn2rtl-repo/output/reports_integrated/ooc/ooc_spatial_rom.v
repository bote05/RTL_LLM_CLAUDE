// OOC harness to MEASURE the spatial conv weights_wide ROM tile cost at INT3 vs INT4.
// Mirrors conv_datapath_mp_k.v:92 EXACTLY: (* rom_style="block", ram_style="block" *)
// reg [WIDE_W-1:0] weights_wide [0:DEPTH-1]; with the same 1-stage registered read
// (conv_datapath_mp_k.v:158  weight_word_q <= weights_wide[addr]).
// conv_284/292/298: DEPTH=16384, WIDE_W=432 (INT3 MP16*MPK9*3) / 576 (INT4 *4), ADDR_W=14.
// conv_288:         DEPTH=16384, WIDE_W=384 (INT3 MP16*MPK8*3) / 512 (INT4 *4), ADDR_W=14.
module ooc_spatial_rom #(
    parameter integer DEPTH    = 16384,
    parameter integer WIDE_W   = 432,
    parameter integer ADDR_W   = 14,
    parameter         MEM_INIT = ""
)(
    input  wire                clk,
    input  wire [ADDR_W-1:0]   addr,
    output reg  [WIDE_W-1:0]   q
);
    (* rom_style = "block", ram_style = "block" *) reg [WIDE_W-1:0] weights_wide [0:DEPTH-1];
    initial if (MEM_INIT != "") $readmemh(MEM_INIT, weights_wide);
    always @(posedge clk) q <= weights_wide[addr];
endmodule
