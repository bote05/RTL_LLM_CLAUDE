module sp_r #(parameter integer W=576, parameter integer DEPTH=256, parameter MEM_INIT_FILE="") (
  input wire clk, input wire [$clog2(DEPTH)-1:0] rd_addr, output wire [W-1:0] rd_data, input wire rd_en );
  (* rom_style="block", ram_style="block" *) reg [W-1:0] mem [0:DEPTH-1];
  initial $readmemh(MEM_INIT_FILE, mem);
  reg [W-1:0] r1; always @(posedge clk) if(rd_en) r1<=mem[rd_addr]; assign rd_data=r1;
endmodule
