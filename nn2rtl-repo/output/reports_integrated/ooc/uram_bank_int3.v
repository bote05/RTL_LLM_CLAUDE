module uram_bank_int3 #(parameter integer DEPTH=39424, parameter integer ADDR_W=17, parameter MEM_INIT_FILE="") (
  input wire clk, input wire [ADDR_W-1:0] rd_addr, output wire [95:0] rd_data, input wire rd_en );
  (* ram_style = "block", cascade_height = 8 *) reg [95:0] mem [0:DEPTH-1];
  initial if (MEM_INIT_FILE!="") $readmemh(MEM_INIT_FILE, mem);
  reg [95:0] r1, r2;
  always @(posedge clk) begin if (rd_en) r1<=mem[rd_addr]; r2<=r1; end
  assign rd_data=r2;
endmodule
