module lbslot #(parameter integer IC=512, parameter integer MEM_DEPTH=58, parameter STYLE="block") (
  input wire clk, input wire write_en, input wire sched_advance,
  input wire [$clog2(MEM_DEPTH)-1:0] wr_col, rd_col,
  input wire [IC*8-1:0] data_in, output wire [IC*8-1:0] q
);
  (* ram_style = STYLE *) reg [IC*8-1:0] mem [0:MEM_DEPTH-1];
  reg [IC*8-1:0] q_reg;
  always @(posedge clk) begin
    if (write_en) mem[wr_col] <= data_in;
    if (sched_advance) q_reg <= mem[rd_col];   // registered + gated (the freeze)
  end
  assign q = q_reg;
endmodule
