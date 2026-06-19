module sfifo_sync #(parameter integer WIDTH=256, parameter integer DEPTH=1024, parameter STYLE="ultra") (
  input wire clk, input wire rst_n,
  input wire in_valid, input wire [WIDTH-1:0] in_data, output wire in_ready,
  output reg out_valid, output reg [WIDTH-1:0] out_data, input wire out_ready );
  localparam AW = $clog2(DEPTH);
  (* ram_style = STYLE *) reg [WIDTH-1:0] mem [0:DEPTH-1];
  reg [AW:0] wr, rd;
  wire empty = (wr==rd);
  wire full  = (wr[AW]!=rd[AW]) && (wr[AW-1:0]==rd[AW-1:0]);
  assign in_ready = !full;
  wire do_wr = in_valid && !full;
  wire do_rd = !empty && (!out_valid || out_ready);   // registered/FWFT load
  always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin wr<=0; rd<=0; out_valid<=0; out_data<=0; end
    else begin
      if(do_wr) wr<=wr+1'b1;
      if(out_valid && out_ready) out_valid<=0;
      if(do_rd) begin out_data<=mem[rd[AW-1:0]]; out_valid<=1; rd<=rd+1'b1; end
    end
  end
  always @(posedge clk) if(do_wr) mem[wr[AW-1:0]]<=in_data;
endmodule
