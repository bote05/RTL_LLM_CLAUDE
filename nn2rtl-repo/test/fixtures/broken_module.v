module broken_module(
  input wire clk,
  input wire rst_n
)
  always @(posedge clk) begin
    if (!rst_n) begin
      missing_semicolon <= 1'b0
    end
  end
endmodule
