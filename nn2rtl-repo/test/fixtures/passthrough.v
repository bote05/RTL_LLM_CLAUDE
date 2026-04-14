module passthrough(
  input wire clk,
  input wire rst_n,
  input wire valid_in,
  output wire ready_in,
  input wire signed [7:0] data_in,
  output reg valid_out,
  output reg signed [7:0] data_out
);
  assign ready_in = 1'b1;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      valid_out <= 1'b0;
      data_out <= 8'sd0;
    end else begin
      valid_out <= valid_in;
      data_out <= data_in;
    end
  end
endmodule
