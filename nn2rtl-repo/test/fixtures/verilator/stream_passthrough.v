module stream_passthrough (
  input  wire       clk,
  input  wire       rst_n,
  input  wire       valid_in,
  input  wire [7:0] data_in,
  output wire       ready_in,
  output reg        valid_out,
  output reg  [7:0] data_out
);
  assign ready_in = 1'b1;

  always @(posedge clk) begin
    if (!rst_n) begin
      valid_out <= 1'b0;
      data_out <= 8'd0;
    end else begin
      valid_out <= valid_in;
      if (valid_in) begin
        data_out <= data_in;
      end
    end
  end
endmodule
