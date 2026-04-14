module stream_stall (
  input  wire       clk,
  input  wire       rst_n,
  input  wire       valid_in,
  input  wire [7:0] data_in,
  output wire       ready_in,
  output reg        valid_out,
  output reg  [7:0] data_out
);
  reg phase;

  assign ready_in = phase;

  always @(posedge clk) begin
    if (!rst_n) begin
      phase <= 1'b0;
      valid_out <= 1'b0;
      data_out <= 8'd0;
    end else begin
      phase <= ~phase;
      if (phase && valid_in) begin
        valid_out <= 1'b1;
        data_out <= data_in;
      end else begin
        valid_out <= 1'b0;
      end
    end
  end
endmodule
