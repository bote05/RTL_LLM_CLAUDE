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
  reg pending_valid;
  reg [7:0] pending_data;

  assign ready_in = phase;

  always @(posedge clk) begin
    if (!rst_n) begin
      phase <= 1'b0;
      pending_valid <= 1'b0;
      pending_data <= 8'd0;
      valid_out <= 1'b0;
      data_out <= 8'd0;
    end else begin
      valid_out <= pending_valid;
      if (pending_valid) begin
        data_out <= pending_data;
      end

      if (phase && valid_in) begin
        pending_valid <= 1'b1;
        pending_data <= data_in;
      end else begin
        pending_valid <= 1'b0;
      end

      phase <= ~phase;
    end
  end
endmodule
