module stream_latency2 (
  input  wire       clk,
  input  wire       rst_n,
  input  wire       valid_in,
  input  wire [7:0] data_in,
  output wire       ready_in,
  output reg        valid_out,
  output reg  [7:0] data_out
);
  reg [7:0] stage0_data;
  reg [7:0] stage1_data;
  reg       stage0_valid;
  reg       stage1_valid;

  assign ready_in = 1'b1;

  always @(posedge clk) begin
    if (!rst_n) begin
      stage0_data <= 8'd0;
      stage1_data <= 8'd0;
      stage0_valid <= 1'b0;
      stage1_valid <= 1'b0;
      valid_out <= 1'b0;
      data_out <= 8'd0;
    end else begin
      valid_out <= stage1_valid;
      data_out <= stage1_data;
      stage1_valid <= stage0_valid;
      stage1_data <= stage0_data;
      stage0_valid <= valid_in;
      stage0_data <= data_in;
    end
  end
endmodule
